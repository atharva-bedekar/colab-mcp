import json
import uuid
from datetime import datetime, timezone

import pytest


def test_strip_xssi_and_uuid_encoding():
    from colab_mcp.models import strip_xssi_prefix, uuid_to_web_safe_base64

    assert strip_xssi_prefix(")]}'\n{\"a\": 1}") == '{"a": 1}'
    assert strip_xssi_prefix(")]}'{\"a\": 1}") == '{"a": 1}'
    assert strip_xssi_prefix('{"a": 1}') == '{"a": 1}'

    u = uuid.UUID("12345678-1234-5678-1234-567812345678")
    encoded = uuid_to_web_safe_base64(u)
    assert len(encoded) == 44
    assert encoded == "12345678_1234_5678_1234_567812345678........"


def test_resolve_runtime_options():
    from colab_mcp.models import Accelerator, Shape, Variant, resolve_runtime_options

    assert resolve_runtime_options() == (Variant.DEFAULT, Accelerator.NONE, Shape.STANDARD)
    assert resolve_runtime_options(high_mem=True) == (
        Variant.DEFAULT,
        Accelerator.NONE,
        Shape.HIGH_RAM,
    )
    assert resolve_runtime_options(gpu="T4", high_mem=True) == (
        Variant.GPU,
        Accelerator.T4,
        Shape.HIGH_RAM,
    )
    assert resolve_runtime_options(gpu="L4", high_mem=True) == (
        Variant.GPU,
        Accelerator.L4,
        None,
    )
    assert resolve_runtime_options(tpu="v5e1", high_mem=True) == (
        Variant.TPU,
        Accelerator.V5E1,
        None,
    )
    with pytest.raises(ValueError, match="Cannot specify both GPU and TPU"):
        resolve_runtime_options(gpu="T4", tpu="v5e1")
    with pytest.raises(ValueError):
        resolve_runtime_options(gpu="INVALID")
    with pytest.raises(ValueError):
        resolve_runtime_options(tpu="INVALID")


def test_pydantic_models_and_aliases():
    from colab_mcp.models import (
        Assignment,
        AssignmentVariant,
        ListedAssignment,
        RuntimeProxyInfo,
        Shape,
    )

    proxy = RuntimeProxyInfo(
        token="tok-1",
        tokenExpiresInSeconds=1200,
        url="https://proxy.colab.google",
    )
    now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert proxy.compute_expires_at(now) == datetime(2026, 1, 1, 0, 20, 0, tzinfo=timezone.utc)

    assign = Assignment(
        endpoint="ep-1",
        accelerator="T4",
        variant="GPU",
        machineShape=Shape.HIGH_RAM,
        runtimeProxyInfo={
            "token": "tok-1",
            "tokenExpiresInSeconds": 3600,
            "url": "https://proxy.colab.google",
        },
    )
    assert assign.machine_shape == Shape.HIGH_RAM
    assert assign.runtime_proxy_info.token_expires_in_seconds == 3600

    int_assign = Assignment(
        endpoint="ep-cpu",
        accelerator="NONE",
        variant=0,
        runtimeProxyInfo={
            "token": "tok-cpu",
            "tokenExpiresInSeconds": 3600,
            "url": "https://proxy.colab.google",
        },
    )
    assert int_assign.variant == AssignmentVariant.DEFAULT

    listed = ListedAssignment(endpoint="ep-2")
    assert listed.runtime_proxy_info is None


def test_notebook_roundtrip(tmp_path):
    from colab_mcp.models import parse_notebook_code_cells, write_notebook_with_outputs

    nb_path = tmp_path / "sample.ipynb"
    nb_payload = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Title"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": ["x = 1\n", "print(x)"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": "print(x + 1)"},
        ],
    }
    nb_path.write_text(json.dumps(nb_payload), encoding="utf-8")

    nb_data, code_cells = parse_notebook_code_cells(nb_path)
    assert len(code_cells) == 2
    assert code_cells[0]["index"] == 1
    assert code_cells[0]["source"] == "x = 1\nprint(x)"
    assert code_cells[1]["index"] == 2
    assert code_cells[1]["source"] == "print(x + 1)"

    out_path = tmp_path / "sample_output.ipynb"
    outputs = {
        1: [{"output_type": "stream", "name": "stdout", "text": ["1\n"]}],
        2: [{"output_type": "stream", "name": "stdout", "text": ["2\n"]}],
    }
    write_notebook_with_outputs(nb_data, outputs, out_path)

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["cells"][1]["outputs"] == outputs[1]
    assert written["cells"][2]["outputs"] == outputs[2]
