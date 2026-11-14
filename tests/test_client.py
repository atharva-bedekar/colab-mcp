import uuid
from unittest.mock import MagicMock

import pytest
import requests


def test_assign_two_step_handshake_and_xssi():
    from colab_mcp.client import ColabClient
    from colab_mcp.models import Accelerator, Shape, Variant

    sess = MagicMock()
    get_resp = MagicMock(status_code=200, text=")]}'\n{\"token\": \"xsrf-123\", \"variant\": \"GPU\", \"acc\": \"T4\"}")
    post_resp = MagicMock(
        status_code=200,
        text=(
            ")]}\'\n"
            '{"endpoint": "gpu-t4-s-1", "accelerator": "T4", "variant": 1, '
            '"machineShape": 1, '
            '"runtimeProxyInfo": {"token": "p-tok", "tokenExpiresInSeconds": 3600, '
            '"url": "https://8080-gpu.colab.googleusercontent.com"}}'
        ),
    )
    sess.request.side_effect = [get_resp, post_resp]
    client = ColabClient(sess)
    res = client.assign(uuid.uuid4(), Variant.GPU, Accelerator.T4, Shape.HIGH_RAM)
    assert res.endpoint == "gpu-t4-s-1"
    assert res.runtime_proxy_info.token == "p-tok"
    assert sess.request.call_count == 2
    post_call = sess.request.call_args_list[1]
    assert post_call.kwargs["headers"]["X-Goog-Colab-Token"] == "xsrf-123"
    assert post_call.kwargs["params"]["shape"] == "hm"

    sess.request.reset_mock()
    sess.request.side_effect = [
        MagicMock(status_code=200, text=")]}'\n{\"token\": \"xsrf-cpu\", \"variant\": \"DEFAULT\", \"acc\": \"NONE\"}"),
        MagicMock(
            status_code=200,
            text=(
                ")]}\'\n"
                '{"endpoint": "cpu-1", "accelerator": "NONE", "variant": 0, '
                '"runtimeProxyInfo": {"token": "p-cpu", "tokenExpiresInSeconds": 3600, '
                '"url": "https://8080-cpu.colab.googleusercontent.com"}}'
            ),
        ),
    ]
    cpu_res = client.assign(uuid.uuid4(), Variant.DEFAULT, Accelerator.NONE, Shape.STANDARD)
    assert cpu_res.endpoint == "cpu-1"
    assert int(cpu_res.variant) == 0


def test_assign_503_retry_recovery_and_412():
    from colab_mcp.client import ColabClient
    from colab_mcp.models import TooManyAssignmentsError

    sess = MagicMock()
    get_tok_1 = MagicMock(status_code=200, text=")]}'\n{\"token\": \"xsrf-1\"}")
    post_503 = MagicMock(status_code=503, text="unavailable")
    get_tok_2 = MagicMock(status_code=200, text=")]}'\n{\"token\": \"xsrf-2\"}")
    post_ok = MagicMock(
        status_code=200,
        text=(
            ")]}\'\n"
            '{"endpoint": "ep-ok", "runtimeProxyInfo": {"token": "p-2", '
            '"tokenExpiresInSeconds": 3600, "url": "https://ep-ok.colab.google"}}'
        ),
    )
    sess.request.side_effect = [get_tok_1, post_503, get_tok_2, post_ok]
    slept = []
    client = ColabClient(sess)
    res = client.assign(uuid.uuid4(), retries=2, backoff=1.0, sleep_fn=slept.append)
    assert res.endpoint == "ep-ok"
    assert slept == [1.0]

    sess.request.reset_mock()
    sess.request.side_effect = [
        MagicMock(status_code=200, text=")]}'\n{\"token\": \"xsrf-3\"}"),
        MagicMock(status_code=412, text="too many assignments"),
    ]
    with pytest.raises(TooManyAssignmentsError):
        client.assign(uuid.uuid4(), sleep_fn=lambda _: None)


def test_list_assignments_unassign_and_keep_alive():
    from colab_mcp.client import ColabClient

    sess = MagicMock()
    sess.request.return_value = MagicMock(
        status_code=200,
        text=')]}\'\n{"assignments": [{"endpoint": "ep-1", "variant": 1, "accelerator": "T4", "machineShape": 0}]}',
    )
    client = ColabClient(sess)
    listed = client.list_assignments()
    assert len(listed) == 1
    assert listed[0].endpoint == "ep-1"

    sess.request.reset_mock()
    sess.request.side_effect = [
        MagicMock(status_code=200, text=')]}\'\n{"token": "unassign-tok"}'),
        MagicMock(status_code=204, text=""),
    ]
    client.unassign("ep-1")
    assert sess.request.call_args_list[1].kwargs["headers"]["X-Goog-Colab-Token"] == "unassign-tok"

    sess.request.reset_mock()
    sess.request.side_effect = requests.exceptions.ReadTimeout("tunnel forward timeout")
    assert client.keep_alive("ep-1") is True
