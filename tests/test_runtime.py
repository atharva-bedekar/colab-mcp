import json
from unittest.mock import MagicMock


def test_execute_code_passes_timeout_and_headers():
    from colab_mcp.runtime import ColabRuntime

    fake_kc = MagicMock()
    fake_kc.id = "k-123"
    fake_kc.execute.return_value = {
        "status": "ok",
        "outputs": [{"output_type": "stream", "name": "stdout", "text": "ready\n"}],
    }
    factory = MagicMock(return_value=fake_kc)
    rt = ColabRuntime("https://proxy.colab.google", "tok-xyz", kernel_client_factory=factory)
    res = rt.execute_code("print('ready')", timeout=600.0, allow_stdin=False, env={"FOO": "bar"}, ensure_cwd=False)

    assert res.status == "ok"
    assert res.stdout == "ready\n"
    assert fake_kc._own_kernel is False
    factory.assert_called_once()
    assert factory.call_args.kwargs["headers"]["X-Colab-Runtime-Proxy-Token"] == "tok-xyz"
    exec_args, exec_kwargs = fake_kc.execute.call_args
    assert "os.environ['FOO'] = 'bar'" in exec_args[0]
    assert exec_kwargs["timeout"] == 600.0
    assert exec_kwargs["allow_stdin"] is False


def test_execute_file_ipynb_and_lifecycle(tmp_path):
    from colab_mcp.runtime import ColabRuntime

    nb_path = tmp_path / "train.ipynb"
    nb_path.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [
                    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": ["a = 1"]},
                    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": ["print(a)"]},
                ],
            }
        ),
        encoding="utf-8",
    )

    fake_kc = MagicMock()
    fake_kc.id = "k-456"
    fake_kc.execution_state = "idle"
    fake_kc.execute.side_effect = [
        {"status": "ok", "outputs": []},  # ensure_content_dir
        {"status": "ok", "outputs": []},
        {"status": "ok", "outputs": [{"output_type": "stream", "name": "stdout", "text": "1\n"}]},
    ]
    rt = ColabRuntime("https://proxy.colab.google", "tok-xyz", kernel_client_factory=MagicMock(return_value=fake_kc))
    res = rt.execute_file(nb_path, save_output_notebook=True)

    assert res.status == "ok"
    assert res.cells_executed == 2
    assert res.output_notebook_path == str(tmp_path / "train_output.ipynb")
    saved = json.loads((tmp_path / "train_output.ipynb").read_text(encoding="utf-8"))
    assert saved["cells"][1]["outputs"][0]["text"] == "1\n"

    assert rt.get_kernel_state() == "idle"
    fake_kc._manager.refresh_model.assert_called_once()
    assert rt.restart() == "k-456"
    fake_kc.restart.assert_called_once()
    rt.stop()
    fake_kc.stop.assert_called_once_with(shutdown_kernel=False)
