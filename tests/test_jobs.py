import ast
import json
from unittest.mock import MagicMock

import pytest

from colab_mcp.jobs import check_background_job, start_background_job, stop_background_job
from colab_mcp.models import ColabError, ExecutionResult


def _mock_runtime(stdout: str = "", *, status: str = "ok", error: str | None = None) -> MagicMock:
    runtime = MagicMock()
    runtime.execute_code.return_value = ExecutionResult(
        status=status,
        stdout=stdout,
        error=error,
    )
    return runtime


def test_start_background_job_returns_running_metadata_and_passes_options():
    expected = {
        "job_id": "train-1",
        "pid": 4321,
        "log_path": "/tmp/colab_mcp_jobs/train-1.log",
        "status": "running",
    }
    runtime = _mock_runtime(json.dumps(expected) + "\n")

    result = start_background_job(
        runtime,
        "python train.py --epochs 10",
        name="train-1",
        cwd="/content/project",
        env={"CUDA_VISIBLE_DEVICES": "0"},
    )

    assert result == expected
    runtime.execute_code.assert_called_once()
    code = runtime.execute_code.call_args.args[0]
    assert runtime.execute_code.call_args.kwargs["compact"] is True
    ast.parse(code)
    assert "python train.py --epochs 10" in code
    assert "/content/project" in code
    assert "CUDA_VISIBLE_DEVICES" in code
    assert "start_new_session=True" in code


def test_start_background_job_generates_default_job_id():
    runtime = _mock_runtime(
        json.dumps(
            {
                "job_id": "job-1234abcd",
                "pid": 100,
                "log_path": "/tmp/colab_mcp_jobs/job-1234abcd.log",
                "status": "running",
            }
        )
    )

    result = start_background_job(runtime, "echo hi")

    assert result["status"] == "running"
    code = runtime.execute_code.call_args.args[0]
    ast.parse(code)
    assert "job_id = 'job-" in code


def test_check_background_job_running():
    payload = {
        "job_id": "train-1",
        "pid": 4321,
        "status": "running",
        "exit_code": None,
        "log_path": "/tmp/colab_mcp_jobs/train-1.log",
        "log_tail": "epoch 1\nepoch 2\n",
    }
    runtime = _mock_runtime(json.dumps(payload) + "\n")

    result = check_background_job(runtime, "train-1", tail_lines=25)

    assert result == payload
    assert result["status"] == "running"
    assert result["exit_code"] is None
    assert result["log_tail"] == "epoch 1\nepoch 2\n"
    code = runtime.execute_code.call_args.args[0]
    assert runtime.execute_code.call_args.kwargs["compact"] is True
    ast.parse(code)
    assert "tail_lines = 25" in code


def test_check_background_job_exited():
    payload = {
        "job_id": "train-1",
        "pid": 4321,
        "status": "exited",
        "exit_code": 0,
        "log_path": "/tmp/colab_mcp_jobs/train-1.log",
        "log_tail": "done\n",
    }
    runtime = _mock_runtime(json.dumps(payload) + "\n")

    result = check_background_job(runtime, "train-1")

    assert result["status"] == "exited"
    assert result["exit_code"] == 0
    assert result["log_tail"] == "done\n"


def test_stop_background_job_returns_stopped():
    payload = {"job_id": "train-1", "pid": 4321, "stopped": True}
    runtime = _mock_runtime(json.dumps(payload) + "\n")

    result = stop_background_job(runtime, "train-1")

    assert result == payload
    code = runtime.execute_code.call_args.args[0]
    assert runtime.execute_code.call_args.kwargs["compact"] is True
    ast.parse(code)
    assert "os.killpg(pid, signal.SIGTERM)" in code


def test_missing_job_raises_file_not_found():
    runtime = _mock_runtime(json.dumps({"not_found": True}) + "\n")

    with pytest.raises(FileNotFoundError, match="missing-job"):
        check_background_job(runtime, "missing-job")

    with pytest.raises(FileNotFoundError, match="missing-job"):
        stop_background_job(runtime, "missing-job")


@pytest.mark.parametrize(
    "bad_id",
    ["../bad", "a/b", "", ".hidden", "-dash", "bad name", "a" * 65, "job\n"],
)
def test_invalid_job_id_raises_value_error(bad_id: str):
    runtime = _mock_runtime()

    with pytest.raises(ValueError):
        start_background_job(runtime, "echo hi", name=bad_id)
    with pytest.raises(ValueError):
        check_background_job(runtime, bad_id)
    with pytest.raises(ValueError):
        stop_background_job(runtime, bad_id)

    runtime.execute_code.assert_not_called()


def test_remote_execution_error_or_invalid_json_raises_colab_error():
    err_runtime = _mock_runtime("", status="error", error="PermissionError: denied")
    with pytest.raises(ColabError, match="PermissionError"):
        start_background_job(err_runtime, "echo hi", name="j1")

    bad_json_runtime = _mock_runtime("not valid json")
    with pytest.raises(ColabError, match="Invalid JSON"):
        check_background_job(bad_json_runtime, "j1")
