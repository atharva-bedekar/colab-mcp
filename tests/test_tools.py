import json
from unittest.mock import MagicMock

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError


@pytest.mark.asyncio
async def test_all_twenty_one_tools_registered_and_callable(tmp_path, monkeypatch):
    import colab_mcp.jobs as jobs_module
    import colab_mcp.transfer as transfer_module
    from colab_mcp.models import (
        AuthStatusResponse,
        ExecutionResult,
        FileExecutionResult,
        RemoteFileEntry,
        SessionInfo,
        SessionSummary,
        UnauthenticatedError,
    )
    from colab_mcp.tools import register_tools

    mcp = FastMCP(name="ColabMCPTest")
    mgr = MagicMock()
    mgr.auth_manager.get_status.return_value = AuthStatusResponse(
        authenticated=True, provider="oauth2", token_path="/tmp/token.json"
    )
    mgr.create_session.return_value = SessionInfo(
        name="trainer",
        endpoint="ep-1",
        variant="GPU",
        accelerator="T4",
        shape="STANDARD",
        runtime_url="https://proxy.colab.google",
    )
    mgr.list_sessions.return_value = [
        SessionSummary(
            name="trainer",
            endpoint="ep-1",
            variant="GPU",
            accelerator="T4",
            shape="STANDARD",
        )
    ]
    fake_contents = MagicMock()
    fake_contents.list_dir.return_value = [
        RemoteFileEntry(name="data.csv", path="/content/data.csv", type="file", size=42)
    ]
    fake_contents.read_file.return_value = {
        "remote_path": "/content/data.csv",
        "content": "a,b\n1,2\n",
        "size_bytes": 8,
        "truncated": False,
    }
    fake_contents.stat.return_value = RemoteFileEntry(
        name="data.csv", path="/content/data.csv", type="file", size=42
    )
    mgr.get_contents_client.return_value = fake_contents

    download_payload = {
        "remote_dir": "/content/data",
        "local_dir": str(tmp_path / "downloaded"),
        "files_count": 2,
        "archive_size_bytes": 128,
        "sha256": "a" * 64,
        "verified": True,
    }
    upload_payload = {
        "local_dir": str(tmp_path / "project"),
        "remote_dir": "/content/project",
        "files_count": 3,
        "archive_size_bytes": 256,
        "sha256": "b" * 64,
        "verified": True,
    }
    job_start_payload = {
        "job_id": "train-1",
        "pid": 4321,
        "log_path": "/tmp/colab_mcp_jobs/train-1.log",
        "status": "running",
    }
    job_check_payload = {
        "job_id": "train-1",
        "pid": 4321,
        "status": "exited",
        "exit_code": 0,
        "log_path": "/tmp/colab_mcp_jobs/train-1.log",
        "log_tail": "done\n",
    }
    job_stop_payload = {"job_id": "train-1", "pid": 4321, "stopped": True}

    mock_download_dir = MagicMock(return_value=download_payload)
    mock_upload_dir = MagicMock(return_value=upload_payload)
    mock_start_job = MagicMock(return_value=job_start_payload)
    mock_check_job = MagicMock(return_value=job_check_payload)
    mock_stop_job = MagicMock(return_value=job_stop_payload)
    monkeypatch.setattr(transfer_module, "download_dir", mock_download_dir)
    monkeypatch.setattr(transfer_module, "upload_dir", mock_upload_dir)
    monkeypatch.setattr(jobs_module, "start_background_job", mock_start_job)
    monkeypatch.setattr(jobs_module, "check_background_job", mock_check_job)
    monkeypatch.setattr(jobs_module, "stop_background_job", mock_stop_job)

    fake_rt = MagicMock()
    fake_rt.execute_code.return_value = ExecutionResult(
        status="ok", stdout="hi\n", duration_s=0.1
    )
    fake_rt.execute_file.return_value = FileExecutionResult(
        file_path=str(tmp_path / "run.py"), status="ok", cells_executed=1, duration_s=0.2
    )
    mgr.get_runtime.return_value = fake_rt

    register_tools(mcp, mgr)

    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert names == {
            "auth_status",
            "auth_start_login",
            "auth_finish_login",
            "create_session",
            "list_sessions",
            "session_status",
            "stop_session",
            "restart_kernel",
            "upload_file",
            "download_file",
            "list_files",
            "delete_file",
            "read_file",
            "stat_file",
            "download_dir",
            "upload_dir",
            "start_background_job",
            "check_background_job",
            "stop_background_job",
            "exec_code",
            "exec_file",
        }

        auth_res = await client.call_tool("auth_status", {"provider": "oauth2"})
        assert json.loads(auth_res.content[0].text)["authenticated"] is True

        create_res = await client.call_tool("create_session", {"name": "trainer", "gpu": "T4"})
        assert json.loads(create_res.content[0].text)["endpoint"] == "ep-1"

        exec_res = await client.call_tool("exec_code", {"session": "trainer", "code": "print('hi')"})
        assert json.loads(exec_res.content[0].text)["stdout"] == "hi\n"
        assert fake_rt.execute_code.call_args.kwargs["compact"] is True
        # Verify running label set and then cleared in finally
        calls = mgr.update_session_execution.call_args_list
        assert calls[0].kwargs["running"] == "exec_code"
        assert calls[-1].kwargs["running"] is None
        assert calls[-1].kwargs["last_execution"][0] == "ok"

        raw_res = await client.call_tool(
            "exec_code",
            {"session": "trainer", "code": "print('hi')", "raw_outputs": True},
        )
        assert json.loads(raw_res.content[0].text)["stdout"] == "hi\n"
        assert fake_rt.execute_code.call_args.kwargs["compact"] is False

        file_res = await client.call_tool("exec_file", {"session": "trainer", "file_path": str(tmp_path / "run.py")})
        assert json.loads(file_res.content[0].text)["status"] == "ok"

        files_res = await client.call_tool("list_files", {"session": "trainer"})
        assert json.loads(files_res.content[0].text)[0]["name"] == "data.csv"

        read_res = await client.call_tool(
            "read_file",
            {"session": "trainer", "remote_path": "/content/data.csv", "tail_lines": 5},
        )
        assert json.loads(read_res.content[0].text)["content"] == "a,b\n1,2\n"
        fake_contents.read_file.assert_called_once_with(
            "/content/data.csv", tail_lines=5, max_bytes=65536
        )

        stat_res = await client.call_tool(
            "stat_file", {"session": "trainer", "remote_path": "/content/data.csv"}
        )
        assert json.loads(stat_res.content[0].text)["name"] == "data.csv"

        download_res = await client.call_tool(
            "download_dir",
            {
                "session": "trainer",
                "remote_dir": "/content/data",
                "local_dir": str(tmp_path / "downloaded"),
                "glob": "*.csv",
            },
        )
        assert json.loads(download_res.content[0].text) == download_payload
        mock_download_dir.assert_called_once_with(
            fake_rt, fake_contents, "/content/data", str(tmp_path / "downloaded"), glob="*.csv"
        )

        upload_res = await client.call_tool(
            "upload_dir",
            {
                "session": "trainer",
                "local_dir": str(tmp_path / "project"),
                "remote_dir": "/content/project",
            },
        )
        assert json.loads(upload_res.content[0].text) == upload_payload
        mock_upload_dir.assert_called_once_with(
            fake_rt, fake_contents, str(tmp_path / "project"), "/content/project", glob=None
        )

        start_res = await client.call_tool(
            "start_background_job",
            {
                "session": "trainer",
                "command": "python train.py",
                "name": "train-1",
                "cwd": "/content/project",
                "env": {"CUDA_VISIBLE_DEVICES": "0"},
            },
        )
        assert json.loads(start_res.content[0].text) == job_start_payload
        mock_start_job.assert_called_once_with(
            fake_rt,
            "python train.py",
            name="train-1",
            cwd="/content/project",
            env={"CUDA_VISIBLE_DEVICES": "0"},
        )

        check_res = await client.call_tool(
            "check_background_job",
            {"session": "trainer", "job_id": "train-1", "tail_lines": 10},
        )
        assert json.loads(check_res.content[0].text) == job_check_payload
        mock_check_job.assert_called_once_with(fake_rt, "train-1", tail_lines=10)

        stop_res = await client.call_tool(
            "stop_background_job", {"session": "trainer", "job_id": "train-1"}
        )
        assert json.loads(stop_res.content[0].text) == job_stop_payload
        mock_stop_job.assert_called_once_with(fake_rt, "train-1")

        mgr.get_runtime.side_effect = UnauthenticatedError("login required")
        with pytest.raises(ToolError, match="login required"):
            await client.call_tool("exec_code", {"session": "trainer", "code": "1+1"})
