import json
from unittest.mock import MagicMock

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError


@pytest.mark.asyncio
async def test_all_fourteen_tools_registered_and_callable(tmp_path):
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
    mgr.get_contents_client.return_value = fake_contents

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
            "exec_code",
            "exec_file",
        }

        auth_res = await client.call_tool("auth_status", {"provider": "oauth2"})
        assert json.loads(auth_res.content[0].text)["authenticated"] is True

        create_res = await client.call_tool("create_session", {"name": "trainer", "gpu": "T4"})
        assert json.loads(create_res.content[0].text)["endpoint"] == "ep-1"

        exec_res = await client.call_tool("exec_code", {"session": "trainer", "code": "print('hi')"})
        assert json.loads(exec_res.content[0].text)["stdout"] == "hi\n"
        # Verify running label set and then cleared in finally
        calls = mgr.update_session_execution.call_args_list
        assert calls[0].kwargs["running"] == "exec_code"
        assert calls[-1].kwargs["running"] is None
        assert calls[-1].kwargs["last_execution"][0] == "ok"

        file_res = await client.call_tool("exec_file", {"session": "trainer", "file_path": str(tmp_path / "run.py")})
        assert json.loads(file_res.content[0].text)["status"] == "ok"

        files_res = await client.call_tool("list_files", {"session": "trainer"})
        assert json.loads(files_res.content[0].text)[0]["name"] == "data.csv"

        mgr.get_runtime.side_effect = UnauthenticatedError("login required")
        with pytest.raises(ToolError, match="login required"):
            await client.call_tool("exec_code", {"session": "trainer", "code": "1+1"})
