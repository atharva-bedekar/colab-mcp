import importlib.util
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_entrypoint_auth_subcommand_and_mcp_wiring(tmp_path):
    import colab_mcp

    args = colab_mcp.parse_args(["auth"])
    assert args.command == "auth"
    assert args.auth == "oauth2"

    adc_args = colab_mcp.parse_args(["--auth", "adc", "--log", str(tmp_path)])
    assert adc_args.command is None
    assert adc_args.auth == "adc"
    assert adc_args.log == str(tmp_path)

    with (
        patch("colab_mcp.AuthManager") as mock_auth_cls,
        patch.object(colab_mcp.mcp, "run_async", new_callable=AsyncMock) as mock_run,
    ):
        await colab_mcp.main_async(["auth", "--log", str(tmp_path)])
        mock_auth_cls.return_value.run_cli_login.assert_called_once_with()
        mock_run.assert_not_awaited()

    fake_mgr = MagicMock()
    fake_mgr.__aenter__ = AsyncMock(return_value=fake_mgr)
    fake_mgr.__aexit__ = AsyncMock(return_value=None)
    with (
        patch("colab_mcp.SessionManager", return_value=fake_mgr) as mock_mgr_cls,
        patch.object(colab_mcp.mcp, "run_async", new_callable=AsyncMock) as mock_run,
    ):
        await colab_mcp.main_async(["--auth", "adc", "--log", str(tmp_path)])
        mock_mgr_cls.assert_called_once()
        assert mock_mgr_cls.call_args.kwargs["default_auth_provider"].value == "adc"
        fake_mgr.__aenter__.assert_awaited_once()
        mock_run.assert_awaited_once()
        fake_mgr.__aexit__.assert_awaited_once()

    tools = await colab_mcp.mcp.get_tools()
    assert len(tools) == 21
    assert importlib.util.find_spec("colab_mcp.session") is None
    assert importlib.util.find_spec("colab_mcp.websocket_server") is None
