import json
import urllib.parse
from unittest.mock import MagicMock, PropertyMock, patch

import pytest


def test_bundled_config_and_remote_login_flow(tmp_path, monkeypatch):
    from colab_mcp.auth import REMOTE_REDIRECT_URI, AuthManager

    monkeypatch.chdir(tmp_path)
    token_path = tmp_path / "cfg" / "token.json"
    mgr = AuthManager(
        token_path=token_path,
        cli_fallback_token_path=tmp_path / "cli_token.json",
    )
    cfg = mgr.load_client_config()
    assert "installed" in cfg
    assert cfg["installed"]["client_id"].startswith("764086051850-")

    start = mgr.start_remote_login()
    assert "sdk.cloud.google.com/applicationdefaultauthcode.html" in urllib.parse.unquote(start.auth_url)
    assert "token_usage=remote" in start.auth_url
    assert mgr._pending_flow.redirect_uri == REMOTE_REDIRECT_URI

    fake_creds = MagicMock()
    fake_creds.to_json.return_value = json.dumps({"token": "access-123", "refresh_token": "ref-1"})
    with (
        patch.object(mgr._pending_flow, "fetch_token") as mock_fetch,
        patch.object(type(mgr._pending_flow), "credentials", new_callable=PropertyMock, return_value=fake_creds),
    ):
        finish = mgr.finish_remote_login("  test-code  ")
        mock_fetch.assert_called_once_with(code="test-code")
    assert finish.authenticated is True
    assert finish.token_path == str(token_path)
    assert mgr._pending_flow is None
    assert json.loads(token_path.read_text(encoding="utf-8"))["token"] == "access-123"


def test_get_status_fallback_and_unauthenticated(tmp_path):
    from colab_mcp.auth import AuthManager
    from colab_mcp.models import UnauthenticatedError

    primary = tmp_path / "primary.json"
    fallback = tmp_path / "fallback.json"
    mgr = AuthManager(token_path=primary, cli_fallback_token_path=fallback)

    status = mgr.get_status()
    assert status.authenticated is False
    assert status.token_path is None

    with pytest.raises(UnauthenticatedError):
        mgr.get_authorized_session()

    fallback.write_text('{"token": "abc"}', encoding="utf-8")
    fake_creds = MagicMock(valid=True, expired=False, refresh_token="ref")
    with patch("colab_mcp.auth.Credentials.from_authorized_user_file", return_value=fake_creds):
        status = mgr.get_status()
        assert status.authenticated is True
        assert status.token_path == str(fallback)
        session = mgr.get_authorized_session()
        assert session is not None


def test_adc_status(tmp_path):
    from colab_mcp.auth import AuthManager
    from colab_mcp.models import AuthProvider

    mgr = AuthManager(token_path=tmp_path / "missing.json")
    fake_creds = MagicMock(valid=True, expired=False, service_account_email="sa@example.iam.gserviceaccount.com")
    with patch("google.auth.default", return_value=(fake_creds, "proj-1")):
        status = mgr.get_status(provider=AuthProvider.ADC)
        assert status.authenticated is True
        assert status.provider == "adc"
        assert status.account == "sa@example.iam.gserviceaccount.com"


def test_keyring_loading_and_local_cli_login(tmp_path):
    from colab_mcp.auth import AuthManager

    token_path = tmp_path / "cfg" / "token.json"
    mgr = AuthManager(
        token_path=token_path,
        cli_fallback_token_path=tmp_path / "cli_token.json",
        use_keyring=True,
    )
    fake_creds = MagicMock(valid=True, expired=False, refresh_token="ref-kr")
    fake_creds.to_json.return_value = json.dumps({"token": "kr-access", "refresh_token": "ref-kr"})

    with (
        patch("colab_mcp.auth.keyring.get_password", return_value=fake_creds.to_json.return_value),
        patch("colab_mcp.auth.Credentials.from_authorized_user_info", return_value=fake_creds),
    ):
        status = mgr.get_status()
        assert status.authenticated is True
        assert status.token_path == "keyring:google-colab-cli/oauth_token"

    mock_flow = MagicMock()
    mock_flow.run_local_server.return_value = fake_creds
    with (
        patch("colab_mcp.auth.InstalledAppFlow.from_client_config", return_value=mock_flow),
        patch("colab_mcp.auth.keyring.set_password") as mock_set_pw,
    ):
        finish = mgr.run_cli_login()
        mock_flow.run_local_server.assert_called_once_with(port=0)
        mock_set_pw.assert_called_once()
        assert finish.authenticated is True
        assert json.loads(token_path.read_text(encoding="utf-8"))["token"] == "kr-access"
