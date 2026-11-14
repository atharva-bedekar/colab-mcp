"""Hybrid OAuth2 / ADC authentication manager for Google Colab."""

from __future__ import annotations

import contextlib
import importlib.resources
import json
import os
import urllib.parse
from pathlib import Path
from typing import Any

import google.auth
import keyring
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from colab_mcp.models import (
    AuthFinishResponse,
    AuthProvider,
    AuthStartResponse,
    AuthStatusResponse,
    UnauthenticatedError,
)

PUBLIC_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
]
REMOTE_REDIRECT_URI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"
KEYRING_SERVICE_NAME = "google-colab-cli"
KEYRING_USERNAME = "oauth_token"


class AuthManager:
    """Manages cached OAuth2 tokens, OS keyring fallback, ADC, and OAuth2 login."""

    def __init__(
        self,
        token_path: Path | None = None,
        cli_fallback_token_path: Path | None = None,
        client_config_path: Path | None = None,
        use_keyring: bool | None = None,
    ) -> None:
        config_root = Path.home() / ".config"
        self.token_path = token_path or (config_root / "colab-mcp" / "token.json")
        self.cli_fallback_token_path = cli_fallback_token_path or (
            config_root / "colab-cli" / "token.json"
        )
        self.client_config_path = client_config_path
        # Isolate custom tmp_path unit tests from the host keyring unless opted in.
        self.use_keyring = (
            (token_path is None and cli_fallback_token_path is None)
            if use_keyring is None
            else use_keyring
        )
        self._pending_flow: InstalledAppFlow | None = None

    def _pending_oauth_path(self) -> Path:
        return self.token_path.parent / "pending_oauth.json"

    def _read_keyring_token(self) -> str | None:
        with contextlib.suppress(Exception):
            return keyring.get_password(KEYRING_SERVICE_NAME, KEYRING_USERNAME)
        return None

    def _write_keyring_token(self, token_json: str) -> None:
        with contextlib.suppress(Exception):
            keyring.set_password(KEYRING_SERVICE_NAME, KEYRING_USERNAME, token_json)

    def load_client_config(self) -> dict[str, Any]:
        """Load OAuth2 client configuration from explicit path or bundled package resource."""
        if self.client_config_path is not None:
            return json.loads(self.client_config_path.read_text(encoding="utf-8"))
        resource = importlib.resources.files("colab_mcp").joinpath("oauth_config.json")
        return json.loads(resource.read_text(encoding="utf-8"))

    def _refresh_credentials(self, creds: Credentials, *, persist: bool) -> bool:
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                return False
            if persist and creds.valid:
                self._write_token_file(creds)
        return bool(creds.valid)

    def _load_oauth2_credentials(self) -> tuple[Credentials | None, Path | str | None]:
        if self.token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(self.token_path))
                if self._refresh_credentials(creds, persist=True):
                    return creds, self.token_path
            except (OSError, ValueError):
                pass

        if self.use_keyring:
            raw_token = self._read_keyring_token()
            if raw_token:
                try:
                    creds = Credentials.from_authorized_user_info(json.loads(raw_token))
                    if self._refresh_credentials(creds, persist=True):
                        return creds, f"keyring:{KEYRING_SERVICE_NAME}/{KEYRING_USERNAME}"
                except (ValueError, json.JSONDecodeError):
                    pass

        if self.cli_fallback_token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(self.cli_fallback_token_path))
                if self._refresh_credentials(creds, persist=False):
                    return creds, self.cli_fallback_token_path
            except (OSError, ValueError):
                pass

        return None, None

    def _write_token_file(self, creds: Credentials) -> None:
        token_json = creds.to_json()
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(self.token_path.parent, 0o700)
        self.token_path.write_text(token_json, encoding="utf-8")
        with contextlib.suppress(OSError):
            os.chmod(self.token_path, 0o600)
        if self.use_keyring:
            self._write_keyring_token(token_json)

    def _save_pending_oauth_state(self, flow: InstalledAppFlow) -> None:
        pending_path = self._pending_oauth_path()
        with contextlib.suppress(OSError):
            pending_path.parent.mkdir(parents=True, exist_ok=True)
            pending_path.write_text(
                json.dumps({"code_verifier": getattr(flow, "code_verifier", None)}),
                encoding="utf-8",
            )
            with contextlib.suppress(OSError):
                os.chmod(pending_path, 0o600)

    def _restore_pending_flow(self) -> InstalledAppFlow | None:
        pending_path = self._pending_oauth_path()
        if not pending_path.exists():
            return None
        try:
            data = json.loads(pending_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        config = self.load_client_config()
        flow = InstalledAppFlow.from_client_config(config, PUBLIC_SCOPES)
        flow.redirect_uri = REMOTE_REDIRECT_URI
        flow.code_verifier = data.get("code_verifier")
        return flow

    def get_status(self, provider: AuthProvider = AuthProvider.OAUTH2) -> AuthStatusResponse:
        """Check whether valid cached OAuth2 or ADC credentials are available."""
        if provider == AuthProvider.ADC:
            try:
                creds, _ = google.auth.default(scopes=PUBLIC_SCOPES)
            except DefaultCredentialsError:
                return AuthStatusResponse(authenticated=False, provider=provider.value)
            if creds.expired and getattr(creds, "refresh_token", None):
                with contextlib.suppress(RefreshError):
                    creds.refresh(Request())
            return AuthStatusResponse(
                authenticated=bool(creds.valid),
                provider=provider.value,
                token_path=None,
                account=getattr(creds, "service_account_email", None),
            )

        creds, location = self._load_oauth2_credentials()
        return AuthStatusResponse(
            authenticated=bool(creds and creds.valid),
            provider=AuthProvider.OAUTH2.value,
            token_path=str(location) if location is not None else None,
            account=None,
        )

    def start_remote_login(self) -> AuthStartResponse:
        """Create a remote copy-paste OAuth2 consent URL without starting a local server."""
        config = self.load_client_config()
        flow = InstalledAppFlow.from_client_config(config, PUBLIC_SCOPES)
        flow.redirect_uri = REMOTE_REDIRECT_URI
        auth_url, _ = flow.authorization_url(prompt="consent", token_usage="remote")
        self._pending_flow = flow
        self._save_pending_oauth_state(flow)
        return AuthStartResponse(
            auth_url=auth_url,
            instructions=(
                "Open auth_url in a browser, approve access, copy the authorization code, "
                "and pass it to auth_finish_login(code)."
            ),
        )

    @staticmethod
    def _extract_authorization_code(code: str) -> str:
        cleaned = code.strip()
        if cleaned.startswith(("http://", "https://")):
            parsed = urllib.parse.urlparse(cleaned)
            query = urllib.parse.parse_qs(parsed.query)
            if "code" in query and query["code"]:
                return query["code"][0].strip()
        return cleaned

    def finish_remote_login(self, code: str) -> AuthFinishResponse:
        """Exchange the pasted authorization code for OAuth2 tokens and persist credentials."""
        flow = self._pending_flow or self._restore_pending_flow()
        if flow is None:
            raise UnauthenticatedError(
                "No pending login flow. Call auth_start_login first."
            )
        cleaned_code = self._extract_authorization_code(code)
        flow.oauth2session.scope = None
        token = flow.fetch_token(code=cleaned_code)
        if isinstance(token, dict):
            flow.oauth2session.scope = token.get("scope")
        creds = flow.credentials
        self._write_token_file(creds)
        self._pending_flow = None
        with contextlib.suppress(OSError):
            self._pending_oauth_path().unlink(missing_ok=True)
        return AuthFinishResponse(authenticated=True, token_path=str(self.token_path))

    def run_cli_login(self, remote: bool = False) -> AuthFinishResponse:
        """Run interactive terminal OAuth2 login (local browser loopback by default)."""
        if remote:
            start = self.start_remote_login()
            print(f"Open this URL in your browser to authenticate:\n\n{start.auth_url}\n")
            code = input("Enter authorization code: ")
            return self.finish_remote_login(code)

        config = self.load_client_config()
        flow = InstalledAppFlow.from_client_config(config, PUBLIC_SCOPES)
        creds = flow.run_local_server(port=0)
        self._write_token_file(creds)
        return AuthFinishResponse(authenticated=True, token_path=str(self.token_path))

    def get_authorized_session(
        self,
        provider: AuthProvider = AuthProvider.OAUTH2,
    ) -> AuthorizedSession:
        """Return an authenticated AuthorizedSession or raise UnauthenticatedError."""
        if provider == AuthProvider.ADC:
            try:
                creds, _ = google.auth.default(scopes=PUBLIC_SCOPES)
            except DefaultCredentialsError as exc:
                raise UnauthenticatedError(
                    "Application Default Credentials not found. Run 'gcloud auth application-default login' "
                    "or authenticate with OAuth2 via auth_start_login / 'colab-mcp auth'."
                ) from exc
            if creds.expired and getattr(creds, "refresh_token", None):
                creds.refresh(Request())
            if not creds.valid:
                raise UnauthenticatedError("ADC credentials are invalid or expired.")
            return AuthorizedSession(creds)

        creds, _ = self._load_oauth2_credentials()
        if creds is None or not creds.valid:
            raise UnauthenticatedError(
                "Not authenticated with Google Colab. Call auth_start_login and auth_finish_login, "
                "or run 'colab-mcp auth' in a terminal."
            )
        return AuthorizedSession(creds)
