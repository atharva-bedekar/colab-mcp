# Headless Google Colab REST & Jupyter Kernel MCP Server Implementation Plan

> **Execution:** Hand off to the AGY orchestrator using bundled subagent-driven-development (or executing-plans in a fresh session). Workers check and self-review; only the controller commits. Independent review and verification follow the last wave. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the browser-dependent WebSocket proxy in `colab_mcp` with a standalone, zero-CLI-dependency Headless Google Colab MCP Server providing runtime lifecycle, file I/O, and configurable kernel execution.

**Architecture:** Port the four core backend layers (`auth.py`, `client.py`, `contents.py`, `runtime.py`) on top of shared Pydantic models (`models.py`), coordinate persistent session state and an in-process `asyncio` keep-alive loop in `state.py`, and expose 14 FastMCP tools in `tools.py` wired through `src/colab_mcp/__init__.py`.

**Tech Stack:** Python >=3.13, `fastmcp==2.14.5`, `mcp[cli]>=1.20.0`, `pydantic>=2.12.3,<3.0.0`, `requests>=2.32.5`, `google-auth-oauthlib>=0.7.0`, `jupyter-kernel-client>=0.9.0`, `pytest>=8.4.2`, `pytest-asyncio>=1.2.0`.

---

## Execution Waves

| Wave | Tasks | Description | Files Touched (Disjoint) |
|---|---|---|---|
| Wave 1 | Task 1 | Foundational enums, Pydantic models, XSSI & `.ipynb` JSON helpers | `src/colab_mcp/models.py`, `tests/test_models.py` |
| Wave 2 | Task 2, Task 3, Task 4, Task 5 | Independent core layers: Auth (`auth.py`), Control-Plane (`client.py`), Contents (`contents.py`), Kernel Runtime (`runtime.py`) | `src/colab_mcp/oauth_config.json`, `src/colab_mcp/auth.py`, `tests/test_auth.py`, `src/colab_mcp/client.py`, `tests/test_client.py`, `src/colab_mcp/contents.py`, `tests/test_contents.py`, `src/colab_mcp/runtime.py`, `tests/test_runtime.py` |
| Wave 3 | Task 6 | Persistent session state (`sessions.json`), proxy token refresh, and background `asyncio` keep-alive loop | `src/colab_mcp/state.py`, `tests/test_state.py` |
| Wave 4 | Task 7 | FastMCP registration for all 14 tools across Auth, Lifecycle, Files, and Execution | `src/colab_mcp/tools.py`, `tests/test_tools.py` |
| Wave 5 | Task 8 | Entrypoint (`__init__.py`) + `colab-mcp auth` CLI command, `pyproject.toml` cleanup, legacy WebSocket proxy removal, and E2E verification | `src/colab_mcp/__init__.py`, `pyproject.toml`, `tests/test_entrypoint.py`, delete `src/colab_mcp/session.py`, `src/colab_mcp/websocket_server.py`, `tests/session_test.py`, `tests/websocket_server_test.py` |

---

## Grounded Codebase & Library References

- Existing entrypoint & CLI argument parser: `parse_args`, `init_logger`, `main_async`, `main` in `src/colab_mcp/__init__.py:31-87` (creates `FastMCP(name="ColabMCP")` at `src/colab_mcp/__init__.py:28`).
- Legacy browser proxy modules to remove in Wave 5: `ColabSessionProxy` in `src/colab_mcp/session.py:352-376`, `ColabWebSocketServer` in `src/colab_mcp/websocket_server.py:36-162`, and old unit tests in `tests/session_test.py:1-323` and `tests/websocket_server_test.py:1-222`.
- Installed `jupyter_kernel_client` signatures verified in `.venv`:
  - `KernelClient.__init__(self, kernel_id: str | None = None, log: logging.Logger | None = None, **kwargs)` in `.venv/Lib/site-packages/jupyter_kernel_client/client.py:162-170`
  - `KernelClient.start(self, name: str = "python3", path: str | None = None, timeout: float = REQUEST_TIMEOUT)` in `.venv/Lib/site-packages/jupyter_kernel_client/client.py:395-416`
  - `KernelClient.execute(self, code: str, silent: bool = False, store_history: bool = True, user_expressions: dict | None = None, allow_stdin: bool | None = False, stop_on_error: bool = True, timeout: float = REQUEST_TIMEOUT, stdin_hook: Callable | None = None, variables: dict | None = None) -> dict[str, Any]` in `.venv/Lib/site-packages/jupyter_kernel_client/client.py:245-311`
  - `KernelClient.restart(self, timeout: float = REQUEST_TIMEOUT) -> None` in `.venv/Lib/site-packages/jupyter_kernel_client/client.py:384-386`
  - `KernelClient.stop(self, shutdown_kernel: bool | None = None, shutdown_now: bool = True, timeout: float = REQUEST_TIMEOUT) -> None` in `.venv/Lib/site-packages/jupyter_kernel_client/client.py:418-438`
  - `KernelHttpManager.__init__(self, server_url: str, token: str | None, username: str = ..., kernel_id: str | None = None, client_kwargs: dict | None = None, headers: dict | None = None, **kwargs)` in `.venv/Lib/site-packages/jupyter_kernel_client/manager.py:66-102`
- Installed `google_auth_oauthlib` signatures verified in `.venv`:
  - `Flow` and `InstalledAppFlow.from_client_config(client_config, scopes, **kwargs)` in `.venv/Lib/site-packages/google_auth_oauthlib/flow.py:74-120`.

---

## Tasks

### Task 1: Domain Models, Enums, XSSI & Notebook Helpers (`models.py`)

**Wave:** 1  
**Depends on:** None  
**Files Touched:**
- Create: `src/colab_mcp/models.py`
- Test: `tests/test_models.py`

#### Interface & Schema Specifications
Define all shared enums, exceptions, Pydantic models, and pure helper functions in `src/colab_mcp/models.py`:
1. **Enums:**
   - `AuthProvider(str, Enum)`: `OAUTH2 = "oauth2"`, `ADC = "adc"`
   - `Variant(str, Enum)`: `DEFAULT = "DEFAULT"`, `GPU = "GPU"`, `TPU = "TPU"`
   - `AssignmentVariant(int, Enum)`: `DEFAULT = 0`, `GPU = 1`, `TPU = 2`
   - `Accelerator(str, Enum)`: `NONE = "NONE"`, `T4 = "T4"`, `L4 = "L4"`, `A100 = "A100"`, `H100 = "H100"`, `V5E1 = "V5E1"`, `V6E1 = "V6E1"`
   - `Shape(int, Enum)`: `STANDARD = 0`, `HIGH_RAM = 1`
2. **Exceptions:**
   - `ColabError(Exception)`: Base error for Colab MCP operations.
   - `ColabRequestError(ColabError)`: Stores `status_code: int | None` and `message: str`.
   - `TooManyAssignmentsError(ColabRequestError)`: Raised when `/tun/m/assign` returns HTTP `412`.
   - `UnauthenticatedError(ColabError)`: Raised when neither valid OAuth2 token nor ADC credentials exist.
3. **Pydantic Models (`pydantic.BaseModel`, `ConfigDict(populate_by_name=True)`):**
   - `RuntimeProxyInfo`: `token: str`, `token_expires_in_seconds: int = Field(default=3600, alias="tokenExpiresInSeconds")`, `url: str`. Include `compute_expires_at(now: datetime | None = None) -> datetime` returning UTC timestamp.
   - `GetAssignmentTokenResponse`: `token: str`, `variant: Variant | str = Variant.DEFAULT`, `accelerator: str = "NONE"`.
   - `Assignment`: `endpoint: str`, `accelerator: str = "NONE"`, `variant: Variant | str = Variant.DEFAULT`, `machine_shape: Shape = Field(default=Shape.STANDARD, alias="machineShape")`, `runtime_proxy_info: RuntimeProxyInfo = Field(alias="runtimeProxyInfo")`.
   - `PostAssignmentResponse`: Same fields as `Assignment` returned on new POST allocation.
   - `ListedAssignment`: `endpoint: str`, `accelerator: str = "NONE"`, `variant: AssignmentVariant | int = AssignmentVariant.DEFAULT`, `machine_shape: Shape = Field(default=Shape.STANDARD, alias="machineShape")`, `runtime_proxy_info: RuntimeProxyInfo | None = Field(default=None, alias="runtimeProxyInfo")`.
   - `SessionState`: `name: str`, `notebook_hash: str`, `endpoint: str`, `variant: str`, `accelerator: str`, `shape: str`, `runtime_url: str`, `proxy_token: str`, `proxy_expires_at: str`, `kernel_id: str | None = None`, `running: str | None = None`, `last_execution: tuple[str, str | None, str] | None = None`.
   - Tool response schemas: `AuthStatusResponse`, `AuthStartResponse`, `AuthFinishResponse`, `SessionInfo`, `SessionSummary`, `SessionStatus`, `RemoteFileEntry`, `ExecutionResult`, `FileExecutionResult`.
4. **Pure Helper Functions:**
   - `strip_xssi_prefix(text: str) -> str`: Removes leading `)]}'\n` or `)]}'` if present.
   - `uuid_to_web_safe_base64(u: uuid.UUID) -> str`: Returns `base64.urlsafe_b64encode(u.bytes).decode("ascii").rstrip("=")` with `.` replacing `-` or standard URL-safe base64 matching Colab `nbh` format (`urlsafe_b64encode(u.bytes).decode("utf-8").rstrip("=")`).
   - `resolve_runtime_options(gpu: str | None = None, tpu: str | None = None, *, high_mem: bool = False) -> tuple[Variant, Accelerator, Shape | None]`: Raises `ValueError("Cannot specify both GPU and TPU")` if both `gpu` and `tpu` are non-null; maps `gpu` (`T4`, `L4`, `A100`, `H100`) or `tpu` (`v5e1`, `v6e1`) and `high_mem` (`None` for single-shape accelerators `L4`, `V5E1`, `V6E1`; `Shape.HIGH_RAM` when `high_mem=True` on `NONE`/`T4`/`A100`/`H100`).
   - `parse_notebook_code_cells(nb_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]` and `write_notebook_with_outputs(nb_data: dict[str, Any], cell_outputs: dict[int, list[dict[str, Any]]], out_path: Path) -> None`: Reads/writes `.ipynb` v4 JSON using stdlib `json`.

- [x] **Step 1: Write the failing test (`tests/test_models.py`)**
  Test `strip_xssi_prefix`, `resolve_runtime_options` (covering CPU default, T4 + high_mem -> `Shape.HIGH_RAM`, L4 + high_mem -> `None`, TPU `v5e1`, and `ValueError` when both `gpu` and `tpu` are set), `RuntimeProxyInfo` alias parsing, and `.ipynb` round-trip cell extraction/saving using `tmp_path`.

```python
def test_strip_xssi_and_resolve_options(tmp_path):
    import pytest
    from colab_mcp.models import Accelerator, Shape, Variant, resolve_runtime_options, strip_xssi_prefix

    assert strip_xssi_prefix(")]}'\n{\"a\": 1}") == '{"a": 1}'
    assert resolve_runtime_options(gpu="T4", high_mem=True) == (Variant.GPU, Accelerator.T4, Shape.HIGH_RAM)
    assert resolve_runtime_options(gpu="L4", high_mem=True) == (Variant.GPU, Accelerator.L4, None)
    with pytest.raises(ValueError):
        resolve_runtime_options(gpu="T4", tpu="v5e1")
```

- [x] **Step 2: Run test to verify it fails (Red)**
  Run: `uv run pytest tests/test_models.py -v`
  Expected: FAIL with `ModuleNotFoundError: No module named 'colab_mcp.models'`.

- [x] **Step 3: Write minimal implementation (`src/colab_mcp/models.py`)**
  Implement all enums, exceptions, Pydantic schemas, and pure helper functions specified above.

- [x] **Step 4: Run test to verify it passes (Green)**
  Run: `uv run pytest tests/test_models.py -v`
  Expected: PASS (all model, option-resolution, XSSI, and `.ipynb` helper tests pass).

- [x] **Step 5: Self-review and report to the controller**
  Report status, changed files, and test output; leave changes uncommitted.

---

### Task 2: Hybrid OAuth2 / ADC Authentication Layer (`auth.py` + `oauth_config.json`)

**Wave:** 2  
**Depends on:** Task 1  
**Files Touched:**
- Create: `src/colab_mcp/oauth_config.json`
- Create: `src/colab_mcp/auth.py`
- Test: `tests/test_auth.py`

#### Interface & Schema Specifications
1. **`src/colab_mcp/oauth_config.json`:**
   Bundled Google Cloud SDK installed-app OAuth configuration with `client_id` `"764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com"`, `client_secret` `"d-FL95Q19q7MQmFpd7hHD0Ty"`, `auth_uri` `"https://accounts.google.com/o/oauth2/auth"`, `token_uri` `"https://oauth2.googleapis.com/token"`, and `redirect_uris` `["http://localhost", "https://sdk.cloud.google.com/applicationdefaultauthcode.html"]`.
2. **`src/colab_mcp/auth.py`:**
   - Constants:
     - `PUBLIC_SCOPES = ["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/colaboratory", "https://www.googleapis.com/auth/cloud-platform"]`
     - `REMOTE_REDIRECT_URI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"`
   - Class `AuthManager`:
     - `__init__(self, token_path: Path | None = None, cli_fallback_token_path: Path | None = None, client_config_path: Path | None = None)`: Defaults `token_path` to `Path.home() / ".config" / "colab-mcp" / "token.json"` and `cli_fallback_token_path` to `Path.home() / ".config" / "colab-cli" / "token.json"`. When `client_config_path` is `None`, loads the bundled config via `importlib.resources.files("colab_mcp").joinpath("oauth_config.json")` (falling back to `~/.colab-cli-oauth-config.json` only if explicitly configured or absent, with bundled resource as guaranteed default).
     - `load_client_config(self) -> dict[str, Any]`: Reads `client_config_path` if provided, otherwise parses JSON from `importlib.resources.files("colab_mcp").joinpath("oauth_config.json")`.
     - `get_status(self, provider: AuthProvider = AuthProvider.OAUTH2) -> AuthStatusResponse`: Checks `token_path`, then `cli_fallback_token_path` (refreshing if expired with `refresh_token`), or `google.auth.default(scopes=PUBLIC_SCOPES)` when `provider == AuthProvider.ADC`. Never blocks on `input()`.
     - `start_remote_login(self) -> AuthStartResponse`: Loads client config via `load_client_config()`, instantiates `InstalledAppFlow.from_client_config(config, scopes=PUBLIC_SCOPES, redirect_uri=REMOTE_REDIRECT_URI)` (ref `.venv/Lib/site-packages/google_auth_oauthlib/flow.py:74-120`), calls `flow.authorization_url(prompt="consent", access_type="offline", token_usage="remote")`, stores `self._pending_flow = flow`, and returns `AuthStartResponse(auth_url=auth_url, instructions=...)`.
     - `finish_remote_login(self, code: str) -> AuthFinishResponse`: Calls `self._pending_flow.fetch_token(code=code.strip())`, writes `credentials.to_json()` to `self.token_path` (creating parent dirs with `0o700` where supported), clears `self._pending_flow`, and returns `AuthFinishResponse(authenticated=True, token_path=str(self.token_path))`.
     - `run_cli_login(self) -> AuthFinishResponse`: Calls `start_remote_login()`, prints URL to stdout, reads code via `input("Enter authorization code: ")`, and calls `finish_remote_login(code)`.
     - `get_authorized_session(self, provider: AuthProvider = AuthProvider.OAUTH2) -> google.auth.transport.requests.AuthorizedSession`: Returns an authenticated `AuthorizedSession` or raises `UnauthenticatedError` with clear instructions to call `auth_start_login` or run `colab-mcp auth`.

- [x] **Step 1: Write the failing test (`tests/test_auth.py`)**
  Using `tmp_path` for `token_path` and `cli_fallback_token_path` and `unittest.mock.patch` on `InstalledAppFlow`, verify:
  1. Default `AuthManager()` loads bundled `oauth_config.json` via `importlib.resources.files("colab_mcp")` regardless of working directory.
  2. `start_remote_login()` sets `redirect_uri == REMOTE_REDIRECT_URI` and passes `token_usage="remote"` without starting a localhost server.
  3. `finish_remote_login("test-code")` exchanges the code and writes `token.json` inside `tmp_path`.
  4. `get_status()` falls back to `cli_fallback_token_path` when primary `token_path` does not exist.
  5. `get_authorized_session()` raises `UnauthenticatedError` when no token file exists.

```python
def test_remote_login_flow_writes_token(tmp_path, monkeypatch):
    from colab_mcp.auth import AuthManager, REMOTE_REDIRECT_URI

    monkeypatch.chdir(tmp_path)
    mgr = AuthManager(token_path=tmp_path / "token.json", cli_fallback_token_path=tmp_path / "cli_token.json")
    cfg = mgr.load_client_config()
    assert "installed" in cfg
    start = mgr.start_remote_login()
    assert "sdk.cloud.google.com/applicationdefaultauthcode.html" in start.auth_url
    assert mgr._pending_flow.redirect_uri == REMOTE_REDIRECT_URI
```

- [x] **Step 2: Run test to verify it fails (Red)**
  Run: `uv run pytest tests/test_auth.py -v`
  Expected: FAIL with `ModuleNotFoundError: No module named 'colab_mcp.auth'`.

- [x] **Step 3: Write minimal implementation (`src/colab_mcp/oauth_config.json` and `src/colab_mcp/auth.py`)**
  Create `src/colab_mcp/oauth_config.json` and implement `AuthManager` in `src/colab_mcp/auth.py`.

- [x] **Step 4: Run test to verify it passes (Green)**
  Run: `uv run pytest tests/test_auth.py -v`
  Expected: PASS.

- [x] **Step 5: Self-review and report to the controller**
  Report status, changed files, and test output; leave changes uncommitted.

---

### Task 3: Colab Session Control-Plane Client (`client.py`)

**Wave:** 2  
**Depends on:** Task 1  
**Files Touched:**
- Create: `src/colab_mcp/client.py`
- Test: `tests/test_client.py`

#### Interface & Schema Specifications
Implement `ColabClient` in `src/colab_mcp/client.py` consuming `requests.Session` and models from `src/colab_mcp/models.py`:
- Constants:
  - `COLAB_RESEARCH_DOMAIN = "https://colab.research.google.com"`
  - `COLAB_SANDBOX_DOMAIN = "https://colab.sandbox.google.com"`
  - `TUN_PREFIX = "/tun/m"`
  - `DEFAULT_REQUEST_TIMEOUT = 120.0`
- Methods on `ColabClient`:
  - `__init__(self, session: requests.Session, colab_domain: str = COLAB_RESEARCH_DOMAIN, sandbox_domain: str = COLAB_SANDBOX_DOMAIN, request_timeout: float | None = None)`
  - `_issue_request(self, method: str, url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None, json_data: Any = None, timeout: float | None = None) -> dict[str, Any] | list[Any] | None`:
    - Injects headers `{"Accept": "application/json", "X-Colab-Client-Agent": "colab-mcp", "X-Colab-Tunnel": "Google"}`.
    - Raises `TooManyAssignmentsError` on HTTP `412` and `ColabRequestError(status_code=resp.status_code, message=...)` on other HTTP 4xx/5xx responses.
    - Returns `None` on HTTP `204` or empty body; otherwise strips XSSI prefix via `strip_xssi_prefix(resp.text)` and returns `json.loads(...)`.
  - `_build_assign_params(notebook_hash: uuid.UUID, variant: Variant, accelerator: Accelerator, shape: Shape | None) -> dict[str, str]`:
    - Always sets `"nbh": uuid_to_web_safe_base64(notebook_hash)` and `"authuser": "0"`.
    - Adds `"variant": variant.value` when `variant != Variant.DEFAULT`, `"accelerator": accelerator.value` when `accelerator != Accelerator.NONE`, and `"shape": "hm"` when `shape == Shape.HIGH_RAM`.
  - `get_assignment(self, notebook_hash: uuid.UUID, variant: Variant = Variant.DEFAULT, accelerator: Accelerator = Accelerator.NONE, shape: Shape | None = None) -> Assignment | GetAssignmentTokenResponse`:
    - `GET {colab_domain}/tun/m/assign` with `_build_assign_params`. If response dict contains `"runtimeProxyInfo"` (or `"endpoint"`), validates and returns `Assignment`; otherwise returns `GetAssignmentTokenResponse`.
  - `assign(self, notebook_hash: uuid.UUID, variant: Variant = Variant.DEFAULT, accelerator: Accelerator = Accelerator.NONE, shape: Shape | None = None, retries: int = 3, backoff: float = 10.0, sleep_fn: Callable[[float], None] = time.sleep) -> Assignment`:
    - Calls `get_assignment(...)`. If already an `Assignment`, returns it.
    - Otherwise `POST {colab_domain}/tun/m/assign` with header `{"X-Goog-Colab-Token": token_resp.token}` and the same query params.
    - Retries on `requests.exceptions.ReadTimeout` or `ColabRequestError` with `status_code in (502, 503, 504)` up to `retries` times, sleeping `backoff * (2 ** attempt)` via `sleep_fn`, reconciling via `get_assignment(...)` before re-POSTing. Re-raises `TooManyAssignmentsError` immediately on `412`.
  - `list_assignments(self) -> list[ListedAssignment]`:
    - `GET {colab_domain}/tun/m/assignments?authuser=0`, strips XSSI, parses `"assignments"` list (or top-level list) into `list[ListedAssignment]`.
  - `unassign(self, endpoint: str) -> None`:
    - Step 1: `GET {sandbox_domain}/tun/m/unassign/{endpoint}?authuser=0` to retrieve `{"token": "..."}`.
    - Step 2: `POST {sandbox_domain}/tun/m/unassign/{endpoint}?authuser=0` with header `{"X-Goog-Colab-Token": token}`.
  - `keep_alive(self, endpoint: str, timeout: float = 15.0) -> bool`:
    - `GET {colab_domain}/tun/m/{endpoint}/keep-alive/?authuser=0` with header `{"X-Colab-Tunnel": "Google"}`. Treats `requests.exceptions.ReadTimeout` as `True` (Colab TFE records activity before forwarding to the VM).

- [x] **Step 1: Write the failing test (`tests/test_client.py`)**
  Test two-step `assign` (GET token -> POST with `X-Goog-Colab-Token` and XSSI-prefixed response), 503 retry recovery, 412 `TooManyAssignmentsError`, two-step `unassign`, `list_assignments`, and `keep_alive` (`ReadTimeout` returning `True`).

```python
def test_assign_two_step_handshake_and_xssi():
    import uuid
    from unittest.mock import MagicMock
    from colab_mcp.client import ColabClient
    from colab_mcp.models import Accelerator, Shape, Variant

    sess = MagicMock()
    get_resp = MagicMock(status_code=200, text=")]}'\n{\"token\": \"xsrf-123\"}")
    post_resp = MagicMock(
        status_code=200,
        text=")]}'\n{\"endpoint\": \"gpu-t4-s-1\", \"accelerator\": \"T4\", \"variant\": \"GPU\", \"runtimeProxyInfo\": {\"token\": \"p-tok\", \"tokenExpiresInSeconds\": 3600, \"url\": \"https://8080-gpu.colab.googleusercontent.com\"}}",
    )
    sess.request.side_effect = [get_resp, post_resp]
    client = ColabClient(sess)
    res = client.assign(uuid.uuid4(), Variant.GPU, Accelerator.T4, Shape.HIGH_RAM)
    assert res.endpoint == "gpu-t4-s-1"
    assert res.runtime_proxy_info.token == "p-tok"
```

- [x] **Step 2: Run test to verify it fails (Red)**
  Run: `uv run pytest tests/test_client.py -v`
  Expected: FAIL with `ModuleNotFoundError: No module named 'colab_mcp.client'`.

- [x] **Step 3: Write minimal implementation (`src/colab_mcp/client.py`)**
  Implement `ColabClient` with all methods and retry/error semantics specified above.

- [x] **Step 4: Run test to verify it passes (Green)**
  Run: `uv run pytest tests/test_client.py -v`
  Expected: PASS.

- [x] **Step 5: Self-review and report to the controller**
  Report status, changed files, and test output; leave changes uncommitted.

---

### Task 4: Remote Filesystem Contents Client (`contents.py`)

**Wave:** 2  
**Depends on:** Task 1  
**Files Touched:**
- Create: `src/colab_mcp/contents.py`
- Test: `tests/test_contents.py`

#### Interface & Schema Specifications
Implement `ContentsClient` in `src/colab_mcp/contents.py` communicating with `<base_url>/api/contents/<quoted_path>`:
- `__init__(self, base_url: str, proxy_token: str, session: requests.Session | None = None, timeout: float = 60.0)`
- `_request(self, method: str, remote_path: str, *, params: dict[str, str] | None = None, json_data: dict[str, Any] | None = None) -> dict[str, Any] | None`:
  - Normalizes `quoted_path = urllib.parse.quote(remote_path.strip("/"), safe="/")`.
  - Sends request to `{base_url.rstrip("/")}/api/contents/{quoted_path}` with query params `{"authuser": "0", "colab-runtime-proxy-token": self.proxy_token, ...(params or {})}` and headers `{"X-Colab-Runtime-Proxy-Token": self.proxy_token, "X-Colab-Client-Agent": "colab-mcp"}`.
  - Raises `FileNotFoundError(f"Remote path not found: {remote_path}")` on HTTP `404` and `ColabRequestError` on other HTTP errors.
  - Returns `None` on `DELETE`; otherwise returns `response.json()`.
- `list_dir(self, remote_path: str = "/content") -> list[RemoteFileEntry]`:
  - Calls `self._request("GET", remote_path)`. If response `"type" == "directory"`, maps `data.get("content", [])` into `list[RemoteFileEntry]`; if single file, returns a single-element list.
- `upload(self, local_path: str | Path, remote_path: str) -> dict[str, Any]`:
  - Reads bytes from `Path(local_path)` (raising `FileNotFoundError` if missing), base64-encodes content, and calls `self._request("PUT", remote_path, json_data={"name": filename, "path": remote_path, "type": "file", "format": "base64", "content": b64_content, "chunk": 1})`.
  - Returns `{"name": filename, "path": remote_path, "size_bytes": len(raw_bytes)}`.
- `download(self, remote_path: str, local_path: str | Path) -> dict[str, Any]`:
  - Calls `self._request("GET", remote_path, params={"content": "1"})`.
  - Raises `IsADirectoryError` if `data.get("type") == "directory"`.
  - Decodes `"base64"` or `"text"` (UTF-8) payload, creates parent directories for `Path(local_path)`, writes bytes, and returns `{"remote_path": remote_path, "local_path": str(local_path), "size_bytes": len(content_bytes)}`.
- `delete(self, remote_path: str) -> dict[str, Any]`:
  - Calls `self._request("DELETE", remote_path)` and returns `{"deleted": True, "remote_path": remote_path}`.

- [x] **Step 1: Write the failing test (`tests/test_contents.py`)**
  Using `tmp_path` for real local files, test `upload`, `download` (both `base64` and `text` formats, plus `IsADirectoryError` on directory), `list_dir`, `delete`, and `404` -> `FileNotFoundError`.

```python
def test_upload_and_download_roundtrip(tmp_path):
    from unittest.mock import MagicMock
    from colab_mcp.contents import ContentsClient

    src_file = tmp_path / "sample.bin"
    src_file.write_bytes(b"\x00\x01colab")
    sess = MagicMock()
    sess.request.return_value = MagicMock(status_code=200, json=lambda: {"name": "sample.bin", "path": "/content/sample.bin"})
    client = ContentsClient("https://proxy.colab.google", "tok-1", session=sess)
    res = client.upload(src_file, "/content/sample.bin")
    assert res["size_bytes"] == 7
```

- [x] **Step 2: Run test to verify it fails (Red)**
  Run: `uv run pytest tests/test_contents.py -v`
  Expected: FAIL with `ModuleNotFoundError: No module named 'colab_mcp.contents'`.

- [x] **Step 3: Write minimal implementation (`src/colab_mcp/contents.py`)**
  Implement `ContentsClient` satisfying all file operation contracts.

- [x] **Step 4: Run test to verify it passes (Green)**
  Run: `uv run pytest tests/test_contents.py -v`
  Expected: PASS.

- [x] **Step 5: Self-review and report to the controller**
  Report status, changed files, and test output; leave changes uncommitted.

---

### Task 5: Jupyter Kernel Execution Runtime (`runtime.py`)

**Wave:** 2  
**Depends on:** Task 1  
**Files Touched:**
- Create: `src/colab_mcp/runtime.py`
- Test: `tests/test_runtime.py`

#### Interface & Schema Specifications
Implement `ColabRuntime` in `src/colab_mcp/runtime.py` wrapping `jupyter_kernel_client.KernelClient` (ref `.venv/Lib/site-packages/jupyter_kernel_client/client.py:108-438` and `.venv/Lib/site-packages/jupyter_kernel_client/manager.py:66-102`):
- `__init__(self, runtime_url: str, proxy_token: str, kernel_id: str | None = None, startup_retries: int = 3, startup_backoff: float = 2.0, kernel_client_factory: Callable[..., Any] = KernelClient)`
- Lazy `kernel_client` property:
  - Instantiates `self._kernel_client = self._kernel_client_factory(server_url=self.runtime_url, token=self.proxy_token, kernel_id=self.kernel_id, headers={"X-Colab-Runtime-Proxy-Token": self.proxy_token, "X-Colab-Client-Agent": "colab-mcp"})`.
  - Sets `self._kernel_client._own_kernel = False` so disconnecting channels never shuts down the remote kernel.
  - Calls `self._kernel_client.start()` with retry on `requests.exceptions.ReadTimeout` / `requests.exceptions.ConnectTimeout` up to `startup_retries` times, sleeping `startup_backoff ** (i + 1)` seconds, and updates `self.kernel_id = self._kernel_client.id`.
- `ensure_content_dir(self) -> None`:
  - Runs `self.execute_code("import os; os.makedirs('/content', exist_ok=True); os.chdir('/content')", timeout=30.0, ensure_cwd=False)`.
- `execute_code(self, code: str, *, timeout: float = 300.0, allow_stdin: bool = False, env: dict[str, str] | None = None, ensure_cwd: bool = True) -> ExecutionResult`:
  - If `ensure_cwd` is `True`, runs `ensure_content_dir()` once per connection.
  - Prepends `_build_env_prelude(env)` (`import os\nos.environ[...] = ...\n`) when `env` is non-empty.
  - Calls `self.kernel_client.execute(full_code, allow_stdin=allow_stdin, timeout=timeout)` (ref `.venv/Lib/site-packages/jupyter_kernel_client/client.py:245-311`).
  - Aggregates `stdout` (`output_type == "stream"` and `name == "stdout"`), `stderr` (`name == "stderr"`), and `error` (`output_type == "error"` or synthesized when `reply.get("status") == "error"`), and returns `ExecutionResult(status=status, outputs=outputs, stdout=stdout, stderr=stderr, error=error_summary, duration_s=elapsed)`.
- `execute_file(self, file_path: str | Path, *, timeout: float = 1800.0, allow_stdin: bool = False, env: dict[str, str] | None = None, save_output_notebook: bool = True) -> FileExecutionResult`:
  - Validates `Path(file_path)` exists locally.
  - If `.py`: executes file contents as a single block via `execute_code(...)` and returns `FileExecutionResult`.
  - If `.ipynb`: uses `parse_notebook_code_cells` from `src/colab_mcp/models.py`, executes each code cell sequentially (stopping on first cell with `status == "error"`), and when `save_output_notebook=True`, writes `<stem>_output.ipynb` via `write_notebook_with_outputs`.
- `restart(self) -> str | None`:
  - Calls `self.kernel_client.restart()` (ref `.venv/Lib/site-packages/jupyter_kernel_client/client.py:384-386`) and returns `self.kernel_id`.
- `get_kernel_state(self) -> str`:
  - Calls `self.kernel_client._manager.refresh_model()` (ref `.venv/Lib/site-packages/jupyter_kernel_client/manager.py:159-191`) and returns `self.kernel_client.execution_state or "unknown"`.
- `stop(self) -> None`:
  - Safely calls `self._kernel_client.stop(shutdown_kernel=False)` inside `contextlib.suppress(Exception)` and clears `self._kernel_client = None`.

- [x] **Step 1: Write the failing test (`tests/test_runtime.py`)**
  Using a fake `kernel_client_factory` and `tmp_path` fixtures for `.py` and `.ipynb` files, test:
  1. `execute_code` forwards `timeout` and `allow_stdin=False`, sets `_own_kernel = False`, passes `X-Colab-Runtime-Proxy-Token` header, and aggregates `stdout`/`stderr`/`error`.
  2. `execute_file` executes a multi-cell `.ipynb` in `tmp_path` and writes `<stem>_output.ipynb` with cell outputs attached.
  3. `restart()` and `stop()` invoke `KernelClient.restart()` and `KernelClient.stop(shutdown_kernel=False)`.

```python
def test_execute_code_passes_timeout_and_headers():
    from unittest.mock import MagicMock
    from colab_mcp.runtime import ColabRuntime

    fake_kc = MagicMock()
    fake_kc.id = "k-123"
    fake_kc.execute.return_value = {
        "status": "ok",
        "outputs": [{"output_type": "stream", "name": "stdout", "text": "ready\n"}],
    }
    factory = MagicMock(return_value=fake_kc)
    rt = ColabRuntime("https://proxy.colab.google", "tok-xyz", kernel_client_factory=factory)
    res = rt.execute_code("print('ready')", timeout=600.0, allow_stdin=False, ensure_cwd=False)
    assert res.status == "ok"
    assert res.stdout == "ready\n"
    assert fake_kc._own_kernel is False
```

- [x] **Step 2: Run test to verify it fails (Red)**
  Run: `uv run pytest tests/test_runtime.py -v`
  Expected: FAIL with `ModuleNotFoundError: No module named 'colab_mcp.runtime'`.

- [x] **Step 3: Write minimal implementation (`src/colab_mcp/runtime.py`)**
  Implement `ColabRuntime` with all methods specified above.

- [x] **Step 4: Run test to verify it passes (Green)**
  Run: `uv run pytest tests/test_runtime.py -v`
  Expected: PASS.

- [x] **Step 5: Self-review and report to the controller**
  Report status, changed files, and test output; leave changes uncommitted.

---

### Task 6: Persistent Session Manager & In-Process Keep-Alive Loop (`state.py`)

**Wave:** 3  
**Depends on:** Task 1, Task 2, Task 3, Task 4, Task 5  
**Files Touched:**
- Create: `src/colab_mcp/state.py`
- Test: `tests/test_state.py`

#### Interface & Schema Specifications
Implement `SessionManager` in `src/colab_mcp/state.py` coordinating `AuthManager`, `ColabClient`, `ContentsClient`, and `ColabRuntime`:
- `__init__(self, state_path: Path | None = None, auth_manager: AuthManager | None = None, client_factory: Callable[[requests.Session], ColabClient] = ColabClient, contents_factory: Callable[[str, str], ContentsClient] = ContentsClient, runtime_factory: Callable[..., ColabRuntime] = ColabRuntime, keep_alive_interval_s: float = 60.0)`:
  - Defaults `state_path` to `Path.home() / ".config" / "colab-mcp" / "sessions.json"`.
- Persistence methods:
  - `load_sessions(self) -> dict[str, SessionState]` and `save_sessions(self, sessions: dict[str, SessionState]) -> None`.
- Proxy token refresh & client accessors:
  - `ensure_fresh_session(self, name: str) -> SessionState`: Looks up `name` in `load_sessions()` (raising `KeyError(f"Session '{name}' not found")` if absent). If `proxy_expires_at` is within 300 seconds of `datetime.now(timezone.utc)`, calls `client.get_assignment(uuid.UUID(s.notebook_hash), ...)` to refresh `proxy_token`, `runtime_url`, and `proxy_expires_at`, invalidating any cached runtime for `name` if token/url changed and persisting updated state.
  - `get_contents_client(self, name: str) -> ContentsClient`: Calls `s = self.ensure_fresh_session(name)` and returns `self._contents_factory(s.runtime_url, s.proxy_token)`.
  - `get_runtime(self, name: str) -> ColabRuntime`: Calls `s = self.ensure_fresh_session(name)` and returns a cached `ColabRuntime` for `name` (or instantiates via `self._runtime_factory(s.runtime_url, s.proxy_token, kernel_id=s.kernel_id)` and caches it in `self._runtimes[name]`).
- Lifecycle operations:
  - `create_session(self, name: str | None = None, gpu: str | None = None, tpu: str | None = None, high_mem: bool = False) -> SessionInfo`:
    - Generates a 6-char hex name if `name` is `None`.
    - Resolves `(variant, accelerator, shape) = resolve_runtime_options(gpu=gpu, tpu=tpu, high_mem=high_mem)`.
    - Allocates via `client.assign(notebook_hash, variant, accelerator, shape)`.
    - Saves `SessionState` to `sessions.json`, runs a pre-flight `client.keep_alive(assignment.endpoint)`, and ensures background keep-alive task is running.
  - `list_sessions(self) -> list[SessionSummary]`:
    - Calls `client.list_assignments()`, cross-references local `sessions.json` by `endpoint`, prunes dead local sessions missing from server assignments, and returns `list[SessionSummary]` (marking untracked server assignments with `name="?"`).
  - `session_status(self, name: str) -> SessionStatus`:
    - Calls `s = self.ensure_fresh_session(name)`, queries kernel state via `self.get_runtime(name).get_kernel_state()`, and returns `SessionStatus`.
  - `stop_session(self, name: str) -> dict[str, Any]`:
    - Safely calls `rt.stop()` on any cached runtime in `self._runtimes.pop(name, None)` (inside `contextlib.suppress(Exception)`), calls `client.unassign(s.endpoint)`, removes `name` from `sessions.json`, and returns `{"stopped": True, "session": name, "endpoint": s.endpoint}`.
  - `restart_kernel(self, name: str) -> dict[str, Any]`:
    - Calls `self.get_runtime(name).restart()`, updates `kernel_id` in `sessions.json`, and returns `{"restarted": True, "session": name, "kernel_id": kernel_id}`.
- Background keep-alive lifecycle:
  - `start_keep_alive_loop(self) -> None` and `stop_keep_alive_loop(self) -> None` (plus `__aenter__` / `__aexit__`): Runs an `asyncio.Task` that every `keep_alive_interval_s` pings `await asyncio.to_thread(client.keep_alive, s.endpoint)` for each stored session. On `__aexit__`, stops the keep-alive loop and calls `rt.stop()` on all cached runtimes.

- [x] **Step 1: Write the failing test (`tests/test_state.py`)**
  Using `tmp_path / "sessions.json"` and mocked factories, test `create_session`, `get_contents_client`, `get_runtime` (including caching and automatic proxy token refresh when `proxy_expires_at` is near expiry), `list_sessions` (including pruning stale local sessions and surfacing `?` untracked server sessions), `stop_session` (verifying `runtime.stop()` is called before `client.unassign()`), and the `asyncio` keep-alive loop.

```python
import pytest

@pytest.mark.asyncio
async def test_session_manager_create_accessors_and_stop(tmp_path):
    from unittest.mock import MagicMock
    from colab_mcp.models import Assignment, RuntimeProxyInfo, Shape, Variant
    from colab_mcp.state import SessionManager

    fake_client = MagicMock()
    fake_client.assign.return_value = Assignment(
        endpoint="ep-1",
        accelerator="T4",
        variant=Variant.GPU,
        machineShape=Shape.STANDARD,
        runtimeProxyInfo=RuntimeProxyInfo(token="tok-1", tokenExpiresInSeconds=3600, url="https://proxy.colab.google"),
    )
    fake_runtime = MagicMock()
    mgr = SessionManager(
        state_path=tmp_path / "sessions.json",
        auth_manager=MagicMock(),
        client_factory=lambda _: fake_client,
        runtime_factory=lambda *a, **kw: fake_runtime,
    )
    info = mgr.create_session(name="trainer", gpu="T4")
    assert info.name == "trainer"
    assert mgr.get_runtime("trainer") is fake_runtime
    assert mgr.get_contents_client("trainer").proxy_token == "tok-1"
    mgr.stop_session("trainer")
    fake_runtime.stop.assert_called_once()
    fake_client.unassign.assert_called_once_with("ep-1")
```

- [x] **Step 2: Run test to verify it fails (Red)**
  Run: `uv run pytest tests/test_state.py -v`
  Expected: FAIL with `ModuleNotFoundError: No module named 'colab_mcp.state'`.

- [x] **Step 3: Write minimal implementation (`src/colab_mcp/state.py`)**
  Implement `SessionManager` with state persistence, token refresh, and background keep-alive loop.

- [x] **Step 4: Run test to verify it passes (Green)**
  Run: `uv run pytest tests/test_state.py -v`
  Expected: PASS.

- [x] **Step 5: Self-review and report to the controller**
  Report status, changed files, and test output; leave changes uncommitted.

---

### Task 7: FastMCP Tool Suite (`tools.py`)

**Wave:** 4  
**Depends on:** Task 6  
**Files Touched:**
- Create: `src/colab_mcp/tools.py`
- Test: `tests/test_tools.py`

#### Interface & Schema Specifications
Implement `register_tools(mcp: FastMCP, manager: SessionManager) -> FastMCP` in `src/colab_mcp/tools.py` registering all 14 tools from the approved Design Spec (`docs/superpowers/specs/2026-11-13-colab-headless-mcp-design.md`):
1. **Authentication Tools (3):**
   - `auth_status(provider: Literal["oauth2", "adc"] = "oauth2") -> dict[str, Any]`
   - `auth_start_login() -> dict[str, Any]`
   - `auth_finish_login(code: str) -> dict[str, Any]`
2. **Runtime & Kernel Lifecycle Tools (5):**
   - `create_session(name: str | None = None, gpu: Literal["T4", "L4", "A100", "H100"] | None = None, tpu: Literal["v5e1", "v6e1"] | None = None, high_mem: bool = False) -> dict[str, Any]`
   - `list_sessions() -> list[dict[str, Any]]`
   - `session_status(session: str) -> dict[str, Any]`
   - `stop_session(session: str) -> dict[str, Any]`
   - `restart_kernel(session: str) -> dict[str, Any]`
3. **Local ↔ Remote File Operation Tools (4):**
   - `upload_file(session: str, local_path: str, remote_path: str) -> dict[str, Any]`
   - `download_file(session: str, remote_path: str, local_path: str) -> dict[str, Any]`
   - `list_files(session: str, remote_path: str = "/content") -> list[dict[str, Any]]`
   - `delete_file(session: str, remote_path: str) -> dict[str, Any]`
4. **Execution Control Tools (2):**
   - `exec_code(session: str, code: str, timeout: float = 300.0, allow_stdin: bool = False, env: dict[str, str] | None = None) -> dict[str, Any]`
   - `exec_file(session: str, file_path: str, timeout: float = 1800.0, allow_stdin: bool = False, env: dict[str, str] | None = None, save_output_notebook: bool = True) -> dict[str, Any]`
- All blocking I/O calls are dispatched via `await asyncio.to_thread(...)` so the FastMCP async event loop stays responsive during long GPU runs and file transfers.
- While `exec_code` or `exec_file` is running, `SessionState.running` is updated in `sessions.json` and cleared in a `finally` block, and `SessionState.last_execution` is recorded.

- [x] **Step 1: Write the failing test (`tests/test_tools.py`)**
  Using `fastmcp.FastMCP` and `fastmcp.Client`, verify all 14 tool names are registered and callable through the MCP client interface against a mocked `SessionManager`.

```python
import pytest
from fastmcp import Client, FastMCP

@pytest.mark.asyncio
async def test_all_fourteen_tools_registered_and_callable():
    from unittest.mock import MagicMock
    from colab_mcp.tools import register_tools

    mcp = FastMCP(name="ColabMCPTest")
    mgr = MagicMock()
    register_tools(mcp, mgr)
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert len(names) == 14
        assert {"create_session", "exec_code", "exec_file", "upload_file", "download_file"} <= names
```

- [x] **Step 2: Run test to verify it fails (Red)**
  Run: `uv run pytest tests/test_tools.py -v`
  Expected: FAIL with `ModuleNotFoundError: No module named 'colab_mcp.tools'`.

- [x] **Step 3: Write minimal implementation (`src/colab_mcp/tools.py`)**
  Implement `register_tools(mcp, manager)` registering all 14 tools with `asyncio.to_thread` execution and error translation to `fastmcp.exceptions.ToolError`.

- [x] **Step 4: Run test to verify it passes (Green)**
  Run: `uv run pytest tests/test_tools.py -v`
  Expected: PASS.

- [x] **Step 5: Self-review and report to the controller**
  Report status, changed files, and test output; leave changes uncommitted.

---

### Task 8: Server Entrypoint (`__init__.py`), `pyproject.toml` Packaging, CLI `auth` Subcommand & Legacy Browser-Proxy Removal

**Wave:** 5  
**Depends on:** Task 7  
**Files Touched:**
- Modify: `src/colab_mcp/__init__.py:1-87`
- Modify: `pyproject.toml:11-29`
- Delete: `src/colab_mcp/session.py:1-376`
- Delete: `src/colab_mcp/websocket_server.py:1-162`
- Delete: `tests/session_test.py:1-323`
- Delete: `tests/websocket_server_test.py:1-222`
- Create: `tests/test_entrypoint.py`

#### Interface & Schema Specifications
1. **Update `src/colab_mcp/__init__.py:1-87`:**
   - Keep `mcp = FastMCP(name="ColabMCP")` (`src/colab_mcp/__init__.py:28`) and `init_logger(logdir)` (`src/colab_mcp/__init__.py:31-41`).
   - Update `parse_args(v)` (`src/colab_mcp/__init__.py:44-63`):
     - Support optional positional/subcommand `command` (`"auth"` to run interactive CLI OAuth login via `AuthManager().run_cli_login()`).
     - Support `-l` / `--log` directory flag and `--auth` (`choices=["oauth2", "adc"]`, default `"oauth2"`).
   - Update `main_async()` (`src/colab_mcp/__init__.py:66-84`):
     - If `args.command == "auth"`, run `AuthManager().run_cli_login()` and return.
     - Otherwise instantiate `SessionManager()`, call `register_tools(mcp, session_manager)`, start the keep-alive loop inside `async with session_manager:`, and await `mcp.run_async()`.
2. **Update `pyproject.toml:11-29`:**
   - Remove unused legacy dependency `requests-oauth2client>=0.4.0` (`pyproject.toml:18`).
   - Add `[tool.hatch.build]` with `artifacts = ["src/colab_mcp/*.json"]` so `oauth_config.json` is always included in sdist and wheel builds.
3. **Delete legacy browser-proxy files** `src/colab_mcp/session.py`, `src/colab_mcp/websocket_server.py`, `tests/session_test.py`, and `tests/websocket_server_test.py`.

- [x] **Step 1: Write the failing test (`tests/test_entrypoint.py`)**
  Test `parse_args(["auth"])`, `parse_args(["--auth", "adc"])`, `main_async()` dispatching `auth` subcommand to `AuthManager.run_cli_login()`, and default MCP startup registering all 14 tools on `colab_mcp.mcp` without importing `colab_mcp.session` or `colab_mcp.websocket_server`.

```python
import pytest

@pytest.mark.asyncio
async def test_entrypoint_auth_subcommand_and_mcp_wiring(monkeypatch):
    import colab_mcp

    args = colab_mcp.parse_args(["auth"])
    assert args.command == "auth"
```

- [x] **Step 2: Run test to verify it fails (Red)**
  Run: `uv run pytest tests/test_entrypoint.py -v`
  Expected: FAIL with `SystemExit: 2` (unrecognized arguments: `auth`).

- [x] **Step 3: Write minimal implementation (`src/colab_mcp/__init__.py`) and remove legacy files**
  Update `src/colab_mcp/__init__.py` and delete `src/colab_mcp/session.py`, `src/colab_mcp/websocket_server.py`, `tests/session_test.py`, and `tests/websocket_server_test.py`.

- [x] **Step 4: Run full test suite and linter to verify Green**
  Run: `uv run pytest -v` and `uv run ruff check src tests`
  Expected: PASS with 0 test failures and 0 ruff errors.

- [x] **Step 5: Self-review and report to the controller**
  Report status, changed files, and full test output; leave changes uncommitted.
