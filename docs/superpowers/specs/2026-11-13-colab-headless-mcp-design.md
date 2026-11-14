# Headless Google Colab REST & Jupyter Kernel MCP Server — Design Specification

**Date:** 2026-11-13  
**Target Workspace:** `c:\Users\atharva.bedekar\PycharmProjects\colab_mcp`  
**Package / Entrypoint:** `colab_mcp` / `colab-mcp`  
**Status:** Pending Checkpoint 1 Approval

---

## 1. Overview & Goals

Replace the existing browser-dependent WebSocket proxy in `colab_mcp` (`src/colab_mcp/session.py` and `src/colab_mcp/websocket_server.py`) with a standalone, zero-CLI-dependency **Headless Google Colab MCP Server** powered directly by ported core layers from `google-colab-cli`:

1. **Authentication Layer (`src/colab_mcp/auth.py` + `src/colab_mcp/oauth_config.json`)**:
   - Hybrid authentication supporting:
     - Automatic token reuse from `~/.config/colab-mcp/token.json` or `~/.config/colab-cli/token.json`
     - Application Default Credentials (`adc` via `google.auth.default()`)
     - Remote OAuth2 copy-paste login (`redirect_uri=https://sdk.cloud.google.com/applicationdefaultauthcode.html`, `token_usage=remote`) exposed both via MCP tools (`auth_status`, `auth_start_login`, `auth_finish_login`) and CLI subcommand (`colab-mcp auth`).
2. **Session Control-Plane Client (`src/colab_mcp/client.py`)**:
   - Two-step XSRF handshake (`GET` + `POST /tun/m/assign` and `/tun/m/unassign/<endpoint>` with `X-Goog-Colab-Token` header)
   - XSSI prefix (`)]}'\n`) stripping and Pydantic validation (`Variant`, `Accelerator`, `Shape`, `RuntimeProxyInfo`, `Assignment`, `PostAssignmentResponse`, `ListedAssignment`)
   - Exponential backoff retry on `ReadTimeout` and HTTP 502/503/504, and `TooManyAssignmentsError` on HTTP 412
   - Active session listing (`GET /tun/m/assignments`)
   - Tunnel Frontend HTTP keep-alive ping (`GET /tun/m/<endpoint>/keep-alive/` with `X-Colab-Tunnel: Google`)
3. **Session State & In-Process Keep-Alive Manager (`src/colab_mcp/state.py`)**:
   - Local state persistence at `~/.config/colab-mcp/sessions.json` mapping friendly session names (e.g., `trainer`) to `endpoint`, `notebook_hash`, `variant`, `accelerator`, `shape`, and cached `RuntimeProxyInfo` (`url`, `token`, `expires_at`)
   - Background `asyncio` keep-alive task inside the MCP server lifecycle that periodically pings `/tun/m/<endpoint>/keep-alive/` and refreshes proxy tokens before expiry
4. **Remote Filesystem Layer (`src/colab_mcp/contents.py`)**:
   - Direct HTTP `GET` / `PUT` / `DELETE` against `<runtime_proxy_url>/api/contents/<remote_path>` using `authuser=0` and `colab-runtime-proxy-token` (kernel-independent file upload, download, directory listing, and deletion)
5. **Jupyter Kernel Execution Layer (`src/colab_mcp/runtime.py`)**:
   - Direct WebSocket kernel connection via `jupyter-kernel-client` (`KernelClient` with `X-Colab-Runtime-Proxy-Token` and `X-Colab-Client-Agent: colab-mcp`, `_own_kernel = False`)
   - Code execution (`execute_code`) with configurable `timeout` ceiling (overriding the 10s `REQUEST_TIMEOUT` default), `allow_stdin=False` fail-fast handling, structured stdout/stderr/error/display output collection, and kernel restart (`restart_kernel`)
   - Script (`.py`) and notebook (`.ipynb`) execution (`execute_file`) with optional `*_output.ipynb` saving

---

## 2. Architecture & Module Layout

```text
src/colab_mcp/
├── __init__.py          # CLI parser (`colab-mcp` MCP stdio runner + `colab-mcp auth` CLI login) & FastMCP server entrypoint
├── oauth_config.json    # Bundled Cloud SDK OAuth2 client config (764086051850-...)
├── models.py            # Enums (Variant, Accelerator, Shape, AuthProvider) & Pydantic schemas (RuntimeProxyInfo, Assignment, ListedAssignment, SessionState, ExecutionOutput)
├── auth.py              # Hybrid credential manager: token cache (~/.config/colab-mcp/token.json + ~/.config/colab-cli/token.json fallback), ADC, and remote OAuth2 flow
├── client.py            # ColabClient: /tun/m/assign, /tun/m/unassign/<endpoint>, /tun/m/assignments, /tun/m/<endpoint>/keep-alive/, XSSI stripping, retry/backoff
├── contents.py          # ContentsClient: /api/contents/<path> (list_dir, upload, download, delete)
├── runtime.py           # ColabRuntime: jupyter-kernel-client wrapper (execute_code, execute_file, restart, get_status, stop)
├── state.py             # SessionManager: ~/.config/colab-mcp/sessions.json persistence, session resolution, and background asyncio keep-alive loop
└── tools.py             # FastMCP tool registrations for Auth, Runtime Lifecycle, File Operations, and Execution Control
```

Legacy browser-proxy files (`src/colab_mcp/session.py`, `src/colab_mcp/websocket_server.py`, and their old tests) are removed.

---

## 3. MCP Tool Contracts

### 3.1 Authentication Tools
| Tool Name | Parameters | Returns | Description |
|---|---|---|---|
| `auth_status` | `provider: Literal["oauth2", "adc"] = "oauth2"` | `{"authenticated": bool, "provider": str, "token_path": str \| None, "account": str \| None}` | Checks whether valid cached OAuth2 or ADC credentials are available. |
| `auth_start_login` | None | `{"auth_url": str, "instructions": str}` | Builds the remote OAuth2 consent URL (`redirect_uri=https://sdk.cloud.google.com/applicationdefaultauthcode.html`, `token_usage=remote`) and stores pending flow state in memory. |
| `auth_finish_login` | `code: str` | `{"authenticated": bool, "token_path": str}` | Exchanges the pasted authorization code for tokens and saves `~/.config/colab-mcp/token.json`. |

### 3.2 Runtime & Kernel Lifecycle Tools
| Tool Name | Parameters | Returns | Description |
|---|---|---|---|
| `create_session` | `name: str \| None = None`, `gpu: Literal["T4", "L4", "A100", "H100"] \| None = None`, `tpu: Literal["v5e1", "v6e1"] \| None = None`, `high_mem: bool = False` | `SessionInfo` (`name`, `endpoint`, `variant`, `accelerator`, `shape`, `runtime_url`, `status`) | Provisions a new Colab VM via `/tun/m/assign`, persists `SessionState`, verifies `/content` working directory, and registers background keep-alive. |
| `list_sessions` | None | `list[SessionSummary]` | Syncs local `sessions.json` with server-side `GET /tun/m/assignments`, marking tracked and untracked (`?`) active runtimes. |
| `session_status` | `session: str` | `SessionStatus` (`name`, `endpoint`, `variant`, `accelerator`, `shape`, `kernel_state`, `running`, `last_execution`, `proxy_expires_at`) | Queries live assignment and Jupyter kernel status for the named session. |
| `stop_session` | `session: str` | `{"stopped": bool, "session": str, "endpoint": str}` | Stops kernel channels, cancels keep-alive, calls two-step `GET` + `POST /tun/m/unassign/<endpoint>`, and removes session from `sessions.json`. |
| `restart_kernel` | `session: str` | `{"restarted": bool, "session": str, "kernel_id": str \| None}` | Restarts the remote Jupyter kernel cleanly without releasing the VM or losing `/content`. |

### 3.3 Local ↔ Remote File Operation Tools
| Tool Name | Parameters | Returns | Description |
|---|---|---|---|
| `upload_file` | `session: str`, `local_path: str`, `remote_path: str` | `{"name": str, "path": str, "size_bytes": int}` | Reads `local_path`, base64-encodes it, and `PUT`s to `/api/contents/<remote_path>` (independent of kernel execution state). |
| `download_file` | `session: str`, `remote_path: str`, `local_path: str` | `{"remote_path": str, "local_path": str, "size_bytes": int}` | `GET`s `/api/contents/<remote_path>?content=1`, decodes base64/UTF-8, and writes to `local_path`. |
| `list_files` | `session: str`, `remote_path: str = "/content"` | `list[RemoteFileEntry]` (`name`, `path`, `type`, `size`, `last_modified`) | Lists files and directories via `GET /api/contents/<remote_path>`. |
| `delete_file` | `session: str`, `remote_path: str` | `{"deleted": bool, "remote_path": str}` | Deletes a remote file or empty directory via `DELETE /api/contents/<remote_path>`. |

### 3.4 Execution Control Tools
| Tool Name | Parameters | Returns | Description |
|---|---|---|---|
| `exec_code` | `session: str`, `code: str`, `timeout: float = 300.0`, `allow_stdin: bool = False`, `env: dict[str, str] \| None = None` | `ExecutionResult` (`status`, `outputs`, `stdout`, `stderr`, `error`, `duration_s`) | Executes Python code on the session's Jupyter kernel with configurable `timeout` ceiling and `allow_stdin=False` fail-fast protection. |
| `exec_file` | `session: str`, `file_path: str`, `timeout: float = 1800.0`, `allow_stdin: bool = False`, `env: dict[str, str] \| None = None`, `save_output_notebook: bool = True` | `FileExecutionResult` (`file_path`, `status`, `cells_executed`, `outputs`, `output_notebook_path`, `duration_s`) | Executes a local `.py` script or `.ipynb` notebook sequentially on the remote kernel, optionally writing `<stem>_output.ipynb`. |

---

## 4. Dependencies (`pyproject.toml`)

All core runtime dependencies are already declared in `pyproject.toml`:
- `fastmcp==2.14.5`, `mcp[cli]>=1.20.0`, `pydantic>=2.12.3,<3.0.0`, `requests>=2.32.5`, `websockets>=15.0.1`, `google-auth-oauthlib>=0.7.0`, `jupyter-kernel-client>=0.9.0`, `anyio>=4.5`
- For `.ipynb` parsing/saving in `exec_file`, `.ipynb` files are standard JSON (v4 schema); we parse and serialize notebook JSON directly with `json` + Pydantic/helper functions so no extra third-party `nbformat` dependency is required.

---

## 5. Autonomous Engineering Defaults
- **XSSI Prefix Handling:** Strip leading `)]}'\n` or `)]}'` on all Colab `/tun/m/` JSON responses before parsing.
- **Proxy Token Refresh:** When a session's `RuntimeProxyInfo.expires_at` is within 5 minutes of expiry (or on 401/403 from `/api/contents`), re-query `_get_assignment(notebook_hash)` to refresh the runtime proxy token transparently.
- **Working Directory Guarantee:** Before executing code in `exec_code` / `exec_file`, ensure `/content` exists and is the current working directory (`os.makedirs('/content', exist_ok=True); os.chdir('/content')`).
- **Keep-Alive Interval:** Background `asyncio` loop pings `/tun/m/<endpoint>/keep-alive/` every 60 seconds for all active tracked sessions; `ReadTimeout` on the tunnel ping is treated as success per Colab TFE behavior.
