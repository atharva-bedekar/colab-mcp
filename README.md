# Colab MCP

Headless Model Context Protocol (MCP) server for provisioning, managing, and executing code on Google Colab runtimes without a browser session.

## Overview

`colab-mcp` connects MCP-compatible AI agents and IDEs directly to Google Colab's runtime control plane and Jupyter kernel WebSocket/REST APIs. It supports CPU, GPU (`T4`, `L4`, `A100`, `H100`), and TPU (`v5e1`, `v6e1`) runtimes, persistent named sessions with background keep-alive, and remote file transfers.

## Requirements

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/)
- A Google account with access to Google Colab (or Application Default Credentials)

## Installation

```bash
git clone https://github.com/atharva-bedekar/colab-mcp.git
cd colab-mcp
uv sync
```

## Authentication

`colab-mcp` supports OAuth2 (including shared credentials with `google-colab-cli` in the OS keyring) and Google Application Default Credentials (ADC).

### OAuth2 (Recommended)

Run the local loopback login flow once in a terminal:

```bash
uv run colab-mcp auth
```

For headless or remote environments without a local browser, use the copy-paste flow:

```bash
uv run colab-mcp auth --remote
```

Credentials are saved to `~/.config/colab-mcp/token.json` and the OS keyring (`google-colab-cli/oauth_token`).

### Application Default Credentials (ADC)

```bash
gcloud auth application-default login
uv run colab-mcp --auth adc
```

## MCP Client Configuration

Add `colab-mcp` to your MCP client configuration (Claude Code, Gemini CLI, Windsurf, Cursor, etc.).

### Using a local checkout

```json
{
  "mcpServers": {
    "colab-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/colab-mcp", "colab-mcp"],
      "timeout": 30000
    }
  }
}
```

### Using `uvx` directly from GitHub

```json
{
  "mcpServers": {
    "colab-mcp": {
      "command": "uvx",
      "args": ["git+https://github.com/atharva-bedekar/colab-mcp"],
      "timeout": 30000
    }
  }
}
```

## Tools

### Authentication

| Tool | Description | Parameters |
| --- | --- | --- |
| `auth_status` | Check whether valid cached OAuth2 or ADC credentials are available. | `provider` (`"oauth2"` \| `"adc"`, default `"oauth2"`) |
| `auth_start_login` | Start remote OAuth2 login and return the consent URL and instructions. | — |
| `auth_finish_login` | Exchange an authorization code or redirect URL for OAuth2 tokens. | `code` (`str`) |

### Runtime Lifecycle

| Tool | Description | Parameters |
| --- | --- | --- |
| `create_session` | Provision a Colab runtime and register it locally. | `name` (`str \| None`), `gpu` (`"T4"` \| `"L4"` \| `"A100"` \| `"H100"`), `tpu` (`"v5e1"` \| `"v6e1"`), `high_mem` (`bool`) |
| `list_sessions` | List active Colab runtimes reconciled with local session state. | — |
| `session_status` | Query live kernel state, background keep-alive health (auto-pinged every 60s), and metadata for a session. | `session` (`str`) |
| `stop_session` | Disconnect kernel channels, unassign the Colab VM, and remove local state. | `session` (`str`) |
| `restart_kernel` | Restart the remote Jupyter kernel cleanly without releasing the VM. | `session` (`str`) |

### Remote Files (`/api/contents`) & Directory Transfers

| Tool | Description | Parameters |
| --- | --- | --- |
| `upload_file` | Upload a local file to the Colab runtime. | `session` (`str`), `local_path` (`str`), `remote_path` (`str`) |
| `download_file` | Download raw bytes of a remote file from the Colab runtime with SHA-256 checksum. | `session` (`str`), `remote_path` (`str`), `local_path` (`str`) |
| `list_files` | List files and directories on the Colab runtime. | `session` (`str`), `remote_path` (`str`, default `"/content"`) |
| `delete_file` | Delete a remote file or empty directory on the Colab runtime. | `session` (`str`), `remote_path` (`str`) |
| `read_file` | Read a remote file as UTF-8 text via `/api/contents`, with optional `tail_lines` or `max_bytes`. | `session` (`str`), `remote_path` (`str`), `tail_lines` (`int \| None`), `max_bytes` (`int`, default `65536`) |
| `stat_file` | Return metadata (`type`, `size`, `last_modified`) for a remote path without downloading content. | `session` (`str`), `remote_path` (`str`) |
| `download_dir` | Archive a remote directory, download with SHA-256 verification, and extract locally. | `session` (`str`), `remote_dir` (`str`), `local_dir` (`str`), `glob` (`str \| None`) |
| `upload_dir` | Archive a local directory, upload with SHA-256 verification, and extract on the Colab runtime. | `session` (`str`), `local_dir` (`str`), `remote_dir` (`str`), `glob` (`str \| None`) |

### Code, Notebook & Background Job Execution

| Tool | Description | Parameters |
| --- | --- | --- |
| `exec_code` | Execute Python code on the session's remote Jupyter kernel (compact deduplicated output by default). | `session` (`str`), `code` (`str`), `timeout` (`float`, default `300.0`), `allow_stdin` (`bool`), `env` (`dict[str, str] \| None`), `raw_outputs` (`bool`, default `false`) |
| `exec_file` | Execute a local `.py` script or `.ipynb` notebook on the remote kernel. | `session` (`str`), `file_path` (`str`), `timeout` (`float`, default `1800.0`), `allow_stdin` (`bool`), `env` (`dict[str, str] \| None`), `save_output_notebook` (`bool`, default `true`) |
| `start_background_job` | Launch a detached shell command on the Colab VM with log/exit tracking under `/tmp/colab_mcp_jobs/`. | `session` (`str`), `command` (`str`), `name` (`str \| None`), `cwd` (`str`, default `"/content"`), `env` (`dict[str, str] \| None`) |
| `check_background_job` | Check liveness, exit code, and trailing log lines of a background job. | `session` (`str`), `job_id` (`str`), `tail_lines` (`int`, default `50`) |
| `stop_background_job` | Terminate a detached background job on the Colab VM. | `session` (`str`), `job_id` (`str`) |

## CLI Reference

```text
colab-mcp [auth] [-l LOG] [--auth {oauth2,adc}] [--remote]
```

- `auth`: Run interactive OAuth2 login and exit.
- `-l`, `--log`: Directory for log files (defaults to a temporary `colab-mcp-logs-*` directory).
- `--auth`: Authentication provider (`oauth2` or `adc`, default `oauth2`).
- `--remote`: Use the remote copy-paste OAuth2 flow instead of a local loopback server.

You can also invoke the CLI via `uv run colab_mcp` or `uv run python -m colab_mcp`.

## Development

```bash
uv run pytest -v
uv run ruff check src tests
```

## License

Apache-2.0
