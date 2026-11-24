"""FastMCP tool registrations for Auth, Runtime Lifecycle, Files, and Execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Literal, TypeVar

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from colab_mcp import jobs, transfer
from colab_mcp.models import AuthProvider, ColabError
from colab_mcp.state import SessionManager

_T = TypeVar("_T")
_HANDLED_ERRORS = (
    ColabError,
    FileNotFoundError,
    IsADirectoryError,
    NotADirectoryError,
    KeyError,
    ValueError,
)


async def _run_blocking(fn: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except _HANDLED_ERRORS as exc:
        raise ToolError(str(exc)) from exc


async def _execute_with_tracking(
    manager: SessionManager,
    session: str,
    label: str,
    fn: Callable[[Any], Any],
) -> dict[str, Any]:
    def _do() -> dict[str, Any]:
        rt = manager.get_runtime(session)
        manager.update_session_execution(session, running=label)
        status = "error"
        error_msg: str | None = None
        try:
            res = fn(rt)
            status = res.status
            error_msg = getattr(res, "error", None)
            return res.model_dump(mode="json")
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            finished_at = datetime.now(timezone.utc).isoformat()
            manager.update_session_execution(
                session,
                running=None,
                last_execution=(status, error_msg, finished_at),
            )

    return await _run_blocking(_do)


def register_tools(mcp: FastMCP, manager: SessionManager) -> FastMCP:
    """Register all 21 Colab MCP tools onto a FastMCP instance."""

    @mcp.tool()
    async def auth_status(provider: Literal["oauth2", "adc"] = "oauth2") -> dict[str, Any]:
        """Check whether valid cached OAuth2 or ADC credentials are available."""
        resp = await _run_blocking(manager.auth_manager.get_status, AuthProvider(provider))
        return resp.model_dump(mode="json")

    @mcp.tool()
    async def auth_start_login() -> dict[str, Any]:
        """Start remote OAuth2 login and return the consent URL and instructions."""
        resp = await _run_blocking(manager.auth_manager.start_remote_login)
        return resp.model_dump(mode="json")

    @mcp.tool()
    async def auth_finish_login(code: str) -> dict[str, Any]:
        """Complete remote OAuth2 login by exchanging an authorization code."""
        resp = await _run_blocking(manager.auth_manager.finish_remote_login, code)
        return resp.model_dump(mode="json")

    @mcp.tool()
    async def create_session(
        name: str | None = None,
        gpu: Literal["T4", "L4", "A100", "H100"] | None = None,
        tpu: Literal["v5e1", "v6e1"] | None = None,
        high_mem: bool = False,
    ) -> dict[str, Any]:
        """Provision a new Colab VM runtime and register it under a session name."""
        info = await _run_blocking(
            manager.create_session,
            name=name,
            gpu=gpu,
            tpu=tpu,
            high_mem=high_mem,
        )
        return info.model_dump(mode="json")

    @mcp.tool()
    async def list_sessions() -> list[dict[str, Any]]:
        """List active Colab runtimes synced with local session state."""
        summaries = await _run_blocking(manager.list_sessions)
        return [item.model_dump(mode="json") for item in summaries]

    @mcp.tool()
    async def session_status(session: str) -> dict[str, Any]:
        """Query live kernel state and metadata for a named Colab session.

        Background VM keep-alive runs automatically every 60s; its state is
        reported via keep_alive_active and last_keep_alive_at.
        """
        status = await _run_blocking(manager.session_status, session)
        return status.model_dump(mode="json")

    @mcp.tool()
    async def stop_session(session: str) -> dict[str, Any]:
        """Stop kernel channels, unassign the Colab VM, and remove local session state."""
        return await _run_blocking(manager.stop_session, session)

    @mcp.tool()
    async def restart_kernel(session: str) -> dict[str, Any]:
        """Restart the remote Jupyter kernel cleanly without releasing the VM."""
        return await _run_blocking(manager.restart_kernel, session)

    @mcp.tool()
    async def upload_file(session: str, local_path: str, remote_path: str) -> dict[str, Any]:
        """Upload a local file to the Colab runtime via /api/contents."""
        def _do() -> dict[str, Any]:
            return manager.get_contents_client(session).upload(local_path, remote_path)

        return await _run_blocking(_do)

    @mcp.tool()
    async def download_file(session: str, remote_path: str, local_path: str) -> dict[str, Any]:
        """Download a remote file from the Colab runtime via /api/contents."""
        def _do() -> dict[str, Any]:
            return manager.get_contents_client(session).download(remote_path, local_path)

        return await _run_blocking(_do)

    @mcp.tool()
    async def list_files(session: str, remote_path: str = "/content") -> list[dict[str, Any]]:
        """List files and directories on the Colab runtime via /api/contents."""
        def _do() -> list[dict[str, Any]]:
            entries = manager.get_contents_client(session).list_dir(remote_path)
            return [entry.model_dump(mode="json") for entry in entries]

        return await _run_blocking(_do)

    @mcp.tool()
    async def delete_file(session: str, remote_path: str) -> dict[str, Any]:
        """Delete a remote file or empty directory on the Colab runtime."""
        def _do() -> dict[str, Any]:
            return manager.get_contents_client(session).delete(remote_path)

        return await _run_blocking(_do)

    @mcp.tool()
    async def read_file(
        session: str,
        remote_path: str,
        tail_lines: int | None = None,
        max_bytes: int = 65536,
    ) -> dict[str, Any]:
        """Read a remote text file, optionally limited to trailing lines or bytes."""
        def _do() -> dict[str, Any]:
            return manager.get_contents_client(session).read_file(
                remote_path, tail_lines=tail_lines, max_bytes=max_bytes
            )

        return await _run_blocking(_do)

    @mcp.tool()
    async def stat_file(session: str, remote_path: str) -> dict[str, Any]:
        """Return metadata for a single remote file or directory."""
        def _do() -> dict[str, Any]:
            return manager.get_contents_client(session).stat(remote_path).model_dump(
                mode="json"
            )

        return await _run_blocking(_do)

    @mcp.tool()
    async def download_dir(
        session: str,
        remote_dir: str,
        local_dir: str,
        glob: str | None = None,
    ) -> dict[str, Any]:
        """Archive a remote directory, verify SHA-256, and extract it locally."""
        def _do() -> dict[str, Any]:
            return transfer.download_dir(
                manager.get_runtime(session),
                manager.get_contents_client(session),
                remote_dir,
                local_dir,
                glob=glob,
            )

        return await _run_blocking(_do)

    @mcp.tool()
    async def upload_dir(
        session: str,
        local_dir: str,
        remote_dir: str,
        glob: str | None = None,
    ) -> dict[str, Any]:
        """Zip a local directory, upload it, verify SHA-256, and extract remotely."""
        def _do() -> dict[str, Any]:
            return transfer.upload_dir(
                manager.get_runtime(session),
                manager.get_contents_client(session),
                local_dir,
                remote_dir,
                glob=glob,
            )

        return await _run_blocking(_do)

    @mcp.tool()
    async def start_background_job(
        session: str,
        command: str,
        name: str | None = None,
        cwd: str = "/content",
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Start a detached background shell command on the remote Colab VM."""
        def _do() -> dict[str, Any]:
            return jobs.start_background_job(
                manager.get_runtime(session),
                command,
                name=name,
                cwd=cwd,
                env=env,
            )

        return await _run_blocking(_do)

    @mcp.tool()
    async def check_background_job(
        session: str,
        job_id: str,
        tail_lines: int = 50,
    ) -> dict[str, Any]:
        """Check status and tail logs for a background job on the remote Colab VM."""
        def _do() -> dict[str, Any]:
            return jobs.check_background_job(
                manager.get_runtime(session), job_id, tail_lines=tail_lines
            )

        return await _run_blocking(_do)

    @mcp.tool()
    async def stop_background_job(session: str, job_id: str) -> dict[str, Any]:
        """Send SIGTERM to a background job's process group on the remote Colab VM."""
        def _do() -> dict[str, Any]:
            return jobs.stop_background_job(manager.get_runtime(session), job_id)

        return await _run_blocking(_do)

    @mcp.tool()
    async def exec_code(
        session: str,
        code: str,
        timeout: float = 300.0,
        allow_stdin: bool = False,
        env: dict[str, str] | None = None,
        raw_outputs: bool = False,
    ) -> dict[str, Any]:
        """Execute Python code on the session's remote Jupyter kernel."""
        return await _execute_with_tracking(
            manager,
            session,
            "exec_code",
            lambda rt: rt.execute_code(
                code,
                timeout=timeout,
                allow_stdin=allow_stdin,
                env=env,
                compact=not raw_outputs,
            ),
        )

    @mcp.tool()
    async def exec_file(
        session: str,
        file_path: str,
        timeout: float = 1800.0,
        allow_stdin: bool = False,
        env: dict[str, str] | None = None,
        save_output_notebook: bool = True,
    ) -> dict[str, Any]:
        """Execute a local .py script or .ipynb notebook on the remote Jupyter kernel."""
        return await _execute_with_tracking(
            manager,
            session,
            f"exec_file:{file_path}",
            lambda rt: rt.execute_file(
                file_path,
                timeout=timeout,
                allow_stdin=allow_stdin,
                env=env,
                save_output_notebook=save_output_notebook,
            ),
        )

    return mcp
