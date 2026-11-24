"""Jupyter kernel WebSocket execution wrapper for Colab runtimes."""

from __future__ import annotations

import contextlib
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests
from jupyter_kernel_client import KernelClient

from colab_mcp.models import (
    ExecutionResult,
    FileExecutionResult,
    parse_notebook_code_cells,
    write_notebook_with_outputs,
)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _build_env_prelude(env: dict[str, str] | None) -> str:
    if not env:
        return ""
    lines = ["import os"]
    for key, val in env.items():
        lines.append(f"os.environ[{key!r}] = {val!r}")
    return "\n".join(lines) + "\n"


def _join_text(value: Any) -> str:
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    if value is None:
        return ""
    return str(value)


class ColabRuntime:
    """Manages a remote Jupyter kernel session over Colab's runtime proxy."""

    def __init__(
        self,
        runtime_url: str,
        proxy_token: str,
        kernel_id: str | None = None,
        startup_retries: int = 3,
        startup_backoff: float = 2.0,
        kernel_client_factory: Callable[..., Any] = KernelClient,
    ) -> None:
        self.runtime_url = runtime_url.rstrip("/")
        self.proxy_token = proxy_token
        self.kernel_id = kernel_id
        self.startup_retries = startup_retries
        self.startup_backoff = startup_backoff
        self._kernel_client_factory = kernel_client_factory
        self._kernel_client: Any | None = None
        self._cwd_initialized = False

    @property
    def kernel_client(self) -> Any:
        """Lazily connect to or start a remote kernel without taking ownership."""
        if self._kernel_client is None:
            client = self._kernel_client_factory(
                server_url=self.runtime_url,
                token=self.proxy_token,
                kernel_id=self.kernel_id,
                headers={
                    "X-Colab-Runtime-Proxy-Token": self.proxy_token,
                    "X-Colab-Client-Agent": "colab-cli",
                },
            )
            client._own_kernel = False
            last_exc: Exception | None = None
            for attempt in range(self.startup_retries + 1):
                try:
                    client.start()
                    last_exc = None
                    break
                except (
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectTimeout,
                ) as exc:
                    last_exc = exc
                    if attempt >= self.startup_retries:
                        break
                    time.sleep(self.startup_backoff ** (attempt + 1))
            if last_exc is not None:
                raise last_exc
            self._kernel_client = client
            self.kernel_id = getattr(client, "id", self.kernel_id)
        return self._kernel_client

    def ensure_content_dir(self) -> None:
        """Ensure /content exists and is the current working directory once per connection."""
        if self._cwd_initialized:
            return
        self.execute_code(
            "import os; os.makedirs('/content', exist_ok=True); os.chdir('/content')",
            timeout=30.0,
            ensure_cwd=False,
        )
        self._cwd_initialized = True

    def execute_code(
        self,
        code: str,
        *,
        timeout: float = 300.0,
        allow_stdin: bool = False,
        env: dict[str, str] | None = None,
        ensure_cwd: bool = True,
        compact: bool = False,
    ) -> ExecutionResult:
        """Execute Python code on the remote kernel and collect structured outputs."""
        if ensure_cwd:
            self.ensure_content_dir()

        full_code = f"{_build_env_prelude(env)}{code}"
        started = time.monotonic()
        reply = self.kernel_client.execute(
            full_code,
            allow_stdin=allow_stdin,
            timeout=timeout,
        )
        elapsed = time.monotonic() - started

        outputs: list[dict[str, Any]] = list(reply.get("outputs", []))
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        error_summary: str | None = None

        for out in outputs:
            out_type = out.get("output_type")
            if out_type == "stream":
                name = out.get("name")
                text = _join_text(out.get("text", ""))
                if name == "stdout":
                    stdout_parts.append(text)
                elif name == "stderr":
                    stderr_parts.append(text)
            elif out_type == "error":
                ename = out.get("ename", "Error")
                evalue = out.get("evalue", "")
                tb = out.get("traceback")
                if compact and tb:
                    if isinstance(tb, str):
                        tb = tb.splitlines()
                    error_summary = "\n".join(_strip_ansi(str(line)) for line in tb)
                else:
                    error_summary = f"{ename}: {evalue}" if evalue else str(ename)

        status = str(reply.get("status", "ok"))
        if status == "error" and error_summary is None:
            ename = reply.get("ename", "Error")
            evalue = reply.get("evalue", "")
            error_summary = f"{ename}: {evalue}" if evalue else str(ename)

        result_outputs = (
            outputs
            if not compact
            else [out for out in outputs if out.get("output_type") not in ("stream", "error")]
        )

        return ExecutionResult(
            status=status,
            outputs=result_outputs,
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
            error=error_summary,
            duration_s=elapsed,
        )

    def execute_file(
        self,
        file_path: str | Path,
        *,
        timeout: float = 1800.0,
        allow_stdin: bool = False,
        env: dict[str, str] | None = None,
        save_output_notebook: bool = True,
    ) -> FileExecutionResult:
        """Execute a local .py script or .ipynb notebook on the remote kernel."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Local file not found: {path}")

        started = time.monotonic()
        if path.suffix.lower() == ".ipynb":
            nb_data, code_cells = parse_notebook_code_cells(path)
            cell_outputs: dict[int, list[dict[str, Any]]] = {}
            all_outputs: list[dict[str, Any]] = []
            cells_executed = 0
            overall_status = "ok"

            for idx, cell in enumerate(code_cells):
                if idx == 0 and env:
                    cell_env = env
                else:
                    cell_env = None
                res = self.execute_code(
                    cell["source"],
                    timeout=timeout,
                    allow_stdin=allow_stdin,
                    env=cell_env,
                    ensure_cwd=True,
                )
                cells_executed += 1
                cell_outputs[cell["index"]] = res.outputs
                all_outputs.extend(res.outputs)
                if res.status == "error":
                    overall_status = "error"
                    break

            out_nb_path: str | None = None
            if save_output_notebook:
                target = path.with_name(f"{path.stem}_output.ipynb")
                write_notebook_with_outputs(nb_data, cell_outputs, target)
                out_nb_path = str(target)

            return FileExecutionResult(
                file_path=str(path),
                status=overall_status,
                cells_executed=cells_executed,
                outputs=all_outputs,
                output_notebook_path=out_nb_path,
                duration_s=time.monotonic() - started,
            )

        code = path.read_text(encoding="utf-8")
        res = self.execute_code(
            code,
            timeout=timeout,
            allow_stdin=allow_stdin,
            env=env,
            ensure_cwd=True,
        )
        return FileExecutionResult(
            file_path=str(path),
            status=res.status,
            cells_executed=1,
            outputs=res.outputs,
            output_notebook_path=None,
            duration_s=time.monotonic() - started,
        )

    def restart(self) -> str | None:
        """Restart the remote Jupyter kernel cleanly."""
        self.kernel_client.restart()
        self._cwd_initialized = False
        self.kernel_id = getattr(self.kernel_client, "id", self.kernel_id)
        return self.kernel_id

    def get_kernel_state(self) -> str:
        """Refresh and return the remote kernel execution state."""
        manager = getattr(self.kernel_client, "_manager", None)
        if manager is not None:
            manager.refresh_model()
        return getattr(self.kernel_client, "execution_state", None) or "unknown"

    def stop(self) -> None:
        """Disconnect local kernel channels without shutting down the remote kernel."""
        if self._kernel_client is not None:
            with contextlib.suppress(Exception):
                self._kernel_client.stop(shutdown_kernel=False)
            self._kernel_client = None
            self._cwd_initialized = False
