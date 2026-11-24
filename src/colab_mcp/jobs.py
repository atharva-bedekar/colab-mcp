"""Detached background job helpers executed on a remote Colab VM."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from colab_mcp.models import ColabError
from colab_mcp.runtime import ColabRuntime

JOBS_DIR = "/tmp/colab_mcp_jobs"
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def _validate_job_id(job_id: str) -> str:
    if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
        raise ValueError(
            f"Invalid job_id {job_id!r}: must match ^[A-Za-z0-9][A-Za-z0-9._-]{{0,63}}$"
        )
    return job_id


def _run_job_snippet(runtime: ColabRuntime, code: str) -> dict[str, Any]:
    res = runtime.execute_code(code, compact=True)
    if res.status != "ok":
        detail = res.error or res.stderr or "Remote background job command failed"
        raise ColabError(detail)

    raw = res.stdout.strip()
    if not raw:
        raise ColabError("Remote background job command produced no output")

    last_line = raw.splitlines()[-1]
    try:
        data = json.loads(last_line)
    except json.JSONDecodeError as exc:
        raise ColabError(f"Invalid JSON from remote background job command: {raw!r}") from exc

    if not isinstance(data, dict):
        raise ColabError(
            f"Expected JSON object from remote background job command, got {type(data).__name__}"
        )
    return data


def _require_found(data: dict[str, Any], job_id: str) -> dict[str, Any]:
    if data.get("not_found"):
        raise FileNotFoundError(f"Background job not found: {job_id}")
    return data


def start_background_job(
    runtime: ColabRuntime,
    command: str,
    *,
    name: str | None = None,
    cwd: str = "/content",
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Start a detached background shell command on the remote Colab VM."""
    job_id = f"job-{uuid.uuid4().hex[:8]}" if name is None else name
    _validate_job_id(job_id)

    code = f"""
import json
import os
import subprocess
from datetime import datetime, timezone

jobs_dir = {JOBS_DIR!r}
job_id = {job_id!r}
command = {command!r}
cwd = {cwd!r}
extra_env = {env!r}

os.makedirs(jobs_dir, exist_ok=True)
meta_path = os.path.join(jobs_dir, f"{{job_id}}.json")
log_path = os.path.join(jobs_dir, f"{{job_id}}.log")
exit_path = os.path.join(jobs_dir, f"{{job_id}}.exit")
if os.path.exists(exit_path):
    os.remove(exit_path)

proc_env = os.environ.copy()
if extra_env:
    proc_env.update(extra_env)

wrapper = 'bash -c "$1"; printf "%s" "$?" > "$2"'
with open(log_path, "w", encoding="utf-8") as log_file:
    proc = subprocess.Popen(
        ["bash", "-c", wrapper, "_", command, exit_path],
        cwd=cwd,
        env=proc_env,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

meta = {{
    "job_id": job_id,
    "pid": proc.pid,
    "command": command,
    "cwd": cwd,
    "log_path": log_path,
    "exit_path": exit_path,
    "started_at": datetime.now(timezone.utc).isoformat(),
}}
with open(meta_path, "w", encoding="utf-8") as meta_file:
    json.dump(meta, meta_file)

print(json.dumps({{
    "job_id": job_id,
    "pid": proc.pid,
    "log_path": log_path,
    "status": "running",
}}))
""".lstrip()

    return _run_job_snippet(runtime, code)


def check_background_job(
    runtime: ColabRuntime,
    job_id: str,
    *,
    tail_lines: int = 50,
) -> dict[str, Any]:
    """Check status and tail logs for a background job on the remote Colab VM."""
    _validate_job_id(job_id)

    code = f"""
import json
import os
from collections import deque

jobs_dir = {JOBS_DIR!r}
job_id = {job_id!r}
tail_lines = {int(tail_lines)!r}

meta_path = os.path.join(jobs_dir, f"{{job_id}}.json")
if not os.path.exists(meta_path):
    print(json.dumps({{"not_found": True}}))
else:
    with open(meta_path, "r", encoding="utf-8") as meta_file:
        meta = json.load(meta_file)

    pid = int(meta["pid"])
    log_path = meta["log_path"]
    exit_path = meta["exit_path"]

    exit_code = None
    status = "running"
    if os.path.exists(exit_path):
        with open(exit_path, "r", encoding="utf-8") as exit_file:
            raw_exit = exit_file.read().strip()
        if raw_exit:
            exit_code = int(raw_exit)
            status = "exited"

    if status == "running":
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = True

        if alive and os.path.exists(f"/proc/{{pid}}/stat"):
            with open(f"/proc/{{pid}}/stat", "r", encoding="utf-8", errors="replace") as stat_file:
                stat_txt = stat_file.read()
            rparen = stat_txt.rfind(")")
            if rparen != -1 and len(stat_txt) > rparen + 2 and stat_txt[rparen + 2] == "Z":
                alive = False

        if not alive:
            status = "exited"

    log_tail = ""
    if tail_lines > 0 and os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="replace") as log_file:
            log_tail = "".join(deque(log_file, maxlen=tail_lines))

    print(json.dumps({{
        "job_id": job_id,
        "pid": pid,
        "status": status,
        "exit_code": exit_code,
        "log_path": log_path,
        "log_tail": log_tail,
    }}))
""".lstrip()

    return _require_found(_run_job_snippet(runtime, code), job_id)


def stop_background_job(runtime: ColabRuntime, job_id: str) -> dict[str, Any]:
    """Send SIGTERM to a background job's process group on the remote Colab VM."""
    _validate_job_id(job_id)

    code = f"""
import json
import os
import signal

jobs_dir = {JOBS_DIR!r}
job_id = {job_id!r}

meta_path = os.path.join(jobs_dir, f"{{job_id}}.json")
if not os.path.exists(meta_path):
    print(json.dumps({{"not_found": True}}))
else:
    with open(meta_path, "r", encoding="utf-8") as meta_file:
        meta = json.load(meta_file)

    pid = int(meta["pid"])
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    print(json.dumps({{
        "job_id": job_id,
        "pid": pid,
        "stopped": True,
    }}))
""".lstrip()

    return _require_found(_run_job_snippet(runtime, code), job_id)
