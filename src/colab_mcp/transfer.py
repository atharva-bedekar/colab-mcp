"""SHA-256-verified directory upload and download over Colab runtime + Contents API."""

from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import json
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from colab_mcp.contents import ContentsClient
from colab_mcp.models import ColabError
from colab_mcp.runtime import ColabRuntime

_CHUNK_SIZE = 65536


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_json_line(stdout: str) -> dict[str, Any]:
    """Parse the last non-empty JSON object line from remote stdout."""
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ColabError("Remote archive command produced no output")
    try:
        data = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ColabError(f"Invalid JSON output from remote archive command: {lines[-1]!r}") from exc
    if not isinstance(data, dict):
        raise ColabError(
            f"Expected JSON object from remote archive command, got {type(data).__name__}"
        )
    return data


def _validate_zip_members(zf: zipfile.ZipFile, target_root: Path) -> list[tuple[zipfile.ZipInfo, Path]]:
    """Validate all zip members stay within target_root (zip-slip protection)."""
    resolved_root = target_root.resolve()
    validated: list[tuple[zipfile.ZipInfo, Path]] = []
    for member in zf.infolist():
        if PurePosixPath(member.filename).is_absolute() or Path(member.filename).is_absolute():
            raise ColabError(f"Unsafe absolute path in zip archive: {member.filename!r}")
        dest = (target_root / member.filename).resolve()
        if not dest.is_relative_to(resolved_root):
            raise ColabError(f"Unsafe path traversal in zip archive: {member.filename!r}")
        if not member.is_dir() and dest == resolved_root:
            raise ColabError(f"Unsafe zip member targets root directory: {member.filename!r}")
        validated.append((member, dest))
    return validated


def _safe_extract_zip(zip_path: Path, target_root: Path) -> int:
    """Safely extract zip_path into target_root after validating all member paths."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = _validate_zip_members(zf, target_root)
        target_root.mkdir(parents=True, exist_ok=True)
        extracted_files = 0
        for member, dest in members:
            if member.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_files += 1
        return extracted_files


def _build_remote_archive_code(remote_dir: str, zip_path: str, glob_pattern: str | None) -> str:
    return f"""
import fnmatch
import hashlib
import json
import zipfile
from pathlib import Path

_root = Path({remote_dir!r})
if not _root.exists():
    raise FileNotFoundError(f"Remote directory not found: {{_root}}")
if not _root.is_dir():
    raise NotADirectoryError(f"Remote path is not a directory: {{_root}}")

_zip_path = Path({zip_path!r})
_glob = {glob_pattern!r}
_files = sorted(p for p in _root.rglob("*") if p.is_file())
_selected = []
for _p in _files:
    _rel = _p.relative_to(_root).as_posix()
    if _glob is None or fnmatch.fnmatch(_rel, _glob):
        _selected.append((_p, _rel))

try:
    with zipfile.ZipFile(_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as _zf:
        for _p, _rel in _selected:
            _zf.write(_p, arcname=_rel)
    _digest = hashlib.sha256()
    with _zip_path.open("rb") as _fh:
        for _chunk in iter(lambda: _fh.read(65536), b""):
            _digest.update(_chunk)
    print(json.dumps({{
        "zip_path": str(_zip_path),
        "sha256": _digest.hexdigest(),
        "files_count": len(_selected),
        "size_bytes": _zip_path.stat().st_size,
    }}))
except Exception:
    _zip_path.unlink(missing_ok=True)
    raise
""".strip()


def _build_remote_extract_code(
    remote_zip_path: str,
    remote_dir: str,
    expected_sha256: str,
) -> str:
    return f"""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

_zip_path = Path({remote_zip_path!r})
_target = Path({remote_dir!r})
_expected_sha = {expected_sha256!r}

try:
    _digest = hashlib.sha256()
    with _zip_path.open("rb") as _fh:
        for _chunk in iter(lambda: _fh.read(65536), b""):
            _digest.update(_chunk)
    _actual_sha = _digest.hexdigest()
    if _actual_sha.lower() != _expected_sha.lower():
        raise ValueError(f"SHA-256 mismatch: expected {{_expected_sha}}, got {{_actual_sha}}")

    _resolved_root = _target.resolve()
    with zipfile.ZipFile(_zip_path, "r") as _zf:
        _validated = []
        for _member in _zf.infolist():
            if PurePosixPath(_member.filename).is_absolute() or Path(_member.filename).is_absolute():
                raise ValueError(f"Unsafe absolute path in zip archive: {{_member.filename!r}}")
            _dest = (_target / _member.filename).resolve()
            if not _dest.is_relative_to(_resolved_root):
                raise ValueError(f"Unsafe path traversal in zip archive: {{_member.filename!r}}")
            if not _member.is_dir() and _dest == _resolved_root:
                raise ValueError(f"Unsafe zip member targets root directory: {{_member.filename!r}}")
            _validated.append((_member, _dest))

        _target.mkdir(parents=True, exist_ok=True)
        for _member, _dest in _validated:
            if _member.is_dir():
                _dest.mkdir(parents=True, exist_ok=True)
                continue
            _dest.parent.mkdir(parents=True, exist_ok=True)
            with _zf.open(_member, "r") as _src, _dest.open("wb") as _dst:
                shutil.copyfileobj(_src, _dst)

    print(json.dumps({{"sha256": _actual_sha, "verified": True}}))
finally:
    _zip_path.unlink(missing_ok=True)
""".strip()


def download_dir(
    runtime: ColabRuntime,
    contents: ContentsClient,
    remote_dir: str,
    local_dir: str | Path,
    *,
    glob: str | None = None,
) -> dict[str, Any]:
    """Archive a remote directory, download it, verify SHA-256, and safely extract locally."""
    candidate_zip = f"/tmp/colab_mcp_dl_{uuid.uuid4().hex}.zip"
    code = _build_remote_archive_code(remote_dir, candidate_zip, glob)
    res = runtime.execute_code(code, compact=True)
    if res.status != "ok":
        err = res.error or "Failed to archive remote directory"
        if "FileNotFoundError" in err:
            raise FileNotFoundError(err)
        raise ColabError(err)

    meta = _parse_json_line(res.stdout)
    remote_zip_path = str(meta.get("zip_path") or "")
    remote_sha256 = str(meta.get("sha256") or "")
    if not remote_zip_path or not remote_sha256:
        raise ColabError(f"Incomplete archive metadata from remote runtime: {meta!r}")

    target_root = Path(local_dir)
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_zip_path = Path(tmp_dir) / "archive.zip"
        try:
            dl_res = contents.download(remote_zip_path, local_zip_path)
        finally:
            with contextlib.suppress(Exception):
                contents.delete(remote_zip_path)

        if local_zip_path.exists():
            local_sha256 = _sha256_file(local_zip_path)
            archive_size_bytes = local_zip_path.stat().st_size
        else:
            dl_sha = dl_res.get("sha256") if isinstance(dl_res, dict) else None
            if not isinstance(dl_sha, str) or not dl_sha:
                raise ColabError(f"Downloaded archive missing at {local_zip_path}")
            local_sha256 = dl_sha
            archive_size_bytes = int(meta.get("size_bytes", 0))

        if local_sha256.lower() != remote_sha256.lower():
            raise ColabError(
                f"SHA-256 mismatch for {remote_zip_path}: "
                f"expected {remote_sha256}, got {local_sha256}"
            )

        extracted_count = _safe_extract_zip(local_zip_path, target_root)

    files_count = int(meta.get("files_count", extracted_count))
    return {
        "remote_dir": remote_dir,
        "local_dir": str(target_root),
        "files_count": files_count,
        "archive_size_bytes": archive_size_bytes,
        "sha256": local_sha256,
        "verified": True,
    }


def upload_dir(
    runtime: ColabRuntime,
    contents: ContentsClient,
    local_dir: str | Path,
    remote_dir: str,
    *,
    glob: str | None = None,
) -> dict[str, Any]:
    """Zip a local directory, upload it, and verify SHA-256 + safe extract on remote."""
    src_root = Path(local_dir)
    if not src_root.exists():
        raise FileNotFoundError(f"Local directory not found: {src_root}")
    if not src_root.is_dir():
        raise ValueError(f"Local path is not a directory: {src_root}")

    selected: list[tuple[Path, str]] = []
    for path in sorted(p for p in src_root.rglob("*") if p.is_file()):
        rel_posix = path.relative_to(src_root).as_posix()
        if glob is None or fnmatch.fnmatch(rel_posix, glob):
            selected.append((path, rel_posix))

    remote_zip_path = f"/tmp/colab_mcp_ul_{uuid.uuid4().hex}.zip"
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_zip_path = Path(tmp_dir) / "archive.zip"
        with zipfile.ZipFile(local_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path, rel_posix in selected:
                zf.write(path, arcname=rel_posix)

        local_sha256 = _sha256_file(local_zip_path)
        archive_size_bytes = local_zip_path.stat().st_size
        contents.upload(local_zip_path, remote_zip_path)

    extract_code = _build_remote_extract_code(remote_zip_path, remote_dir, local_sha256)
    res = runtime.execute_code(extract_code, compact=True)
    if res.status != "ok":
        raise ColabError(res.error or "Failed to extract uploaded archive")

    return {
        "local_dir": str(src_root),
        "remote_dir": remote_dir,
        "files_count": len(selected),
        "archive_size_bytes": archive_size_bytes,
        "sha256": local_sha256,
        "verified": True,
    }
