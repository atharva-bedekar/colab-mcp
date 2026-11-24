import hashlib
import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from colab_mcp.models import ColabError, ExecutionResult
from colab_mcp.transfer import download_dir, upload_dir


def _build_zip_bytes(entries: dict[str, bytes | str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_download_dir_happy_path(tmp_path):
    zip_bytes = _build_zip_bytes({"a.txt": "hello", "sub/b.bin": b"\x01\x02\x03"})
    expected_sha = hashlib.sha256(zip_bytes).hexdigest()
    remote_zip = "/tmp/colab_mcp_dl_test123.zip"

    runtime = MagicMock()
    runtime.execute_code.return_value = ExecutionResult(
        status="ok",
        stdout=json.dumps(
            {
                "zip_path": remote_zip,
                "sha256": expected_sha,
                "files_count": 2,
                "size_bytes": len(zip_bytes),
            }
        ),
    )

    def fake_download(remote_path, local_path):
        assert remote_path == remote_zip
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(zip_bytes)
        return {
            "remote_path": remote_path,
            "local_path": str(dst),
            "size_bytes": len(zip_bytes),
            "sha256": expected_sha,
        }

    contents = MagicMock()
    contents.download.side_effect = fake_download

    dest_dir = tmp_path / "downloaded"
    result = download_dir(runtime, contents, "/content/data", dest_dir)

    runtime.execute_code.assert_called_once()
    assert runtime.execute_code.call_args.kwargs["compact"] is True
    contents.delete.assert_called_once_with(remote_zip)
    assert (dest_dir / "a.txt").read_text(encoding="utf-8") == "hello"
    assert (dest_dir / "sub" / "b.bin").read_bytes() == b"\x01\x02\x03"
    assert result == {
        "remote_dir": "/content/data",
        "local_dir": str(dest_dir),
        "files_count": 2,
        "archive_size_bytes": len(zip_bytes),
        "sha256": expected_sha,
        "verified": True,
    }


def test_download_dir_sha256_mismatch_raises_and_leaves_no_files(tmp_path):
    zip_bytes = _build_zip_bytes({"secret.txt": "payload"})
    remote_zip = "/tmp/colab_mcp_dl_mismatch.zip"

    runtime = MagicMock()
    runtime.execute_code.return_value = ExecutionResult(
        status="ok",
        stdout=json.dumps(
            {
                "zip_path": remote_zip,
                "sha256": "0" * 64,
                "files_count": 1,
                "size_bytes": len(zip_bytes),
            }
        ),
    )

    def fake_download(_remote_path, local_path):
        Path(local_path).write_bytes(zip_bytes)
        return {"size_bytes": len(zip_bytes), "sha256": hashlib.sha256(zip_bytes).hexdigest()}

    contents = MagicMock()
    contents.download.side_effect = fake_download

    dest_dir = tmp_path / "dest"
    with pytest.raises(ColabError, match="SHA-256 mismatch"):
        download_dir(runtime, contents, "/content/data", dest_dir)

    contents.delete.assert_called_once_with(remote_zip)
    assert not dest_dir.exists()


def test_download_dir_zip_slip_rejected(tmp_path):
    zip_bytes = _build_zip_bytes({"good.txt": "ok", "../escape.txt": "pwned"})
    expected_sha = hashlib.sha256(zip_bytes).hexdigest()
    remote_zip = "/tmp/colab_mcp_dl_slip.zip"

    runtime = MagicMock()
    runtime.execute_code.return_value = ExecutionResult(
        status="ok",
        stdout=json.dumps(
            {
                "zip_path": remote_zip,
                "sha256": expected_sha,
                "files_count": 2,
                "size_bytes": len(zip_bytes),
            }
        ),
    )

    def fake_download(_remote_path, local_path):
        Path(local_path).write_bytes(zip_bytes)
        return {"size_bytes": len(zip_bytes), "sha256": expected_sha}

    contents = MagicMock()
    contents.download.side_effect = fake_download

    dest_dir = tmp_path / "dest"
    with pytest.raises(ColabError, match="traversal"):
        download_dir(runtime, contents, "/content/data", dest_dir)

    contents.delete.assert_called_once_with(remote_zip)
    assert not (tmp_path / "escape.txt").exists()
    assert not (dest_dir / "good.txt").exists()


def test_download_dir_remote_missing_raises_file_not_found(tmp_path):
    runtime = MagicMock()
    runtime.execute_code.return_value = ExecutionResult(
        status="error",
        error="FileNotFoundError: Remote directory not found: /content/missing",
    )
    contents = MagicMock()

    with pytest.raises(FileNotFoundError):
        download_dir(runtime, contents, "/content/missing", tmp_path / "out")

    contents.download.assert_not_called()
    contents.delete.assert_not_called()


def test_upload_dir_happy_path_with_glob(tmp_path):
    src_dir = tmp_path / "proj"
    (src_dir / "nested").mkdir(parents=True)
    (src_dir / "keep.py").write_text("print('keep')", encoding="utf-8")
    (src_dir / "nested" / "also_keep.py").write_text("x = 1", encoding="utf-8")
    (src_dir / "ignore.txt").write_text("skip", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_upload(local_path, remote_path):
        data = Path(local_path).read_bytes()
        captured["zip_bytes"] = data
        captured["remote_path"] = remote_path
        return {"name": Path(remote_path).name, "path": remote_path, "size_bytes": len(data)}

    contents = MagicMock()
    contents.upload.side_effect = fake_upload

    runtime = MagicMock()
    runtime.execute_code.return_value = ExecutionResult(status="ok", stdout='{"verified": true}\n')

    result = upload_dir(runtime, contents, src_dir, "/content/proj", glob="*.py")

    uploaded_bytes = captured["zip_bytes"]
    assert isinstance(uploaded_bytes, bytes)
    expected_sha = hashlib.sha256(uploaded_bytes).hexdigest()
    remote_zip_path = captured["remote_path"]
    assert isinstance(remote_zip_path, str)
    assert remote_zip_path.startswith("/tmp/colab_mcp_ul_")

    with zipfile.ZipFile(io.BytesIO(uploaded_bytes), "r") as zf:
        assert sorted(zf.namelist()) == ["keep.py", "nested/also_keep.py"]

    runtime.execute_code.assert_called_once()
    exec_code_arg = runtime.execute_code.call_args.args[0]
    assert runtime.execute_code.call_args.kwargs["compact"] is True
    assert expected_sha in exec_code_arg
    assert "/content/proj" in exec_code_arg
    assert remote_zip_path in exec_code_arg

    assert result == {
        "local_dir": str(src_dir),
        "remote_dir": "/content/proj",
        "files_count": 2,
        "archive_size_bytes": len(uploaded_bytes),
        "sha256": expected_sha,
        "verified": True,
    }


def test_upload_dir_missing_and_non_directory_errors(tmp_path):
    runtime = MagicMock()
    contents = MagicMock()

    with pytest.raises(FileNotFoundError):
        upload_dir(runtime, contents, tmp_path / "does_not_exist", "/content/proj")

    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        upload_dir(runtime, contents, file_path, "/content/proj")


def test_upload_dir_remote_extract_error_raises_colab_error(tmp_path):
    src_dir = tmp_path / "proj"
    src_dir.mkdir()
    (src_dir / "a.txt").write_text("data", encoding="utf-8")

    contents = MagicMock()
    runtime = MagicMock()
    runtime.execute_code.return_value = ExecutionResult(
        status="error",
        error="ValueError: SHA-256 mismatch",
    )

    with pytest.raises(ColabError, match="SHA-256 mismatch"):
        upload_dir(runtime, contents, src_dir, "/content/proj")
