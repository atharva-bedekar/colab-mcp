import base64
from unittest.mock import MagicMock

import pytest


def test_upload_and_download_roundtrip(tmp_path):
    from colab_mcp.contents import ContentsClient

    src_file = tmp_path / "sample.bin"
    src_file.write_bytes(b"\x00\x01colab")
    sess = MagicMock()
    sess.request.return_value = MagicMock(
        status_code=200,
        json=lambda: {"name": "sample.bin", "path": "/content/sample.bin"},
    )
    client = ContentsClient("https://proxy.colab.google/", "tok-1", session=sess)
    res = client.upload(src_file, "/content/sample.bin")
    assert res["size_bytes"] == 7
    put_kwargs = sess.request.call_args.kwargs
    assert put_kwargs["json"]["format"] == "base64"
    assert base64.b64decode(put_kwargs["json"]["content"]) == b"\x00\x01colab"
    assert put_kwargs["headers"]["X-Colab-Runtime-Proxy-Token"] == "tok-1"

    out_bin = tmp_path / "nested" / "out.bin"
    sess.request.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(b"hello-bytes").decode("ascii"),
        },
    )
    dl = client.download("/content/out.bin", out_bin)
    assert dl["size_bytes"] == 11
    assert out_bin.read_bytes() == b"hello-bytes"

    out_txt = tmp_path / "out.txt"
    sess.request.return_value = MagicMock(
        status_code=200,
        json=lambda: {"type": "file", "format": "text", "content": "plain text"},
    )
    dl_txt = client.download("/content/out.txt", out_txt)
    assert dl_txt["size_bytes"] == 10
    assert out_txt.read_text(encoding="utf-8") == "plain text"


def test_list_dir_delete_and_errors(tmp_path):
    from colab_mcp.contents import ContentsClient

    sess = MagicMock()
    sess.request.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "type": "directory",
            "content": [
                {"name": "a.py", "path": "content/a.py", "type": "file", "size": 12},
                {"name": "sub", "path": "content/sub", "type": "directory"},
            ],
        },
    )
    client = ContentsClient("https://proxy.colab.google", "tok-1", session=sess)
    entries = client.list_dir("/content")
    assert len(entries) == 2
    assert entries[0].name == "a.py"

    sess.request.return_value = MagicMock(
        status_code=200,
        json=lambda: {"type": "directory", "content": []},
    )
    with pytest.raises(IsADirectoryError):
        client.download("/content", tmp_path / "dir_dl")

    sess.request.return_value = MagicMock(status_code=404, text="not found")
    with pytest.raises(FileNotFoundError):
        client.list_dir("/content/missing")

    sess.request.return_value = MagicMock(status_code=204, text="")
    assert client.delete("/content/a.py") == {"deleted": True, "remote_path": "/content/a.py"}


def test_download_sends_format_params_and_sha256(tmp_path):
    import hashlib

    from colab_mcp.contents import ContentsClient

    payload = b"sha-me"
    sess = MagicMock()
    sess.request.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(payload).decode("ascii"),
        },
    )
    client = ContentsClient("https://proxy.colab.google", "tok-1", session=sess)
    out = tmp_path / "sha.bin"
    res = client.download("/content/sha.bin", out)
    params = sess.request.call_args.kwargs["params"]
    assert params["content"] == "1"
    assert params["type"] == "file"
    assert params["format"] == "base64"
    assert out.read_bytes() == payload
    assert res["sha256"] == hashlib.sha256(payload).hexdigest()
    assert len(res["sha256"]) == 64


def test_download_json_dict_fallback(tmp_path):
    import json

    from colab_mcp.contents import ContentsClient

    sess = MagicMock()
    sess.request.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "type": "file",
            "format": "json",
            "content": {"b": 2, "a": [1, 3]},
        },
    )
    client = ContentsClient("https://proxy.colab.google", "tok-1", session=sess)
    out = tmp_path / "data.json"
    client.download("/content/data.json", out)
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed == {"b": 2, "a": [1, 3]}


def test_stat_returns_entry_with_content_zero():
    from colab_mcp.contents import ContentsClient
    from colab_mcp.models import RemoteFileEntry

    sess = MagicMock()
    sess.request.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "name": "a.py",
            "path": "content/a.py",
            "type": "file",
            "size": 12,
            "last_modified": "2026-01-01T00:00:00Z",
        },
    )
    client = ContentsClient("https://proxy.colab.google", "tok-1", session=sess)
    entry = client.stat("/content/a.py")
    assert isinstance(entry, RemoteFileEntry)
    assert entry.name == "a.py"
    assert entry.size == 12
    assert sess.request.call_args.kwargs["params"]["content"] == "0"


def test_read_file_tail_lines():
    from colab_mcp.contents import ContentsClient

    raw = b"one\ntwo\nthree\nfour"
    sess = MagicMock()
    sess.request.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(raw).decode("ascii"),
        },
    )
    client = ContentsClient("https://proxy.colab.google", "tok-1", session=sess)

    full = client.read_file("/content/log.txt", tail_lines=10)
    assert full["content"] == "one\ntwo\nthree\nfour"
    assert full["total_lines"] == 4
    assert full["truncated"] is False
    assert full["size_bytes"] == len(raw)

    tail = client.read_file("/content/log.txt", tail_lines=2)
    assert tail["content"] == "three\nfour"
    assert tail["truncated"] is True
    assert tail["total_lines"] == 4

    none = client.read_file("/content/log.txt", tail_lines=0)
    assert none["content"] == ""
    assert none["truncated"] is True


def test_read_file_max_bytes_truncation():
    from colab_mcp.contents import ContentsClient

    raw = b"abcdefghij"
    sess = MagicMock()
    sess.request.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(raw).decode("ascii"),
        },
    )
    client = ContentsClient("https://proxy.colab.google", "tok-1", session=sess)

    cut = client.read_file("/content/f.txt", max_bytes=4)
    assert cut["content"] == "abcd"
    assert cut["size_bytes"] == 10
    assert cut["truncated"] is True

    whole = client.read_file("/content/f.txt", max_bytes=100)
    assert whole["content"] == "abcdefghij"
    assert whole["truncated"] is False


def test_relative_paths_resolve_under_content(tmp_path):
    from colab_mcp.contents import ContentsClient, _normalize_remote_path

    assert _normalize_remote_path("rpo_bundle.zip") == "/content/rpo_bundle.zip"
    assert _normalize_remote_path("./sub/file.txt") == "/content/sub/file.txt"
    assert _normalize_remote_path(".hidden") == "/content/.hidden"
    assert _normalize_remote_path("content/rpo_bundle.zip") == "/content/rpo_bundle.zip"
    assert _normalize_remote_path("/content/rpo_bundle.zip") == "/content/rpo_bundle.zip"
    assert _normalize_remote_path("/tmp/archive.zip") == "/tmp/archive.zip"
    assert _normalize_remote_path("/") == "/"
    assert _normalize_remote_path(".") == "/content"

    src = tmp_path / "rpo_bundle.zip"
    src.write_bytes(b"zipdata")
    sess = MagicMock()
    sess.request.return_value = MagicMock(status_code=200, json=lambda: {})
    client = ContentsClient("https://proxy.colab.google", "tok-1", session=sess)

    up = client.upload(src, "rpo_bundle.zip")
    assert up["path"] == "/content/rpo_bundle.zip"
    assert sess.request.call_args.kwargs["url"] == (
        "https://proxy.colab.google/api/contents/content/rpo_bundle.zip"
    )
    assert sess.request.call_args.kwargs["json"]["path"] == "/content/rpo_bundle.zip"

    sess.request.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(b"zipdata").decode("ascii"),
        },
    )
    dl = client.download("rpo_bundle.zip", tmp_path / "out.zip")
    assert dl["remote_path"] == "/content/rpo_bundle.zip"
    assert sess.request.call_args.kwargs["url"] == (
        "https://proxy.colab.google/api/contents/content/rpo_bundle.zip"
    )

    # Absolute /tmp path remains rooted at /
    client.download("/tmp/archive.zip", tmp_path / "tmp_out.zip")
    assert sess.request.call_args.kwargs["url"] == (
        "https://proxy.colab.google/api/contents/tmp/archive.zip"
    )
