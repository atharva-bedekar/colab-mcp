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
