"""Remote filesystem client backed by Jupyter Contents REST API (/api/contents)."""

from __future__ import annotations

import base64
import urllib.parse
from pathlib import Path
from typing import Any

import requests

from colab_mcp.models import ColabRequestError, RemoteFileEntry


class ContentsClient:
    """Kernel-independent file operations against a Colab runtime proxy."""

    def __init__(
        self,
        base_url: str,
        proxy_token: str,
        session: requests.Session | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.proxy_token = proxy_token
        self.session = session or requests.Session()
        self.timeout = timeout

    def _request(
        self,
        method: str,
        remote_path: str,
        *,
        params: dict[str, str] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        quoted_path = urllib.parse.quote(remote_path.strip("/"), safe="/")
        url = f"{self.base_url}/api/contents/{quoted_path}"
        query_params = {
            "authuser": "0",
            "colab-runtime-proxy-token": self.proxy_token,
        }
        if params:
            query_params.update(params)

        resp = self.session.request(
            method=method,
            url=url,
            params=query_params,
            headers={
                "X-Colab-Runtime-Proxy-Token": self.proxy_token,
                "X-Colab-Client-Agent": "colab-cli",
            },
            json=json_data,
            timeout=self.timeout,
        )
        if resp.status_code == 404:
            raise FileNotFoundError(f"Remote path not found: {remote_path}")
        if resp.status_code >= 400:
            raise ColabRequestError(
                status_code=resp.status_code,
                message=resp.text or f"HTTP {resp.status_code}",
            )
        if method == "DELETE" or resp.status_code == 204:
            return None
        return resp.json()

    def list_dir(self, remote_path: str = "/content") -> list[RemoteFileEntry]:
        """List files/directories at remote_path (or return single file metadata)."""
        data = self._request("GET", remote_path)
        if not data:
            return []
        if data.get("type") == "directory":
            return [RemoteFileEntry.model_validate(item) for item in data.get("content", [])]
        return [RemoteFileEntry.model_validate(data)]

    def upload(self, local_path: str | Path, remote_path: str) -> dict[str, Any]:
        """Upload a local file to remote_path via base64 PUT."""
        src = Path(local_path)
        if not src.exists():
            raise FileNotFoundError(f"Local path not found: {src}")
        raw_bytes = src.read_bytes()
        b64_content = base64.b64encode(raw_bytes).decode("ascii")
        filename = Path(remote_path).name or src.name
        self._request(
            "PUT",
            remote_path,
            json_data={
                "name": filename,
                "path": remote_path,
                "type": "file",
                "format": "base64",
                "content": b64_content,
                "chunk": 1,
            },
        )
        return {"name": filename, "path": remote_path, "size_bytes": len(raw_bytes)}

    def download(self, remote_path: str, local_path: str | Path) -> dict[str, Any]:
        """Download a remote file to local_path."""
        data = self._request("GET", remote_path, params={"content": "1"})
        if not data:
            raise FileNotFoundError(f"Remote path not found: {remote_path}")
        if data.get("type") == "directory":
            raise IsADirectoryError(f"Remote path is a directory: {remote_path}")

        fmt = data.get("format")
        payload = data.get("content") or ""
        if fmt == "base64":
            content_bytes = base64.b64decode(payload)
        else:
            content_bytes = str(payload).encode("utf-8")

        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(content_bytes)
        return {
            "remote_path": remote_path,
            "local_path": str(dst),
            "size_bytes": len(content_bytes),
        }

    def delete(self, remote_path: str) -> dict[str, Any]:
        """Delete a remote file or empty directory."""
        self._request("DELETE", remote_path)
        return {"deleted": True, "remote_path": remote_path}
