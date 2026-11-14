"""Colab session control-plane HTTP client (/tun/m/*)."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from typing import Any

import requests

from colab_mcp.models import (
    Accelerator,
    Assignment,
    ColabRequestError,
    GetAssignmentTokenResponse,
    ListedAssignment,
    Shape,
    TooManyAssignmentsError,
    Variant,
    strip_xssi_prefix,
    uuid_to_web_safe_base64,
)

COLAB_RESEARCH_DOMAIN = "https://colab.research.google.com"
COLAB_SANDBOX_DOMAIN = "https://colab.sandbox.google.com"
TUN_PREFIX = "/tun/m"
DEFAULT_REQUEST_TIMEOUT = 120.0
_RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})


class ColabClient:
    """REST client for Colab runtime assignment, listing, unassignment, and keep-alive."""

    def __init__(
        self,
        session: requests.Session,
        colab_domain: str = COLAB_RESEARCH_DOMAIN,
        sandbox_domain: str = COLAB_RESEARCH_DOMAIN,
        request_timeout: float | None = None,
    ) -> None:
        self.session = session
        self.colab_domain = colab_domain.rstrip("/")
        self.sandbox_domain = sandbox_domain.rstrip("/")
        self.request_timeout = request_timeout or DEFAULT_REQUEST_TIMEOUT

    def _issue_request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json_data: Any = None,
        timeout: float | None = None,
    ) -> dict[str, Any] | list[Any] | None:
        req_headers = {
            "Accept": "application/json",
            "X-Colab-Client-Agent": "colab-cli",
            "X-Colab-Tunnel": "Google",
        }
        if headers:
            req_headers.update(headers)

        resp = self.session.request(
            method=method,
            url=url,
            params=params,
            headers=req_headers,
            json=json_data,
            timeout=timeout or self.request_timeout,
        )
        if resp.status_code == 412:
            raise TooManyAssignmentsError(
                status_code=412,
                message=resp.text or "Too many active Colab assignments",
            )
        if resp.status_code >= 400:
            raise ColabRequestError(
                status_code=resp.status_code,
                message=resp.text or f"HTTP {resp.status_code}",
            )
        if resp.status_code == 204 or not resp.text or not resp.text.strip():
            return None
        return json.loads(strip_xssi_prefix(resp.text))

    @staticmethod
    def _build_assign_params(
        notebook_hash: uuid.UUID,
        variant: Variant,
        accelerator: Accelerator,
        shape: Shape | None,
    ) -> dict[str, str]:
        params: dict[str, str] = {
            "nbh": uuid_to_web_safe_base64(notebook_hash),
            "authuser": "0",
        }
        if variant != Variant.DEFAULT:
            params["variant"] = variant.value
        if accelerator != Accelerator.NONE:
            params["accelerator"] = accelerator.value
        if shape == Shape.HIGH_RAM:
            params["shape"] = "hm"
        return params

    def get_assignment(
        self,
        notebook_hash: uuid.UUID,
        variant: Variant = Variant.DEFAULT,
        accelerator: Accelerator = Accelerator.NONE,
        shape: Shape | None = None,
    ) -> Assignment | GetAssignmentTokenResponse:
        """Query an existing assignment or obtain an XSRF token for POST /tun/m/assign."""
        url = f"{self.colab_domain}{TUN_PREFIX}/assign"
        params = self._build_assign_params(notebook_hash, variant, accelerator, shape)
        data = self._issue_request("GET", url, params=params)
        if not isinstance(data, dict):
            raise ColabRequestError("Unexpected response format from /tun/m/assign")
        if "runtimeProxyInfo" in data or "endpoint" in data:
            return Assignment.model_validate(data)
        return GetAssignmentTokenResponse.model_validate(data)

    def assign(
        self,
        notebook_hash: uuid.UUID,
        variant: Variant = Variant.DEFAULT,
        accelerator: Accelerator = Accelerator.NONE,
        shape: Shape | None = None,
        retries: int = 3,
        backoff: float = 10.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> Assignment:
        """Allocate or retrieve a Colab runtime using the two-step XSRF handshake."""
        url = f"{self.colab_domain}{TUN_PREFIX}/assign"
        params = self._build_assign_params(notebook_hash, variant, accelerator, shape)

        current = self.get_assignment(notebook_hash, variant, accelerator, shape)
        if isinstance(current, Assignment):
            return current

        token_resp = current
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                data = self._issue_request(
                    "POST",
                    url,
                    params=params,
                    headers={"X-Goog-Colab-Token": token_resp.token},
                )
                if not isinstance(data, dict):
                    raise ColabRequestError("Unexpected POST /tun/m/assign response")
                return Assignment.model_validate(data)
            except TooManyAssignmentsError:
                raise
            except (requests.exceptions.ReadTimeout, ColabRequestError) as exc:
                if isinstance(exc, ColabRequestError) and exc.status_code not in _RETRYABLE_STATUS_CODES:
                    raise
                last_exc = exc
                if attempt >= retries:
                    break
                sleep_fn(backoff * (2**attempt))
                reconciled = self.get_assignment(notebook_hash, variant, accelerator, shape)
                if isinstance(reconciled, Assignment):
                    return reconciled
                token_resp = reconciled

        assert last_exc is not None
        raise last_exc

    def list_assignments(self) -> list[ListedAssignment]:
        """List active Colab runtime assignments for the authenticated user."""
        url = f"{self.colab_domain}{TUN_PREFIX}/assignments"
        data = self._issue_request("GET", url, params={"authuser": "0"})
        if data is None:
            return []
        raw_items = data.get("assignments", []) if isinstance(data, dict) else data
        return [ListedAssignment.model_validate(item) for item in raw_items]

    def unassign(self, endpoint: str) -> None:
        """Release a Colab runtime via the two-step sandbox unassign handshake."""
        url = f"{self.sandbox_domain}{TUN_PREFIX}/unassign/{endpoint}"
        params = {"authuser": "0"}
        token_data = self._issue_request("GET", url, params=params)
        if not isinstance(token_data, dict) or "token" not in token_data:
            raise ColabRequestError(f"Missing unassign XSRF token for endpoint {endpoint}")
        self._issue_request(
            "POST",
            url,
            params=params,
            headers={"X-Goog-Colab-Token": str(token_data["token"])},
        )

    def keep_alive(self, endpoint: str, timeout: float = 15.0) -> bool:
        """Ping Colab Tunnel Frontend keep-alive; ReadTimeout counts as success."""
        url = f"{self.colab_domain}{TUN_PREFIX}/{endpoint}/keep-alive/"
        try:
            self._issue_request(
                "GET",
                url,
                params={"authuser": "0"},
                headers={"X-Colab-Tunnel": "Google"},
                timeout=timeout,
            )
            return True
        except requests.exceptions.ReadTimeout:
            return True
