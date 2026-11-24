"""Persistent session state manager and in-process asyncio keep-alive loop."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from colab_mcp.auth import AuthManager
from colab_mcp.client import ColabClient
from colab_mcp.contents import ContentsClient
from colab_mcp.models import (
    Accelerator,
    Assignment,
    AssignmentVariant,
    AuthProvider,
    ListedAssignment,
    RuntimeProxyInfo,
    SessionInfo,
    SessionState,
    SessionStatus,
    SessionSummary,
    Shape,
    Variant,
    resolve_runtime_options,
)
from colab_mcp.runtime import ColabRuntime

logger = logging.getLogger("colab-mcp")

_REFRESH_WINDOW_SECONDS = 300


def _parse_shape(shape_str: str | None) -> Shape | None:
    if not shape_str or shape_str == "NONE":
        return None
    if shape_str == "HIGH_RAM" or shape_str == "1":
        return Shape.HIGH_RAM
    if shape_str == "STANDARD" or shape_str == "0":
        return Shape.STANDARD
    return Shape[shape_str]


def _assignment_variant_name(variant: AssignmentVariant | Variant | int | str) -> str:
    if isinstance(variant, Variant):
        return variant.value
    try:
        return AssignmentVariant(variant).name
    except ValueError:
        return str(variant)


def _find_local_session(
    sessions: dict[str, SessionState], name_or_endpoint: str
) -> tuple[str, SessionState] | None:
    if name_or_endpoint in sessions:
        return name_or_endpoint, sessions[name_or_endpoint]
    for key, state in sessions.items():
        if state.endpoint == name_or_endpoint:
            return key, state
    return None


def _find_server_assignment(
    client: ColabClient, endpoint: str
) -> ListedAssignment | None:
    for item in client.list_assignments():
        if item.endpoint == endpoint:
            return item
    return None


def _adopt_listed_assignment(item: ListedAssignment) -> SessionState:
    assert item.runtime_proxy_info is not None
    return SessionState(
        name=item.endpoint,
        notebook_hash=str(uuid.uuid4()),
        endpoint=item.endpoint,
        variant=_assignment_variant_name(item.variant),
        accelerator=str(item.accelerator),
        shape=item.machine_shape.name,
        runtime_url=item.runtime_proxy_info.url,
        proxy_token=item.runtime_proxy_info.token,
        proxy_expires_at=item.runtime_proxy_info.compute_expires_at().isoformat(),
    )


class SessionManager:
    """Coordinates auth, ColabClient, ContentsClient, ColabRuntime, and sessions.json."""

    def __init__(
        self,
        state_path: Path | None = None,
        auth_manager: AuthManager | None = None,
        client_factory: Callable[[requests.Session], ColabClient] = ColabClient,
        contents_factory: Callable[[str, str], ContentsClient] = ContentsClient,
        runtime_factory: Callable[..., ColabRuntime] = ColabRuntime,
        keep_alive_interval_s: float = 60.0,
        default_auth_provider: AuthProvider = AuthProvider.OAUTH2,
    ) -> None:
        self.state_path = state_path or (
            Path.home() / ".config" / "colab-mcp" / "sessions.json"
        )
        self.auth_manager = auth_manager or AuthManager()
        self._client_factory = client_factory
        self._contents_factory = contents_factory
        self._runtime_factory = runtime_factory
        self.keep_alive_interval_s = keep_alive_interval_s
        self.default_auth_provider = default_auth_provider
        self._runtimes: dict[str, ColabRuntime] = {}
        self._lock = threading.RLock()
        self._keep_alive_task: asyncio.Task[None] | None = None
        self._last_keep_alive_at: dict[str, str] = {}

    def _get_client(self) -> ColabClient:
        session = self.auth_manager.get_authorized_session(self.default_auth_provider)
        return self._client_factory(session)

    def load_sessions(self) -> dict[str, SessionState]:
        """Load persisted sessions from sessions.json."""
        with self._lock:
            if not self.state_path.exists():
                return {}
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return {}
            return {
                name: SessionState.model_validate(payload)
                for name, payload in raw.items()
            }

    def save_sessions(self, sessions: dict[str, SessionState]) -> None:
        """Persist sessions dict to sessions.json."""
        with self._lock:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            serialized = {
                name: state.model_dump(mode="json")
                for name, state in sessions.items()
            }
            self.state_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

    def update_session_execution(
        self,
        name: str,
        *,
        running: str | None,
        last_execution: tuple[str, str | None, str] | None = None,
    ) -> None:
        """Update execution tracking fields for a stored session."""
        with self._lock:
            sessions = self.load_sessions()
            state = sessions.get(name)
            if state is None:
                return
            state.running = running
            if last_execution is not None:
                state.last_execution = last_execution
            sessions[name] = state
            self.save_sessions(sessions)

    def _apply_proxy_refresh(
        self,
        key: str,
        state: SessionState,
        sessions: dict[str, SessionState],
        proxy_info: RuntimeProxyInfo,
    ) -> None:
        new_token = proxy_info.token
        new_url = proxy_info.url
        new_expires = proxy_info.compute_expires_at().isoformat()

        if new_token != state.proxy_token or new_url != state.runtime_url:
            old_rt = self._runtimes.pop(key, None)
            if old_rt is not None:
                with contextlib.suppress(Exception):
                    old_rt.stop()

        state.proxy_token = new_token
        state.runtime_url = new_url
        state.proxy_expires_at = new_expires
        sessions[key] = state
        self.save_sessions(sessions)

    def ensure_fresh_session(self, name: str) -> SessionState:
        """Look up or adopt a session and refresh proxy token if within 300s of expiry."""
        with self._lock:
            sessions = self.load_sessions()
            matched = _find_local_session(sessions, name)
            if matched is None:
                client = self._get_client()
                listed = _find_server_assignment(client, name)
                if listed is None or listed.runtime_proxy_info is None:
                    raise KeyError(f"Session '{name}' not found")
                adopted = _adopt_listed_assignment(listed)
                sessions[adopted.name] = adopted
                self.save_sessions(sessions)
                return adopted

            key, state = matched
            expires_at = datetime.fromisoformat(state.proxy_expires_at)
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)

            if expires_at - datetime.now(timezone.utc) <= timedelta(
                seconds=_REFRESH_WINDOW_SECONDS
            ):
                client = self._get_client()
                assignment = client.get_assignment(
                    uuid.UUID(state.notebook_hash),
                    Variant(state.variant),
                    Accelerator(state.accelerator),
                    _parse_shape(state.shape),
                )
                refreshed_proxy: RuntimeProxyInfo | None = None
                if isinstance(assignment, Assignment):
                    refreshed_proxy = assignment.runtime_proxy_info
                else:
                    listed = _find_server_assignment(client, state.endpoint)
                    if listed is not None:
                        refreshed_proxy = listed.runtime_proxy_info
                if refreshed_proxy is not None:
                    self._apply_proxy_refresh(key, state, sessions, refreshed_proxy)

            return state

    def get_contents_client(self, name: str) -> ContentsClient:
        """Return a ContentsClient bound to the session's current proxy URL and token."""
        state = self.ensure_fresh_session(name)
        return self._contents_factory(state.runtime_url, state.proxy_token)

    def get_runtime(self, name: str) -> ColabRuntime:
        """Return a cached or newly created ColabRuntime for the named session."""
        state = self.ensure_fresh_session(name)
        with self._lock:
            rt = self._runtimes.get(state.name)
            if rt is None:
                rt = self._runtime_factory(
                    state.runtime_url,
                    state.proxy_token,
                    kernel_id=state.kernel_id,
                )
                self._runtimes[state.name] = rt
            return rt

    def create_session(
        self,
        name: str | None = None,
        gpu: str | None = None,
        tpu: str | None = None,
        high_mem: bool = False,
    ) -> SessionInfo:
        """Allocate a new Colab runtime, persist state, and start keep-alive."""
        session_name = name or uuid.uuid4().hex[:6]
        with self._lock:
            sessions = self.load_sessions()
            if session_name in sessions:
                raise ValueError(f"Session '{session_name}' already exists")

        variant, accelerator, shape = resolve_runtime_options(
            gpu=gpu,
            tpu=tpu,
            high_mem=high_mem,
        )
        notebook_hash = uuid.uuid4()
        client = self._get_client()
        assignment = client.assign(notebook_hash, variant, accelerator, shape)

        shape_label = shape.name if shape is not None else "STANDARD"
        state = SessionState(
            name=session_name,
            notebook_hash=str(notebook_hash),
            endpoint=assignment.endpoint,
            variant=variant.value,
            accelerator=accelerator.value,
            shape=shape_label,
            runtime_url=assignment.runtime_proxy_info.url,
            proxy_token=assignment.runtime_proxy_info.token,
            proxy_expires_at=assignment.runtime_proxy_info.compute_expires_at().isoformat(),
        )
        with self._lock:
            sessions = self.load_sessions()
            sessions[session_name] = state
            self.save_sessions(sessions)

        with contextlib.suppress(Exception):
            client.keep_alive(assignment.endpoint)
            self._last_keep_alive_at[assignment.endpoint] = datetime.now(
                timezone.utc
            ).isoformat()

        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop()
            self.start_keep_alive_loop()

        return SessionInfo(
            name=state.name,
            endpoint=state.endpoint,
            variant=state.variant,
            accelerator=state.accelerator,
            shape=state.shape,
            runtime_url=state.runtime_url,
            status="running",
        )

    def list_sessions(self) -> list[SessionSummary]:
        """Reconcile local sessions.json against live server assignments."""
        client = self._get_client()
        server_assignments = client.list_assignments()
        server_by_endpoint = {item.endpoint: item for item in server_assignments}

        with self._lock:
            local_sessions = self.load_sessions()
            alive_sessions = {
                name: state
                for name, state in local_sessions.items()
                if state.endpoint in server_by_endpoint
            }
            for dead_name in set(local_sessions) - set(alive_sessions):
                dead_rt = self._runtimes.pop(dead_name, None)
                if dead_rt is not None:
                    with contextlib.suppress(Exception):
                        dead_rt.stop()
            if len(alive_sessions) != len(local_sessions):
                self.save_sessions(alive_sessions)

        tracked_endpoints = {state.endpoint for state in alive_sessions.values()}
        summaries = [
            SessionSummary(
                name=state.name,
                endpoint=state.endpoint,
                variant=state.variant,
                accelerator=state.accelerator,
                shape=state.shape,
                tracked=True,
                running=state.running,
            )
            for state in alive_sessions.values()
        ]
        for assignment in server_assignments:
            if assignment.endpoint in tracked_endpoints:
                continue
            summaries.append(
                SessionSummary(
                    name="?",
                    endpoint=assignment.endpoint,
                    variant=_assignment_variant_name(assignment.variant),
                    accelerator=assignment.accelerator,
                    shape=assignment.machine_shape.name,
                    tracked=False,
                    running=None,
                )
            )
        return summaries

    def session_status(self, name: str) -> SessionStatus:
        """Query live kernel state and stored metadata for a named session."""
        state = self.ensure_fresh_session(name)
        kernel_state = self.get_runtime(name).get_kernel_state()
        return SessionStatus(
            name=state.name,
            endpoint=state.endpoint,
            variant=state.variant,
            accelerator=state.accelerator,
            shape=state.shape,
            kernel_state=kernel_state,
            running=state.running,
            last_execution=state.last_execution,
            proxy_expires_at=state.proxy_expires_at,
            keep_alive_active=self._keep_alive_task is not None
            and not self._keep_alive_task.done(),
            last_keep_alive_at=self._last_keep_alive_at.get(state.endpoint),
        )

    def stop_session(self, name: str) -> dict[str, Any]:
        """Disconnect kernel channels, unassign the Colab VM, and remove local state."""
        with self._lock:
            sessions = self.load_sessions()
            matched = _find_local_session(sessions, name)
            if matched is not None:
                key, state = matched
                rt = self._runtimes.pop(key, None)
                endpoint = state.endpoint
            else:
                key = None
                rt = None
                endpoint = None
        if rt is not None:
            with contextlib.suppress(Exception):
                rt.stop()

        client = self._get_client()
        if endpoint is None:
            listed = _find_server_assignment(client, name)
            if listed is None:
                raise KeyError(f"Session '{name}' not found")
            endpoint = listed.endpoint

        client.unassign(endpoint)
        self._last_keep_alive_at.pop(endpoint, None)

        if key is not None:
            with self._lock:
                sessions = self.load_sessions()
                sessions.pop(key, None)
                self.save_sessions(sessions)

        return {"stopped": True, "session": key or name, "endpoint": endpoint}

    def restart_kernel(self, name: str) -> dict[str, Any]:
        """Restart the session's Jupyter kernel and persist any updated kernel_id."""
        rt = self.get_runtime(name)
        kernel_id = rt.restart()
        with self._lock:
            sessions = self.load_sessions()
            if name in sessions:
                sessions[name].kernel_id = kernel_id
                self.save_sessions(sessions)
        return {"restarted": True, "session": name, "kernel_id": kernel_id}

    async def _keep_alive_worker(self) -> None:
        while True:
            await asyncio.sleep(self.keep_alive_interval_s)
            sessions = self.load_sessions()
            if not sessions:
                continue
            try:
                client = self._get_client()
            except Exception as exc:
                logger.debug("Keep-alive skipped (auth/client error): %s", exc)
                continue
            for state in sessions.values():
                try:
                    await asyncio.to_thread(client.keep_alive, state.endpoint)
                    self._last_keep_alive_at[state.endpoint] = datetime.now(
                        timezone.utc
                    ).isoformat()
                except Exception as exc:
                    logger.debug("Keep-alive failed for %s: %s", state.endpoint, exc)

    def start_keep_alive_loop(self) -> None:
        """Start the background asyncio keep-alive task if not already running."""
        if self._keep_alive_task is not None and not self._keep_alive_task.done():
            return
        self._keep_alive_task = asyncio.create_task(self._keep_alive_worker())

    def stop_keep_alive_loop(self) -> None:
        """Cancel the background keep-alive task if running."""
        if self._keep_alive_task is not None:
            self._keep_alive_task.cancel()
            self._keep_alive_task = None

    async def __aenter__(self) -> SessionManager:
        self.start_keep_alive_loop()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        task = self._keep_alive_task
        self.stop_keep_alive_loop()
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        with self._lock:
            runtimes = list(self._runtimes.values())
            self._runtimes.clear()
        for rt in runtimes:
            with contextlib.suppress(Exception):
                rt.stop()
