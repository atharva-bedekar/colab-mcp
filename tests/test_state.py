from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest


@pytest.mark.asyncio
async def test_session_manager_create_accessors_and_stop(tmp_path):
    from colab_mcp.models import Assignment, RuntimeProxyInfo, Shape, Variant
    from colab_mcp.state import SessionManager

    fake_client = MagicMock()
    fake_client.assign.return_value = Assignment(
        endpoint="ep-1",
        accelerator="T4",
        variant=Variant.GPU,
        machineShape=Shape.STANDARD,
        runtimeProxyInfo=RuntimeProxyInfo(
            token="tok-1",
            tokenExpiresInSeconds=3600,
            url="https://proxy.colab.google",
        ),
    )
    fake_runtime = MagicMock()
    fake_runtime.get_kernel_state.return_value = "idle"
    fake_runtime.restart.return_value = "k-new"

    mgr = SessionManager(
        state_path=tmp_path / "sessions.json",
        auth_manager=MagicMock(),
        client_factory=lambda _: fake_client,
        runtime_factory=lambda *a, **kw: fake_runtime,
    )

    info = mgr.create_session(name="trainer", gpu="T4")
    assert info.name == "trainer"
    assert info.endpoint == "ep-1"
    fake_client.keep_alive.assert_called_once_with("ep-1")

    assert mgr.get_runtime("trainer") is fake_runtime
    assert mgr.get_runtime("trainer") is fake_runtime  # cached
    assert mgr.get_contents_client("trainer").proxy_token == "tok-1"

    status = mgr.session_status("trainer")
    assert status.kernel_state == "idle"

    restarted = mgr.restart_kernel("trainer")
    assert restarted == {"restarted": True, "session": "trainer", "kernel_id": "k-new"}
    assert mgr.load_sessions()["trainer"].kernel_id == "k-new"

    stopped = mgr.stop_session("trainer")
    assert stopped == {"stopped": True, "session": "trainer", "endpoint": "ep-1"}
    fake_runtime.stop.assert_called_once()
    fake_client.unassign.assert_called_once_with("ep-1")
    assert "trainer" not in mgr.load_sessions()
    assert "ep-1" not in mgr._last_keep_alive_at


@pytest.mark.asyncio
async def test_token_refresh_list_reconciliation_and_keep_alive(tmp_path):
    import asyncio
    import uuid

    from colab_mcp.models import (
        Assignment,
        AssignmentVariant,
        ListedAssignment,
        RuntimeProxyInfo,
        SessionState,
        Shape,
        Variant,
    )
    from colab_mcp.state import SessionManager

    fake_client = MagicMock()
    fake_client.get_assignment.return_value = Assignment(
        endpoint="ep-1",
        accelerator="T4",
        variant=Variant.GPU,
        machineShape=Shape.STANDARD,
        runtimeProxyInfo=RuntimeProxyInfo(
            token="tok-refreshed",
            tokenExpiresInSeconds=3600,
            url="https://proxy-new.colab.google",
        ),
    )
    fake_client.list_assignments.return_value = [
        ListedAssignment(endpoint="ep-1", accelerator="T4", variant=AssignmentVariant.GPU),
        ListedAssignment(endpoint="ep-untracked", accelerator="NONE", variant=AssignmentVariant.DEFAULT),
    ]

    old_runtime = MagicMock()
    new_runtime = MagicMock()
    runtimes = [new_runtime]

    mgr = SessionManager(
        state_path=tmp_path / "sessions.json",
        auth_manager=MagicMock(),
        client_factory=lambda _: fake_client,
        runtime_factory=lambda *a, **kw: runtimes.pop(0),
        keep_alive_interval_s=0.02,
    )

    soon = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
    later = (datetime.now(timezone.utc) + timedelta(seconds=3600)).isoformat()
    mgr.save_sessions(
        {
            "active": SessionState(
                name="active",
                notebook_hash=str(uuid.uuid4()),
                endpoint="ep-1",
                variant="GPU",
                accelerator="T4",
                shape="STANDARD",
                runtime_url="https://proxy-old.colab.google",
                proxy_token="tok-old",
                proxy_expires_at=soon,
            ),
            "dead": SessionState(
                name="dead",
                notebook_hash=str(uuid.uuid4()),
                endpoint="ep-dead",
                variant="DEFAULT",
                accelerator="NONE",
                shape="STANDARD",
                runtime_url="https://dead.colab.google",
                proxy_token="tok-dead",
                proxy_expires_at=later,
            ),
        }
    )

    # Seed old cached runtime to verify invalidation on token/url change
    mgr._runtimes["active"] = old_runtime
    rt = mgr.get_runtime("active")
    assert rt is new_runtime
    old_runtime.stop.assert_called_once()
    assert mgr.load_sessions()["active"].proxy_token == "tok-refreshed"

    summaries = mgr.list_sessions()
    assert [s.name for s in summaries] == ["active", "?"]
    assert "dead" not in mgr.load_sessions()

    async with mgr:
        await asyncio.sleep(0.06)
    assert fake_client.keep_alive.call_count >= 1
    new_runtime.stop.assert_called_once()


@pytest.mark.asyncio
async def test_adopt_and_stop_untracked_server_session(tmp_path):
    from colab_mcp.models import (
        AssignmentVariant,
        ListedAssignment,
        RuntimeProxyInfo,
        Shape,
    )
    from colab_mcp.state import SessionManager

    fake_client = MagicMock()
    fake_client.list_assignments.return_value = [
        ListedAssignment(
            endpoint="m-s-orphan-1",
            accelerator="NONE",
            variant=AssignmentVariant.DEFAULT,
            machineShape=Shape.STANDARD,
            runtimeProxyInfo=RuntimeProxyInfo(
                token="tok-orphan",
                tokenExpiresInSeconds=3600,
                url="https://orphan.colab.google",
            ),
        ),
        ListedAssignment(
            endpoint="m-s-orphan-2",
            accelerator="NONE",
            variant=AssignmentVariant.DEFAULT,
            machineShape=Shape.STANDARD,
            runtimeProxyInfo=None,
        ),
    ]
    orphan_runtime = MagicMock()
    mgr = SessionManager(
        state_path=tmp_path / "sessions.json",
        auth_manager=MagicMock(),
        client_factory=lambda _: fake_client,
        runtime_factory=lambda *a, **kw: orphan_runtime,
    )

    # Auto-adopt untracked server session with runtimeProxyInfo by endpoint
    rt = mgr.get_runtime("m-s-orphan-1")
    assert rt is orphan_runtime
    adopted = mgr.load_sessions()["m-s-orphan-1"]
    assert adopted.endpoint == "m-s-orphan-1"
    assert adopted.variant == "DEFAULT"
    assert adopted.proxy_token == "tok-orphan"

    # Stop untracked server session directly by endpoint even without local state
    stopped = mgr.stop_session("m-s-orphan-2")
    assert stopped == {"stopped": True, "session": "m-s-orphan-2", "endpoint": "m-s-orphan-2"}
    fake_client.unassign.assert_called_with("m-s-orphan-2")


def test_session_status_keep_alive_inactive_and_last_keep_alive_set(tmp_path):
    from colab_mcp.models import Assignment, RuntimeProxyInfo, Shape, Variant
    from colab_mcp.state import SessionManager

    fake_client = MagicMock()
    fake_client.assign.return_value = Assignment(
        endpoint="ep-1",
        accelerator="T4",
        variant=Variant.GPU,
        machineShape=Shape.STANDARD,
        runtimeProxyInfo=RuntimeProxyInfo(
            token="tok-1",
            tokenExpiresInSeconds=3600,
            url="https://proxy.colab.google",
        ),
    )
    fake_runtime = MagicMock()
    fake_runtime.get_kernel_state.return_value = "idle"

    mgr = SessionManager(
        state_path=tmp_path / "sessions.json",
        auth_manager=MagicMock(),
        client_factory=lambda _: fake_client,
        runtime_factory=lambda *a, **kw: fake_runtime,
    )

    mgr.create_session(name="trainer", gpu="T4")
    status = mgr.session_status("trainer")
    assert status.keep_alive_active is False
    assert status.last_keep_alive_at is not None


@pytest.mark.asyncio
async def test_session_status_keep_alive_active_inside_loop(tmp_path):
    import asyncio
    import uuid

    from colab_mcp.models import SessionState
    from colab_mcp.state import SessionManager

    fake_client = MagicMock()
    fake_runtime = MagicMock()
    fake_runtime.get_kernel_state.return_value = "idle"

    mgr = SessionManager(
        state_path=tmp_path / "sessions.json",
        auth_manager=MagicMock(),
        client_factory=lambda _: fake_client,
        runtime_factory=lambda *a, **kw: fake_runtime,
        keep_alive_interval_s=0.02,
    )
    later = (datetime.now(timezone.utc) + timedelta(seconds=3600)).isoformat()
    mgr.save_sessions(
        {
            "trainer": SessionState(
                name="trainer",
                notebook_hash=str(uuid.uuid4()),
                endpoint="ep-1",
                variant="GPU",
                accelerator="T4",
                shape="STANDARD",
                runtime_url="https://proxy.colab.google",
                proxy_token="tok-1",
                proxy_expires_at=later,
            )
        }
    )

    async with mgr:
        await asyncio.sleep(0.06)
        status = mgr.session_status("trainer")
        assert status.keep_alive_active is True
        assert status.last_keep_alive_at is not None
