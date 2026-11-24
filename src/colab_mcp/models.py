"""Shared domain enums, exceptions, Pydantic models, and pure helpers."""

from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AuthProvider(str, Enum):
    OAUTH2 = "oauth2"
    ADC = "adc"


class Variant(str, Enum):
    DEFAULT = "DEFAULT"
    GPU = "GPU"
    TPU = "TPU"


class AssignmentVariant(int, Enum):
    DEFAULT = 0
    GPU = 1
    TPU = 2


class Accelerator(str, Enum):
    NONE = "NONE"
    G4 = "G4"
    T4 = "T4"
    L4 = "L4"
    A100 = "A100"
    H100 = "H100"
    V5E1 = "V5E1"
    V6E1 = "V6E1"


class Shape(int, Enum):
    STANDARD = 0
    HIGH_RAM = 1


_SINGLE_SHAPE_ACCELERATORS = frozenset({Accelerator.L4, Accelerator.V5E1, Accelerator.V6E1})
_GPU_ACCELERATORS = frozenset(
    {Accelerator.G4, Accelerator.T4, Accelerator.L4, Accelerator.A100, Accelerator.H100}
)
_TPU_ACCELERATORS = frozenset({Accelerator.V5E1, Accelerator.V6E1})


class ColabError(Exception):
    """Base error for Colab MCP operations."""


class ColabRequestError(ColabError):
    """HTTP or control-plane error returned by a Colab endpoint."""

    def __init__(self, message: str = "", status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class TooManyAssignmentsError(ColabRequestError):
    """Raised when Colab rejects assignment due to quota/session limit (HTTP 412)."""


class UnauthenticatedError(ColabError):
    """Raised when no valid OAuth2 or ADC credentials are available."""


class _ColabBaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class RuntimeProxyInfo(_ColabBaseModel):
    token: str
    token_expires_in_seconds: int = Field(default=3600, alias="tokenExpiresInSeconds")
    url: str

    def compute_expires_at(self, now: datetime | None = None) -> datetime:
        base = now or datetime.now(timezone.utc)
        return base + timedelta(seconds=self.token_expires_in_seconds)


class GetAssignmentTokenResponse(_ColabBaseModel):
    token: str
    variant: AssignmentVariant | Variant | int | str = Variant.DEFAULT
    accelerator: str = Field(default="NONE", alias="acc")


class Assignment(_ColabBaseModel):
    endpoint: str
    accelerator: str = "NONE"
    variant: AssignmentVariant | Variant | int | str = Variant.DEFAULT
    machine_shape: Shape = Field(default=Shape.STANDARD, alias="machineShape")
    runtime_proxy_info: RuntimeProxyInfo = Field(alias="runtimeProxyInfo")


class PostAssignmentResponse(Assignment):
    pass


class ListedAssignment(_ColabBaseModel):
    endpoint: str
    accelerator: str = "NONE"
    variant: AssignmentVariant | int = AssignmentVariant.DEFAULT
    machine_shape: Shape = Field(default=Shape.STANDARD, alias="machineShape")
    runtime_proxy_info: RuntimeProxyInfo | None = Field(default=None, alias="runtimeProxyInfo")


class SessionState(_ColabBaseModel):
    name: str
    notebook_hash: str
    endpoint: str
    variant: str
    accelerator: str
    shape: str
    runtime_url: str
    proxy_token: str
    proxy_expires_at: str
    kernel_id: str | None = None
    running: str | None = None
    last_execution: tuple[str, str | None, str] | None = None


class AuthStatusResponse(_ColabBaseModel):
    authenticated: bool
    provider: str
    token_path: str | None = None
    account: str | None = None


class AuthStartResponse(_ColabBaseModel):
    auth_url: str
    instructions: str


class AuthFinishResponse(_ColabBaseModel):
    authenticated: bool
    token_path: str


class SessionInfo(_ColabBaseModel):
    name: str
    endpoint: str
    variant: str
    accelerator: str
    shape: str
    runtime_url: str
    status: str = "running"


class SessionSummary(_ColabBaseModel):
    name: str
    endpoint: str
    variant: str
    accelerator: str
    shape: str
    tracked: bool = True
    running: str | None = None


class SessionStatus(_ColabBaseModel):
    name: str
    endpoint: str
    variant: str
    accelerator: str
    shape: str
    kernel_state: str
    running: str | None = None
    last_execution: tuple[str, str | None, str] | None = None
    proxy_expires_at: str
    keep_alive_active: bool = False
    last_keep_alive_at: str | None = None


class RemoteFileEntry(_ColabBaseModel):
    name: str
    path: str
    type: str
    size: int | None = None
    last_modified: str | None = None


class ExecutionResult(_ColabBaseModel):
    status: str
    outputs: list[dict[str, Any]] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    duration_s: float = 0.0


class FileExecutionResult(_ColabBaseModel):
    file_path: str
    status: str
    cells_executed: int = 0
    outputs: list[dict[str, Any]] = Field(default_factory=list)
    output_notebook_path: str | None = None
    duration_s: float = 0.0


def strip_xssi_prefix(text: str) -> str:
    """Remove Google's leading XSSI protection prefix if present."""
    for prefix in (")]}'\r\n", ")]}'\n", ")]}'"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def uuid_to_web_safe_base64(u: uuid.UUID) -> str:
    """Encode a UUID into Colab's 44-character web-safe notebook hash (nbh) format."""
    uuid_str = str(u)
    return uuid_str.replace("-", "_") + "." * (44 - len(uuid_str))


def resolve_runtime_options(
    gpu: str | None = None,
    tpu: str | None = None,
    *,
    high_mem: bool = False,
) -> tuple[Variant, Accelerator, Shape | None]:
    """Validate and map user-facing runtime flags to Colab assignment parameters."""
    if gpu is not None and tpu is not None:
        raise ValueError("Cannot specify both GPU and TPU")

    if gpu is not None:
        accel = Accelerator(gpu.upper())
        if accel not in _GPU_ACCELERATORS:
            raise ValueError(f"Unsupported GPU accelerator: {gpu}")
        variant = Variant.GPU
    elif tpu is not None:
        accel = Accelerator(tpu.upper())
        if accel not in _TPU_ACCELERATORS:
            raise ValueError(f"Unsupported TPU accelerator: {tpu}")
        variant = Variant.TPU
    else:
        accel = Accelerator.NONE
        variant = Variant.DEFAULT

    if accel in _SINGLE_SHAPE_ACCELERATORS:
        shape: Shape | None = None
    else:
        shape = Shape.HIGH_RAM if high_mem else Shape.STANDARD

    return variant, accel, shape


def parse_notebook_code_cells(nb_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read an .ipynb v4 JSON file and extract code cells with their cell indices."""
    nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells: list[dict[str, Any]] = []
    for idx, cell in enumerate(nb_data.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        raw_source = cell.get("source", "")
        source = "".join(raw_source) if isinstance(raw_source, list) else str(raw_source)
        code_cells.append({"index": idx, "source": source, "cell": cell})
    return nb_data, code_cells


def write_notebook_with_outputs(
    nb_data: dict[str, Any],
    cell_outputs: dict[int, list[dict[str, Any]]],
    out_path: Path,
) -> None:
    """Write a copy of notebook JSON with updated code cell outputs."""
    updated = copy.deepcopy(nb_data)
    cells = updated.get("cells", [])
    for idx, outputs in cell_outputs.items():
        if 0 <= idx < len(cells):
            cells[idx]["outputs"] = outputs
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
