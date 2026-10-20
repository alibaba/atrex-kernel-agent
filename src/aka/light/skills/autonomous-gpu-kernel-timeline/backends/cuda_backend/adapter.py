"""Host-side construction helpers for the ATREX timeline buffer."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Sequence


MAGIC = 0x31544C5845525441
ABI_MAJOR = 1
ABI_MINOR = 0
HEADER_STRUCT = struct.Struct("<Q4H10IQ")
RECORD_BYTES = 16
CLAIM_BYTES = 8


def _dimensions(value: Sequence[int], field: str) -> tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError(f"{field} must contain x, y, z")
    dimensions = tuple(int(item) for item in value)
    if any(item < 1 or item > 0xFFFFFFFF for item in dimensions):
        raise ValueError(f"{field} dimensions must fit positive uint32")
    return dimensions  # type: ignore[return-value]


def make_header(
    *,
    owner_count: int,
    records_per_owner: int,
    grid: Sequence[int],
    block: Sequence[int],
    launch_id: int,
) -> bytes:
    """Pack the exact 64-byte ABI header with status cleared."""

    owners = int(owner_count)
    per_owner = int(records_per_owner)
    if owners < 1 or per_owner < 1:
        raise ValueError("owner_count and records_per_owner must be positive")
    capacity = owners * per_owner
    if capacity > 0xFFFFFFFF:
        raise ValueError("capacity does not fit the v1 header")
    gx, gy, gz = _dimensions(grid, "grid")
    bx, by, bz = _dimensions(block, "block")
    launch = int(launch_id)
    if launch < 0 or launch > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("launch_id must fit uint64")
    return HEADER_STRUCT.pack(
        MAGIC,
        ABI_MAJOR,
        ABI_MINOR,
        HEADER_STRUCT.size,
        RECORD_BYTES,
        capacity,
        owners,
        per_owner,
        0,
        gx,
        gy,
        gz,
        bx,
        by,
        bz,
        launch,
    )


def allocation_bytes(owner_count: int, records_per_owner: int) -> int:
    capacity = int(owner_count) * int(records_per_owner)
    if capacity < 1:
        raise ValueError("timeline allocation must contain at least one record")
    return HEADER_STRUCT.size + capacity * RECORD_BYTES + int(owner_count) * CLAIM_BYTES


def allocate_torch_buffer(
    *,
    owner_count: int,
    records_per_owner: int,
    grid: Sequence[int],
    block: Sequence[int],
    launch_id: int,
    device: object = "cuda",
):
    """Create a zeroed one-dimensional ``torch.uint8`` buffer on the requested device."""

    import torch

    header = make_header(
        owner_count=owner_count,
        records_per_owner=records_per_owner,
        grid=grid,
        block=block,
        launch_id=launch_id,
    )
    host = torch.zeros(allocation_bytes(owner_count, records_per_owner), dtype=torch.uint8)
    host[: len(header)] = torch.tensor(tuple(header), dtype=torch.uint8)
    return host.to(device=device)


def save_torch_buffer(buffer: object, destination: str | Path) -> Path:
    """Synchronize, copy, and save an allocated timeline byte tensor."""

    import torch

    if not isinstance(buffer, torch.Tensor) or buffer.dtype != torch.uint8 or buffer.ndim != 1:
        raise TypeError("expected a one-dimensional torch.uint8 timeline buffer")
    if buffer.is_cuda:
        torch.cuda.synchronize(buffer.device)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.detach().cpu().contiguous().numpy().tobytes())
    return path


def header_path() -> Path:
    return Path(__file__).with_name("atrex_timeline.cuh")


def embed_header(source: str, *, source_name: str = "kernel.cu") -> str:
    """Prepend the backend header to embedded NVRTC source and restore source coordinates."""

    escaped_name = source_name.replace("\\", "\\\\").replace('"', '\\"')
    header = header_path().read_text(encoding="utf-8")
    return f'{header}\n#line 1 "{escaped_name}"\n{source}'
