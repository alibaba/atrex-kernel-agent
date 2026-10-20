"""Host-side construction helpers for the PPU fixed-slot timeline buffer."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Sequence


MAGIC = 0x0000314C54555050
ABI_MAJOR = 1
ABI_MINOR = 0
HEADER_STRUCT = struct.Struct("<Q4H10IQ")
RECORD_BYTES = 16
CLAIM_BYTES = 8
UINT32_MAX = 0xFFFFFFFF


def _dimensions(value: Sequence[int], field: str) -> tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError(f"{field} must contain x, y, z")
    dimensions = tuple(int(item) for item in value)
    if any(item < 1 or item > UINT32_MAX for item in dimensions):
        raise ValueError(f"{field} dimensions must fit positive uint32")
    return dimensions  # type: ignore[return-value]


# Explicit host/device allocation budget; callers may choose a different positive cap.
DEFAULT_MAX_ALLOCATION_BYTES = 256 * 1024 * 1024


def _validated_geometry(
    owner_count: int, records_per_owner: int
) -> tuple[int, int, int]:
    owners, per_owner = int(owner_count), int(records_per_owner)
    if not 1 <= owners <= UINT32_MAX or not 1 <= per_owner <= UINT32_MAX:
        raise ValueError("owner_count and records_per_owner must fit positive uint32")
    capacity = owners * per_owner
    if capacity > UINT32_MAX:
        raise ValueError("capacity does not fit the v1 header")
    return owners, per_owner, capacity


def make_header(
    *,
    owner_count: int,
    records_per_owner: int,
    grid: Sequence[int],
    block: Sequence[int],
    launch_id: int,
) -> bytes:
    """Pack the exact 64-byte ABI header with status cleared."""

    owners, per_owner, capacity = _validated_geometry(owner_count, records_per_owner)
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
    owners, _, capacity = _validated_geometry(owner_count, records_per_owner)
    return HEADER_STRUCT.size + capacity * RECORD_BYTES + owners * CLAIM_BYTES


def allocate_torch_buffer(
    *,
    owner_count: int,
    records_per_owner: int,
    grid: Sequence[int],
    block: Sequence[int],
    launch_id: int,
    device: object = "cuda",
    max_allocation_bytes: int = DEFAULT_MAX_ALLOCATION_BYTES,
):
    """Create an initialized one-dimensional ``torch.uint8`` PPU buffer."""

    header = make_header(
        owner_count=owner_count,
        records_per_owner=records_per_owner,
        grid=grid,
        block=block,
        launch_id=launch_id,
    )
    fields = HEADER_STRUCT.unpack(header)
    size = HEADER_STRUCT.size + fields[5] * RECORD_BYTES + fields[6] * CLAIM_BYTES
    if (
        not isinstance(max_allocation_bytes, int)
        or isinstance(max_allocation_bytes, bool)
        or max_allocation_bytes <= 0
        or size > max_allocation_bytes
    ):
        raise ValueError(
            f"timeline allocation {size} exceeds max_allocation_bytes={max_allocation_bytes}"
        )
    import torch

    buffer = torch.zeros(size, dtype=torch.uint8, device=device)
    buffer[: len(header)] = torch.tensor(
        tuple(header), dtype=torch.uint8, device=device
    )
    return buffer


def save_torch_buffer(buffer: object, destination: str | Path) -> Path:
    """Synchronize, copy, and save an allocated PPU timeline byte tensor."""

    import torch

    if (
        not isinstance(buffer, torch.Tensor)
        or buffer.dtype != torch.uint8
        or buffer.ndim != 1
    ):
        raise TypeError("expected a one-dimensional torch.uint8 timeline buffer")
    if buffer.is_cuda:
        torch.cuda.synchronize(buffer.device)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.detach().cpu().contiguous().numpy().tobytes())
    return path


def header_path() -> Path:
    return Path(__file__).with_name("ppu_timeline.cuh")


def instrument_source(
    source: str,
    *,
    source_name: str = "kernel.cu",
    enabled: bool = True,
    header: str | Path | None = None,
) -> str:
    """Include the PPU recorder before a JIT source and restore its coordinates."""

    escaped_name = source_name.replace("\\", "\\\\").replace('"', '\\"')
    selected_header = header_path() if header is None else Path(header)
    escaped_header = str(selected_header).replace("\\", "\\\\").replace('"', '\\"')
    enabled_define = "#define PPU_TIMELINE_ENABLED 1\n" if enabled else ""
    return (
        "#include <cuda_runtime.h>\n"
        f'{enabled_define}#include "{escaped_header}"\n'
        f'#line 1 "{escaped_name}"\n{source}'
    )
