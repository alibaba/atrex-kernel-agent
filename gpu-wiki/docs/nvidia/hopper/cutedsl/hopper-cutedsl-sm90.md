# CuTeDSL SM90 (Hopper) Special Features


**Last updated**: 2026-06-30

## Warpgroup MMA (WGMMA)

Hopper introduces **warpgroup-level** MMA (128 threads = 4 warps collaboration), implemented via WGMMA instructions.

### MmaF16BF16Op — FP16/BF16 WGMMA

```python
from cutlass.cute.nvgpu.warpgroup import MmaF16BF16Op, OperandSource, OperandMajorMode

mma_op = MmaF16BF16Op(
 ab_dtype=cutlass.Float16, # A B type(f16 or bf16)
 acc_dtype=cutlass.Float32, # type
    instruction_shape=(64, 128, 16),    # MxNxK
 a_src=OperandSource.smem, # A registeror SMEM
 a_major_mode=OperandMajorMode.K, # A majorness
 b_major_mode=OperandMajorMode.K, # B majorness
)
```

### MmaF8Op — FP8 WGMMA

```python
from cutlass.cute.nvgpu.warpgroup import MmaF8Op

mma_op = MmaF8Op(
 a_dtype=cutlass.Float8E4M3, # A FP8 type
 b_dtype=cutlass.Float8E5M2, # B different A
    acc_dtype=cutlass.Float32,
    instruction_shape=(64, 128, 32),
    a_src=OperandSource.smem,
    a_major_mode=OperandMajorMode.K,
    b_major_mode=OperandMajorMode.K,
)
```

### WGMMA Workflow

```python
from cutlass.cute.nvgpu.warpgroup import fence, commit_group, wait_group

fence # 1. wgmma.fence, MMA
# ... execute MMA ...
commit_group # 2. pending WGMMA group
wait_group(0) # 3. wait committed group complete(0=complete)
```

---

## SmemLayoutAtomKind

SM90's SMEM layout atom is the "minimal compact layout that satisfies TMA and UMMA legality," used recoveryto construct the SMEM layout for operands A/B via blocked product.

| Kind | Meaning |
|------|---------|
| `MN_INTER` | MN-major, no swizzle |
| `MN_SW32` | MN-major, 32B swizzle |
| `MN_SW64` | MN-major, 64B swizzle |
| `MN_SW128` | MN-major, 128B swizzle |
| `K_INTER` | K-major, no swizzle |
| `K_SW32` | K-major, 32B swizzle |
| `K_SW64` | K-major, 64B swizzle |
| `K_SW128` | K-major, 128B swizzle |

```python
from cutlass.cute.nvgpu.warpgroup import make_smem_layout_atom, SmemLayoutAtomKind

atom = make_smem_layout_atom(SmemLayoutAtomKind.MN_SW128, cutlass.Float16)
```

---

## TMA (Tensor Memory Accelerator)

TMA is a hardware data movement unit in the Hopper architecture. A single TMA instruction copies an **entire multi-dimensional tile**, eliminating element-wise address computation.

### TMA Core Concepts

1. **TMA Descriptor**: A packed multi-dimensional tensor description (base ptr, dtype, shape, stride, swizzle). Must be created on the host side.
2. **TMA Instruction**: Only requires the descriptor pointer + SMEM pointer + coordinates to initiate a copy.
3. **TMA Tensor**: A coordinate tensor built using ArithTuple iterators and Basis Element Stride. It can be tiled/sliced/partitioned just like a regular CuTe tensor.

### Basis Element Stride

The stride of a TMA tensor is not an ordinary integer but a **basis element** — a unit element that marks the coordinate dimension:

```python
E(0) # (1, 0, ...) - 0 unit
E(1) # (0, 1, ...) - 1 unit
5*E(1) # (0, 5, ...) - 1 5
```

The inner product of coordinate `(i, j)` and stride `(E(0), E(1))` equals `(i, j)`, directly yielding the TMA coordinate.

### Copy Operation Classes

```python
from cutlass.cute.nvgpu.cpasync import (
 CopyG2SOp, # bulk cp.async(GMEM -> SMEM)
    CopyBulkTensorTileG2SOp,       # TMA bulk load（GMEM → SMEM）
    CopyBulkTensorTileG2SMulticastOp,  # TMA multicast load
    CopyBulkTensorTileS2GOp,       # TMA store（SMEM → GMEM）
    CopyReduceBulkTensorTileS2GOp, # TMA reduction store
    CopyDsmemStoreOp,              # DSMEM async store
)
```### make_tiled_tma_atom

Core factory function for building TMA copy atoms:

```python
from cutlass.cute.nvgpu.cpasync import make_tiled_tma_atom

copy_atom, tma_tensor = make_tiled_tma_atom(
 op=CopyBulkTensorTileG2SOp, # TMA type
 gmem_tensor=mA, # globalmemory tensor
 smem_layout_=smem_layout, # SMEM layout( stage mode)
    cta_tiler=cta_tiler,              # CTA tiler
 num_multicast=1, # multicast
 internal_type=None, # optionaltype
)
```

Returns:
- `CopyAtom` — TMA copy atom
- `TMA Tensor` — Coordinate tensor that maps logical coordinates to TMA coordinates using basis strides

### tma_partition

```python
from cutlass.cute.nvgpu.cpasync import tma_partition

smem_partitioned, gmem_partitioned = tma_partition(
    atom=copy_atom,
    cta_coord=cta_coord,
    cta_layout=cta_layout,
    smem_tensor=sA,
    gmem_tensor=tma_tensor,
)
```

### TMA Multicast

Broadcast data across multiple CTAs within a cluster:

```python
from cutlass.cute.nvgpu.cpasync import create_tma_multicast_mask

mask = create_tma_multicast_mask(
    cta_layout_vmnk=cta_layout,
    cta_coord_vmnk=cta_coord,
 mcast_mode=1, # multicast tensor mode
)
```

### TMA Descriptor Management

```python
from cutlass.cute.nvgpu.cpasync import (
 prefetch_descriptor, # TMA descriptor cache
 copy_tensormap, # copy descriptor SMEM
 update_tma_descriptor, # descriptor(base, shape, stride)
    fence_tma_desc_acquire,  # descriptor acquire barrier
    fence_tma_desc_release,  # descriptor release barrier
)
```

### make_tiled_tma_atom_A / _B

SM90/SM100 common TMA atom factory, accounting for M/K or N/K projection of TiledMMA:

```python
from cutlass.cute.nvgpu import make_tiled_tma_atom_A, make_tiled_tma_atom_B

copy_atom_a, tma_a = make_tiled_tma_atom_A(
    op=CopyBulkTensorTileG2SMulticastOp(),
    gmem_tensor=mA,
    smem_layout=smem_layout_a,
    mma_tiler_mnk=(tile_m, tile_n, tile_k),
    tiled_mma=tiled_mma,
    cluster_shape_vmnk=cluster_shape,
)
# A multicast N mode, B multicast M mode
```

---

## PipelineTmaAsync — SM90 Mainloop Pipeline

```python
from cutlass.pipeline import PipelineTmaAsync, CooperativeGroup, Agent

pipeline = PipelineTmaAsync.create(
    num_stages=4,
    producer_group=CooperativeGroup(Agent.ThreadBlock, size=32),
    consumer_group=CooperativeGroup(Agent.ThreadBlock, size=128),
    tx_count=tile_bytes,
    barrier_storage=smem_mbar_ptr,
    cta_layout_vmnk=cluster_layout,
    mcast_mode_mn=(1, 1),
)

producer = pipeline.make_producer()
consumer = pipeline.make_consumer()
```

Key characteristics:
- Producer `commit` is a noop — TMA instructions automatically update transaction count
- Consumer `release` conditionally sends empty buffer signal
- Internally computes multicast signaling threads automatically

---

## SM90 Utility Functions

### SMEM Layout Construction

```python
from cutlass.utils.sm90 import make_smem_layout_a, make_smem_layout_b, make_smem_layout_epi

# A tensor SMEM layout(4 ):
# 1. MMA tiler shape
# 2. majorness/dtype SMEM layout atom
# 3. tile atom MMA tile shape
# 4. by num_stages
smem_a = make_smem_layout_a(
    a_layout=LayoutEnum.ColumnMajor,
    mma_tiler_mnk=(128, 128, 32),
    a_dtype=cutlass.Float16,
    num_stages=4,
)

smem_b = make_smem_layout_b(b_layout, mma_tiler_mnk, b_dtype, num_stages)
smem_epi = make_smem_layout_epi(epi_dtype, epi_layout, epi_tile, epi_stage)
```### Store Operation Selection

```python
from cutlass.utils.sm90 import get_smem_store_op

store_atom = get_smem_store_op(
    layout_d=LayoutEnum.ColumnMajor,
    elem_ty_d=cutlass.Float16,
    elem_ty_acc=cutlass.Float32,
)
# returns SmemStoreMatrix or SimtSyncCopy
```

### Epilogue Tile Computation

```python
from cutlass.utils.sm90 import compute_tile_shape_or_override

epi_tile = compute_tile_shape_or_override(
    tile_shape_mnk=(128, 128, 32),
    element_type=cutlass.Float16,
    is_cooperative=False,
 epi_tile_override=None, # optionalmanual
)
```

---

## Warp Specialization

### Producer-Consumer Warp Group Coordination

Hopper divides the threadblock into:

- **Producer warp group** (typically 1 warp): performs global → shared loading via TMA
- **Consumer warp group(s)** (typically 2-3 warp groups): performs WGMMA computation

Producer waits for consumer to release an empty buffer → TMA load → marks buffer full
Consumer waits for a full buffer → WGMMA compute → releases buffer

### Ping-Pong Design

Two consumer warp groups alternate working on **different output tiles**:

```
Consumer A: [Compute tile 0] [Epilogue 0] [Compute tile 2] [Epilogue 2]
Consumer B:        [Compute tile 1] [Epilogue 1] [Compute tile 3]
```

Consumer A's epilogue overlaps with Consumer B's compute → maximizes tensor core utilization.

The producer uses an **Ordered Sequence Barrier** to alternately fill buffers for the two consumers.

---

## Warp-Level ldmatrix/stmatrix

```python
from cutlass.cute.nvgpu.warp import (
    LdMatrix8x8x16bOp,     # 8x8 ldmatrix (.m8n8)
    LdMatrix16x8x8bOp,     # 16x8 ldmatrix with transpose
 LdMatrix16x16x8bOp, # 16x16 ldmatrix (.m16n16), 4b/6b/8b
    StMatrix8x8x16bOp,     # 8x8 stmatrix (.m8n8)
    StMatrix16x8x8bOp,     # 16x8 stmatrix (.m16n8)
)

ld_op = LdMatrix8x8x16bOp(transpose=False, num_matrices=1)
```

`LdMatrix16x8x8bOp` has no direct PTX counterpart—it lowers to `.m16n16` ldmatrix with additional address/value rearrangement, used for vectorized Ampere-style 8x8 matrix thread-value layouts.

---

## Warp-Level MMA (Non-Warpgroup)

```python
from cutlass.cute.nvgpu.warp import MmaF16BF16Op as WarpMmaF16BF16Op

# Warp MMA(32 ), warpgroup
warp_mma = WarpMmaF16BF16Op(
    ab_dtype=cutlass.Float16,
    acc_dtype=cutlass.Float32,
    shape_mnk=(16, 8, 16),
)
```

---

## SM90 Key Hardware Parameters

| Parameter | Value |
|------|------|
| Warp size | 32 threads |
| Warpgroup size | 128 threads (4 warps) |
| SMEM capacity | 228 KB/SM (configurable) |
| Registers/SM | 65,536 |
| Registers/thread | Up to 255 |
| Cluster support | Up to 16 CTAs |
| TMA support | 1-5 dimensional tensors |
| WGMMA support | FP16, BF16, FP8 (E4M3, E5M2) |


## Related

- [CUTLASS/CuTe Core Concepts and Layout Algebra](../../common/cutedsl/cutlass-cute-fundamentals.md)
- [nvidia/hopper/cutedsl](README.md)
