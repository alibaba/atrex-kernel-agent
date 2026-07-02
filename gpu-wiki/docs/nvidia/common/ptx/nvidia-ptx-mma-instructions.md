# PTX MMA Instruction Evolution (NVIDIA General)


**Last updated**: 2026-06-30

## MMA Instruction Lineage

```
wmma (SM70+) → mma.sync (SM75+) → wgmma.mma_async (SM90+) → tcgen05.mma (SM100+)
  32 threads    32 threads            128 threads(warpgroup)       128 threads(warpgroup)
  High-level API Low-level precise control  Async+SMEM input   TMEM accumulator
```

---

## wmma — Warp Matrix Multiply-Accumulate (SM70+)

High-level API; the compiler manages fragment layout (opaque layout).

### Basic Operation

```
// Declare fragments (opaque layout, compiler-managed)
wmma.load.a.sync.aligned.row.m16n16k16.f16     {%r0,...,%r7}, [%sptr], %ldm;
wmma.load.b.sync.aligned.col.m16n16k16.f16     {%r0,...,%r7}, [%sptr], %ldm;
wmma.load.c.sync.aligned.row.m16n16k16.f32     {%r0,...,%r7}, [%sptr], %ldm;

// MMA computation
wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32
    {%d0,...,%d7}, {%a0,...,%a7}, {%b0,...,%b7}, {%c0,...,%c7};

// Store result
wmma.store.d.sync.aligned.row.m16n16k16.f32    [%ptr], {%d0,...,%d7}, %ldm;
```

### Supported Shapes and Types (SM70+)

| Shape | A/B Type | C/D Type | Minimum Architecture |
|------|---------|---------|---------|
| m16n16k16 | f16 | f16/f32 | SM70 |
| m32n8k16 | f16 | f16/f32 | SM70 |
| m8n32k16 | f16 | f16/f32 | SM70 |
| m16n16k16 | bf16 | f32 | SM80 |
| m16n16k16 | tf32 | f32 | SM80 |
| m16n16k8 | tf32 | f32 | SM80 |
| m16n16k16 | s8/u8 | s32 | SM72 |
| m8n8k128 | b1 | s32 | SM75 (XOR/AND popc) |
| m16n16k16 | s4/u4 | s32 | SM75 |
| m8n8k4 | f64 | f64 | SM80 |

### wmma Sub-byte Types

```
// 4-bit integer MMA
wmma.load.a.sync.aligned.row.m8n8k32.s4   {%r0}, [%ptr], %ldm;
wmma.mma.sync.aligned.row.col.m8n8k32.s32.s4
    {%d0, %d1}, {%a0}, {%b0}, {%c0, %c1};

// 1-bit MMA (SM75+)
wmma.mma.xor.popc.sync.aligned.row.col.m8n8k128.s32.b1
    {%d0, %d1}, {%a0}, {%b0}, {%c0, %c1};
```

---

## mma.sync — Low-level MMA (SM75+)

Directly maps to hardware instructions; fragment layout is fixed and documented.

### SM75 (Turing) — m8n8k16 / m16n8k8

```
// INT8 m8n8k16
mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32
    {%d0, %d1}, {%a0}, {%b0}, {%c0, %c1};

// FP16 m16n8k8
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
    {%d0,%d1,%d2,%d3}, {%a0,%a1}, {%b0}, {%c0,%c1,%c2,%c3};
```

### SM80 (Ampere)

Adds m16n8k16 (primary shape) and more data types:

```
// FP16 m16n8k16
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%d0,%d1,%d2,%d3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%c0,%c1,%c2,%c3};

// BF16 m16n8k16
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
    {%d0,%d1,%d2,%d3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%c0,%c1,%c2,%c3};

// TF32 m16n8k8
mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    {%d0,%d1,%d2,%d3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%c0,%c1,%c2,%c3};

// FP64 m8n8k4
mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64
    {%d0,%d1}, {%a0}, {%b0}, {%c0,%c1};

// INT8 m16n8k32
mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
    {%d0,%d1,%d2,%d3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%c0,%c1,%c2,%c3};
```

### SM89 (Ada Lovelace) — FP8

```
// FP8 m16n8k32
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
    {%d0,%d1,%d2,%d3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%c0,%c1,%c2,%c3};

mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32
    {%d0,%d1,%d2,%d3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%c0,%c1,%c2,%c3};
// A B usedifferent FP8
```### Fragment Layout (m16n8k16 f16)

In each warp (32 threads), threads hold fixed positions of the matrix:

```
A matrix (16x16, row-major):
  4 registers/thread, each register 2 f16
  Thread T holds part of row [T%16, T%16+8]

B matrix (8x16, col-major):
  2 registers/thread

D/C matrix (16x8):
  4 registers/thread, each register 2 f32
```

---

## wgmma.mma_async — Warpgroup MMA (SM90+)

128 threads (4 warps = 1 warpgroup) collaborate to execute asynchronous MMA. The A operand can come from SMEM (no need to load into registers).

### Basic Usage

```
// fence + MMA + commit + wait four-step workflow

// 1. Fence: separate different MMA groups
wgmma.fence.sync.aligned;

// 2. MMA instruction (async launch)
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
    {%d0,...,%d63},                    // D: 64 f32 registers
    %desc_a,                           // A: SMEM descriptor
    %desc_b,                           // B: SMEM descriptor
    1,                                 // scale_d: 1=accumulate, 0=overwrite
    -1, 1, 0, 1;                       // scale_a, scale_b, trans_a, trans_b

// 3. Commit: submit to async group
wgmma.commit_group.sync.aligned;

// 4. Wait: wait for group completion
wgmma.wait_group.sync.aligned  0;     // Wait for all groups to complete
```

### Supported Shapes and Types

| M | N Range | K | A/B Type | D Type |
|---|--------|---|---------|--------|
| 64 | 8-256 (stride 8) | 16 | f16 | f16/f32 |
| 64 | 8-256 | 16 | bf16 | f32 |
| 64 | 8-256 | 8 | tf32 | f32 |
| 64 | 8-256 | 32 | e4m3/e5m2 | f16/f32 |
| 64 | 8-256 | 32 | s8/u8 | s32 |
| 64 | 8-256 | 64 | e4m3/e5m2 | f16/f32 |

**N Dimension Flexibility:** N can be any multiple of 8 from 8 to 256, allowing fine-grained tuning of tile sizes.

### A Operand Source

```
// A from SMEM (descriptor) — saves registers
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
    {%d0,...}, %desc_a, %desc_b, ...;

// A from registers — for epilogue fusion or special data flow
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
    {%d0,...}, {%a0,...,%a7}, %desc_b, ...;
```

### SMEM Descriptor Format

```
// 64-bit descriptor encoding:
// [13:0]   start_address     — 14-bit SMEM address (128B aligned)
// [29:16]  leading_byte_offset — continuous dimension stride
// [45:32]  stride_byte_offset  — non-continuous dimension stride
// [48:46]  base_offset        — base address offset
// [51:49]  swizzle mode       — 0-4 (none/32B/64B/128B)
```

### Warpgroup Fence Semantics

```
wgmma.fence.sync.aligned;
// - Ensures registers written by previous MMA group are visible to subsequent instructions
// - Must be called after modifying A/B SMEM operands and before next MMA
// - Similar to acquire-release semantics but specific to WGMMA

wgmma.commit_group.sync.aligned;
// - Groups all pending wgmma instructions into a group
// - Similar to cp.async.commit_group

wgmma.wait_group.sync.aligned  N;
// - Wait until ≤N groups pending
// - N=0 means all completed
```

---

## tcgen05.mma — Fifth-Generation Tensor Core (SM100+)

The Blackwell architecture introduces TMEM (Tensor Memory) as accumulator storage, further decoupling computation and data movement.

### Basic Usage

```
// TMEM
tcgen05.alloc %tmem_addr, %num_columns;

// Fence
tcgen05.fence::before_thread_sync;
bar.sync 0;

// MMA
tcgen05.mma.cta_group::1.kind::f16
    [%tmem_addr], %desc_a, %desc_b, %idesc, %enable, %scale_d;

// Commit + Wait
tcgen05.commit.cta_group::1;
tcgen05.wait.cta_group::1  0;

// TMEM
tcgen05.ld [%tmem_addr], {%r0,...};

// TMEM
tcgen05.dealloc %tmem_addr, %num_columns;
```

### Supported Data Types

| Kind | A/B Type | Accumulator | Characteristics |
|------|---------|--------|------|
| `f16` | f16, bf16 | f32 | Standard mixed precision |
| `tf32` | tf32 | f32 | High precision |
| `f8f6f4` | e4m3, e5m2, e2m3, e3m2, e2m1 | f32 | Narrow-precision MMA |
| `i8` | s8, u8 | s32 | Integer MMA |
| `mxf8` | MX FP8 | f32 | Block-scaled |
| `mxf4` | MX FP4 | f32 | Block-scaled |
| `mxf4nvf4` | MX FP4 NV | f32 | NV custom format |### CTA Group

```
// 1 CTA
tcgen05.mma.cta_group::1.kind::f16   [%tmem], %desc_a, %desc_b, ...;

// 2 CTA ( CTA compute tile)
tcgen05.mma.cta_group::2.kind::f16   [%tmem], %desc_a, %desc_b, ...;
```

### TMEM Operations

```
// English comment
tcgen05.alloc              %addr, %num_cols;

// English comment
tcgen05.dealloc            %addr, %num_cols;

// TMEM loadregister
tcgen05.ld                 [%tmem_addr+offset], {%r0,...};
tcgen05.ld.16x256b [%tmem_addr+offset], {%r0,...}; // 16 column x 256 bytes

// store TMEM
tcgen05.st                 [%tmem_addr+offset], {%r0,...};
tcgen05.st.16x32b [%tmem_addr+offset], {%r0,...}; // 16 column x 32 bytes

// TMEM -> SMEM directstore
tcgen05.st.shared::cluster [%smem_ptr], [%tmem_addr], %num_cols;
```

---

## MMA Sparse Variants

### Structured Sparsity (SM80+)

2:4 structured sparsity: only 2 out of every 4 elements are non-zero, doubling throughput.

```
// mma.sync
mma.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32
    {%d0,...,%d3}, {%a0,...,%a3}, {%b0,...,%b3}, {%c0,...,%c3};
// A matrix 2:4 , metadata register

// wgmma(SM90+)
wgmma.mma_async.sp.sync.aligned.m64n128k32.f32.f16.f16
    {%d0,...}, %desc_a, %desc_b, %meta, %selector, ...;
// %meta: modedata
// %selector: data
```

---

## MMA Instruction Selection Guide

| Scenario | Recommended Instruction | Rationale |
|------|---------|------|
| Simple GEMM (portable) | `wmma` | Compiler-managed fragments, simplest |
| Performance-critical (SM75-SM89) | `mma.sync` | Precise layout control, hand-tunable |
| Hopper GEMM | `wgmma.mma_async` | Async + SMEM input, maximum throughput |
| Blackwell GEMM | `tcgen05.mma` | TMEM accumulator, latest architecture |
| Inference (sparse models) | `mma.sp` / `wgmma.sp` | 2:4 sparse acceleration |

### Throughput Evolution

| Architecture | Instruction | M×N×K | FP16 TFLOPS (typical) |
|------|------|-------|-------------------|
| SM70 (Volta) | `wmma` | 16×16×16 | ~125 |
| SM75 (Turing) | `mma.sync` | 16×8×8 | ~130 |
| SM80 (Ampere) | `mma.sync` | 16×8×16 | ~312 |
| SM89 (Ada) | `mma.sync` | 16×8×32 (FP8) | ~660 |
| SM90 (Hopper) | `wgmma` | 64×N×16 | ~990 |
| SM100 (Blackwell) | `tcgen05` | — | ~2500+ |

---

## ldmatrix Evolution
```
// SM90+: stmatrix (reverse operation)
stmatrix.sync.aligned.m8n8.x4.shared.b16  [%sptr], {%r0,...,%r3};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16  [%sptr], {%r0,...,%r3};

// SM100+: m16n8 variants
ldmatrix.sync.aligned.m16n8.x1.shared.b8  {%r0}, [%sptr];
ldmatrix.sync.aligned.m16n8.x2.shared.b8  {%r0,%r1}, [%sptr];
ldmatrix.sync.aligned.m16n8.x4.shared.b8  {%r0,...,%r3}, [%sptr];
```
## Related

- **Prerequisites**: [PTX Programming Model](ptx-programming-model.md) → [PTX Instruction Set](ptx-instruction-set.md) → [PTX Synchronization & Asynchrony](nvidia-ptx-sync-and-async.md)
- **AMD Counterpart**: [AMD MFMA Matrix Core Programming Guide](../../../amd/common/amd-mfma-matrix-cores.md) — AMD matrix core instructions
- **CuTeDSL Wrapper**: [CuTeDSL SM90 Special Features](../../hopper/cutedsl/hopper-cutedsl-sm90.md) — High-level DSL wrapper for wgmma
- **⚠️ FP8 Differences**: NVIDIA uses OCP standard `.e4m3`/`.e5m2`, AMD CDNA3 uses non-standard FNUZ format (different bias)
