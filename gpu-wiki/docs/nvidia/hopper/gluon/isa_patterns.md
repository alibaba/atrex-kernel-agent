# Hopper (sm_90) SASS Instruction Patterns and Optimization Reference

**Last updated**: 2026-03-18

---

## 1. Global Memory Instructions

### LDG (Global Load) Series

| SASS Instruction | Width | Bytes | Throughput (Relative) |
|----------|------|------|---------------|
| `LDG.E.32` | 32-bit | 4B | 1x |
| `LDG.E.64` | 64-bit | 8B | ~2x |
| `LDG.E.128` | 128-bit | 16B | ~4x (optimal) |
| `LDG.E.U16` | 16-bit | 2B | 0.5x |

- A coalesced `LDG.E.128` from a warp (32 threads) accesses 32 × 16B = 512B
- Hopper supports 128-byte sector-granularity coalescing
- Uncoalesced accesses are split into multiple sector requests, severely reducing bandwidth utilization

### STG (Global Store) Series

| SASS Instruction | Width | Bytes |
|----------|------|------|
| `STG.E.32` | 32-bit | 4B |
| `STG.E.64` | 64-bit | 8B |
| `STG.E.128` | 128-bit | 16B (optimal) |

### LDGSTS (CP_ASYNC: Global → Shared)

| SASS Instruction | Meaning | Description |
|----------|------|------|
| `LDGSTS.E.128` | CP_ASYNC DMA 128-bit | Global → Shared, bypasses registers |
| `LDGSTS.E.64` | CP_ASYNC DMA 64-bit | Global → Shared |

**`LDGSTS` is a performance-critical instruction on Hopper**: Corresponds to `async_copy_global_to_shared()`, where data is DMA'd directly from HBM to shared memory without going through registers. It is **50%+ faster** than the two-step transfer of `LDG` + `STS`.

> Seeing `LDG` + `STS` instead of `LDGSTS` = async_copy not being used → must fix

### Local Memory Instructions (Register Spill Indicator)

| SASS Instruction | Meaning |
|----------|------|
| `STL` | Store to Local Memory (register spill) |
| `LDL` | Load from Local Memory (register spill) |

**STL/LDL present = register spill to DRAM, must optimize**

---

## 2. Shared Memory Instructions

### STS (Shared Store) Series

| SASS Instruction | Width | Bytes |
|----------|------|------|
| `STS.32` | 32-bit | 4B |
| `STS.64` | 64-bit | 8B |
| `STS.128` | 128-bit | 16B (optimal) |

### LDS (Shared Load) Series

| SASS Instruction | Width | Bytes |
|----------|------|------|
| `LDS.32` | 32-bit | 4B |
| `LDS.64` | 64-bit | 8B |
| `LDS.128` | 128-bit | 16B (optimal) |
| `LDS.U.32` | 32-bit uniform | 4B (scalar broadcast) |

### LDSM (Load Shared to Matrix Register)

| SASS Instruction | Meaning | Description |
|----------|------|------|
| `LDSM.16.M88.2` | Load smem to matrix register | wgmma data preparation |
| `LDSM.16.M88.4` | Load smem to matrix register (4x) | wgmma data preparation |

> `LDSM` is the data loading instruction for Tensor Cores, which loads data arranged in a specific layout from shared memory into matrix registers in preparation for `HMMA`/`WGMMA`.

### Bank Conflict Rules

Hopper: **32 banks, 4B/bank** (same as AMD CDNA3)

```
bank_id = (byte_address / 4) % 32
```

In the same clock cycle, when multiple threads in the same warp access the same bank but different addresses → N-way bank conflict → latency × N

**Hopper warp = 32 threads** (AMD = 64 threads):
- 32 threads accessing 32 banks → one bank per thread → **zero conflicts (ideal)**
- If the address pattern causes multiple threads to hit the same bank → conflict

**Swizzle to eliminate conflicts**:
- `NVMMASharedLayout` has built-in swizzle modes, controlled by the `swizzle_byte_width` parameter
- `SwizzledSharedLayout` controls swizzling via `vec`, `perPhase`, `maxPhase` parameters

---

## 3. Tensor Core Instructions

### WGMMA (Warp Group Matrix Multiply-Accumulate)

| SASS Instruction | Input Type | Shape | Description |
|----------|---------|------|------|
| `WGMMA.MMA_ASYNC.SYNC.ALIGNED.M64NxK16.F32.BF16.BF16` | BF16 | 64×N×16 | N=8..256 |
| `WGMMA.MMA_ASYNC.SYNC.ALIGNED.M64NxK16.F32.F16.F16` | FP16 | 64×N×16 | N=8..256 |
| `WGMMA.MMA_ASYNC.SYNC.ALIGNED.M64NxK32.F32.E4M3.E4M3` | FP8 | 64×N×32 | N=8..256 |

### HMMA (Older Tensor Core)

| SASS Instruction | Description |
|----------|------|
| `HMMA.16816.F32` | 16×8×16 half-precision matrix multiply |

### Key Tensor Core Characteristics

- **wgmma is asynchronous**: Launched without blocking the warp, must be used with `WGMMA_FENCE` + `WGMMA_COMMIT_GROUP` + `WGMMA_WAIT` in coordination
- **Operands reside in shared memory**: wgmma reads A and B matrices directly from smem (via `NVMMASharedLayout`), with the accumulator in registers
- **Fundamental difference from AMD MFMA**: AMD MFMA operands are in registers (loaded from smem via `DotOperandLayout` before execution)

### wgmma Related Synchronization Instructions

| SASS Instruction | Corresponding Gluon API | Description |
|----------|----------------|------|
| `WGMMA.FENCE` | `fence_async_shared()` | Ensures smem writes are completed |
| `WGMMA.COMMIT_GROUP` | (internal) | Commits a set of wgmma |
| `WGMMA.WAIT` | `warpgroup_mma_wait()` | Waits for wgmma to complete |

---

## 4. Synchronization and Wait Instructions

| SASS Instruction | Description | Performance Impact |
|----------|------|---------|
| `BAR.SYNC` | Thread block barrier | All warps synchronize (full block-level) |
| `BAR.ARRIVE` | Barrier arrive (non-blocking) | Arrives at barrier without waiting |
| `BAR.WAIT` | Wait barrier | Waits for all threads to arrive |
| `DEPBAR.LE` | Dependency barrier | Waits for async operations to complete |
| `FENCE.PROXY.ASYNC.SHARED` | Shared memory fence | Ensures smem writes are visible |

### CP_ASYNC Synchronization

| SASS Instruction | Corresponding Gluon API | Description |
|----------|----------------|------|
| `LDGSTS` | `async_copy_global_to_shared()` | Initiates async DMA |
| `DEPBAR.LE SB0, <count>` | `async_copy.commit_group()` + `wait_group()` | Waits for async copy to complete |

### Key Optimization Points

- Excessive `BAR.SYNC` usage → Check for unnecessary barriers
- The count parameter of `DEPBAR.LE` controls how many outstanding async_copy groups are allowed → Affects pipeline depth
- Ideal pattern: `LDGSTS` → independent computation → `DEPBAR` → use data

---

## 5. Scalar and Control Instructions

| SASS Instruction | Description |
|----------|------|
| `MOV` | Register move |
| `IADD3` | Integer three-operand addition |
| `IMAD` | Integer multiply-add |
| `ISETP` | Integer compare + set predicate |
| `BRA` | Unconditional branch |
| `@P0 BRA` | Conditional branch |
| `EXIT` | Exit kernel |
| `NOP` | No-op (scheduling padding) |

A large number of `BRA` may indicate overly complex control flow. A large number of `NOP` indicates the scheduler needs to insert bubbles (insufficient latency hiding).

---

## 6. Floating-Point Instructions

| SASS Instruction | Description | Precision |
|----------|------|------|
| `FADD` | Floating-point addition | FP32 |
| `FMUL` | Floating-point multiplication | FP32 |
| `FFMA` | Floating-point multiply-add | FP32 |
| `HADD2` | Half-precision dual-element addition | FP16/BF16 |
| `HMUL2` | Half-precision dual-element multiplication | FP16/BF16 |
| `HFMA2` | Half-precision dual-element multiply-add | FP16/BF16 |
| `MUFU.EX2` | Special function (exp2) | FP32 |
| `MUFU.RCP` | Special function (reciprocal) | FP32 |
| `MUFU.RSQ` | Special function (rsqrt) | FP32 |

---

## 7. Instruction Pattern Recognition Reference

### Healthy Patterns ✅

```sass
; load + async copy + wgmma pipeline
LDGSTS.E.128 [smem_addr], [global_addr]  ; CP_ASYNC: global → shared (128-bit)
LDGSTS.E.128 [smem_addr+16], [global_addr+16]
DEPBAR.LE SB0, 0x0 ; wait async copy complete
FENCE.PROXY.ASYNC.SHARED                 ; shared memory fence
WGMMA.MMA_ASYNC.SYNC... ; wgmma matrixmultiplication
WGMMA.WAIT ; wait wgmma complete

; store
STG.E.128 [global_addr], reg            ; 128-bit global store
```

### Problematic Patterns ❌

```sass
; load ( LDG.E.128 or LDGSTS.E.128)
LDG.E.32 R0, [global_addr] ; 32-bit load -> 1/4
LDG.E.32 R1, [global_addr+4]

; use async_copy ( LDGSTS)
LDG.E.128 R0, [global_addr] ; load register
STS.128 [smem_addr], R0 ; store shared memory
; -> LDGSTS.E.128 [smem_addr], [global_addr]

; register
STL [local_addr], R0                     ; store to local memory (DRAM!)
English description
LDL R0, [local_addr]                     ; load from local memory (DRAM!)

; shared memory access
LDS.32 R0, [smem_addr] ; 32-bit smem load -> 1/4
; -> LDS.128
```

---

## Reference Documentation

- [NVIDIA CUDA ISA (PTX/SASS)](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)
- [Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA C++ Programming Guide - Warp Level Primitives](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)


## Related

- [Hopper (sm_90) General ISA Optimization Checklist](common_optimizations.md)
- [Fused Attention (Prefill / Paged Attention) Optimization Guide](fused_attention.md)
- [Chunk Linear Attention / Recurrent State Update Optimization Guide](linear_attention.md)
- [Standard GEMM / Batched GEMM Optimization Guide](matmul.md)
- [Gluon Kernel Performance Optimization Guide (NVIDIA Hopper)](optimization-guide.md)
- [CDNA3 (gfx942) ISA Instruction Patterns and Optimization Reference](../../../amd/gluon/gfx942/isa_patterns.md)
- [Document Relationship Diagram](../../../RELATIONS.md)
