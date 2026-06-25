# CDNA3 (gfx942) ISA Instruction Patterns and Optimization Reference

**Last Updated**: 2026-03-06

---

## 1. Global Memory Instructions

### buffer_load Series

| Instruction | Width | Bytes | Throughput (Relative) |
|-------------|-------|-------|-----------------------|
| `buffer_load_dword` | 1 dword | 4B | 1x |
| `buffer_load_dwordx2` | 2 dwords | 8B | ~2x |
| `buffer_load_dwordx4` | 4 dwords | 16B | ~4x (optimal) |

- Up to 4 adjacent `buffer_load_dwordx4` instructions can be merged into a single clause, corresponding to a single data fabric transaction
- In the ideal case, coalesced access from the entire wavefront (64 threads) covers 64 × 16B = 1024B

### buffer_store Series

| Instruction | Width | Bytes |
|-------------|-------|-------|
| `buffer_store_dword` | 1 dword | 4B |
| `buffer_store_dwordx2` | 2 dwords | 8B |
| `buffer_store_dwordx4` | 4 dwords | 16B (optimal) |

### Scratch Instructions (Register Spill Indicator)

| Instruction | Meaning |
|-------------|---------|
| `scratch_load_dword[x2/x4]` | Load from scratch space (VGPR spill) |
| `scratch_store_dword[x2/x4]` | Write to scratch space (VGPR spill) |

**Presence of scratch instructions = VGPR usage must be optimized**

---

## 2. LDS (Shared Memory) Instructions

### ds_read Series

| Instruction | Width | Bytes | Bank Access Groups per Wavefront |
|-------------|-------|-------|----------------------------------|
| `ds_read_b32` | 32-bit | 4B | 1 bank per thread |
| `ds_read_b64` | 64-bit | 8B | 2 banks per thread |
| `ds_read_b128` | 128-bit | 16B | 4 banks per thread (optimal) |
| `ds_read2_b32` | 2×32-bit | 8B | 2 independent bank accesses |
| `ds_read2_b64` | 2×64-bit | 16B | 2 independent bank accesses |
| `ds_read2st64_b32` | 2×32-bit | 8B | 2 bank accesses with stride=64 |
| `ds_read2st64_b64` | 2×64-bit | 16B | 2 bank accesses with stride=64 |

### ds_write Series

| Instruction | Width | Bytes |
|-------------|-------|-------|
| `ds_write_b32` | 32-bit | 4B |
| `ds_write_b64` | 64-bit | 8B |
| `ds_write_b128` | 128-bit | 16B (optimal) |
| `ds_write2_b32` | 2×32-bit | 8B |
| `ds_write2_b64` | 2×64-bit | 16B |
| `ds_write2st64_b32` | 2×32-bit | 8B |
| `ds_write2st64_b64` | 2×64-bit | 16B |

### ds_bpermute

| Instruction | Latency | Description |
|-------------|---------|-------------|
| `ds_bpermute_b32` | ~50 cycles | Cross-lane data read (uses LDS hardware but does not write to LDS) |
| `ds_permute_b32` | ~50 cycles | Cross-lane data write |

### Bank Conflict Rules

CDNA3: 32 banks, 4B/bank
```
bank_id = (byte_address / 4) % 32
```

Multiple threads in the same wavefront accessing the same bank but different addresses in the same clock cycle → N-way bank conflict → latency × N

**ds_write_b128 grouping** (8 groups, 8 lanes each):
```
 0: lane 0-7
 1: lane 8-15
 2: lane 16-23
 3: lane 24-31
 4: lane 32-39
 5: lane 40-47
 6: lane 48-55
 7: lane 56-63
```
Bank conflicts are checked independently within each group.

**ds_read_b128 grouping** (more complex interleaved pattern):
```
 0: lane 0-3, 20-23
 1: lane 4-7, 16-19
 2: lane 8-11, 28-31
 3: lane 12-15, 24-27
 4: lane 32-35, 52-55
 5: lane 36-39, 48-51
 6: lane 40-43, 60-63
 7: lane 44-47, 56-59
```

---

## 3. MFMA Instructions

### CDNA3 MFMA Variants (gfx942)

| Instruction | Input Type | Shape (M×N×K) | Output Type | Cycles per Instruction |
|-------------|------------|---------------|-------------|------------------------|
| `v_mfma_f32_16x16x16_f16` | FP16 | 16×16×16 | FP32 | 16 |
| `v_mfma_f32_16x16x16_bf16` | BF16 | 16×16×16 | FP32 | 16 |
| `v_mfma_f32_32x32x8_f16` | FP16 | 32×32×8 | FP32 | 32 |
| `v_mfma_f32_32x32x8_bf16` | BF16 | 32×32×8 | FP32 | 32 |
| `v_mfma_f32_16x16x32_f8f6f4` | FP8 | 16×16×32 | FP32 | 16 |
| `v_mfma_f32_32x32x16_f8f6f4` | FP8 | 32×32×16 | FP32 | 32 |

### MFMA Latency and Throughput

- 16×16 variants: ~8 cycles latency, 1/cycle throughput (per SIMD)
- 32×32 variants: ~16 cycles latency, 1/2-cycle throughput (per SIMD)
- Uses AGPR as accumulator registers### MFMA Stall Diagnosis

If the Idle column for `v_mfma_*` instructions is high → waiting for operands to be ready → better prefetch / pipeline needed

---

## 4. Synchronization and Wait Instructions

| Instruction | Description | Performance Impact |
|------|------|---------|
| `s_waitcnt vmcnt(N)` | Allows the most recent N VMEM operations to pass through, stalls all prior operations until completion | N=0 is the strictest (wait for all to complete), N>0 allows async |
| `s_waitcnt lgkmcnt(N)` | Allows the most recent N LDS/GDS/SMEM operations to pass through, stalls all prior operations until completion | Same as above |
| `s_waitcnt expcnt(N)` | Allows the most recent N exports to pass through, stalls all prior operations until completion | Typically only in graphics pipeline |
| `s_barrier` | workgroup barrier | All waves synchronized |

### Key Optimization Points

- If `s_waitcnt vmcnt(0)` is immediately preceded by `buffer_load` → load-use distance is too short, needs to be stretched out
- Ideal: `buffer_load` → sufficient independent computation → `s_waitcnt vmcnt(0)` → use data

---

## 5. Scalar Instructions

| Instruction | Description |
|------|------|
| `s_load_dword[x2/x4/x8/x16]` | Load from constant memory pointed to by SGPR |
| `s_mov_b32/b64` | Scalar move |
| `s_add_u32`, `s_mul_i32` etc. | Scalar arithmetic |
| `s_cbranch_*` | Conditional branch |

Scalar instructions are not a focus for optimization (low overhead), but a large number of `s_cbranch` may indicate overly complex control flow.

---

## 6. Data Movement Instructions

| Instruction | Direction | Latency | Description |
|------|------|------|------|
| `v_accvgpr_write_b32` | VGPR → AGPR | ~4 cycles | MFMA requires AGPR input |
| `v_accvgpr_read_b32` | AGPR → VGPR | ~4 cycles | Reading MFMA accumulator results |
| `v_mov_b32` | VGPR → VGPR | ~4 cycles | Register move |

A large number of `v_accvgpr_read/write` may indicate:
- MFMA accumulators being frequently moved between AGPR and VGPR (block size too large)
- Approaching VGPR spill boundary

---

## 7. Instruction Pattern Recognition Quick Reference

### Healthy Patterns ✅

```asm
; load + MFMA pipeline
buffer_load_dwordx4 v[0:3], ...        ; 128-bit load
buffer_load_dwordx4 v[4:7], ...        ; 128-bit load (clause)
; ... independentcompute ...
s_waitcnt vmcnt(0)
ds_write_b128 v8, v[0:3], ...          ; 128-bit LDS write
ds_read_b128 v[12:15], v9, ...         ; 128-bit LDS read
v_mfma_f32_16x16x16_bf16 a[0:3], ...  ; MFMA
```

### Problematic Patterns ❌

```asm
; load ( dwordx4)
buffer_load_dword v0, ... ; 32-bit load -> 1/4
buffer_load_dword v1, ...

; scratch
scratch_store_dwordx4 off, v[0:3], ... ; VGPR
English description
scratch_load_dwordx4 v[0:3], off, ... ; VGPR

; ds_bpermute
ds_bpermute_b32 v0, v1, v2 ; lane shuffle
ds_bpermute_b32 v3, v4, v5 ; shuffle
; ... loop ...

; LDS access
ds_read_b32 v0, v1 ; 32-bit read -> 1/4
ds_read_b32 v2, v3 ; ds_read_b128
```

---

## Reference Documentation

- [CDNA3 ISA Manual](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [LLVM gfx942 Instruction Syntax](https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX940.html)
- [AMDGPU Kernel Optimization Guide](https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md)
- [LDS Bank Conflict Analysis](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html)
- [Machine-Readable ISA (XML)](https://gpuopen.com/machine-readable-isa/)
