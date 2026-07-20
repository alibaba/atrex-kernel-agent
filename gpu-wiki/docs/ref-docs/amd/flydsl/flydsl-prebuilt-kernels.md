# FlyDSL Pre-built Kernel Library Reference

Applicability: backend: flydsl; hardware: amd; topic: reference

FlyDSL provides a set of production-grade pre-built GPU kernels covering common operators such as Normalization, Softmax, GEMM, and MoE, implemented using the `@flyc.kernel`/`@flyc.jit` API.

---

## 1. Kernel Overview

| Kernel | Build Function | Data Types | Key Features |
|--------|---------|----------|---------|
| **LayerNorm** | `build_layernorm_module(M, N, dtype)` | f32, f16, bf16 | Two-phase vectorized normalization |
| **RMSNorm** | `build_rmsnorm_module(M, N, dtype)` | f32, f16, bf16 | LDS-cached 3-stage pipeline |
| **Softmax** | `build_softmax_module(M, N, dtype)` | f32, f16, bf16 | Online softmax, adaptive block size |
| **GEMM** | `compile_preshuffle_gemm_a8(...)` | fp8, int8, int4, fp16, bf16, fp4 | Preshuffle B, ping-pong LDS, MFMA 16×16 |
| **Blockscale GEMM** | `compile_blockscale_preshuffle_gemm(...)` | fp8 + per-block scaling | Block-scale quantized GEMM |
| **MoE GEMM** | `moe_gemm_2stage` | fp8, f16, bf16, int8, int4 | Gate/Up + Reduce two-phase |
| **MoE Blockscale** | `moe_blockscale_2stage` | fp8 + per-block scaling | Block-scale MoE |
| **Mixed MoE** | `mixed_moe_gemm_2stage` | Mixed precision | Different precision per expert |
| **Flash Attention** | `flash_attn_func` | f16, bf16 | Flash Attention |
| **Flash Attention GQA D=256** | `flash_attn_func_gqa_d256` | bf16 | GQA Flash Attention for head_dim=256, MI355X optimized. 10-15% faster than CK. |
| **Paged Attention** | `pa_decode_fp8` | fp8 | FP8 Paged Attention decode |
| **Fused RoPE** | `fused_rope_cache_kernel` | f16, bf16 | RoPE positional encoding fusion |

---

## 2. Normalization Kernel

### 2.1 LayerNorm

Computes `LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta`.

```python
from kernels.layernorm_kernel import build_layernorm_module

executor = build_layernorm_module(M=32768, N=8192, dtype_str="bf16")
```

**Configuration Parameters**:

| Constant | Value | Description |
|------|-----|------|
| `BLOCK_THREADS` | 256 | Threads per block |
| `WARP_SIZE` | 64 | AMD wavefront size |
| `VEC_WIDTH` | 8 | Vector load/store width |
| `EPS` | 1e-5 | Numerical stability epsilon |
| `USE_NONTEMPORAL` | True | Non-temporal store |

**Algorithm**:
- **Two-phase normalization**: Pass 1 computes mean/variance, Pass 2 performs affine transform
- **Fast path**: When `N == BLOCK_THREADS * VEC_WIDTH * 4` (e.g., N=8192), full register-resident computation
- **bf16 handling**: gfx942 uses software RNE packing; gfx950+ uses hardware `cvt_pk_bf16_f32`
- **Warp reduction**: XOR-shuffle intra-wave reduction (shift 32,16,8,4,2,1), LDS cross-wave sync

### 2.2 RMSNorm

Computes `RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma`.

```python
from kernels.rmsnorm_kernel import build_rmsnorm_module

executor = build_rmsnorm_module(M=32768, N=8192, dtype_str="bf16")
```

**Algorithm (3-stage + LDS cache)**:
1. **Pass 0**: Global → LDS row cache (single global read, vectorized)
2. **Pass 1**: Compute sum of squares from LDS row cache
3. **Pass 2**: Normalize + gamma multiplication + store, with Gamma software prefetch pipeline

---

## 3. Softmax Kernel

Row-wise softmax: `softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))`.

```python
from kernels.softmax_kernel import build_softmax_module

executor = build_softmax_module(M=32768, N=8192, dtype_str="bf16")
```

**Configuration**:
- `BLOCK_SIZE`: `min(256, next_power_of_2(N))`, minimum 32
- `VEC_WIDTH`: 8

**Algorithm (6-stage)**:
1. **Load Data**: Vectorized global load into register buffer
2. **Local Max**: Per-thread vector reduction (`maxnumf`)
3. **Global Max**: Block-wide shuffle reduction
4. **Local Exp + Sum**: `exp2(x * log2(e))` approximation + partial sum accumulation
5. **Global Sum**: Block-wide sum reduction
6. **Normalize + Store**: Divide by sum, type conversion, vectorized store

## 4. GEMM Kernel

### 4.1 Preshuffle GEMM

MFMA 16×16 GEMM with B matrix pre-shuffled: `C[M,N] = A[M,K] @ B[N,K]^T`.

```python
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8

launch_fn = compile_preshuffle_gemm_a8(
    M=16, N=5120, K=8192,
    tile_m=16, tile_n=128, tile_k=256,
    in_dtype="fp8",
    lds_stage=2,
    use_cshuffle_epilog=False,
)

# English note
launch_fn(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, M_val, N_val, stream)
```

**Parameters**:

| Parameter | Type | Description |
|------|------|------|
| `M, N, K` | int | GEMM dimensions. M/N = 0 indicates dynamic |
| `tile_m, tile_n, tile_k` | int | Block tile size |
| `in_dtype` | str | `"fp8"`, `"int8"`, `"int4"`, `"fp16"`, `"bf16"`, `"fp4"` |
| `lds_stage` | int | `2` = ping-pong LDS, `1` = single buffer |
| `use_cshuffle_epilog` | bool | CK-style LDS CShuffle epilogue |
| `waves_per_eu` | int | Occupancy hint (1-4) |
| `use_async_copy` | bool | Async DMA A tile transfer |

**Constraint**: `tile_k * elem_bytes` must be divisible by 64 (K64-byte micro-step).

### 4.2 Core Optimization Techniques

#### Ping-pong LDS Buffering (`lds_stage=2`)
Two LDS buffers are used for A tiles. A0 prefetch across tiles overlaps with VMEM reads and LDS reads.

#### XOR16 Swizzle
Byte-level XOR address remapping eliminates LDS bank conflicts:
```python
col_swizzled = col_bytes ^ ((row % k_blocks16) << 4)
```

#### B Matrix Preshuffle Layout
Shape: `(N/16, K/64, KLane=4, NLane=16, kpack_bytes)`, pre-shuffles B for coalesced MFMA access.

#### K64-byte Micro-step
Each pipeline step issues 2× K32 MFMA operations.

#### CShuffle Epilogue
CK-style: writes C tiles to LDS (row-major), remaps threads, and performs half2 packing via `ds_bpermute`.

#### Non-temporal Stores
Output writes are marked as non-temporal, with write-through behavior.

---

## 5. MoE Kernel

### 5.1 Standard MoE GEMM

Two-stage MoE: Stage 1 (Gate GEMM + SiLU activation + Up GEMM), Stage 2 (Down GEMM + reduction).

```python
# fp8, f16, bf16, int8, int4
from kernels.moe_gemm_2stage import moe_gemm_2stage
```

**SiLU Fast Path**: Uses `v_exp2` + `v_rcp` instead of `exp` + `div`, achieving ~4× speedup.

### 5.2 Blockscale MoE

MoE with per-block FP8 scaling:

```python
from kernels.moe_blockscale_2stage import moe_blockscale_2stage
```

### 5.3 Mixed MoE

Different experts use different precisions:

```python
from kernels.mixed_moe_gemm_2stage import mixed_moe_gemm_2stage
```

---

## 6. Shared Utilities

### 6.1 Reduction Utilities (`kernels/reduce.py`)

| Function | Description |
|------|------|
| `reduce_vec_max(vec, VEC_WIDTH)` | Vector reduction to find maximum (`maxnumf`) |
| `reduce_vec_sum(vec, VEC_WIDTH)` | Vector reduction summation |
| `make_block_reduce(tid, BLOCK_SIZE)` | Block-wide reduction: intra-wave XOR shuffle → LDS cross-wave sync |
| `make_block_reduce_add(tid)` | Addition block reduction (single-wave fast path) |
| `make_block_reduce_add2(tid)` | Dual independent scalar reduction |

**Reduction Pattern**:
1. Intra-wave: XOR shuffle, shift 32, 16, 8, 4, 2, 1 (wave64)
2. Lane 0 writes per-wave partial results to LDS
3. Barrier
4. Wave 0 reduces `NUM_WAVES` partial results from LDS

### 6.2 MFMA Epilogue (`kernels/mfma_epilogues.py`)

| Function | Description |
|------|------|
| `default_epilog(...)` | Standard row iterator |
| `c_shuffle_epilog(...)` | CK-style LDS CShuffle: write LDS → barrier → remap threads → half2 store |
| `mfma_epilog(use_cshuffle, ...)` | Dispatcher |

### 6.3 Preshuffle Pipeline (`kernels/mfma_preshuffle_pipeline.py`)

| Function | Description |
|------|------|
| `make_preshuffle_b_layout(...)` | Build B preshuffle layout |
| `load_b_pack_k32(...)` | Load B pack (K32 MFMA micro-step) |
| `buffer_copy_gmem16_dwordx4(...)` | 16-byte global buffer load |
| `lds_store_16b_xor16(...)` | LDS 16B store with XOR16 swizzle |
| `lds_load_pack_k32(...)` | Load A pack from LDS (K32 micro-step) |
| `swizzle_xor16(...)` | XOR swizzle to avoid LDS bank conflicts |

## 7. MFMA Instruction Reference

| Instruction | Data Type | M×N×K | Architecture |
|-------------|-----------|-------|--------------|
| `mfma_f32_16x16x16f16` | FP16 | 16×16×16 | GFX942+ |
| `mfma_f32_16x16x32_fp8_fp8` | FP8 | 16×16×32 | GFX942+ |
| `mfma_i32_16x16x32_i8` | INT8 | 16×16×32 | GFX942+ |
| `mfma_f32_32x32x8f16` | FP16 | 32×32×8 | GFX942+ |
| `mfma_f32_16x16x16bf16_1k` | BF16 1K | 16×16×16 | GFX942+ |
| `mfma_scale_x128` | MXFP4 | 16×16×128 | GFX950 |

---

## 8. Instruction Scheduling Control

```python
from flydsl.expr import rocdl

rocdl.sched_mfma(cnt) # wait cnt MFMA complete
rocdl.sched_vmem(cnt) # wait cnt VMEM readcomplete
rocdl.sched_dsrd(cnt) # wait cnt LDS readcomplete
rocdl.sched_dswr(cnt) # wait cnt LDS writecomplete
```

Used arg0 manual control of instruction scheduling, ensuring overlap of computation and data movement.

---

## 9. Kernel Selection Decision Tree

```
requires？
│
├── Normalization
│ ├── requires bias (beta)？ -> LayerNorm (layernorm_kernel.py)
│ └── none bias？ -> RMSNorm (rmsnorm_kernel.py)
│
├── Softmax
│ └── row softmax -> Softmax (softmax_kernel.py)
│
├── matrixmultiplication (GEMM)
│ ├── standard GEMM -> compile_preshuffle_gemm_a8
│ │ : fp8 / int8 / int4(W4A8) / fp16 / bf16 / fp4
│   └── Block-scale GEMM → compile_blockscale_preshuffle_gemm()
│
├── MoE (Mixture of Experts)
│ ├── standard MoE -> moe_gemm_2stage
│   ├── Blockscale MoE → moe_blockscale_2stage
│   └── Mixed MoE → mixed_moe_gemm_2stage
│
├── Attention
│   ├── Flash Attention → flash_attn_func
│   └── Paged Attention → pa_decode_fp8
│
└── block
 ├── Warp/Block reduction -> reduce.py
    ├── MFMA epilogue    → mfma_epilogues.py
 └── Preshuffle -> mfma_preshuffle_pipeline.py
```

---

## 10. Performance Data

### Fused MoE (tokens=16384, E=384, topk=8, MI300X)

| Comparison | BF16 | W4A16 |
|------------|------|-------|
| vs Triton comparison baseline | 1.39× | 3.22× |
| vs PyTorch | 13.8× | 13.4× |

### End-to-End (RTP-LLM, Kimi-K2.5 MoE)

| Metric | Improvement |
|--------|-------------|
| TPOT | -69.2% |
| Throughput | +162.4% |

---

## Related Docs

- [FlyDSL Programming Guide](flydsl-programming-guide.md) — Compilation pipeline, core API
- [FlyDSL Layout Algebra](flydsl-layout-algebra.md) — Detailed layout algebra reference
- [Fused MoE Optimization (FlyDSL)](../../../kernel-opt/amd/flydsl/gfx942/cdna3-fused-moe-flydsl.md) — MI300X MoE optimization case study
- [MoE 2-Stage Fusion](../../../kernel-opt/amd/common/hands-on/moe-2stage-fusion.md) — Expert GEMM + SiLU fusion pattern
- [Preshuffle B Layout](../../../kernel-opt/amd/common/hands-on/preshuffle-b-layout.md) — Pre-shuffle weight matrix technique
- [AMD MFMA Matrix Core Programming Guide](../common/amd-mfma-matrix-cores.md) — MFMA instruction details
