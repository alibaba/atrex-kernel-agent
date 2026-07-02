# SM120 GDN Decode: cp.async + GLOBAL Cache Quick Reference (kernel-opt)

> This is an **optimization highlights quick reference**. See "Further Reading" below for the complete 18-version iteration journey + 12 pitfalls.


**Last updated**: 2026-06-30

## Trigger Conditions

You are writing a CuTeDSL kernel targeting **sm_120** (RTX PRO 5000/4000 Blackwell, RTX 50xx GeForce), and ncu shows:
- L2 throughput **>= 95%** (appears saturated)
- DRAM Throughput **< 50%** (actually far from saturated)
- L1/TEX Hit Rate **>= 50%**

→ **This is L2 false-saturation**, not a real bottleneck. The following 4-piece set can break through it.

## The 4-Piece Set (Apply in Order)

### 1. Unlock 16-byte cp.async Alignment

```python
mH0 = from_dlpack(h0.contiguous(), assumed_align=16)
mHt = from_dlpack(ht,                assumed_align=16)
```

PyTorch CUDA tensors are actually 256-byte aligned, but `from_dlpack` assumes conservative defaults.
Without this line → cp.async cp_size=128b will fail the compile-time verifier.

### 2. cp.async G2S with `LoadCacheMode.GLOBAL`

```python
from cutlass.cute.nvgpu import cpasync
cp_atom = cute.make_copy_atom(
    cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
    cutlass.Float32,
    num_bits_per_copy=128,   # 16B vec
)
```

The `GLOBAL` mode makes cp.async go directly from DRAM → SMEM, **bypassing L1**, eliminating the doubling of L2 traffic.

### 3. Thread Layout Satisfies 16B Alignment

```python
# Per-thread V slice = 4 × fp32 = 16 bytes contiguous
thr_layout = cute.make_layout((K_thr, V_thr), stride=(V_thr, 1))
val_layout = cute.make_layout((K_per_t, V_per_t), stride=(V_per_t, 1))
# REQUIRE: V_per_t * sizeof(dtype) == 16 (e.g. V_per_t=4 for fp32, =8 for bf16)
```

### 4. SMEM Staging + Latency Hiding

```python
# Issue cp.async, commit
cute.copy(tiled_copy_h0, thr_gH0, thr_sH0)
cute.arch.cp_async_commit_group()

# Window for compute that doesn't depend on h0:
#   - load q/k/v via autovec
#   - L2 norm (warp shuffle)
#   - exp(g), beta load

cute.arch.cp_async_wait_group(0)
cute.arch.barrier()

# Now read state from SMEM (cheap)
state[ki, vi] = sH0[k_off + ki, v_smem_base + vi]
```

## Measured Benefits (GDN decode, fp32 state, B=64, Pro5000)

| Version | wall-clock | DRAM throughput | Memory throughput |
|------|-----------|-----------------|-------------------|
| baseline (default ld.ca) | 389 μs | 34% | 411 GB/s |
| **+ 4-piece set** | **246 μs** | **87.7%** | **1.04 TB/s** = 100.8% memcpy ceiling |

Matches FLA Triton wall-clock (1.00× at B=64).

## Anti-Patterns Quick Reference (see pitfalls for details)

| Don't Try | Because |
|-------|------|
| `cute.arch.load(cop='cg')` element-wise load | Loses vectorization, 2.6× slower |
| `num_bits_per_copy=64` cp.async | Not supported, only 128b |
| 4-warp K-distributed | Cross-warp SMEM reduce eats occupancy gains |
| Hand-written mbarrier | Use `pipeline.PipelineTmaAsync`, hand-written easily deadlocks |
| Increasing BV to chase higher ncu Max BW % | Actually L1 reuse↑ → DRAM% ↓ |
| `cpasync.CopyOp()` as a generic atom | Abstract class, use concrete subclasses like `CopyG2SOp` |

## Further Reading

- **Complete optimization journey + 18-version data**:
  [`docs/ref-docs/nvidia/cutedsl/sm120/sm120-gdn-decode-fp32state-bf16qkv-optimization.md`](sm120-gdn-decode-fp32state-bf16qkv-optimization.md)
- **12 pitfalls explained in detail (trap → symptom → why → lesson)**:
  [`docs/pitfalls/nvidia/cutedsl/gdn-decode-pitfalls.md`](pitfalls/gdn-decode-pitfalls.md)
- **Production code**:
  [`reference-kernels/nvidia/blackwell-geforce/cutedsl/gdn_decode/sm120_gdn_fwd_T1_v13.py`](../../../../reference-kernels/nvidia/blackwell-geforce/cutedsl/gdn_decode/sm120_gdn_fwd_T1_v13.py)
- **Same kernel hopper bf16-state variant** (sister project):
  [`reference-kernels/nvidia/hopper/cutedsl/flashinfer/gdn_decode_*.py`](../../../../reference-kernels/nvidia/hopper/cutedsl/flashinfer/)


## Related

- [Stage 3 Closeout — Path-1 fused sigmoid·gate + NVFP4 quant on sm_120](sm120-fused-fa-epilogue-nvfp4-bf16-optimization.md)
- [CuTeDSL Gated DeltaNet Chunk Forward (bf16, Precomputed Neumann) on SM120](sm120-gdn-chunk-fwd-bf16-neumann-optimization.md)
- [CuteDSL GDN Decode (fp32 state, bf16 q/k/v) on sm_120 — Optimization Journey](sm120-gdn-decode-fp32state-bf16qkv-optimization.md)
- [SM120 INT32 MoE Data-Prep — Optimization Journey](sm120-moe-data-prep-optimization.md)
- [SM120 MoE Data-Prep — Quick Reference](sm120-moe-data-prep.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
