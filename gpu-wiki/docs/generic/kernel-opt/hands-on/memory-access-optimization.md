# Memory Access Optimization

## Pattern: tl.max_contiguous + tl.multiple_of Hints

**Source**: `flash-attention/flash_attn_triton_*.py`

```python
# Tell the compiler about pointer alignment and contiguity to generate better memory access instructions
offs_k = tl.arange(0, BLOCK_K)
offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

k_ptrs = K + offs_k[None, :] * stride_kk + offs_n[:, None] * stride_kn
k = tl.load(k_ptrs)
```

**Effect**: The compiler can generate vectorized loads (e.g., 128-bit load), reducing the number of memory access instructions.

## Pattern: EVEN_M/EVEN_N Bounds Check Elimination

```python
@triton.jit
def matmul_kernel(..., EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    if EVEN_M:
        a = tl.load(a_ptrs)  # No mask, compiler generates faster code
    else:
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
```

**Practical Experience**:
- When M/N is an integer multiple of BLOCK_M/BLOCK_N, set `EVEN_M=True`
- This can be automatically determined and set in the autotune `pre_hook`
- Eliminating masks can yield a 5-15% performance improvement (by reducing predicate instructions)

---

## Related Documents

- **Same Series**: [Autotune Configuration and Pruning](autotune-config-pruning.md) — EVEN_M/EVEN_N is often used as an autotune constexpr parameter
- **Same Series**: [Fused Kernel Patterns](fused-kernel-patterns.md) — Fusion reduces the number of HBM read/write operations
- **Prerequisites**: [GPU Memory Hierarchy](../../ref-docs/gpu-memory-hierarchy.md) — Shared memory, coalescing, and vectorized load basics
- **Prerequisites**: [GPU Execution Model](../../ref-docs/gpu-execution-model.md) — Warp-level coalesced access
- **Index**: [Triton Kernel Optimization Patterns in Practice](README.md) — Overview of all patterns
