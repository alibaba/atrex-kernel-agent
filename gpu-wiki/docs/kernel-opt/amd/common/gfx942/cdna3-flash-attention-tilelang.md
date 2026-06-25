# Flash Attention Optimization (TileLang on MI308X)

## TileLang Programming Model

TileLang uses "Tile" as the fundamental unit, providing declarative GPU programming:

```python
@T.prim_func
def flash_attention(Q: T.Tensor(q_shape, dtype), K: T.Tensor(kv_shape, dtype),
                    V: T.Tensor(kv_shape, dtype), Output: T.Tensor(q_shape, dtype)):
    with T.Kernel(num_split_q, batch * heads, threads=threads) as (b_split, byz):
        # Memory allocation
        Q_shared = T.alloc_shared([block_M, dim], dtype)       # LDS
        acc_o = T.alloc_fragment([block_M, dim], accum_dtype)  # Registers
        
        # Data movement
        T.copy(Q[bz, offset:offset+block_M, by, :], Q_shared,
               coalesced_width=qk_coalesced_width)
        
        # Computation
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
               k_pack=k_pack, policy=GemmWarpPolicy.FullRow)
        
        # Reduction
        T.reduce_max(acc_s, m_i, dim=1, clear=False)
        T.reduce_sum(acc_s, row_sum, dim=1)
```

## Memory Hierarchy Usage

| Level | Allocation Method | Purpose | Data Type |
|------|---------|------|---------|
| HBM | Input/Output tensor | Persistent storage | float16 |
| LDS (Shared Memory) | `T.alloc_shared` | Tile cache reuse | float16 |
| Registers | `T.alloc_fragment` | Accumulator and intermediate values | float32/float16 |

## MI308X Specific Optimizations

- **Swizzle/Rasterization**: `T.use_swizzle(panel_size, enable=enable_rasterization)` improves memory access patterns
- **Memory Coalescing**: `coalesced_width` parameter controls coalescing width (QK and V operations can be configured independently)
- **Warp Scheduling**: `GemmWarpPolicy.FullRow` adapts to the MI308X warp scheduling architecture
- **Data Packing**: `k_pack` parameter optimizes data packing for compute units
- **Pipelining**: `T.Pipelined(loop_end, num_stages=N)` overlaps prefetching and computation

## Performance Results

batch=1, heads=8, seq_len=4096, dim=128:

| Implementation | Latency (ms) | Relative PyTorch Speedup | Relative Triton Speedup |
|------|----------|------------------|-----------------|
| PyTorch | 0.97 | — | — |
| Triton | 0.55 | 1.76x | — |
| **TileLang** | **0.36** | **2.69x** | **1.53x** |

Autotuning searched 108 configurations in approximately 1 second. Optimal configuration: block_M=128, block_N=32, threads=512, enable_rasterization=True.

---

## Related Documents

- [MI308X (CDNA3) Kernel Optimization Practices (Index)](cdna3-mi308x-kernel-practices.md) -- Index of the case study collection this document belongs to
- [AMD GPU Kernel Optimization Framework Overview](../../../../ref-docs/amd/common/amd-kernel-optimization-frameworks.md) -- TileLang's position within the AMD optimization framework
- Triton Kernel Optimization Patterns in Practice -- General Flash Attention optimization patterns
