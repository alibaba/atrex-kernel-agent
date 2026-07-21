# TMA (Tensor Memory Accelerator)

## Mode: Creating TMA Descriptors on the Host Side

**Source**: `cutedsl/cutlass/dense_gemm_persistent_warp_specialized_sm90.py`

```python
from cutlass.cute.runtime import from_dlpack

# Host side: create TMA descriptor (zero-copy pass to kernel)
tma_a = cute.make_tma_copy(
    cute.SM90_TMA_LOAD,           # TMA operation type
    a_tensor,                      # source tensor
    smem_layout_a,                 # shared memory target layout
    tile_shape_a,                  # tile size
    cluster_shape                  # cluster shape (multi-CTA cooperation)
)
```

**Usage Inside Kernel**:

```python
# Inside kernel: TMA async load global → shared
cute.copy(tma_a, tma_a.get_slice(coord), smem_a)
# No need to manually manage address calculation, bounds checking, or padding — TMA hardware handles it automatically
```

**Practical Experience**:
- TMA descriptors are created on the host side, and the overhead is completed before kernel launch
- TMA automatically handles out-of-bounds accesses (zero-fill), so no mask is needed
- Supports 2D/3D/4D tensors, automatically handling stride and swizzle
- TMA multicast can broadcast data to the shared memory of multiple SMs within a cluster

## Mode: TMA Store (Epilogue Write-back)

```python
# TMA store writes directly from shared memory back to global memory
tma_store_c = cute.make_tma_copy(cute.SM90_TMA_STORE, c_tensor, smem_layout_c, tile_shape_c)

# Inside the kernel
cute.copy(tma_store_c, smem_c, tma_store_c.get_slice(coord))
```

**Advantage**: Compared to `global_store`, TMA store does not consume register file or CUDA core resources, and can fully overlap with computation.

---

## Related Documentation

- **CuTeDSL Basics**: [CuTeDSL Programming Model](../../../common/ref-docs/cutedsl/cutedsl-programming-model.md)
- **CuTeDSL Pipeline**: [CuTeDSL Software Pipeline and Synchronization Patterns](../../../common/ref-docs/cutedsl/cutedsl-pipeline-patterns.md)
- **Hardware Specifications**: [Hopper Hardware Specs](../../hardware-specs/hardware_specs_hopper.md) — H100/H20 peak TFLOPS
- **Reference Kernels**: `reference-kernels/nvidia/hopper/` — 21 Hopper kernel source files
