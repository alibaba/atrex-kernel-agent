## Overview

CuTe (CUDA Templates) DSL is the primary abstraction layer in CUTLASS 4.5.0 for Blackwell kernels. FlashAttention-4 was implemented entirely in CuTe-DSL (Python variant), achieving 20-30× faster compilation than C++ templates.

## SM100 MMA Atoms

```python
# CuTe-DSL: SM100 MMA atom for BF16
from cutlass.cute import *

# 1-SM MMA: m128 x n256 x k16
mma_atom = SM100_MMA_F16BF16_SS  # inputs from shared memory
# Accumulator goes to TMEM automatically

# 2-SM MMA: m256 x n256 x k16
mma_atom_2sm = SM100_MMA_F16BF16_SS_2SM
```

## TMEM as CuTe Locale

```python
# TMEM allocation in CuTe
tmem_tensor = make_tensor(
    make_tmem_ptr(tmem_addr),
    make_layout(make_shape(128, 256))  # rows x cols
)

# Copy TMEM → registers for epilogue
copy(tmem_tensor, reg_tensor)  # tcgen05.ld under the hood
```

## TMA Copy Atoms

```python
# TMA bulk copy: global → shared
tma_copy = SM100_TMA_LOAD_2D

# Setup TMA descriptor
tma_desc = make_tma_copy(
    tma_copy,
    global_tensor,
    smem_layout,
    tile_shape,
    cluster_shape
)
```

## Warp-Specialized Kernel Skeleton

```python
@cute_kernel
def blackwell_gemm(A, B, C):
    # Warp 0: TMA producer
    if warp_id == 0:
        for stage in pipeline:
            tma_copy(A_tile, smem_A[stage])
            tma_copy(B_tile, smem_B[stage])
            arrive(mbarrier[stage])

    # Warp 1: MMA consumer
    elif warp_id == 1:
        tmem = alloc_tmem(256)  # columns
        for stage in pipeline:
            wait(mbarrier[stage])
            fence_after_thread_sync()
            mma(smem_A[stage], smem_B[stage], tmem)
        signal_epilogue()

    # Warps 2+: Epilogue
    else:
        wait_epilogue()
        regs = load_tmem(tmem)
        store_global(C, regs)
        dealloc_tmem(tmem)
```

## Why CuTe-DSL for Blackwell

1. **20-30× faster compilation** than C++ CUTLASS templates
2. Python-native: easier to iterate and debug
3. Same performance as hand-written C++ (FlashAttention-4: 1605 TFLOPS)
4. First-class TMEM and tcgen05 support in CUTLASS 4.5.0
5. Automatic layout computation and swizzle handling

## Full Examples (verbatim upstream code shipped locally)


| File | Purpose | Size |
|---|---|---|

The `fp16_gemm_{0..6}.py` series from `examples/python/CuTeDSL/blackwell/tutorial_gemm/` in NVIDIA/cutlass PR-3106 is the authoritative CuTe DSL learning path: it walks from a naive FP16 GEMM baseline through 2CTA MMA with TMA multicast, warp specialization, static persistent scheduling, preferred / dynamic clusters, TMA prefetch, and ends with Programmatic Dependent Launch (PDL). Reading them in order is the recommended on-ramp.

## Related
- [tcgen05-mma](../hardware/tcgen05-mma.md) -- Underlying MMA instruction
- [flash-attention-4](../kernels/flash-attention-4.md) -- CuTe-DSL implementation
- `nvidia-cutlass-blackwell` -- Official CUTLASS Blackwell reference
