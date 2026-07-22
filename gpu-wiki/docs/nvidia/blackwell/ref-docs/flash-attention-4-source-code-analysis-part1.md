# FlashAttention-4 Source Code Analysis — Part 1: Architecture Overview

A structural walkthrough of the FlashAttention-4 codebase, covering the entry points, call chain, CuTe DSL compilation pipeline, and architecture-specific dispatch for SM90/SM100.

---

## 1. Introduction

FlashAttention-4 achieves 1605 TFLOPs/s peak performance. This document analyzes the source structure to understand what optimizations are applied and how the code is organized.

The FlashAttention series has a clear evolution path:

| Version | Implementation Language | Target Architecture | Underlying Technology |
|---------|----------------------|--------------------|-----------------------|
| FA2 | C++ CUDA / CUTLASS | Ampere (SM80) | Early hand-written CUDA, later CUTLASS refactored |
| FA3 | C++ CUTLASS / CuTe | Hopper (SM90) | C++ CuTe templates |
| FA4 | Python CuTe DSL | Blackwell (SM100+) | Python DSL → MLIR → PTX |

Note: Python CuTe DSL and CuTe's Python bindings are not the same thing — there is no hierarchical dependency between them.

## 2. Source Code Structure

### 2.1 Repository

Repository: `Dao-AILab/flash-attention`, tag `fa4-v4.0.0.beta4`.

### 2.2 Entry Point Analysis

**Package-level entry** `flash_attn/__init__.py`:

```python
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
```

**Core interface file** `flash_attn/flash_attn_interface.py`:

| Function | Purpose |
|----------|---------|
| `flash_attn_func(q, k, v, ...)` | Standard FlashAttention (Q, K, V passed separately) |
| `flash_attn_qkvpacked_func(qkv, ...)` | QKV packed optimized version |
| `flash_attn_varlen_func(q, k, v, cu_seqlens, ...)` | Variable-length sequences |
| `flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, ...)` | Variable-length + packed |
| `flash_attn_with_kvcache(...)` | KV Cache inference |

**CuTe DSL core implementation** `flash_attn/cute/interface.py`: DSL implementation / architecture-specific dispatch (SM90/SM100) / JIT compilation and caching.

**FlexAttention integration entry:**

```python
from torch.nn.attention.flex_attention import flex_attention
flex_flash = torch.compile(
    partial(flex_attention, kernel_options={"BACKEND": "FLASH"}),
    dynamic=False
)
```

**Call chain:**

```
User call
  ↓
flash_attn_func() / flash_attn_qkvpacked_func()  [flash_attn_interface.py]
  ↓
_flash_attn_forward() / _flash_attn_backward()  [Python layer]
  ↓
CuTe DSL Kernel (JIT compiled)                   [flash_attn/cute/*.py]
  ↓
PTX → SASS (GPU machine code)
```

**Architecture-specific entries** `flash_attn/cute/`:

- `flash_fwd_sm100.py`: Blackwell (SM100) forward
- `flash_fwd_sm90.py`: Hopper (SM90) forward
- `flash_bwd.py`: Backward
- `pipeline.py`: Async pipeline management

## 3. CuTe DSL

### 3.1 Programming Interface

Design goal: maintain consistency with CuTe C++, provide Python usability, support Ampere → Blackwell. Compilation time drastically reduced compared to template-based approaches.

### 3.2 CuTe DSL vs cuTile

| Feature | CuTe DSL | cuTile |
|---------|----------|--------|
| Abstraction level | Low-level (Thread-based) | High-level (Tile-based) |
| Programming model | SIMT | Array-based |
| Control granularity | Explicit thread/memory/sync control | Only specify tile and math ops |
| Hardware details | Manual TC / SMEM management | Compiler automatic |
| Intermediate representation | MLIR → LLVM IR → PTX | Tile IR → PTX |
| Architecture support | Ampere, Hopper, Blackwell | Blackwell only (SM10x/SM12x) |
| Target users | Performance engineers, library developers | AI researchers, productivity-focused |

**Why both coexist:** CuTe DSL targets "speed-of-light" extreme performance (FA4's choice). cuTile targets "good performance with less effort" (comparable to Triton — 86 lines of Python generating 1900 lines of PTX, with automatic barrier/leader election handling). Tile IR is a higher-level virtual ISA than PTX, enabling automatic forward compatibility across Tensor Core generations.

NVIDIA currently has five different Python DSLs (OpenAI Triton, CuTe Python, cuTile Python, Numba, Warp), with teams using multiple DSLs in competition.

### 3.3 CuTe DSL Compilation Pipeline

Three stages:

```
Python source
  ↓
Pre-Staging (AST rewrite)        → Insert callbacks to capture control flow structure
  ↓
Meta-Staging (Python interpreter) → Execute and generate MLIR IR
  ↓
Object-Staging (MLIR compiler)    → Lower to PTX/SASS
```

Generated MLIR uses the CuTe dialect + standard MLIR dialects (scf, cf, gpu):

```mlir
!memref_gmem_f16 = !cute.memref<f16, gmem, "(128,256):(256,1)">
func.func @cutlass_kernel_Epilogue(%A: !memref_gmem_f16,
                                    %B: !memref_gmem_f16,
                                    %C: !memref_gmem_f16) {
    scf.for ... {
        %fA = cute.slice ...
        %fB = cute.slice ...
        %fC = cute.slice ...
        cute.gemm(%fA, %fB, %fC)
    }
}
```

The CuTe DSL MLIR dialect compiler is **not currently open-source**. Similar to Triton, compilation intermediate files are cached in `.cache/cutedsl`.

## 4. Testing Notes

### 4.1 B200 Testing

Measured results may not reach the paper's 1700 TFLOPS — approximately 1300 TFLOPS observed, potentially due to environment configuration. Large-size GEMM also only reaches ~1500 TFLOPS.

### 4.2 Architecture Compatibility

- **5090 (SM120):** FA4 cannot run. In the latest code, SM80/SM120 implementations use FA2, SM90 uses FA3, and only SM100/103/110 use FA4. The 5090 lacks TMEM and does not support tcgen05, so FA4 optimizations are inapplicable.
- **Hopper (SM90):** Current main branch FA4 performance on Hopper may not exceed FA3 — pending further testing.

## 5. Summary

Understanding MLIR may be useful for deeper FA4 analysis, since beyond reading the Python code's layout and pipeline design, examining compilation intermediates in the `.cache` directory can provide additional insight.

The FA series optimization philosophy: through tiling + recomputation + kernel fusion, push attention's effective compute throughput toward large-GEMM levels by solving the memory wall (IO) problem.
