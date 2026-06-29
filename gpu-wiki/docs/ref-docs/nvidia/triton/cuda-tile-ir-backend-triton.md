# CUDA Tile IR Backend: Advancing OpenAI Triton GPU Programming

Overview of NVIDIA's work integrating CUDA Tile as a backend for OpenAI Triton, enabling Triton kernels to compile directly to tile-native IR instead of thread-level PTX.

---

## 1. Introduction

NVIDIA CUDA Tile is a GPU programming model designed to provide portability for NVIDIA Tensor Cores, unlocking peak GPU performance. **A key advantage of CUDA Tile is allowing developers to build custom DSLs on top of it.** This document describes NVIDIA's ongoing work integrating CUDA Tile as a backend for OpenAI Triton.

OpenAI Triton is an open-source Python DSL purpose-built for writing deep learning kernels on GPUs. It supports block-based computation, partitioning data and computation into smaller blocks. Triton includes an MLIR-based compiler that generates PTX code, enabling researchers without CUDA experience to write efficient GPU code.

---

## 2. What are CUDA Tile and CUDA Tile IR?

**CUDA Tile** extends the CUDA programming model with native tile programming support. **First introduced in CUDA 13.1**, it marks a significant evolution in GPU programming.

Unlike requiring developers to think in terms of SIMT with individual threads, **the tile-based model lets developers express computation at a higher abstraction level**: developers operate on data blocks (tiles) while the compiler and runtime automatically handle thread scheduling, hardware mapping, and resource allocation.

**CUDA Tile IR** is an MLIR-based intermediate representation and compiler infrastructure. The entire CUDA Tile development flow is driven by the CUDA Tile IR specification, which formally defines the **semantics, operations, and type system** required for tile computation on NVIDIA GPUs.

---

## 3. What is Triton-to-TileIR?

The Triton-to-TileIR backend is a bridge layer for Triton that **targets CUDA Tile IR (rather than PTX) as output code**. It extends the Triton compiler ecosystem, enabling developers to compile and run GPU kernels written in OpenAI Triton on the new CUDA Tile IR backend — **without rewriting code** — seamlessly leveraging modern hardware capabilities.

> Triton is fundamentally a tile-based programming language whose philosophy has developers compute in terms of data blocks (tiles) rather than individual threads. This design aligns naturally with CUDA Tile IR. This alignment enables a more direct backend compilation path: **Triton-to-TileIR no longer converts Triton's tile-level abstractions into thread-level SIMT code; instead, it preserves tile-level semantics and compiles directly to CUDA Tile IR which natively supports tile-granularity computation.**

Switching is simple: **set an environment variable** to redirect compilation from the PTX backend to the CUDA Tile IR backend. Triton users can flexibly choose PTX or CUDA Tile IR backend per-kernel based on specific requirements.

---

## 4. Development Roadmap

An incubation project under the triton-lang organization. The roadmap includes:

1. **Core conversion infrastructure**: Implement MLIR-based language conversion mechanism mapping Triton operations to corresponding CUDA Tile IR operations.
2. **Testing and validation**: Build comprehensive test suites verifying semantic correctness of control flow, memory access patterns, numerical precision, and edge-case conversions.
3. **Performance benchmarking**: Systematically compare kernel performance when compiled via TileIR vs PTX across matrix multiplication, convolution, elementwise operations, and reductions.
4. **Open-source project integration**: Close collaboration with the open-source community to advance CUDA Tile IR backend support in projects like **Helion**.

---

## 5. How to Use Triton-to-TileIR

### 5.1 Prerequisites

- **CUDA version**: CUDA 13.1 or higher
- **GPU architecture**: NVIDIA **Blackwell architecture GPU** (e.g., GeForce RTX 5080); future CUDA versions are expected to add support for earlier GPU architectures

### 5.2 Building from Source

> reference source: github.com/triton-lang/Triton-to-tile-IR

Clone the Triton-to-tile-IR repository and install in editable mode:

```bash
cd Triton-to-tile-IR
pip install -e .
```

### 5.3 Verifying Tile IR Compilation

```bash
cd python/tutorials
export ENABLE_TILE=1
python 01-vector-add.py
```

When the Tile IR backend is active, **Triton uses `.tileIR` file extension for cached compiled kernels**, rather than the standard cubin files used by the SIMT backend. The Triton cache directory is typically at `~/.triton/cache`.

---

## 6. Limitations

### 6.1 Unsupported Operations

The Tile IR backend does not yet fully cover all Triton-supported operations. Consult the "currently unsupported or partially covered operations" list for details.

### 6.2 Tensor-of-Pointer Performance Regression

On CUDA 13.1's Tile IR backend, **Triton's "tensor-of-pointer" pattern** (using pointer-composed tensors to describe memory access patterns) does not yet achieve optimal performance. This is a **temporary limitation**. For affected workloads:

- Temporarily switch critical operations back to the SIMT backend
- Monitor future releases for improvements
- **Optimize code to use TMA load/store API**

### 6.3 Optimization: Before and After

**Before (tensor-of-pointer style):**

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
offs_k = tl.arange(0, BLOCK_K)
a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
a = tl.load(a_ptrs)
b = tl.load(b_ptrs)
```

> Each element in `a_ptrs` is an explicitly computed pointer within the kernel, even when the tile is contiguous and its layout can be fully described by (shape, strides, block_shape).

**After (TMA-style):**

```python
desc_a = tl.make_tensor_descriptor(
    a,                                # base pointer
    shape=(M, K),
    strides=(stride_am, stride_ak),
    block_shape=(BLOCK_M, BLOCK_K)    # tile size
)
desc_b = tl.make_tensor_descriptor(
    b,
    shape=(K, N),
    strides=(stride_bk, stride_bn),
    block_shape=(BLOCK_K, BLOCK_N)
)
offs_m = pid_m * BLOCK_M
offs_n = pid_n * BLOCK_N
a_tile = desc_a.load([offs_m, 0])         # [BLOCK_M, BLOCK_K]
b_tile = desc_b.load([0, offs_n])         # [BLOCK_K, BLOCK_N]
desc_c.store([offs_m, offs_n], acc)       # TMA-backed store
```

---

## 7. Strategic Significance

The Triton-to-TileIR project represents a pivotal step in GPU programming evolution, **bridging the gap between developer productivity and hardware efficiency**:

- Connects Triton's accessible, tile-oriented programming model with the **CUDA Tile IR virtual instruction set**.
- Provides existing Triton developers a smooth upgrade path, requiring minimal code changes to experience next-generation GPU architectures.
- Demonstrates the strategic value of **deep collaboration between language designers and hardware vendors** to the broader GPU programming ecosystem: simpler access to advanced hardware capabilities without sacrificing high-level abstraction iteration speed.

The ultimate metric is straightforward: **can researchers without deep GPU expertise write near-optimal Triton code on NVIDIA GPUs?**
