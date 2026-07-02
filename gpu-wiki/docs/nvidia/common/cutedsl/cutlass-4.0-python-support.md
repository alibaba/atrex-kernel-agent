# CUTLASS 4.0 Python Support

Overview of NVIDIA CUTLASS 4.0's new Python DSL interface, which brings CuTe abstractions to Python without sacrificing performance.


**Last updated**: 2026-06-30

---

## 1. GEMM as a Foundational Computation

General Matrix Multiplication (GEMM) is the most frequently used and computationally intensive fundamental operation in modern high-performance applications — from computational fluid dynamics and quantum chemistry to neural network training and inference. GPUs excel at this core computation. NVIDIA has long provided closed-source cuBLAS/cuDNN libraries, but these require familiarity with low-level C/C++ APIs and are tightly coupled to specific GPU architectures. Porting to new architectures demands significant optimization effort, creating a non-trivial barrier.

---

## 2. CUTLASS Overview

**CUDA Template Abstractions for Linear Algebra Subroutines** — an open-source library released by NVIDIA in 2017, serving as the underlying implementation paradigm reference for cuBLAS/cuDNN.

### 2.1 Template Abstraction Coverage

1. **Mixed precision:** FP64, FP32, TF32, FP16, BF16 multiply-accumulate abstractions
2. **Specialized data movement:** Asynchronous memory management + TiledCopy components for overlapping data transfer and computation
3. **Tensor Core FP32 emulation** + 8-bit floating point (e5m2 / e4m3)
4. **Block-scaled data types:** NVIDIA NVFP4 + OCP standard MXFP4 / MXFP6 / MXFP8; provides parallelization hierarchy primitives at different levels
5. **Narrow integers:** 4-bit / 8-bit signed and unsigned
6. **1-bit binary** (where architecture natively supports it)
7. **Architecture span:** Volta / Turing / Ampere / Ada / Hopper / Blackwell

Through C++ template encapsulation of thread hierarchies, data layouts, and memory access logic, hardware details are abstracted away. When GPU hardware is updated, code requires virtually no modification — only recompilation with the new compiler version.

### 2.2 No Performance Compromise

CUTLASS primitives are highly efficient. Device-wide GEMM kernels built from them achieve near-optimal theoretical peak throughput utilization. CUTLASS 3.8 achieves high percentages of theoretical peak utilization across various input/output data types on NVIDIA Blackwell SM100 architecture GPUs. Since CUTLASS 3.1, performance on H100 has seen continuous improvement (CUTLASS 3.5.1 compiled with CUDA 12.5u1, Tensor Core using mma / wgmma instructions).

---

## 3. CuTe: The Backend Core Library

Introduced in CUTLASS 3.0 (2024) as a major new capability for describing and manipulating tensors of threads and data:

- Provides **Layout** and **Tensor** objects that compactly pack dtype/shape/storage/layout while handling complex indexing for the user
- Core abstraction is **hierarchical multidimensional layouts**, capable of expressing virtually everything needed for efficient dense linear algebra
- Layouts can be combined and manipulated through function **composition**; tiling, partitioning, and other common operations are built on top
- CUTLASS 3.0+ adopts CuTe throughout the entire GEMM hierarchy template, simplifying design and improving composability and readability

---

## 4. CUTLASS 4.0: New Python Support

Python is one of the most widely adopted and beginner-friendly programming languages. CUTLASS 4.0 builds on the rich C++ kernel programming abstraction ecosystem of previous versions by providing a native Python interface in the form of **DSL (Domain-Specific Languages)** for writing high-performance CUDA kernels based on core CUTLASS and CuTe concepts — **with zero performance overhead**.

**Advantages:**

- Smoother learning curve
- Faster compilation times
- Native integration with deep learning frameworks without glue code
- More intuitive metaprogramming without requiring deep C++ expertise

NVIDIA positions the CUTLASS DSL as **a family of DSLs**. CuTe DSL, released with 4.0, is the first member. It is a low-level programming model that:

- Maintains full consistency with CuTe C++ abstractions
- Exposes layouts, tensors, hardware atoms, and other core concepts
- Provides complete control over hardware thread and data hierarchies
- Demonstrates optimal matrix multiplication and other linear algebra operations for programmable, high-throughput Tensor Cores on NVIDIA Ampere / Hopper / Blackwell architectures

CuTe DSL is currently in **public beta**, with general availability planned for **late summer 2025**.


## Related

- [CuTeDSL API Reference Guide](cutedsl-api-reference-guide.md)
- [CuTeDSL Inline PTX Writing Overview](cutedsl-inline-ptx-patterns.md)
- [CuTeDSL Software Pipeline and Synchronization Patterns](cutedsl-pipeline-patterns.md)
- [CuTeDSL Programming Model](cutedsl-programming-model.md)
- [CUTLASS 3.x Architecture](cutlass-3x-architecture.md)
- [CUTLASS GEMM Optimization Strategy](cutlass-gemm-optimization.md)
- [Community GEMM Optimization Practical Summary](../../../generic/gemm-optimization-guide.md)
- [CUTLASS/CuTe Core Concepts and Layout Algebra](cutlass-cute-fundamentals.md)
