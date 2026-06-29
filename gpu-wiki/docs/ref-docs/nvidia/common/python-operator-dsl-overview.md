# Emerging Python Operator DSLs: Triton, CuTeDSL, Mojo Overview

A survey of emerging Python-based domain-specific languages for GPU kernel development, comparing Triton, CuTeDSL, TileLang, cuTile, Mojo, and other frameworks across abstraction levels, ecosystems, and performance characteristics.

---

## 1. Introduction

DSL (Domain-Specific Language) refers to languages designed for specific domains (HTML/SQL/regex). More precisely, the tools discussed here are **eDSLs** (embedded DSLs) that reuse Python syntax while employing compilers to change how code executes. AI model development typically occurs in Python but runs on GPGPUs. Since Python cannot run on GPUs, OpenAI built Triton. The fundamental question: can Python DSLs achieve both peak performance and usability? This is inherently a tradeoff — no silver bullet exists. NVIDIA has embraced this direction (announcing cuTile to consolidate their fragmented library landscape), making CUDA embrace Python.

---

## 2. Triton

AI kernel computations are highly regular — tiling alone yields good performance. Triton's design **sacrifices some generality for DSL simplicity**: no need to manage thread organization, only focus on tiles and core float configurations.

### 2.1 Overview

- Tillet's design paper (MAPL 2019): multi-level tiling + automatic optimization
- Subsequent MLIR refactor + Python frontend; **2023-03: PyTorch 2.0 integrated via Inductor**
- Triton leverages Layout design + optimization passes to reduce kernel-writing burden
- **After FlagTree open-sourced, Triton has 12 open-source backends**: NVIDIA / AMD / Intel / CPU / Huawei Ascend / Moore Threads / MetaX / Kunlunxin / ARM China / Tsingmicro / Tianshu / Cambricon (partial)
- On matrix multiply kernels, easily leverages TMA on B200 achieving **nearly 5x speedup** versus native CUDA kernels

> Triton upstream is developing kernel benchmarks and MoE kernel implementations: `python/triton_kernels`.

### 2.2 Extensions

- **ByteDance-Seed/Triton-distributed**: Seed's extension adding communication ops to Triton. MegaKernel and FlashDMoE had significant impact upon release.
- **facebookexperimental/triton TLX**: Triton Low-level Language Extensions — brings warp-aware, hardware-near capabilities back to Triton; essentially hand-writing ttgir and ttngir.
- **microsoft/triton-shared**: First to explore lowering to linalg-level dialect; many subsequent projects build on this.
- **NPU/DSA adaptation**: Triton's op definitions are insufficient for coarse-grained hardware; current `python/triton/language/standard.py` implements sigmoid etc. via math, but NPU/DSA often have their own lowering paths.
- **Triton mainline ≠ these extensions**: The core team strictly controls the programming model like a language committee.
- **Gluon**: Triton upstream's answer for fine-grained control over memory, layout, and scheduling — essentially writing Triton GPU IR. Located at `triton-lang/triton/main/python/tutorials/gluon`.

### 2.3 Ecosystem

- **pytorch-labs/tritonparse**: Visualization and analysis tool. Triton's print functionality is limited; debugging IR is routine.
- **Mogball/triton_lite**: Triton-style interface in Mojo; provides `torch.compile` to replace Triton with Mojo — pushing frontend unification while swapping compiler backends.
- **Integration**: PyTorch, vLLM, SGLang, flash-attention all integrate Triton. Educational resources like `srush/Triton-Puzzles` provide excellent tutorials.

---

## 3. Helion

A Tensor-oriented DSL with **higher abstraction than Triton**. Achieving performance at this level is challenging, but **kernels compile to Triton**, directly inheriting Triton's performance. New chips can leverage this approach for ecosystem compatibility: **abstract is all you need**.

---

## 4. NVIDIA CUTLASS (CuTeDSL)

NVIDIA's response to Triton's success. CuTeDSL operates at **thread level with CuTe abstractions at the center**, similar to CUDA.

- MLIR adoption **significantly improved compilation speed**; includes PyTorch integration
- Supports **DLPack interface** for zero-copy cross-framework data interop with PyTorch tensors
- **`mark_layout_dynamic`** converts static layouts to dynamic layouts, avoiding repeated JIT function recompilation
- Direct AI model integration — the same seamless integration pattern that PyTorch/vLLM/SGLang use with Triton
- NVIDIA exposes some interfaces; the `python/CuTeDSL` package is visible in site-packages after installation

---

## 5. TileLang

Built on TVM thread-level primitives with **three programming interfaces**: explicit memory declaration, explicit thread control, or hands-off mode. **All three syntaxes can appear in the same program.**

> TileLang's strength in inference deployment is substantial — easy to extract performance from. CuTeDSL will soon compete with it directly. Alternatively, like Helion connecting to Triton, **TileLang can connect to CuTeDSL** — abstract is all you need; if you can't beat them, join them.

---

## 6. Apache TVM

A comprehensive deep learning framework providing DSL operator authoring. Popularity has declined in recent years as PyTorch became the de facto standard.

- **TE (Tensor Expression)**: Rich parallel abstractions — `s.bind(axis, te.thread_axis("blockIdx.x"))` binds to GPU thread blocks; supports unroll/vectorize/cache_read/cache_write for shared memory. Supports AutoTVM / AutoScheduler.
- **TensorIR**: The stage between frontend operator modeling and hardware code generation. Fully schedulable loop-level structure using TVMScript-style Python AST (explicit loops + `with T.block`), imperative-leaning.
- **Relax**: Primarily for compute graph description.

---

## 7. Mojo

Chris Lattner's vision to build an **AI infrastructure company** serving various hardware vendors toward AI democratization.

- Mojo syntax closely resembles CUDA — **thread level**
- **Strong typing** without implicit type conversions
- `@parameter` serves as a **compile-time constant parameter modifier**

---

## 8. Halide

Provides Python bindings (no C++ required). Targets image processing, tensor operations, signal processing, and other scenarios with strong data locality. Features **independent compute + schedule syntax**.

---

## 9. Tiramisu

Inspired by Halide but more oriented toward **deeply nested loops, complex scheduling structures, and polyhedral analysis**. Built on ISL with manual schedule focus.

---

## 10. NVIDIA cuTile

NVIDIA's DSL designed to **counter Triton**, directly competing with it. As a vendor, NVIDIA has inherent advantages in extracting performance — likely won't be open-sourced.

> cuTile's performance needs to significantly exceed Triton to displace Triton's established position. Without exposing low-level interfaces it seems difficult, but it can borrow optimization experience from CuTeDSL or leverage unexposed hardware interfaces. Triton is open source — users can modify source code to extract performance. Open source is critically important for customers wanting to squeeze every last drop of performance.

---

## 11. Performance Comparison (tritonbench)

**Test environment**: GPU H20 SXM 96GB x1, CPU 16-core, Memory 154 GB. CUDA 12.8 / Driver 550.127.05 / Python 3.12.11 / torch 2.8.0.dev / pytorch-triton 3.3.1+gitc8757738 / flash_attn_3 3.0.0b1 / tilelang 0.1.3 / tk 87fa717.

Benchmarks covered:

1. flash_attention (aten / sdpa / triton / FA3 / tilelang / tk)
2. gemm `(256, 256, 256)`
3. fp8gemm
4. int4_gemm
5. layer_norm
6. softmax
7. Triton launch_latency

---

## 12. Other Notable Projects

- **HazyResearch/ThunderKittens** ("tk"): Framework-oriented with DSL-style kernel definition and schedule API; strong performance, C++.
- **jax-ml/jax**: NumPy-style high-performance numerical computing with autodiff + JIT + GPU/TPU support.
- **NVIDIA/warp**: Python-based physics simulation (robotics, cloth, soft body, springs) — **1:1 CUDA replica** with framework capabilities.
