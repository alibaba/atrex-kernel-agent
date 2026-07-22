# FlashAttention 1–4: GPU Generational Evolution

A comprehensive analysis of FlashAttention versions 1 through 4, tracing how each generation co-designs with NVIDIA GPU architectures from Ampere to Blackwell, covering IO-aware tiling, online softmax, warp specialization, TMEM utilization, and 2-CTA MMA techniques.

---

## 0. Background: Why Attention Needs "Flash"

Standard Self-Attention (where \(S\in\mathbb{R}^{N\times N}\), \(O\in\mathbb{R}^{N\times d}\)):

\[S = QK^\top,\qquad P = \mathrm{softmax}(S),\qquad O = PV.\]

Two major pain points:

| Problem | Cause |
|---------|-------|
| Memory \(\Theta(N^2)\) | Must store the full \(N \times N\) attention matrix S and P in HBM |
| IO bottleneck | S and P are repeatedly transferred between HBM and SRAM, while HBM bandwidth is far lower than compute throughput |

On A100: HBM bandwidth is ~2 TB/s, FP16 compute is 312 TFLOPS. Matrix multiplication itself is compute-bound, but the extensive intermediate result reads/writes in attention make the overall operation memory-bound.

The FlashAttention series' core idea: design the algorithm to be IO-aware, keeping data on-chip as much as possible.

The read path can be understood as: large tensors reside in HBM by default and can be cached on-chip; FlashAttention proactively retains hot data in the SRAM/register path for fused computation.

## 1. FlashAttention-1 (2022) — IO-Aware Tiling

**Target Hardware:** A100 (Ampere)

**Core Contribution:** Tiled computation + Online Softmax + Kernel Fusion, eliminating \(\Theta(N^2)\) intermediate matrix memory

### 1.1 GPU Architecture Features (Ampere/A100)

| Feature | Value | Significance for FA1 |
|---------|-------|---------------------|
| HBM Bandwidth | ~2 TB/s | The bottleneck; FA1's optimization target |
| SRAM (Shared Memory) / SM | 192 KB (configurable) | Upper bound on tile size |
| FP16 Tensor Core throughput | 312 TFLOPS | Compute is not the bottleneck |
| cp.async | Supported | Global → Shared async copy, bypassing registers |
| MMA instruction | mma.sync 16×8×16 | Warp-level synchronous matrix multiply |
| ldmatrix | Supported | Warp-level vectorized load, layout-matched to Tensor Core |

Ampere's `cp.async` allows data to be asynchronously copied from Global Memory directly to Shared Memory without register intermediation (a pain point in the Volta era). This provides the hardware foundation for FA1's tiled pipeline.

### 1.2 Tiling Strategy

Q, K, V are divided into blocks that fit in SRAM:

- Outer loop iterates over K, V blocks \((K_j, V_j)\), block size \(B_c \times d\)
- Inner loop iterates over Q blocks \((Q_i)\), block size \(B_r \times d\)

For each iteration:
1. Load \(Q_i, K_j\) from HBM to SRAM
2. Compute \(S_{ij} = Q_i K_j^\top\) (\(B_r \times B_c\)), entirely in SRAM
3. Perform online softmax update (see below)
4. Load \(V_j\) from HBM to SRAM
5. Update output accumulator \(O_i \mathrel{+=} P_{ij} V_j\)

**Key insight:** The full \(N \times N\) matrix S never materializes in HBM. Each \(B_r \times B_c\) block is computed in SRAM, immediately used, and discarded.

### 1.3 Online Softmax — Solving Global Normalization in Tiled Computation

Softmax is a global operation: \(\mathrm{softmax}(x) = \exp(x) / \sum \exp(\cdot)\) requires the entire row of key positions. When tiling, only \(S_{ij}\) is locally visible, so online softmax (Milakov & Gimelshein 2018) maintains running row statistics \(m_i\) (running max) and \(\ell_i\) (unnormalized exponential sum).

For a fixed Q block \(i\) and K/V block \(j\), with \(S_{ij} = Q_i K_j^\top\) already on-chip:

\[\tilde{m}_{ij} = \operatorname{rowmax}(S_{ij}).\]

\[\tilde{P}_{ij} = \exp\left(S_{ij} - \tilde{m}_{ij}\right)\quad\text{(element-wise, block-local stabilization)}.\]

\[\tilde{\ell}_{ij} = \operatorname{rowsum}\!\left(\tilde{P}_{ij}\right).\]

\[m_i^{\mathrm{new}} = \max\left(m_i,\, \tilde{m}_{ij}\right).\]

\[\ell_i^{\mathrm{new}} = \exp\left(m_i - m_i^{\mathrm{new}}\right)\ell_i + \exp\left(\tilde{m}_{ij} - m_i^{\mathrm{new}}\right)\tilde{\ell}_{ij}.\]

\[O_i \leftarrow \operatorname{diag}\!\left(\ell_i^{\mathrm{new}}\right)^{-1}\!\left(\operatorname{diag}(\ell_i)\exp\left(m_i - m_i^{\mathrm{new}}\right) O_i + \exp\left(\tilde{m}_{ij} - m_i^{\mathrm{new}}\right)\tilde{P}_{ij} V_j\right).\]

After processing all \(j\), this yields \(\mathrm{softmax}(QK^\top)V\) (Theorem 1). If logits include scaling \(1/\sqrt{d}\), it is already incorporated into \(S_{ij}\).

### 1.4 Kernel Fusion

All operations — \(S_{ij}\) matrix multiplication, softmax statistics update, \(P_{ij}V_j\) matrix multiplication, output accumulator update — are fused into a single GPU kernel:

- **HBM reads:** \(Q_i, K_j, V_j\) (each block used to completion)
- **SRAM operations:** Compute \(S_{ij}\) → online softmax → multiply \(V_j\) → update \(O_i\)
- **HBM writes:** Final O and softmax statistics \((m, \ell)\)

Intermediate results \(\tilde{P}_{ij}\) and \(S_{ij}\) exist only in SRAM/registers and are never written back to HBM.

### 1.5 IO Complexity Analysis

**Notation:** \(N\) = sequence length, \(d\) = head dimension, \(M\) = SRAM size per SM (bytes).

**Standard Attention HBM access:**

\[\textbf{1.}\quad S = QK^\top \qquad \mathcal{O}(Nd + N^2)\]
\[\textbf{2.}\quad P = \mathrm{softmax}(S) \qquad \mathcal{O}(N^2)\]
\[\textbf{3.}\quad O = PV \qquad \mathcal{O}(N^2 + Nd)\]

Total: \(\Theta(Nd + N^2)\), with \(N^2\) dominating when \(N \gg d\).

**FlashAttention HBM access:**

With block sizes \(B_r = \lceil M/(4d)\rceil\), \(B_c = \min(\lceil M/(4d)\rceil, d)\), guaranteeing that \(Q_i\) (\(B_r \times d\)), \(K_j, V_j\) (\(B_c \times d\)) and intermediates fit in \(M\) bytes of SRAM:

- Outer blocks: \(T_c = \lceil N/B_c \rceil\); inner Q blocks: \(T_r = \lceil N/B_r \rceil\)
- Each inner iteration: read/write \(Q_i, O_i, \ell_i, m_i\), totaling \(\mathcal{O}(B_r d)\)
- Each outer iteration: read \(K_j, V_j\) as \(\mathcal{O}(B_c d)\); inner repeats \(T_r\) times giving \(\mathcal{O}(Nd)\)

Total:

\[\text{HBM} = T_c \cdot \mathcal{O}(B_c d + Nd) = \mathcal{O}(N^2 d / B_c)\]

Substituting \(B_c = \Theta(M/d)\): \(\Theta(N^2 d^2 / M)\).

**Comparison:**

| Method | HBM Access Order |
|--------|-----------------|
| Standard Attention | \(O(Nd + N^2)\) |
| FlashAttention | \(O(N^2 d^2 / M)\) |

For \(N=2048, d=64, M \approx 100\,\text{KB}\): FlashAttention's HBM access is approximately 1/6 of standard.

When \(M = \Theta(Nd)\), FA degrades to \(\Theta(Nd)\) (linear) — SRAM is sufficient to cover the full row's working set.

**Optimality proof:** Via reduction to matrix multiplication IO lower bounds (Red-Blue Pebble Game, Hong & Kung 1981), HBM access lower bound is \(\Omega(N^2 d^2 / M)\) for exact attention with SRAM capacity \(M\). FlashAttention achieves this bound (asymptotically optimal).

### 1.6 Backward Pass: Recomputation Instead of Storage

Standard backprop requires the \(\Theta(N^2)\) matrix P. FA1 instead:
- Forward pass stores only O, m, \(\ell\) (all \(\Theta(N)\))
- Backward pass recomputes \(S_{ij}\) and \(P_{ij}\) from Q, K

This trades extra FLOPs for massive memory savings. Since matrix multiplication is compute-bound, the recomputation cost is relatively acceptable.

### 1.7 Performance

- BERT-large (seq 512): 15% faster than MLPerf 1.1 record
- GPT-2 (seq 1K): 3× speedup
- First to achieve better-than-chance on Path-X (seq 16K) and Path-256 (seq 64K)
- Memory: \(\Theta(N)\) vs \(\Theta(N^2)\)
- A100: ~124 TFLOPS, only 25-40% of theoretical peak

## 2. FlashAttention-2 (2023) — Maximizing Parallelism and Work Distribution

**Target Hardware:** A100 (Ampere)

**Core Contribution:** Loop reordering + Warp partitioning optimization + sequence-dimension parallelism, doubling throughput

### 2.1 Three Inefficiencies in FA1

FA1 achieves only 25-40% of theoretical peak on A100 due to:

| Inefficiency | Description |
|-------------|-------------|
| Excessive non-matmul FLOPs | A100 matmul throughput is 312 TFLOPS (FP16), non-matmul is only ~19.5 TFLOPS (FP32); each non-matmul FLOP costs the equivalent of 16 matmul FLOPs |
| Insufficient parallelism | FA1 parallelizes only over (batch, heads); small batch/few heads leave many SMs idle |
| Inter-warp communication overhead | FA1's split-K scheme requires warp synchronization through Shared Memory |

### 2.2 Optimization 1: Reducing Non-Matmul FLOPs

FA1 normalizes the output accumulator \(O_i\) with \(\operatorname{diag}(\ell_i)^{-1}\) after every block. FA2 defers this:

- **FA1:** Normalize after every block, incurring extra linear algebra overhead
- **FA2:** Maintain an unnormalized accumulator throughout, perform a single final \(O_i \gets O_i / \ell_i\)

Also streamlines boundary checks and causal masking overhead.

### 2.3 Optimization 2: Loop Reordering — Sequence-Dimension Parallelism

FA1 scans K/V in the outer loop and Q in the inner loop, with one block per (batch, head). FA2 reverses this — outer Q, inner K/V — enabling the sequence dimension to span multiple thread blocks so that even small batches fully occupy all SMs.

With \(N = 8\text{K}\) and block size 128, approximately 64 Q blocks can run in parallel; multiplied by batch × heads, utilization of 108 SMs improves significantly.

### 2.4 Optimization 3: Warp Partitioning — Split-K → Split-Q

**FA1 (split-K):** Multiple warps each handle different K slices; the same Q block is shared. Each warp produces only a partial accumulation of \(QK^\top\) along the key dimension, requiring Shared Memory staging + `__syncthreads` for reduction. High inter-warp synchronization cost.

**FA2 (split-Q):** Multiple warps each handle different Q row bands; \(K^\top\) and V are broadcast to all warps from Shared Memory. Each warp independently maintains output for its own query rows — no cross-warp reduction of \(QK^\top\) needed.

### 2.5 Causal Masking Optimization

For causal attention (positions where key > query are masked), FA2 skips blocks that fall entirely below the diagonal. For triangular masks, this saves approximately 50% of computation.

## 3. FlashAttention-3 (2024) — Deep Hopper Architecture Adaptation

**Target Hardware:** H100 (Hopper)

**Core Contribution:** Async WGMMA/TMA + Warp specialization + ping-pong scheduling + FP8

### 3.1 GPU Architecture Features (Hopper/H100)

Hopper introduces three major hardware changes, each deeply exploited by FA3:

#### 3.1.1 Warpgroup Asynchronous MMA (wgmma)

| Feature | Ampere mma.sync | Hopper wgmma |
|---------|----------------|--------------|
| Participating threads | Warp (32) | Warpgroup (4 Warps = 128) |
| Execution mode | Synchronous (hardware interlock) | Asynchronous (fire-and-continue) |
| MMA shape | 16×8×16 | m64×n(8-256)×k16 |
| Operand B | Registers | Shared Memory (saves registers) |
| SASS instruction | HMMA | HGMMA / QGMMA |

Without wgmma, only about 2/3 of Hopper Tensor Core peak is achievable.

**Key:** wgmma's asynchronous nature means that after issuing a matrix multiply, CUDA Cores can simultaneously perform other work (such as softmax exponential computation). This is the hardware foundation for all FA3 pipeline optimizations.

#### 3.1.2 Tensor Memory Accelerator (TMA)

| Aspect | Conventional Transfer (warp load, cp.async, etc.) | TMA (Hopper+, cp.async.bulk semantics) |
|--------|--------------------------------------------------|----------------------------------------|
| Initiation | Multiple threads generate addresses, transfer per-element/grouped | Single thread issues one bulk block copy |
| Typical path | Global→Registers→SMEM, or async Global→SMEM (still occupies warp resources) | Global↔SMEM bulk async transfer via dedicated unit |
| Register pressure | Address generation occupies register file | Thread can perform unrelated computation, reducing register pressure |
| Cluster optimization | Same data pulled by multiple SMs independently → L2/HBM pressure | Supports multicast: single load shared across SMs via DSMEM, reducing redundant traffic |

#### 3.1.3 Thread Block Cluster and Distributed Shared Memory (DSMEM)

Shared memory views of CTAs within a cluster are connected into DSMEM, enabling inter-SM access through a dedicated path (bypassing L2). This mechanism is related to TMA multicast and FA4's 2-CTA cooperation.

#### 3.1.4 FP8 Data Types

| Format | Exponent | Mantissa | vs FP16 Throughput |
|--------|----------|----------|-------------------|
| E4M3 | 4 | 3 | 2× FP16 (989 → 1978 TFLOPS) |
| E5M2 | 5 | 2 | 2× FP16 |

### 3.2 Core Challenge: Asymmetric Throughput Gap

Severely asymmetric compute on H100:

| Operation | Throughput | Ratio |
|-----------|-----------|-------|
| FP16 matmul (Tensor Core) | 989 TFLOPS | 1× |
| Special functions (exp, SFU) | ~3.9 TFLOPS | 256× slower |

For head dim = 128, matmul FLOPs are approximately 512× the exponential operations. But due to the 256× throughput gap, exponential computation can occupy up to 50% of matmul time.

**FA3's core objective:** Hide softmax's non-matmul work in the "shadow" of matrix multiplication.

### 3.3 Technique 1: Inter-Warpgroup Ping-Pong Scheduling

Two Warpgroups alternate phases via `bar.sync`: while one drives GEMM (Tensor Core), the other performs softmax (CUDA Core / SFU), aligning the idle periods of both unit types.

**Effect:** FP16 forward improves from ~570 TFLOPS to ~620 TFLOPS.

### 3.4 Technique 2: Intra-Warpgroup GEMM-Softmax Pipeline

Leveraging wgmma async, within the same Warpgroup, WGMMA0 → softmax → WGMMA1 are staggered across tiles: while tile \(i\) performs softmax, tile \(i+1\)'s \(QK^\top\) can be issued, and tile \(i-1\)'s PV can overlap with current softmax.

In steady state, three pipeline stages advance simultaneously. The cost is increased register pressure.

**Effect:** ~620 TFLOPS → ~640–660 TFLOPS.

**Are ping-pong and intra-warpgroup pipelining complementary?**

Yes. Ping-pong staggers two WGs in phase at the SM level — keeping both Tensor Core (GEMM) and CUDA Core/SFU (softmax) occupied simultaneously. But within a single tile chain, S → softmax → PV remains sequential per WG.

Intra-warpgroup pipelining overlaps WGMMA0—softmax—WGMMA1 across different tiles within the same WG, using preceding/following GEMM stages to cover softmax latency and reduce TC bubbles within a WG.

The former focuses on SM-level gap-filling; the latter on single-WG instruction-level pipelining.

### 3.5 Technique 3: Warp Specialization (Producer-Consumer)

The classic GEMM division of labor applied to attention: Producers handle data movement exclusively; Consumers handle computation exclusively.

### 3.6 FP8 + Incoherent Processing

**Problem:** LLM activations contain outlier values with magnitudes far exceeding normal features. FP8 quantization suffers severe precision loss from these outliers.

**Solution — Incoherent Processing (QuIP# approach):** Apply a random orthogonal transform to Q, K before attention (typically Hadamard with random signs, \(\mathcal{O}(d\log d)\)), spreading outlier energy across all dimensions. Can be fused with RoPE at minimal extra cost.

**Effect:** Quantization error improves approximately 2.6× relative to naive FP8.

## 4. FlashAttention-4 (2026) — Blackwell Co-Design

**Target Hardware:** B200 (Blackwell)

**Core Contribution:** TMEM/UMMA + software exponential + conditional rescaling + 2-CTA MMA + CuTe-DSL

### 4.1 GPU Architecture Features (Blackwell/B200)

#### 4.1.1 Tensor Memory (TMEM) — Dedicated Storage

**Specifications:** 128 lanes × 512 columns × 4 bytes/cell → approximately 256 KB/SM (comparable to the register file).

In tiled GEMM, the accumulator D has far more read/write accesses than A or B; placing D in dedicated storage closest to the Tensor Core helps saturate compute throughput. TMEM requires explicit allocation and data movement.

A single MMA is expressed as \(D \leftarrow \alpha AB + \beta D\) (commonly \(\alpha=1, \beta=1\), accumulating along K).

| Symbol | Role | Description |
|--------|------|-------------|
| A | Left operand | Current tile from the left matrix (rows × inner dim), performs rank-K update with B |
| B | Right operand | Current tile from the right matrix (inner dim × cols), multiplied with A and added to accumulator |
| D | Accumulator/output | The result block on the M×N output surface; updated every inner-dim step, thus most read/write intensive. On Blackwell, defaults to TMEM |

**Mapping to attention:** For \(S = QK^\top\), Q and \(K^\top\) serve as A and B, with the S tile as D. For \(PV \to O\), P and V serve as A and B, with the O tile as D.

#### 4.1.2 Fifth-Generation Tensor Core (tcgen05.mma / UMMA)

| Feature | Hopper wgmma | Blackwell tcgen05.mma |
|---------|-------------|----------------------|
| Operand A | Registers or SMEM | SMEM or TMEM |
| Operand B | SMEM | SMEM |
| Accumulator D | Registers | TMEM |
| Issuing thread | Warpgroup (128) | Single thread |
| Max tile | ~128×128×16 | 128×256×16 (1-CTA) |
| SASS instruction | HGMMA/QGMMA | UTCHMMA/UTCQMMA/UTCOMMA |

Single-thread issuance eliminates complex data layout from thread-register mapping, freeing registers for other computation.

#### 4.1.3 CTA Pair and MMA.2SM

**CTA pair:** Within a Thread Block Cluster, two CTAs whose ranks differ only in the lowest bit (e.g., 0 and 1, 4 and 5) form a CTA pair. They map to the same TPC (containing two SMs); cooperation at this granularity enables operand sharing, reducing per-SM pressure on SMEM capacity and bandwidth.

**MMA.2SM:** Two SMs collaboratively complete a single MMA, operating at CTA-pair granularity; initiated by a thread in the pair's leader CTA. Compared to single-SM mode, the output M dimension expands approximately 2×: each SM loads different A and D partitions; B is split along a suitable dimension, with both halves aligned between SMs via DSMEM.

**Why useful:** Blackwell SMEM capacity per SM did not double relative to Hopper, but dual-SM cooperation means each SM only handles approximately half the operand footprint — effectively "larger on-chip staging" better matched to faster Tensor Cores.

#### 4.1.4 New Data Types

FP16/BF16/FP8 continue from previous generations; Blackwell's main additions are FP6, FP4, MXFP8/6/4, and NVFP4 entering the GEMM mainline. FA4's engineering partially targets NVFP4 and other ultra-low precision formats.

### 4.2 Core Challenge: Extreme Asymmetric Scaling

Scaling from H100 to B200 is extremely uneven:

| Resource | H100 | B200 | Scale Factor |
|----------|------|------|--------------|
| BF16 Tensor Core | 1 PFLOPS | 2.25 PFLOPS | 2.25× |
| SFU (exp operations) | Baseline | Unchanged | 1× |
| Shared Memory bandwidth | Baseline | Unchanged | 1× |

Tensor Core is 2.25× faster, but SFU and SMEM bandwidth are unchanged → bottlenecks that FA3 barely hid become the dominant constraint on Blackwell.

**Feeds-and-Speeds Analysis (M=N=D=128, per SM):**

| Resource | Forward (2 MMA + MN exp) | Backward (5 MMA + MN exp) |
|----------|--------------------------|---------------------------|
| Tensor Core | 1,024 cycles | 2,560 cycles |
| SFU (exp) | 1,024 cycles | 1,024 cycles |
| Shared Memory | 768 cycles | 3,328 cycles |
| **Bottleneck** | **Compute + Exp tied** | **Shared Memory bandwidth** |

Forward bottleneck is SFU; backward bottleneck is SMEM — this dictates FA4's different optimization strategies for forward and backward passes.

### 4.3 Forward Pass Optimizations

#### 4.3.1 Ping-Pong Q Tile Scheduling

Compared to FA3's ping-pong (§3.3):

- **FA3:** Two Warpgroups alternate phases within the same attention kernel — one WG drives Tensor Core (WGMMA) while the other handles softmax/CUDA Core/SFU, targeting H100's GEMM-much-faster-than-exp asymmetry.
- **FA4:** Each CTA holds two query tiles (\(Q^H\), \(Q^L\), each 128 tokens), two WGs each manage one tile's softmax, with `bar.sync` ensuring the two tiles' exp operations never run concurrently — targeting Blackwell's further TC speedup with unchanged SFU bandwidth causing SFU contention.

**Summary:** FA3 = GEMM vs softmax role rotation; FA4 = two softmax streams time-multiplexing SFU exclusively.

#### 4.3.2 Software Exponential Function (Major Innovation)

Hardware `MUFU.EX2` remains relatively slow compared to Tensor Core on Blackwell. FA4 uses Cody–Waite decomposition to split \(2^x\) into \(2^n \cdot 2^f\) (where \(f \in [0,1)\)), then approximates \(2^f\) with a Horner polynomial (coefficients tuned via Sollya), and reassembles using exponent-domain bit operations. Tunable parameters time-share between hardware exponential and FMA software paths, keeping both unit types occupied.

#### 4.3.3 Conditional Online Softmax Rescaling

- **Standard:** Rescale O whenever the running max changes.
- **FA4:** Only perform full rescale \(\exp(m_{j-1} - m_j)\) when \(m_j - m_{j-1} > \tau\); otherwise skip. When \(m_j - m_{j-1} \le \tau\), use an approximate path with the old max.

The final result still uses correct statistics for normalization, ensuring exactness. Decisions are made at warp granularity to avoid thread divergence.

#### 4.3.4 Dedicated Correction Warpgroup

Rescaling operations are offloaded to an independent Warpgroup, removing them from the critical path.

### 4.4 Backward Pass Optimizations

#### 4.4.1 SMEM Bottleneck Analysis

The backward pass chain contains 5 MMAs (recomputing S, computing dQ, dK, dP, dV) plus element-wise operations. On Blackwell, SMEM bandwidth (not FLOPs) is the bottleneck.

#### 4.4.2 MMA-Softmax Overlap

With accumulators in TMEM and multiple MMAs in-flight, CUDA Cores can process block \(j\)'s softmax while Tensor Cores advance block \(j-1\)'s dK and dQ MMAs.

#### 4.4.3 Transposed Recomputation

The backward pass recomputes S and P in transposed tile layout so that \(S^\mathsf{T}\) and \(P^\mathsf{T}\) exist in TMEM in the A-operand layout required by dV and dK MMAs — eliminating SMEM intermediation.

#### 4.4.4 TMEM Column Reuse

TMEM cannot simultaneously hold all 5 accumulators. FA4 reuses TMEM columns across pipeline stages:
- S and P share one group
- dP, dS, and dQ share another group

#### 4.4.5 2-CTA Backward Pass

\(\mathrm{d}Q = \mathrm{d}S \cdot K\) (and symmetric variants) are large GEMMs. The approach:

1. Logical tiling with CTA mapping to SMs
2. Before exchange: each CTA holds \(M \times N\) dS and \(N \times d\) K; single inner iteration reduction dimension is only N (short relative to UMMA preference)
3. DSMEM exchanges upper/lower halves of dS between CTA pairs
4. After exchange: each CTA gets \(M/2 \times 2N\) and \(2N \times d\), expanding the reduction dimension to 2N — better matching UMMA's access pattern
5. Each CTA writes only its own row band of dQ (\(M/2 \times d\)); two pieces concatenate to the full output

Through data path partitioning, the two CTAs' dQ write sets are non-overlapping, eliminating multi-writer contention and reducing global atomic/reduction pressure.

Paper results at \(N=d=128\): 1-CTA (\(M=128\)) vs 2-CTA (\(M=256\)): total SMEM cycles drop from 3328 to 2688 (~19% reduction) — trading manageable DSMEM traffic for operand and dQ read/write SMEM savings.

#### 4.4.6 Deterministic Mode

FA4 offers optional deterministic backward: semaphores + fences fix dQ accumulation order, CTA swizzling mitigates lock contention, causal paths use SPT sorting. Throughput is approximately 85–90% of non-deterministic mode.

### 4.5 Engineering Innovation: CuTe-DSL

FA4 is written entirely in CuTe-DSL (CUTLASS's Python kernel DSL):

```python
# Installation
pip install flash-attn-4

# Usage
from flash_attn.cute import flash_attn_func
```

- Python → lowered to PTX → CUDA toolkit compilation
- Provides CuTe/CUTLASS abstractions + PTX escape interfaces
- Compilation speed 20-30× faster than C++ CUTLASS templates

### 4.6 Performance

| Metric | Value |
|--------|-------|
| Peak throughput | 1,605 TFLOPS (B200 utilization 71%) |
| vs cuDNN 9.13 (forward) | 1.1-1.3× faster |
| vs Triton (forward) | 2.1-2.7× faster |

## 5. Cross-Generational Evolution Summary

### 5.1 Performance Evolution

| Version | Year | Target GPU | Architecture Gen | Peak TFLOPS | GPU Utilization | vs Previous |
|---------|------|-----------|-----------------|-------------|-----------------|-------------|
| FA1 | 2022 | A100 | Ampere (3rd) | ~124 | 25-40% | — |
| FA2 | 2023 | A100 | Ampere (3rd) | ~230 | 50-73% | ~2× |
| FA3 | 2024 | H100 | Hopper (4th) | ~740 | ~75% | ~2× (vs FA2 on H100) |
| FA4 | 2026 | B200 | Blackwell (5th) | ~1,605 | ~71% | ~2× |

Each generation achieves approximately 2× improvement, driven primarily by deep utilization of new hardware capabilities + algorithm-hardware co-design.

### 5.2 Algorithm-Architecture Correspondence

| FA Technique | Required GPU Feature | First Appeared In |
|-------------|---------------------|-------------------|
| Tiling + kernel fusion | Shared Memory, Tensor Core | Volta (V100) |
| Async data loading | cp.async | Ampere (A100) |
| Async matmul overlapping softmax | wgmma async execution | Hopper (H100) |
| TMA hardware loading | TMA unit | Hopper (H100) |
| Warp specialization | Producer-Consumer model | Hopper (H100) |
| FP8 attention | FP8 Tensor Core | Hopper (H100) |
| TMEM accumulator | Tensor Memory (256KB/SM) | Blackwell (B200) |
| Single-thread MMA issuance | tcgen05.mma (UMMA) | Blackwell (B200) |
| 2-CTA MMA | MMA.2SM | Blackwell (B200) |
| Software exponential | FMA unit idle (SFU bottleneck) | Blackwell (B200) |
| DSMEM data exchange | Cluster + distributed shared memory | Hopper, deeply used in Blackwell |

### 5.3 Bottleneck Migration

Each generation pushes down the previous bottleneck, exposing the next layer — forming a chain of "what is the constraint → how FlashAttention responds."

### 5.4 Operand Location Evolution — Attention Perspective

| Version | Q, K, V Source | Intermediate S, P | Accumulator O |
|---------|---------------|-------------------|---------------|
| FA1/FA2 | HBM → Reg → TC | SRAM/Reg (tiled) | Registers |
| FA3 | HBM → TMA → SMEM → TC | SMEM/Reg (async) | Registers |
| FA4 | HBM → TMA → SMEM → TC | SMEM/TMEM | TMEM |

This aligns exactly with GPU architecture operand location evolution:
- Volta/Ampere: A=Reg, B=Reg, D=Reg (everything in registers)
- Hopper: A=Reg/SMEM, B=SMEM, D=Reg (B moves to SMEM)
- Blackwell: A=SMEM/TMEM, B=SMEM, D=TMEM (D moves to TMEM)

### 5.5 Parallelism Granularity Evolution

| Version | Thread Organization | Parallel Dimensions |
|---------|-------------------|-------------------|
| FA1 | Warp (32) | batch × heads |
| FA2 | Warp (32) | batch × heads × seq_len |
| FA3 | Warpgroup (128) + Warp specialization | batch × heads × seq_len |
| FA4 | Single-thread UMMA issuance + CTA Pair | batch × heads × seq_len × 2-SM |

## References

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," NeurIPS 2022 — arXiv:2205.14135
- Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," ICLR 2024 — arXiv:2307.08691
- Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision," 2024 — arXiv:2407.08608
- Dao et al., "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling," 2026 — arXiv:2603.05451
- Patel & Chen (SemiAnalysis), "NVIDIA Tensor Core Evolution: From Volta To Blackwell," 2025
- NVIDIA, "CUDA C++ Programming Guide"
- NVIDIA, "Parallel Thread Execution (PTX ISA)"
