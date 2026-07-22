# From H100 to B200: GPGPU and LLM Scaling Deep Analysis

A comprehensive analysis of GPU microarchitecture, cluster networking, collective communication, and distributed training scaling strategies based on the Google DeepMind report "How to Scale Your Model."

---

## 1. Introduction: GPGPU Evolution and AI Infrastructure

In the deep learning era, GPUs have evolved into GPGPUs (General-Purpose GPUs) focused on parallel computation. While Google's TPU excels in specific domains, NVIDIA GPUs remain the mainstream choice for large model training due to their versatility and strong software ecosystem (CUDA).

Understanding GPU architecture, memory hierarchy, and inter-node communication is essential for designing efficient distributed training systems. This document covers four dimensions: chip architecture (Silicon), networking, communication primitives (Collectives), and scaling strategies.

---

## 2. Chip Architecture: The GPU Microworld

Unlike TPUs (systolic arrays with SIMD), GPUs emphasize flexibility and multi-threaded parallelism.

### 2.1 Streaming Multiprocessor (SM)

H100 has 132 SMs; B200 increases to 148. Each SM contains:
- **Tensor Cores**: Dedicated matrix multiplication accelerators (dominant FLOPS contributor)
- **CUDA Cores**: General-purpose vector operations (element-wise)
- **Warp Schedulers**: Thread warp scheduling and execution

Compared to TPU's centralized large-core design, GPU's many independent SMs provide extreme parallelism and scheduling flexibility, but impose higher requirements on cache coherence and synchronization.

### 2.2 Memory Hierarchy

| Level | Description | Key Metric |
|-------|-------------|------------|
| HBM | Model weights, activations, optimizer states | H100: 3.35 TB/s; B200: 8 TB/s |
| L2 Cache | Chip-wide shared cache | H100: 50 MB; B200: 126 MB |
| SMEM/L1 | Per-SM on-chip shared memory | 256 KB/SM (extremely low latency) |
| Registers | Per-thread private storage | Fixed per-SM |

The L2 cache growth from H100's 50 MB to B200's 126 MB is critical for small-batch inference scenarios.

### 2.3 Tensor Core Evolution: Key Breakthroughs

**Hopper's TMA (Tensor Memory Accelerator)**:
- Traditional: CUDA Cores execute LDGSTS, manually moving data from GMEM to SMEM
- TMA: Dedicated hardware unit performs async transfers; CUDA Cores only initiate requests
- Result: 40% latency reduction, 30% bandwidth improvement

**Blackwell's 2nd-Gen Transformer Engine** supports dynamic FP8/FP6/FP4 precision:
- Weights: FP4/FP6 storage (halved/75% memory access)
- Activations: FP8 computation (maintaining accuracy)
- Gradients: BF16/FP32 accumulation (preventing overflow)

B200 theoretical inference throughput: 9,000 TFLOPS (FP4) — 9x over H100.

### 2.4 Roofline Model

The critical metric is **Arithmetic Intensity** (computation per memory access).

H100: 990 TFLOPS (BF16) / 3.35 TB/s = ~295 ops/byte threshold. Tasks below 295 ops/byte are memory-bound; above are compute-bound.

**Blackwell's performance gains** come from three factors:
- Die area: B200 ~1600 mm² (dual-die) vs. H100 ~800 mm² (2x)
- Power: GB200 liquid-cooled at 1200W vs. H100 at 700W (+71%)
- 2nd-Gen Transformer Engine: FP4 support reaching 9 PFLOPS

Cost analysis: BF16 FLOPS/mm² improves only 14%; BF16 FLOPS/W improves 47%. The real leap requires FP8/FP4 quantization and software-hardware co-optimization.

---

## 3. Networking: Foundation for Cluster Scaling

### 3.1 Intra-Node (Scale Up): NVLink

Within a node (typically 8 GPUs), NVLink provides high-speed interconnect via NVSwitch for all-to-all connectivity. H100 NVLink unidirectional bandwidth: 450 GB/s.

Application: Tensor Parallelism (TP) — frequent AllReduce at every layer.

### 3.2 Inter-Node (Scale Out): InfiniBand

Cross-node communication uses InfiniBand with Fat Tree topology. Full bisection bandwidth ensures constant inter-node bandwidth regardless of cluster size. Typical bandwidth: 400 GB/s (lower than intra-node NVLink, determining parallelism strategy choices).

---

## 4. Collective Communication Primitives

### 4.1 Core Types

- **AllReduce**: Reduce + broadcast. Used in DP for gradient synchronization
- **AllGather**: Collect distributed data to all nodes. Used in FSDP for parameter gathering
- **ReduceScatter**: Reduce then scatter. Often paired with AllGather
- **AllToAll**: Each node sends different data to all others. Core pattern for MoE models

### 4.2 Intra-Node Performance (Ring Algorithm)

For 1 GB data on H100's 450 GB/s NVLink:
- AllGather/ReduceScatter: ~2.2 ms (actual: 87.5% efficiency due to ring algorithm)
- AllReduce (without SHARP): ~4.4 ms (2x AllGather)
- AllToAll: ~0.28 ms (8x faster — each GPU receives only 1/8 of total data)

### 4.3 Cross-Node Performance

For 1 GB data on 400 GB/s inter-node bandwidth:
- AllGather/ReduceScatter: ~2.5 ms (constant regardless of cluster scale due to Fat Tree)
- AllToAll degrades severely: from intra-node 0.28 ms to cross-2-node ~1.25 ms (4.5x slowdown)

This explains why MoE expert parallelism is typically limited to few nodes.

### 4.4 SHARP: In-Network Reduction

SHARP offloads reduction computation to network switches:
- Theory: 2x AllReduce bandwidth
- Practice: 370 GB/s → 480 GB/s (~30% actual gain)

Gap due to switch compute limitations, protocol overhead, and sync latency.

### 4.5 Multi-Axis Sharding Optimization

When arrays are sharded across multiple axes simultaneously: if the inner axis spans multiple nodes, outer-axis communication cost decreases proportionally.

Example: DeepSeek V3 with 64-way EP + 16-way PP:
- 1024-way model parallelism pushes DP AllReduce to spine layer
- 2-way DP further doubles bandwidth
- Communication time reduces to 1/16 of single-axis parallelism

---

## 5. Scaling Strategies

The golden rule: **compute time must exceed communication time** (Roofline principle).

### 5.1 Data Parallelism (DP / FSDP)

Critical batch size on H100: 990 TFLOPS / 400 GB/s ≈ 2,475 tokens/GPU.

Below this threshold: GPU idle waiting for communication. Above: compute dominates.

**Blackwell impact**: B200's critical batch size rises to ~5,625 tokens/GPU (2.27x H100). However, FP8 quantization halves gradient transfer (offsetting the 2x compute increase), keeping effective DP efficiency roughly equivalent.

**FSDP (ZeRO-3) memory management** for LLaMA-3 70B (BF16 training):
- Weights: 140 GB; Optimizer states: 560 GB; Total: 700 GB
- On 8× H100 (80 GB each): Pure DP = 87.5 GB/GPU → OOM
- ZeRO-3: 700 GB / (8×512) = 0.17 GB/GPU → feasible (but requires AllGather at each forward pass)

### 5.2 Tensor Parallelism (TP)

Splits matrix multiplication across GPUs. Communication is extremely frequent — strictly limited to NVLink domain (intra-node). Typical TP size ≤ 8.

Performance requires three-level tiling:
- Block tiling (cross-SM parallelism)
- Warp tiling (cross-warp-scheduler parallelism)
- Thread tiling (instruction-level parallelism)

This hierarchical design improves arithmetic intensity, pushing performance from 16 TFLOPS to 22 TFLOPS (approaching cuBLAS's 23.2 TFLOPS).

### 5.3 Pipeline Parallelism (PP)

Partitions model by layers across devices with pipelined execution. Communication: point-to-point (minimal volume). Cost: pipeline bubbles cause idle compute.

### 5.4 Expert Parallelism (EP)

MoE-specific: tokens routed to different expert GPUs. Efficiency depends on FFN hidden size (F) vs. network bandwidth ratio. When F is large enough, cross-node EP is efficient.

DeepSeek V3 configuration: E=256, k=8, F=1536, 64-way EP across 8 nodes. Feasible because:
- Small F (1536): compute-dominated
- High sparsity (k/E = 3.125%): ~97% communication reduction

---

## 6. GB200 NVL72: Scale-Up Renaissance

### 6.1 Architecture

72 GPUs across 9 trays, each containing 8 GB200 Superchips + 4 NVSwitch 5.0. Full 72×72 all-to-all connectivity:
- 72 GPUs = 2,592 NVLink connections
- Per GPU: 18 ports × 50 GB/s = 900 GB/s bidirectional

### 6.2 Key Advantages

- **TP revival**: TP-72 across 9 trays (vs. H100's TP-8 limit)
- **Reduced IB dependency**: 900 GB/s intra-domain >> 400 GB/s inter-node
- **Simplified parallelism**: TP-72 × DP-8 = 576 GPUs (fewer PP stages needed)
- **Memory breakthrough**: 13.82 TB can hold 72T parameters (BF16) or 220T MoE

### 6.3 Performance Example

TP-72 on LLaMA-405B (single MLP layer, batch=4096):
- Compute: ~5.6 ms
- AllGather: ~0.15 ms
- Communication overhead: 2.7% (1/3 of H100 TP-8)

---

## 7. Communication Wall Challenge

GPU compute growth far outpaces bandwidth growth:
- Compute: A100 312 → B200 2,250 TFLOPS (7.2x)
- Intra-node BW: A100 300 → B200 900 GB/s (3x only)
- Inter-node BW: stagnant at 400 GB/s

Mitigation strategies:
- Sparse computation (MoE)
- Low-precision communication (FP8/FP4)
- Larger NVLink domains (GB200 NVL72)
- Aggressive compute-communication overlap

### 7.1 Blackwell's Inference Revolution

NVIDIA's claimed 30x inference improvement decomposition:
- 4x from FP4 compute (vs. H100 FP8)
- 2.4x from HBM bandwidth (8 TB/s vs. 3.35 TB/s)
- 2.5x from L2 cache expansion (126 MB vs. 50 MB)
- 1.25x from TMA + 2nd-gen Transformer Engine
- Total: 4 × 2.4 × 2.5 × 1.25 = 30x (best-case: FP4 quantized + small-batch inference)

For training, actual improvement is only 2–3x due to communication wall constraints.

---

## 8. Summary

Key insights:

1. **Communication wall is the new bottleneck**: Critical batch size grows from H100's 2,500 to B200's 5,625 tokens/GPU
2. **Tensor Core evolution drives performance**: Hopper TMA reduces latency 40%; Blackwell FP4 enables up to 30x inference improvement (best case)
3. **Scale-up renaissance**: GB200 NVL72 expands NVLink domain to 72 GPUs, enabling TP-72
4. **Engineering capability is decisive**: Hardware peak ≠ actual performance (MFU typically 42–55%); CUDA kernel optimization, memory management, and automatic parallelism search are core competencies

Future evolution depends on: algorithmic innovation (efficient attention, better quantization), system optimization (communication-computation overlap, intelligent parallelism), and architecture innovation (optical interconnects, compute-in-memory, specialized AI chips).
