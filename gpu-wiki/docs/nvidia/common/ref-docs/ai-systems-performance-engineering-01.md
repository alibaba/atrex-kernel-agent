# AI Systems Performance Engineering (Part 1)

Introduction and overview of AI systems performance engineering, covering the role definition, hardware landscape (GB200 NVL72), and the path toward 100-trillion-parameter models.

---

## 1. Chapter Overview

In late 2024, Chinese startup DeepSeek.AI trained a frontier LLM without access to the latest NVIDIA GPUs. Constrained by export restrictions to H800 hardware, they maximized performance through custom kernels and model distillation. The resulting DeepSeek-R1 approached the reasoning capabilities of frontier models trained on the most advanced NVIDIA silicon.

> This case underscores that **skilled AI systems performance engineers can extract maximum value from available hardware — regardless of constraints**.

---

## 2. The AI Systems Performance Engineer

### 2.1 Role Definition

A specialized role focused on optimizing AI model performance and underlying systems, ensuring AI training and inference pipelines are fast, efficient, reliable, and highly available. The larger the scale, the more critical this role becomes.

### 2.2 Core Competencies

- Deep understanding of hardware architecture, software optimization, system-level integration, low-level OS, memory hierarchies, networking
- Multi-language proficiency (Python, C++) + frameworks (PyTorch, Triton, CUDA)
- Daily work: inspecting low-level GPU kernel efficiency, optimizing OS thread scheduling, analyzing memory access patterns, improving network throughput, debugging distributed training algorithms

### 2.3 Key Responsibilities

1. **Benchmarking and Profiling**: Latency/throughput/memory metrics + nsys/ncu/PyTorch profiler + automated performance tests for early regression detection
2. **Debugging and Optimization**: Sub-optimal CUDA kernels, communication overhead, load imbalance → NCCL/NIXL/all-reduce optimization / parallelism strategy tuning
3. **Scaling Distributed Training and Inference**: 8 GPUs → 80,000 GPUs; DP/TP/PP/EP
4. **Efficient Resource Management**: CPU core pinning, GPU memory orchestration, MIG partitioning, thread pinning to reduce context switching
5. **Cross-team Collaboration**: Researchers, data scientists, application developers, infrastructure teams

### 2.4 Transparency and Reproducibility

DeepSeek's **Open Source Week (February 2025)** released the full stack:

| Project | Purpose |
| --- | --- |
| FlashMLA | Optimized attention kernel (CUDA C++) |
| DeepGEMM | FP8-optimized matrix multiplication library |
| DeepEP | Efficient communication library for MoE models |
| EPLB | Expert parallel load balancer |
| DualPipe | Bidirectional pipeline parallelism algorithm |
| 3FS | High-performance distributed file system |

MLPerf v5.0 (2025):

- **Training**: Blackwell GB200 NVL72 achieves **+2.6x** training throughput over Hopper
- **Inference**: Blackwell NVL72 achieves **+3.4x** inference throughput over Hopper

---

## 3. DeepSeek: Scaling to ~680B Parameters Under Hardware Constraints

| Feature | H100 | H800 |
| --- | --- | --- |
| NVLink bandwidth | ~900 GB/s | ~400 GB/s |
| Memory bandwidth | 3.35 TB/s | Restricted |
| HBM capacity | Similar | Similar |

**DeepSeek-V3 MoE**: ~680B total parameters, only ~37B activated per token, 1 shared + 8 routed experts (256 total), 9 active experts per token.

**Innovative Solutions**:

- DualPipe parallelism algorithm carefully overlaps computation and communication, masking H800 NVLink weakness
- Custom CUDA kernels bypass default NCCL collective operations, coordinating data transfer with computation

Result: Training completed at a fraction of the GPU-hours and cost of comparable frontier models, approaching GPT-4 on standardized benchmarks, matching or slightly exceeding on some tests.

---

## 4. Toward 100-Trillion-Parameter Models

| Estimate | Value |
| --- | --- |
| Dense 100T training ~29T tokens | ~1.2 × 10²⁹ FLOPS |
| 16-bit model memory | ~182 TB |
| Single B200 memory | 180 GB |
| Memory multiple | 1000x |
| Estimated GPU count | ~1000 B200 or 700 B300 |

MoE advantage: FLOPS per token remains constant. DeepSeek-V3/R1 (680B total, 37B activated), Google Switch Transformer (1.6T MoE, 7x faster training).

---

## 5. NVIDIA "AI Supercomputer in a Rack"

### Grace Blackwell GB200/GB300 NVL72

| Metric | Value |
| --- | --- |
| Grace Blackwell Superchips | 36 |
| Blackwell GPUs | 72 |
| Grace CPUs (72-core) | 36 |
| FP4 theoretical performance | ~1.44 EFLOPS |
| FP8 theoretical performance | ~720 PFLOPS |
| HBM3e total capacity | ~13.5 TB |
| Total memory (including CPU) | ~30 TB |
| NVLink aggregate bandwidth | ~130 TB/s |
| Rack power | 120-132 kW |

### NVLink 5 / NVSwitch

- Per-GPU bidirectional 1.8 TB/s (18 links × 100 GB/s bidirectional) = 2x NVLink 4
- 9 NVSwitch trays, 2 NVSwitch chips per tray = **18 NVSwitch chips**, any GPU reachable in single hop

Cloud availability: AWS / GCP / Azure / CoreWeave / Lambda Labs one-click provisioning.

---

## 6. Mechanical Sympathy

Derived from F1 champion Jackie Stewart's deep understanding of racing car mechanics. Martin Thompson introduced the concept to software engineering. In AI = **co-designing algorithms with hardware capabilities to maximize performance**.

- **FlashAttention**: Tiling + minimizing HBM reads/writes → 2-4x long-sequence speedup
- **DeepSeek MLA**: Restructured attention leveraging NVIDIA memory hierarchy + Tensor Cores, surpassing FlashAttention on H800
- **Transformer Engine**: FP8 / FP4 + microscaling + dedicated exponent units to accelerate softmax

> Co-design loop: New hardware → New algorithms → New hardware → ...

---

## 7. Goodput: Effective Throughput

> The amount of **useful work** completed per unit time, excluding contributions that do not directly advance training/inference.

Example: 8 GPUs processing 100K tokens in 10 seconds = 10K token/s; per-GPU peak 1500 token/s × 8 = 12K → efficiency **83.3%**.

Meta research: 100% cluster utilization ≠ 70-75% effective compute. The gap is consumed by communication latency, suboptimal parallelism, data delays, and failure recovery.

Common bottlenecks and strategies:
- Data loading latency → caching / async prefetch
- Gradient synchronization waiting → overlap computation and communication
- Network congestion → optimize topology and routing

---

## 8. Book Roadmap

| Chapter | Topic |
| --- | --- |
| 2 | NVIDIA AI hardware deep dive (GB200/GB300 NVL72, Grace Blackwell Superchip, NVLink networks) |
| 3-5 | OS / network / storage optimization (CPU + memory pinning, Docker/K8s, GPU network I/O) |
| 6-12 | CUDA programming fundamentals and kernel optimization, FlashAttention, DeepSeek MLA, Transformer attention mechanisms |
| 13-14 | PyTorch / Triton optimization, DDP/FSDP/TP/PP/CP/MoE, activation checkpointing / sharded optimizers / offloading |
| 15-19 | Disaggregated prefill/decode, vLLM/SGLang/TensorRT-LLM, NVIDIA Dynamo, NIXL KV cache, speculative decoding, model compression |
| 20 | AI-assisted kernel / system performance optimization, self-improving AI systems |
| Appendix | Performance optimization and cost-saving checklist |

---

## 9. Key Takeaways

1. **Measure goodput beyond raw FLOPS or utilization**: Use Nsight / PyTorch profiler to focus on effective GPU utilization
2. **Skilled engineering > brute-force spending**: DeepSeek trained frontier models on constrained H800 at a fraction of the cost
3. **Order-of-magnitude incremental optimization**: Small percentage efficiency × scale = millions of dollars
4. **Profile-driven tuning**: Let data guide optimization, targeting real bottlenecks
5. **Holistic perspective**: Hardware + software co-design; any layer can become the bottleneck
6. **Stay current with hardware / software / algorithms**: Unified CPU-GPU memory, interconnects, and new precision formats continuously change optimal strategies
