# Comprehensive Guide to NVIDIA Blackwell Architecture

An overview of the NVIDIA Blackwell and Blackwell Ultra platforms, covering key architectural innovations, product configurations, and scaling law implications.


**Last updated**: 2026-06-30

---

## 1. NVIDIA Blackwell and Blackwell Ultra Overview

NVIDIA Blackwell and Blackwell Ultra are designed to address the escalating complexity of artificial intelligence — including growing model sizes and inference demands — incorporating multiple groundbreaking innovations.

---

## 2. Eight Architectural Innovations

### 2.1 GPU Design

- **208 billion transistors** — over 2.5x that of the Hopper GPU
- **TSMC 4NP process** (a 4nm high-performance variant, essentially part of the 5nm family)
- Single-chip compute capability up to **20 petaFLOPS**
- Two dies connected via **10 TB/s NV-HBI** forming a fully coherent chip
- Blackwell Ultra GB300 NVL72 delivers 1.5x AI performance over GB200 NVL72
- Compared to Hopper systems: inference productivity +50x, inference speed +35x, energy efficiency +30x, cost per token reduced 25x

### 2.2 5th Generation Tensor Core

- Supports **FP4** and other new numeric formats, including community-defined microscaling (OCP) formats
- Delivers massive throughput and energy efficiency improvements over standard FP/INT/FMA operations

### 2.3 2nd Generation Transformer Engine

- Employs **Micro-Tensor Scaling** for dynamic range management
- Optimizes inference performance and accuracy with FP4 AI computation
- **Doubles both HBM parameter bandwidth and single-GPU model capacity**
- Co-optimized with Dynamo, TensorRT-LLM, and NeMo frameworks (4-bit quantization, expert-parallel custom kernels, disaggregation techniques)
- Training: synergizes with NeMo + Megatron-Core expert parallelism + 5th-gen NVLink

### 2.4 Attention Layer Acceleration (Blackwell Ultra Exclusive)

- New instructions optimize performance for long input sequences
- **2x attention layer compute speed** compared to base Blackwell GPU
- Reduces latency to improve AI inference decision-making speed

### 2.5 Confidential Computing

- First GPU supporting **TEE-I/O**
- End-to-end protection via NVLink
- Throughput nearly identical to unencrypted mode

### 2.6 5th Generation NVLink

- Single link: 50 GB/s bidirectional
- Per GPU: **18 links = 1.8 TB/s total bandwidth (900 GB/s per direction)**
- Over 14x PCIe Gen5 bandwidth
- NVL72 domain aggregate bandwidth: **130 TB/s**
- SHARP FP8 support provides 4x bandwidth efficiency improvement

### 2.7 Decompression Engine

- Decompression rate: **800 GB/s**
- Paired with GB200 single-GPU **8 TB/s HBM3e**
- Supports LZ4, Snappy, Deflate
- TPC-H Q4 query: Blackwell is 18x CPU, 6x H100

### 2.8 RAS Engine

- Dedicated RAS engine with AI-driven predictive management
- Continuously monitors thousands of data points
- Activates spare capacity upon detecting component degradation, minimizing performance loss

---

## 3. GB300 NVL72 / GB200 NVL72

**GB300 NVL72**: 36 Grace CPUs + 72 Blackwell Ultra GPUs, liquid-cooled rack, 37 TB fast memory per rack, >1 PFlops FP4 compute.

**Key performance gains (vs. H100)**:
- Inference productivity (throughput): **50x** (Dynamo dynamic orchestration, fixed 1 MW power)
- DeepSeek-R1 inference: **35x**
- Energy efficiency: **30x**
- TCO reduction: **25x**

Target metrics: 32K input sequence / 8K output sequence / 200 TPS per user / 2-second time-to-first-token.

**GB200 NVL72**: On GPT-MoE 1.8T and other ultra-large models, 30x speed improvement over H100, 25x TCO and energy reduction; 4x training performance over Hopper, 9x rack space reduction.

---

## 4. Real-Time Video Generation

Current state-of-the-art LLM context windows handle only 128K tokens, while 5 seconds of video requires 4 million tokens. The Blackwell Ultra platform supports real-time video generation with NVIDIA Cosmos foundation models, delivering 30x performance over Hopper. Cosmos-1.0-Diffusion-7B-Video2World achieves 720p at 60 FPS.

---

## 5. HGX B300 / HGX B200

- **HGX B300**: 7x AI compute over Hopper, 2 TB+ HBM3e, integrated ConnectX-8 SuperNIC
- **HGX B200**: x86 platform, 144 PFlops AI performance, 15x over HGX H100, 12x TCO reduction, up to 1000 W per GPU

---

## 6. Three Scaling Laws

1. **Pre-training scaling**: More training data, parameters, and compute → 50-million-fold compute demand increase within five years
2. **Post-training scaling**: Teaching models to "think" requires ~30x pre-training compute
3. **Test-time scaling / Deep thinking**: Agent inference "thinking" requires ~100x traditional inference compute

Blackwell is positioned as a "once-in-a-decade" platform capable of effectively supporting all three scaling laws.


## Related

- [GPGPU Architecture: Blackwell Instruction Analysis](blackwell-architecture-instruction-analysis.md)
- [Blackwell GPGPU Architecture New Features Overview](blackwell-gpgpu-new-features-overview.md)
- [NVIDIA Blackwell Tensor Core Analysis (Part 2): B300](blackwell-tensor-core-analysis-b300.md)
- [NVIDIA Blackwell Tensor Core Analysis (Part 1)](blackwell-tensor-core-analysis-part1.md)
- [Blackwell Ultra (B300): NVIDIA AI Chip Evolution and Roadmap](blackwell-ultra-b300-chip-evolution.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
