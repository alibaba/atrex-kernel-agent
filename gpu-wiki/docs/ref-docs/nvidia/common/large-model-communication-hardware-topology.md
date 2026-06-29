# Large Model Communication: Hardware Topology (MNNVL/NVL72)

Fundamentals of communication hardware topology for large model training and inference, covering NVLink, NVL72, MNNVL, and collective operation patterns.

---

## 1. Overview

This document covers the communication hardware topology underlying large-scale model training, focusing on NVIDIA's interconnect hierarchy from intra-node NVLink to multi-node NVLink (MNNVL) and NVL72 rack-scale configurations.

---

## 2. Key Topology Concepts

- **NVLink**: High-bandwidth, low-latency GPU-to-GPU interconnect within a node
- **NVL72**: 72-GPU rack-scale configuration using NVLink 5 and NVSwitch, providing 1.8 TB/s per-GPU bidirectional bandwidth with single-hop any-to-any connectivity
- **MNNVL (Multi-Node NVLink)**: Extension of NVLink across multiple nodes, enabling unified memory addressing across racks
- **NVSwitch**: Custom silicon providing full-bisection bandwidth between all GPUs in an NVL72 domain

---

## 3. Communication Hierarchy

| Level | Interconnect | Bandwidth | Latency |
|---|---|---|---|
| Intra-SM | Registers / SMEM | TB/s+ | ~ns |
| Intra-GPU | L2 / HBM | 8 TB/s (B200) | ~100 ns |
| Intra-NVL72 | NVLink 5 + NVSwitch | 1.8 TB/s/GPU | ~µs |
| Cross-rack | InfiniBand / RoCE | 400-800 Gb/s | ~10 µs |

---

## 4. Collective Operations

- **All-Reduce**: Aggregate gradients across all GPUs; ring-based or tree-based algorithms
- **All-to-All**: Token routing in MoE expert parallelism
- **All-Gather / Reduce-Scatter**: FSDP parameter reconstruction and gradient sharding
- **Point-to-Point**: Pipeline parallelism activation transfers

---

## 5. Implications for Parallelism Strategy

The NVL72 topology fundamentally changes optimal parallelism strategies:
- Traditional TP limited to 8 GPUs can now extend to 72 within the NVLink domain
- Pipeline parallelism becomes less relevant when all GPUs have uniform high-bandwidth connectivity
- Expert parallelism (EP) benefits most from the all-to-all bandwidth within NVL72
