# Blackwell Ultra (B300): NVIDIA AI Chip Evolution and Roadmap

A first-principles analysis of the B300 (Blackwell Ultra) GPU, with projections for the Rubin and Rubin Next platforms through 2030.


**Last updated**: 2026-06-30

---

## 1. Introduction

NVIDIA first announced the Rubin platform at Computex 2024 on June 2, 2024, with a planned 2026 launch. This document provides a first-principles analysis of B300 (Blackwell Ultra) and extrapolates the Rubin and Rubin Next roadmap.

Through 2030, AI compute chips are expected to maintain a doubling cadence every two years across compute, memory, and interconnect. Each platform generation is projected to deliver: compute +2.25x, memory capacity and bandwidth +1.5–2x, interconnect bandwidth +2x, with power growth <1.5x.

---

## 2. Process Technology Roadmap

By 2030, semiconductor technology will enter the 1nm era, with single-die transistor counts reaching an estimated 200 billion. Blackwell's 4NP process yields 104 billion transistors per die. Each process generation provides ~30% density improvement; a full density doubling requires 3 generations. Projected mapping:

- Blackwell: 4NP
- **Rubin: 3nm**
- **Rubin Next: 2nm**
- 1nm: the generation after Rubin Next

Note: SMIC's DUV multi-patterning approach will plateau at 7nm/5nm equivalent levels; domestic AI chips must compensate through systemic innovation beyond process scaling.

---

## 3. Advanced Packaging

TSMC CoWoS roadmap:
- 2023: 3.3-reticle, 8 HBM (Blackwell)
- 2026: 5.5-reticle, 12 HBM (Rubin / Rubin Ultra)
- 2027: >8-reticle, 12 HBM (Rubin Next, possibly 3.5D)
- 2026+: Silicon-based SoW (System-on-Wafer), >40-reticle, 60+ HBM

Rubin Next more likely uses relatively mature 3.5D packaging; SoW may target the generation after Rubin Next.

TSMC CoWoS-R (organic interposer) prototype: 97×95 mm² package, 5.5x CoWoS-R interposer, integrating 4 SoCs + 12 HBM stacks, routing density >1100 lines/mm, supporting 64-channel 32 Gbps UCIe. This serves as a **reference design for the Rubin GPU package**.

---

## 4. HBM Roadmap

- HBM4: mass production 2025 → Rubin
- HBM4E: 2027 → Rubin Next
- Samsung developing custom HBM4 for CSPs, targeting 2025 mass production
- SK Hynix: 48 GB 16-layer HBM3E (announced at November 2025 SK AI Summit)
- Jensen Huang requested SK Hynix accelerate HBM4 supply by 6 months

HBM4 pin count exceeds 2000 (double HBM3). From HBM3 to HBM4, the data bus widens from 1024 to 2048 bits. Unchanged bump pitch causes PHY area expansion that encroaches on compute die area — driving the need for customized cHBM solutions.

---

## 5. Blackwell Ultra (GB300 & B300) Analysis

B300 is a full redesign on 4NP process, delivering **+50% overall FLOPS over B200**. Performance gains come from:

1. **Increased power**: GB300 / B300 HGX TDP rises to 1.4 kW / 1.2 kW (GB200 series: 1.2 kW / 1 kW)
2. **Architecture optimization**: CPU↔GPU dynamic power allocation (Power Sloshing)
3. **Memory upgrade**: HBM3E 8-layer → 12-layer (12-Hi HBM3E), per-GPU **192 GB → 288 GB**, bandwidth maintains 8 TB/s

### 5.1 Core Value of Memory Upgrades

H100 → H200 case study:
- **43% interactivity improvement**: bandwidth 3.35 → 4.8 TB/s
- **~3x cost reduction**: capacity 80 → 141 GB HBM3e enables higher batch sizes, 3x tokens/second

Memory optimization is a key factor in improving inference efficiency and economics. A single mid-generation memory upgrade achieves 3x hardware performance improvement — faster than Moore's Law or Huang's Law.

### 5.2 NVL72 → NVL288

NVL72 inference advantages: low latency, long reasoning chains, cost optimization, accuracy improvements. For long reasoning chains, NVL72 economics improve by **over 10x** — it is the only solution supporting 100K+ token high-batch inference.

---

## 6. GB300 Supply Chain Changes

- GB200: complete Bianca board (GPU + Grace CPU + 512 GB LPDDR5X + VRM) + switch tray + copper backplane
- GB300: only SXM Puck module with B300 GPU + BGA Grace CPU + Axiado HMC; remaining components sourced by OEM
- Second-tier memory: LPDDR5X → **LPCAMM pluggable modules**
- Introduction of **800G ConnectX-8** (48 PCIe lanes + SpectrumX; GB200 removed CX-8)

NVIDIA provides reference designs while delegating non-core components to the supply chain, retaining switch tray + copper backplane as differentiated competitive advantages (NVLink & NVSwitch networking is the core of NVL72).

---

## 7. Rubin / Rubin Ultra Architecture Projections

Evolution path: Hopper (single die) → Blackwell (dual die) → **Rubin (quad die)**. Three potential approaches:
- Option 1: 2.5D, 4 compute dies + 8 HBM
- Option 2: 2.5D, 4 compute dies + 12 HBM
- Option 3: 3D, 3 base dies + 3 (or 12) compute dies + 12 HBM

All fit within a 5.5x reticle area. Compute die area ×1.92 + 30% process improvement → throughput ×2.5. Memory bandwidth and capacity maintain ×1.5 scaling.

Rubin Ultra's 11×13 mm² HBM modules push to 6.4x reticle area, potentially requiring custom HBM size reduction or compute die shrinkage to stay within the 5.5x roadmap.

---

## 8. Rubin NVL288 Projections

Integrating 4 NVL72 units into a single rack yields 4x compute density. Total rack power and cooling reaches the 1 MW range. The architecture may employ **orthogonal compute and switch boards**: 72 GPUs distributed across 36 compute boards, with each logical partition equivalent to an NVL36 — conceptually rotating the existing NVL36 switch board to the back, orthogonal to compute boards.

**Blackwell → Rubin (2x) + NVL72 → NVL288 (4x) = 8x performance growth**. Combined with software and algorithmic optimization, over 10x total performance improvement is achievable.


## Related

- [Comprehensive Guide to NVIDIA Blackwell Architecture](blackwell-architecture-comprehensive-guide.md)
- [GPGPU Architecture: Blackwell Instruction Analysis](blackwell-architecture-instruction-analysis.md)
- [Blackwell GPGPU Architecture New Features Overview](blackwell-gpgpu-new-features-overview.md)
- [NVIDIA Blackwell Tensor Core Analysis (Part 2): B300](blackwell-tensor-core-analysis-b300.md)
- [NVIDIA Blackwell Tensor Core Analysis (Part 1)](blackwell-tensor-core-analysis-part1.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
