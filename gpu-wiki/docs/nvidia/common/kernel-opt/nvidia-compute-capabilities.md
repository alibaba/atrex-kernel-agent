# NVIDIA Compute Capability Reference Table

Different compute capabilities (CC) correspond to different hardware specifications and feature support. You must reference the target CC when selecting optimization strategies.

## SM and Thread Specifications

| Specification | CC 7.5 | CC 8.0 | CC 8.6 | CC 8.9 | CC 9.0 | CC 10.x | CC 12.x |
|------|--------|--------|--------|--------|--------|---------|---------|
| Max Resident Threads/SM | 1024 | 2048 | 1536 | 1536 | 2048 | 1536 | 1536 |
| Max Resident Blocks/SM | 16 | 32 | 16 | 16 | 32 | 24 | 24 |
| Max Resident Warps/SM | 32 | 64 | 48 | 48 | 64 | 48 | 48 |
| Max Threads/Block | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 |
| Warp Size | 32 | 32 | 32 | 32 | 32 | 32 | 32 |

### Representative GPU Models

| CC | Architecture | Representative GPUs |
|----|------|----------|
| 7.5 | Turing | T4, RTX 2080 |
| 8.0 | Ampere | A100, A30 |
| 8.6 | Ampere | A40, RTX 3090 |
| 8.9 | Ada Lovelace | L40, RTX 4090 |
| 9.0 | Hopper | H100, H200, H20 |
| 10.x | Blackwell | B100, B200 |
| 12.x | Rubin (expected) | TBD |

## Memory Specifications

| Specification | CC 7.5 | CC 8.0 | CC 8.6 | CC 8.9 | CC 9.0 | CC 10.x |
|------|--------|--------|--------|--------|--------|---------|
| 32-bit Registers/SM | 64K | 64K | 64K | 64K | 64K | 64K |
| Max Registers/Thread | 255 | 255 | 255 | 255 | 255 | 255 |
| Max Shared Memory/SM | 64 KB | 164 KB | 100 KB | 100 KB | 228 KB | 228 KB |
| Max Shared Memory/Block | 64 KB | 163 KB | 99 KB | 99 KB | 227 KB | 227 KB |
| Unified Data Cache | 96 KB | 192 KB | 128 KB | 128 KB | 256 KB | 256 KB |
| Local Memory/Thread | 512 KB | 512 KB | 512 KB | 512 KB | 512 KB | 512 KB |
| Constant Memory | 64 KB | 64 KB | 64 KB | 64 KB | 64 KB | 64 KB |
| Shared Memory Banks | 32 | 32 | 32 | 32 | 32 | 32 |

## FP32:FP64 Throughput Ratio

| CC 7.5 | CC 8.0 | CC 8.6 | CC 8.9 | CC 9.0 | CC 10.x |
|--------|--------|--------|--------|--------|---------|
| 32:1 | **2:1** | 64:1 | 64:1 | **2:1** | 64:1 |

- CC 8.0 (A100) and CC 9.0 (H100) have full double-precision support (FP64 = FP32/2)
- Other CCs have extremely low double-precision throughput (FP64 = FP32/32 or /64); avoid using `double` on these GPUs

## Tensor Core Support Matrix

| Data Type | CC 7.5 | CC 8.0 | CC 8.6 | CC 8.9 | CC 9.0 | CC 10.x |
|----------|--------|--------|--------|--------|--------|---------|
| FP64 | - | ✓ | - | - | ✓ | - |
| TF32 | - | ✓ | ✓ | ✓ | ✓ | ✓ |
| BF16 | - | ✓ | ✓ | ✓ | ✓ | ✓ |
| FP16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| FP8 | - | - | - | ✓ | ✓ | ✓ |
| FP6 | - | - | - | - | - | ✓ |
| FP4 | - | - | - | - | - | ✓ |
| INT8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Key Features Introduced by CC

### CC 8.0+ (Ampere)

- **L2 Cache Persistence Control**: `cudaAccessPropertyPersisting` pins hot data in L2
- **Hardware `memcpy_async`**: asynchronous copy from global memory to shared memory, bypassing registers
- **Async Barrier (split arrive/wait barrier)**
- **BF16 Tensor Core Support**

### CC 9.0 (Hopper)

- **Thread Block Cluster**: cooperative groups spanning multiple blocks (distributed shared memory)
- **TMA (Tensor Memory Accelerator)**: hardware-accelerated tensor data movement
- **Distributed Shared Memory**: direct access to other blocks' shared memory within a cluster
- **wgmma Instruction**: warp group level matrix multiply (feeds data directly from shared memory)
- **FP8 Tensor Core**

### CC 10.x (Blackwell)

- **FP4/FP6 Tensor Core**
- **128-bit Atomic Operations**
- **Warp reduce function** (hardware native)
- **128-bit Floating-Point Operations**
