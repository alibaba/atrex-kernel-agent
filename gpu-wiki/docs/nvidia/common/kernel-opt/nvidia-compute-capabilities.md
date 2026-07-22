# NVIDIA Compute Capability Reference Table

Different compute capabilities (CC) correspond to different hardware specifications and feature support. You must reference the target CC when selecting optimization strategies.

## SM and Thread Specifications

| Specification | CC 7.5 | CC 8.0 | CC 8.6 | CC 8.9 | CC 9.0 | CC 10.0 | CC 12.0 |
|------|--------|--------|--------|--------|--------|---------|---------|
| Max Resident Threads/SM | 1024 | 2048 | 1536 | 1536 | 2048 | 2048 | 1536 |
| Max Resident Blocks/SM | 16 | 32 | 16 | 24 | 32 | 32 | 24* |
| Max Resident Warps/SM | 32 | 64 | 48 | 48 | 64 | 64 | 48 |
| Max Threads/Block | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 |
| Warp Size | 32 | 32 | 32 | 32 | 32 | 32 | 32 |

> **CC 12.0 block-limit caveat**: CUDA 13.1 Programming Guide Table 27
> lists 24 resident blocks/SM, while the current Blackwell Tuning Guide says
> 32. Treat 24 as the programming-guide value and query
> `cudaDevAttrMaxBlocksPerMultiprocessor` on the deployed GPU before relying on
> the limit.

### Representative GPU Models

| CC | Architecture | Representative GPUs |
|----|------|----------|
| 7.5 | Turing | T4, RTX 2080 |
| 8.0 | Ampere | A100, A30 |
| 8.6 | Ampere | A40, RTX 3090 |
| 8.9 | Ada Lovelace | L40, RTX 4090 |
| 9.0 | Hopper | H100, H200, H20 |
| 10.0 | Blackwell data center | B100, B200, GB200 |
| 10.3 | Blackwell Ultra | B300, GB300 |
| 12.0 | Blackwell GeForce/workstation | RTX PRO 5000/6000 Blackwell, GeForce RTX 50 series |

## Memory Specifications

| Specification | CC 7.5 | CC 8.0 | CC 8.6 | CC 8.9 | CC 9.0 | CC 10.0 | CC 12.0 |
|------|--------|--------|--------|--------|--------|---------|---------|
| 32-bit Registers/SM | 64K | 64K | 64K | 64K | 64K | 64K | 64K |
| Max Registers/Thread | 255 | 255 | 255 | 255 | 255 | 255 | 255 |
| Max Shared Memory/SM | 64 KB | 164 KB | 100 KB | 100 KB | 228 KB | 228 KB | 128 KB |
| Max Shared Memory/Block | 64 KB | 163 KB | 99 KB | 99 KB | 227 KB | 227 KB | 99 KB |
| Unified Data/Shared Memory Pool | 96 KB | 192 KB | 128 KB | 128 KB | 256 KB | 256 KB | 128 KB |
| Local Memory/Thread | 512 KB | 512 KB | 512 KB | 512 KB | 512 KB | 512 KB | 512 KB |
| Constant Memory | 64 KB | 64 KB | 64 KB | 64 KB | 64 KB | 64 KB | 64 KB |
| Shared Memory Banks | 32 | 32 | 32 | 32 | 32 | 32 | 32 |

## FP32:FP64 Throughput Ratio

| CC 7.5 | CC 8.0 | CC 8.6 | CC 8.9 | CC 9.0 | CC 10.0 | CC 12.0 |
|--------|--------|--------|--------|--------|---------|---------|
| 32:1 | **2:1** | 64:1 | 64:1 | **2:1** | **~2:1** | 64:1 |

- CC 8.0 (A100), CC 9.0 (H100), and CC 10.0 B200-class GPUs have strong double-precision support.
- Do not generalize the CC 10.0 B200 ratio to CC 10.3 B300: the official HGX table gives about 75 FP32 versus 1.25 FP64 TFLOPS per B300 GPU.
- CC 12.0 client/workstation GPUs expose FP64 primarily for compatibility; avoid using `double` for throughput-sensitive kernels.

## Tensor Core Support Matrix

| Data Type | CC 7.5 | CC 8.0 | CC 8.6 | CC 8.9 | CC 9.0 | CC 10.0 |
|----------|--------|--------|--------|--------|--------|---------|
| FP64 | - | ✓ | - | - | ✓ | ✓ |
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

### CC 10.0 (Blackwell Data Center)

- **FP4/FP6 Tensor Core**
- **128-bit Atomic Operations**
- **Warp reduce function** (hardware native)
- **128-bit Floating-Point Operations**

## Official Sources

- [CUDA 13.1 Programming Guide, Technical Specifications per Compute Capability](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability)
- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#occupancy)
- [CUDA GPUs by Compute Capability](https://developer.nvidia.com/cuda-gpus)
- [NVIDIA HGX specifications](https://www.nvidia.com/en-us/data-center/hgx/)
