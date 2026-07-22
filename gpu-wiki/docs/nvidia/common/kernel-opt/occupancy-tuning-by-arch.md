# Occupancy Tuning Differences Across Architectures

| Parameter | CC 7.5 | CC 8.0 | CC 9.0 | CC 10.0 |
|------|--------|--------|--------|---------|
| Target occupancy | > 50% | > 50% | Flexible | Flexible |
| Recommended block size | 128-256 | 128-256 | 128-256 | 128-256 |
| Shared memory limit/block | 64 KB | 163 KB | 227 KB | 227 KB |
| 32-bit registers/SM | 64K | 64K | 64K | 64K |

## Hopper (CC 9.0) Special Considerations

- wgmma instructions require warp group (4 warps = 128 threads) coordination
- Block size typically needs to be a multiple of 128 (rather than the traditional 32)
- Larger shared memory allows larger tiles → higher data reuse → may be faster even with reduced occupancy

## Blackwell Data Center (CC 10.0) Special Considerations

- Maximum 32 blocks, 64 warps, and 2048 threads per SM
- FP4/FP6 Tensor Core available → more flexible mixed-precision strategies

## Blackwell GeForce/Workstation (CC 12.0) Caveat

- Maximum 48 warps and 1536 threads per SM.
- CUDA 13.1 Programming Guide Table 27 lists 24 resident blocks/SM, while the
  Blackwell Tuning Guide lists 32. Query
  `cudaDevAttrMaxBlocksPerMultiprocessor` on the deployed GPU.
- Shared memory is 128 KB per SM and 99 KB per block, not the CC 10.0
  228/227 KB limits.

## Related Documentation

- [NVIDIA Architecture-Specific Optimization Techniques (Index)](nvidia-arch-specific-optimization.md)
- [NVIDIA Compute Capability Reference](nvidia-compute-capabilities.md) — Thread/register/shared memory limits per architecture
- [GPU Execution Model and Thread Optimization](../../../generic/ref-docs/gpu-execution-model.md) — General occupancy optimization theory
- [CUDA 13.1 Technical Specifications](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability)
- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#occupancy)
