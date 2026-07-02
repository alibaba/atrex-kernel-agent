# Occupancy Tuning Differences Across Architectures

| Parameter | CC 7.5 | CC 8.0 | CC 9.0 | CC 10.x |

**Last updated**: 2026-06-30

|------|--------|--------|--------|---------|
| Target occupancy | > 50% | > 50% | Flexible | Flexible |
| Recommended block size | 128-256 | 128-256 | 128-256 | 128-256 |
| Shared memory limit/block | 64 KB | 163 KB | 227 KB | 227 KB |
| Available registers/block (100% occ) | 2K | 1K | 1K | ~1.3K |

## Hopper (CC 9.0) Special Considerations

- wgmma instructions require warp group (4 warps = 128 threads) coordination
- Block size typically needs to be a multiple of 128 (rather than the traditional 32)
- Larger shared memory allows larger tiles → higher data reuse → may be faster even with reduced occupancy

## Blackwell (CC 10.x) Special Considerations

- Maximum 24 blocks per SM (higher than CC 8.6's 16, lower than CC 9.0's 32)
- Maximum 1536 threads per SM (lower than CC 9.0's 2048)
- FP4/FP6 Tensor Core available → more flexible mixed-precision strategies

## Related

- [NVIDIA Architecture-Specific Optimization Techniques (Index)](nvidia-arch-specific-optimization.md)
- [NVIDIA Compute Capability Reference](nvidia-compute-capabilities.md) — Thread/register/shared memory limits per architecture
- [GPU Execution Model and Thread Optimization](../../generic/gpu-execution-model.md) — General occupancy optimization theory
