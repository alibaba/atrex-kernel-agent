# L2 Cache Persistence Control (CC 8.0+)

On Ampere and later architectures, you can reserve a portion of the L2 cache for frequently accessed data to prevent it from being evicted.


**Last updated**: 2026-06-30

## Applicable Scenarios

- Hot data (e.g., attention KV cache) that is accessed repeatedly
- Streaming data (e.g., large matrix tiling) that is accessed only once → mark as streaming to avoid cache pollution

## Usage

```c
// 1. Set persistent L2 cache size
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persistSize);

// 2. Create access policy window
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr = (void*)hotDataPtr;
attr.accessPolicyWindow.num_bytes = hotDataSize;
attr.accessPolicyWindow.hitRatio = 1.0f;          // 100% attempt to persist
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

// 3. Apply to stream
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);

// 4. Kernel execution...

// 5. Reset (release L2 for other uses)
attr.accessPolicyWindow.num_bytes = 0;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
cudaCtxResetPersistingL2Cache();
```

## Tuning Recommendations

- `num_bytes` should not exceed the L2 set-aside size, otherwise it will be truncated
- `hitRatio` can be lowered (e.g., to 0.6) to allow some accesses to go through the streaming path
- Policy windows for different streams take effect independently
- Measured performance improvements of ~50% (when hot data fits exactly)

## Related

- [NVIDIA Architecture-Specific Optimization Techniques (Index)](nvidia-arch-specific-optimization.md)
- [GPU Memory Hierarchy and Optimization](../../generic/gpu-memory-hierarchy.md) — General memory hierarchy principles
- [NVIDIA Compute Capability Reference Table](nvidia-compute-capabilities.md) — L2 cache capacity per architecture
