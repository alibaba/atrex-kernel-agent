# Composable Kernel (CK) Programming Model (MI308X)


**Last updated**: 2026-06-30

## TensorDescriptor Core Concepts

CK uses a transformation tree to manage multidimensional data layout:

```cpp
// Create a base tensor descriptor
auto desc = make_naive_tensor_descriptor(
    make_tuple(M, K),    // shape: (256, 128)
    make_tuple(K, 1)     // strides: row-major
);

// Apply transformations
auto transformed = transform_tensor_descriptor(
    desc,
    make_tuple(unmerge, passthrough),
    make_tuple(Sequence<0>{}, Sequence<1>{}),       // lower dim ids
    make_tuple(Sequence<0, 1>{}, Sequence<2>{})     // upper dim ids
);
// Result shape: (4, 64, 128)
```

## Four Core Transformations

| Transformation | Function | Example |
|------|------|------|
| **Embed** | Multidimensional coordinates → Linear address | Implicit insertion |
| **Unmerge** | Split one dimension | (256) → (4, 64) |
| **Merge** | Merge multiple dimensions | (64, 128) → (8192) |
| **PassThrough** | Identity transformation | Keep dimensions unchanged |

## Vectorized Memory Access

```cpp
// 16 float registers, can be viewed as different shapes
vector_type<float, 16> thread_local_buf;
auto& a = thread_local_buf.AsType<d4_t>();   // 4 float4 vectors
auto& b = thread_local_buf.AsType<d1_t>();   // 16 individual floats

// Vectorized read
auto buf = make_dynamic_buffer<AddressSpaceEnum::Global>(src, desc.GetElementSpaceSize());
a(Number<i>{}) = buf.Get<d4_t>(desc.CalculateOffset({x+i, y}), true);
```

## Performance (2560x32 Matrix Transpose)

- PyTorch: 8.4 us
- CK: **5.82 us** (+44.3% throughput improvement)

Design highlights: 64 threads/block × 80 blocks (matching MI308X's 80 CUs), no LDS usage, all completed in VGPRs.

---

## Related

- [MI308X (CDNA3) Kernel Optimization Practices (Index)](cdna3-mi308x-kernel-practices.md) -- Index of the case study collection this document belongs to
- [AMD GPU Kernel Optimization Frameworks Overview](../amd-kernel-optimization-frameworks.md) -- CK's position within the AMD optimization framework
- [AMD MFMA Matrix Core Programming Guide](../amd-mfma-matrix-cores.md) -- MFMA instructions used at the CK low level
