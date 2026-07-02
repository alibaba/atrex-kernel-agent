# Occupancy Optimization


**Last updated**: 2026-06-30

## Relationship Between VGPR and Occupancy

Each EU (SIMD unit) has 512 VGPRs, allocated in groups of 16. Occupancy = number of wavefronts that can run simultaneously.

```
Actual VGPR = ceil(kernel_vgpr / 16) * 16
max_waves_per_eu = floor(512 / Actual VGPR)
```

| VGPR Usage | Max Waves/SIMD |
|------------|-----------------|
| ≤96 | 5+ |
| 97-128 | 4 |
| 129-170 | 3 |
| 171-256 | 2 |
| 257-512 | 1 |

## Tips for Reducing Register Pressure

1. **Set `__launch_bounds__`**: Tell the compiler the actual block size (default assumes 1024)
2. **Defer variable definitions**: Move variable definitions closer to their first use
3. **Avoid `pow(x, 2.0)`**: Replace with `x*x` to reduce registers
4. **Control loop unrolling**: `#pragma unroll` increases register requirements
5. **Use `__restrict__`**: Reduces SGPRs (may slightly increase VGPRs)
6. **Manual spill to LDS**: Store long-lived variables in LDS
7. **`waves_per_eu=N`**: In Triton, hint the compiler to compress VGPRs to achieve target occupancy

## Inspecting Register Usage

```bash
# Compile time
hipcc -Rpass-analysis=kernel-resource-usage kernel.cpp

# Assembly inspection
hipcc --save-temps kernel.cpp
# Check .vgpr_count, .vgpr_spill_count
```

---

## Related

- **Index**: AMD GPU Kernel Tuning Guide — Complete tuning topic index
- **Hardware Specifications**: [Hardware Specification Comparison](../hardware-specs/hardware-comparison-cdna3-cdna4.md) — Hardware parameters such as VGPR file size
- **General Execution Model**: [GPU Execution Model and Thread Optimization](../../generic/gpu-execution-model.md) — SIMT execution model, general theory of occupancy optimization
- **Triton Tuning**: Triton Kernel Tuning Parameters — Occupancy control at the Triton level such as `waves_per_eu`
