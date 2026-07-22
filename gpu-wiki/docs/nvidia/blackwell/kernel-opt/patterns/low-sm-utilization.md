# Low SM Utilization

## Symptom

SM utilization below 60% despite sufficient occupancy. Nsight Compute shows idle SMs during portions of kernel execution.

## Likely Causes

1. **Tail effect**: Last wave of tiles leaves most SMs idle (see [tail-effect](tail-effect.md))
2. **Load imbalance**: Some tiles take longer than others (variable computation per tile)
3. **Static scheduling**: Fixed tile-to-SM assignment doesn't adapt to runtime conditions
4. **Grid too small**: Fewer threadblocks than SMs

## Candidate Techniques

| Technique | Applicability | Effect |
|---|---|---|
| [CLC](../hardware/clc.md) | SM100 only | Dynamic tile assignment, eliminates load imbalance |
| [Persistent kernels](../techniques/persistent-kernels.md) | SM90+ | Eliminates tail effect, one-time launch overhead |
| [Tile scheduling](../techniques/tile-scheduling.md) | SM90+ | Better L2 locality, reduce load variance |

## Examples

```
// tcgen05 tutorial progression:
// Without persistent/CLC: 86% of cuBLAS (some SMs idle at wave boundaries)
// With persistent + CLC:  98% of cuBLAS (all SMs stay busy)
```

## Caveats
- CLC only available on SM100 datacenter GPUs (not SM120 consumer)
- Persistent kernels complicate debugging and profiling
- For non-persistent kernels, ensure grid size >> SM count
