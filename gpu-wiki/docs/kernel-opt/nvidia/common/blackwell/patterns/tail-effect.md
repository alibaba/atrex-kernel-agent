# Tail Effect -- Last Wave Underutilization

## Symptom

Performance drops for problem sizes where total_tiles % num_SMs != 0. The last wave of tiles runs with many SMs idle.

## Likely Causes

1. **Wave quantization**: Grid of N tiles on M SMs takes ceil(N/M) waves; last wave may use only N%M SMs
2. **Static assignment**: stride-by-gridDim leaves remainder tiles on few SMs
3. **Non-persistent launch**: each kernel launch has fixed grid, no dynamic rebalancing

## Candidate Techniques

| Technique | Effect |
|---|---|
| [CLC](../hardware/clc.md) | Hardware dynamic scheduling, SMs grab tiles on-demand |
| [Persistent kernels](../techniques/persistent-kernels.md) | SM-count grid, iterate over tiles, no wave boundary |
| [Tile scheduling](../techniques/tile-scheduling.md) | Raster order, swizzle patterns for better distribution |

## Example

```
// B200 example target: 148 SMs
// Problem: 156 tiles
// Without CLC: 2 waves (148 + 8), last wave uses only 8 SMs (5.4%)
// With CLC: single persistent wave, all 148 SMs stay busy
//
// Impact: 86% → 98% of cuBLAS (tcgen05 tutorial data)
```

## Caveats
- Only significant for moderate tile counts (< 4× SM count)
- For very large problems, tail effect is amortized across many waves
- CLC only on SM100 datacenter
