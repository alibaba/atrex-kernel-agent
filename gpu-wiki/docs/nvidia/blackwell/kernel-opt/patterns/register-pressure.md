# Register Pressure -- Low Occupancy

## Symptom

Occupancy below target due to high register usage per thread. Nsight Compute shows register spilling to local memory.

## Likely Causes

1. **Accumulator registers**: On Hopper, large MMA tiles consume many registers for accumulators
2. **Epilogue state**: Data transformation in epilogue requires additional registers
3. **Complex control flow**: Many live variables across branches

## Candidate Techniques

| Technique | Applicability | Effect |
|---|---|---|
| [TMEM](../hardware/tmem.md) | SM100 only | Moves accumulators to dedicated 256KB memory |
| [Warp specialization](../techniques/warp-specialization.md) | SM100+ | Different warps handle different roles, reducing per-warp register needs |
| [Register-to-TMEM migration](../migration/register-to-tmem.md) | SM90→SM100 | Systematic approach to moving accumulators off registers |

## Blackwell Solution

```
// Hopper: 64×256 MMA tile accumulator = 64*256*4 bytes in registers per warp group
//   → ~128 registers per thread just for accumulators
//
// Blackwell: TMEM holds accumulators
//   → 0 registers for accumulators
//   → ~128 registers freed for other work or higher occupancy
//
// TMEM: 256 KB per SM, 128 rows × 512 columns × 32-bit
// Largest 1-SM MMA uses half → double-buffering possible
```

## Caveats
- TMEM only available on SM100 datacenter (not SM120 consumer)
- TMEM requires explicit alloc/dealloc lifecycle
- TMEM→register transfer adds latency (offset by freeing registers)
