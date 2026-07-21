# LDS Bank Conflict Optimization

## Bank Architecture

- **32 physical banks** (CDNA3), **64 banks** (CDNA4)
- Each bank serves 4 bytes per cycle
- Conflicts occur when multiple lanes access different addresses within the same bank

## Conflict Checking for `ds_read_b128` and `ds_write_b128`

**Write (ds_write_b128)**: Checked in groups of 8 consecutive lanes
```
{0-7}, {8-15}, {16-23}, {24-31}, {32-39}, {40-47}, {48-55}, {56-63}
```

**Read (ds_read_b128)**: Uses interleaved grouping (different from writes!)
```
{0:3,20:23}, {4:7,16:19}, {8:11,28:31}, {12:15,24:27},
{32:35,52:55}, {36:39,48:51}, {40:43,60:63}, {44:47,56:59}
```

**Key:** A layout that is conflict-free for writes may have conflicts on reads.

## XOR Swizzle for Conflict Elimination

CK-Tile's XOR transform remaps indices without consuming additional LDS:

```
K0' = K0 ^ (M % (KPerBlock / Kpack * MLdsLayer))
```

Better than padding: zero extra LDS consumption while eliminating both read and write conflicts.

---

## Related Documents

- **Index**: AMD GPU Kernel Tuning Guide — Complete tuning topic index
- **Hardware Specs**: [Hardware Specification Comparison](../hardware-specs/hardware-comparison-cdna3-cdna4.md) — LDS bank count differences between CDNA3 vs CDNA4
- **General Memory Hierarchy**: [GPU Memory Hierarchy & Optimization](../../../generic/ref-docs/gpu-memory-hierarchy.md) — General shared memory / LDS optimization strategies
- **Hands-on**: [LDS Bank Conflict Swizzle Hands-on](hands-on/lds-bank-conflict-swizzle.md) — XOR swizzle code examples
