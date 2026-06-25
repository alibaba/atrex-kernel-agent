# GDN Decode and Complex State Management

## Pattern: Multi-Stage State Update Kernel

**Source**: `cutedsl/flashinfer/gdn_decode_*.py`

```python
# GDN (Gated Delta Network) decode: need to manage multiple states
# 1. gate state
# 2. delta state  
# 3. output state

# Each decode step requires:
# a. Load all states from previous step
# b. Compute gate = sigmoid(W_g @ x)
# c. Compute delta = gate * (W_d @ x)
# d. Update state state = state + delta
# e. Compute output y = W_o @ state
# f. Save new state
```

**Practical Experience**:
- Decode kernels are typically memory-bound (only one token processed per step)
- The key is to minimize state read/write volume: coalesce multiple small states into a single contiguous read/write
- Using TMA to load state can reduce address computation overhead

---

## Related Documents

- **Hopper Pitfalls**: [Hopper Practical Pitfalls](../../../../../ref-docs/nvidia/gluon/sm90/pitfalls.md) — Case studies including chunk_gated_delta_rule
- **Linear Attention**: [Chunk Linear Attention Optimization](../../../../../ref-docs/nvidia/gluon/sm90/linear_attention.md) — Similar state management patterns
- **Hardware Specs**: [Hopper Hardware Specification Table](../../../../../hardware-specs/hardware_specs_hopper.md) — Memory bandwidth
- **Reference Kernels**: `reference-kernels/nvidia/hopper/` — 21 Hopper kernel source files
