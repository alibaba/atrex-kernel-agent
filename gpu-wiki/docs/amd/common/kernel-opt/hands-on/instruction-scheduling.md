# Instruction Scheduling Control

Instruction scheduling optimization patterns extracted from `reference-kernels/amd/`, controlling pipeline overlap via instructions such as `sched_barrier`.

---

## Pattern: sched_barrier / sched_dsrd / sched_mfma

**Source**: `cdna/flydsl/FlyDSL/`, `cdna4/gluon/triton/`

```python
# AMD GPU compilers respond well to ISA reordering (opposite of NVIDIA)
# Manual instruction scheduling can be used to optimize pipeline overlap

# In FlyDSL
flyc.sched_barrier(0)    # Prevent instructions from being reordered across the barrier
flyc.sched_mfma(8)       # Hint that 8 MFMA instructions follow

# In Gluon
from triton.experimental.gluon.language.amd import sched_barrier

sched_barrier(0)  # Establish a scheduling barrier
# Load instructions
a = tl.load(a_ptr)
b = tl.load(b_ptr)
sched_barrier(0)  # Ensure loads complete before MFMA
# Compute instructions
c = tl.dot(a, b)
```

**Hands-on Experience**:
- AMD compilers (Gluon for gfx942/gfx950) respond **well** to manual scheduling hints
- In stark contrast to NVIDIA Hopper: Hopper's compiler performs more aggressive global optimizations, and manual intervention can easily disrupt the compiler's strategy
- `sched_barrier` Most common use case: ensuring loads and MFMA are correctly interleaved
- Overusing `sched_barrier` can also degrade performance (it limits the compiler's optimization space)

---

## Related Documents

- **CDNA3 ISA**: [CDNA3 ISA Instruction Patterns](../../../cdna3/ref-docs/gluon/isa_patterns.md)
- **Profiling**: [AMD rocprofv3 Profiling Guide](../../ref-docs/rocprofv3-profiling-guide.md) — General usage of rocprofv3
- **General Instruction Optimization**: [GPU Instruction-Level Optimization](../../../../generic/ref-docs/gpu-instruction-optimization.md)
