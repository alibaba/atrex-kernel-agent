# AMD GPU Optimization Knowledge

AMD GPU kernel optimization knowledge base, organized by DSL/programming language → hardware architecture.

## Optimization Tips

For ISA, the key points to focus on are:
1. The total stall time should be as short as possible, and the instruction count should also be minimized;
2. Independent instructions should be inserted after long-latency instructions as much as possible, fully utilizing the SIMD instruction setotations to process data rather than scalar instruction loops;
3. Data access patterns should adhere to spatial locality and temporal locality to reduce cache misses;
4. If waves_per_eu=2 is set, AGPR can be avoided by having the two waves each use two sets of VGPRs.

---

| Directory | Description |
|-----------|-------------|
| [common/](common/) | AMD Common: optimization framework, MFMA programming, tuning guide, profiling tools (DSL-agnostic) |
| [gluon/](gluon/) | Gluon DSL kernel optimization on AMD GPUs (gfx942, gfx950) |
| [flydsl/](flydsl/) | FlyDSL (MLIR) programming framework and kernel optimization |
