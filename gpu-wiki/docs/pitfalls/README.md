# GPU Kernel Pitfalls

Non-obvious traps encountered while implementing or porting GPU kernels.
Each entry includes the trap, the symptom, the root cause, and the lesson.

Organized by GPU vendor → DSL/framework → kernel category.

---

| Directory | Description |
|-----------|-------------|
| [amd/](amd/) | AMD CDNA pitfalls: FlyDSL kernel traps (FlashAttention, Chunk-GDN, Fused MoE) |
| [nvidia/](nvidia/) | NVIDIA Blackwell pitfalls: CUDA, CuTeDSL, Gluon, Triton kernel traps |

## How to add a new entry

1. File path: `pitfalls/<vendor>/<framework>/<short-name>.md`
2. Each pitfall section: trap → symptom → reality → why → lesson.
3. Cross-link the optimization journey doc in `ref-docs/` and the
   reference impl in `reference-kernels/`.

