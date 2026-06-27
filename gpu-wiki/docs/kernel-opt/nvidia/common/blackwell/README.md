# Blackwell Kernel Optimization Knowledge Cards

This directory collects NVIDIA Blackwell / Hopper kernel optimization knowledge cards cleaned from the kernel-pilot wiki, covering hardware mechanisms, typical kernels, language interfaces, migration rules, bottleneck patterns, and reusable optimization techniques.

These contents are short knowledge points, pattern cards, and quick references that can be reused across projects; complete optimization reports are still placed in `docs/ref-docs/`.

---

| Directory | Description |
|-----------|-------------|
| [hardware/](hardware/) | Hardware mechanisms: Blackwell / Hopper related hardware mechanisms and architectural primitives. |
| [kernels/](kernels/) | Kernel cases: Typical high-performance kernels, model operators, and implementation roadmaps. |
| [languages/](languages/) | Languages & DSLs: Programming interfaces such as CUDA C++, PTX, CuTeDSL, Triton, and Blackwell differences. |
| [migration/](migration/) | Migration guide: Rules for migrating from Hopper / register accumulator paths to Blackwell / TMEM / tcgen05. |
| [patterns/](patterns/) | Bottleneck patterns: Bottleneck identification patterns based on profiling or performance phenomena. |
| [techniques/](techniques/) | Optimization techniques: Reusable kernel optimization technique cards. |
