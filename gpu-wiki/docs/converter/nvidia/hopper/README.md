# Hopper (H100/H20, sm_90) Conversion Tools

Conversion rules and patterns for Triton → Gluon, specific to NVIDIA Hopper sm_90.

---

| File | Description |
|------|------|
| [Triton → Gluon Conversion Guide (NVIDIA Hopper)](conversion-guide.md) | Hopper sm_90 specific Triton → Gluon conversion |
| [API Mapping Table (NVIDIA Hopper)](api_mapping.md) | Hopper Triton → Gluon API mapping (including async_copy) |
| [Pipeline Conversion Pattern (NVIDIA Hopper)](pipeline.md) | CP_ASYNC DMA pipeline: async global → shared |
| [Matrix Multiply Pattern (NVIDIA Hopper wgmma)](matrix_multiply.md) | tl.dot → wgmma warpgroup-level MMA |
| [Memory Access Pattern (NVIDIA Hopper)](memory_access.md) | Hopper 2D block pointer load/store |
| [Layouts (NVIDIA Hopper)](layouts.md) | Hopper TTGIR → Gluon layout mapping (including version) |
| [Common Errors and Solutions (NVIDIA Hopper)](common_pitfalls.md) | Misusing AMD API on Hopper causing LLVM crash, etc. |
