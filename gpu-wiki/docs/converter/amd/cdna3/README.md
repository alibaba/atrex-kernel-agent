# CDNA3 (MI300X/gfx942) Conversion Tools

CDNA3/MI300X-specific Triton→Gluon conversion rules and patterns.

---

| File | Description |
|------|------|
| [Triton → Gluon Conversion Guide (AMD CDNA3)](conversion-guide.md) | CDNA3/MI300-specific Triton→Gluon conversion |
| [API Mapping Reference](api_mapping.md) | Triton→Gluon API mapping for CDNA3 |
| [Pipeline Conversion Patterns](pipeline.md) | Triton→Gluon software pipeline: prologue/main-loop/epilogue |
| [Matrix Multiply Patterns](matrix_multiply.md) | tl.dot → Gluon MFMA + shared memory allocation |
| [Memory Access Patterns](memory_access.md) | 2D block pointer load/store conversion |
| [Layouts](layouts.md) | TTGIR→Gluon layout field mapping rules |
| [Common Pitfalls and Solutions](common_pitfalls.md) | Mask block type, layout mismatch, etc. |
