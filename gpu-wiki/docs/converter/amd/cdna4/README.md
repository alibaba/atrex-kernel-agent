# CDNA4 (MI355X/gfx950) Conversion Tools

CDNA4/MI355X-specific Triton→Gluon conversion rules and patterns.

---

| File | Description |
|------|-------------|
| [Triton → Gluon Conversion Guide (AMD CDNA4)](conversion-guide.md) | CDNA4/MI355X-specific Triton→Gluon conversion |
| [API Mapping Reference (CDNA4 / MI355X)](api_mapping.md) | CDNA4-specific API mapping (including mfma_scaled, async_copy) |
| [Pipeline Conversion Pattern (CDNA4)](pipeline.md) | Hardware async copy (DMA) pipeline, differs from CDNA3 |
| [Matrix Multiply Pattern](matrix_multiply.md) | tl.dot → Gluon MFMA (CDNA4) |
| [Memory Access Pattern](memory_access.md) | CDNA4 2D block pointer load/store conversion |
| [Layouts](layouts.md) | CDNA4 TTGIR→Gluon layout mapping |
| [Common Errors and Solutions (CDNA4 / gfx950)](common_pitfalls.md) | CDNA4-specific conversion errors and solutions |
