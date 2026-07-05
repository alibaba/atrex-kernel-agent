# AMD Triton→Gluon conversion sheets

One sheet per architecture (see the transition router in [../README.md](../README.md)).

| File | Target |
|------|--------|
| [cdna3.md](cdna3.md) | CDNA3 (MI300X, gfx942) — buffer_load / mfma, software pipeline |
| [cdna4.md](cdna4.md) | CDNA4 (MI355X, gfx950) — inherits CDNA3 + hardware async_copy, mfma_scaled |
