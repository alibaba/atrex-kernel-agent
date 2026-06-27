# SM120 CuTeDSL

Summary of optimization and pitfalls for CuTeDSL on SM120 / Blackwell GeForce.

---

| File | Description |
|------|-------------|
| [sm120-gdn-decode-cpasync-cache-mode.md](sm120-gdn-decode-cpasync-cache-mode.md) | GDN decode quick reference: cp.async + `LoadCacheMode.GLOBAL` + `assumed_align=16` 4-piece toolkit to break L2 false-saturation. Extends to ref-docs/sm120/ optimization journey and pitfalls/. |
| [sm120-moe-data-prep.md](sm120-moe-data-prep.md) | INT32 MoE data prep quick reference: CUDA C++ via `load_inline` + V6 multi-CTA split + V7 contention-free per-CTA offsets + V9-A 4-way bank-replicated histogram. **0.706× of vLLM CG at T=6144**. Includes anti-pattern table (warp-agg / CUB sort / early-exit warp-specialization all regress). Extends to ref-docs/sm120/ optimization journey and pitfalls/, and cross-architecture NCU meta-rule. |
