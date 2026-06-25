# AMD CDNA Triton Kernel

Triton kernel reference implementations on the AMD CDNA architecture.

---

| Directory | Description |
|------|------|
| [aiter/](aiter/) | Aiter inference operator library Triton kernels (Attention, GEMM, MoE, Norm, Quant, and 80+ other files) |
| [chunk_gdn/](chunk_gdn/) | RTP's currently tuned Chunk-GDN Triton back-half baseline: `recompute_w_u_fwd -> fwd_h -> fwd_o` |
