# Pipeline Pattern Comparison

## New Pipeline Types Introduced by Blackwell

| Pipeline Type | Usage | Source |
|---|---|---|
| `PipelineTmaUmma` | TMA load + tcgen05 MMA (most common) | GEMM, FMHA |
| `PipelineUmmaAsync` | TMEM double-buffered async pipeline | Large tile GEMM |
| `PipelineTmaStore` | TMA store async write-back | Epilogue |

## Comparison with Hopper

| Feature | Hopper | Blackwell |
|---|---|---|
| Load pipeline | `PipelineTmaAsync` | `PipelineTmaUmma` |
| Accumulator location | Register | TMEM |
| Epilogue execution | MMA warp serial | Epilogue warp parallel |
| Number of warp roles | 2 (TMA + MMA) | 3 (TMA + MMA + Epilogue) |
| Store pipeline | MMA warp direct store | `PipelineTmaStore` |

---

## Related Documentation

- **tcgen05 MMA and TMEM**: [tcgen05 MMA and TMEM](tcgen05-mma-tmem.md) — Relationship between TMEM accumulator and pipeline
- **Three-Role Warp Specialization**: [Three-Role Warp Specialization](three-role-warp-specialization.md) — How three-role architecture drives pipeline changes
- **Hopper Hands-On**: [Hopper Optimization Hands-On](README.md) — Hopper pipeline comparison
- **CuTeDSL Pipeline**: [CuTeDSL Pipeline Patterns](../../../common/ref-docs/cutedsl/cutedsl-pipeline-patterns.md) — Producer/consumer patterns
