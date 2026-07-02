# SM120 RMSNorm-MLP NVFP4 PDL and Row-Ready Sources

> **Usability status:** `diagnostic-archive`
>
> This package captures a specific SM120 investigation. It may require `$CUTLASS_DIR` and shape-specific assumptions before it can run.

CUDA/vLLM source snapshots and component probes for the SM120
`residual add -> RMSNorm -> MLP input NVFP4 quant -> parent MLP C1 GEMM`
boundary.

These files are archived to preserve the source behind the PDL and row-ready
lessons. They are correctness and diagnostic references only: the no-PDL route
was served-neutral, whole-A PDL regressed served TTFT, row-chunk PDL regressed
component timing, and row-ready wait-cache remained operator-negative.

---

| File | Scope |
|---|---|
| [nvfp4_quant_rmsnorm_fp4_sm120_pdl.cu](nvfp4_quant_rmsnorm_fp4_sm120_pdl.cu) | Fused add/RMSNorm/NVFP4 quant producer source with PDL and row-ready variants. |
| [nvfp4_mlp_fusion_entry_rmsnorm_pdl_sm120.cu](nvfp4_mlp_fusion_entry_rmsnorm_pdl_sm120.cu) | vLLM stable-libtorch entry points for parent fusion, PDL handoff, and row-chunk routes. |
| [nvfp4_mlp_fusion_sm120_rowready_diagnostic.cu](nvfp4_mlp_fusion_sm120_rowready_diagnostic.cu) | SM120 CUTLASS/vLLM parent-fusion diagnostic source with row-ready and Warp1/LoadMN probes. |
| [cutlass_sm120_rowready_wait_hook_diagnostic.hpp](cutlass_sm120_rowready_wait_hook_diagnostic.hpp) | CUTLASS SM120 mainloop hook showing where C1 waits before A/input-scale loads. |
| [rmsnorm_mlp_parent_quant_probe.py](rmsnorm_mlp_parent_quant_probe.py) | Component correctness/timing probe for no-PDL fused RMSNorm + input quant. |
| [row_chunk_parent_probe.py](row_chunk_parent_probe.py) | Row-chunk parent-pipeline correctness/timing probe. |
| [ready_wait_component_probe.py](ready_wait_component_probe.py) | Row-ready wait-cache component timing probe. |

## Related Docs

- [SM120 RMSNorm-MLP NVFP4 Fusion and PDL Handoff Report](../../../../../docs/nvidia/blackwell-geforce/cuda/sm120-rmsnorm-mlp-pdl-fusion-report.md)
- [SM120 RMSNorm-MLP PDL Pitfalls](../../../../../docs/nvidia/blackwell-geforce/cuda/pitfalls/sm120-rmsnorm-mlp-pdl-pitfalls.md)
