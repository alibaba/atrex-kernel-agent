# SM120 NVFP4 Prefill GEMM Experiments

> **Usability status:** `diagnostic-archive`
>
> This package captures a specific SM120 investigation. It may require `$CUTLASS_DIR` and shape-specific assumptions before it can run.

Experimental CUDA and vLLM integration sources for SM120 NVFP4 prefill GEMM
work. These files capture the source boundaries behind the prefill restart,
forced-backend, direct-layout, and M-chunk experiments.

The current knowledge-base conclusion is conservative: production-shape
inventory and current-backend baselines come first, and the stronger direction
is a higher-level dense MLP fusion boundary rather than broad forced GEMM
routing. Treat this directory as source-map evidence, not a promoted backend.

---

| File | Scope |
|---|---|
| [flashinfer_prefill_backend_router_sm120_experimental.py](flashinfer_prefill_backend_router_sm120_experimental.py) | vLLM FlashInfer router with prefill-specific backend, allowlist, custom `.so`, and prepack guards. |
| [prefill_mma_padded_sm120_experimental.cu](prefill_mma_padded_sm120_experimental.cu) | CUDA NVFP4 prefill GEMM candidate with padded-M and direct-layout entry points. |
| [cutlass_mchunk_split_nvfp4_sm120_experimental.cu](cutlass_mchunk_split_nvfp4_sm120_experimental.cu) | Standalone CUTLASS split-chain launcher for M-chunk prefill probes. |
| [build_prefill_mma_padded_sm120_experimental.sh](build_prefill_mma_padded_sm120_experimental.sh) | Build helper for the archived padded prefill CUDA candidate. |

## Related Docs

- [SM120 NVFP4 Decode GEMM Production Lessons](../../../../../docs/nvidia/blackwell-geforce/ref-docs/cuda/sm120-nvfp4-decode-gemm-production-lessons.md)
- [SM120 NVFP4 Decode GEMM Production Pitfalls](../../../../../docs/nvidia/blackwell-geforce/pitfalls/cuda/sm120-nvfp4-decode-gemm-production-pitfalls.md)
