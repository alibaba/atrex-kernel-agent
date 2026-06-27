# Blackwell CuTeDSL CUTLASS Kernels

CUTLASS framework reference kernel implementations using CuTeDSL on the Blackwell (SM100) architecture.

---

| Directory | Description |
|------|------|
| [blockwise_gemm/](blockwise_gemm/) | Blockwise GEMM and Grouped GEMM |
| [distributed/](distributed/) | Distributed communication (All-Reduce, All-Gather GEMM, Reduce-Scatter) |
| [epilogue/](epilogue/) | Epilogue fused computation patterns |
| [mamba2_ssd/](mamba2_ssd/) | Mamba2 SSD state space model |
| [mixed_input_fmha/](mixed_input_fmha/) | Mixed-precision FMHA (Decode, Prefill) |
| [mixed_input_gemm/](mixed_input_gemm/) | Mixed-precision GEMM |
| [mla/](mla/) | MLA (Multi-head Latent Attention) Decode |
| [tutorial_gemm/](tutorial_gemm/) | GEMM tutorial series (FP16, NVFP4) |

| Kernel | Description |
|--------|------|
| [dense_blockscaled_gemm_persistent.py](dense_blockscaled_gemm_persistent.py) | Dense Block-Scaled GEMM Persistent mode |
| [dense_blockscaled_gemm_persistent_amax.py](dense_blockscaled_gemm_persistent_amax.py) | Dense Block-Scaled GEMM Persistent + Amax |
| [dense_blockscaled_gemm_persistent_prefetch.py](dense_blockscaled_gemm_persistent_prefetch.py) | Dense Block-Scaled GEMM Persistent + Prefetch |
| [dense_gemm.py](dense_gemm.py) | Dense GEMM basic version |
| [dense_gemm_alpha_beta_persistent.py](dense_gemm_alpha_beta_persistent.py) | Dense GEMM Alpha-Beta Persistent mode |
| [dense_gemm_persistent.py](dense_gemm_persistent.py) | Dense GEMM Persistent mode |
| [dense_gemm_persistent_dynamic.py](dense_gemm_persistent_dynamic.py) | Dense GEMM Persistent dynamic version |
| [dense_gemm_persistent_prefetch.py](dense_gemm_persistent_prefetch.py) | Dense GEMM Persistent + Prefetch |
| [dense_gemm_software_pipeline.py](dense_gemm_software_pipeline.py) | Dense GEMM software pipeline version |
| [experimental_dense_block_scaled_gemm.py](experimental_dense_block_scaled_gemm.py) | Experimental Dense Block-Scaled GEMM |
| [experimental_dense_gemm.py](experimental_dense_gemm.py) | Experimental Dense GEMM |
| [experimental_dense_gemm_2sm.py](experimental_dense_gemm_2sm.py) | Experimental Dense GEMM dual SM |
| [experimental_dense_gemm_cute_pipeline.py](experimental_dense_gemm_cute_pipeline.py) | Experimental Dense GEMM CuTe pipeline |
| [experimental_dense_gemm_ptr_array.py](experimental_dense_gemm_ptr_array.py) | Experimental Dense GEMM pointer array |
| [fmha.py](fmha.py) | Fused Multi-Head Attention |
| [fmha_bwd.py](fmha_bwd.py) | FMHA backward pass |
| [grouped_blockscaled_gemm.py](grouped_blockscaled_gemm.py) | Grouped Block-Scaled GEMM |
| [grouped_gemm.py](grouped_gemm.py) | Grouped GEMM |
| [programmatic_dependent_launch.py](programmatic_dependent_launch.py) | Programmatic Dependent Launch (PDL) |
| [reduce.py](reduce.py) | Reduction kernel |
| [rmsnorm.py](rmsnorm.py) | RMSNorm kernel |
| [sm103_dense_blockscaled_gemm_persistent.py](sm103_dense_blockscaled_gemm_persistent.py) | SM103 Dense Block-Scaled GEMM Persistent |
