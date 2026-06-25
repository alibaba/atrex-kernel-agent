# QuACK — Hopper (SM90) CuTeDSL Kernels

Dao-AILab QuACK ("Quirky Assortment of CuTe Kernels") high-performance GPU kernel library, written entirely in CuTeDSL (Python). This directory contains SM90+ common code (48 .py files + sort/ subdirectory). For SM100/SM120 specific code, see the corresponding architecture directories.

> Source: QuACK upstream repository mirror (`quack/`)

## Kernel Files

### Reduction / Memory-Bound Kernels
| File | Description |
|------|------|
| `reduction_base.py` | ReductionBase framework: thread→warp→block→cluster 4-level reduction |
| `rmsnorm.py` | RMSNorm fwd/bwd, based on ReductionBase |
| `softmax.py` | Softmax fwd/bwd, online softmax algorithm |
| `cross_entropy.py` | CrossEntropy fwd/bwd, log-sum-exp stable computation |
| `topk.py` | TopK kernel, bitonic sort network |
| `reduce.py` | General-purpose reduce kernel |
| `rms_final_reduce.py` | RMSNorm final reduce (cluster-level) |
| `sort/bitonic_sort.py` | Bitonic sort network |
| `sort/sorting_networks.py` | Sort network generator |
| `sort/generate_sorting_networks.py` | Sort network code generation |
| `sort/utils.py` | Sort utility functions |

### GEMM / Compute-Bound Kernels
| File | Description |
|------|------|
| `gemm.py` | GEMM entry point: multi-architecture dispatch (SM90/SM100/SM120) |
| `gemm_sm90.py` | SM90 GEMM: WGMMA + TMA + pingpong + cluster multicast |
| `gemm_act.py` | GEMM + Activation fusion |
| `gemm_dact.py` | GEMM + dActivation backward fusion |
| `gemm_norm_act.py` | GEMM + Norm + Activation fusion |
| `gemm_sq_reduce.py` | GEMM + SmoothQuant reduce fusion |
| `gemm_symmetric.py` | Symmetric GEMM |
| `gemm_blockscaled_interface.py` | Block-scaled GEMM interface |
| `gemm_config.py` | GEMM configuration parameters |
| `gemm_default_epi.py` | GEMM default epilogue |
| `gemm_interface.py` | GEMM general interface |
| `gemm_wrapper_utils.py` | GEMM wrapper utilities |
| `gemm_tvm_ffi_utils.py` | TVM FFI binding utilities |

### Composable Epilogue
| File | Description |
|------|------|
| `epi_ops.py` | Epilogue operator definitions (bias/activation/scaling/reduce) |
| `epi_composable.py` | Composable Epilogue composition system |
| `epi_utils.py` | Epilogue utility functions |

### Fused High-Level Ops
| File | Description |
|------|------|
| `linear.py` | Fused Linear (GEMM + epilogue) |
| `linear_cross_entropy.py` | Fused Linear + CrossEntropy |
| `mlp.py` | Fused MLP (with activation recomputation) |

### Infrastructure / Utils
| File | Description |
|------|------|
| `autotuner.py` | Autotuner: tile/cluster/swap_ab search + disk cache |
| `cache_utils.py` | Multi-layer cache: memory dict + SHA-256 file + parallel compilation |
| `compile_utils.py` | CuTeDSL compilation utilities |
| `_compile_worker.py` | Parallel compilation worker |
| `cute_dsl_utils.py` | CuTeDSL helper functions |
| `cute_dsl_ptxas.py` | PTX assembler interface |
| `varlen_utils.py` | Variable-length / Ragged sequence support (TMA ptr_shift) |
| `tensormap_manager.py` | TMA TensorMap manager |
| `tile_scheduler.py` | Tile scheduler (persistent/stream-k) |
| `pipeline.py` | Software pipelining |
| `activation.py` | Activation function definitions |
| `fast_math.py` | Fast math functions |
| `rounding.py` | Rounding modes |
| `layout_utils.py` | Layout utilities |
| `copy_utils.py` | Data copy utilities |
| `broadcast_utils.py` | Broadcast utilities |
| `mx_utils.py` | MX (Microscaling) format utilities |
| `blockscaled_gemm_utils.py` | Block-scaled GEMM utilities |
| `nvmmh_heuristic.py` | NVMM heuristic selection |
| `sm90_utils.py` | SM90 architecture-specific utilities |
| `trace.py` | Trace / debugging utilities |
| `utils.py` | General-purpose utility functions |
| `__init__.py` | Package initialization |
