# Ampere CuTeDSL Flash Attention Kernels

CuTeDSL reference implementation of Flash Attention on the Ampere architecture.

---

| Kernel | Description |
|--------|-------------|
| [compute_block_sparsity.py](compute_block_sparsity.py) | Block sparsity computation |
| [flash_bwd.py](flash_bwd.py) | Flash Attention backward pass |
| [flash_bwd_postprocess.py](flash_bwd_postprocess.py) | Flash Attention backward post-processing |
| [flash_bwd_preprocess.py](flash_bwd_preprocess.py) | Flash Attention backward pre-processing |
| [flash_fwd.py](flash_fwd.py) | Flash Attention forward pass |
| [flash_fwd_combine.py](flash_fwd_combine.py) | Flash Attention forward combine |
