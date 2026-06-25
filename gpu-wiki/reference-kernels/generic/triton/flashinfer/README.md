# FlashInfer Project Triton Kernel Reference Implementations

Triton kernel reference implementations from the FlashInfer project, covering GEMM, quantization, normalization, cascaded attention, and other inference optimization operators.

---

| Kernel | Description |
|--------|-------------|
| [activation.py](activation.py) | Activation function kernel |
| [cascade.py](cascade.py) | Cascaded attention state merging kernel |
| [gemm.py](gemm.py) | GEMM kernel |
| [norm.py](norm.py) | Normalization kernel |
| [page.py](page.py) | Paged KV Cache management kernel |
| [quant.py](quant.py) | Quantization kernel |
| [sm_constraint_gemm.py](sm_constraint_gemm.py) | SM-constrained GEMM kernel |
| [ssd_chunk_state.py](ssd_chunk_state.py) | SSD/Mamba chunk state update kernel |
