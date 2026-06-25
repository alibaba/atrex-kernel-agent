# triton-kernels Project Kernel Reference Implementations

Triton kernel reference implementations from the triton-kernels project, covering distributed communication, reduction, matrix multiplication, numerical precision, compaction, and other modules.

---

| Directory | Description |
|-----------|-------------|
| [compaction_details/](compaction_details/) | Masked compaction implementation |
| [matmul_details/](matmul_details/) | Matmul and persistent matmul implementation |
| [numerics_details/](numerics_details/) | Numerical precision (flexpoint, MXFP) implementation |
| [swiglu_details/](swiglu_details/) | SwiGLU activation function implementation |
| [topk_details/](topk_details/) | Top-K forward and backward implementation |

| Kernel | Description |
|--------|-------------|
| [distributed.py](distributed.py) | Distributed communication kernel |
| [reduce.py](reduce.py) | Reduction kernel |
