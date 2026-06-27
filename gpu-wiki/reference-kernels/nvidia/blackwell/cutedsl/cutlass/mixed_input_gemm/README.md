# Blackwell CUTLASS Mixed-Input GEMM Kernels

Reference implementations of mixed-precision GEMM on the Blackwell architecture using the CUTLASS framework.

---

| Kernel | Description |
|--------|-------------|
| [grouped_mixed_input_gemm.py](grouped_mixed_input_gemm.py) | Grouped mixed-precision GEMM |
| [grouped_mixed_input_gemm_acc_scale.py](grouped_mixed_input_gemm_acc_scale.py) | Grouped mixed-precision GEMM (with accumulation scaling) |
| [mixed_input_gemm.py](mixed_input_gemm.py) | Mixed-precision GEMM |
| [mixed_input_host_utils.py](mixed_input_host_utils.py) | Mixed-precision host-side utility functions |
