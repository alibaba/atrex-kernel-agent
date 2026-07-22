# Blackwell GeForce CuTeDSL FlashInfer Kernels

> **Usability status:** `diagnostic-archive`
>
> This package captures a specific SM120 investigation. It may require `$CUTLASS_DIR` and shape-specific assumptions before it can run.

Reference kernel implementations and diagnostic forks of the FlashInfer project
on Blackwell GeForce using CuTeDSL.

---

| Kernel | Description |
|--------|-------------|
| [dense_blockscaled_gemm_sm120_task39_diagnostic.py](dense_blockscaled_gemm_sm120_task39_diagnostic.py) | omoExplore SM120 b12x CuTe DSL diagnostic fork for NVFP4 prefill/gate-up SF-layout experiments. |
| [task39_b12x_runner_diagnostic.py](task39_b12x_runner_diagnostic.py) | Runner that mirrors FlashInfer b12x input contracts for same-layout diagnostic comparisons. |
