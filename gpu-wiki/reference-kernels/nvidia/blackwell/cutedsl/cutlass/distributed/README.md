# Blackwell CUTLASS Distributed Kernels

Reference implementations of distributed communication and computation using the CUTLASS framework on the Blackwell architecture.

---

| Kernel | Description |
|--------|-------------|
| [all_reduce_one_shot_lamport.py](all_reduce_one_shot_lamport.py) | All-Reduce single-round Lamport protocol |
| [all_reduce_simple.py](all_reduce_simple.py) | All-Reduce simple implementation |
| [all_reduce_tma.py](all_reduce_tma.py) | All-Reduce TMA accelerated version |
| [all_reduce_two_shot_multimem.py](all_reduce_two_shot_multimem.py) | All-Reduce two-round MultiMem version |
| [distributed_all_gather_gemm_blackwell.py](distributed_all_gather_gemm_blackwell.py) | Distributed All-Gather GEMM |
| [distributed_gemm_all_reduce_blackwell.py](distributed_gemm_all_reduce_blackwell.py) | Distributed GEMM All-Reduce |
| [distributed_gemm_reduce_scatter_blackwell.py](distributed_gemm_reduce_scatter_blackwell.py) | Distributed GEMM Reduce-Scatter |
