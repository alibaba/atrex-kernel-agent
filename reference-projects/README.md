# Reference Projects

This directory contains GPU kernel optimization reference projects managed as git submodules.

## Usage

```bash
# Initialize and clone all submodules (shallow)
git submodule update --init

# Update all submodules to latest
git submodule update --remote
```

## Included Projects

| Project | Description |
|---------|-------------|
| cutlass | NVIDIA CUTLASS - CUDA Templates for Linear Algebra Subroutines |
| cutex | CUDA Template Extensions |
| cuLA | inclusionAI CUDA Linear Algebra |
| flash-attention | Flash Attention |
| flashinfer | FlashInfer - Kernel Library for LLM Serving |
| FlyDSL | ROCm FlyDSL |
| triton | Triton Language and Compiler |
| DeepGEMM | DeepSeek DeepGEMM |
| LeetCUDA | LeetCUDA - CUDA Learning |
| FlashMLA | DeepSeek FlashMLA |
| composable_kernel | ROCm Composable Kernel |
| cute-gemm | CuTe GEMM Examples |
| hpc-ops | Tencent HPC Ops |
| aiter | ROCm AIter |
| quack | Dao-AILab Quack |
| tilelang | TileLang |
| how-to-optim-algorithm-in-cuda | CUDA kernels and engineering notes covering CUTLASS/CuTe, Triton, PTX, PyTorch, and LLM systems; upstream currently has no repository-level license, so treat it as read-only reference material |
