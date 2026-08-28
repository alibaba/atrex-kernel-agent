# Reference Projects

GPU kernel optimization reference projects, managed as git submodules.

These are **design references, read for ideas only**. Executing, importing, or copying
this source into a candidate is out of contract — see `orchestrator/prompts/framework_baseline.md`.

## Usage

```bash
# Initialize and clone all submodules (full history; these are not shallow)
git submodule update --init

# Update all submodules to latest
git submodule update --remote
```

The t-head PPU projects are cloned over SSH (`git@github.com:t-head/...`), so an SSH key
with access to that org is required for them.

## Included Projects

Pick by target hardware first: a PPU kernel is written in HGGC against AIU and tensor-cell
intrinsics, so NVIDIA and AMD material describes strategy but not the instructions available.

### T-Head PPU (ZW-M890P, HGGC / SAIL)

| Project | Vendor/Arch | DSL | Operators | When to use |
|---|---|---|---|---|
| flash-attention-for-sail | ppu / zwm890p | cuda (hggc) | flash-attention, paged-attention | Vendor FA2/FA3 port. `flash_attn_with_kvcache` is a superset of the production operator contract; `hopper/paged_kv.h` and `hopper/pack_gqa.h` hold the AIU load and GQA packing strategy. |
| actlize | ppu / zwm890p | cuda (hggc) | gemm, conv | CUTLASS 3.6.0 / CuTe for PPU. `include/cute/arch/mma_ppu0015.hpp` and `copy_ppu0015_aiu.hpp` are the MMA and AIU/swizzle atoms; `examples/08_ppu_basic_tensor_op_gemm` is the way in. |
| triton-for-sail | ppu / zwm890p | triton, gluon | any | The PPU Triton compiler itself. `python/test/unit/ppu/aiu/` is the per-op AIU lowering contract; `perf/06-fused-attention-aiu.py` is fused attention on the AIU path. |
| DeepGEMM-for-sail | ppu / zwm890p | cuda (hggc) | gemm, mqa-logits | PPU GEMM tiling and heuristics; `deep_gemm/include/deep_gemm/ppu_paged_mqa_logits.cuh` for paged MQA. |
| FlashMLA-for-sail | ppu / zwm890p | cuda (hggc) | mla, paged-attention | MLA attention on PPU; `csrc/kerutils/include/kerutils/device/ppu/` has PPU softmax, mask, and dequant primitives. |
| sailify | ppu / zwm890p | cuda (hggc) | any | CUDA -> HGGC translator. `sailify/cuda_to_ppu_mappings.py` is the authoritative symbol map (`cudaStream_t` -> `hggcStream_t`, cublas -> acblas, NCCL -> pccl). |
| hggc-samples | ppu / zwm890p | cuda (hggc) | any | Official SAIL 2.1 samples. `Samples/3_HGGC_Features/bf16_tensor_cell_gemm` for tensor-cell usage; `Common/helper_hggc.h` for the error-checking idiom. |

### NVIDIA

| Project | Vendor/Arch | DSL | Operators | When to use |
|---|---|---|---|---|
| cutlass | nvidia / ampere-blackwell | cuda, cutedsl | gemm, conv | CuTe layouts, MMA and TMA atoms, collective builders. |
| flash-attention | nvidia / ampere-hopper | cuda | flash-attention, paged-attention | Upstream FA2/FA3, including the varlen and paged KV paths. |
| flashinfer | nvidia / ampere-blackwell | cuda, triton | paged-attention, decode, sampling | LLM serving kernels; page-table and append-KV layouts. |
| FlashMLA | nvidia / hopper | cuda | mla | DeepSeek MLA attention. |
| DeepGEMM | nvidia / hopper-blackwell | cuda | gemm | FP8 GEMM with JIT tiling heuristics. |
| cute-gemm | nvidia / ampere-hopper | cuda | gemm | Small, readable CuTe GEMM walkthroughs. |
| cutex | nvidia / any | cuda | any | CUDA template extensions. |
| cuLA | nvidia / any | cuda | gemm, linalg | Linear algebra kernels. |
| LeetCUDA | nvidia / any | cuda | many | Wide catalogue of short, self-contained CUDA kernels. |
| quack | nvidia / hopper-blackwell | cutedsl | norm, softmax, elementwise | CuTeDSL memory-bound kernels. |

### AMD

| Project | Vendor/Arch | DSL | Operators | When to use |
|---|---|---|---|---|
| composable_kernel | amd / cdna3-cdna4 | hip | gemm, conv, attention | CK tile programming model and pipelines. |
| aiter | amd / cdna3-cdna4 | hip, triton | attention, gemm, moe | AMD's tuned operator library. |
| FlyDSL | amd / cdna3-cdna4 | flydsl | any | AMD FlyDSL kernels and lowering. |

### Vendor-neutral

| Project | Vendor/Arch | DSL | Operators | When to use |
|---|---|---|---|---|
| triton | any | triton, gluon | any | Upstream Triton and Gluon, including per-vendor backends. |
| tilelang | any | tilelang | gemm, attention | TileLang DSL, layouts, and pipelining. |
| hpc-ops | any | cuda, hip | many | Tencent HPC operator collection. |
