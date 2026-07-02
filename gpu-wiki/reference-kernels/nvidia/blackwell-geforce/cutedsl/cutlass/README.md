# Blackwell GeForce CuTeDSL CUTLASS Kernels

> **Usability status:** `requires-external-checkout`
>
> Set `$CUTLASS_DIR` to a compatible CUTLASS checkout before building or running these files.

Reference kernel implementations in the CUTLASS framework using CuTeDSL on the Blackwell GeForce (SM120) architecture.

---

| Kernel | Description |
|--------|-------------|
| [dense_gemm.py](dense_gemm.py) | Dense GEMM |
| [sm120_nvfp4_inline_ptx_gemm.py](sm120_nvfp4_inline_ptx_gemm.py) | NVFP4 `m16n8k64` inline PTX **atom demo** on SM120 (single-tile correctness reference) |
| [test_sm120_nvfp4_inline_ptx_gemm.py](test_sm120_nvfp4_inline_ptx_gemm.py) | 10-seed regression test for the above NVFP4 demo |
| [sm120_nvfp4_persistent_gemm_pro5000.py](sm120_nvfp4_persistent_gemm_pro5000.py) | SM120 NVFP4 **fully optimized persistent GEMM** (tuned for RTX PRO 5000): BLOCK_K=128, STAGES=4, compressed SF, SF cp.async prefetching, persistent NUM_CTAS=110. **581 TFLOPS on 4096³ = 71% of CUTLASS C++** (8.7× improvement vs v15 baseline). See [sm120-nvfp4-persistent-gemm-pro5000-optimization.md](../../../../../docs/nvidia/blackwell-geforce/cutedsl/sm120-nvfp4-persistent-gemm-pro5000-optimization.md) and [nvfp4-gemm-pitfalls.md](../../../../../docs/nvidia/blackwell-geforce/cutedsl/pitfalls/nvfp4-gemm-pitfalls.md) for details. |
| [sm120_nvfp4_pack_helpers.py](sm120_nvfp4_pack_helpers.py) | SF packing helpers for the above persistent GEMM: `pack_sf_per_block` provides an 8× compact SF gmem layout (replacing the `pack_sf_per_atom` default layout, where 7/8 of each atom's `(128, 4)` bytes are zero-padded). |
