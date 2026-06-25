# Ampere CuTeDSL CUTLASS Kernels

Reference kernel implementations of the CUTLASS framework using CuTeDSL on the Ampere architecture.

---

| Kernel | Description |
|--------|-------------|
| [call_bypass_dlpack.py](call_bypass_dlpack.py) | Kernel invocation bypassing DLPack |
| [call_from_jit.py](call_from_jit.py) | JIT compilation invocation example |
| [cooperative_launch.py](cooperative_launch.py) | Cooperative kernel launch |
| [dynamic_smem_size.py](dynamic_smem_size.py) | Dynamic shared memory size configuration |
| [elementwise_add.py](elementwise_add.py) | Element-wise addition kernel |
| [elementwise_add_autotune.py](elementwise_add_autotune.py) | Element-wise addition (auto-tuned version) |
| [elementwise_apply.py](elementwise_apply.py) | Element-wise generic operator |
| [experimental_memcpy_simt_universal_copy.py](experimental_memcpy_simt_universal_copy.py) | Experimental SIMT universal memory copy |
| [flash_attention_v2.py](flash_attention_v2.py) | Flash Attention V2 implementation |
| [hstu_attention.py](hstu_attention.py) | HSTU Attention implementation |
| [inline_ptx.py](inline_ptx.py) | Inline PTX instruction example |
| [sgemm.py](sgemm.py) | Single-precision GEMM (SGEMM) |
| [smem_allocator.py](smem_allocator.py) | Shared memory allocator |
| [tensorop_gemm.py](tensorop_gemm.py) | Tensor Core GEMM |
