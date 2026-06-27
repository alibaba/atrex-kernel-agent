# Gluon CDNA4 (gfx950) Kernel Collection

Reference kernel implementations of the Gluon framework on the AMD CDNA4 (gfx950) architecture.

---

| File | Description |
|------|-------------|
| [matmul_triton.py](matmul_triton.py) | Triton raw matmul baseline |
| [bench_matmul.py](bench_matmul.py) | Matmul performance benchmark |
| [matmul_gluon_gfx950_256x256x64_2stage.py](matmul_gluon_gfx950_256x256x64_2stage.py) | 256x256x64 2-stage pipeline |
| [matmul_gluon_gfx950_256x256x64_2stage_pingpong.py](matmul_gluon_gfx950_256x256x64_2stage_pingpong.py) | 256x256x64 2-stage ping-pong |
| [matmul_gluon_gfx950_256x256x64_2stage_scheduling.py](matmul_gluon_gfx950_256x256x64_2stage_scheduling.py) | 256x256x64 2-stage instruction scheduling |
| [matmul_gluon_gfx950_256x256x32_3stage.py](matmul_gluon_gfx950_256x256x32_3stage.py) | 256x256x32 3-stage pipeline |
| [matmul_gluon_gfx950_256x256x32_3stage_pingpong.py](matmul_gluon_gfx950_256x256x32_3stage_pingpong.py) | 256x256x32 3-stage ping-pong |
| [matmul_gluon_gfx950_256x256x32_3stage_scheduling.py](matmul_gluon_gfx950_256x256x32_3stage_scheduling.py) | 256x256x32 3-stage instruction scheduling |

**aiter Production-Grade Gluon Kernels (source: [aiter](https://github.com/ROCm/aiter))**

| File | Description |
|------|-------------|
| [gemm_a8w8.py](gemm_a8w8.py) | A8W8 GEMM (Gluon MFMA control) |
| [gemm_a8w8_blockscale.py](gemm_a8w8_blockscale.py) | A8W8 Block-Scale GEMM (AMDMFMALayout + SwizzledSharedLayout) |
| [gemm_afp4wfp4.py](gemm_afp4wfp4.py) | MXFP4 GEMM |
| [gemm_a16w16.2stage.gfx950.py](gemm_a16w16.2stage.gfx950.py) | A16W16 2-stage GEMM |
| [gemm_a16w16.2stage.pingpong.gfx950.py](gemm_a16w16.2stage.pingpong.gfx950.py) | A16W16 2-stage ping-pong |
| [gemm_a16w16.3stage.gfx950.py](gemm_a16w16.3stage.gfx950.py) | A16W16 3-stage GEMM |
| [gemm_a16w16.3stage.pingpong.gfx950.py](gemm_a16w16.3stage.pingpong.gfx950.py) | A16W16 3-stage ping-pong |
| [pa_decode_gluon.py](pa_decode_gluon.py) | Paged Attention Decode (Gluon) |
| [pa_mqa_logits.py](pa_mqa_logits.py) | Paged Attention MQA Logits (Gluon) |
