# NVIDIA A100 SXM 80 GB Hardware Compute Specifications

**Last Updated**: 2026-07-20

This page uses NVIDIA's published A100 80 GB SXM figures. Values in the
"With sparsity" column assume NVIDIA's documented structured sparsity mode;
use the dense column for ordinary kernels.

## NVIDIA A100 SXM 80 GB (Ampere / sm_80)

| Precision | Dense peak | With sparsity |
|------|------:|------:|
| FP64 (CUDA Core) | 9.7 TFLOPS | — |
| FP64 (Tensor Core) | 19.5 TFLOPS | — |
| TF32 (Tensor Core) | 156 TFLOPS | 312 TFLOPS |
| FP16 / BF16 (Tensor Core) | 312 TFLOPS | 624 TFLOPS |
| INT8 (Tensor Core) | 624 TOPS | 1,248 TOPS |

### Memory and execution resources

| Parameter | Value |
|------|------|
| VRAM | 80 GB HBM2e |
| Memory bandwidth | 2,039 GB/s |
| Streaming Multiprocessors | 108 |
| CUDA cores | 6,912 |
| Third-generation Tensor Cores | 432 |
| L2 cache | 40 MB |
| Configurable shared memory per SM | Up to 164 KB |
| SXM TDP | 400 W |
| Compute capability | 8.0 (`sm_80`) |

## Dense Roofline Ridge Points

| Precision | Calculation | Ridge point |
|------|------|------:|
| FP16 / BF16 Tensor | 312 / 2.039 | ~153 FLOPs/Byte |
| TF32 Tensor | 156 / 2.039 | ~77 FLOPs/Byte |
| FP64 Tensor | 19.5 / 2.039 | ~9.6 FLOPs/Byte |
| FP64 CUDA | 9.7 / 2.039 | ~4.8 FLOPs/Byte |

Do not use Hopper `wgmma`, TMA, thread-block clusters, or Blackwell
`tcgen05`/TMEM guidance for A100. Ampere Tensor Core kernels use the SM80
warp-level MMA path; asynchronous global-to-shared pipelines can use
`cp.async`.

## Official Sources

- [NVIDIA A100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/a100/)
- [NVIDIA A100 Tensor Core GPU datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf)
- [Ampere architecture tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)

