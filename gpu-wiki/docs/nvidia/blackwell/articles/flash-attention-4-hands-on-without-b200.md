# FlashAttention-4 Hands-On Without B200 Hardware

A practical guide to experimenting with FlashAttention-4 on cloud platforms without owning Blackwell hardware, including benchmarking and code modification workflows.


**Last updated**: 2026-06-30

---

## 1. Running FA4 Without B200

FlashAttention-4 targets the Blackwell architecture (sm_100). Consumer cards like the 5090 cannot use TMEM. For hands-on experimentation — testing performance, understanding the source of gains, or modifying code — cloud platforms provide access.

The recommended approach is **Modal**, which provides B200 and H100 instances. The free tier (30 USD/month) is more than sufficient for operator testing without training workloads.

## 2. Benchmark Script

Usage is straightforward:

```bash
pip install modal
modal run fa4_benchmark.py
```

No other installation required — works out of the box.

## 3. Modifying the Code

Start with the official FlashAttention-4 implementation. The main kernel logic is in `flash_attn/flash_fwd_sm100.py`.

As a test: changing `block_m` from 128 to 64 (intentionally not saturating tcgen05) produced an expected performance drop — but **only a 2.5% decrease**, less than anticipated. This suggests the pipeline design is robust to tile size variations within a range.

## 4. Learning Resources

For those unfamiliar with CuTe DSL, Simon's blog (veitner.bearblog.dev) provides an accessible introduction to the concepts underlying the DSL.

## 5. Practical Notes

- **Nsight Compute profiling:** Modal does not support ncu — cloud platforms lack hardware counter permissions.
- **Alternative hardware:** Jetson Thor (3499 USD, sm110 architecture with tcgen05 support) is an option for local profiling access.


## Related

- [Comprehensive Guide to NVIDIA Blackwell Architecture](blackwell-architecture-comprehensive-guide.md)
- [GPGPU Architecture: Blackwell Instruction Analysis](blackwell-architecture-instruction-analysis.md)
- [Blackwell GPGPU Architecture New Features Overview](blackwell-gpgpu-new-features-overview.md)
- [NVIDIA Blackwell Tensor Core Analysis (Part 2): B300](blackwell-tensor-core-analysis-b300.md)
- [NVIDIA Blackwell Tensor Core Analysis (Part 1)](blackwell-tensor-core-analysis-part1.md)
- [FlashAttention 1–4: GPU Generational Evolution](../../common/flash-attention-1-to-4-gpu-evolution.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
