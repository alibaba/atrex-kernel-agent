# Software Pipeline (Cross-Architecture Differences)

Software pipeline patterns extracted from `reference-kernels/amd/`, covering differences across three architectures: CDNA3 (gfx942), CDNA4 (gfx950), and RDNA4 (gfx1250).


**Last updated**: 2026-06-30

---

## CDNA3: Register-Based Pipeline

**Source**: `cdna/flydsl/FlyDSL/`, CDNA3 converter docs

```python
# CDNA3 does not have hardware async copy
# Pipeline implemented via register buffers: global → register → LDS

# Prologue: Prefetch first batch of data
a_reg = buffer_load(a_ptr)       # global → register
b_reg = buffer_load(b_ptr)       # global → register

for k in range(1, K_tiles):
    # Write current data to LDS
    lds_store(smem_a, a_reg)
    lds_store(smem_b, b_reg)

    # Simultaneously prefetch next batch
    a_reg_next = buffer_load(a_ptr + k * stride)
    b_reg_next = buffer_load(b_ptr + k * stride)

    barrier()

    # Compute
    acc = mfma(smem_a, smem_b, acc)

    a_reg = a_reg_next
    b_reg = b_reg_next
```

**CDNA3 Insights**:
- Data must go through register staging (global → reg → LDS), consuming additional VGPRs
- Double buffering requires 2 sets of register buffers, putting high pressure on VGPRs
- Triple buffering is generally not worthwhile (VGPR spilling leads to scratch access)

---

## CDNA4: Hardware Async DMA Pipeline

```python
# CDNA4 supports hardware async copy: global → LDS, bypassing registers
from triton.experimental.gluon.language.amd.cdna4 import async_copy

for k in range(K_tiles):
    # Async copy: global → LDS, no register usage
    async_copy.buffer_load_to_shared(smem_a, a_ptr + k * stride)
    async_copy.buffer_load_to_shared(smem_b, b_ptr + k * stride)
    async_copy.commit()

    # Wait for previous copy to complete
    async_copy.wait(0)

    # Compute
    acc = tl.dot(smem_a, smem_b, acc)
```

**CDNA4 Insights**:
- Async copy bypasses registers, significantly reducing VGPR pressure
- Supports deeper pipelines (3-4 stages) since it does not consume registers
- Similar to NVIDIA Hopper's cp.async, but with a different implementation mechanism

---

## RDNA4: TDM Async Copy

```python
# RDNA4 (gfx1250) introduces TDM (Tensor Data Mover)
# Similar to NVIDIA TMA, supports 2D descriptor

from triton.experimental.gluon.language.amd.rdna4 import tdm

# Create 2D descriptor
desc = tdm.make_2d_descriptor(base_ptr, width, height, stride)

# Async load
tdm.async_copy_2d(smem, desc, offset_x, offset_y)
```

---

## Related

- **Tuning Guide**: AMD GPU Kernel Tuning Guide — CDNA3 vs CDNA4 Hardware Specification Comparison
- **General Execution Model**: [GPU Execution Model and Thread Optimization](../../../generic/gpu-execution-model.md)
- **CDNA4 FP8 in Practice**: [CDNA4 FP8 GEMM Optimization in Practice](../gfx950/cdna4-fp8-gemm-optimization.md)
- **General Triton Patterns**: [Triton Optimization Patterns in Practice](../../../generic/hands-on/README.md)
