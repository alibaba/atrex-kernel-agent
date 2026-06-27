# Matrix Multiplication Patterns (NVIDIA Hopper wgmma)

## Hopper wgmma Patterns

### Triton (BEFORE)
```python
accumulator = tl.dot(a, b, accumulator)
```

### Gluon Hopper (AFTER)
```python
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    warpgroup_mma,
    warpgroup_mma_wait,
)

# Step 1: Place operands in shared memory (NVMMASharedLayout)
a_smem = gl.allocate_shared_memory(
    element_ty=a.dtype,
    shape=[BLOCK_M, BLOCK_K],
    layout=nvmma_shared_a,  # NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, transposed=False)
    value=a
)

b_smem = gl.allocate_shared_memory(
    element_ty=b.dtype,
    shape=[BLOCK_K, BLOCK_N],
    layout=nvmma_shared_b,  # NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, transposed=True/False)
    value=b
)

# Step 2: fence → wgmma → wait three steps
fence_async_shared()
acc = warpgroup_mma(a_smem, b_smem, acc, is_async=True)
acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
```

---

### ⚠️ Key Differences from AMD MFMA

| Aspect | AMD MFMA | Hopper wgmma |
|------|----------|-------------|
| Operand Source | register (loaded from smem via `DotOperandLayout`) | **shared memory** (direct read) |
| Requires DotOperandLayout | ✅ Required | ❌ Not required |
| Synchronization Requirement | No special requirements | `fence_async_shared()` + `warpgroup_mma_wait()` |
| Async Mode | N/A | `is_async=True` |
| Accumulator Layout | `AMDMFMALayout` | `NVMMADistributedLayout` |
| Shared Layout | `SwizzledSharedLayout` | `NVMMASharedLayout` |

```python
# AMD MFMA (requires DotOperandLayout, load from smem to register):
dot_op0 = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=4)
dot_op1 = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=4)
a_dot = a_smem.load(dot_op0)
b_dot = b_smem.load(dot_op1)
acc = gl.amd.cdna3.mfma(a_dot, b_dot, acc)

# Hopper wgmma (operands are directly smem descriptors, no DotOperandLayout needed):
fence_async_shared()
acc = warpgroup_mma(a_smem, b_smem, acc, is_async=True)
acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
```

---

## Layout Selection

Extracted from TTGIR (Hopper-specific mappings):
- `#ttg.nvidia_mma<{...}>` → `NVMMADistributedLayout(version=[major,minor], ...)`
- `#ttg.nvmma_shared<{...}>` → `NVMMASharedLayout(...)`
- `#ttg.swizzled_shared<{...}>` → `SwizzledSharedLayout(...)` (non-wgmma scenarios)

### Example
```python
# TTGIR: #mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
mma: gl.constexpr = gl.NVMMADistributedLayout(
    version=[3, 0],
    warps_per_cta=[4, 1],
    instr_shape=[16, 16, 16],
)

# TTGIR: #shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
shared_w: gl.constexpr = gl.NVMMASharedLayout(
    swizzle_byte_width=128,
    element_bitwidth=16,
    transposed=False,
)

# TTGIR: #shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
shared_k: gl.constexpr = gl.NVMMASharedLayout(
    swizzle_byte_width=128,
    element_bitwidth=16,
    transposed=True,
)
```

---

## Complete wgmma Loop Example

```python
# (NVMMADistributedLayout)
accumulator = gl.zeros((64, BV), dtype=gl.float32, layout=mma)

# smem (double-buffered)
smem_w = gl.allocate_shared_memory(gl.bfloat16, [2, BT, 64], shared_w)
smem_k = gl.allocate_shared_memory(gl.bfloat16, [2, 64, BT], shared_k)

# PROLOGUE: async_copy prefetch
async_copy.async_copy_global_to_shared(smem_w.index(0), w_ptr_tensor, mask=w_mask)
async_copy.async_copy_global_to_shared(smem_k.index(0), k_ptr_tensor, mask=k_mask)
async_copy.commit_group()
async_copy.wait_group(num_outstanding=0)

# MAIN LOOP
for i_t in range(NT):
    cur_slot = i_t % 2

 # registerdatacreatetemporary smem -> wgmma
    h_smem = gl.allocate_shared_memory(gl.bfloat16, [64, BV], shared_v, value=b_h.to(gl.bfloat16))
    fence_async_shared()
    b_v_new = warpgroup_mma(smem_w.index(cur_slot), h_smem, b_v_new, is_async=True)
    b_v_new = warpgroup_mma_wait(num_outstanding=0, deps=(b_v_new,))

    # k × v_new
    vn_smem = gl.allocate_shared_memory(gl.bfloat16, [BT, BV], shared_v, value=b_v_new_bf16)
    fence_async_shared()
    b_h = warpgroup_mma(smem_k.index(cur_slot), vn_smem, b_h, is_async=True)
    b_h = warpgroup_mma_wait(num_outstanding=0, deps=(b_h,))

 # prefetchnextiteration
    if i_t < NT - 1:
        async_copy.async_copy_global_to_shared(smem_w.index((i_t+1) % 2), ...)
        async_copy.async_copy_global_to_shared(smem_k.index((i_t+1) % 2), ...)
        async_copy.commit_group()
```

## Key Points

1. **wgmma operands must be in shared memory with NVMMASharedLayout**
2. **DotOperandLayout is not needed** (the biggest difference from AMD)
3. **Three-step pattern: `fence_async_shared()` → `warpgroup_mma(is_async=True)` → `warpgroup_mma_wait(deps=...)`**
4. **Accumulator uses float32, converted to `.to(gl.bfloat16)` at the end**
5. **transposed parameter**: see the `transposed` field of `nvmma_shared` in TTGIR, the k matrix usually requires `transposed=True`

## Related Documents

- **Cross-architecture reference**: [CDNA3 Matrix Multiplication](../../amd/cdna3/matrix_multiply.md) (mfma) | [CDNA4 Matrix Multiplication](../../amd/cdna4/matrix_multiply.md) (mfma)
- **⚠️ Key difference**: This article uses wgmma (operands read directly from shared memory), while AMD uses mfma (operands read from registers)
- **ISA reference**: [PTX MMA Instruction Evolution](../../../ref-docs/nvidia/common/nvidia-ptx-mma-instructions.md)
- **Optimization downstream**: [Hopper GEMM Optimization](../../../ref-docs/nvidia/gluon/sm90/matmul.md)
