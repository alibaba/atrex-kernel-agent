# Gluon Tutorial 06: Tensor Memory and tcgen05_mma

This tutorial covers Blackwell's Tensor Memory (TMEM) — a dedicated on-chip memory space for asynchronous MMA instructions — and demonstrates how to use tcgen05_mma to build tiled matrix multiplication kernels with pipeline optimization.

---

## 1. Tensor Memory Introduction

Tensor Memory is a 2D memory space with 128 rows × 512 columns of 32-bit cells per CTA in the Blackwell architecture. Key characteristics:

- Faster access than shared memory, but with additional constraints
- Each warp can only access 32 rows based on its warp ID; the full warp group (4 warps) is needed to access all 128 physical rows
- Allocated by column count, which must be in [32, 512] and a power of 2
- In Gluon, load/store operations require 4 or 8 warps
- Only 2D tensors can interact with Tensor Memory
- Data can be asynchronously copied from shared memory to Tensor Memory (API not yet exposed in Gluon)

Additional notes:

- TMEM is essentially an extra register file: 128 × 512 = 65,536 32-bit cells = 256KB, matching the total register file size per SM
- TMEM can be used independently of MMA instructions as an alternative to shared memory for data transfer
- TMEM is dynamically allocated on-SM; allocation blocks if space is insufficient (does not directly affect occupancy)

### 1.1 Tensor Memory Layout

```python
TensorMemoryLayout(
    block=(blockM, blockN),
    unpacked=True,
)
```

The tensor is partitioned into (blockM, blockN) blocks where:
- blockM must be 64 or 128 (logical; physically each warp accesses only 32 rows)
- blockN must be a power of 2 in [1, 256]

When blockM=64, tensors with multiple blocks are packed in TMEM to fully utilize all 128 physical rows. The underlying `tcgen05.st` and `tcgen05.ld` instructions are warp-level and access TMEM in specific patterns, constraining register layouts.

## 2. TMEM Usage Example

The following kernel demonstrates reading and writing 2D data through Tensor Memory:
`input: global memory → registers → Tensor Memory → output: registers → global memory`

```python
@gluon.jit
def tmem_example_kernel(in_ptr, out_ptr, M: gl.constexpr, N: gl.constexpr, num_warps: gl.constexpr):
    global_memory_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])
    offs_m = gl.arange(0, M, gl.SliceLayout(1, global_memory_layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, global_memory_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]

    # global memory -> registers
    input = gl.load(in_ptr + offs)

    # Define Tensor memory layout
    tmem_layout: gl.constexpr = TensorMemoryLayout(
        block=(64, 64),
        col_stride=32 // in_ptr.dtype.element_ty.primitive_bitwidth,
    )

    # Allocate Tensor memory
    tmem = allocate_tensor_memory(
        element_ty=in_ptr.dtype.element_ty, shape=[M, N], layout=tmem_layout,
    )

    # Get register layout required for TMEM access
    tmem_reg_layout: gl.constexpr = get_tmem_reg_layout(
        in_ptr.dtype.element_ty, (M, N), tmem_layout, num_warps=num_warps,
    )

    # Convert register layout for TMEM access
    input = gl.convert_layout(input, tmem_reg_layout)

    # registers -> Tensor memory
    tmem.store(input)

    # Tensor memory -> registers
    output = tmem.load(tmem_reg_layout)

    # Convert to global memory layout
    output = gl.convert_layout(output, global_memory_layout)

    # registers -> global memory
    gl.store(out_ptr + offs, output)
```

## 3. tcgen05_mma: Single-Tile Matrix Multiply

The `tcgen05_mma` API computes D = A @ B + C on a single tensor block. Key constraints:
- Accumulator must reside in TMEM
- LHS operand can be in SMEM or TMEM
- RHS operand must be in SMEM
- SMEM operands must use `NVMMASharedLayout`

```python
@gluon.jit
def small_mma_kernel(a_desc, b_desc, c_desc, d_desc, tmem_block: gl.constexpr,
                     LHS_IN_TMEM: gl.constexpr, USE_COMMIT: gl.constexpr, num_warps: gl.constexpr):
    # Load A, B, C tiles via TMA
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)
    c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)

    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes + c_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [0, 0], bar, a_smem)
    tma.async_copy_global_to_shared(b_desc, [0, 0], bar, b_smem)
    tma.async_copy_global_to_shared(c_desc, [0, 0], bar, c_smem)
    mbarrier.wait(bar, phase=0)

    # Reinitialize barrier (reusing between TMA and tcgen05_mma may cause UB)
    mbarrier.invalidate(bar)
    mbarrier.init(bar, count=1)

    # Set up accumulator in TMEM
    acc_tmem_layout: gl.constexpr = TensorMemoryLayout(tmem_block.value, col_stride=32 // d_desc.dtype.primitive_bitwidth)
    acc_tmem = allocate_tensor_memory(d_desc.dtype, [M, N], acc_tmem_layout)
    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(d_desc.dtype, (M, N), acc_tmem_layout, num_warps)

    # Initialize accumulator from C (via tcgen05_copy or register transfer)
    if M == 128:
        tcgen05_copy(c_smem, acc_tmem)
        tcgen05_commit(bar)
        mbarrier.wait(bar, phase=0)
        mbarrier.invalidate(bar)
        mbarrier.init(bar, count=1)
    else:
        acc = c_smem.load(acc_reg_layout)
        acc_tmem.store(acc)

    # Optionally place LHS in TMEM
    if LHS_IN_TMEM:
        lhs_tmem = allocate_tensor_memory(a_desc.dtype, [M, K], lhs_tmem_layout)
        lhs = a_smem.load(lhs_reg_layout)
        lhs_tmem.store(lhs)
        a = lhs_tmem
    else:
        a = a_smem

    # Issue async MMA and wait
    if USE_COMMIT:
        tcgen05_mma(a, b_smem, acc_tmem)
        tcgen05_commit(bar)
    else:
        tcgen05_mma(a, b_smem, acc_tmem, mbarriers=[bar], mbarrier_preds=[True])
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    # Store result via TMA
    d_smem = gl.allocate_shared_memory(d_desc.dtype, d_desc.block_type.shape, d_desc.layout)
    acc = acc_tmem.load(acc_reg_layout)
    d_smem.store(acc)
    fence_async_shared()
    tma.async_copy_shared_to_global(d_desc, [0, 0], d_smem)
    tma.store_wait(pendings=0)
```

Important notes on `tcgen05_mma`:
- It is asynchronous; accumulator memory and operand memory must not be read/written until completion
- Completion is tracked via mbarrier with `tcgen05_commit`
- A `fence_async_shared()` is required between SMEM writes and `tcgen05_mma` to ensure memory ordering
- `use_acc=False` efficiently zero-initializes the accumulator

## 4. Blocked Matrix Multiplication with tcgen05_mma

Each program computes one output tile: C = A @ B.

```python
@gluon.jit
def blocked_matmul_kernel(a_desc, b_desc, c_desc, TRANSPOSE_B: gl.constexpr, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    K = a_desc.shape[1]

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_smem = gl.allocate_shared_memory(dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, b_desc.block_type.shape, b_desc.layout)
    tma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())

    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    use_acc = False
    for k in range(0, K, BLOCK_K):
        mbarrier.expect(tma_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, k], tma_bar, a_smem)
        tma.async_copy_global_to_shared(b_desc, [k, off_n], tma_bar, b_smem)
        mbarrier.wait(tma_bar, phase=phase)

        # Transpose B via permuted SMEM view (hardware handles transposition)
        b = b_smem.permute((1, 0)) if TRANSPOSE_B else b_smem

        tcgen05_mma(a_smem, b, acc_tmem, use_acc=use_acc)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase=phase)
        use_acc = True
        phase ^= 1

    # Epilogue: store result
    acc = acc_tmem.load(acc_reg_layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)
```

### 4.1 Performance Results (No Pipeline)

Matrix size: 8192×8192×16K, fp16:

| BLOCK_M | BLOCK_N | BLOCK_K | num_warps | Time (ms) | TFLOPS |
|---------|---------|---------|-----------|-----------|--------|
| 64 | 64 | 64 | 4 | 3.27 | 671.77 |
| 64 | 64 | 128 | 4 | 3.33 | 660.93 |
| 128 | 128 | 64 | 4 | 2.45 | 898.61 |
| 128 | 128 | 128 | 4 | 2.16 | 1019.46 |

Achieves 1020 TFLOPS without pipelining.

## 5. Implicit Pipeline Property of tcgen05_mma

Since `tcgen05_mma` is asynchronous, hardware guarantees in-order execution without explicit synchronization in these cases:

1. **Consecutive tcgen05_mma** with same shape and accumulator dtype — hardware ensures sequential execution
2. **tcgen05_mma followed by tcgen05_commit** — commit waits for MMA completion
3. **tcgen05_copy followed by tcgen05_mma** — copy completes before MMA starts

This enables issuing multiple MMA operations without explicit barriers between them, allowing fine-grained pipeline scheduling when combined with mbarrier completion tracking.

## 6. Pipelined Matrix Multiplication

The pipelined kernel processes two vertically adjacent output tiles (Upper and Lower) simultaneously, using double-buffered operands to overlap TMA loads with MMA computation.

Design:
- Each program computes 2×BLOCK_M rows of output
- U = Upper A tile, V = Lower A tile, B = shared B tile
- Load-to-MMA ratio is 3:2 (3 TMA loads per 2 MMA operations)
- Double-buffered: current iteration computes while next iteration's data loads

SMEM usage (BLOCK_M=BLOCK_N=128, BLOCK_K=128):
- u_bufs: 2 × 128 × 128 × 2B = 64 KB
- v_bufs: 2 × 128 × 128 × 2B = 64 KB
- b_bufs: 2 × 128 × 128 × 2B = 64 KB
- Total: 192 KB (within 228 KB SM limit)

Pipeline schedule:
1. Prefetch first 2 iterations (U1,B1,V1 and U2,B2,V2)
2. Main loop: Wait → Compute UB → Wait → Compute VB → Wait MMA complete → Load next
3. Epilogue: Store Upper result, then Lower result via TMA

### 6.1 Pipeline Performance Results

| BLOCK_M | BLOCK_N | BLOCK_K | num_warps | Time (ms) | TFLOPS |
|---------|---------|---------|-----------|-----------|--------|
| 128 | 128 | 64 | 4 | 2.20 | 1000.51 |
| 128 | 128 | 64 | 8 | 1.97 | 1113.49 |
| 128 | 128 | 128 | 4 | 2.21 | 1040.27 |
| 128 | 128 | 128 | 8 | 2.17 | 1011.47 |

Best configuration achieves **1113 TFLOPS** (BLOCK_K=64, 8 warps), a ~9% improvement over the non-pipelined version.

### 6.2 Performance Analysis

**BLOCK_K=64 vs 128**: Smaller BLOCK_K uses less SMEM (96KB vs 192KB), enabling 2 blocks/SM instead of 1. Higher occupancy provides better latency hiding despite doubling iteration count. This indicates the kernel is memory-latency-bound.

**num_warps=8 benefit**: With BLOCK_K=64, the epilogue (TMEM load → layout conversion → SMEM store → TMA writeback) constitutes ~15% of runtime. More warps accelerate these parallel operations. With BLOCK_K=128, the epilogue is only ~5% of runtime, so extra warps provide negligible benefit.

## 7. Future Directions

Warp specialization enables more effective fine-grained pipeline optimization by dedicating different warps to load, compute, and store tasks, further reducing idle time.
