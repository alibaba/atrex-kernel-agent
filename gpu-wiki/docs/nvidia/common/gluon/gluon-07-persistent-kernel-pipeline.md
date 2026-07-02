# Gluon Tutorial 07: Persistent Kernels and Pipeline Optimization

This tutorial covers building a unified MMA abstraction layer across Hopper and Blackwell, implementing pipelined matrix multiplication, persistent kernel design with tile schedulers, and outer-loop pipeline optimization using the STEALB buffer-borrowing technique.


**Last updated**: 2026-06-30

---

## 1. Unified MMA Abstraction Layer

To support both Hopper (WGMMA) and Blackwell (tcgen05_mma) with a single kernel, a unified abstraction is built using Gluon's `@aggregate` decorator. The API is designed around WGMMA's more constrained interface.

### 1.1 WGMMA Wrapper

```python
@aggregate
class WGMMA:
    acc: Union[warpgroup_mma_accumulator, gl.tensor]
    use_acc: gl.tensor

    @gluon.jit
    def initialize(dtype: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, num_warps: gl.constexpr):
        mma_layout: gl.constexpr = t5.pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
        acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)
        return WGMMA(acc, gl.to_tensor(False))

    @gluon.jit
    def issue_async_mma(self, a, b):
        acc = warpgroup_mma(a, b, self.acc, is_async=True, use_acc=self.use_acc)
        return WGMMA(acc, gl.to_tensor(True))

    @gluon.jit
    def wait_num_outstanding(self, num_outstanding: gl.constexpr):
        acc = warpgroup_mma_wait(num_outstanding, (self.acc, ))
        return WGMMA(acc, self.use_acc)

    @gluon.jit
    def take_result(self):
        return self.acc, WGMMA(self.acc, gl.to_tensor(False))
```

### 1.2 Blackwell MMAv5 Wrapper

The Blackwell wrapper allocates barriers and tracks issued MMA count to implement `wait_num_outstanding`:

```python
@aggregate
class MMAv5:
    use_acc: gl.tensor
    acc_tmem: tensor_memory_descriptor
    bar: gl.shared_memory_descriptor
    counter: gl.tensor
    reg_layout: gl.constexpr

    @gluon.jit
    def initialize(dtype: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, num_warps: gl.constexpr):
        layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
        acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], layout)
        bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(bar, count=1)
        reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), layout, num_warps)
        return MMAv5(gl.to_tensor(False), acc_tmem, bar, gl.to_tensor(0), reg_layout)

    @gluon.jit
    def issue_async_mma(self, a, b):
        tcgen05_mma(a, b, self.acc_tmem, use_acc=self.use_acc)
        tcgen05_commit(self.bar)
        return MMAv5(gl.to_tensor(True), self.acc_tmem, self.bar, self.counter + 1, self.reg_layout)

    @gluon.jit
    def wait_num_outstanding(self, num_outstanding: gl.constexpr):
        mbarrier.wait(self.bar, (self.counter - 1 - num_outstanding) & 1)
        return self

    @gluon.jit
    def take_result(self):
        next = MMAv5(gl.to_tensor(False), self.acc_tmem, self.bar, self.counter, self.reg_layout)
        return self.acc_tmem.load(self.reg_layout), next
```

Why Blackwell can use 4 warps while Hopper needs 8: On Blackwell, the accumulator resides in Tensor Memory rather than registers, significantly reducing register pressure. Hopper's 128×256 accumulator requires 256 registers/thread at 4 warps, hitting the limit.

## 2. Pipelined Matrix Multiplication

The kernel pipelines both loads and MMA operations using multiple buffers. Reusable components:

```python
@gluon.jit
def issue_loads(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs,
                num_buffers: gl.constexpr, pred=True):
    index = producer % num_buffers
    producer += 1
    bar = bars.index(index)
    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes, pred=pred)
    tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_bufs.index(index), pred)
    tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_bufs.index(index), pred)
    return producer

@gluon.jit
def issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers: gl.constexpr):
    index = consumer % num_buffers
    phase = consumer // num_buffers & 1
    consumer += 1
    mbarrier.wait(bars.index(index), phase)
    mma = mma.wait_num_outstanding(0)
    mma = mma.issue_async_mma(a_bufs.index(index), b_bufs.index(index))
    return consumer, mma
```

The main kernel prefetches `num_buffers - 2` loads, then alternates loads and MMA in steady state, draining remaining MMA at the end.

### 2.1 Performance Results

Matrix size: 8192×8192×16K, BLOCK_M=128, BLOCK_N=256:

| BLOCK_K | num_buffers | num_warps | Blackwell TFLOPS | Hopper TFLOPS |
|---------|-------------|-----------|-----------------|---------------|
| 128 | 2 | 4 | 735.96 | — |
| 128 | 2 | 8 | 697.97 | 489.26 |
| 64 | 3 | 4 | 1054.00 | — |
| 64 | 3 | 8 | 973.94 | 673.67 |
| 64 | 4 | 4 | 1175.70 | — |
| 64 | 4 | 8 | 1072.83 | 669.16 |

Hopper saturates at 3 buffers; Blackwell continues improving with 4 buffers, reflecting its higher MMA-to-memory throughput ratio.

## 3. Persistent Kernel Implementation

A persistent kernel wraps the computation in an outer loop, processing multiple tiles per program. Programs remain resident on-GPU until all work is done.

### 3.1 Tile Scheduler

Basic row-major scheduler with static work distribution:

```python
@aggregate
class PersistentTileScheduler:
    pid_start: gl.tensor
    pid_end: gl.tensor
    num_pid_m: gl.tensor

    @gluon.jit
    def initialize(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
        kernel_id = gl.program_id(axis=0)
        num_kernels = gl.num_programs(axis=0)
        num_pid_m = gl.cdiv(M, BLOCK_M)
        num_pid_n = gl.cdiv(N, BLOCK_N)
        num_pid = num_pid_m * num_pid_n
        pid_per_kernel = gl.cdiv(num_pid, num_kernels)
        pid_start = kernel_id * pid_per_kernel
        pid_end = min(pid_start + pid_per_kernel, num_pid)
        return PersistentTileScheduler(pid_start, pid_end, num_pid_m)

    @gluon.jit
    def get_num_tiles(self):
        return self.pid_end - self.pid_start

    @gluon.jit
    def get_tile(self, idx):
        pid = self.pid_start + idx
        pid_m = pid % self.num_pid_m
        pid_n = pid // self.num_pid_m
        return pid_m, pid_n
```

The persistent kernel initializes MMA state and barriers once before the outer loop, reusing them across tiles. Operand buffers are scoped inside the inner loop so the SMEM allocator can reuse memory with TMA store buffers.

### 3.2 Persistent Matmul Performance

| BLOCK_K | num_buffers | num_warps | Blackwell TFLOPS | Hopper TFLOPS |
|---------|-------------|-----------|-----------------|---------------|
| 128 | 2 | 4 | 712.25 | — |
| 64 | 3 | 4 | 1032.16 | — |
| 64 | 4 | 4 | 1142.26 | — |
| 64 | 3 | 8 | 938.81 | 661.11 |
| 64 | 4 | 8 | 1071.46 | 658.84 |

Hopper slightly improves, but Blackwell slightly regresses. NCU profiling shows persistent kernel L2 hit rate drops ~10% (52.93% vs 61.11%) due to unfavorable global memory access patterns.

## 4. Grouped Tile Scheduler

To improve L2 locality, tiles are grouped along the M dimension so that adjacent tiles share B matrix data in cache:

```python
def GroupedPersistentTileScheduler(GROUP_SIZE_M):
    GROUP_SIZE_M = gl.constexpr(GROUP_SIZE_M)

    @aggregate
    class GroupedPersistentTileSchedulerImpl:
        start_pid: gl.tensor
        num_pid_m: gl.tensor
        num_pid_in_group: gl.tensor
        num_pid: gl.tensor

        @gluon.jit
        def initialize(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
            start_pid = gl.program_id(axis=0)
            num_pid_m = gl.cdiv(M, BLOCK_M)
            num_pid_n = gl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_pid = num_pid_m * num_pid_n
            return GroupedPersistentTileSchedulerImpl(start_pid, num_pid_m, num_pid_in_group, num_pid)

        @gluon.jit
        def get_tile(self, idx):
            tile_id = self.start_pid + idx * gl.num_programs(axis=0)
            group_id = tile_id // self.num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(self.num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % self.num_pid_in_group) // group_size_m
            return pid_m, pid_n

    return GroupedPersistentTileSchedulerImpl
```

Results with GROUP_SIZE_M=8: Blackwell recovers to 1179.94 TFLOPS (L2 hit rate improves to 70%). On Hopper, grouped scheduling slightly hurts performance due to load imbalance.

## 5. Outer-Loop Pipeline Optimization

By deferring the TMA store wait, the current tile's store operation overlaps with the next tile's computation. This is most beneficial when K is small (epilogue constitutes a larger fraction of runtime).

### 5.1 The STEALB Technique

Challenge: Overlapping store with next-tile loads makes the C store buffer lifetime overlap with operand buffers, potentially exceeding 228KB SMEM.

Solution: Allocate one extra B buffer (5 total instead of 4). The inner loop uses A buffers with modulo-4 indexing and B buffers with modulo-5 indexing. During epilogue, the B buffer at `producer % 5` is guaranteed to be free (its MMA consumption completed long ago) and can be reinterpreted as the C store buffer.

Buffer allocation:
- A_bufs[4]: 4 × 16KB = 64KB
- B_bufs[5]: 5 × 32KB = 160KB
- Total: 224KB < 228KB limit

```python
@gluon.jit
def issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs,
                       stealb: gl.constexpr, num_buffers: gl.constexpr, pred=True):
    index = producer % num_buffers          # A uses modulo num_buffers
    b_index = producer % (num_buffers + stealb)  # B uses modulo (num_buffers + 1)
    producer += 1
    bar = bars.index(index)
    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes, pred=pred)
    tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_bufs.index(index), pred)
    tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_bufs.index(b_index), pred)
    return producer
```

### 5.2 Pipelined Persistent Kernel Structure

The kernel fuses the current tile's drain phase with the next tile's prologue:

1. **Initial Prologue**: Prefetch first `num_buffers - 1` loads for tile 0
2. **Main loop per tile**:
   - First MMA (consuming oldest buffered data)
   - `tma.store_wait(pendings=0)` — wait for previous tile's store
   - Steady state: alternate Load and MMA
   - Fused drain + next prologue: issue next tile's loads while consuming current tile's remaining MMA
3. **Epilogue**: Wait MMA → take result → cast to fp16 → borrow B buffer → store → TMA to global

### 5.3 Final Performance Comparison

Matrix: 8192×8192, varying K. Best configs (Blackwell: 4 buffers, 4 warps, grouped scheduler; Hopper: 3 buffers, 8 warps):

| K | Non-persistent | Persistent | Pipelined | cuBLAS |
|---:|---------------:|-----------:|----------:|-------:|
| 512 | 615.86 | 828.70 | 993.50 | 1108.11 |
| 1024 | 997.16 | 1077.28 | 1173.31 | 1347.44 |
| 2048 | 1152.74 | 1190.55 | 1133.37 | 1435.01 |
| 4096 | 1164.05 | 1120.92 | 1143.47 | 1563.98 |
| 8192 | 1160.93 | 1074.97 | 1185.40 | 1491.84 |
| 16384 | 1185.62 | 1096.34 | 1296.93 | 1548.42 |

(Blackwell results in TFLOPS)

As expected, the pipelined persistent kernel shows the largest improvement at small K where epilogue overhead is proportionally larger.

On Hopper, the pipelined kernel matches cuBLAS at medium-to-large K:

| K | Non-persistent | Persistent | Pipelined | cuBLAS |
|---:|---------------:|-----------:|----------:|-------:|
| 512 | 491.74 | 485.01 | 539.88 | 588.15 |
| 1024 | 554.24 | 575.02 | 602.52 | 588.32 |
| 2048 | 573.87 | 594.72 | 625.91 | 615.58 |
| 4096 | 609.36 | 630.10 | 640.48 | 646.30 |
| 8192 | 629.44 | 646.22 | 661.57 | 661.11 |
| 16384 | 653.79 | 660.29 | 670.00 | 665.49 |

(Hopper results in TFLOPS)

## 6. Key Observations

Performance gaps relative to cuBLAS stem from:

1. **2-CTA matmul**: cuBLAS uses distributed shared memory across 2 CTAs for 256×256 instruction shapes, feeding MMA more efficiently. Especially important on Blackwell where MMA-to-TMA throughput ratio is higher.
2. **Warp specialization**: cuBLAS uses dedicated warps for load/compute/store, essential for fully hiding epilogue at small K.
3. **Blackwell-specific optimizations**: Our unified API does not double-buffer the accumulator or fully utilize TMEM's 256 columns.
4. **Dynamic scheduling**: Blackwell supports Cluster Launch Control for GPU-cooperative dynamic scheduling, combining static optimization potential with dynamic load balancing.

## 7. Summary

- Persistent kernels replace GPU-native block scheduling with static scheduling, enabling better resource coordination and computation overlap across tiles at the cost of dynamic load balancing
- Grouped tile scheduling improves L2 cache locality by co-locating tiles that share B matrix data
- Outer-loop pipelining overlaps the current tile's epilogue with the next tile's prologue, most effective at small K
- The STEALB technique borrows free B buffers for C storage, avoiding SMEM overflow while maintaining 4-buffer pipeline depth
- Persistent kernels are especially effective for small problem sizes but provide benefits at all scales


## Related

- [Software Pipeline Depth Optimization](../software-pipeline-depth-optimization.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
- [Document Relationship Diagram](../../../RELATIONS.md)
