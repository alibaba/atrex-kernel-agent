# Minimal test: 256x128 register -> 128x256 TMEM
# Print partition_D and partition_S shapes for TMEM store

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl.cutlass import const_expr
import cutlass.pipeline as pipeline
import cutlass.utils as utils

@cute.struct
class SharedStorage:
    tmem_holding_buf: cutlass.Int32

@cute.kernel
def test_tmem_store_shapes():
    tidx, _, _ = cute.arch.thread_idx()

    # TMEM tensor: 128 rows x 256 columns
    tmem_rows = 128
    tmem_cols = 256
    tmem_shape = ((tmem_rows, tmem_cols), 1, 1)

    # Create identity tensor for TMEM
    tmem_tensor = cute.make_identity_tensor(tmem_shape)

    # Register tensor: 256 rows x 128 columns
    reg_rows = 128
    reg_cols = 256
    reg_shape = ((reg_rows, reg_cols), 1, 1)
    reg_tensor = cute.make_identity_tensor(reg_shape)

    # Create tiled copy with St32x32bOp
    store_atom = cute.make_copy_atom(
        tcgen05.St32x32bOp(tcgen05.Repetition.x2),
        cutlass.Float32,
    )

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=128,
    )
    tmem = cutlass.utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
    )
    num_tmem_cols = 256
    tmem.allocate(num_tmem_cols)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(
        cutlass.Float32,
    )

    # Swap the pointer in tCtAcc
    tmem_tensor = cute.make_tensor(tmem_ptr, tmem_tensor.layout)

    # Partition for store: S = TMEM (dest), D = register (src)
    tiled_copy = tcgen05.make_tmem_copy(store_atom, tmem_tensor)
    thr_copy = tiled_copy.get_slice(tidx)

    tmem_partition = thr_copy.partition_S(tmem_tensor)  # TMEM dest
    reg_partition = thr_copy.partition_D(reg_tensor)  # Register src

    # Print shapes
    if tidx == 0:
        cute.printf("TMEM tensor shape: (%d, %d)\n", tmem_rows, tmem_cols)
        cute.printf("Register tensor shape: (%d, %d)\n", reg_rows, reg_cols)
        cute.printf("\n")
        cute.printf(
            "tmem_partition (partition_S) inner shape: ((%d, %d), ...)\n",
            cute.size(tmem_partition, mode=[0, 0]),
            cute.size(tmem_partition, mode=[0, 1]),
        )
        cute.printf(
            "reg_partition (partition_D) inner shape: ((%d, %d), ...)\n",
            cute.size(reg_partition, mode=[0, 0]),
            cute.size(reg_partition, mode=[0, 1]),
        )
        cute.printf(
            "inner_m_S = %d, inner_n_S = %d\n",
            cute.size(tmem_partition, mode=[0, 0]),
            cute.size(tmem_partition, mode=[0, 1]),
        )
        cute.printf(
            "inner_m_D = %d, inner_n_D = %d\n",
            cute.size(reg_partition, mode=[0, 0]),
            cute.size(reg_partition, mode=[0, 1]),
        )


@cute.jit
def host_function():
    test_tmem_store_shapes().launch(grid=(1, 1, 1), block=(128, 1, 1))


if __name__ == "__main__":
    from cuda.bindings import driver as cu_driver

    cu_driver.cuInit(0)
    err, device_count = cu_driver.cuDeviceGetCount()
    if err != cu_driver.CUresult.CUDA_SUCCESS or device_count < 1:
        raise RuntimeError("A GPU is required")

    print("Testing TMEM store shapes:")
    print("  Register tensor: 256 rows x 128 columns")
    print("  TMEM tensor: 128 rows x 256 columns")
    print("  Store op: St32x32bOp x2")
    print()

    compiled = cute.compile(host_function)
    compiled()
