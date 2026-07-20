# Register Pressure and Warp Occupancy

Registers are the invisible killer of GPU kernel performance. This document explains from the hardware architecture level how the register file affects occupancy, and how to design register budgets in warp-specialized kernels.

> Related Documents: [Occupancy Tuning](../../../kernel-opt/nvidia/common/occupancy-tuning-by-arch.md) | [Warp Specialization Design Principles](warp-specialization-design-principles.md) | [PTX Synchronization and Asynchronous Operations](nvidia-ptx-sync-and-async.md)

---

## 1. Register File Architecture

### Hardware Specifications

Each SM has a fixed-size register file shared by all warps resident on that SM:

| Architecture | Registers per SM | Register Width | Total Capacity |
|--------------|------------------|---------------|----------------|
| Volta (SM70) | 65536 | 32-bit | 256 KB |
| Turing (SM75) | 65536 | 32-bit | 256 KB |
| Ampere (SM80) | 65536 | 32-bit | 256 KB |
| Hopper (SM90) | 65536 | 32-bit | 256 KB |
| Blackwell (SM100) | 65536 | 32-bit | 256 KB |

The number of registers remains unchanged across generations (65536), but each generation's architecture continuously improves register usage efficiency.

### Register Allocation Fundamentals

Key formula for register allocation:

```
Max warps per SM = min(
    Registers per SM / (Registers per thread * 32),     // Register limit
    Max threads per SM / 32,                             // Architecture limit
    SMEM per SM / SMEM per block * block_warps          // SMEM limit
)
```

**Concrete Calculation Example** (Hopper SM90):

| Registers per Thread | Registers per Warp | Max Warps | Occupancy |
|----------------------|-------------------|-----------|-----------|
| 32 | 1024 | 64 (hardware limit) | 100% |
| 64 | 2048 | 32 | 50% |
| 128 | 4096 | 16 | 25% |
| 192 | 6144 | 10 | ~16% |
| 232 | 7424 | 8 | 12.5% |
| 256 | 8192 | 8 | 12.5% |

Note: Register allocation is done at the warp granularity requirements and rounded up to the hardware allocation unit (usually a multiple of 8).

---

## 2. Register Pressure in GEMM Kernels

### The Dominance of Accumulators

In GEMM kernels, the largest consumer of registers is the **MMA accumulator**:

```
Number of accumulator registers = TileM * TileN / threads_per_warp_group * sizeof(AccumType) / sizeof(uint32_t)
```

| Tile Shape (M x N) | Accumulator Type | Threads per Warp Group | Accumulator Registers per Thread |
|---------------------|------------------|------------------------|----------------------------------|
| 64 x 64 | FP32 | 128 | 32 |
| 128 x 64 | FP32 | 128 | 64 |
| 128 x 128 | FP32 | 128 | 128 |
| 128 x 256 | FP32 | 128 | 256 |
| 256 x 128 | FP32 | 128 | 256 |
| 256 x 256 | FP32 | 256 (cooperative) | 256 |

Calculation method in CUTLASS:

```cpp
// From CUTLASS sm90_gemm_tma_warpspecialized_pingpong.hpp
static constexpr int RegsPerThread =
    (size<0>(TileShape{}) * size<1>(TileShape{}) * sizeof(ElementAccumulator))
    / (NumMMAThreads * sizeof(uint32_t));
```

### Other Register Consumers

In addition to accumulators, each thread requires extra registers:

| Purpose | Registers (Estimated) |
|---------|----------------------|
| Loop counters / indices | 4-8 |
| Address pointers (64-bit) | 4-8 |
| Pipeline state | 2-4 |
| GMMA descriptors | 4-8 |
| Temporary variables | 8-16 |
| Compiler spill reserve | 8-16 |
| **Subtotal** | ~40-60 |

Therefore, the total register demand for a GEMM consumer warp ≈ accumulators + 40-60.

### Why High Registers with Low Occupancy Makes Sense for GEMM

GEMM is a **compute-bound** kernel:
Arithmetic intensity = 2*M*N*K / (M*K + K*N + M*N) ≈ O(M*N*K) / O(M*K)

For M=N=K=4096: Arithmetic intensity ≈ 2730 FLOP/byte
H100 Tensor Core: 989 TFLOPS
H100 HBM bandwidth: 3.35 TB/s
Inflexion point = 989e12 / 3.35e12 ≈ 295 FLOP/byte

GEMM far exceeds inflexion point → compute-bound → No high occupancy needed to hide latency
Occupancy guidelines for compute-bound kernels:
- 8 warps/SM (12.5% occupancy) is usually sufficient for GEMM
- Further increasing occupancy may actually degrade performance (smaller tile → less data reuse)

---

## 3. `setmaxnreg` Instruction: Dynamic Register Reallocation

### Basic Mechanism

Starting from Hopper (SM90), PTX provides the `setmaxnreg` instruction, which allows runtime reallocation of registers between warp groups within the same CTA:

```
PTX syntax:
  setmaxnreg.inc.sync.aligned.u32 Rn;  // Increase current warp group register limit
  setmaxnreg.dec.sync.aligned.u32 Rn;  // Decrease current warp group register limit
```Corresponding CUDA C++ wrapper:

```cpp
// Increase registers (consumer uses)
template<uint32_t RegCount>
__device__ void warpgroup_reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

// Decrease registers (producer uses)
template<uint32_t RegCount>
__device__ void warpgroup_reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}
```

### Constraints

1. **Total CTA register count remains unchanged**: Reducing registers on one side increases them on the other. The two must match.
2. **Synchronization semantics**: `sync.aligned` requires all threads within a warp group to execute simultaneously.
3. **Granularity**: RegCount must be a multiple of 8.
4. **Timing**: Must be executed at the kernel entry point, before any branches. Dealloc first, then alloc.
5. **Warp group granularity**: SM90 reallocates at the warp group (128 threads) level.

### Usage in Warp-Specialized Kernels

A typical warp-specialized kernel entry:

```cpp
__global__ void gemm_kernel(...) {
    int warp_group_idx = threadIdx.x / 128;

    if (warp_group_idx == 0) {
        // Producer: Release unneeded registers
        warpgroup_reg_dealloc<40>();   // Reduce to 40 registers

        // ... TMA load code (only needs few registers)
    }
    else {
        // Consumer: Acquire more registers
        warpgroup_reg_alloc<232>();    // Increase to 232 registers

        // ... WGMMA code (needs many accumulator registers)
    }
}
```

**Execution order is critical**: The producer must first execute `dealloc` to release registers into the "pool," then the consumer can `alloc` to acquire those registers. CUTLASS ensures this by making the `WarpGroupRole::Producer` code path appear before the `Consumer` code path.

---

## 4. Register Spilling

### What is Spilling

When the compiler allocates insufficient registers, variables are stored to **local memory** (actually per-thread stack space located in L1 cache / DRAM):

```
Normal register access:  ~1 cycle
Local memory access: ~30-100 cycles (L1 hit) to ~200-800 cycles (L2/DRAM)
```

### Detecting Spills

Use `--ptxas-options=-v` at compile time to view register and spill information:

```bash
$ nvcc --ptxas-options=-v -o kernel kernel.cu
ptxas info : Used 128 registers, 0 bytes smem, 0 bytes cmem, 24 bytes spill stores, 24 bytes spill loads
#                                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                                             spill > 0 indicates register spilling
```

Key metrics for detecting spills in NCU:

| Metric | Meaning |
|------|------|
| `l1tex__data_pipe_lsu_wavefronts_mem_lg_op_ld.sum` | Contains local memory loads |
| `l1tex__data_pipe_lsu_wavefronts_mem_lg_op_st.sum` | Contains local memory stores |
| `smsp__inst_executed_pipe_lsu.sum` | LSU pipeline execution count (high = potentially more spills) |
| Source Panel "Stack frame" | Per-thread stack frame size |

### Performance Impact of Spilling

```
Each spill = 1 store (write to local memory) + 1 load (read back from local memory)
Worst case: Each spill ≈ 200 cycles (L1 miss → L2/DRAM)
Typical case: Each spill ≈ 30-50 cycles (L1 hit)

If kernel has N spills/restores, additional latency ≈ N * 30-50 cycles
```

### Strategies Giacomo to Reduce Spilling

| Strategy | Method | Impact |
|------|------|------|
| Reduce tile size | 128x128 → 64x64 | Accumulator reduced by 4x |
| Use `__launch_bounds__` | Limit compiler's register allocation | May introduce spills |
| Recomputation instead of storage | Recompute values when needed instead of caching | Increases computation |
| TMEM (SM100+) | Store accumulators in TMEM instead of registers | Hardware supported |
| Reduce live variables | Restructure loops to reduce simultaneously live variables | Significant code changes |

---

## 5. `__launch_bounds__` and Occupancy Control

### Basic Syntax

```cpp
__global__ void __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
kernel(...) {
    // ...
}
```

| Parameter | Meaning | Impact |
|------|------|------|
| `maxThreadsPerBlock` | Maximum threads per block | Compiler optimizes accordingly |
| `minBlocksPerMultiprocessor` | Minimum blocks per SM | Compiler limits registers to meet this target |

### Impact of Register Allocation

The compiler determines the maximum allowed number of registers based on `minBlocksPerMultiprocessor`:

```
max_regs_per_thread = 65536 / (maxThreadsPerBlock * minBlocksPerMultiprocessor)

Example 1: __launch_bounds__(256, 1) → max_regs = 65536 / 256 = 256
Example 2: __launch_bounds__(256, 2) → max_regs = 65536 / 512 = 128
Example 3: __launch_bounds__(384, 1) → max_regs = 65536 / 384 = 170
```

### Practice in Production Code

CUTLASS's approach is to **not specify `minBlocksPerMultiprocessor`**, but instead precisely control the registers for each warp group through `setmaxnreg`:

```cpp
// Set MaxThreadsPerBlock but not MinBlocks
static constexpr uint32_t MaxThreadsPerBlock = 384;
static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
// → Compiler allows up to 65536/384 ≈ 170 registers per thread

// Then fine-tune through setmaxnreg:
// Producer: dealloc to 40 regs (actually only uses addresses + loop variables)
// Consumer: alloc to 232 regs (accumulators + auxiliary variables)
```

### Trade-off Analysis

```
More blocks/SM (high occupancy):
  + Better latency hiding (more warps to switch)
  + Higher memory-bound kernel performance
  - Fewer registers per thread → potential spilling
  - Smaller tile → less data reuse

Fewer blocks/SM (low occupancy):
  + More registers → no spilling
  + Larger tile → better data reuse
  + More shared memory available
  - Less latency hiding capability
```

---

## 6. Register Budget Design for Warp-Specialized Kernels

### Design Steps

```Step 1: Determine CTA total threads and occupancy target
  Example: 384 threads/CTA, 1 CTA/SM → total budget = 65536 regs

Step 2: List register requirements for each role (profile or estimate)
  Producer (TMA load):  Address pointers + loop variables ≈ 24-40 regs
  Consumer (MMA):       Accumulators + auxiliary variables ≈ 200-256 regs

Step 3: Calculate if allocation is feasible
  Ensure Total * CTA_Threads ≤ 65536

Step 4: Implement allocation using setmaxnreg```

### Worked Example 1: Hopper Pingpong (384 Threads)

```
CTA structure:
  Producer WG: 128 threads (1 warp group)
  Consumer0:   128 threads (1 warp group)
  Consumer1:   128 threads (1 warp group)
  Total:       384 threads

Register budget:
  Total available registers (1 CTA/SM): 65536
  Average per thread: 65536 / 384 = 170.6 → compiler initial allocation ~170/thread

setmaxnreg reallocation:
  Producer: dealloc to 40 regs → release (170-40) * 128 = 16640 regs
  Consumer0: alloc to 232 regs → acquire (232-170) * 128 = 7936 regs
  Consumer1: alloc to 232 regs → acquire (232-170) * 128 = 7936 regs
  Verify: 16640 ≈ 7936 + 7936 = 15872  ✓ (small margin)

Actual consumption:
  Producer: 40 * 128 = 5120 regs
  Consumer0: 232 * 128 = 29696 regs
  Consumer1: 232 * 128 = 29696 regs
  Total: 64512 / 65536 = 98.4% utilization
```

**Adaptive Strategy in CUTLASS**:

```cpp
// From sm90_gemm_tma_warpspecialized_pingpong.hpp
static constexpr bool HeavyRegisterPressure = RegsPerThread >= 208;
static constexpr uint32_t LoadRegisterRequirement = !HeavyRegisterPressure ? 40 : 24;
static constexpr uint32_t MmaRegisterRequirement = !HeavyRegisterPressure ? 232 : 240;
```

When the accumulator exceeds 208 regs/thread, further compress the producer's registers (40 → 24), giving the saved registers to the consumer (232 → 240).

### Worked Example 2: Blackwell 5-Warp (160 Threads)

```
CTA structure:
  MMA:           32 threads (1 warp)
  Scheduler:     32 threads (1 warp)
  Mainloop Load: 32 threads (1 warp)
  Epilogue Load: 32 threads (1 warp)
  Epilogue:      32 threads (1 warp)
  Total:         160 threads

Register budget (1 CTA/SM):
  Available: 65536
  Average: 65536 / 160 = 409.6 regs/thread — very abundant

Key difference: Blackwell's MMA accumulators stored in TMEM, not occupying registers!
  MMA warp:      No accumulator registers needed, only MMA descriptor + addresses
  Epilogue warp: TMEM → register read + epilogue computation

Estimated allocation:
  MMA:           ~96 regs (descriptor + loop + pipeline state)
  Scheduler:     ~32 regs (CLC address + counters)
  Mainloop Load: ~40 regs (TMA address + loop)
  Epilogue Load: ~40 regs (TMA address + loop)
  Epilogue:      ~128 regs (TMEM read + type conversion + store)

Total consumption: (96+32+40+40+128) * 32 = 10752 regs
Supported CTA count: 65536 / 10752 ≈ 6 CTA/SM

→ Blackwell can support more CTA concurrency, significantly higher than Hopper's 1-2 CTA/SM
```### Worked Example 3: Mamba2 SSD (384 threads)

```
From CUTLASS calculation:
  MaxThreadsPerBlock = (2 + 1) * 128 = 384

  LoadRegisterRequirement = 40 - 2*8 = 24  // Extreme compression
  TotalRegisterSupply = (65536 / 384 / 1 / 8) * 8 * 384 / 128 = 504
  MmaRegisterRequirement = ((504 - 24) / 2 / 8) * 8 = 240

  // Note: TotalRegisterSupply is total supply per warp group
  // Actually producer 24 regs, each consumer 240 regs
```

---

## 7. Strategies Condición reduce register pressure

### 7.1 Reduce Tile Size

The most straightforward approach, but sacrifices data reuse:

```
128x128 tile → 128 regs/thread (accumulators)
64x128 tile  → 64 regs/thread
64x64 tile   → 32 regs/thread

Trade-off:
  Compute unchanged, but SMEM load count increases
  Suitable for memory-bound scenarios or scenarios requiring high occupancy
```

### 7.2 TMEM (SM100+)

Blackwell introduces Tensor Memory (TMEM), dedicated to storing MMA accumulators:

```
Traditional method (SM90):
  Accumulators → Register file → 128-256 regs/thread

TMEM method (SM100):
  Accumulators → TMEM (dedicated hardware) → Register file almost unaffected
  TMEM capacity: 128 columns * 512 rows per SM = 65536 32-bit values

Performance:
  TMEM read/write latency ≈ Register file (design goal)
  But TMEM is independent storage space, not occupying register file
```

This is the key reason Blackwell can reduce MMA to a single warp — without register pressure constraints.

### 7.3 RS Mode vs SS Mode (SM90)

WGMMA instructions have two operand sources:

```
SS mode (smem-smem):
  A from SMEM descriptor, B from SMEM descriptor
  → Less register usage (no need to store A fragment)
  → But may have SMEM bank conflict

RS mode (reg-smem):
  A from registers (first SMEM→REG copy), B from SMEM descriptor
  → Additional A fragment registers (adds 16-32 regs)
  → Avoid SMEM read conflict
  → Must use RS for mixed-precision (A needs dequant)
```

Selection criteria:
- If registers are abundant and A needs preprocessing → RS mode
- If registers are tight → SS mode

### 7.4 Staged Accumulation

For very long K dimensions, accumulation can be staged to shared memory:

```cpp
// Staged accumulation: Write accumulators to SMEM every K_CHUNK tiles
for (int k_chunk = 0; k_chunk < K; k_chunk += K_CHUNK) {
    float accum[TILE_M * TILE_N / THREADS];  // Local accumulators in registers

    // Normal accumulation within K_CHUNK
    for (int k = 0; k < K_CHUNK; k += TILE_K) {
        mma(accum, smem_A[k], smem_B[k]);
    }

    // Accumulate to SMEM (release registers)
    atomicAdd(smem_partial, accum);
    __syncthreads();
}
```

Drawback: additional SMEM atomic operations and synchronization. Typically only used under extreme register pressure.

---

## 8. Rules of Thumb

1. **200-256 regs/thread is normal for GEMM kernels**. GEMM is compute-bound; 8 warps/SM (12.5% occupancy) is sufficient.

2. **Don't panic if you see spill > 0**. A small amount of spill (<100 bytes) costs very little when it hits the L1 cache. Only large spills (>500 bytes) need optimization.

3. **`setmaxnreg.dec` before `setmaxnreg.inc`**. The consumer can only acquire registers after the producer releases them. Wrong ordering causes deadlock.

4. **24-40 regs is enough for a producer warp**. TMA load only needs address pointers, loop counters, and pipeline state. Giving more registers to the producer is wasteful.

5. **Use `--ptxas-options=-v` to check actual register usage**. The compiler may use more registers than you expect. If it exceeds 256, consider reducing the tile size or using `__launch_bounds__`.

6. **Blackwell's TMEM is a game changer**. If you're on SM100+, accumulators should live in TMEM, and register pressure is no longer the primary bottleneck.

7. **Register allocation is in multiples of 8**. The `setmaxnreg` parameter must be a multiple of 8 (24, 32, 40, ..., 232, 240, 248, 256). Misalignment causes compilation errors.

8. **Higher occupancy is not always better**. For compute-bound kernels, high occupancy may mean smaller tiles, less data reuse, more SMEM load operations, which can actually degrade performance.

---

## Further Reading

- Warp Specialization architecture design: [Warp Specialization Design Principles](warp-specialization-design-principles.md)
- Occupancy differences across architectures: [Occupancy Tuning](../../../kernel-opt/nvidia/common/occupancy-tuning-by-arch.md)
- NCU performance analysis: [NCU Profiling Guide](ncu-profiling-guide.md)
- CUTLASS source code reference:
  - `arch/reg_reconfig.h` -- `setmaxnreg` wrapper
  - `sm90_gemm_tma_warpspecialized_pingpong.hpp` -- Hopper register allocation
  - `sm100_gemm_tma_warpspecialized.hpp` -- Blackwell 5-warp register allocation
