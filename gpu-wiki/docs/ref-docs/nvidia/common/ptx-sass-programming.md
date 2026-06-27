# PTX/SASS Low-Level Programming Practices

> This article synthesizes multiple technical posts from the Zhihu community, covering PTX inline assembly, SASS instruction scheduling optimization, DeepSeek engineering practices, and more.

---

## 1. PTX Programming Fundamentals

### 1.1 Compilation Pipeline: CUDA C++ → PTX → SASS

The NVIDIA GPU compilation pipeline consists of two stages:

1. **Front-end Compilation**: CUDA C++ is compiled by `nvcc` into PTX (Parallel Thread Execution) intermediate representation
2. **Back-end Compilation**: PTX is assembled by `ptxas` into SASS (Shader Assembly) native machine instructions

PTX is a virtual ISA with cross-architecture compatibility (forward-compatible); SASS is the binary instruction directly executed by GPU hardware, with different encodings for each generation of architecture.

```
CUDA C++ (.cu)
 │ nvcc
    ▼
PTX (.ptx) ← , ,
 │ ptxas
    ▼
SASS (.cubin) ← ,
```

### 1.2 PTX Syntax Elements

**State Spaces**:

| State Space | Description | Latency (cycles) |
|-------------|-------------|------------------|
| `.reg` | Register, thread-private | 1 |
| `.shared` | Shared memory, visible within a block | ~5 |
| `.global` | Global memory (HBM) | ~500 |
| `.local` | Local memory (spills to HBM) | ~500 |
| `.const` | Constant memory (cached) | 1–5 |
| `.param` | Parameter space | - |

**Register Declaration and Usage**:

```
.reg .f32 %f<32>;        // 32 32-bit floating-point registers
.reg .b64 %rd<16>;       // 16 64-bit general-purpose registers
.reg .pred %p<8>;        // 8 predicate registers

mov.f32 %f0, 0f3F800000; // %f0 = 1.0f
add.f32 %f2, %f0, %f1;   // %f2 = %f0 + %f1
```

**Inline PTX Assembly** (embedding PTX in CUDA C++):

```cpp
// Basic format
asm("instruction" : output_operands : input_operands);

// Example: FMA fused multiply-add
float result;
asm("fma.rn.f32 %0, %1, %2, %3;"
    : "=f"(result)        // Output: floating-point register
    : "f"(a), "f"(b), "f"(c));  // Input

// Constraint specifiers:
// "f" = .f32 floating-point register
// "d" = .f64 double-precision register
// "r" = .u32/.s32 integer register
// "l" = .u64/.s64 long integer register
// "=" = output operand
// "+" = read-write operand
```

### 1.3 Key PTX Instruction Categories

**Integer Arithmetic Instructions**:

```
add.s32 %r2, %r0, %r1;       // 32-bit signed addition
mul.lo.s32 %r2, %r0, %r1;    // Low 32-bit multiplication
mad.lo.s32 %r3, %r0, %r1, %r2; // Multiply-add
shl.b32 %r1, %r0, 2;         // Left shift (multiply by 4)
bfe.u32 %r1, %r0, 8, 4;      // Bit field extract
```

**Floating-Point and MMA Instructions**:

```
// Tensor Core MMA（Hopper wgmma）
wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16
    {%f0, ..., %f127},        // Accumulators
    desc_a,                    // A matrix descriptor
    desc_b,                    // B matrix descriptor
    1, 1, 1, 0, 0;            // scale/satfinite parameters
```

**Synchronization and Barrier Instructions**:

```
bar.sync 0;                   // Block synchronization
bar.arrive 0, 128;            // Barrier arrival
membar.gl;                    // Global memory barrier
fence.proxy.async;            // Asynchronous proxy barrier

// mbarrier（Hopper）
mbarrier.init.shared.b64 [%smem_addr], %thread_count;
mbarrier.arrive.shared.b64 _, [%smem_addr];
```

**Asynchronous Copy Instructions**:

```
// Ampere cp.async
cp.async.ca.shared.global [dst], [src], 16;  // 16B copy
cp.async.commit_group;
cp.async.wait_group 0;

// Hopper TMA
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    [dst], [tensor_map, {x, y}], [mbar];
```

---

## 2. Hand-Written PTX Optimization Techniques

### 2.1 FMA Fusion and Instruction Selection

The compiler sometimes cannot automatically fuse `mul + add` into `fma`. Hand-written PTX can enforce the fusion:

```cpp
// Compiler may generate separate mul and add
float result = a * b + c;

// Hand-written PTX enforces FMA
float result;
asm("fma.rn.f32 %0, %1, %2, %3;"
    : "=f"(result) : "f"(a), "f"(b), "f"(c));
```

Advantages of FMA:
- Completes multiply-add in a single instruction, doubling throughput
- Intermediate results retain full precision (reducing rounding errors)
- Suitable for polynomial evaluation (Horner's method)

### 2.2 Vectorized Memory Access

Using 128-bit vector loads to reduce the number of instructions and transactions:

```cpp
// Scalar load: 4 instructions, 4 memory transactions
float a0 = ptr[0]; float a1 = ptr[1];
float a2 = ptr[2]; float a3 = ptr[3];

// Vectorized load: 1 instruction, 1 128-bit transaction
float4 vec;
asm("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];"
    : "=f"(vec.x), "=f"(vec.y), "=f"(vec.z), "=f"(vec.w)
    : "l"(ptr));
```Notes on vectorization:
- Address must be aligned to `sizeof(vector type)` (16B alignment for float4)
- On H200/B200, vectorization yields approximately 21-25% bandwidth utilization improvement
- Tail elements must be handled (when N is not divisible by the vector width)

### 2.3 Warp-Level Shuffle Operations

The `shfl` instruction enables direct data exchange between threads within a warp, bypassing shared memory:

```cpp
// Warp-wide broadcast: lane 0 value broadcast to all lanes
asm("shfl.sync.idx.b32 %0, %1, 0, 31, 0xffffffff;"
    : "=r"(val) : "r"(val));

// Warp-wide reduction sum
for (int offset = 16; offset > 0; offset >>= 1) {
    asm("shfl.sync.down.b32 %0, %1, %2, 31, 0xffffffff;"
        : "=r"(tmp) : "r"(val), "r"(offset));
    val += tmp;
}
```

### 2.4 SFU Special Function Unit

SFU provides fast approximations of transcendental functions (e.g., `sin`, `cos`, `exp2`, `rcp`, `rsqrt`), with approximately 22-23 bits of precisioney and throughput of 8 operations per SM per cycle:

```
// SFU fast reciprocal
rcp.approx.f32 %f1, %f0;     // 1/x approximation

// SFU fast reciprocal square root
rsqrt.approx.f32 %f1, %f0;   // 1/sqrt(x) approximation

// Compiler option --use_fast_math automatically uses SFU version
```

### 2.5 Eliminating Shared Memory Bank Conflicts

PTX enables precise control over shared memory address computation, eliminating bank conflicts:

```cpp
// 32 banks × 4B = 128B per row
// Add padding to eliminate conflict
__shared__ float smem[32][33];  // 33 not 32, avoid column access conflict

// Or use PTX swizzle mode
// Hopper architecture supports 128B swizzle
```

### 2.6 Data Prefetching

Explicit prefetching loads data from global memory into the L1/L2 cache to hide memory latency:

```
// L1 prefetch
prefetch.global.L1 [%addr];
// L2 prefetch
prefetch.global.L2 [%addr];

// Use inline PTX prefetch in CUDA C++
asm("prefetch.global.L2 [%0];" :: "l"(next_ptr));
```

A typical software pipelining pattern:

```
// Phase 1: Prefetch next round data
prefetch(data[i+1])
// Phase 2: Compute current round data
compute(data[i])
// Phase 3: Store previous round result
store(result[i-1])
```

---

## 3. DeepSeek's PTX Engineering Practices

DeepSeek extensively uses hand-written PTX in three core components — FlashMLA, DeepEP, and DeepGEMM — demonstrating best practices for production-grade PTX optimization.

### 3.1 Asynchronous Pipeline Control in FlashMLA

FlashMLA uses `cp.async.wait_group` to control the depth of the asynchronous copy pipeline:

```cpp
// Wait until no more than N incomplete copy groups
asm volatile("cp.async.wait_group %0;" :: "n"(N));

// Typical double buffering pattern:
// Step 1: Initiate copy to buffer[1]
cp.async(buffer[1], global_ptr + offset);
cp.async.commit_group;

// Step 2: Wait for buffer[0] copy completion
cp.async.wait_group 1;  // Wait until only 1 incomplete group

// Step 3: Compute from buffer[0]
compute(buffer[0]);

// Step 4: Swap buffer indices, loop
```

This fine-grained pipeline control completely hides memory latency behind computation.

### 3.2 Endianness Handling in DeepEP

DeepEP needs to handle network byte order conversion in All-to-All communication, using PTX bit-manipulation instructionspects to achieve efficient 64-bit byte reversal:

```cpp
__device__ __forceinline__ int64_t HtoBE64(int64_t val) {
    int32_t lo, hi;
    asm("mov.b64 {%0, %1}, %2;" : "=r"(lo), "=r"(hi) : "l"(val));

 // use prmt rowbytescolumn(byte permutation)
    int32_t rlo, rhi;
    asm("prmt.b32 %0, %1, %2, 0x0123;"
        : "=r"(rhi) : "r"(lo), "r"(0));
    asm("prmt.b32 %0, %1, %2, 0x0123;"
        : "=r"(rlo) : "r"(hi), "r"(0));

    int64_t result;
    asm("mov.b64 %0, {%1, %2};" : "=l"(result) : "r"(rlo), "r"(rhi));
    return result;
}
```

The `prmt.b32` instruction is a powerful byte permutation instruction that can rearrange any 4 bytes within a single cycle.

### 3.3 WGMMA Asynchronous Matrix Multiplication in DeepGEMM

DeepGEMM uses Hopper's `wgmma.mma_async` instruction to achieve high-throughput matrix multiplication:

```cpp
// 128 threads collaboratively execute 64×256×16 matrix multiplication
asm volatile(
    "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
    "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
    " %8,  %9,  %10, %11, %12, %13, %14, %15, "
 " ...(128register)}, "
    "%33, %34, "          // A、B matrix descriptors
    "1, 1, 1, 0, 0;"     // scale_d, scale_a, scale_b, ...
    : "+f"(d0), "+f"(d1), ..., "+f"(d127)
    : "l"(desc_a), "l"(desc_b));
```Key Design Features:
- 128 threads form a warpgroup that collaboratively executes a WGMMA instruction
- Input matrices are read from shared memory via descriptors, eliminating the need for explicit loads
- The accumulator is retained in registers, supporting multiple WGMMA accumulations
- Used in conjunction with TMA asynchronous copy to achieve compute-copy overlap

---

## 4. SASS Instruction Scheduling Optimization

### 4.1 SASS Control Bits and Dependency Counters

Modern NVIDIA GPU SASS instructions embed control information to guide the hardware scheduler:

**Control Word Format** (every 3 instructions share a 128-bit control word):

| Field | Bits | Description |
|------|------|------|
| Stall Count | 4 bits | Number of cycles to wait before instruction issue (0–15) |
| Yield Flag | 1 bit | Whether to yield warp scheduling priority |
| Write Barrier | 3 bits | Write barrier index (0–5), marking a long-latency write operation |
| Read Barrier | 3 bits | Read barrier index, waiting for the corresponding write to complete |
| Wait Mask | 6 bits | Wait for specified barriers to complete |

**How Dependency Counters Work**:
- The GPU maintains 6 dependency counters (barriers 0–5)
- Long-latency instructions (e.g., global memory loads) set a write barrier, incrementing the counter by 1
- Instructions that depend on that result set a read barrier or wait mask, waiting for the counter to reach zero
- Short-latency instructions (e.g., register operations) use Stall Count for direct delay

Example:
```
// SASS instruction format example
[B------:R-:W-:-:S04]  IMAD.MOV.U32 R4, RZ, RZ, c[0x0][0x168]
// B------: Wait for no barriers
// R-:     No read barrier
// W-:     No write barrier
// S04:    stall 4 cycles

[B------:R-:W0:-:S01]  LDG.E R2, [R4]
// W0:     Set barrier 0 (mark load start)
// S01:    stall 1 cycle

[B0-----:R-:W-:-:S02]  IADD3 R5, R2, R3, RZ
// B0:     Wait for barrier 0 completion (wait for load result)
```

### 4.2 CuAsmRL: Reinforcement Learning-Based SASS Instruction Scheduling

CuAsmRL (CGO'25 paper) proposes using reinforcement learning to automatically optimize SASS instruction scheduling:

**Core Idea**:
- The ptxas compiler's instruction scheduling is not optimal and leaves room for improvement
- Model instruction scheduling as a reinforcement learning problem: state = current instruction sequence, action = select the next instruction, reward = kernel execution time
- Use the CuAsm tool to disassemble the cubin, modify instruction order and control bits, and reassemble

**Experimental Results**:
- On CUTLASS GEMM kernels, an average speedup of 9% and a maximum speedup of 26%
- Significant speedups were also achieved on Flash Attention kernels
- Demonstrates that compiler-generated instruction scheduling has room for optimization

**Optimization Techniques**:
1. **Instruction Reordering**: Adjust the execution order of independent instructions TResult to better hide latency
2. **Stall Count Adjustment**: Reduce unnecessary wait cycles
3. **Barrier Allocation**: Utilize the 6 dependency counters more efficiently
4. **Yield Strategy**: Optimize warp switching points

### 4.3 ptxas Compiler Behavior and "Black Magic"

**The `cutlass_` Prefix Triggers Different Optimizations**:

A little-known fact is that when a kernel function name begins with `cutlass_`, the ptxas compiler triggers a different optimization path. Testing on the SM100 (Blackwell) architecture:

```bash
# Use cutlass_ prefix
$ nvcc -arch=sm_100 -o test_cutlass test.cu  # kernel name: cutlass_gemm
# cubin size: 4096 bytes

# Don't use prefix
$ nvcc -arch=sm_100 -o test_normal test.cu   # kernel name: gemm
# cubin size: 3584 bytes
```

The generated SASS instruction sequences differ, indicating that ptxas internally applies special optimization strategies Inhaltsverzeichnis for CUTLASS kernels. This behavior is not documented in official documentation.

**Influencing Factors**:
- Behavior may differ across compiler versions
- Obvious differences have only been observed on specific architectures (e.g., SM100)
- This characteristic can be leveraged in production, but requires thorough performance validation

---

## 5. PTX Generation in the Triton Compiler

### 5.1 Pipeline Pass PTX Instruction Selection

The Triton compiler's Pipeline Pass is responsible for converting high-level `tl.load` / `tl.store` into asynchronous memory instructions. Different PTX instructions are selected based on the architecture:

**Ampere (SM80) Path**:
```
tl.load → cp.async.ca.shared.global → cp.async.commit_group → cp.async.wait_group
```
- Uses `cp.async` to implement asynchronous copies from global memory to shared memory
- Software pipeline depth is controlled via `wait_group(N)`

**Hopper (SM90) Path**:
```
tl.load → TMA bulk copy → mbarrier.arrive → mbarrier.try_wait
```
- Uses the TMA (Tensor Memory Accelerator) hardware unit
- A single thread initiates large-block transfers; other threads do not participate
- Cooperates with `wgmma.mma_async` to achieve full compute-copy pipelining

### 5.2 Choosing the Number of Pipeline Stages

The number of pipeline stages directly affects performance and shared memory usage:

| Stages | Shared Memory Usage | Latency Hiding Capability | Applicable Scenarios |
|--------|---------------------|---------------------------|----------------------|
| 2 (double buffering) | 2× tile size | Basic | Small tiles, tight shared memory |
| 3 (triple buffering) | 3× tile size | Good | General GEMM |
| 4+ | 4+× tile size | Ample | Large matrices, abundant memory |

---

## 6. Debugging and Analysis Tools

### 6.1 Viewing PTX

```bash
# Compile to generate PTX
nvcc -ptx -arch=sm_90 kernel.cu -o kernel.ptx

# Disassemble SASS from cubin
cuobjdump -sass kernel.cubin

# Use CuAsm to disassemble and reassemble
python -m cuasm disassemble kernel.cubin
python -m cuasm assemble kernel.cuasm
```### 6.2 Source-SASS Correlation in Nsight Compute

Using the `-lineinfo` compile option allows you to correlate CUDA source code with SASS instructions in Nsight Compute:

```bash
nvcc -O3 -lineinfo -arch=sm_90 kernel.cu -o kernel
ncu --set detailed ./kernel
```

In the Source page of Nsight Compute, you can inspect line by line:
- The stall reason and stall cycle count for each SASS instruction
- Instruction throughput compared to theoretical peak
- Memory access coalescing efficiency

---

## Related Documents

- [PTX Programming Model](ptx-programming-model.md) — Programming model and execution semantics of the PTX virtual ISA
- [PTX Instruction Set Reference](ptx-instruction-set.md) — Complete classification and syntax of PTX instructions
- [PTX MMA Instructions](nvidia-ptx-mma-instructions.md) — Detailed explanation of Tensor Core MMA instructions
- [PTX Synchronization and Asynchronous Primitives](nvidia-ptx-sync-and-async.md) — Barriers mad asynchronous copy instructions
- [Asynchronous Global-to-Shared Memory Copy](../../../kernel-opt/nvidia/common/async-global-to-shared-copy.md) — cp.async and TMA practices
- [NCU Performance Analysis Guide](ncu-profiling-guide.md) — Source-SASS analysis in Nsight Compute
- [Hopper SM90 Optimization Hands-On](../../../kernel-opt/nvidia/common/sm90/hands-on) — WGMMA and TMA pipeline hands-on
