# GPU Instruction-Level Optimization


**Last updated**: 2026-06-30

## Arithmetic Instruction Optimization

### Use High-Throughput Math Functions

Standard math functions (``sin``, ``exp``, ``log``, etc.) map to multiple hardware instructions. GPUs typically provide low-precision but high-throughput fast math intrinsics, which offer slightly lower precision but are much faster.

Common functions that can be accelerated: ``sin``, ``cos``, ``exp``, ``log``, ``pow``, ``rsqrt``, etc.

Compilers typically offer options to globally enable fast math functions (trading precision for speed), which is suitable for scenarios where precision requirements are not strict.

### Avoid Implicit Double-Precision Promotion

The double-precision throughput of many GPUs is only 1/32 or 1/64 of single-precision throughput.

````c
// ❌ 3.14 is double, causing entire expression to be computed in double precision
float result = x * 3.14;

// ✅ Add f suffix to keep single precision
float result = x * 3.14f;
````

### FMA (Fused Multiply-Add)

The compiler automatically fuses ``a * b + c`` into a single FMA instruction:
- Only one rounding is performed (more precise than separate multiply + add)
- Throughput is doubled (one instruction performs two operations)
- Note: FMA results may have slight numerical differences from separate computations

### Integer Division and Modulo Optimization

Integer division and modulo operations are very expensive (~20 instructions):

````c
// ❌ Expensive division and modulo
int q = n / divisor;
int r = n % divisor;

// ✅ Use bitwise operations when divisor is power of 2
int q = n >> log2_divisor;
int r = n & (divisor - 1);
````

The compiler can automatically optimize division by literal powers of 2, but variable division requires manual optimization.

## Control Flow Optimization

### Minimize Warp Divergence

Branch divergence is a GPU performance killer. Optimization strategies:

1. **Align branch conditions by warp**: Make the entire warp follow the same path
2. **Reduce the number of conditional branches**: Replace simple branches with arithmetic operations
3. **Move divergent branches to warp boundaries**

````c
// ❌ Each thread may take different paths
if (data[tid] > threshold) {
    result = expensive_compute(data[tid]);
} else {
    result = 0.0f;
}

// ✅ Use multiplication to eliminate branches (when both paths are short)
float flag = (data[tid] > threshold) ? 1.0f : 0.0f;
result = flag * expensive_compute(data[tid]);
// Note: expensive_compute will still execute, only suitable when computation is minimal
````

### Loop Unrolling

````c
// Compiler hint to unroll
#pragma unroll
for (int i = 0; i < 4; i++) {
    sum += a[i] * b[i];
}

// Fully unrolled to:
// sum += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
````

Unrolling reduces loop control overhead and increases ILP, but excessive unrolling increases register pressure.

## Data Type Selection

### Vectorized Load/Store

Use vector types to load multiple elements at once, reducing the instruction count:

````c
// Scalar load: 4 load instructions
float a = data[idx];
float b = data[idx+1];
float c = data[idx+2];
float d = data[idx+3];

// Vector load: 1 load instruction (128-bit)
float4 vec = reinterpret_cast<float4*>(data)[idx/4];
````

### Half-Precision and Mixed Precision

- FP16 (``half``): Storage bandwidth is halved, and matrix cores (Tensor Core / Matrix Core) can accelerate it
- BF16: Same dynamic range as FP32, slightly lower precision, commonly used in deep learning training
- Mixed-precision strategy: Use FP16/BF16 for computation and FP32 for accumulation

## Performance Analysis Methods

### Profile-Driven Optimization

**Always profile first, then optimize**. Use tools to locate bottlenecks:

- Various platforms provide dedicated kernel profilers and system-level tracers
- Focus on instruction throughput, memory bandwidth utilization, and warp/wavefront activity

### Key Performance Metrics

| Metric | Meaning | Target |
|------|------|------|
| Effective Bandwidth | Actual data transferred / time | Close to theoretical peak |
| Compute Utilization | Actual FLOPS / Peak FLOPS | Should be close to peak when compute-bound |
| Occupancy | Active warps / Max warps | Typically > 50%, but not absolute |
| SM Utilization | Percentage of SMs with active warps | Close to 100% |

### Roofline Model

Determine whether a kernel is **compute-bound** or **memory-bound**:

Arithmetic Intensity (AI) = FLOPS / bytes accessed
If AI < Peak FLOPS / Peak Bandwidth (ridge point) → memory-bound
If AI > ridge point → compute-bound

- Memory-bound kernel: Optimize memory access patternsestic, reduce redundant accesses
- Compute-bound kernel: Use Tensor Core, improve ILP, optimize algorithms

## Related

- **Prerequisites**: [GPU Execution Model](gpu-execution-model.md) — warp divergence fundamentals
- **Prerequisites**: [GPU Memory Hierarchy](gpu-memory-hierarchy.md) — memory access pattern fundamentals
- **Content Overlap**: The warp divergence section of this article overlaps with ``gpu-execution-model.md``; the latter focuses on definitions while this article focuses on optimization techniques
- **NVIDIA Profiling**: [NCU Profiling Guide](../nvidia/common/profiling/ncu-profiling-guide.md) — practical tool usage for Roofline analysis
- **AMD Profiling**: [rocprofv3 Instruction-Level Profiling](../amd/gluon/gfx942/profiling_guide.md)
