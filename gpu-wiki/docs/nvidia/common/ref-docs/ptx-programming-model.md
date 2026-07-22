# PTX Programming Model and Basics

## PTX Overview

PTX (Parallel Thread Execution) is NVIDIA GPU's virtual ISA (Intermediate Representation), sitting between CUDA C++ and hardware SASS. PTX provides a stable programming interface, and the compiler (ptxas) translates PTX into SASS instructions for a specific GPU architecture.

```
CUDA C++ / CuTeDSL Python → nvcc/MLIR → PTX → ptxas → SASS (GPU execution)
```

**Key Features:**
- Forward compatibility: Old PTX can run on new GPUs (ptxas recompiles)
- Virtual architecture: PTX hides hardware details, providing a unified programming model
- Inline embedding: PTX can be embedded in CUDA C++ via `asm volatile`

---

## State Spaces

PTX divides memory into multiple logical state spaces, mapped to different hardware storage levels:

| State Space | Keyword | Scope | Hardware Mapping | Characteristics |
|-------------|---------|-------|------------------|-----------------|
| Register | `.reg` | Thread-private | Register file | Fastest, limited quantity (max 255 per thread) |
| Shared Memory | `.shared` | Shared within CTA | On-chip SMEM | Low latency, requires manual synchronization |
| Global Memory | `.global` | Visible to all threads | DRAM + Cache | High latency, large capacity |
| Local Memory | `.local` | Thread-private | DRAM (register spill) | Logically private, physically in DRAM |
| Constant Memory | `.const` | Read-only for all threads | Dedicated cache | 64 KB limit, broadcast optimization |
| Parameter Space | `.param` | Kernel parameters | Constant cache/register | Kernel argument passing |

### .shared Subtypes (SM90+)

```
.shared::cta // current CTA access(default)
.shared::cluster // Cluster CTA access(DSMEM)
```

### .const bank

```
.const[0] .const[10] // 11 independent bank
// bank 0: compilation
// bank 1-10: (driver )
```

### Generic Address Space `.generic`

SM20+ supports **generic address space**, with the actual space determined at runtime:

```
// pointer global, shared or local
ld.generic [ptr], %r1; // row

// conversion
cvta.to.global   %gptr, %ptr;    // generic → global
cvta.to.shared   %sptr, %ptr;    // generic → shared
cvta.shared.to.generic %ptr, %sptr;  // shared → generic
```

---

## Basic Data Types

### Integer Types

| Type | Width | Signedness |
|------|-------|------------|
| `.pred` | 1-bit | Predicate (boolean) |
| `.b8`, `.b16`, `.b32`, `.b64` | 8/16/32/64-bit | Untyped bit container |
| `.u8`, `.u16`, `.u32`, `.u64` | 8/16/32/64-bit | Unsigned integer |
| `.s8`, `.s16`, `.s32`, `.s64` | 8/16/32/64-bit | Signed integer |

### Floating-Point Types

| Type | Width | Precision | Introduced in |
|------|-------|-----------|---------------|
| `.f16` | 16-bit | IEEE FP16 | SM53+ |
| `.bf16` | 16-bit | BFloat16 | SM80+ |
| `.f32` | 32-bit | IEEE FP32 | All |
| `.f64` | 64-bit | IEEE FP64 | All |
| `.tf32` | 19-bit (32-bit storage) | TF32 | SM80+ (MMA only) |
| `.e4m3` | 8-bit | FP8 E4M3 | SM89+ (MMA only) |
| `.e5m2` | 8-bit | FP8 E5M2 | SM89+ (MMA only) |
| `.e2m3` | 6-bit | FP6 E2M3 | SM100+ (MMA only) |
| `.e3m2` | 6-bit | FP6 E3M2 | SM100+ (MMA only) |
| `.e2m1` | 4-bit | FP4 E2M1 | SM100+ (MMA only) |

### Vector Types

```
.v2 // 2 vector: .v2.f32, .v2.u32
.v4 // 4 vector: .v4.f32, .v4.b8
.v8 // 8 ( .b16/.b32)
```

---

## Variable Declaration

``````
// Registers
.reg .f32 %f<32>;        // 32 f32 registers
.reg .pred %p;              // Predicate registers

// Shared memory
.shared .align 128 .b8 smem[16384];  // 16 KB, 128-byte aligned

// Global memory
.global .align 4 .f32 data[1024];

// Parameters
.param .u64 %ptr;
.param .align 16 .b8 %struct[64];    // Structure parameters
``````

---

## Special Registers

### Thread and CTA Identifiers

| Register | Meaning | Usage |
|----------|---------|-------|
| `%tid.x/y/z` | Thread index within CTA | `mov.u32 %r, %tid.x;` |
| `%ntid.x/y/z` | CTA dimensions (block size) | |
| `%ctaid.x/y/z` | CTA index in Grid | |
| `%nctaid.x/y/z` | Grid dimensions | |
| `%laneid` | Lane within warp (0-31) | |
| `%warpid` | Warp ID within CTA | Not guaranteed contiguous |

### Cluster Identifiers (SM90+)

| Register | Meaning |
|----------|---------|
| `%clusterid.x/y/z` | Cluster index in Grid |
| `%nclusterid.x/y/z` | Number of clusters in Grid |
| `%cluster_ctaid.x/y/z` | CTA index within Cluster |
| `%cluster_nctaid.x/y/z` | Number of CTAs in Cluster |
| `%cluster_ctarank` | CTA rank within Cluster |
| `%cluster_nctarank` | Total number of CTAs in Cluster |

### Performance and Debug Registers

| Register | Description |
|----------|-------------|
| `%clock`, `%clock_hi` | 32/64-bit cycle counter |
| `%clock64` | 64-bit cycle counter |
| `%globaltimer` | 64-bit global timer (ns) |
| `%dynamic_smem_size` | Dynamic shared memory size (bytes) |
| `%total_smem_size` | Total shared memory size |
| `%lanemask_eq/lt/le/gt/ge` | Current lane relative mask |

---

## Predicated Execution

PTX makes extensive use of predicates identity for conditional execution, avoiding branches:

```
// Set predicate
setp.gt.f32 %p, %f1, 0.0;       // p = (f1 > 0.0)

// Conditional execution
@%p  add.f32 %f2, %f2, %f1;     // if (p) f2 += f1
@!%p mov.f32 %f2, 0.0;          // if (!p) f2 = 0.0

// Conditional branch
@%p  bra TARGET;                  // if (p) goto TARGET
```

### setp Comparison Operations

```
// Integer comparison
setp.eq/ne/lt/le/gt/ge.u32 %p, %a, %b;

// Floating-point comparison (with NaN handling)
setp.lt.f32     %p, %a, %b;     // ordered: NaN → false
setp.ltu.f32    %p, %a, %b;     // unordered: NaN → true
setp.num.f32    %p, %a, %b;     // Are both non-NaN?
setp.nan.f32    %p, %a, %b;     // Is either NaN?
```

---

## Control Flow

```
// Branch
bra         LABEL;               // Unconditional jump
bra.uni     LABEL;               // Uniform jump (all threads same target)
@%p bra     LABEL;               // Conditional jump

// Function call
call (%ret), func, (%arg1, %arg2);
call.uni (%ret), func, (%arg1);  // Uniform call

// Kernel return
ret;
exit;                            // Terminate current thread
```

---

## Performance Tuning Directives

```
// Maximum registers per thread
.maxnreg 128                    // Limit compiler to use ≤128 registers

// Maximum threads per CTA
.maxntid 256, 1, 1              // Hint compiler for max block size 256 threads
.reqntid 256, 1, 1              // Require block to be exactly 256 threads

// Minimum CTA/SM
.minnctapersm 2                 // At least 2 CTA/SM (increase occupancy)

// Pragma
.pragma "nounroll";             // Disable loop unrolling

// Cluster size (SM90+)
.reqnctapercluster 4            // Require 4 CTA/cluster
.maxclusterrank 8               // Cluster max rank
```

---

## Memory Consistency Model (SM70+)

### Scope

| Scope | Description |
|-------|-------------|
| `.cta` | Threads within the current CTA |
| `.cluster` | All CTAs within the cluster (SM90+) |
| `.gpu` | All threads on the same GPU |
| `.sys` | System-level (including CPU and other GPUs) |

### Semantics (Ordering)

| Semantic | Description |
|----------|-------------|
| `.relaxed` | No ordering guarantee (atomicity only) |
| `.acquire` | Acquire: subsequent operations see the releasing side's writes |
| `.release` | Release: prior writes are visible to the acquiring side |
| `.acq_rel` | Combined acquire + release |
| `.sc` | Sequential consistency (strongest, SM70+) |

### Fence Instruction

```
fence.sc.gpu; // GPU range fence
fence.acq_rel.sys; // range acquire-release fence
fence.proxy.alias; // proxy fence
fence.proxy.async; // asynchronous proxy fence(cp.async complete)
fence.proxy.async.global; // asynchronous global proxy fence
fence.proxy.async.shared::cta; // asynchronous shared proxy fence
```

### Membar (Legacy Interface, Still Supported)

```
membar.cta; // CTA rangememorybarrier
membar.gl; // globalmemorybarrier
membar.sys; // memorybarrier
```

---

## Type Conversion

```
// Basic conversions
cvt.f32.f16   %f, %h;           // f16 → f32
cvt.rn.f16.f32 %h, %f;          // f32 → f16 (round nearest)
cvt.rz.f32.s32 %f, %i;          // s32 → f32 (round toward zero)

// Rounding modes
.rn    // round to nearest even (default)
.rz    // round toward zero
.rm    // round toward minus infinity
.rp    // round toward plus infinity

// Integer saturation
cvt.sat.u8.s32 %b, %i;          // Clamp to [0, 255]

// FP8 conversion (SM89+)
cvt.rn.satfinite.e4m3.f32 %e, %f;  // f32 → fp8 e4m3
cvt.f32.e5m2 %f, %e;               // fp8 e5m2 → f32
```## PTX Syntax Essentials

### Instruction Format

```
opcode.modifier1.modifier2.type  dest, src1, src2;
```

- **opcode**: Operation code (add, mul, ld, st, ...)
- **modifier**: Modifier (.rn, .sat, .ftz, .lo, .wide, ...)
- **type**: Data type (.f32, .u64, .b16, ...)
- **dest**: Destination operand
- **src**: Source operand

### Common Modifiers

| Modifier | Applicable To | Meaning |
|--------|------|------|
| `.rn/.rz/.rm/.rp` | Float | Rounding mode |
| `.ftz` | FP32 | Flush subnormals to zero |
| `.sat` | Float/Integer | Saturate to [0, 1] or type range |
| `.lo/.hi/.wide` | Integer multiply | Low/high half/full-width result |
| `.uni` | Branch/Call | All threads follow the same path |
| `.approx` | Special functions | Use approximate instructions (fast path) |

---

## PTX Kernel Declaration

```
.version 8.6
.target sm_90
.address_size 64

.visible .entry my_kernel (
    .param .u64 param_ptr,
    .param .u32 param_N
)
{
    .reg .u32 %r<10>;
    .reg .f32 %f<10>;
    .reg .pred %p;

    ld.param.u64 %rd0, [param_ptr];
    ld.param.u32 %r0, [param_N];

    // kernel body

    ret;
}
```
```
- `.version` — PTX version
- `.target` — Target architecture (sm_70, sm_80, sm_90, sm_100)
- `.address_size` — Address width (32 or 64)
- `.visible .entry` — Externally visible kernel entry
- `.func` — Device function (non-entry)
```
