# PTX Core Instruction Set

## Integer Arithmetic Instructions

### Basic Arithmetic

```
// Addition
add.u32      %r, %a, %b;         // r = a + b
add.cc.u32   %r, %a, %b;         // With carry output
addc.u32     %r, %a, %b;         // With carry input
add.s64      %rd, %ra, %rb;      // 64-bit signed addition

// Subtraction
sub.u32      %r, %a, %b;
sub.cc.u32   %r, %a, %b;         // With borrow output
subc.u32     %r, %a, %b;         // With borrow input

// Multiplication
mul.lo.u32   %r, %a, %b;         // Low 32-bit result
mul.hi.u32   %r, %a, %b;         // High 32-bit result
mul.wide.u32 %rd, %a, %b;        // 32x32 → 64-bit result

// Multiply-add
mad.lo.u32   %r, %a, %b, %c;     // r = a*b + c (low 32 bits)
mad.hi.u32   %r, %a, %b, %c;     // r = (a*b)_hi + c
mad.wide.u32 %rd, %a, %b, %rc;   // 32x32+64 → 64-bit

// Division and modulo
div.u32      %r, %a, %b;
rem.u32      %r, %a, %b;
```

### Bit Manipulation Instructions

```
abs.s32      %r, %a;             // Absolute value
neg.s32      %r, %a;             // Negation
min.u32      %r, %a, %b;
max.s32      %r, %a, %b;
popc.b32     %r, %a;             // Population count (count of 1s)
clz.b32      %r, %a;             // Count leading zeros
bfind.u32    %r, %a;             // Most significant bit position
brev.b32     %r, %a;             // Bit reversal
bfe.u32      %r, %a, %pos, %len; // Bit field extraction
bfi.b32      %r, %a, %b, %pos, %len;  // Bit field insertion
```

### Dot Product Instructions (SM61+)

```
// 4-element 8-bit dot product, accumulated to 32-bit
dp4a.u32.u32 %r, %a, %b, %c;    // r = c + dot4(a[0:3], b[0:3])
dp4a.s32.s32 %r, %a, %b, %c;    // Signed version
dp4a.u32.s32 %r, %a, %b, %c;    // Mixed signedness

// 2-element 16-bit dot product
dp2a.lo.u32.u32 %r, %a, %b, %c; // r = c + a[0]*b[0] + a[1]*b[1]
dp2a.hi.u32.u32 %r, %a, %b, %c; // Uses high 16 bits of b
```

---

## Floating-Point Arithmetic Instructions

### Basic Operations

```
// English comment
add.rn.f32   %f, %fa, %fb;       // round to nearest
sub.rz.f32   %f, %fa, %fb;       // round toward zero
mul.rm.f32   %f, %fa, %fb;       // round toward -inf

// FMA(Fused Multiply-Add) -
fma.rn.f32 %f, %fa, %fb, %fc; // f = a*b + c
fma.rn.f64 %fd, %da, %db, %dc; // accuracy FMA

// MAD(Multiply-Add) - (SM20+ compilation FMA)
mad.rn.f32   %f, %fa, %fb, %fc;

// division
div.rn.f32 %f, %fa, %fb; // completeaccuracy
div.approx.f32 %f, %fa, %fb; //
div.full.f32 %f, %fa, %fb; // completerange
```

### Special Functions

```
// English comment
rcp.rn.f32 %f, %fa; // 1/a(completeaccuracy)
rcp.approx.f32 %f, %fa; // 1/a(, 1 )
sqrt.rn.f32      %f, %fa;        // sqrt(a)
rsqrt.approx.f32 %f, %fa; // 1/sqrt(a)

// English comment
sin.approx.f32 %f, %fa; // sin(a), inputrange [-pi, pi]
cos.approx.f32   %f, %fa;        // cos(a)
lg2.approx.f32   %f, %fa;        // log2(a)
ex2.approx.f32   %f, %fa;        // 2^a

// Tanh（SM75+）
tanh.approx.f32  %f, %fa;        // tanh(a)
```

### Floating-Point Modifiers

| Modifier | Meaning | Typical Use |
|--------|------|---------|
| `.ftz` | Flush denormals to zero | Avoid denormal performance penalty |
| `.sat` | Saturate to [0.0, 1.0] | Normalized output |
| `.approx` | Use hardware approximation unit | Performance-critical path |
| `.rn` | Round to nearest even | Default IEEE rounding |
| `.rz` | Round toward zero | Truncation semantics |

### Half Precision Operations (SM53+)

```
// FP16 arithmetic (paired operations)
add.rn.f16x2   %h, %ha, %hb;     // 2 f16 simultaneous addition
mul.rn.f16x2   %h, %ha, %hb;     // 2 f16 simultaneous multiplication
fma.rn.f16x2   %h, %ha, %hb, %hc;

// BF16 arithmetic (SM80+)
add.rn.bf16x2  %h, %ha, %hb;
fma.rn.bf16x2  %h, %ha, %hb, %hc;

// Single f16/bf16 operation
add.rn.f16     %h, %ha, %hb;
mul.rn.bf16    %h, %ha, %hb;

// Mixed precision
fma.rn.f32     %f, %ha, %hb, %fc;  // f16 inputs, f32 accumulate
```### Floating-Point Comparison and Selection

```
// min/max
min.f32       %f, %fa, %fb;
max.ftz.f32 %f, %fa, %fb; // flush-to-zero
min.NaN.f32 %f, %fa, %fb; // NaN (SM80+)

// English comment
selp.f32      %f, %fa, %fb, %p;  // f = p ? fa : fb
slct.f32.s32  %f, %fa, %fb, %c;  // f = (c >= 0) ? fa : fb
```

---

## Logical and Shift Instructions

### Bitwise Logic

```
and.b32 %r, %a, %b; // bybit
or.b32 %r, %a, %b; // bybitor
xor.b32 %r, %a, %b; // bybitor
not.b32 %r, %a; // bybit
cnot.b32   %r, %a;               // r = (a == 0) ? 1 : 0
```

### lop3 — Three-Input Arbitrary Boolean (SM50+)

```
lop3.b32  %r, %a, %b, %c, immLut;
// r[i] = F(a[i], b[i], c[i]), F 8-bit LUT immLut definition
//
// LUT :
//   a = 0xF0, b = 0xCC, c = 0xAA
// immLut =
//
// example:
//   AND3:  immLut = 0xF0 & 0xCC & 0xAA = 0x80
//   OR3:   immLut = 0xF0 | 0xCC | 0xAA = 0xFE
//   XOR3:  immLut = 0xF0 ^ 0xCC ^ 0xAA = 0x96
//   (a & b) | c: immLut = (0xF0 & 0xCC) | 0xAA = 0xEA
```

**LUT calculation method:** Substitute `a=0xF0, b=0xCC, c=0xAA` into the desired boolean expression to obtain the immLut value. A single instruction implements any three-input boolean function.

### Shift

```
shl.b32 %r, %a, %n; // left
shr.u32 %r, %a, %n; // logicalright
shr.s32 %r, %a, %n; // right
```

### shf — Funnel Shift

```
// registerbit, 32 bit
shf.l.wrap.b32 %r, %lo, %hi, %n; // leftbit
shf.r.clamp.b32 %r, %lo, %hi, %n; // rightbit, clamp bit

// English comment
// 1. 64-bit bithigh
// 2. bit
// 3. registerbytes
```

---

## Data Movement Instructions

### Load (ld)

```
// load
ld.global.f32 %f, [%ptr]; // global memory
ld.shared.v4.f32 {%f0,%f1,%f2,%f3}, [%sptr]; // vector load

// Cache
ld.global.ca.f32 %f, [%ptr]; // cache at all levels(default)
ld.global.cg.f32  %f, [%ptr];   // cache at L2 only
ld.global.cs.f32 %f, [%ptr]; // cache streaming( evict data)
ld.global.lu.f32 %f, [%ptr]; // last use(access)
ld.global.cv.f32 %f, [%ptr]; // cache volatile( L1)

// load(SM70+)
ld.global.acquire.gpu.f32 %f, [%ptr]; // acquire
ld.global.relaxed.sys.f32     %f, [%ptr];  // relaxed + sys scope
```

### Store (st)

```
// store
st.global.f32     [%ptr], %f;
st.shared.v2.f32  [%sptr], {%f0, %f1};

// Cache
st.global.wb.f32 [%ptr], %f; // write-back(default)
st.global.cg.f32  [%ptr], %f;   // cache at L2 only
st.global.cs.f32  [%ptr], %f;   // cache streaming
st.global.wt.f32  [%ptr], %f;   // write-through

// store(SM70+)
st.global.release.gpu.f32     [%ptr], %f;
```

### Cache Policy Selection Guide

| Load Scenario | Recommended | Reason |
|-----------|------|------|
| Frequently reused | `.ca` | Cached at all levels |
| Streaming access (traverse once) | `.cs` | Does not pollute cache |
| Data no longer needed | `.lu` | Hint evict |
| Cross-CTA communication | `.cv` | Bypass L1enges to see latest value |

| Store Scenario | Recommended | Reason |
|------------|------|------|
| General write-back | `.wb` | Leverage cache coalescing |
| Cross-CTA visible | `.wt` | Write-through to L2 immediately |
| Streaming write-out | `.cs` | Does not pollute cache |

### mov — Register Move

```
mov.f32 %f, %g; // register
mov.u32 %r, 42; //
mov.b64 %rd, {%r0, %r1}; // 32-bit 64-bit
mov.b32 {%h0, %h1}, %r; // 32-bit 16-bit
```

### prmt — Byte Permutation

```
// Byte permutation
prmt.b32  %r, %a, %b, %selector;

// Each 4-bit nibble of selector selects a byte:
//   0-3: from a's byte 0-3
//   4-7: from b's byte 0-3
//   MSB bit: sign-extension/zero-extension control

// Predefined permutation patterns
prmt.b32.f4e  %r, %a, %b, %c;    // forward 4 extract
prmt.b32.b4e  %r, %a, %b, %c;    // backward 4 extract
prmt.b32.ecl  %r, %a, %b, %c;    // edge clamp left
prmt.b32.ecr  %r, %a, %b, %c;    // edge clamp right
```### shfl.sync — Warp Shuffle

```
// Intra-warp thread exchange via shuffle, bypassing shared memory
shfl.sync.up.b32    %r, %src, %delta, 0x1F, 0xFFFFFFFF;
shfl.sync.down.b32  %r, %src, %delta, 0x1F, 0xFFFFFFFF;
shfl.sync.bfly.b32  %r, %src, %laneMask, 0x1F, 0xFFFFFFFF;
shfl.sync.idx.b32   %r, %src, %srcLane, 0x1F, 0xFFFFFFFF;

// Modes:
//   .up   — read from lane - delta
//   .down — read from lane + delta
//   .bfly — read from lane XOR laneMask (butterfly exchange)
//   .idx  — read from specified srcLane

// Last two parameters:
//   clamp/mask value (control warp partition)
//   member mask (which lanes participate)
```

---

## Atomic Operations

```
// Basic atomic operations
atom.global.add.u32         %old, [%ptr], %val;     // Atomic add
atom.shared.cas.b32         %old, [%ptr], %cmp, %val; // Compare-and-swap
atom.global.exch.b32        %old, [%ptr], %val;     // Atomic exchange
atom.global.and.b32         %old, [%ptr], %val;     // Atomic AND
atom.global.or.b32          %old, [%ptr], %val;     // Atomic OR
atom.global.xor.b32         %old, [%ptr], %val;     // Atomic XOR
atom.global.min.s32         %old, [%ptr], %val;     // Atomic minimum
atom.global.max.u32         %old, [%ptr], %val;     // Atomic maximum
atom.global.inc.u32         %old, [%ptr], %val;     // Atomic increment (modulo val)
atom.global.dec.u32         %old, [%ptr], %val;     // Atomic decrement (modulo val)
atom.global.add.f32         %old, [%ptr], %val;     // Atomic float add

// Atomics with semantics (SM70+)
atom.global.acquire.gpu.add.f32  %old, [%ptr], %val;
atom.global.acq_rel.sys.cas.b64  %old, [%ptr], %cmp, %val;
atom.global.relaxed.gpu.add.u32  %old, [%ptr], %val;

// Reductions (no return old value, possibly faster)
red.global.add.u32          [%ptr], %val;
red.global.relaxed.gpu.add.f32 [%ptr], %val;

// FP16 atomics (SM70+)
atom.global.add.noftz.f16x2  %old, [%ptr], %val;   // 2 f16 atomic add

// BF16 atomics (SM90+)
atom.global.add.noftz.bf16x2 %old, [%ptr], %val;
```

---

## Warp-Level Voting and Reduction

### Vote Instructions

```
// All threads participate in voting
vote.sync.all.pred   %p, %q, 0xFFFFFFFF;   // Are all lanes' q true?
vote.sync.any.pred   %p, %q, 0xFFFFFFFF;   // Is any lane's q true?
vote.sync.uni.pred   %p, %q, 0xFFFFFFFF;   // Are all lanes' q the same?
vote.sync.ballot.b32 %r, %q, 0xFFFFFFFF;   // Per-lane 1-bit voting result

// Typical usage: check warp branch
vote.sync.all.pred %p, %valid, 0xFFFFFFFF;
@%p bra ALL_VALID;
```

### Match Instructions (SM70+)

```
// Find lanes with same value in warp
match.any.sync.b32   %mask, %val, 0xFFFFFFFF;   // Return mask of lanes with same value
match.all.sync.b32   %mask, %val, %p, 0xFFFFFFFF; // Do all lanes have same value? p = whether all identical
```

### Redux — Warp Reduction (SM80+)

```
redux.sync.add.s32   %r, %val, 0xFFFFFFFF;   // Warp summation
redux.sync.min.u32   %r, %val, 0xFFFFFFFF;   // Warp minimum
redux.sync.max.s32   %r, %val, 0xFFFFFFFF;   // Warp maximum
redux.sync.and.b32   %r, %val, 0xFFFFFFFF;   // Warp bitwise AND
redux.sync.or.b32    %r, %val, 0xFFFFFFFF;   // Warp bitwise OR
redux.sync.xor.b32   %r, %val, 0xFFFFFFFF;   // Warp bitwise XOR
```

### Elect Instructions (SM90+)

```
// Select a leader from active threads
elect.sync %pred, 0xFFFFFFFF;
// pred is true only for selected leader
@%pred st.global.u32 [%ptr], %val;  // Only leader executes
```

### Activemask

```
activemask.b32 %mask;             // Get current active thread mask
// No synchronization! Just queries current convergence status
```
