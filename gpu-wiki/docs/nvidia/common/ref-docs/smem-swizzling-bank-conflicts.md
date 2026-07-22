# Shared Memory Swizzling and Bank Conflict Elimination

## Overview

NVIDIA GPU's Shared Memory (SMEM) achieves high bandwidth through **parallel bank access**, but when multiple threads access the same bank simultaneously, **bank conflicts** occur, leading to serialization. This article explains the SMEM banking model, common conflict scenarios, and the principles and practices of using XOR swizzle to eliminate conflicts.

---

## 1. NVIDIA SMEM Banking Model

### 1.1 Basic Parameters

| Parameter | Value | Description |
|------|-----|------|
| Number of Banks | 32 | Consistent across all SM architectures (Volta to Blackwell) |
| Bank Width | 4 bytes | Each bank can serve one 4B read/write per cycle |
| Bandwidth/Cycle | 128 bytes | 32 banks x 4B = 128B/cycle |
| Minimum Allocation Granularity | 32B sector | SMEM is accessed with 32B alignment |

### 1.2 Address-to-Bank Mapping

Consecutive 4-byte words map to consecutive banks:

```
Address (bytes)    Bank
0x00 - 0x03    Bank 0
0x04 - 0x07    Bank 1
0x08 - 0x0B    Bank 2
...
0x7C - 0x7F    Bank 31
0x80 - 0x83    Bank 0    ← Back to Bank 0 after 128B
0x84 - 0x87    Bank 1
...
```

Formula: `bank_id = (byte_address / 4) % 32`

### 1.3 Multi-Word Access Patterns

| Access Width | Banks Spanned | Description |
|----------|-------------|------|
| 4B (float) | 1 bank | Most basic unit |
| 8B (double/float2) | 2 banks | Two consecutive banks |
| 16B (float4) | 4 banks | Four consecutive banks, requires 16B alignment |

> Note: 8B and 16B accesses occupy multiple banks, but as long as each bank is exclusively used by a different thread, there is no conflict.

### 1.4 Bank Conflict Definition

When **multiple threads within the same warp** access **different addresses of the same bank** in the **same cycle**, an N-way bank conflict occurs:

- 2-way conflict: 2 threads access the same bank → 2 serialized accesses
- 32-way conflict: All 32 threads access Bank 0 → 32 serialized accesses = worst case

Special case: If multiple threads access **the same address of the same bank** (the same 4B word), the hardware broadcasts and no conflict occurs.

---

## 2. Common Bank Conflict Scenarios

### 2.1 Column-Major Access on Row-Major Data — Matrix Transpose

This is the most classic source of bank conflicts.

```cuda
// Assume TILE = 32, float = 4B
__shared__ float smem[32][32];  // Row-major storage

// No conflict: each thread reads different columns of same row
float val = smem[row][threadIdx.x];  // stride-1 access, 32 threads → 32 different banks

// 32-way conflict! Each thread reads different rows of same column
float val = smem[threadIdx.x][col];  // stride-32 access
// threadIdx.x=0 → bank (0*32+col)%32 = col%32
// threadIdx.x=1 → bank (1*32+col)%32 = col%32  ← Same bank!
// All 32 threads land on bank (col%32)
```

Root cause analysis: Each row in `smem[32][32]` is exactly 32 floats = 128 bytes, which exactly covers one full round of banks. Therefore, the same column position in adjacent rows falls on the same bank.

### 2.2 Bank Conflicts in Matrix Transpose

```cuda
__shared__ float tile[32][32];

// Write (row-wise): No conflict
tile[threadIdx.y][threadIdx.x] = input[gy][gx];

// Read (column-wise transpose): 32-way conflict!
output[gx][gy] = tile[threadIdx.x][threadIdx.y];
```

### 2.3 Reduction Operations

In tree-based reductions, bank conflicts easily occur when the stride is a power of 2:

```cuda
__shared__ float sdata[256];

// stride = 1: No conflict
// stride = 2: 2-way conflict (threads 0,16 both access bank 0)
// stride = 16: 16-way conflict
// stride = 32: 32-way conflict
for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
}
```

---

## 3. Solution 1: Padding

The simplest method: add padding at the end of each row to break bank alignment.

```cuda
// Original: Each row 32 floats = 128 bytes → Row spacing = multiple of 32 banks
__shared__ float smem[32][32];  // 32-way conflict on column access

// Fixed: Each row 33 floats → Row spacing = 33 × 4B = 132B → No longer multiple of 128B
__shared__ float smem[32][33];  // No conflict!
```

Principle: bank of `smem[i][col]` = `(i * 33 + col) % 32`, so the same column in adjacent rows no longer falls on the same bank.

| Advantages | Disadvantages |
|------|------|
| Simple to implement, one line of code | Wastes SMEM (approximately 3% overhead) |
| Easy to understand and maintain | Breaks alignment, preventing 128B vectorized loads |
| | Tile sizes are not powers of 2, limiting compiler optimizations |

> For scenarios like GEMM kernels that require efficient vectorized access, the disadvantages of padding are significant.

---

## 4. Solution 2: XOR Swizzle (Key Focus)

### 4.1 Core Idea

Swizzle uses **bitwise operations (XOR)** to remap SMEM addresses, distributing originally conflicting access patterns across different banks while preserving data locality and vectorization capability.

Key constraint: the swizzle function must be an **involution** (self-inverse function), i.e., `f(f(x)) = x`, so that the same function can be used for both writes and reads. XOR naturally satisfies this property.

### 4.2 Swizzle Formula

Given an SMEM byte offset `offset`, the swizzled address is:

```
swizzled_offset = offset ^ (extract_bits(offset, Y_pos, B) << Z_pos)
```

Where:
- `B` = number of swizzle bits (1, 2, or 3 bits)
- `Y_pos` = starting position of the source bit field (upper region)
- `Z_pos` = starting position of the destination bit field (lower region, near bank selection bits)

Equivalent expression: take B bits from the Y bit field and XOR them with the B bits of the Z bit field in the offset.

### 4.3 Three-Parameter Swizzle Model: `Swizzle<B, M, S>`

The CUTLASS/CuTe libraries use three parameters to precisely describe swizzle:

```
Swizzle<BBits, MBase, SShift>

- BBits: Number of swizzle bits (1, 2, or 3) → Affects 2^B rows as a group
- MBase: Lowest bits that remain unchanged (skipped starting from bit 0)
- SShift: Y bit field offset relative to Z bit field (|SShift| >= BBits)
```

Bit structure of the offset:

```
bit position:  ...  [Y: B bits]  ...  [Z: B bits]  [M bits]
                ↑                       ↑              ↑
 High bits, row index Bank selection Invariant (intra-word offset)

Operation: ZZZ ^= YYY (B bits from the Z bit field are XORed with B bits from the Y bit field)```

Operation: `ZZZ ^= YYY` (B bits from the Z bit field are XORed with B bits from the Y bit field)

### 4.4 Standard Swizzle Patterns

For NVIDIA GPUs (M=4, i.e., the lower 4 bits are reserved, corresponding to a minimum SMEM access granularity of 16 bytes = 128 bits):

| Name | Parameters | BBits | MBase | SShift | XOR Granularity | Applicable Scenario |
|------|------|-------|-------|--------|---------|---------|
| **B32** | `Swizzle<1,4,3>` | 1 | 4 | 3 | 32B (2 sectors) | Small tile, 32B row width |
| **B64** | `Swizzle<2,4,3>` | 2 | 4 | 3 | 64B (4 sectors) | Medium tile, 64B row width |
| **B128** | `Swizzle<3,4,3>` | 3 | 4 | 3 | 128B (8 sectors) | Large tile, 128B row width (most common) |

> M=4 means the lower 4 bits (intra-16-byte offset) do not participate in swizzle; S=3 means the Y bit field is offset by 3 bits above Z.

Blackwell (SM100) also introduces a special mode:
- **B128_32B** = `Swizzle<2,5,2>`: M=5 (32-byte unchanged region), used for tcgen05 32B base alignment

### 4.5 Swizzle Visualization: B128 Mode (`Swizzle<3,4,3>`)

Using a matrix of type `half` (2B), 8 rows × 64 columns as an example (each row 128B = one B128 block):

**Bank Distribution Without Swizzle:**

```
Row 0: [Bank 0..7 ][Bank 8..15 ][Bank 16..23][Bank 24..31]
Row 1: [Bank 0..7 ][Bank 8..15 ][Bank 16..23][Bank 24..31]
Row 2: [Bank 0..7 ][Bank 8..15 ][Bank 16..23][Bank 24..31]
  ...    ↑ All rows same column in column access → Same bank → 8-way conflict
```

**After B128 Swizzle:**

```
Row 0: [Bank 0..7 ][Bank 8..15 ][Bank 16..23][Bank 24..31]  (row_idx=0, XOR=000)
Row 1: [Bank 8..15 ][Bank 0..7 ][Bank 24..31][Bank 16..23]  (row_idx=1, XOR=001)
Row 2: [Bank 16..23][Bank 24..31][Bank 0..7 ][Bank 8..15 ]  (row_idx=2, XOR=010)
Row 3: [Bank 24..31][Bank 16..23][Bank 8..15 ][Bank 0..7 ]  (row_idx=3, XOR=011)
Row 4: [Bank 0..7 ][Bank 8..15 ][Bank 16..23][Bank 24..31]  (row_idx=4, XOR=100→back to 000)
Row 5: [Bank 8..15 ][Bank 0..7 ][Bank 24..31][Bank 16..23]  (row_idx=5, XOR=101)
Row 6: [Bank 16..23][Bank 24..31][Bank 0..7 ][Bank 8..15 ]  (row_idx=6, XOR=110)
Row 7: [Bank 24..31][Bank 16..23][Bank 8..15 ][Bank 0..7 ]  (row_idx=7, XOR=111)
```

During column access, the same logical column falls on different banks in different rows → conflicts eliminated.

### 4.6 Bitwise Swizzle Example

Taking `Swizzle<3,4,3>` (B128) as an example, assuming access to `float16 smem[8][64]` (128B per row):

```
bytes = row * 128 + col * 2

row=0, col=0:  offset = 0x000 = 0b 000 0000 0000
row=1, col=0:  offset = 0x080 = 0b 000 1000 0000
row=2, col=0:  offset = 0x100 = 0b 001 0000 0000
row=3, col=0:  offset = 0x180 = 0b 001 1000 0000

bit(Swizzle<3,4,3>):
bit:   10 9 8 | 7 6 5 4 | 3 2 1 0
       Y Y Y  | Z Z Z   | M M M M
 row bank offset

Swizzle: Z' = Z ^ Y
row=0: Y=000, Z=000 → Z'=000 → Bank unchanged
row=1: Y=001, Z=000 → Z'=001 → Bank offset +1 group
row=2: Y=010, Z=000 → Z'=010 → Bank offset +2 groups
row=3: Y=011, Z=000 → Z'=011 → Bank offset +3 groups
```### 4.7 TMA Hardware Swizzle

Starting from Hopper (SM90), TMA (Tensor Memory Accelerator) supports swizzle at the hardware level:

- Set `swizzle_mode` in the TMA descriptor (None / 32B / 64B / 128B)
- Hardware **automatically** applies swizzle during GMEM→SMEM transfer
- MMA instructions (WGMMA/UMMA) automatically de-swizzle on read
- No need Kirk for software to explicitly compute swizzled addresses

```
TMA Swizzle Mode corresponds to:
  None  → No swizzle
  32B   → Swizzle<1,4,3>  (B32)
  64B   → Swizzle<2,4,3>  (B64)
  128B  → Swizzle<3,4,3>  (B128)
```

### 4.8 Data Type Width and Swizzle Selection

Consider the byte count per row when choosing a swizzle pattern:

| Data Type | Per Element | Bytes per Row (K=64) | Recommended Swizzle |
|----------|--------|-------------|-------------|
| FP32 | 4B | 256B | B128 |
| FP16/BF16 | 2B | 128B | B128 |
| FP8/INT8 | 1B | 64B | B64 |
| FP4 (packed) | 0.5B | 32B | B32 |

Principle: **The swizzle granularity should not exceed the number of bytes per row**. If a row is only 64B, using B128 swizzle will result in out-of-bounds access.

---

## 5. Diagnosing Bank Conflicts

### 5.1 NCU Metrics

Use NVIDIA Nsight Compute (NCU) to detect bank conflicts:

```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,\
              l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st \
    ./my_kernel
```

Key Metrics:

| Metric | Meaning |
|--------|------|
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` | Number of conflicts on SMEM loads |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st` | Number of conflicts on SMEM stores |
| `smem_bank_conflicts_per_request` | Derived metric: average conflicts per request |

### 5.2 Interpreting Conflict Counts

- **0**: Completely conflict-free
- **< warp count × 0.1**: Negligible
- **> warp count × 1.0**: Severe, optimization needed
- Distinguish LD from ST: In GEMM, LD conflicts are usually more critical (MMA instructions read from SMEM far more frequently than they write)

### 5.3 Rules of Thumb

1. **GEMM kernels**: Must use swizzle. B128 is the first choice unless the tile K dimension is too small
2. **Attention kernels**: Q/K/V SMEM layouts all require swizzle
3. **Simple elementwise**: Usually not needed (stride-1 access by threads is inherently conflict-free)
4. **Reduction kernels**: Using padding or warp shuffle instead of SMEM reduction is preferable
5. **Transpose kernels**: Must be handled; either swizzle or padding works

---

## 6. Architecture Differences

| Architecture | Codename | Bank Configuration | Swizzle Support |
|------|------|----------|-------------|
| Volta (SM70) | V100 | 32 banks × 4B | Software swizzle only |
| Ampere (SM80) | A100 | 32 banks × 4B | Software swizzle + cp.async |
| Hopper (SM90) | H100/H20 | 32 banks × 4B | TMA hardware swizzle (B32/B64/B128) |
| Blackwell (SM100) | B200 | 32 banks × 4B | TMA hardware swizzle + B128_32B mode |

### Volta/Ampere Era

- Swizzled addresses must be explicitly computed in software
- `cp.async` supports vectorized GMEM→SMEM copies, but does not automatically swizzle
- Typical pattern: Embed XOR computation in SMEM indexing

### Hopper/Blackwell Era

- TMA hardware performs swizzle, transparent to the programmer
- MMA instructions (WGMMA/UMMA) read directly from swizzled SMEM
- 128B swizzle is the default choice
- Blackwell adds `Swizzle<2,5,2>` mode (32B alignment on top of 128B, optimized for tcgen05)

---

## 7. General Swizzle Implementation Reference

Below is a framework-agnostic XOR swizzle implementation:

```cuda
// Generic B128 swizzle function
// Applicable when each row is 128B (e.g., FP16 x 64 columns)
__device__ int swizzle_offset(int row, int col, int elem_bytes) {
    int row_bytes = col * elem_bytes;
 int base_offset = row * 128 + row_bytes; // row 128B

 // Swizzle<3,4,3>: bit[9:7] XOR bit[6:4]
    int y_bits = (base_offset >> 7) & 0x7;  // 3 bits from row region
    int swizzled = base_offset ^ (y_bits << 4);
    return swizzled;
}

// useexample
__shared__ char smem_raw[8 * 128]; // 8 row x 128B

// write( GMEM load SMEM)
int offset = swizzle_offset(row, col, sizeof(half));
*reinterpret_cast<half*>(&smem_raw[offset]) = input_val;

// read(same swizzle function)
int offset = swizzle_offset(row, col, sizeof(half));
half val = *reinterpret_cast<half*>(&smem_raw[offset]);
```In production kernels, swizzle parameters are typically determined at compile time, using templates or `constexpr` to avoid runtime calculation overhead.

---

## 8. Summary and Best Practices

| Scenario | Recommended Solution |
|------|---------|
| Simple matrix transpose | Padding (`[N][N+1]`) is the easiest |
| SMEM layout for GEMM tiles | B128 swizzle (auto-applied via TMA on Hopper+) |
| Small tiles / narrow data types | B64 or B32 swizzle |
| Hopper+ architecture | Set swizzle mode in the TMA descriptor; no manual coding needed |
| Volta/Ampere | Embed XOR in SMEM address computation |

**Key Takeaways:**

1. 32 banks, 4B each, 128B per cycle
2. Stride that is a multiple of 32 floats will always cause conflicts
3. XOR swizzle is a zero-overhead conflict elimination technique
4. B128 (`Swizzle<3,4,3>`) is the general-purpose go-to choice
5. Hopper+ TMA hardware auto-swizzle

---

## Related Documents

- [Async Copy and Synchronization Primitives](nvidia-ptx-sync-and-async.md)
- Occupancy Tuning
- [NCU Profiling Practice](nsight-profiling-practice.md)
- [GPU Architecture Deep Dive](gpu-architecture-deep-dive.md)
