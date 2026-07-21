# CDNA4 FP8 GEMM Kernel Optimization in Practice

## CDNA4 vs CDNA3 Hardware Comparison

| Feature | CDNA4 (MI355X, gfx950) | CDNA3 (MI300X, gfx942) |
|------|----------------------|----------------------|
| LDS Capacity | 160 KB/CU | 64 KB/CU |
| LDS Bank Count | 64 | 32 |
| LDS Read Bandwidth | 256 bytes/clock | 128 bytes/clock |
| `GLOBAL_LOAD_LDS` per lane | Up to 128 bits | Up to 32 bits |
| Wavefront Size | 64 (wave64) | 64 (wave64) |
| FP4/FP6 Dense MFMA | Supported | Not supported |
| Block-scaled MFMA | Supported | Not supported |
| FP16/BF16 New Shapes | 16x16x32, 32x32x16 | Up to 16x16x16, 32x32x8 |

---

## CDNA4 MFMA Instructions

### Core MFMA: V_MFMA_F32_16X16X128_F8F6F4

FP8 inputs (E4M3FN), FP32 accumulation, 16x16x128 tile. **A single instruction executes 65,536 FLOPs** (compared to 128 FLOPs for FMA, **512× density**).

### MFMA Compute Efficiency Measurements

| Kernel Type | MFMA_INSTS/SQ_CYCLES | FMA_INSTS/SQ_CYCLES | MFMA FLOPs/cycle | FMA FLOPs/cycle | Total FLOPs/cycle |
|------------|---------------------|---------------------|-----------------|----------------|---------------|
| LDS-tiled SIMT | 0 | 0.466 | 0 | 59.6 | 59.6 |
| MFMA matrix-core | 0.009 | 0.001 | 617.3 | 0.2 | 617.4 |

The MFMA kernel achieves **~10.35×** higher FLOPs/cycle.

### Block-scaled MFMA (New in CDNA4)

```
V_MFMA_SCALE_F32_16X16X128_F8F6F4    // 16x16x128 block-scaled
V_MFMA_SCALE_F32_32X32X64_F8F6F4     // 32x32x64 block-scaled
```

Supports FP8, FP6, FP4 inputs with block scaling factors for MX-format inference.

---

## FP8 GEMM Implementation Details

### Problem Definition

C = A * B^T, where A is M×K (row-major), B is N×K (row-major), and C is M×N.

- Input: FP8 (E4M3FN)
- Accumulation: FP32
- Output: BF16

### Wave-Lane Mapping (16×16×128)

Each wave has 64 lanes, with each lane handling fixed positions in the matrix:

```c
const int lane = lane_in_wave;          // [0, 63]
const int row_in_tile = lane & 15; // [0, 15] row
const int row_group = lane >> 4; // [0, 3] K
const int k_chunk0 = row_group * 16;    // 0, 16, 32, 48
const int k_chunk1 = k_chunk0 + 64;     // 64, 80, 96, 112
```

Per lane per K tile (128 elements):
- **A**: 32 FP8 elements (two 16-element chunks)
- **B**: 32 FP8 elements (two 16-element chunks)
- **Output**: 4 FP32 accumulators

### Vectorized Load

Each load fetches 16 FP8 values (16 bytes), using `uint4`:

```c
using fp8x16_t = __attribute__((vector_size(16))) fp8_t;

static inline __device__ fp8x16_t load_fp8x16_u4(const fp8_t* p) {
    const uint4 v = *reinterpret_cast<const uint4*>(p);
    return *reinterpret_cast<const fp8x16_t*>(&v);
}
```

### Direct Global-to-LDS Load

CDNA4 supports bypassing registers and loading directly from Global to LDS (`GLOBAL_LOAD_LDS`):

```c
extern "C" __device__ void llvm_amdgcn_raw_buffer_load_lds(
    i32x4 rsrc, as3_uint32_ptr lds_ptr, int size,
    int voffset, int soffset, int offset, int aux)
    __asm("llvm.amdgcn.raw.buffer.load.lds");

struct buffer_resource {
    uint64_t ptr;
    uint32_t range;
    uint32_t config;  // 0x110000
};
```

Data path: **Global → LDS → Registers → MFMA**, skipping the register intermediate step.

---

## LDS Swizzle Optimization

### Bank Conflict Problem

The `ds_read_b128` instruction executes in **four phases**, with each phase requiring conflict-free bank access. Thread grouping:

```
stage 1: T0-T3, T12-T15, T20-T23, T24-T27
stage 2: T32-T35, T44-T47, T52-T55, T56-T59
stage 3: T4-T7, T8-T11, T16-T19, T28-T31
stage 4: T36-T39, T40-T43, T48-T51, T60-T63
```

### Swizzle Function

Apply row-based XOR remapping to the 16×128 tile (16-byte columns):

```c
int swizzle_col(int row, int col) {
    const int pair = (row >> 1) & 7;
    const int perm = pair ^ (((pair >> 1) ^ (pair >> 2)) & 1);
    const int mask = perm << 4;
    return col ^ mask;
}
```XOR is self-inverse: the same function is used for both forward and inverse transformations.

---

## Double Buffering

Two LDS slots alternate (ping-pong): one is being used for computation while the other is being filled with the next K tile.

```c
LdsTile A_lds[2], B_lds[2];
int cur = 0, nxt = 1;

prefetch_tile_to_lds(A_lds[cur], B_lds[cur], 0);
wait_for_global_loads();
block_sync();

for (int t = 0; t < num_k_tiles; ++t) {
    if (t + 1 < num_k_tiles)
        prefetch_tile_to_lds_async(A_lds[nxt], B_lds[nxt], t + 1);

    fragments_a = read_fragments_from_lds(A_lds[cur]);
    fragments_b = read_fragments_from_lds(B_lds[cur]);
    acc = mfma(acc, fragments_a, fragments_b);

    if (t + 1 < num_k_tiles) {
        wait_for_global_loads();
        block_sync();
        cur ^= 1; nxt ^= 1;
    }
}
```

Performance impact: single-buffer swizzled (497 TFLOPS) → double-buffer swizzled (1166 TFLOPS) = **2.34x improvement**.

---

## Multi-Wave Configuration

Each thread block runs multiple waves that share LDS tiles:

| Configuration | Output Tile | Threads | Waves | TFLOPS (M=N=K=4096) |
|------|----------|--------|---------|---------------------|
| 128x128_t512 | 128x128 | 512 | 8 | 1828.74 |
| **256x256_t512** | **256x256** | **512** | **8** | **2288.16** |
| 256x256_t1024 | 256x256 | 1024 | 16 | 2228.01 |

**256x256_t512 (8 waves)** delivers the best performance. The wave layout is a 2x4 arrangement, with 8 waves cooperatively filling a 256x128 LDS tile. B's LDS-to-register reads are grouped by fixed wave pairs: (W0,W4), (W1,W5), (W2,W6), (W3,W7).

---

## 8-Wave Ping-Pong Scheduling

Based on the HipKittens paper design. Each block has 8 waves, with 2 resident waves per SIMD. Waves within the same SIMD alternate between executing memory and MFMA instructions.

### Key Compiler Intrinsics

```c
__builtin_amdgcn_s_barrier // wave, increaseexecute
__builtin_amdgcn_s_setprio(x) // compilation wave (0-3)
__builtin_amdgcn_sched_barrier(x) // barrier
// sched_barrier(0) =
```

### Wave Allocation

```c
int waveid = threadIdx.x / 64;  // 0...7
int wave_m = waveid / 4;        // 0...1
int wave_n = waveid % 4;        // 0...3

// SIMD mapping:
// Waves 0,4 → SIMD 0
// Waves 1,5 → SIMD 1
// Waves 2,6 → SIMD 2
// Waves 3,7 → SIMD 3
```

### LDS Layout

```c
__shared__ fp8 A_lds[2][2][128*128]; // [block][data]
__shared__ fp8 B_lds[2][2][128*128];
```

Per-wave register allocation:
- `a_reg[4][32]` — A fragments
- `b_reg0[2][32]`, `b_reg1[2][32]` — B fragments
- `c_reg0..c_reg3[8][4]` — FP32 accumulators

### Hot Loop Structure

Using `#pragma unroll 2` to reduce register pressure and eliminate spilling. Each iteration:

1. Issue `ds_read_b128` read B_lds[tic][0] → b_reg0
2. Issue `ds_read_b128` read A_lds[tic][0] → a_reg
3. Issue buffer loads to write A_lds[toc][1]
4. Wait for LDS reads to complete, barrier, then `setprio(1)` wrap MFMA computation
5. Issue B_lds[tic][1] read → b_reg1
6. buffer loads + barrier + MFMA (c_reg1)
7. A_lds[tic][1] read + B buffer loads + barrier + MFMA (c_reg2)
8. B_lds[tic][1] buffer loads + `s_waitcnt vmcnt(6)` + barrier + MFMA (c_reg3)
9. Swap tic/toc (`tic^=1, toc^=1`)

The last two K iterations are manually unrolled (epilogue) because memory instructions are still in flight.

### Prologue Wait Count

```c
// 8 buffer loads :
s_waitcnt vmcnt(4); // wait 4 pending
// 6 buffer loads :
s_waitcnt vmcnt(6); // wait 6 pending
```

---

## Complete Performance Evolution

| Optimization Stage | Average Time (ms) | TFLOPS | M=N=K |
|---------|-------------|--------|-------|
| Naive baseline | 119.60 | 1.15 | 4096 |
| LDS tiling | 28.64 | 4.80 | 4096 |
| Matrix-core baseline | 4.57 | 30.05 | 4096 |
| + Vectorized load | 0.408 | 336.88 | 4096 |
| + Direct Global-to-LDS | 0.271 | 506.70 | 4096 |
| + LDS swizzle | 0.276 | 497.43 | 4096 |
| + Double buffering | 0.118 | 1166.41 | 4096 |
| Multi-Wave 128x128, 512t | 0.075 | 1828.74 | 4096 |
| Multi-Wave 256x256, 512t | 0.060 | 2288.16 | 4096 |
| **8-wave ping-pong** | **0.051** | **2680.33** | **4096** |
| hipBLASLt reference | 0.050 | 2750.42 | 4096 |
| **8-wave ping-pong** | **0.343** | **3204.15** | **8192** |
| hipBLASLt reference | 0.351 | 3130.21 | 8192 |**Key Conclusions:**
- 8-wave ping-pong achieves **97.5%** of hipBLASLt at M=N=K=4096
- At M=N=K=8192, it **surpasses** hipBLASLt by 2.4%
- From naive to final: **~2330×** total speedup
- All written in HIP/C++, no hand-written assembly

### System Configuration

- GPU: AMD Instinct MI355X (gfx950)
- ROCm 7.1.0
- Uses compiler built-in functions and inline asm for scheduling control
