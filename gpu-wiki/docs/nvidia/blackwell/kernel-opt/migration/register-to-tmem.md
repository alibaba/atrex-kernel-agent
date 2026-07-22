# Register Accumulators to TMEM

## Overview

On Hopper (SM90), `wgmma` stores MMA accumulators in **registers**. A single m64xn256xk16 BF16 wgmma requires 128 FP32 registers per thread in the warpgroup for the accumulator alone. This extreme register pressure limits tile sizes, reduces occupancy, and forces complex register management.

On Blackwell (SM100), `tcgen05.mma` stores accumulators in **Tensor Memory (TMEM)** -- a dedicated 256KB per-SM memory. This eliminates accumulator register pressure entirely, freeing registers for data movement, epilogue computation, and enabling larger tile sizes.

## The Register Pressure Problem on Hopper

### Register Budget Analysis

Each SM90 SM has **65,536 registers** (256KB). A typical GEMM warpgroup needs:

```
Hopper wgmma m64xn256xk16 BF16 register budget (per thread):

  Accumulator:        128 FP32 registers (m64 x n256 / 128 threads)
  A operand:           16 registers (via ldmatrix)
  B descriptor:         2 registers
  Loop variables:       8 registers
  SMEM pointers:        4 registers
  TMA state:            6 registers
  Misc (indexing):     10 registers
  ─────────────────────────────
  Total per thread:   ~174 registers

  Per warpgroup (128 threads): 174 * 128 = 22,272 registers
  Per SM (65,536 regs): max 2 warpgroups = 2 CTAs at this tile size

  Occupancy: 2 CTAs per SM (limited by registers)
```

If the tile is larger (e.g., m64xn256xk32 with double-buffered accumulators), register pressure becomes even worse:

```
Double-buffered accumulator: 256 registers per thread
Total per thread:            ~300 registers
Per warpgroup:              38,400 registers
Per SM:                     max 1 warpgroup -> 1 CTA per SM

Occupancy: 1 CTA per SM (severe underutilization)
```

### Register Spilling on Hopper

When register pressure exceeds the budget, the compiler spills registers to local memory (L1 cache backed by GMEM). This is catastrophic for performance:

```
Register access:  ~4 cycles
L1 local access:  ~30 cycles
L2 spill access:  ~200 cycles

A single spilled accumulator register accessed every MMA iteration
can cost 200 cycles * K_tiles additional latency.
```

## Blackwell TMEM Solution

### TMEM Budget Analysis

```
Blackwell tcgen05 m128xn256xk16 BF16 register budget (per thread):

  Accumulator:          0 registers (stored in TMEM!)
  SMEM descriptors:     4 registers
  Loop variables:       8 registers
  TMA state:            6 registers
  TMEM address:         1 register
  Misc (indexing):     10 registers
  ─────────────────────────────
  Total per thread:   ~29 registers

  Per CTA (256 threads): 29 * 256 = 7,424 registers
  Per SM (65,536 regs): max 8 CTAs (register-wise)
  Actual limit: TMEM capacity (512 columns)

  TMEM usage per CTA: 256 columns for 128x256 tile
  Max CTAs per SM: 512 / 256 = 2 CTAs (TMEM-limited, not register-limited)
```

The key insight: **registers are no longer the bottleneck**. TMEM capacity becomes the binding constraint for occupancy, and the freed registers enable complex epilogues without spilling.

## Migration: Accumulator Lifecycle

### Hopper: Register-Based Lifecycle

```cuda
// HOPPER: Accumulator lives and dies in registers
__global__ void hopper_kernel(/* ... */) {
    // 1. Declare accumulator in registers
    float acc[4][32];  // 128 registers per thread!

    // 2. Zero-initialize
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 32; ++j)
            acc[i][j] = 0.0f;

    // 3. Mainloop: wgmma accumulates into registers
    for (int k = 0; k < K_tiles; ++k) {
        // Load A via ldmatrix
        uint32_t a_frag[4];
        ldmatrix(a_frag, smem_a + k * TILE_K);

        // wgmma: reads A from registers, B from SMEM descriptor
        // Accumulates into acc[] registers
        wgmma_m64n256k16(acc, a_frag, smem_b_desc);
        wgmma_commit();
        wgmma_wait();
    }

    // 4. Epilogue: acc is directly in registers -- fast access
    // But: if epilogue needs temporary storage, registers are exhausted
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            float val = acc[i][j];
            val += bias[col_base + j];     // bias add
            val = fmaxf(val, 0.0f);         // ReLU
            // Need to convert and store -- no registers left for temp!
            C[row * N + col_base + j] = __float2half(val);
        }
    }

    // 5. Accumulator freed implicitly when CTA exits
}
```

### Blackwell: TMEM-Based Lifecycle

```cuda
// BLACKWELL: Accumulator lives in TMEM
__global__ void blackwell_kernel(/* ... */) {
    // 1. Allocate TMEM (explicit, must be done once)
    uint32_t tmem_acc;
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.b32 %0, %1;"
            : "=r"(tmem_acc) : "r"(256)
        );
    }
    tmem_acc = __shfl_sync(0xFFFFFFFF, tmem_acc, 0);

    // 2. Zero-initialize TMEM
    for (int c = 0; c < 256; c += 4) {
        asm volatile(
            "tcgen05.st.sync.aligned.32x1b.x4.b32 [%0], {%1,%2,%3,%4};"
            : : "r"(tmem_acc + c),
                "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f)
        );
    }

    // 3. Mainloop: tcgen05 accumulates into TMEM
    for (int k = 0; k < K_tiles; ++k) {
        // NO ldmatrix -- tcgen05 reads directly from SMEM
        if (threadIdx.x == 0) {
            asm volatile(
                "tcgen05.mma.cta_group::1.kind::f16 "
                "[%0], %1, %2, %3, 1;"
                : : "r"(tmem_acc), "l"(desc_a), "l"(desc_b), "r"(0)
            );
        }
        // NO commit/wait -- fully async
    }

    // 4. Fence before reading
    asm volatile("tcgen05.mma.fence::before_thread_sync;");
    __syncthreads();

    // 5. Epilogue: read from TMEM to registers (in small batches)
    // PLENTY of registers available for epilogue temporaries!
    for (int c = 0; c < 256; c += 4) {
        float4 vals;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x1b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=f"(vals.x), "=f"(vals.y), "=f"(vals.z), "=f"(vals.w)
            : "r"(tmem_acc + c)
        );

        // Apply epilogue ops with register headroom
        vals.x += bias[col_base + c];
        vals.y += bias[col_base + c + 1];
        vals.z += bias[col_base + c + 2];
        vals.w += bias[col_base + c + 3];

        vals.x = fmaxf(vals.x, 0.0f);  // ReLU
        vals.y = fmaxf(vals.y, 0.0f);
        vals.z = fmaxf(vals.z, 0.0f);
        vals.w = fmaxf(vals.w, 0.0f);

        // Vectorized store -- registers available for conversion
        half2 h01 = __floats2half2_rn(vals.x, vals.y);
        half2 h23 = __floats2half2_rn(vals.z, vals.w);
        *reinterpret_cast<half2*>(&C[row * N + col_base + c]) = h01;
        *reinterpret_cast<half2*>(&C[row * N + col_base + c + 2]) = h23;
    }

    // 6. Deallocate TMEM (explicit, MUST be done in persistent kernels)
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            : : "r"(tmem_acc), "r"(256)
        );
    }
}
```

## Impact on Kernel Design

### Tile Size Freedom

With accumulators in TMEM, the tile size is no longer constrained by register count:

```cuda
// Hopper: large tiles cause spilling
// m64xn256 uses 128 acc registers/thread -> barely fits
// m128xn256 would need 256 acc registers/thread -> spills guaranteed

// Blackwell: tile size limited by TMEM columns (512) and SMEM, not registers
// m128xn256: 256 TMEM cols, 0 acc registers -> fits easily
// m128xn512: 512 TMEM cols, 0 acc registers -> uses full TMEM budget
// m256xn256 (2-SM): 256 TMEM cols/SM, 0 acc registers -> fits with cooperative
```

### Epilogue Complexity

On Hopper, complex epilogues (bias + activation + quantization + scaling) often spill because registers are already exhausted by the accumulator. On Blackwell, the epilogue has the full register file available:

```cuda
// Blackwell epilogue: full register file available
__device__ void rich_epilogue(uint32_t tmem_acc, float* output,
                               const float* bias, const float* scale,
                               int M, int N) {
    // Read TMEM in chunks
    for (int c = 0; c < 256; c += 8) {
        // Load 8 values from TMEM (8 registers -- trivial)
        float v[8];
        tmem_load_f32x4(tmem_acc + c, &v[0]);
        tmem_load_f32x4(tmem_acc + c + 4, &v[4]);

        // Apply bias (8 registers for bias values)
        float b[8];
        load_global_f32x4(&b[0], bias + c);
        load_global_f32x4(&b[4], bias + c + 4);

        #pragma unroll
        for (int i = 0; i < 8; ++i) v[i] += b[i];

        // Apply GELU activation (needs temp registers for polynomial)
        #pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = gelu(v[i]);

        // Apply per-channel scale
        float s[8];
        load_global_f32x4(&s[0], scale + c);
        load_global_f32x4(&s[4], scale + c + 4);

        #pragma unroll
        for (int i = 0; i < 8; ++i) v[i] *= s[i];

        // Quantize to FP8 for next layer
        uint8_t q[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) q[i] = float_to_e4m3(v[i]);

        // Store quantized output
        store_global_u8x8(output + c, q);

        // Total temp registers used: ~24 -- NO SPILLING
        // On Hopper this epilogue would need 24 + 128 (acc) = 152 per thread
    }
}
```

### Overlapped Epilogue and MMA

The most powerful pattern enabled by TMEM: overlapping the current tile's epilogue with the next tile's MMA computation. This is impossible with register accumulators because the accumulator registers are still in use.

```cuda
__global__ void overlapped_gemm(/* ... */) {
    // Two TMEM accumulator buffers
    uint32_t tmem_a = tmem_alloc(256);
    uint32_t tmem_b = tmem_alloc(256);
    uint32_t* tmem_cur = &tmem_a;
    uint32_t* tmem_nxt = &tmem_b;

    // Compute first tile into tmem_a
    compute_tile(*tmem_cur, tile_0);
    fence_and_sync();

    for (int t = 1; t < num_tiles; ++t) {
        // START: next tile MMA into tmem_nxt (background, async)
        zero_tmem(*tmem_nxt);
        start_mma(*tmem_nxt, tile_t);

        // SIMULTANEOUSLY: epilogue of current tile from tmem_cur
        // MMA is running in the background while we read the old accumulator!
        epilogue_from_tmem(*tmem_cur, output_tile[t-1]);

        // Wait for MMA to complete
        fence_and_sync();

        // Swap buffers
        uint32_t* tmp = tmem_cur;
        tmem_cur = tmem_nxt;
        tmem_nxt = tmp;
    }

    // Final epilogue
    epilogue_from_tmem(*tmem_cur, output_tile[num_tiles-1]);

    tmem_dealloc(tmem_a, 256);
    tmem_dealloc(tmem_b, 256);
}
```

## FlashAttention Case Study

FlashAttention is a prime example where the register-to-TMEM migration unlocks major gains.

### Hopper FlashAttention Register Pressure

```
FlashAttention on Hopper needs registers for:
  - QK^T accumulator:   64 registers (m64xn64 attention scores)
  - PV accumulator:     128 registers (m64xn256 output)
  - Softmax state:      4 registers (rowmax, rowsum)
  - Q fragment:         16 registers
  - K fragment:         16 registers
  - V fragment:         16 registers
  - Loop state:         10 registers
  ──────────────────────────────
  Total:               ~254 registers per thread

  Available: 256 registers/thread at 1 CTA/SM occupancy
  Margin: 2 registers (!!)

  Result: compiler must spill, or tile sizes must shrink
```

### Blackwell FlashAttention with TMEM

```
FlashAttention on Blackwell:
  - QK^T accumulator:   0 registers (TMEM buffer 1, 64 cols)
  - PV accumulator:     0 registers (TMEM buffer 2, 256 cols)
  - Softmax state:      4 registers (rowmax, rowsum)
  - SMEM descriptors:   6 registers
  - Loop state:         10 registers
  ──────────────────────────────
  Total:               ~20 registers per thread

  Available: 256 registers/thread
  Margin: 236 registers -- massive headroom

  Result: Can use for ping-pong scheduling, software exp emulation,
          larger tiles, more pipeline stages
```

FlashAttention-4 exploits this headroom for software-emulated exponentials (distributing `2^x` across FMA units instead of waiting for the SFU), achieving **1605 TFLOPS on B200** at 71% utilization.

## Common Mistakes During Migration

### Mistake 1: Treating TMEM Like Registers

```cuda
// WRONG: Trying to use TMEM values directly in expressions
float result = tmem_acc[row][col] + bias;  // TMEM is not directly addressable!

// CORRECT: Load from TMEM to register, then compute
float val = tmem_load_f32(tmem_col);
float result = val + bias;
```

### Mistake 2: Forgetting TMEM Latency

```cuda
// WRONG: Fine-grained TMEM access in a tight loop (high latency)
for (int i = 0; i < 256; ++i) {
    float v = tmem_load_f32(tmem_acc + i);  // 420 cycle latency each!
    output[i] = v;
}

// CORRECT: Vectorized loads to amortize latency
for (int i = 0; i < 256; i += 4) {
    float4 v = tmem_load_f32x4(tmem_acc + i);  // Single 420-cycle access
    output[i]   = v.x;
    output[i+1] = v.y;
    output[i+2] = v.z;
    output[i+3] = v.w;
}
```

### Mistake 3: Not Re-tuning Tile Sizes

```cuda
// Hopper: tile size was m64xn128 to avoid register spilling
// Migrating to Blackwell: keep same tile = leaving performance on the table

// Blackwell should use at minimum m128xn256 (1-SM) or m256xn256 (2-SM)
// The freed registers allow larger tiles without any spilling risk
```

### Mistake 4: TMEM Leak in Persistent Kernels

```cuda
// WRONG: Forgetting to dealloc in a persistent kernel
__global__ void persistent_kernel(/* ... */) {
    uint32_t tmem = tmem_alloc(256);  // Allocated once

    while (has_work()) {
        compute_tile(tmem);
        // BUG: If the kernel exits early (e.g., error path),
        // TMEM is leaked. Next CTA on this SM cannot allocate.
    }
    // Missing: tmem_dealloc(tmem, 256);
}

// CORRECT: Always dealloc, even on error paths
__global__ void persistent_kernel(/* ... */) {
    uint32_t tmem = tmem_alloc(256);

    while (has_work()) {
        compute_tile(tmem);
    }

    tmem_dealloc(tmem, 256);  // Always reached
}
```

## Summary: What Changes, What Stays

| Aspect | Register (Hopper) | TMEM (Blackwell) |
|---|---|---|
| Accumulator declaration | `float acc[N]` in registers | `tmem_alloc(cols)` |
| Zero-initialization | Loop over register array | `tcgen05.st` zero pattern |
| MMA accumulation | Implicit (wgmma writes regs) | Implicit (tcgen05 writes TMEM) |
| Reading results | Direct register access | `tcgen05.ld` to register |
| Cleanup | Implicit (CTA exit) | Explicit `tcgen05.dealloc` |
| Register pressure | ~128-256 regs for accumulator | ~0 regs for accumulator |
| Epilogue headroom | Minimal (spill risk) | Ample (full register file) |
| Double-buffering | Doubles register pressure (2x acc) | Uses 2 TMEM regions (no reg impact) |
| Max practical tile | m64xn256 (limited by registers) | m128xn256 or m256xn256 (limited by TMEM cols) |
