# Flash Attention GQA D=256 Optimization on MI355X (CDNA4 / gfx950)

Applicability: backend: flydsl; hardware: amd; topic: reference

**Last updated**: 2026-04-28
**Platform**: MI355X (gfx950, CDNA4), 256 CUs, 8 XCDs, 160KB LDS/CU, 8 TB/s HBM, 2.5 PFLOPS dense BF16 Matrix
**Framework**: FlyDSL
**Baseline**: aiter CK `flash_attn_varlen_func` (BlockFmhaPipelineQRKSVS)

---

## Target Workload

Qwen 3.5 397B prefill, TP=2, bf16 causal attention with GQA.

| Shape (B,S,Hq,Hkv,D) | Description |
|---|---|
| (1, 32768, 32, 2, 256) | TP2 per-rank, 32K prefill |
| (1, 16384, 32, 2, 256) | TP2 per-rank, 16K prefill |
| (1, 65536, 16, 2, 256) | TP2 per-rank, 64K prefill, fewer Q-heads |
| (1, 65536, 8, 2, 256) | TP2 per-rank, 64K prefill, 8 Q-heads |

---

## Final Results: 10-15% faster than aiter CK

| Shape | aiter CK (ms) | FlyDSL (ms) | Speedup | TFLOPS |
|---|---|---|---|---|
| (1,32768,32,2,256) | 26.1 | **22.7** | **1.15x** | 388 |
| (1,16384,32,2,256) | 6.6 | **5.9** | **1.12x** | 371 |
| (1,65536,16,2,256) | 51.5 | **45.2** | **1.14x** | 390 |
| (1,65536,8,2,256) | 26.2 | **23.8** | **1.10x** | 369 |

Correctness: cos_sim >= 0.999993 vs aiter, max_abs_diff <= 0.0312.

---

## Kernel Architecture

Based on the MI308X reference flash attention kernel (`flash_attn_func_mi308x.py`) with three key additions for D=256 on CDNA4.

### Base Design
- **MFMA**: `v_mfma_f32_32x32x16_bf16` (CDNA4 K=16)
- **Tile**: BLOCK_M=128, BLOCK_N=64, BLOCK_N_OUT=128 (N128 path, 2 sub-tiles)
- **Waves**: 4 per WG (256 threads), occupancy 1
- **LDS**: K double-buffer (64KB) + V double-buffer (64KB) = 128KB / 160KB
- **DMA**: `buffer_load_dwordx4 ... lds` for K and V (gfx950 16B DMA-to-LDS)
- **GQA**: `kv_head_idx = head_idx // GQA_GROUP_SIZE`, separate `KV_STRIDE_TOKEN`
- **Causal**: `tile_needs_mask` branch skips masking on non-diagonal tiles
- **Online softmax**: `rocdl.exp2` (single `v_exp_f32` instruction)

### Three Key Optimizations

#### 1. K DMA after GEMM1 (+7-11%)

**Problem**: ATT trace showed 4 `s_waitcnt vmcnt(0)` stalls totaling 11.9M cycles. K DMA for the next sub-tile was launched at the top of the sub-tile (after barrier), giving only GEMM2 time (~500 cycles) to hide DMA latency.

**Fix**: Move K DMA launch from the sub-tile top to right after GEMM1. This gives softmax + O-rescale + GEMM2 time (~2000 cycles) for DMA to complete.

```python
# BEFORE: K DMA at sub-tile top
rocdl.s_waitcnt(0)
gpu.barrier()
coop_dma_k(next_sub_tile, next_buf)  # ← only GEMM2 time to hide
# ... K reads, GEMM1, softmax, waitcnt, barrier, GEMM2 ...

# AFTER: K DMA after GEMM1
rocdl.s_waitcnt(0)
gpu.barrier()
# ... K reads ...
# ... GEMM1 ...
coop_dma_k(next_sub_tile, next_buf)  # ← softmax + GEMM2 time to hide
# ... softmax, waitcnt, barrier, GEMM2 ...
```

**Evidence**: `s_waitcnt vmcnt(0)` stall at sub-tile top dropped from 3.3M to ~0.9M cycles.

#### 2. V Double-Buffering (+1%)

**Problem**: V used 1 LDS buffer. V DMA and K LDS reads conflicted when launched simultaneously.

**Fix**: `NUM_PREFETCH_V = 2` — V alternates between 2 LDS buffers (`v_buf_base(kv_sub % 2)`). V DMA for current sub-tile writes to one buffer while K reads from a separate region.

**LDS layout**: K[2 × 64 × 256 × 2 = 64KB] + V[2 × 64 × 256 × 2 = 64KB] = 128KB / 160KB.

#### 3. Tile-Grouped Block Ordering for L3 KV Reuse (+3-5%)

**Problem**: With `head_fast` ordering, consecutive workgroups process different heads for the same Q-tile. Even with GQA (16 Q-heads share 1 KV-head), KV data is loaded redundantly from HBM because different XCDs have separate L2 caches (32MB/XCD).

**Fix**: Group `G=8` consecutive Q-tiles per KV-head in the block ID ordering:

```python
# pid → (chunk, kv_head, q_head_in_group, tile_in_chunk)
tile_in_chunk = logical_pid % TILE_GROUP
rest = logical_pid // TILE_GROUP
q_head_in_group = rest % GQA_GROUP_SIZE
rest2 = rest // GQA_GROUP_SIZE
kv_head = rest2 % NUM_KV_HEADS
chunk = rest2 // NUM_KV_HEADS
```

This way, `GQA_GROUP × G = 16 × 8 = 128` consecutive PIDs share the same KV-head and overlapping Q-tile range. KV tiles loaded by earlier PIDs stay warm in L3 (256MB shared across XCDs).

**Evidence**: TCC_MISS dropped 73% (304M → 81M cache lines = 19.5 → 5.2 GB HBM traffic).

**Zigzag**: Applied on top to balance causal workload across XCDs (reverses odd waves of 8 PIDs).

---

## Profiling Setup

### rocprofv3 with PyTorch

PyTorch bundles `librocprofiler-sdk.so` which conflicts with rocprofv3. Fix:

```bash
TORCH_LIB=$(python -c "import torch,os; print(os.path.dirname(torch.__file__))")/lib
mv $TORCH_LIB/librocprofiler-register.so $TORCH_LIB/librocprofiler-register.so.orig
mv $TORCH_LIB/librocprofiler-sdk.so $TORCH_LIB/librocprofiler-sdk.so.orig
ln -s /opt/rocm/lib/librocprofiler-register.so $TORCH_LIB/librocprofiler-register.so
ln -s /opt/rocm/lib/librocprofiler-sdk.so $TORCH_LIB/librocprofiler-sdk.so
```

### ATT Trace

```bash
# Install trace decoder
cp <path>/librocprof-trace-decoder.so /opt/rocm/lib/

# input_att.yaml
jobs:
  - kernel_iteration_range: "[3]"
    output_directory: tt_test
    output_format: [json, csv]
    truncate_kernels: true
    sys_trace: false
    advanced_thread_trace: true
    att_target_cu: 1
    att_shader_engine_mask: "0x1"
    att_simd_select: "0xf"
    att_buffer_size: "0x6000000"

# Run
env LD_LIBRARY_PATH=/opt/rocm/lib64:/opt/rocm/lib:$LD_LIBRARY_PATH \
    rocprofv3 --att --att-library-path /opt/rocm/lib \
    -i input_att.yaml -- python profile.py
```

---

## Hardware Counter Analysis

### PMC Counters (shape 0, 22.7ms)

| Counter | Total | Per-wave | Interpretation |
|---|---|---|---|
| SQ_WAVES | 32,768 | — | 8192 WGs × 4 waves |
| SQ_INSTS_VALU | 3.80B | 116,046 | Dominant instruction type |
| SQ_INSTS_MFMA | 539M | 16,448 | 128 MFMAs × ~128 avg KV tiles |
| SQ_INSTS_LDS | 808M | 24,672 | K + V LDS reads |
| SQ_INSTS_VMEM | 136M | 4,160 | DMA + Q preload + O store |
| SQ_WAIT_ANY | 6.28B | 191K | 80µs stall per wave |
| TCC_READ | 1.08B | — | 69 GB L2 traffic |
| TCC_MISS | 81M | — | 5.2 GB HBM (L3 hit rate ~92%) |

### Time Breakdown (per wave)

| Phase | Time | % |
|---|---|---|
| MFMA compute | 110 µs | 62% |
| VALU (hidden) | 48 µs | (overlapped) |
| Stalls (DMA + LDS + barrier) | 67 µs | 38% |
| **Total** | **177 µs** | **100%** |

### ISA Profile (2920 instructions)

| Instruction | Count | Notes |
|---|---|---|
| v_mfma_f32_32x32x16_bf16 | 128 | 64 GEMM1 + 64 GEMM2 |
| ds_read_b64_tr_b16 | 128 | V reads (HW transpose) |
| ds_read_b128 | 64 | K reads |
| buffer_load_dwordx4 (DMA) | 40 | K+V DMA to LDS |
| v_exp_f32 | 66 | softmax exp2 |
| v_accvgpr_read/write | 326 | AGPR ↔ VGPR transfers |
| v_pk_mul_f32 | 177 | O rescaling |
| s_nop | 88 | Compiler NOPs |

### Register Budget

| Resource | Count | Limit (occ 2) |
|---|---|---|
| VGPRs (unified) | 392 | 256 |
| AGPRs | 136 | (included in unified) |
| SGPRs | 36 | 106 |
| LDS | 128 KB | 160 KB |
| Private (scratch) | 0 | — |

Occupancy 1 is forced by 392 VGPRs > 256 limit. Main consumers:
- O accumulator: 128 AGPRs (8 D-chunks × v16f32)
- Q preload: 32 VGPRs (16 × v8bf16)
- Softmax/P values: 32 VGPRs
- Index/temp: ~60 VGPRs

---

## Optimization Journey

| Version | Change | Result | Delta |
|---|---|---|---|
| V0 | MI308X ref kernel, N32 path | 26.5ms, 4x slower | baseline |
| V1 | Enable N128 for D=256 | 21.0ms | -5.5ms |
| V2 | DMA auto-enabled (N128 on gfx950) | 6.5ms, matches CK | -14.5ms |
| V3 | waves_per_eu=1 | 25.1ms (shape0), 1.04x | +slight |
| V4 | enable-post-misched=False | 25.0ms, 1.04x | best LLVM config |
| V5 | ISA analysis (392 VGPRs, 128 MFMAs) | insight | — |
| V6 | ATT trace: K DMA stalls 11.9M cycles | insight | — |
| **V7** | **K DMA after GEMM1** | **23.6ms, 1.11x** | **-6.6%** |
| **V8** | **V double-buffer** | **23.4ms, 1.12x** | **-0.8%** |
| V9 | Split-D (occ 2 attempt) | 31ms | REJECTED |
| V10 | Zigzag XCD remap | ~0% (Hq≥NUM_XCDS) | harmless |
| V11 | q_fast for L3 reuse | inconsistent | REJECTED |
| **V13** | **Tile-grouped G=8 + zigzag** | **22.7ms, 1.15x** | **-3%** |
| V14 | QK prefetch depth 3 | 45.2ms (shape2) | +0.3% |

### What Didn't Help
- O=3 monkey-patch (-8%): over-aggressive scheduling
- enable-post-misched=True (-8%)
- ds_bpermute reduction (-5%)
- Q reload from global (-50%): latency too high
- BLOCK_M=256 (-75%): register spill
- Split-D occupancy 2 (-20%): 2x GEMM1 recompute > occ 2 gain
- Scheduling hint sweep (dsrd/mfma 1-4): all within noise
- Hi-before-lo MFMA order: compiler reorders anyway

---

## Remaining Bottleneck

**62% MFMA utilization** with 38% stalls. ATT trace shows the stall is from GEMM1 hi-MFMA waiting for K_hi ds_read_b128:

```asm
ds_read_b128 K_hi[ks]         ; LDS read issued
s_nop 0                       ; compiler NOP (waste)
mfma_lo K_lo[ks]              ; Stall=0 (K_lo ready)
s_waitcnt lgkmcnt(2)
mfma_hi K_hi[ks]              ; Stall=104K cycles! (K_hi not ready)
```

Fixing requires ISA-level manual scheduling — below FlyDSL's abstraction.

---

## Key Architectural Insights (MI355X)

1. **L2 is per-XCD (32MB), L3 is shared (256MB)**: L2 reuse requires same-XCD scheduling. L3 reuse works across XCDs via tile-grouped ordering.
2. **DMA-to-LDS bypasses L1**: `buffer_load_dwordx4 ... lds` goes TCP → L2/L3 → LDS. Each DMA is one TCP_TCC_READ_REQ.
3. **Unified VGPR/AGPR on gfx950**: Total budget = arch + accum. With D=256, O accumulator (128 AGPRs) + arch (196) = 332 > 256 → occupancy 1.
4. **XCD round-robin dispatch**: Hardware assigns consecutive PIDs to XCDs mod 8. Can't pin blocks to specific XCDs without persistent kernels.
5. **Zigzag helps when `batch × Hq < NUM_XCDS`**: For our shapes (Hq ≥ 8 = NUM_XCDS), natural ordering already balances.

---

## Files

- **Kernel**: `reference-kernels/amd/cdna4/flydsl/FlyDSL/flash_attn_func_gqa_d256.py`
- **Pitfalls**: `docs/amd/cdna4/pitfalls/flydsl/flash-attn-d256-pitfalls.md`
- **Base kernel**: `reference-kernels/amd/cdna3/flydsl/FlyDSL/flash_attn_func_mi308x.py`
