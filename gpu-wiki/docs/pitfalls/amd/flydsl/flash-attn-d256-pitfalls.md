# Flash Attention head_dim=256 Pitfalls (CDNA4 / gfx950)

Pitfalls encountered during flash attention optimization for head_dim=256 on MI355X. Supplements the generic `flash-attn-pitfalls.md`.

---

## 1. N128 path not auto-selected for D=256

**Symptom**: Kernel 4x slower than CK despite using the same tile shape.

**Cause**: The reference kernel's `PATH_TAG` auto-selection only triggers `N128` for `head_dim == 128`:
```python
elif dtype_str in ("f16", "bf16") and causal and head_dim == 128:
    PATH_TAG = "N128"
```

**Fix**: Extend condition to `head_dim in (128, 256)`. With D=256, K tile = 64×256×2 = 32KB, double-buffered = 64KB. V tile = 32KB, double-buffered = 64KB. Total 128KB < 160KB LDS. The N128 path (2 sub-tiles of 64, DMA-to-LDS) fits and is essential for performance.

**Impact**: 4x speedup (26ms → 6.5ms on S=16K).

---

## 2. Occupancy 1 is unavoidable with D=256

**Symptom**: `waves_per_eu` target 2 or 3 produces "failed to meet occupancy target" warning.

**Cause**: O accumulator = 8 D-chunks × v16f32 = 128 AGPRs. With arch VGPRs (~196) + 128 AGPRs = 332 total, exceeding 256 limit for occupancy 2 on gfx950's unified VGPR/AGPR file.

**Workaround**: Set `waves_per_eu=1` to avoid confusing the register allocator. The kernel works fine at occupancy 1 — the bottleneck is DMA latency, not occupancy.

**Split-D attempt**: Processing D in 2 passes of 128 can achieve occupancy 2 (247 VGPRs with N32 path), but the 2x GEMM1 recompute overhead makes it 20% slower overall.

---

## 3. V cooperative load uses HEAD_DIM, not V_HEAD_DIM

**Symptom**: Garbage output when `v_head_dim != head_dim` (split-D mode).

**Cause**: Several V loading functions use `HEAD_DIM` for stride and thread decomposition:
- `_v_store_row_major` uses `load_col_base` (computed from `HEAD_DIM`)
- `coop_load_v` / `coop_load_v_global` use `THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH`
- `coop_store_v_lds` uses `NUM_BATCHES_KV` and `ROWS_PER_BATCH_LOAD` from K constants

**Fix**: Create separate V loading constants (`V_THREADS_PER_ROW`, `V_ROWS_PER_BATCH`, `v_load_col_base`, `v_lds_col_base`) derived from `V_HEAD_DIM`. Also add `D_OFFSET` to V global addresses and O store addresses.

---

## 4. DMA-to-LDS V address needs D_OFFSET for split-D

**Symptom**: V DMA loads wrong columns when `d_offset > 0`.

**Cause**: `coop_dma_v` computes global byte address as `row * KV_STRIDE_TOKEN * 2 + kv_head * HEAD_DIM * 2 + col_byte`. With split-D, need to add `D_OFFSET * 2` to the byte offset.

**Fix**:
```python
global_byte = (global_row * arith.index(KV_STRIDE_TOKEN * 2)
               + kv_head_idx * arith.index(HEAD_DIM * 2)
               + arith.index(D_OFFSET * 2)  # ← added
               + col_byte)
```

---

## 5. K DMA placement affects performance by 7-11%

**Symptom**: K DMA double-buffer shows 3.3M cycle stalls per invocation despite being prefetched.

**Cause**: Default placement launches K DMA for the next sub-tile at the top of the next sub-tile (after barrier). This gives only GEMM2 time (~500 cycles, 32 MFMAs) to hide DMA latency, insufficient for D=256 where K DMA = 8 instructions × HBM latency.

**Fix**: Launch K DMA after GEMM1 of the current sub-tile, before softmax. This gives softmax + O-rescale + GEMM2 (~2000 cycles) to hide DMA.

**Warning**: Moving K DMA even earlier (before GEMM1, right after barrier) causes DMA-LDS write conflicts with K LDS reads and regresses by 2%.

---

## 6. V DMA early launch conflicts with K LDS reads

**Symptom**: Launching V DMA right after barrier (alongside K DMA) regresses by 2%.

**Cause**: V DMA writes to LDS V region while K is being read from LDS K region simultaneously. Even though they're different LDS addresses, the concurrent DMA writes and LDS reads contend for LDS bank bandwidth.

**Fix**: Issue V DMA after K prefetch reads are initiated (existing position), or use V double-buffering so V DMA writes to a separate buffer.

---

## 7. Tile-grouped ordering needs correct total_blocks calculation

**Symptom**: `arith.index_cast(T.index, x)` fails when `x` is already index type.

**Cause**: `num_chunks * NUM_KV_HEADS * chunk_heads - 1` produces an index value. `arith.index_cast(T.index, ...)` rejects index→index cast.

**Fix**: Use the expression directly without `arith.index_cast`:
```python
_max_pid = num_chunks * NUM_KV_HEADS * chunk_heads - 1
logical_pid = arith.MinUIOp(logical_pid, _max_pid).result
```

---

## 8. q_fast ordering causes causal load imbalance

**Symptom**: q_fast (q_tile fastest, head slowest) regresses 18% on shapes with many heads (Hq=32).

**Cause**: With causal attention, the last Q-tiles are heavy (many KV tiles). q_fast ordering puts all heavy tiles for head 0 at the end of head 0's block range. Some XCDs get all-heavy work while others idle.

**Fix**: Use tile-grouped ordering with G=8 instead of pure q_fast. This provides L3 KV reuse (88% hit rate) while keeping causal imbalance < 3%. Apply zigzag on top to balance odd/even wave assignments.

---

## 9. rocprofv3 crashes with PyTorch's bundled rocprofiler

**Symptom**: `SIGABRT` with "Configuration request occurred outside of valid rocprofiler configuration period".

**Cause**: PyTorch (via `torch/lib/librocprofiler-sdk.so`) loads an older rocprofiler that conflicts with rocprofv3's version.

**Fix**: Replace PyTorch's bundled libraries with symlinks to system ROCm:
```bash
TORCH_LIB=$(python -c "import torch,os; print(os.path.dirname(torch.__file__))")/lib
mv $TORCH_LIB/librocprofiler-register.so $TORCH_LIB/librocprofiler-register.so.orig
mv $TORCH_LIB/librocprofiler-sdk.so $TORCH_LIB/librocprofiler-sdk.so.orig
ln -s /opt/rocm/lib/librocprofiler-register.so $TORCH_LIB/librocprofiler-register.so
ln -s /opt/rocm/lib/librocprofiler-sdk.so $TORCH_LIB/librocprofiler-sdk.so
```

---

## 10. GEMM1 hi-MFMA stalls from compiler scheduling

**Symptom**: ATT trace shows hi-MFMA stalling 104K cycles per K-step while lo-MFMA has 0 stall.

**Cause**: Compiler schedules K_hi ds_read immediately before K_lo ds_read, giving K_hi only ~16 cycles of LDS pipeline lead time (one MFMA). K_lo gets more because it was read earlier.

**Status**: Cannot fix via `sched_dsrd` / `sched_mfma` hints — the compiler ignores them for this pattern. Needs ISA-level manual scheduling or custom LLVM backend changes. This is the **remaining 38% stall** in the optimized kernel.
