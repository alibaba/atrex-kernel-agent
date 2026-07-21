# sm_120 ncu trap: l1tex__t_sector_hit_rate.pct includes ld.shared hits

## trap

On sm_120 cute 4.4.2 + ncu 2025.2.1, `l1tex__t_sector_hit_rate.pct`
(displayed as "L1/TEX Hit Rate" in Speed-of-Light + Memory Workload sections)
**includes shared-memory ld.shared hits**, not just global memory hits.
This is because L1 and texture units share the same physical SRAM unit on
sm_120 (same as Ampere/Ada).

## symptom

TMA-based kernel reads only ld.global ~ O(grid_dim) for scalar / control,
but ncu reports L1/TEX Hit Rate 30-50% even when global loads are 100%
TMA-bulk (bypassing L1 by design). Looks like TMA "isn't bypassing L1",
which contradicts both flashinfer NVFP4QuantizeTMAKernel sm_120
validation and persistent_gemm sm_120 evidence.

## reality

TMA does bypass L1. The hit rate counter is shared-memory-polluted.

## why

Verified on sm_120 cute 4.4.2 V2-TMA kernel (M=6144, K=4096, kernel_opt_attn_fp4_fusion):

| Metric                                                     | Value  |
|---|---|
| `sm__inst_executed_pipe_tma.sum`                           |   6144 |
| `smsp__inst_executed_op_global_ld.sum`                     |    990 |
| `smsp__inst_executed_op_shared_ld.sum`                     | 196608 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`           |    990 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum`|    880 (88.9 %) |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum`|   110 |
| `l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum`            |      0 |
| `sm__sass_inst_executed_op_ldsm.sum`                       |      0 |
| `l1tex__t_sector_hit_rate.pct`                             |  49.93 % |

The 49.93 % "L1/TEX Hit Rate" reduces to: 880 global hits + most of the
196608 shared loads (also accounted as L1/TEX) divided by total L1/TEX
sectors. The global-only hit rate is 88.9 % but on only 990 ops — that's
the mGlobalScale scalar broadcast across 110 SMs, expected.

## lesson

On sm_120, to confirm TMA actually bypasses L1, ALWAYS check:

1. `sm__inst_executed_pipe_tma.sum` — should equal expected TMA tile count
   (e.g. row_tiles × col_chunks × num_buffers)
2. `smsp__inst_executed_op_global_ld.sum` — should be ~0 for tile data
   (only scalar broadcasts and SF-padding scalar writes remain)
3. `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` — should match (2)

NEVER rely on the headline `l1tex__t_sector_hit_rate.pct` alone.

This pitfall does NOT apply on Hopper sm_90 (separate L1/SMEM accounting
in ncu) — it's sm_120-and-earlier specific. (Same physical L1/TEX/SMEM
unit on Ampere/Ada/Blackwell-Geforce; Hopper-Datacenter has split.)

## evidence

- `kernel_opt_attn_fp4_fusion/profiles/v1_tma/v1_tma_l1_disambiguation.log`
  (raw ncu --csv output)
- `kernel_opt_attn_fp4_fusion/profiles/v1_tma/v1_tma_ncu_summary.txt`
  (Addendum section, full diagnosis)
- commit `4b0f3b2` in `kernel_opt_attn_fp4_fusion` working tree

## related sm_120 wiki references

- `reference-kernels/nvidia/blackwell/cutedsl/flashinfer/nvfp4_quantize.py`
  — `NVFP4QuantizeTMAKernel` `_should_use_tma` heuristic (lines 1126-1144) claims
  TMA is faster on sm_120; this trap explains why naive interpretation of L1 hit
  rate would suggest otherwise.
- `reference-kernels/nvidia/blackwell-geforce/cutedsl/cutlass/sm120_nvfp4_persistent_gemm_pro5000.py`
  — confirms `cpasync.CopyBulkTensorTileG2SOp` works on sm_120 cute 4.4.2.
