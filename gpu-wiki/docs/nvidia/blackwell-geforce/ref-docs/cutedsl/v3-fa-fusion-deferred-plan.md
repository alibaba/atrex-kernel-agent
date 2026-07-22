# V3 deferred plan: cute-DSL FlashAttention forward + sigmoid·gate + NVFP4 quant single-kernel fusion (sm_120)

**Status**: DEFERRED. Design complete; implementation blocked on `cutlass-dsl >= 4.5` (cluster currently 4.4.2). See `wiki_drafts/sm120-cutedsl-vendor-pitfalls.md` for the cutlass version blocker. Resume when cluster cutlass upgrades or `cutlass + flash_attn` are bundled together.

**Target shape**: Path-1, M=6144 (varlen single seq), Q (6144, 16, 256) bf16, K/V (6144, 2, 256) bf16, GQA ratio 8, gate (6144, 4096) bf16, causal=True, output NVFP4 (e2m1 packed + e4m3 SF, group_size=16, swizzled-128x4 layout).

**Hardware**: NVIDIA RTX PRO 5000 Blackwell-GeForce, sm_120, 110 SMs, 99 KB SMEM/CTA, GDDR7 512-bit, 1,344 GB/s official bandwidth (measured memcpy ceiling 1099 GB/s @ ≥115 MB).

**Projected wall-clock**: ~130 us for the fused kernel (77 MB traffic at memcpy ceiling), end-to-end **~14× over SDPA+V0** (1825 us → ~130 us). ~9× even if upstream FA on sm_120 stays slow because we eliminate the 50 MB attn_out DRAM round-trip.

---

## §1 The structural change (one paragraph)

Subclass `FlashAttentionForwardSm120` (which already subclasses `FlashAttentionForwardSm80`) and **override only `epilogue()`**. The current `epilogue()` (`flash_fwd.py:324-442` in the upstream / wiki copy) takes `acc_O: cute.Tensor` (FP32 accumulator fragment, already in registers per-thread, layout `(MMA_atom × MMA_M × MMA_N)` per the FA80 mma scheme), converts it to bf16 `rO`, then writes to gmem `mO`. **Replace the gmem write** with: (a) load gate per-thread for the same per-thread `(m, k)` slice, (b) run `bfloat2_sigmoid_mul(rO_pair, gate_pair)` element-wise, (c) per 16-element SF-block run V0's amax → e4m3 scale → e2m1 pack chain, (d) write `x_fp4` (`stg.E.64`) and `x_bs` (swizzled SF byte) to gmem. The `mO` argument becomes `mOutput` (uint8 fp4-packed) + `mScales` (uint8 swizzled SF). Mainloop (Q/K/V loading, softmax, P·V mma) is unmodified. **Eliminates the 50 MB attn_out DRAM round-trip** — the whole point.

---

## §2 Wiki references (line-precise, all confirmed present in this `gpu-wiki/` checkout)

| Topic | File | Lines |
|---|---|---|
| FA forward sm_120 subclass (extension point, SMEM cap check, `arch=80` to keep cp.async paths) | `reference-kernels/nvidia/blackwell-geforce/cutedsl/flash-attention/flash_fwd_sm120.py` | 14-59 (full file) |
| FA forward sm_80 base class — has `__call__`, `kernel`, `epilogue`, `compute_one_n_block` | `reference-kernels/nvidia/ampere/cutedsl/flash-attention/flash_fwd.py` | 572-736 (Sm80 class body) |
| FA80 epilogue — THE override target | same | 324-442 (full `def epilogue`) |
| Where epilogue gets called in main `kernel()` body | same | 1056-1075 (just after `softmax.finalize()` + `softmax.rescale_O`) |
| FA80 `_get_shared_storage_cls` — has `Q_in_regs` toggle that recycles sQ for sV | same | 594-613 |
| FA80 `can_implement` SMEM accounting | same | 112-168 |
| Sm120 SMEM cap override (99 KB call) | flash_fwd_sm120 | 45-56 |
| In-register sigmoid·mul fusion pattern (`cute.arch.mul_packed_f32x2` + `cute.math.exp2`) | `reference-kernels/nvidia/blackwell/cutedsl/flashinfer/blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py` | 2598-2666 |
| In-register fp4-quant epilogue lambda (amax → SFC → quantize_and_pack_16) | same | 2668-2798 |
| fp4-quant kernel structure parallel (norm fused with fp4 in one kernel) | `reference-kernels/nvidia/blackwell/cutedsl/flashinfer/rmsnorm_fp4quant.py` | 76-200, 301-650 |
| All V0 PTX cvt helpers reused | `kernel_opt_attn_fp4_fusion/cute_helpers.py` | unchanged |

---

## §3 SMEM budget (D=256, sm_120 99 KB)

Per-stage SMEM cost @ tile_m / tile_n with D=256:
```
sQ  = tile_m * D * 2  (bf16)             # never staged
sK  = tile_n * D * 2 * num_stages
sV  = tile_n * D * 2 * num_stages
sO  = tile_m * D * 2  (recycles sQ via Q_in_regs=True)
sGate = tile_m * D * 2 (epilogue only, allocated after sV dies)
```

Candidate configurations (D=256), all assume bf16:

| tile_m | tile_n | stages | Q_in_regs | sQ | sK | sV | sGate | Total live max | Fits 99 KB? |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 2 | True | 32 KB | 32 KB | shared with sQ via `max(sQ,sV)=32 KB` | 32 KB (epilogue only) | **64 KB** | ✓ comfortable, 35 KB headroom |
| 64 | 64 | 2 | False | 32 KB | 32 KB | 32 KB | 32 KB | **128 KB** | ✗ |
| 64 | 64 | 1 | False | 32 KB | 32 KB | 32 KB | 32 KB | **128 KB** | ✗ (no pipeline either) |
| 32 | 64 | 2 | True | 16 KB | 32 KB | max=32 KB | 16 KB | **64 KB** | ✓ smaller per-CTA work |
| 128 | 32 | 1 | True | 64 KB | 16 KB | max=64 KB | 64 KB | **144 KB** | ✗ |

**Recommendation**: **`tile_m=64, tile_n=64, num_stages=2, Q_in_regs=True`** = 64 KB total live SMEM. Block Limit Shared Mem = 1 / SM (98 KB / 64 KB). Achieved Occupancy will be ~18-25 % (FA D=256 norm). Same regime as V2-TMA which we know is functional on sm_120.

`gate` strategy: cp.async G2S into sGate (32 KB) in the epilogue, single-stage, after sV dies. Issue 1 `cp.async G2S + LoadCacheMode.GLOBAL` per warp. `cp_async_commit_group(); cp_async_wait_group(0); barrier()` once. L1 bypass via `LoadCacheMode.GLOBAL` (V1 cp.async path proved structurally correct).

---

## §4 Subclass design (concrete override map)

```python
class FlashAttentionForwardSm120Fp4Quant(FlashAttentionForwardSm120):

    @staticmethod
    def can_implement(dtype, head_dim, head_dim_v, tile_m, tile_n,
                      num_stages, num_threads, is_causal, Q_in_regs=False) -> bool:
        ok = FlashAttentionForwardSm120.can_implement(
            dtype, head_dim, head_dim_v, tile_m, tile_n,
            num_stages, num_threads, is_causal, Q_in_regs)
        if not ok:
            return False
        smem_gate = tile_m * head_dim_v * 2  # gate same shape as O per output tile
        smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_120")  # 99 KB
        smem_used = (
            tile_m * head_dim * 2  # sQ
            + tile_n * head_dim * num_stages * 2  # sK
            + (max(tile_m * head_dim * 2, tile_n * head_dim_v * num_stages * 2)
               if Q_in_regs
               else tile_m * head_dim * 2 + tile_n * head_dim_v * num_stages * 2)
            + smem_gate
        )
        return smem_used <= smem_capacity

    def _get_shared_storage_cls(self):
        # Extend FA80 SharedStorage with sGate (allocated after sV dies in epilogue)
        base = super()._get_shared_storage_cls()
        sGate_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, self.tile_m * self.tile_hdimv], 1024
        ]
        @cute.struct
        class SharedStorageFp4(base):
            sGate: sGate_struct
        return SharedStorageFp4

    def epilogue(self, acc_O, lse, mGate, mOutput, mScales,
                 mGlobalScale, sO, sGate, seqlen,
                 gmem_tiled_copy_O, gmem_tiled_copy_Gate,
                 tma_atom_O, tiled_mma, tidx, m_block, head_idx, batch_idx):
        # 1. Convert acc_O FP32 fragment to bf16 fragment (UNCHANGED from FA80)
        rO = cute.make_fragment_like(acc_O, self.dtype)
        rO.store(acc_O.load().to(self.dtype))
        cute.arch.barrier(barrier_id=int(NamedBarrierFwd.Epilogue),
                          number_of_threads=self.num_epilogue_threads)

        # 2. cp.async G2S load gate into sGate
        gGate = cute.local_tile(
            mGate[(None, None)],
            (self.tile_m, self.tile_hdimv),
            (m_block, head_idx),
        )
        thr_copy_gate = gmem_tiled_copy_Gate.get_slice(tidx)
        cute.copy(gmem_tiled_copy_Gate,
                  thr_copy_gate.partition_S(gGate),
                  thr_copy_gate.partition_D(sGate))
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier(barrier_id=int(NamedBarrierFwd.Epilogue),
                          number_of_threads=self.num_epilogue_threads)

        # 3. RECOMMENDED PATH: round-trip rO → sO, then re-read in V0 thread layout
        #    Reason: mma_pv layout doesn't expose 16-element SF-block boundaries
        #    cleanly; round-trip lets us reuse V0 epilogue mathematics 1:1.
        #    Alternative: work out per-thread SF-block extraction from
        #    thr_mma.partition_C(sO) — defer to V4 optimization.
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)  # rO → sO (V0-compatible layout)
        cute.arch.barrier()
        # Now sO holds (sigmoid_mul not yet applied) attn_out in V0-friendly layout

        # 4. Re-read sO in V0 thread layout (256 thr × 16 SF blocks per row × ...)
        # then apply sigmoid_mul + V0 fp4-quant pipeline VERBATIM.
        # See V0 kernel.py for the read + epilogue body.
        # Inline gate read from sGate, paired with attn read from sO.
        # x = sigmoid(gate) * attn_out — V0's bfloat2_sigmoid_mul UNCHANGED.
        # Then V0's amax + cvt_f32_to_e4m3 + bfloat2x8_to_e2m1x16_packed + st_global_u64.
```

---

## §5 ISA-level targets

| Metric | Current (SDPA + V0 standalone) | V3 fused target | Why |
|---|---|---|---|
| End-to-end wall-clock | SDPA 1742 + V0 88 = ~1830 us | **≤ 200 us** total | SDPA replaced by FA fwd ~99 us, fp4 epilogue absorbed into FA mainloop |
| Fused-kernel duration (ncu) | n/a | **≤ 130 us** | 77 MB / 1099 GB/s memcpy floor ≈ 70 us; +cvt overhead ≈ 130 us realistic |
| DRAM throughput | V0: 1010 GB/s on 114 MB | ≥ 1000 GB/s on 77 MB (~91 % memcpy ceiling for that traffic) | same memory-bound regime, less traffic |
| Traffic reduction | 114 MB (V0 standalone) | **77 MB total** | -33 % (50 MB attn_out round-trip eliminated; +27 MB FA Q+K+V net) |
| L1/TEX Hit Rate (global only) | V0 46 %, V2 50 % (shared inflated) | ≤ 30 % global | bypass via TMA / cp.async-GLOBAL |
| Achieved Occupancy | V0 84 %, V2 19 % | 25-50 % | FA D=256 normal regime |
| long_scoreboard / mbar_wait | V0 34 cyc, V2 6.5 cyc mbar | ≤ 15 cyc | 2-stage K/V pipeline + epilogue gate load behind compute |
| Eligible Warps Per Sched | V0 0.35, V2 0.31 | ≥ 0.6 | 2-stage pipeline keeps warps issuable during softmax |
| validate.py PASS | rel_err 1.56e-4 | rel_err < 5e-3 vs vllm reference | oracle = `vllm.vllm_flash_attn.flash_attn_varlen_func` |
| Reg / thread | V0 37, V2 66 | ≤ 96 | FA D=256 register-heavy; ptxas may spill |

---

## §6 Risk catalog (ranked by likelihood)

| # | Risk | Likelihood | Symptom | Fix |
|---|---|---|---|---|
| 1 | **`flash_attn` package not on cluster + cutlass version mismatch** | **CONFIRMED** (this is the deferral reason) | shim chain explodes (P1→P2→P3→P4...) | Wait for cluster cutlass 4.5+ OR bundle cutlass + flash_attn vendor together |
| 2 | D=256 tile_m=64 doesn't fit even with Q_in_regs | LOW (math says it fits) | `can_implement` returns False | Drop to tile_m=32 (16+32+32+16=96 KB tight but feasible) |
| 3 | mma_pv layout doesn't expose 16-element SF-block boundaries cleanly | HIGH | quantize step is ugly | Round-trip via SMEM (recommended above), reuse V0 epilogue verbatim |
| 4 | SDPA causal + GQA + varlen interaction in epilogue | MED | rel_err > 5e-3 vs oracle | Bisect: first verify rO ≈ vllm flash_attn output (skip fp4 path), then add fp4 |
| 5 | num_epilogue_threads != V0's 512 thr | MED | thread layout mismatch when computing global_row / sf_col | FA80 default `num_epilogue_threads = 128`; recompute thread→(row, sf_col) mapping for new thread count |
| 6 | gate cp.async fails because num_threads doesn't divide cleanly into gate tile | LOW | compile error | Adjust thread layout for gate copy atom |
| 7 | `Q_in_regs=True` corrupts sO when epilogue writes rO to sO before reading sGate | MED | rel_err >> 5e-3 | Order matters: load sGate FIRST, barrier, THEN write rO to sO |
| 8 | sm_120 cute 4.4.2 traps from V2 (PEP 563, `@cute.struct` lexical scope, recast hoisting, ceil_div Int32) | MED-HIGH | Same compile errors V2 hit | Apply V2's 5 documented workarounds (see `wiki_drafts/sm120-tma-warp-spec-pitfalls.md`) — note these may or may not persist on cutlass 4.5+ |

---

## §7 Validation oracle

`vllm.vllm_flash_attn.flash_attn_varlen_func` — confirmed importable + functionally correct on sm_120 (probe Q3 PASS, 2026-04-27). Used as numerical oracle only; speed irrelevant for oracle role.

```python
from vllm.vllm_flash_attn import flash_attn_varlen_func
attn_out_oracle = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                                          max_seqlen_q, max_seqlen_k,
                                          softmax_scale=1.0/math.sqrt(256),
                                          causal=True)
# attn_out_oracle is bf16 (N, H_q, D)
# then run V0 fp4_quant on attn_out_oracle * sigmoid(gate)
# for x_fp4_oracle / x_bs_oracle to compare against V3 fused output.
```

NOTE: vllm's flash_attn_varlen_func is **slow on sm_120** (1645 us, ~16× slower than memcpy ceiling — see `wiki_drafts/sm120-flash-attn-vllm-no-fast-path.md`). This is fine for oracle role but means the V3 plan is the only known route to actually-fast attention on sm_120 RTX PRO 5000.

---

## §8 Implementation sequence (when resumed)

1. **Verify cluster `nvidia_cutlass_dsl >= 4.5`** OR bundle cutlass+flash_attn vendor together. Re-run probe to check P1-P4 traps from `wiki_drafts/sm120-cutedsl-vendor-pitfalls.md` are gone.
2. **Commit "V3-1: env ready"** — flash_attn imports work, cute-DSL FA80 base class instantiates.
3. **Commit "V3-2: oracle wired up"** — `validate.py` updated to compare against `flash_attn_oracle(q,k,v) → sigmoid_mul → V0_fp4` reference.
4. **Commit "V3-3: subclass skeleton + can_implement override"** — class compiles, `can_implement(bf16, 256, 256, 64, 64, 2, 128, True, Q_in_regs=True)` returns True.
5. **Commit "V3-4: stub epilogue (write attn_out as bf16, no fp4 yet)"** — verifies the FA pass alone is bit-equivalent to oracle.
6. **Commit "V3-5: add gate cp.async load + sigmoid·mul in epilogue"** — verifies sigmoid_mul stage matches `flash_attn_oracle * sigmoid(gate)` in bf16.
7. **Commit "V3-6: full fp4 epilogue"** — V3 = full pipeline, validate.py PASS.
8. **Commit "V3-7: ncu profile to profiles/v3/"** — gates check; if needed, V3.1 = SF write coalescing / FFMA fusion.

---

## §9 What stays unchanged from V0/V1/V2 (DON'T touch)

- All inline-PTX helpers in `cute_helpers.py` for the epilogue: `bfloat2_sigmoid_mul`, `bfloat2_max_abs_8`, `bfloat2_hmax_reduce_to_f32`, `bfloat2x8_to_e2m1x16_packed`, `cvt_f32_to_e4m3`, `nvfp4_compute_output_scale`, `compute_sf_index_swizzled_128x4_gpu`, `st_global_u64`, `get_ptr_as_int64` — all reused as-is.
- `_bootstrap_cutedsl()` module-level call (sz6wd8l56pnf cluster bogus cutlass==0.1.0 cleanup) — still needed if cluster moves back.
- `from_dlpack(attn_out, assumed_align=16)` and `from_dlpack(gate, assumed_align=16)` — V0 already correct; TMA NEEDS 16-B aligned gmem ptrs.

---

## related

- `wiki_drafts/stage3-closeout.md` — full Stage 3 narrative including why this plan was deferred
- `wiki_drafts/sm120-cutedsl-vendor-pitfalls.md` — the cutlass 4.4.2 vs 4.5+ vendor blocker
- `wiki_drafts/sm120-flash-attn-vllm-no-fast-path.md` — why hybrid fallback only got 1.07× (the gap this plan would close)
- `wiki_drafts/sm120-tma-warp-spec-pitfalls.md` — 5 cute 4.4.2 traps (some may persist on cute 4.5+; verify when resuming)
- `wiki_drafts/sm120-ncu-l1-hit-rate-includes-shared.md` — ncu metric pitfall to avoid when profiling V3
