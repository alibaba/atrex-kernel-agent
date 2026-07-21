pattern: matmul
type: pipeline
priority: ⭐⭐⭐
performance_gain: +27%
applicable_scenarios:
  - Large tile GEMM (256×256 and above)
  - Scenarios requiring extreme performance optimization
  - Configurations with manageable VGPR pressure
pitfalls:
  - 9
---

# Warp Pipeline Stage Full Packaging (Key GEMM Optimization)

> ⚠️ **Most important prerequisite**: When using `warp_pipeline_stage`, `buffer_load` **must NOT pass the `other` parameter** (e.g., `other=0.0`).
> Violating this rule causes the WarpPipeliner ping-pong orchestration to completely fail, degrading performance from 204 to 134 TFLOPS (-34%).
> See pitfall 9 for details.

## Objective

Use `warp_pipeline_stage` to package **all stages** of the GEMM loop (prep/compute/store, etc.) and hand them to the compiler, allowing the compiler to automatically perform ping-pong pipeline scheduling.

## Background

`warp_pipeline_stage` is not simply a "scheduling hint" — it packages different process stages of the GEMM and hands them to the compiler's WarpPipeliner, enabling automatic ping-pong pipelining across iterations. Once the compiler sees the complete `prep → compute → prep → compute → ...` sequence, it can automatically overlap the compute of iteration N with the prep of iteration N+1.

## ❌ Common Mistake: Only Packaging Compute

```python
# ❌ MFMA (compute), ds_read
# compilation compute , none prep/compute iteration
for k_iter in range(num_k_iter - 1):
    a0 = a_tile.slice(0, 16, dim=1).load(layout=dot_layout_a)
    b0 = b_tile.slice(0, 16, dim=0).load(layout=dot_layout_b)

    with warp_pipeline_stage("compute"):
        acc = ttgl.amd.cdna3.mfma(a0, b0, acc)

 # ... subslices compute
    gl.barrier()
    a_tile.store(next_a_val)
    b_tile.store(next_b_val)
    gl.barrier()
```

**Result**: 161 TFLOPS (GEMM 256×256×64 bf16, MI308X)

## ✅ Correct Approach (Round 1): Package All Stages

```python
# ✅ ds_read "prep", MFMA "compute", ds_write "prep"
# compilationcomplete pipeline stage column, automatic ping-pong
for k_iter in range(num_k_iter - 1):
    with warp_pipeline_stage("prep"):
        a0 = a_tile.slice(0, 16, dim=1).load(layout=dot_layout_a)
        b0 = b_tile.slice(0, 16, dim=0).load(layout=dot_layout_b)

    with warp_pipeline_stage("compute"):
        acc = ttgl.amd.cdna3.mfma(a0, b0, acc)

 # ... subslices ...

 # ds_write prep
    with warp_pipeline_stage("prep"):
        a_tile.store(next_a_val)
        b_tile.store(next_b_val)
 # manual gl.barrier
```

**Result**: 182 TFLOPS (+13%, reaching 91% of Triton)

## ✅✅ Round 2 Optimization: Place buffer_load Between Stages + Split A/B

```python
# ✅✅ buffer_load pipeline stage , A/B differentbit
# subslice 2+3 MFMA compute block
for k_iter in range(num_k_iter - 1):
 next_a_val = ttgl.amd.cdna3.buffer_load(...) # ① A
    # prep→compute (subslice 0)
    # prep→compute (subslice 1)
 next_b_val = ttgl.amd.cdna3.buffer_load(...) # ② B subslice 1
 # prep (subslice 2&3 ds_read coalesced)
 # compute (subslice 2&3 MFMA coalesced)
    # prep (ds_write)
```

**Result**: 193 TFLOPS (+6.3%, reaching 96.7% of Triton)

## ✅✅✅ Best Practice (Round 3): Insert ds_write Between MFMAs to Achieve Compute↔Store Overlap

```python
# ✅✅✅ critical: ds_write "prep" stage subslice 2 3 "compute"
# subslice 2+3 MFMA . compilation MFMA execute ds_write .
# ⚠️ buffer_load do not other=0.0, otherwise WarpPipeliner pingpong failure！
for k_iter in range(num_k_iter - 1):
 # ① buffer_load A loop( stage , other)
    next_a_val = ttgl.amd.cdna3.buffer_load(a_ptr, next_offs_a, mask=next_k_mask_a)

    with warp_pipeline_stage("prep"):        # subslice 0: ds_read
        a0 = a_tile.slice(0, 16, dim=1).load(layout=dot_layout_a)
        b0 = b_tile.slice(0, 16, dim=0).load(layout=dot_layout_b)

    with warp_pipeline_stage("compute"):     # subslice 0: MFMA (32 ops)
        acc = ttgl.amd.cdna3.mfma(a0, b0, acc)

    with warp_pipeline_stage("prep"):        # subslice 1: ds_read
        a1 = a_tile.slice(16, 16, dim=1).load(layout=dot_layout_a)
        b1 = b_tile.slice(16, 16, dim=0).load(layout=dot_layout_b)

    with warp_pipeline_stage("compute"):     # subslice 1: MFMA (32 ops)
        acc = ttgl.amd.cdna3.mfma(a1, b1, acc)

 # ② buffer_load B subslice 1 after( stage , other)
    next_b_val = ttgl.amd.cdna3.buffer_load(b_ptr, next_offs_b, mask=next_k_mask_b)

 with warp_pipeline_stage("prep"): # subslice 2&3: ds_read (coalescedread)
        a2 = a_tile.slice(32, 16, dim=1).load(layout=dot_layout_a)
        b2 = a_tile.slice(32, 16, dim=0).load(layout=dot_layout_b)
        a3 = a_tile.slice(48, 16, dim=1).load(layout=dot_layout_a)
        b3 = a_tile.slice(48, 16, dim=0).load(layout=dot_layout_b)

 with warp_pipeline_stage("compute"): # ③ subslice 2 MFMA (32 ops)
        acc = ttgl.amd.cdna3.mfma(a2, b2, acc)

 with warp_pipeline_stage("prep"): # ④ ds_write subslice 2 3 ！
        a_tile.store(next_a_val)
        b_tile.store(next_b_val)

 with warp_pipeline_stage("compute"): # ⑤ subslice 3 MFMA (32 ops)
        acc = ttgl.amd.cdna3.mfma(a3, b3, acc)
 # manual gl.barrier

# EPILOGUE: K-subslicing, directloadcompilationoptimization
a_full = a_tile.load(layout=dot_layout_a)
b_full = b_tile.load(layout=dot_layout_b)
acc = ttgl.amd.cdna3.mfma(a_full, b_full, acc)
```**Result**: 204.7 TFLOPS (+5.8% more, **surpassing Triton's 202 TFLOPS!**)

## Why Placing buffer_load Outside the Stage is Better

| Placement | Effect | Reason |
|-----------|--------|--------|
| buffer_load inside `"prep"` stage | 182 TFLOPS | WarpPipeliner manages all prep operations uniformly, but global load (400+ cycle) and ds_read (20 cycle) have completely different latency characteristics — unified scheduling is suboptimal |
| buffer_load **outside** the stage | **193 TFLOPS** | WarpPipeliner only handles LDS↔MFMA ping-pong, while global loads are delegated to the LLVM standard scheduler. The two schedulers each do their own job without interfering with each other |

## Why Splitting A/B Loads to Different Positions

```
loop:
 buffer_load A ────── ( ds_write ~4 subslice )
  subslice 0: prep → compute
  subslice 1: prep → compute
 buffer_load B ────── ( ds_write ~2 subslice )
  subslice 2&3: prep → compute
 ds_write A,B ─────── ( A/B complete)
```

- Load for A is issued earliest, with 4 subslices of compute time to hide the 400+ cycle latency
- Load for B is issued slightly later, with 2 subslices (64 MFMA instructions) to hide latency
- If A and B are issued simultaneously, they compete for global memory bandwidth without any additional latency-hiding benefit

## Why Inserting ds_write Between MFMAs Works Better

**Round 2** (subslice 2+3 MFMA merged → ds_write at the end):
```
subslice 2&3: 64 MFMA → ds_write → barrier
```
ds_write must wait for all 64 MFMA instructions to complete before it can be issued; the ds_write latency cannot be hidden by any MFMA.

**Round 3** (ds_write inserted between subslice 2 and 3):
```
subslice 2: 32 MFMA → ds_write → subslice 3: 32 MFMA → barrier
```
ds_write is issued immediately after subslice 2's 32 MFMA instructions complete, then subslice 3's 32 MFMA instructions can execute **in parallel** with ds_write's LDS write. This is compute↔store overlap.

**Assembly Evidence** (Round 3):
```asm
; subslice 2: 32× v_mfma (line 911-942)
sched_barrier mask(0x00000000)
s_barrier ; subslice 2
sched_barrier mask(0x00000000)
s_waitcnt vmcnt(6) ; buffer_load A 2 complete
ds_write2st64_b64 ... ; ds_write A ( LDS)
s_waitcnt vmcnt(4)
ds_write2st64_b64 ...
...
s_waitcnt vmcnt(0) ; buffer_load complete
ds_write2st64_b64 ...              ; ds_write B
sched_barrier mask(0x00000000)
s_waitcnt lgkmcnt(0)
s_barrier ; ds_write complete
sched_barrier mask(0x00000000)
; subslice 3: 32× v_mfma (line 967-998) ← ds_write LDS ！
```

Key point: The decreasing pattern of `s_waitcnt vmcnt(6/4/2/0)` indicates that the compiler progressively waits for buffer_load to complete between ds_write instructions, rather than waiting for all of them at once. This creates a pipelining effect for the buffer_load→ds_write chain as well.

## Key Rules

1. **All LDS operations and compute must be packed** — ds_read uses `"prep"`, MFMA uses `"compute"`, ds_write uses `"prep"`; all three are essential
2. **Do not manually write `gl.barrier()`** — let the compiler's Membar analysis automatically insert barriers (see explanation below)
3. **Insert ds_write between MFMAs** — do not merge adjacent subslices' MFMA instructions into a single `"compute"` block. Insert the ds_write's `"prep"` between subslice 2's compute and subslice 3's compute to achieve compute↔store overlap
4. **Place buffer_load between pipeline stages, not wrapped inside any stage** — global memory access is handled by the LLVM standard scheduler, LDS↔MFMA ping-pong is handled by WarpPipeliner; each does its own job
5. **Split A/B buffer_load to different positions** — place A at the very beginning of the loop and B after subslice 1. Maximize latency hiding by spacing them apart (see detailed explanation above)
6. **Do not use the `other=0.0` parameter for buffer_load** — the `other` parameter causes the compiler to generate additional `v_cndmask_b32` instructions for each `buffer_load` (replacing out-of-bounds data with 0), which disrupts WarpPipeliner's ping-pong instruction scheduling and causes it to degrade to bulk-phase scheduling (134 TFLOPS vs 204 TFLOPS). The correct approach is to only pass `mask`, not `other`
7. **Do not apply K-subslicing in the epilogue** — for the final iteration (epilogue), directly load the full BLOCK_K using `smem.load(layout=dot_op)` and compute with a single `mfma`, letting the compiler handle it automatically. Manually splitting into 4×16 subslices in the epilogue yields no performance benefit and only increases code complexity

## Why Not Manually Write gl.barrier()

| Method | Low-Level Implementation | Effect |
|--------|--------------------------|--------|
| `gl.barrier()` | `gpu::BarrierOp` → only `s_barrier` (**without** `s_waitcnt`) | May cause data races |
| Compiler auto-inserted | `triton::gpu::BarrierOp(Local)` → `s_waitcnt lgkmcnt(0)` + `s_barrier` | Correct synchronization |**`gl.barrier()` does not include `s_waitcnt lgkmcnt(0)`**: when you manually insert `gl.barrier()`, it only synchronizes thread execution positions and does not wait for LDS reads/writes to complete. If you place `ds_write` after `gl.barrier()` and before MFMA, `ds_read` from other wavefronts may get overwritten before it completes.

The compiler's Membar analysis will automatically insert `triton::gpu::BarrierOp(Local)` at `ds_read`/`ds_write` conflicts. This barrier includes `s_waitcnt`, guaranteeing correctness.

**Additionally, removing manual barriers gives the compiler more freedomurable to perform ping-pong scheduling**, no longer cutting the pipeline short with manual barriers.

## WPS Diagnostic Methods

```bash
# 1. check warp_pipeline
# if sched_barrier(0) + s_barrier , description WarpPipeliner success
grep -c 'sched_barrier\|s_barrier' kernel.amdgcn

# 2. checkwhether there is scratch (ping-pong increase VGPR )
grep 'private_segment_fixed_size' kernel.amdgcn
# : 0 bytes

# 3. optimization s_waitcnt count
grep -c 's_waitcnt' kernel.amdgcn
# ping-pong success s_waitcnt countdecreaseor
```

## Related Documentation

- **Prerequisites**: [CDNA3 ISA Instruction Patterns](../../../ref-docs/gluon/isa_patterns.md)
- **GEMM Chain Upstream**: [Optimization Strategy](../../kernel-opt/gluon/optimization_strategy.md) → This document → [Stopping Conditions](../../kernel-opt/gluon/final_config_template.md)
- **Converter Reference**: [CDNA3 Pipeline Conversion](../../../converter/cdna3.md) — Pipeline code produced by conversion is optimized by this document
- **🔴 Conflict Notice**: The WPS +27% conclusion in this document only applies to pure GEMM. [CDNA4 pitfalls #7](../../../../cdna4/ref-docs/gluon/pitfalls.md) found that WPS is a **negative optimization** for attention (which has cross-iteration dependencies from online softmax)
