# FlyDSL Flash Attention Forward (fp16, no-mask) on MI308X

## Target hardware

- **Chip**: AMD MI308X (CDNA3, gfx942)
- **Framework**: FlyDSL rebuilt with ISA scheduling improvements
- **Target shape**: `B=1024, H=8, S=316 padded to 320, D=64`
- **Input dtype**: fp16
- **Feature mode**: no mask, non-causal
- **Reference CK command**:
  `<composable_kernel-build>/bin/tile_example_fmha_fwd -b=1024 -h=8 -v=0 -d=64 -s=316`

## Algorithm baseline

The kernel is the MI308X FlashAttention forward path with all mask logic compiled
out (`HAS_MASK=False`). It uses the same high-level structure as the earlier
MI308X FlyDSL attention kernels:

- `BLOCK_M=128`, `BLOCK_N=64`, D64 native.
- `K @ Q^T` for QK so the score/probability tile stays in MFMA32 layout.
- Online softmax in registers.
- Register-resident P feeding `V^T @ P` without an LDS roundtrip.
- K/V staged through LDS, with an accepted FP16 overlay that lets K and V share
  the same LDS slot.

This is **not** a replacement for the existing MI308X causal/GQA kernel or the
bit-packed-mask kernel. The archived source is a separate file:
[`flash_attn_func_fp16_nomask_mi308x.py`](../../../../../reference-kernels/amd/cdna3/flydsl/FlyDSL/flash_attn_func_fp16_nomask_mi308x.py).

## Kernel resource footprint (final)

| Item | Value |
|---|---:|
| rocprofv3 P50 | 3.327254 ms |
| Throughput | 62.938751 TFLOPS |
| Correctness | `rel_err vs fp32 = 0.019502` (`< 0.02`) |
| LDS | 8704 B |
| Scratch | 0 B |
| VGPR | 128 |
| AccVGPR | 0 |
| SGPR | 96 |

Final ATT instruction counts from `profiles/fp16_v25_final_default_att/codeobj5_disasm.s`:

| Instruction / class | FlyDSL final | CK reference |
|---|---:|---:|
| `v_mfma_f32_32x32x8_f16` | 32 | 32 |
| `ds_read_b128` | 0 | 16 |
| `ds_read2_b32` | 32 | 0 |
| `ds_write2_b32` | 12 | 4 |
| `s_waitcnt` | 38 | 29 |
| `s_nop` | 262 | 55 |
| `s_barrier` | 3 | 5 |

The important result is that MFMA count parity is not enough. CK wins through
descriptor-level LDS layouts, `ds_read_b128`, correct K dword-to-LDS staging,
and a much tighter wait/nop schedule.

## Optimization journey

### V0 - fp16 no-mask baseline

Added dtype-aware `verify_nomask.py` and `profile_nomask.py` support. The first
accepted fp16 no-mask baseline measured:

- Correctness: `rel_err vs fp32 = 0.012494`
- rocprofv3 P50: `3.5877335 ms`
- Throughput: `58.369221 TFLOPS`
- Resources: LDS 16896 B, scratch 0, VGPR 128, AccVGPR 8, SGPR 112

This was only 86.17% of the then-used CK95 target, so the optimization pivoted
to fp16-specific codegen and CK ISA comparison.

### V1 - P probability pack with `v_cvt_pkrtz_f16_f32`

Packing pairs of fp32 probabilities directly to fp16 with
`v_cvt_pkrtz_f16_f32` removed half of the scalar `v_cvt_f16_f32` /
`v_pack_b32_f16` sequence:

- Default same-run: `58.580825 TFLOPS`
- P-pack pkrtz: `60.195358 TFLOPS`
- ISA: `v_cvt_pkrtz_f16_f32=16`, fp16 MFMA still 32

The first attempt was rejected because it increased AccVGPR and did not reach
CK95. Later rounds retained the P-pack path only after the LDS overlay and final
resource shape made it part of the best stable default.

### V15 - WPE and pkrtz sweeps

The best official sweep point was P-pack pkrtz plus `waves_per_eu=2`:

- Correctness: `rel_err vs fp32 = 0.012814`
- Throughput: `60.200904 TFLOPS`
- Resources: LDS 16896 B, scratch 0, VGPR 124, AccVGPR 12, SGPR 112

O-store pkrtz removed additional scalar conversions but stayed flat
(`60.136240 TFLOPS`). The lesson was that this fp16 no-mask D64 shape
requires structural changes (LDS descriptor shape, dword-to-LDS staging)
rather than compiler-level improvements alone.

### V18-V24 - K/V LDS overlay and small scheduling defaults

The accepted structural change was enabling K/V LDS overlay for fp16. It reduced
LDS pressure and moved the kernel into the low-62 TFLOPS range. The stable
defaults retained by the final file are:

- FP16 K/V LDS overlay enabled.
- FP16 P-pack via `v_cvt_pkrtz_f16_f32`.
- FP16 no-mask softmax `sched_barrier` disabled by default.
- Builder, verify, and profile default to `waves_per_eu=2` for fp16.

Sweeps of QK prefetch depth 2/4, reduction mode, `LDS_VEC16=0`, and pkrtz
disable were flat or worse.

### V25 - CK ISA pairpack / dword-DMA experiments

The local CK reference was rerun and captured with rocprofv3 ATT:

- CK P50: `2.912971 ms`
- CK throughput: `71.889906 TFLOPS`
- CK95 target: `68.295411 TFLOPS`
- CK ISA: 32 fp16 MFMAs, 16 `ds_read_b128`, 0 `ds_read2_b32`, 55 `s_nop`

Experiments tried to mimic the CK ISA at the FlyDSL source level:

- K pairpack layout forced `ds_read_b128=8` and passed correctness, but reached
  only `62.050298 TFLOPS`.
- V pairpack layout passed correctness but regressed to about `50.28 TFLOPS`.
- `K_PAD=0` with a D64-safe swizzle produced non-finite output.
- K dword-to-LDS staging no longer faulted in this build but produced
  non-finite output.

The retained final default is therefore conservative: keep the stable overlay
and pkrtz changes; leave CK-like pairpack and dword-DMA paths as default-off
diagnostics.

## Final perf vs baseline

| Implementation | P50 (ms) | TFLOPS | Notes |
|---|---:|---:|---|
| CK fp16 no-mask | 2.912971 | 71.889906 | Local CK command above |
| CK95 target | 3.066286 equivalent | 68.295411 | 95% of local CK throughput |
| FlyDSL fp16 no-mask baseline | 3.5877335 | 58.369221 | First accepted fp16 baseline |
| FlyDSL pkrtz + WPE sweep | 3.478572 | 60.200904 | Good but not final resource shape |
| FlyDSL final default | 3.327254 | 62.938751 | 87.55% of CK, 92.16% of CK95 |
| K pairpack diagnostic | - | 62.050298 | `ds_read_b128=8`, slower |
| V pairpack diagnostic | - | ~50.28 | Correct but much slower |

## Remaining bottlenecks

1. **LDS read shape**: FlyDSL final still uses 32 `ds_read2_b32`; CK uses
   16 `ds_read_b128`. Source-level pairpack can change the instruction shape,
   but not the full CK descriptor layout and schedule.
2. **Wait/nop schedule**: FlyDSL final has 262 `s_nop` and 38 `s_waitcnt`; CK
   has 55 `s_nop` and 29 `s_waitcnt`.
3. **Direct-to-LDS staging**: CK's K pipeline uses dword-to-LDS staging correctly.
   The FlyDSL diagnostic path compiled but produced non-finite output.
4. **Source-level scheduling limits**: Inline wait hints and `sched_barrier`
   changes can be moved or reinterpreted by LLVM unless real data dependencies
   pin them. They are not a substitute for CK's template-level scheduler.

## What would close the remaining gap

- Implement CK-equivalent LDS descriptors in FlyDSL lowering so K/V can be read
  as `ds_read_b128` without the source-level pairpack overhead.
- Make K dword-to-LDS staging correct for gfx942 D64, then re-profile with ATT.
- If staying in source-level FlyDSL, use diagnostics only to prove an ISA shape;
  do not promote a knob unless rocprofv3 kernel-trace and correctness both pass.
- For manual ISA work, compare against real `rocprofv3 --att` CK disassembly,
  not a presumed CK schedule.

## Sustained recipe

1. Measure CK locally for the exact dtype and shape:
   `<composable_kernel-build>/bin/tile_example_fmha_fwd -b=1024 -h=8 -v=0 -d=64 -s=316`.
2. Validate FlyDSL with:
   `FLASH_ATTN_DTYPE=f16 /opt/conda310/envs/vllm/bin/python verify_nomask.py`.
3. Profile with rocprofv3 kernel trace, 10 warmup + 50 measured dispatches.
4. Capture ATT and count `ds_read_b128`, `ds_read2_b32`, `s_waitcnt`, and `s_nop`
   before claiming CK-like progress.
5. Keep final defaults: K/V LDS overlay, P-pack pkrtz, no-mask softmax barrier
   off, fp16 `waves_per_eu=2`.
6. Keep pairpack, K dword-DMA, K chunk32, and K swizzle toggles default-off until
   they pass finite correctness and improve rocprofv3 P50.

## Related docs

- Reference kernel: [`flash_attn_func_fp16_nomask_mi308x.py`](../../../../../reference-kernels/amd/cdna3/flydsl/FlyDSL/flash_attn_func_fp16_nomask_mi308x.py)
- Pitfalls: [`flash-attn-pitfalls.md`](../../../../pitfalls/amd/flydsl/flash-attn-pitfalls.md)
- Existing MI308X causal/GQA report: [`cdna3-flash-attention-bf16-gqa-optimization.md`](cdna3-flash-attention-bf16-gqa-optimization.md)
- Existing MI308X bit-packed mask report: [`cdna3-flash-attention-bf16-mask-optimization.md`](cdna3-flash-attention-bf16-mask-optimization.md)
- Generic CDNA FlyDSL baseline: [`flash_attn_func.py`](../../../../../reference-kernels/amd/cdna/flydsl/FlyDSL/flash_attn_func.py)

## Difference notes against existing docs

- This report uses **fp16 no-mask** inputs. The earlier mask and causal/GQA
  reports are bf16-focused and should not be used as fp16 CK95 evidence.
- In this fp16 D64 no-mask path, the performance gap is dominated by LDS
  descriptor shape and dword-to-LDS correctness, not instruction scheduling.
