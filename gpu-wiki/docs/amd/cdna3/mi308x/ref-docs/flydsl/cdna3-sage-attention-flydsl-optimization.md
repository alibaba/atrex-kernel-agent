# FlyDSL SageAttention Step-by-Step Optimization Practice (MI308X gfx942)

Applicability: backend: flydsl; hardware: amd; topic: reference

This document records the complete process of implementing and progressively optimizing the SageAttention (quantized Flash Attention) kernel from scratch on the AMD MI308X (CDNA3, gfx942, 80 CUs) using FlyDSL.

Final performance: S=8192 achieves **191.9 TFLOPS**, S=32768 achieves **199.1 TFLOPS** (42.7% peak). Compared to the Gluon reference implementation (62-64 TFLOPS), it is **3x faster**.

## Target Architecture

| Parameter | Value |
|------|-----|
| GPU | AMD Instinct MI308X |
| Architecture | CDNA3 (gfx942) |
| CU Count | 80 |
| SIMD / CU | 4 |
| FP8 Peak | ~466 TFLOPS |
| HBM Bandwidth | 5.3 TB/s |
| LDS Capacity/CU | 64 KB |
| Clock Frequency | ~1.42 GHz |

## SageAttention Algorithm

SageAttention is a quantized Flash Attention variant that reduces computation and bandwidth requirements through mixed precision:

```
input: Q, K [B, S, H, D] INT8 quantization, per-block descale
 V [B, S, H, D] FP8 (E4M3FN) quantization, per-head descale

GEMM1 (QK): mfma_i32_32x32x16_i8 → INT32 → FP32 → online softmax
GEMM2 (PV): mfma_f32_32x32x16_fp8_fp8 -> FP32
output: O [B, S, H, D] BF16
```

**Key Features**:
- INT8 QK GEMM throughput is 2x that of FP16
- FP8 PV GEMM improves bandwidth efficiency
- Online softmax (log2-based) avoids traversing KV twice

## Kernel Architecture Design

### Tile Configuration

| Parameter | Value | Reason |
|------|-----|------|
| BLOCK_M | 128 | Q-tile size, matching 4 waves × 32 rows/wave |
| BLOCK_N | 64 | KV-tile size, balancing LDS usage and loop count |
| BLOCK_SIZE | 256 | 4 waves (64 threads/wave) |
| K_STRIDE | 128 (= HEAD_DIM) | Row stride of K in LDS |
| VT_STRIDE | 72 (= BLOCK_N + 8) | Row stride of V^T in LDS, +8 alignment eliminates bank conflict |

### Data Layout

- **Q**: BSHD flattened to 1D, with 32 rows per wave preloaded into registers as the B operand and held for the entire kernel lifetime
- **K**: BSHD → LDS [BLOCK_N, HEAD_DIM], with XOR swizzle to eliminate bank conflict
- **V**: Pre-transposed to [B, H, D, S] on the Python side, stored in LDS as V^T [HEAD_DIM, BLOCK_N], supporting vectorized ds_write_b128

### Kernel Flow

```
Phase 0: Setup
 ├── raw pointers
 ├── block_id -> batch, head, q_tile (q_tile-major column)
 ├── load Q i64 B-operand packs (entire kernel register)
 └── load descale factors (Q scale sm_scale * log2(e))

Phase 1: prefetch KV tile LDS -> barrier

Phase 2: Main KV Loop ( [m_old, l_old, buf_id, o_acc_0..3])
 ├── 2a. next tile globalload (non-blocking VMEM)
 ├── 2b. GEMM1 (QK): INT8 MFMA, prefetch depth=2
 │ sched_dsrd(2) + sched_mfma(2)
 │ sched_barrier(0) QK MFMA
 ├── 2c. Row-Max: 32 i32->f32, scaling max, XOR shuffle lane reduction
 ├── 2d. scaling + correction factor, rescale o_accs[0]
  ├── 2e. P-Packing: FMA(s, qk_scale, -m_new) → exp2 → cvt_pk_fp8_f32
 │ local_sum
 ├── 2f. GEMM2 (PV): load V -> MFMA loop, sched_mfma(4)
 │ o_accs[1..3] sub-tile setup rescale
 └── 2g. lane sum, barrier, storeprefetch K/V LDS, barrier

Phase 3: Output Writeback
  ├── O = O_acc / l_final * v_scale
 ├── Shift-based F32->BF16 (bit, conversion)
 └── MFMA32 outputregistermapping + guarded store
```

## Optimization Journey (V1 → V27)

### V1-V3: Baseline Implementation (Functionally Correct, Unoptimized)

- V1: Basic functional implementation, single buffering
- V2: K/V double-buffered LDS loading
- V3: Software-pipelined LDS reads for QK and PV

### V4: Core Arithmetic Optimization → 69.3 TFLOPS

**Four key optimizations committed together:**

1. **`rocdl.exp2` replaces `math.exp2`** — Directly maps to `v_exp_f32`, eliminating the `v_ldexp + v_cmp + v_cndmask` wrapper. VALU reduced by 38%
2. **Deferred row-max scaling** — Take unscaled max first, multiply by `qk_scale` only once after cross-lane reduction. **-31 MUL/iter**
3. **FMA fusion** — `fma(s_f32, qk_scale, -m_new)` replaces `mul + sub`. **-32 VALU/iter**
4. **QK scheduling hint** — `sched_dsrd(2) + sched_mfma(2)` interleaves LDS reads with MFMA**Lesson**: Arithmetic-level optimization (reducing instruction count) is the top priority. Under the same instruction scheduling and memory strategy, fewer instructions means faster execution.

### V5: V Pre-transpose → ~2.5% Improvement

Pre-transpose V on the Python side to [B, H, D, S], changing LDS store from `ds_write_b8` (byte-by-byte scatter) to `ds_write_b128` (16-byte vector write). DS instructions reduced from 86 to 26 (**-70%**).

### V6: PV Scheduling Hint → 73.8 TFLOPS

Added `sched_dsrd + sched_mfma` hint for the PV stage.

### V7: VT_STRIDE Alignment Fix → 131.3 TFLOPS (+78%)

**The largest single-version improvement in this optimization effort.**

Changed V^T LDS row stride from 66 to 72 (BLOCK_N + 8) to ensure `ds_read_b64` (FP8 MFMA B operand) meets the 8-byte alignment requirement.

**Before fix**: 16.7M bank conflicts, 10.5M LDS wait cycles.
**After fix**: Bank conflicts essentially eliminated.

**Lesson**: LDS stride alignment is an invisible performance killer on CDNA3. `ds_read_b64` requires 8-byte alignment, and `ds_read_b128` requires 16-byte alignment. Misalignment does not cause errors, but bank conflicts will cause performance to plummet. **Always use padding to ensure alignment**.

### V8: P Value Pre-packing → 133.9 TFLOPS (+2%)

Pack all softmax P values into FP8 i64 in one batch before the PV MFMA loop. Separating packing from MFMA computation allows the compiler to schedule better.

**Lesson**: The compiler prefers the "dense VALU blocks + dense MFMA blocks" pattern. Interleaving VALU and MFMA actually hinders the compiler's global optimization.

### V9-V11: Micro-optimizations (Code Structure Improvements)

- V9: Deferred O rescaling — o_accs[1..3] are rescaled during PV sub-tile 0 setup, overlapping VALU and MFMA pipelines
- V10: Merged row-max and f32 value extraction into a single pass
- V11: Increased QK prefetch depth from 1 to 2

### V12: Single-buffer V → 168 TFLOPS (+25%)

**The second-largest performance leap.**

Changed V from double-buffered to single-buffered, reducing LDS from 34816B to 25600B, increasing occupancy from 1 wave/SIMD to 2 waves/SIMD.

**Lesson**: In a compute-bound kernel, occupancy matters more than pipeline overlap. Double buffering hides memory latency, but consuming too much LDS leads to insufficient occupancy, which in turn prevents the MFMA pipeline from being filled by instructions from other waves.

### V13: Single-buffer K+V → 186.1 TFLOPS (+11%)

**The third-largest performance leap.**

Also changed K to single-buffered, reducing total LDS to 17408B (64KB / 17408 = 3.76, floor = 3 workgroups/CU). Occupancy increased from 2 to 3 waves/SIMD.

```
LDS compute:
  K: BLOCK_N × HEAD_DIM = 64 × 128 = 8192 bytes
  V: HEAD_DIM × VT_STRIDE = 128 × 72 = 9216 bytes
 : 17408 bytes -> 64KB / 17408 = 3 workgroups/CU -> 3 waves/SIMD
```

**Lesson**: Single buffering + barrier synchronization introduces global sync overhead, but 3-wave occupancy fully compensates for this cost. For MFMA-heavy kernels, **pursuing maximum occupancy is the most effective strategy**.

### V14-V27: Fine-tuning Phase (186 → 192 TFLOPS)

- V14: Removed unnecessary sched_barrier for PV (+0.3%)
- V15: LLVM flag `lsr-drop-solution=True` (Loop Strength Reduction optimization)
- V17: PV sub-tile preloads all V data, inner loop becomes pure MFMA
- V21: `waves_per_eu=3` matches actual LDS-limited occupancy
- V23: LLVM flags `amdgpu-early-inline-all=True`, `misched-postra-direction=2`
- V25: Q-tile-major block ordering improves L2 cache locality
- V27: Added `sched_mfma(4)` hint for PV MFMA loop

## Dead Ends and Lessons Learned

All of the following attempts were made from a ~191 TFLOPS baseline and were all reverted:

### 1. Excessive Use of Scheduling Hints

| Attempt | Result | Analysis |
|------|------|------|
| `sched_dsrd(4) + sched_mfma(4)` for QK | No improvement | (2,2) is already the optimal balance point |
| `sched_dsrd(1) + sched_mfma(1)` | No improvement | Hint too weak |
| Removing `sched_barrier(0)` after QK | **179.8 (-6%)** | Compiler interleaves QK MFMA with softmax VALU, breaking data dependencies |
| Adding `sched_barrier(0)` to PV stage | **190.8 (-0.5%)** | Overly restricts compiler optimization space |

**Lesson**: `sched_barrier(0)` is **critically essential** at the QK→softmax transition point, preventing the compiler from reordering across GEMM boundaries. But it is not needed inside the PV inner loop — the compiler handles this well on its own. The optimal strategy for scheduling hints is: "set barriers at critical points, give the compiler freedom elsewhere."

### 2. Wave Priority and Arbitration Control

| Attempt | Result | Analysis |
|------|------|------|
| `s_setprio(2)` / `s_setprio(0)` | **188.2 (-1.8%)** | Lowering softmax VALU priority also lowers priority of subsequent instructions |
| `disable_xdl_arb_stall` | **187.9 (-2.1%)** | XDL arbitration mechanism actually helps this kernel |
| `iglp_opt(0)` | 191.6 (neutral) | Ineffective but harmless |

**Lesson**: In high-occupancy (3 waves) scenarios, the hardware's default arbitration policy is already near-optimal. Manual intervention in wave priority can easily introduce negative side effects.

### 3. Instruction Interleaving Strategies

| Attempt | Result | Analysis |
|------|------|------|
| Fused P+PV (softmax and PV MFMA interleaved) | **174.8 (-9%)** | **Major regression** |
| PV dc-outer nesting | No improvement | Changing loop nesting order was ineffective |
| V preload interspersed with P-packing VALU | No improvement | |**Experience**: The AMD compiler **strongly prefers the "compute-dense block" pattern**: complete all VALU (softmax P computation) first, then perform dense MFMA (PV accumulation). Manually interleaving VALU and MFMA hinders the compiler's global scheduling optimization. This is counterintuitive—you might think manual interleaving fills MFMA pipeline bubbles, but in reality the compiler already handles this with multiple waves.

### 4. Tile Configuration Changes

| Attempt | Result | Analysis |
|------|------|------|
| BLOCK_M=256 | Not feasible | Q register pressure doubled, VGPR spill |
| waves_per_eu=4 | No improvement | LDS limits actual occupancy to 3 |
| QK prefetch depth=3 | **189.6 (-1.2%)** | Extra register consumption caused spill |

**Experience**: With VGPR at 168, any change that increases register pressure will trigger a spill. The optimization space has been locked by the ISA instruction count—219 VALU/iter and 32 MFMA/iter are the algorithmically determined lower bound.

## Performance Trajectory Summary

| Version | S=8192 TFLOPS | Key Change | Improvement |
|------|------|------|------|
| V1 | (Not tested) | Functional baseline | — |
| V4 | 69.3 | rocdl.exp2 + delayed max + FMA fusion | Baseline |
| V6 | 73.8 | PV scheduling hints | +6.5% |
| **V7** | **131.3** | **VT_STRIDE alignment** | **+78%** |
| V8 | 133.9 | Pre-packed P values | +2% |
| **V12** | **168.0** | **Single-buffer V, occupancy=2** | **+25%** |
| **V13** | **186.1** | **Single-buffer K+V, occupancy=3** | **+11%** |
| V14 | 186.4 | Removed PV redundant barrier | +0.2% |
| V27 | **191.9** | Cumulative micro-optimizations | +3% |

### Multi-Size Final Performance (V27)

| Config | Time | TFLOPS |
|--------|------|--------|
| B=1 S=8192 H=32 D=128 | 5.72ms | 191.9 |
| B=1 S=16384 H=32 D=128 | 22.41ms | 195.4 |
| B=1 S=32768 H=32 D=128 | 87.78ms | 199.1 |
| B=2 S=8192 H=32 D=128 | 11.47ms | 191.5 |
| B=2 S=16384 H=32 D=128 | 44.63ms | 195.7 |

### vs Gluon Reference Implementation

| Config | FlyDSL V27 | Gluon (per_thread) | Ratio |
|--------|-----------|-------------------|------|
| B=1 S=8192 | 191.9 | 61.2 | **3.1x** |
| B=1 S=16384 | 195.4 | 63.8 | **3.1x** |
| B=1 S=32768 | 199.1 | 63.3 | **3.1x** |

Note: The Gluon implementation uses the `gl.amd.cdna4.*` API (CDNA4-specific) and has not been adapted for CDNA3, so its performance is not directly comparable.

## Compiler Flags

```python
compile_hints = {
    "fast_fp_math": True,
    "unsafe_fp_math": True,
    "llvm_options": {
 "enable-post-misched": True, # MachineScheduler
 "lsr-drop-solution": True, # LSR: compilation
 "amdgpu-early-inline-all": True, # function
 "misched-postra-direction": 2, # RA schedulinguse bottom-up
    },
}

# Function attributes
"rocdl.waves_per_eu": 3 # LDS occupancy
"denormal-fp-math-f32": "preserve-sign" # flush denormals (accuracy)
"no-nans-fp-math": "true"
"unsafe-fp-math": "true"
```

## ISA-Level Statistics (Confirmed via rocprofv3)

| Metric | Value |
|------|-----|
| VGPR Usage | 168 |
| VALU / iter | 219 |
| MFMA / iter | 32 |
| Waves / SIMD | 3 |
| LDS / workgroup | 17408 bytes |
| Asymptotic Upper Bound (S→∞) | ~199 TFLOPS |

## Core Optimization Principles Summary

### 1. Occupancy Takes Priority Over Pipeline Overlap
The occupancy improvement from single-buffer + barrier synchronization (1→3 waves) far outweighs the pipeline overlap benefits of double-buffering. For MFMA-heavy kernels, **LDS usage is the top-priority optimization target**.

### 2. LDS Stride Alignment is a Hidden Performance Killer
`ds_read_b64` requires 8-byte alignment, and `ds_read_b128` requires 16-byte alignment. Misalignment does not report an error but causes a sharp increase in bank conflicts. **Use padding to ensure stride alignment**. Changing VT_STRIDE from 66→72 delivered a +78% improvement.

### 3. The Compiler Prefers the "Compute-Dense Block" Pattern
Do not manually interleave VALU and MFMA. The AMD compiler optimizes best with the "complete all VALU first, then dense MFMA" pattern. Multi-wave occupancy naturally fills MFMA pipeline bubbles.

### 4. Arithmetic Optimizations Yield Real Gains
- `rocdl.exp2` replacing `math.exp2` (-38% VALU)
- Delayed row-max scaling (-31 MUL/iter)
- FMA fusion (-32 VALU/iter)
- Every eliminated instruction is real savings.

### 5. Proper Use of Scheduling Hints
- `sched_barrier(0)` is **required** at GEMM boundaries (QK→softmax)
- `sched_dsrd(N) + sched_mfma(N)` is effective within MFMA loops, but the optimal N requires experimentation
- Inner loops do not need barriers—give the compiler freedom
- Overusing scheduling hints actually restricts the compiler's optimization space

### 6. Hardware Arbitration Mechanisms Are Usually Correct
`s_setprio`, `disable_xdl_arb_stall`, and other hardware controls rarely have positive effects in high-occupancy scenarios. The default policy is nearly optimal.

### 7. 219 VALU Is the Algorithmic Lower Bound
SageAttention's INT8/FP8 quantization + online softmax dictates that each iteration requires at least 219 VALU and 32 MFMA. Without changing the algorithm, ~199 TFLOPS is the theoretical asymptotic upper limit. Breaking past 200+ requires algorithm-level changes (such as reducing softmax precision or changing the quantization strategy).

## Related Documents

- **FlyDSL Programming Guide**: [flydsl-programming-guide.md](../../../../common/ref-docs/flydsl/flydsl-programming-guide.md)
- **FlyDSL API Reference**: [flydsl-ref-api.md](../../../../common/ref-docs/flydsl/flydsl-ref-api.md)
- **FlyDSL Kernel Authoring**: [flydsl-ref-kernel-authoring.md](../../../../common/ref-docs/flydsl/flydsl-ref-kernel-authoring.md)
- **AMD MFMA Instructions**: [amd-mfma-matrix-cores.md](../../../../common/ref-docs/amd-mfma-matrix-cores.md)
- **LDS Bank Conflict Optimization**: [lds-bank-conflict-optimization.md](../../../../common/kernel-opt/lds-bank-conflict-optimization.md)
- **Instruction Scheduling Control**: [instruction-scheduling.md](../../../../common/kernel-opt/hands-on/instruction-scheduling.md)
- **Occupancy Optimization**: [occupancy-optimization.md](../../../../common/kernel-opt/occupancy-optimization.md)
- **Fused MoE BF16 Optimization (same document series)**: [cdna3-fused-moe-bf16-optimization.md](cdna3-fused-moe-bf16-optimization.md)
- **Reference Implementation**: [sage_attn_flydsl.py](../../../../../../reference-kernels/amd/cdna/flydsl/FlyDSL/sage_attn_flydsl.py)
