# NVIDIA Blackwell Tensor Core Analysis (Part 1)

A preliminary analysis of Blackwell (B200/SM100) Tensor Core features based on PTX ISA 8.7 documentation, covering compatibility, instruction set, memory layout, block scaling, and synchronization.


**Last updated**: 2026-06-30

---

## 1. Key Takeaways

- B200 is **not backward-compatible** with H100 Tensor Core code; separate development is required
- B200 Tensor Core code will likely **not be forward-compatible** with the next generation (Rubin)
- B200 Tensor Core programming may be **more complex** than H100
- B200 adds **256 KB of Tensor Memory** per SM
- B200 Tensor Core instructions support cluster size of 2
- In addition to microscaling formats, NVIDIA introduces its own NVFP4

---

## 2. Compute Capability

- B200: SM100 / SM101
- RTX 50-series (consumer): SM120

---

## 3. Compatibility

In PTX ISA, instructions with an "a" suffix on the compute capability version are exclusive to that version. `sm_90` instructions cannot run on `sm_80` but can run on `sm_100`; however, `sm_90a` instructions cannot run on any other version.

Hopper WGMMA Tensor Core instructions require `sm_90a` only — code using this feature must be redeveloped for Blackwell. Blackwell's 5th-gen Tensor Core instructions require `sm_100a` only (not even usable on the 5090), and will likely be unsupported on next-generation architectures.

| Hardware | WMMA (FP16) | WMMA (FP6/FP4) | WGMMA | 5th-Gen TC |
|----------|-------------|-----------------|-------|------------|
| H100 (SM_90) | Y | N | Y | N |
| B200 (SM_100) | Y | N | N | Y |
| 5090 (SM_120) | Y | Y | N | N |

FlashAttention-2 runs on A100, H100, and B200, but performance optimization requires FlashAttention-3 for H100 and FlashAttention-4 for B200. The 5090 needs a patch on FA2. The next architecture will likely require yet another version (FlashAttention-5).

---

## 4. 5th Generation Tensor Core Instructions

Generations 1–3 are called WMMA, generation 4 is WGMMA, and generation 5 is simply referred to as "5th gen."

### 4.1 CTA Pair

Two CTAs within a cluster whose `cta_rank / 2` values are equal form a CTA Pair (e.g., (0,1), (2,3)...). This enables two SM cores' Tensor Cores to collaboratively compute larger matrices.

### 4.2 Tensor Memory

Each SM has a 256 KB Tensor Memory block organized as 128 rows × 512 columns with 4-byte element size. Its primary purpose is storing the D matrix (accumulator), sparse metadata, and block scaling factors.

**Allocation and deallocation**: Unlike shared memory (primarily software-static allocation), Tensor Memory is allocated by hardware instructions (`alloc`/`dealloc`).

```
tcgen05.alloc.cta_group.sync.aligned{.shared::cta}.b32 [dst], nCols;
tcgen05.dealloc.cta_group.sync.aligned.b32 taddr, nCols;
tcgen05.relinquish_alloc_permit.cta_group.sync.aligned;
.cta_group = { .cta_group::1, .cta_group::2 }
```

- `alloc` allocates by columns only: [32, 64, 128, 256, 512] columns × 128 rows; the resulting pointer is written to shared memory at `[dst]`
- `alloc` is synchronous — it blocks until allocation succeeds (waits indefinitely if TMEM is exhausted)
- Within the same CTA, allocation size cannot increase after the first allocation
- `relinquish_alloc_permit` informs hardware that this CTA will not issue further `alloc` instructions (enabling earlier scheduling of the next CTA for concurrent execution)

### 4.3 Tensor Memory Data Paths

Tensor Memory supports LD/ST access to/from the register file, and unidirectional copy from shared memory to Tensor Memory.

```
// LD
tcgen05.ld.sync.aligned.shape1.num{.pack}.b32     r, [taddr];
// ST
tcgen05.st.sync.aligned.shape1.num{.unpack}.b32   [taddr], r;
// wait
tcgen05.wait_operation.sync.aligned;
.wait_operation = { .wait::ld, .wait::st }
```

WarpGroup (introduced in Hopper): each SM has 4 warp schedulers with static warp-to-scheduler mapping (`warp_scheduler_id = warp_id % 4`).

For Tensor Memory LD/ST instructions, each warp can only access 1/4 of the TMEM; a full warpgroup is required to access the entire TMEM. Traditional LD/ST instructions rely on SASS read/write scoreboards + wait bitmask control fields for synchronization; Tensor Memory LD/ST requires explicit `wait` instructions. Each LD/ST instruction reads/writes a fixed-size block — for example, `ld.16x128b` loads 16 rows × 4 columns from TMEM to the register file per warp per instruction.

### 4.4 SharedMemory → TensorMemory Copy

```
tcgen05.cp.cta_group.shape{.multicast}{.dst_fmt.src_fmt} [taddr], s-desc;
.cta_group = { .cta_group::1, .cta_group::2 }
.src_fmt = { .b6x16_p32 , .b4x16_p64 }
.dst_fmt = { .b8x16 }
.shape = { .128x256b, .4x256b, .128x128b, .64x128b**, .32x128b*** }
.multicast = { .warpx2::02_13** , .warpx2::01_23**, .warpx4*** }
```

This is a unidirectional copy from shared memory to Tensor Memory, with optional 4-bit/6-bit → 8-bit decompression support.

---

## 5. 5th Generation MMA Operations

```
// 1. Floating-point type without block scaling:
tcgen05.mma.cta_group.kind  [d-tmem], a-desc, b-desc, idesc, { disable-output-lane }, enable-input-d {, scale-input-d};
tcgen05.mma.cta_group.kind  [d-tmem], [a-tmem], b-desc, idesc, { disable-output-lane }, enable-input-d {, scale-input-d};
.kind = { .kind::f16, .kind::tf32, .kind::f8f6f4 }
.cta_group = { .cta_group::1, .cta_group::2 }

// 2. Floating-point type with block scaling:
tcgen05.mma.cta_group.kind.block_scale{.scale_vec_size}  [d-tmem], a-desc, b-desc, idesc, [scale-A-tmem], [scale-B-tmem], enable-input-d;
tcgen05.mma.cta_group.kind.block_scale{.scale_vec_size}  [d-tmem], [a-tmem], b-desc, idesc, [scale-A-tmem], [scale-B-tmem], enable-input-d;
.kind = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
.cta_group = { .cta_group::1, .cta_group::2 }
.scale_vec_size = { .scale_vec::1X, .scale_vec::2X, .scale_vec::4X }
```

Supported data types:
- `.kind::f16`: FP16 and BF16
- `.kind::tf32`: TF32
- `.kind::f8f6f4`: arbitrary FP8/FP6/FP4 combinations
- `.kind::i8`: signed/unsigned INT8
- `.kind::mxf8f6f4` / `.kind::mxf4`: MX floating point
- `.kind::mxf4nvf4`: MXFP4 + NVIDIA custom 4-bit type (shared scaling factor)

Notable MNK dimensions:
- **mxf8f6f4**: A/B matrices use e4m3/e5m2/e2m3/e3m2/e2m1 + ue8m0 in any combination; D matrix is FP32; M={128,256}, N={16,32,...,256}, K=32
- **mxfp4**: A/B use e2m1 + ue8m0; D is FP32; M={128,256}, N={16,32,...,256}, K=64

---

## 6. Packing Formats in Tensor/Shared Memory

Tensor Core output elements are always 32-bit. If the output is FP16, each element's upper 16 bits are zero with the lower 16 bits containing the FP16 value.

For F8F6F4 mixed-input types with input in Tensor Memory, each element occupies a separate 8-bit slot:
- E2M1: `[00SEEM00]`
- E3M2: `[00SEEEMM]`
- E2M3: `[00SEEMMM]`

If input is in Shared Memory:
- FP4: 16×FP4 grouped + 8-byte padding
- FP6: 16×FP6 grouped + 4-byte padding

For `.kind::mxf4` and `.kind::mxf4nvf4`, both TMEM and SMEM pack 2 FP4 elements per byte.

---

## 7. Matmul Tensor Memory Layout

Since Tensor Core output element size is always 32-bit regardless of input type (TF32 or FP4), the D matrix has only 8 possible layouts in Tensor Memory.

For Layout A: two SMs concatenate along the M dimension to form a larger A matrix; the B matrix data is shared. Using TMA multicast to transfer the B matrix saves 50% bandwidth.

**Design question**: What is the inherent benefit of Tensor Core clustering? Can it improve B-matrix shared-memory-to-Tensor-Core bandwidth?

---

## 8. Block Scaling

Microscaling format tensors have corresponding BlockScaling tensors. On Blackwell, both A and B matrix BlockScaling tensors must reside in Tensor Memory in column-major layout.

Given that the microscaling block size is 32 (1 scale element covers K=32): mxf8f6f4 has K=32, while mxf4/mxf4nvf4 has K=64. The `scale_vec` parameter represents the number of scale elements along the K dimension per MMA instruction. NVFP4 uses `scale_vec=4` with block size=16.

Since Tensor Memory storage granularity is 32-bit but scaling factors are 8-bit, for `scale_vec` values less than 4, the MMA instruction requires a parameter specifying which bytes within a column to use.

---

## 9. Sparse Matrices

All Blackwell data types support structured sparsity on the A matrix. Notably, FP4 uses 4:8 sparsity (in pairs) — grouping 2 FP4 elements together, effectively implementing 8-bit 2:4 sparsity.

---

## 10. Tensor Core Collector Buffer

The Tensor Core contains an internal collector buffer that can cache A or B matrix data. Collector buffer usage requires manual programmer control and must be specified at compile time:

```
MMA.a.fill A[0], B[0]   <- load A[0]
MMA.a.use A[0], B[1]
MMA.a.use A[0], B[2]
MMA.a.lastuse A[0], B[3]
MMA.a.fill A[1], B[0]   <- load A[1]
MMA.a.use A[1], B[1]
MMA.a.use A[1], B[2]
MMA.a.lastuse A[1], B[3]
```

---

## 11. Memory Consistency Model

Certain Tensor Core instructions are asynchronous and require explicit programmer-managed synchronization (see PTX ISA 9.7.16.5.4):

1. **Pipelined tcgen05 instructions**: In certain cases, instruction ordering is naturally preserved. For example, two MMA instructions with identical MNK shape and accumulator address execute in issue order.
2. **mbarrier-based completion mechanism**: MMA/cp/shift instructions support TMA-style mbarrier synchronization with cross-thread synchronization capability.
3. **tcgen05.wait instruction-based completion mechanism**: Tensor Memory LD/ST instructions use dedicated wait instructions for synchronization.


## Related

- [Comprehensive Guide to NVIDIA Blackwell Architecture](blackwell-architecture-comprehensive-guide.md)
- [GPGPU Architecture: Blackwell Instruction Analysis](blackwell-architecture-instruction-analysis.md)
- [Blackwell GPGPU Architecture New Features Overview](blackwell-gpgpu-new-features-overview.md)
- [NVIDIA Blackwell Tensor Core Analysis (Part 2): B300](blackwell-tensor-core-analysis-b300.md)
- [Blackwell Ultra (B300): NVIDIA AI Chip Evolution and Roadmap](blackwell-ultra-b300-chip-evolution.md)
- [Tensor Core from Volta to Blackwell](../../common/tensor-core-volta-to-blackwell.md)
- [PTX Programming Model and Basics](../../common/ptx/ptx-programming-model.md)
- [PTX Core Instruction Set](../../common/ptx/ptx-instruction-set.md)
