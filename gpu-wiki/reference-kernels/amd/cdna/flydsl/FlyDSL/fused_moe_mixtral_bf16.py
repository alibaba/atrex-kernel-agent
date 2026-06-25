"""Fused MoE kernel for AMD MI308X (gfx942) using FlyDSL.

V18: In-kernel X loading for S1 via sorted_token_ids indirection.
- V17: Fused atomic f32 accumulation in S2 epilogue (181 TFLOPS at tc=4096).
  Remaining E2E overhead: ~863us from host-side ops (sort, pre-gather, alloc).
- V18: Eliminate sorted_x pre-gather by loading X in-kernel. S1 kernel
  receives original hidden_states + sorted_token_ids, precomputes token
  indices per thread once, then uses them for indirect A loads.
- Stage 1: hidden_states[sorted_token_ids] @ w1^T (gate + up) + SiLU → activated
- Stage 2: activated @ w2^T → output (fused weight * atomic f32 add)
- MFMA 16x16x16 BF16 (mfma_f32_16x16x16bf16_1k)
- tile_n=128, tile_k=128
"""

import functools
from contextlib import contextmanager
import torch
import numpy as np
import triton
import triton.language as tl

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T
from flydsl.expr import range_constexpr, arith, vector, gpu, rocdl, buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf, memref, fly


@contextmanager
def _if_then(if_op):
    """Context manager for scf.IfOp then-block (early-exit guard pattern)."""
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            scf.YieldOp([])
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.compiler.protocol import fly_values


# ============================================================
# Helpers
# ============================================================
def swizzle_xor16(row, col_in_bytes, k_blocks16):
    return col_in_bytes ^ ((row % k_blocks16) * 16)


class GTensor:
    """Global memory tensor with buffer_ops access."""
    def __init__(self, memref_val, dtype, shape, stride=None):
        self.rsrc = buffer_ops.create_buffer_resource(memref_val, max_size=True)
        self.dtype = dtype
        self.shape = shape
        if stride is None:
            self.stride = tuple((np.cumprod(shape[::-1])[::-1].tolist() + [1])[1:])
        else:
            self.stride = stride

    def _offset(self, idxs):
        off = fx.Index(0)
        for i in range_constexpr(len(idxs)):
            off = off + idxs[i] * self.stride[i]
        return off

    def load(self, idxs, vec_size=1):
        off = self._offset(idxs)
        return buffer_ops.buffer_load(self.rsrc, off, vec_width=vec_size, dtype=self.dtype)

    def vec_load(self, idxs, vec_size):
        return self.load(idxs, vec_size=vec_size)

    def store(self, idxs, value):
        off = self._offset(idxs)
        buffer_ops.buffer_store(value, self.rsrc, off)

    def vec_store(self, idxs, value, vec_size):
        off = self._offset(idxs)
        buffer_ops.buffer_store(value, self.rsrc, off)


class STensor:
    """Shared memory (LDS) tensor."""
    def __init__(self, smem_ptr, dtype, shape, stride=None):
        self.memptr = smem_ptr.get()
        self.dtype = dtype
        self.shape = shape
        if stride is None:
            self.stride = tuple((np.cumprod(shape[::-1])[::-1].tolist() + [1])[1:])
        else:
            self.stride = stride

    def _offset(self, idxs):
        off = fx.Index(0)
        for i in range_constexpr(len(idxs)):
            off = off + idxs[i] * self.stride[i]
        return off

    def vec_load(self, idxs, vec_size):
        off = self._offset(idxs)
        vec_t = T.vec(vec_size, self.dtype)
        return vector.load_op(vec_t, self.memptr, [off])

    def vec_store(self, idxs, value, vec_size):
        off = self._offset(idxs)
        vector.store(value, self.memptr, [off], alignment=16)

    def scalar_store(self, idxs, value):
        off = self._offset(idxs)
        vec_t = T.vec(1, self.dtype)
        vec = vector.from_elements(vec_t, [value])
        vector.store(vec, self.memptr, [off], alignment=16)


# ============================================================
# Weight preshuffle for direct MFMA-aligned GMEM loads
# ============================================================
def preshuffle_weight(W):
    """Rearrange weight [N, K] bf16 for coalesced per-lane GMEM loads.

    Output layout: [N//16, K//32, 4, 16, 8] flattened to 1D.
    Lane l loads 16 contiguous bytes from [..., l//16, l%16, :8].
    K-ordering matches our A LDS layout (stride-8 per lane group).
    """
    N, K = W.shape
    assert N % 16 == 0 and K % 32 == 0, f"N={N} must be %16, K={K} must be %32"
    # [N, K] → [N/16, 16, K/32, 4, 8]
    #           n0    n_lane k0   k_grp k_elem
    W = W.reshape(N // 16, 16, K // 32, 4, 8)
    # Permute to [n0, k0, k_grp, n_lane, k_elem]
    W = W.permute(0, 2, 3, 1, 4).contiguous()
    return W.reshape(-1).contiguous()


# ============================================================
# Host-side MoE token sorting
# ============================================================
@triton.jit
def _moe_fused_sort_kernel(
    expert_ids_ptr, token_idx_ptr, weights_ptr,
    out_token_ids_ptr, out_weights_ptr,
    out_tile_expert_ids_ptr, out_num_m_tiles_ptr,
    N,
    NUM_EXPERTS: tl.constexpr, TILE_M: tl.constexpr,
    MAX_TPE: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    """Single-block fused sort: count + pad + scatter + tile expert_ids."""
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    expert = tl.load(expert_ids_ptr + offs, mask=mask, other=-1).to(tl.int32)
    token_id = tl.load(token_idx_ptr + offs, mask=mask, other=0)
    weight = tl.load(weights_ptr + offs, mask=mask, other=0.0)

    running_dst = tl.zeros([1], dtype=tl.int32)
    running_tiles = tl.zeros([1], dtype=tl.int32)
    tile_offs = tl.arange(0, MAX_TPE)

    for e in tl.static_range(NUM_EXPERTS):
        e_mask = (expert == e) & mask
        count = tl.sum(e_mask.to(tl.int32))
        padded = ((count + TILE_M - 1) // TILE_M) * TILE_M
        padded = tl.where(count > 0, padded, 0)
        n_tiles = padded // TILE_M

        local_pos = tl.cumsum(e_mask.to(tl.int32), axis=0) - 1
        dst_scalar = tl.sum(running_dst)
        dst = dst_scalar + local_pos
        tl.store(out_token_ids_ptr + dst, token_id, mask=e_mask)
        tl.store(out_weights_ptr + dst, weight, mask=e_mask)

        tiles_scalar = tl.sum(running_tiles)
        tile_mask = tile_offs < n_tiles
        tl.store(out_tile_expert_ids_ptr + tiles_scalar + tile_offs,
                 tl.full([MAX_TPE], e, dtype=tl.int32), mask=tile_mask)
        running_dst = running_dst + padded
        running_tiles = running_tiles + n_tiles

    tl.store(out_num_m_tiles_ptr, tl.sum(running_tiles))


_flat_token_idx_cache = {}
_sort_workspace = {}


def _get_sort_workspace(total_slots, num_experts, tile_m, device):
    key = (total_slots, num_experts, tile_m, device)
    if key not in _sort_workspace:
        max_padded = total_slots + num_experts * (tile_m - 1)
        max_tiles = max_padded // tile_m
        _sort_workspace[key] = {
            'out_token_ids': torch.zeros(max_padded, dtype=torch.int32, device=device),
            'out_weights': torch.zeros(max_padded, dtype=torch.float32, device=device),
            'out_tile_expert_ids': torch.zeros(max_tiles, dtype=torch.int32, device=device),
            'out_num_m_tiles': torch.zeros(1, dtype=torch.int32, device=device),
        }
    return _sort_workspace[key]


def moe_sort_tokens(topk_ids, topk_weights, num_experts, tile_m):
    """Sort tokens by expert ID for grouped GEMM.

    V27: Fused single-block Triton sort for N<=1024.
    Returns (sorted_ids, sorted_weights, tile_expert_ids, num_m_tiles_tensor).
    num_m_tiles_tensor is a GPU int32[1] — caller should defer .item() sync
    to overlap with other GPU work (e.g. output zeroing).
    """
    token_count = topk_ids.shape[0]
    top_k = topk_ids.shape[1]
    device = topk_ids.device
    total_slots = token_count * top_k
    max_padded = total_slots + num_experts * (tile_m - 1)
    max_tiles = max_padded // tile_m

    flat_expert_ids = topk_ids.reshape(-1)
    flat_weights = topk_weights.reshape(-1)

    cache_key = (token_count, top_k, device)
    if cache_key not in _flat_token_idx_cache:
        _flat_token_idx_cache[cache_key] = torch.arange(
            token_count, device=device, dtype=torch.int32
        ).unsqueeze(1).expand(-1, top_k).reshape(-1).contiguous()
    flat_token_idx = _flat_token_idx_cache[cache_key]

    if total_slots <= 1024:
        ws = _get_sort_workspace(total_slots, num_experts, tile_m, device)
        out_token_ids = ws['out_token_ids']
        out_weights = ws['out_weights']
        out_tile_expert_ids = ws['out_tile_expert_ids']
        out_num_m_tiles = ws['out_num_m_tiles']
        out_weights[:max_padded].zero_()

        max_tpe_raw = total_slots // tile_m + 1
        MAX_TPE = 1
        while MAX_TPE < max_tpe_raw:
            MAX_TPE *= 2

        _moe_fused_sort_kernel[(1,)](
            flat_expert_ids, flat_token_idx, flat_weights,
            out_token_ids, out_weights,
            out_tile_expert_ids, out_num_m_tiles,
            total_slots,
            NUM_EXPERTS=num_experts, TILE_M=tile_m,
            MAX_TPE=MAX_TPE, BLOCK_SIZE=1024,
        )
        # Return GPU tensor for deferred sync
        return out_token_ids[:max_padded], out_weights[:max_padded], \
               out_tile_expert_ids[:max_tiles], out_num_m_tiles

    else:
        sort_idx = flat_expert_ids.long().argsort(stable=True)
        sorted_token_idx = flat_token_idx[sort_idx]
        sorted_weights_flat = flat_weights[sort_idx]

        counts_t = torch.bincount(flat_expert_ids.long(), minlength=num_experts)
        counts = counts_t.cpu().tolist()
        padded_counts = [((c + tile_m - 1) // tile_m) * tile_m if c > 0 else 0 for c in counts]
        total_padded = sum(padded_counts)

        out_token_ids = torch.zeros(total_padded, dtype=torch.int32, device=device)
        out_weights = torch.zeros(total_padded, dtype=torch.float32, device=device)
        m_tile_expert_ids = []
        src_offset = 0
        dst_offset = 0
        for expert_id in range(num_experts):
            c = counts[expert_id]
            pc = padded_counts[expert_id]
            if c == 0:
                continue
            out_token_ids[dst_offset:dst_offset + c] = sorted_token_idx[src_offset:src_offset + c]
            out_weights[dst_offset:dst_offset + c] = sorted_weights_flat[src_offset:src_offset + c]
            m_tile_expert_ids.extend([expert_id] * (pc // tile_m))
            src_offset += c
            dst_offset += pc

        expert_ids_out = torch.tensor(m_tile_expert_ids, dtype=torch.int32, device=device)
        num_m_tiles = expert_ids_out.shape[0]
        # Return CPU int for num_m_tiles (already synced via .cpu().tolist())
        return out_token_ids, out_weights, expert_ids_out, num_m_tiles


# ============================================================
# Stage 1 Kernel: Grouped GEMM + SiLU fusion
# A through LDS (double-buffer), B from preshuffle GMEM
# ============================================================
@functools.lru_cache(maxsize=64)
def compile_moe_stage1(
    *,
    inter_dim: int,
    hidden_size: int,
    num_experts: int,
    tile_m: int = 64,
    tile_n: int = 128,
    tile_k: int = 128,
):
    GPU_ARCH = get_rocm_arch()
    DTYPE_BYTES = 2
    WARP_SIZE = 64
    LDG_VEC_SIZE = 8

    WMMA_M, WMMA_N, WMMA_K = 16, 16, 16
    WMMA_C_FRAG = 4
    MFMA_PER_WARP_K = 2
    WARP_ATOM_K = WMMA_K * MFMA_PER_WARP_K  # 32

    BLOCK_M_WARPS = 1
    BLOCK_N_WARPS = 4
    BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE  # 256

    WARP_M_STEPS = tile_m // (BLOCK_M_WARPS * WMMA_M)
    WARP_N_STEPS = tile_n // (BLOCK_N_WARPS * WMMA_N)
    WARP_M = WARP_M_STEPS * WMMA_M
    WARP_N = WARP_N_STEPS * WMMA_N
    WARP_K_STEPS = tile_k // WARP_ATOM_K

    # A cooperative loading params
    LDG_X_THREADS = tile_k // LDG_VEC_SIZE
    LDG_REG_COUNT = (tile_m * tile_k) // (LDG_VEC_SIZE * BLOCK_THREADS)

    C_FRAGS_LEN = WARP_M_STEPS * WARP_N_STEPS
    B_FRAGS_LEN = WARP_K_STEPS * WARP_N_STEPS

    STAGES = 2
    BLOCK_K_BYTES = tile_k * DTYPE_BYTES
    BLOCK_K_LOOPS = hidden_size // tile_k

    # B preshuffle constants
    PS_K0_STRIDE = 512  # 4 * 16 * 8 elements per K32 block
    PS_N0_STRIDE = (hidden_size // 32) * PS_K0_STRIDE  # elements per n-block
    PS_K32_PER_TILE = tile_k // 32

    # LDS: only A (no B!) — 32KB for tile_m=64,tile_k=128 → 2 blocks/CU
    allocator = SmemAllocator(None, arch=GPU_ARCH, global_sym_name="smem_s1")
    smem_a_offset = allocator._align(allocator.ptr, 16)
    A_LDS_BYTES = STAGES * tile_m * tile_k * DTYPE_BYTES
    EPILOGUE_BYTES = tile_m * tile_n * DTYPE_BYTES
    A_LDS_BYTES = max(A_LDS_BYTES, EPILOGUE_BYTES)
    allocator.ptr = smem_a_offset + A_LDS_BYTES

    LDG_C_X_THREADS = tile_n // LDG_VEC_SIZE
    LDG_REG_C_COUNT = (tile_m * tile_n) // (LDG_VEC_SIZE * BLOCK_THREADS)

    @flyc.kernel
    def moe_stage1_kernel(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w1_ps: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_num_valid_tiles: fx.Tensor,
    ):
        # V27: Early-exit guard for zero-sync sort
        _guard_bid_m = fx.block_idx.y
        _nv_rsrc = buffer_ops.create_buffer_resource(arg_num_valid_tiles, max_size=True)
        _nv_i32 = buffer_ops.buffer_load(_nv_rsrc, fx.Index(0), vec_width=1, dtype=T.i32)
        _blk_valid = arith.cmpi(arith.CmpIPredicate.ult, _guard_bid_m, _nv_i32)
        _if_blk = scf.IfOp(_blk_valid)
        with _if_then(_if_blk):
            dtype_ = T.bf16
            acc_init = arith.constant_vector(0.0, T.f32x4)

            X_ = GTensor(arg_x, dtype=dtype_, shape=(-1, hidden_size))
            W1_PS = GTensor(arg_w1_ps, dtype=dtype_, shape=(num_experts * 2 * inter_dim * hidden_size,))
            OUT_ = GTensor(arg_out, dtype=dtype_, shape=(-1, inter_dim))
            EXP_ = GTensor(arg_expert_ids, dtype=T.i32, shape=(-1,))
            # V18: buffer resource for sorted_token_ids (in-kernel X loading)
            sorted_tids_rsrc = buffer_ops.create_buffer_resource(arg_sorted_token_ids, max_size=True)

            # LDS: only A
            base_ptr = allocator.get_base()
            smem_a_ptr = SmemPtr(base_ptr, smem_a_offset, dtype_, shape=(STAGES * tile_m * tile_k,))
            as_ = STensor(smem_a_ptr, dtype_, shape=(STAGES, tile_m, tile_k))
            smem_c_ptr = SmemPtr(base_ptr, smem_a_offset, dtype_, shape=(tile_m * tile_n,))
            cs_ = STensor(smem_c_ptr, dtype_, shape=(tile_m, tile_n))

            tid = fx.Int32(fx.thread_idx.x)
            wid = tid // WARP_SIZE
            w_tid = tid % WARP_SIZE

            # Grid swapped: (bn, num_m_tiles) for L2 A-tile reuse
            bid_n = fx.block_idx.x
            bid_m = fx.block_idx.y

            m_offset = fx.Index(bid_m * tile_m)
            n_offset = fx.Index(bid_n * tile_n)
            k_blocks16 = fx.Int32(BLOCK_K_BYTES // 16)

            expert_id = EXP_.load((fx.Index(bid_m),))
            expert_id_idx = arith.index_cast(T.index, expert_id)

            # Preshuffle B: compute per-warp n-block base offsets
            wid_n = wid % BLOCK_N_WARPS
            wid_n_idx = arith.index_cast(T.index, wid_n)
            bid_n_idx = fx.Index(bid_n)

            # V10: each warp handles WARP_N_STEPS n16 blocks (2 with tile_n=128)
            gate_n0 = expert_id_idx * fx.Index(2 * inter_dim // 16) + bid_n_idx * fx.Index(tile_n // 16) + wid_n_idx * fx.Index(WARP_N_STEPS)
            up_n0 = gate_n0 + fx.Index(inter_dim // 16)
            gate_base = gate_n0 * fx.Index(PS_N0_STRIDE)
            up_base = up_n0 * fx.Index(PS_N0_STRIDE)

            # Per-lane fixed offset for preshuffle loads
            lane_off = (w_tid // 16) * 128 + (w_tid % 16) * 8
            lane_off_idx = arith.index_cast(T.index, lane_off)

            warp_m_idx = wid // BLOCK_N_WARPS * WARP_M
            warp_n_idx = wid % BLOCK_N_WARPS * WARP_N

            ldmatrix_a_m_idx = w_tid % WMMA_M
            ldmatrix_a_k_vec_idx = w_tid // WMMA_M * 4 * MFMA_PER_WARP_K

            c_gate = [acc_init] * C_FRAGS_LEN
            c_up = [acc_init] * C_FRAGS_LEN

            # V18: Precompute token_ids for in-kernel X loading (one-time, outside pipeline)
            precomp_token_idx = []
            for i in range_constexpr(LDG_REG_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local = global_tid // LDG_X_THREADS
                sorted_row = m_offset + fx.Index(m_local)
                token_id = buffer_ops.buffer_load(sorted_tids_rsrc, sorted_row,
                                                   vec_width=1, dtype=T.i32)
                token_id_idx = arith.index_cast(T.index, token_id)
                precomp_token_idx.append(token_id_idx)

            # ---- A loading: GMEM → regs → LDS (V18: indirect via token_ids) ----
            def ldg_a(k_offset):
                vecs = []
                for i in range_constexpr(LDG_REG_COUNT):
                    global_tid = BLOCK_THREADS * i + tid
                    k_local = global_tid % LDG_X_THREADS * LDG_VEC_SIZE
                    vec = X_.vec_load((precomp_token_idx[i], fx.Index(k_offset + k_local)), LDG_VEC_SIZE)
                    vecs.append(vec)
                return vecs

            def sts_a(vecs, stage):
                for i in range_constexpr(LDG_REG_COUNT):
                    global_tid = BLOCK_THREADS * i + tid
                    m_local = global_tid // LDG_X_THREADS
                    k_local = global_tid % LDG_X_THREADS * LDG_VEC_SIZE
                    col_bytes = k_local * DTYPE_BYTES
                    col_bytes = swizzle_xor16(m_local, col_bytes, k_blocks16)
                    as_.vec_store((fx.Index(stage), m_local, col_bytes // DTYPE_BYTES), vecs[i], LDG_VEC_SIZE)

            # ---- B loading: direct from preshuffle GMEM (per-warp, coalesced) ----
            def load_b_ps(bki, base_offset):
                frags = [0] * B_FRAGS_LEN
                for jj in range_constexpr(WARP_N_STEPS):
                    n0_off = base_offset + fx.Index(jj * PS_N0_STRIDE)
                    for kk in range_constexpr(WARP_K_STEPS):
                        k0 = bki * PS_K32_PER_TILE + kk
                        off = n0_off + fx.Index(k0 * PS_K0_STRIDE) + lane_off_idx
                        frags[kk * WARP_N_STEPS + jj] = W1_PS.vec_load((off,), 8)
                return frags

            # ---- MFMA helpers ----
            def _i64_to_v4i16(x_i64):
                """Convert i64 scalar → i16x4 MFMA operand (1 VALU)."""
                v1 = vector.from_elements(T.vec(1, T.i64), [x_i64])
                return vector.bitcast(T.i16x4, v1)

            def extract_bf16_halves(frag):
                """Extract i64 pair from bf16x8 fragment for MFMA (3 VALU)."""
                i64x2 = vector.bitcast(T.i64x2, frag)
                h0 = vector.extract(i64x2, static_position=[0], dynamic_position=[])
                h1 = vector.extract(i64x2, static_position=[1], dynamic_position=[])
                return h0, h1

            def mfma_k32(a0v, a1v, b0, b1, acc):
                """2x MFMA with pre-extracted A halves, raw B halves (2 MFMA + 2 VALU for B)."""
                b0v = _i64_to_v4i16(b0)
                b1v = _i64_to_v4i16(b1)
                acc_mid = rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [a0v, b0v, acc, 0, 0, 0])
                return rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [a1v, b1v, acc_mid, 0, 0, 0])

            # ---- A0 LDS prefetch helper ----
            # V15: Prefetch first A fragment (kk=0, ii=0) from LDS right after barrier.
            def prefetch_a0_from_lds(stage):
                """Load first A fragment for (kk=0, ii=0) from LDS."""
                s = fx.Index(stage)
                wm = warp_m_idx  # ii=0
                row = wm + ldmatrix_a_m_idx
                col_bytes = ldmatrix_a_k_vec_idx * DTYPE_BYTES  # kk=0, wk=0
                col_bytes = swizzle_xor16(row, col_bytes, k_blocks16)
                a_frag = as_.vec_load((s, row, col_bytes // DTYPE_BYTES), 4 * MFMA_PER_WARP_K)
                a0, a1 = extract_bf16_halves(a_frag)
                return a0, a1

            # ---- Inline compute: load A per K-step, MMA immediately ----
            # V12: Extract A halves once, reuse across all jj (WARP_N_STEPS) iterations.
            # V15: Accept optional a0_prefetch for (kk=0, ii=0) to hide LDS latency.
            def compute_tile(stage, bg_frags, bu_frags, c_gate, c_up, a0_prefetch=None):
                s = fx.Index(stage)
                for kk in range_constexpr(WARP_K_STEPS):
                    for ii in range_constexpr(WARP_M_STEPS):
                        if a0_prefetch is not None and kk == 0 and ii == 0:
                            # V15: Use prefetched A fragment (LDS latency hidden)
                            a0, a1 = a0_prefetch
                        else:
                            wm = warp_m_idx + ii * WMMA_M
                            wk = kk * WARP_ATOM_K
                            row = wm + ldmatrix_a_m_idx
                            col_bytes = (wk + ldmatrix_a_k_vec_idx) * DTYPE_BYTES
                            col_bytes = swizzle_xor16(row, col_bytes, k_blocks16)
                            a_frag = as_.vec_load((s, row, col_bytes // DTYPE_BYTES), 4 * MFMA_PER_WARP_K)
                            a0, a1 = extract_bf16_halves(a_frag)
                        a0v = _i64_to_v4i16(a0)
                        a1v = _i64_to_v4i16(a1)
                        for jj in range_constexpr(WARP_N_STEPS):
                            b_idx = kk * WARP_N_STEPS + jj
                            c_idx = ii * WARP_N_STEPS + jj
                            bg0, bg1 = extract_bf16_halves(bg_frags[b_idx])
                            c_gate[c_idx] = mfma_k32(a0v, a1v, bg0, bg1, c_gate[c_idx])
                            bu0, bu1 = extract_bf16_halves(bu_frags[b_idx])
                            c_up[c_idx] = mfma_k32(a0v, a1v, bu0, bu1, c_up[c_idx])

            # ---- Scheduling hints for hot loop ----
            # V10: mfma_group = WARP_N_STEPS * 2 (gate+up) = 4 with tile_n=128
            MFMA_GROUP = WARP_N_STEPS * 2  # MFMAs per (kk, ii) sub-step
            MFMA_TOTAL = WARP_K_STEPS * WARP_M_STEPS * MFMA_GROUP * 2  # *2 for 2xMFMA per mfma_bf16
            MFMA_PER_ITER = 2 * MFMA_GROUP  # 2 MFMAs per mfma_bf16 call
            SCHED_ITERS = MFMA_TOTAL // MFMA_PER_ITER if MFMA_PER_ITER > 0 else 0

            def hot_loop_scheduler():
                rocdl.sched_dsrd(2)
                rocdl.sched_mfma(2)
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(1)
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(1)
                dswr_tail = LDG_REG_COUNT
                if dswr_tail > SCHED_ITERS:
                    dswr_tail = SCHED_ITERS
                dswr_start = SCHED_ITERS - dswr_tail
                for _si in range_constexpr(SCHED_ITERS):
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(MFMA_GROUP)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(MFMA_GROUP)
                    if _si >= dswr_start - 1:
                        rocdl.sched_dswr(1)
                rocdl.sched_barrier(0)

            # ---- Pipeline: A through LDS, B direct from preshuffle GMEM ----
            sts_a(ldg_a(0), 0)
            bg_frags = load_b_ps(0, gate_base)
            bu_frags = load_b_ps(0, up_base)
            gpu.barrier()
            rocdl.sched_barrier(0)
            # V15: Prefetch first A fragment from LDS right after initial barrier
            a0_pf = prefetch_a0_from_lds(0)

            for bki in range_constexpr(BLOCK_K_LOOPS - 1):
                ns = (bki + 1) % 2
                # Prefetch next tile: issue all GMEM loads first for latency hiding
                a_regs = ldg_a((bki + 1) * tile_k)
                bg_next = load_b_ps(bki + 1, gate_base)
                bu_next = load_b_ps(bki + 1, up_base)
                # Compute on current data (inline A from LDS, with A0 prefetched)
                compute_tile(bki % 2, bg_frags, bu_frags, c_gate, c_up, a0_prefetch=a0_pf)
                # Pipeline A through LDS
                sts_a(a_regs, ns)
                hot_loop_scheduler()  # V15: moved before barrier (matching reference)
                gpu.barrier()
                # V15: Prefetch A0 for next compute_tile immediately after barrier
                a0_pf = prefetch_a0_from_lds(ns)
                bg_frags = bg_next
                bu_frags = bu_next

            # Last tile
            compute_tile((BLOCK_K_LOOPS - 1) % 2, bg_frags, bu_frags, c_gate, c_up, a0_prefetch=a0_pf)

            # ---- Epilogue: SiLU fusion + store through LDS ----
            stmatrix_c_m_vec_idx = w_tid // WMMA_N * WMMA_C_FRAG
            stmatrix_c_n_idx = w_tid % WMMA_N

            gpu.barrier()
            for ii in range_constexpr(WARP_M_STEPS):
                wm = warp_m_idx + ii * WMMA_M
                for jj in range_constexpr(WARP_N_STEPS):
                    wn = warp_n_idx + jj * WMMA_N
                    c_idx = ii * WARP_N_STEPS + jj
                    gate_vec = c_gate[c_idx]
                    up_vec = c_up[c_idx]
                    for kk in range_constexpr(WMMA_C_FRAG):
                        gate_val = vector.extract(gate_vec, static_position=[kk], dynamic_position=[])
                        up_val = vector.extract(up_vec, static_position=[kk], dynamic_position=[])
                        t = gate_val * (-1.4426950408889634)
                        emu = rocdl.exp2(T.f32, t)
                        den = 1.0 + emu
                        sig = rocdl.rcp(T.f32, den)
                        activated = gate_val * sig * up_val
                        activated_bf16 = arith.trunc_f(T.bf16, activated)
                        lds_m = fx.Index(wm + stmatrix_c_m_vec_idx + kk)
                        lds_n = fx.Index(wn + stmatrix_c_n_idx)
                        cs_.scalar_store((lds_m, lds_n), activated_bf16)

            # Cooperative vectorized GMEM store from LDS
            gpu.barrier()
            for i in range_constexpr(LDG_REG_C_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local = fx.Index(global_tid // LDG_C_X_THREADS)
                n_local = fx.Index(global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE)
                vec = cs_.vec_load((m_local, n_local), LDG_VEC_SIZE)
                OUT_.vec_store((m_offset + m_local, n_offset + n_local), vec, LDG_VEC_SIZE)

    @flyc.jit
    def launch_stage1(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w1_ps: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_num_valid_tiles: fx.Tensor,
        num_m_tiles: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bn = inter_dim // tile_n
        moe_stage1_kernel(arg_out, arg_x, arg_w1_ps, arg_expert_ids,
                          arg_sorted_token_ids,
                      arg_num_valid_tiles).launch(
            grid=(bn, num_m_tiles, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_stage1


# ============================================================
# Stage 2 Kernel: Grouped GEMM (activated @ w2^T)
# A through LDS, B from preshuffle GMEM
# ============================================================
@functools.lru_cache(maxsize=64)
def compile_moe_stage2(
    *,
    inter_dim: int,
    hidden_size: int,
    num_experts: int,
    tile_m: int = 64,
    tile_n: int = 128,
    tile_k: int = 128,
):
    GPU_ARCH = get_rocm_arch()
    DTYPE_BYTES = 2
    WARP_SIZE = 64
    LDG_VEC_SIZE = 8

    WMMA_M, WMMA_N, WMMA_K = 16, 16, 16
    WMMA_C_FRAG = 4
    MFMA_PER_WARP_K = 2
    WARP_ATOM_K = WMMA_K * MFMA_PER_WARP_K

    BLOCK_M_WARPS = 1
    BLOCK_N_WARPS = 4
    BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE  # 256

    WARP_M_STEPS = tile_m // (BLOCK_M_WARPS * WMMA_M)
    WARP_N_STEPS = tile_n // (BLOCK_N_WARPS * WMMA_N)
    WARP_M = WARP_M_STEPS * WMMA_M
    WARP_N = WARP_N_STEPS * WMMA_N
    WARP_K_STEPS = tile_k // WARP_ATOM_K

    LDG_A_X_THREADS = tile_k // LDG_VEC_SIZE
    LDG_REG_A_COUNT = (tile_m * tile_k) // (LDG_VEC_SIZE * BLOCK_THREADS)

    C_FRAGS_LEN = WARP_M_STEPS * WARP_N_STEPS
    B_FRAGS_LEN = WARP_K_STEPS * WARP_N_STEPS

    STAGES = 2
    BLOCK_K_BYTES = tile_k * DTYPE_BYTES
    BLOCK_K_LOOPS = inter_dim // tile_k

    # B preshuffle constants (K dimension = inter_dim for Stage 2)
    PS_K0_STRIDE = 512
    PS_N0_STRIDE = (inter_dim // 32) * PS_K0_STRIDE
    PS_K32_PER_TILE = tile_k // 32

    # LDS: A double-buffer only (V17: epilogue uses atomic GMEM writes, no LDS needed)
    allocator = SmemAllocator(None, arch=GPU_ARCH, global_sym_name="smem_s2")
    smem_a_offset = allocator._align(allocator.ptr, 16)
    A_LDS_BYTES = STAGES * tile_m * tile_k * DTYPE_BYTES
    allocator.ptr = smem_a_offset + A_LDS_BYTES

    @flyc.kernel
    def moe_stage2_kernel(
        arg_out: fx.Tensor,
        arg_activated: fx.Tensor,
        arg_w2_ps: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_tiles: fx.Tensor,
    ):
        # V27: Early-exit guard for zero-sync sort
        _guard_bid_m = fx.block_idx.y
        _nv_rsrc = buffer_ops.create_buffer_resource(arg_num_valid_tiles, max_size=True)
        _nv_i32 = buffer_ops.buffer_load(_nv_rsrc, fx.Index(0), vec_width=1, dtype=T.i32)
        _blk_valid = arith.cmpi(arith.CmpIPredicate.ult, _guard_bid_m, _nv_i32)
        _if_blk = scf.IfOp(_blk_valid)
        with _if_then(_if_blk):
            dtype_ = T.bf16
            acc_init = arith.constant_vector(0.0, T.f32x4)

            ACT_ = GTensor(arg_activated, dtype=dtype_, shape=(-1, inter_dim))
            W2_PS = GTensor(arg_w2_ps, dtype=dtype_, shape=(num_experts * hidden_size * inter_dim,))
            EXP_ = GTensor(arg_expert_ids, dtype=T.i32, shape=(-1,))
            # V17: output is f32 (pre-zeroed), atomic accumulation
            out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
            # Buffer resources for sorted token IDs and weights
            sorted_tids_rsrc = buffer_ops.create_buffer_resource(arg_sorted_token_ids, max_size=True)
            sorted_wts_rsrc = buffer_ops.create_buffer_resource(arg_sorted_weights, max_size=True)

            base_ptr = allocator.get_base()
            smem_a_ptr = SmemPtr(base_ptr, smem_a_offset, dtype_, shape=(STAGES * tile_m * tile_k,))
            as_ = STensor(smem_a_ptr, dtype_, shape=(STAGES, tile_m, tile_k))

            tid = fx.Int32(fx.thread_idx.x)
            wid = tid // WARP_SIZE
            w_tid = tid % WARP_SIZE

            # Grid swapped: (bn, num_m_tiles) for L2 A-tile reuse
            bid_n = fx.block_idx.x
            bid_m = fx.block_idx.y

            m_offset = fx.Index(bid_m * tile_m)
            n_offset = fx.Index(bid_n * tile_n)
            k_blocks16 = fx.Int32(BLOCK_K_BYTES // 16)

            expert_id = EXP_.load((fx.Index(bid_m),))
            expert_id_idx = arith.index_cast(T.index, expert_id)

            # Preshuffle B base
            wid_n = wid % BLOCK_N_WARPS
            wid_n_idx = arith.index_cast(T.index, wid_n)
            bid_n_idx = fx.Index(bid_n)
            # V10: each warp handles WARP_N_STEPS n16 blocks
            b_n0 = expert_id_idx * fx.Index(hidden_size // 16) + bid_n_idx * fx.Index(tile_n // 16) + wid_n_idx * fx.Index(WARP_N_STEPS)
            b_base = b_n0 * fx.Index(PS_N0_STRIDE)

            lane_off = (w_tid // 16) * 128 + (w_tid % 16) * 8
            lane_off_idx = arith.index_cast(T.index, lane_off)

            warp_m_idx = wid // BLOCK_N_WARPS * WARP_M
            warp_n_idx = wid % BLOCK_N_WARPS * WARP_N

            ldmatrix_a_m_idx = w_tid % WMMA_M
            ldmatrix_a_k_vec_idx = w_tid // WMMA_M * 4 * MFMA_PER_WARP_K

            c_frags = [acc_init] * C_FRAGS_LEN

            def ldg_a(k_offset):
                vecs = []
                for i in range_constexpr(LDG_REG_A_COUNT):
                    global_tid = BLOCK_THREADS * i + tid
                    m_local = global_tid // LDG_A_X_THREADS
                    k_local = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                    vec = ACT_.vec_load((m_offset + fx.Index(m_local), fx.Index(k_offset + k_local)), LDG_VEC_SIZE)
                    vecs.append(vec)
                return vecs

            def sts_a(vecs, stage):
                for i in range_constexpr(LDG_REG_A_COUNT):
                    global_tid = BLOCK_THREADS * i + tid
                    m_local = global_tid // LDG_A_X_THREADS
                    k_local = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                    col_bytes = k_local * DTYPE_BYTES
                    col_bytes = swizzle_xor16(m_local, col_bytes, k_blocks16)
                    as_.vec_store((fx.Index(stage), m_local, col_bytes // DTYPE_BYTES), vecs[i], LDG_VEC_SIZE)

            def load_b_ps(bki, base_offset):
                frags = [0] * B_FRAGS_LEN
                for jj in range_constexpr(WARP_N_STEPS):
                    n0_off = base_offset + fx.Index(jj * PS_N0_STRIDE)
                    for kk in range_constexpr(WARP_K_STEPS):
                        k0 = bki * PS_K32_PER_TILE + kk
                        off = n0_off + fx.Index(k0 * PS_K0_STRIDE) + lane_off_idx
                        frags[kk * WARP_N_STEPS + jj] = W2_PS.vec_load((off,), 8)
                return frags

            # ---- MFMA helpers (same as S1) ----
            def _i64_to_v4i16_s2(x_i64):
                v1 = vector.from_elements(T.vec(1, T.i64), [x_i64])
                return vector.bitcast(T.i16x4, v1)

            def extract_bf16_halves_s2(frag):
                i64x2 = vector.bitcast(T.i64x2, frag)
                h0 = vector.extract(i64x2, static_position=[0], dynamic_position=[])
                h1 = vector.extract(i64x2, static_position=[1], dynamic_position=[])
                return h0, h1

            def mfma_k32_s2(a0v, a1v, b0, b1, acc):
                b0v = _i64_to_v4i16_s2(b0)
                b1v = _i64_to_v4i16_s2(b1)
                acc_mid = rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [a0v, b0v, acc, 0, 0, 0])
                return rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [a1v, b1v, acc_mid, 0, 0, 0])

            # ---- A0 LDS prefetch helper (S2) ----
            def prefetch_a0_from_lds_s2(stage):
                s = fx.Index(stage)
                wm = warp_m_idx
                row = wm + ldmatrix_a_m_idx
                col_bytes = ldmatrix_a_k_vec_idx * DTYPE_BYTES
                col_bytes = swizzle_xor16(row, col_bytes, k_blocks16)
                a_frag = as_.vec_load((s, row, col_bytes // DTYPE_BYTES), 4 * MFMA_PER_WARP_K)
                a0, a1 = extract_bf16_halves_s2(a_frag)
                return a0, a1

            # ---- Inline compute: load A per K-step, MMA immediately ----
            # V12: Extract A halves once, reuse across WARP_N_STEPS
            # V15: Accept optional a0_prefetch for (kk=0, ii=0)
            def compute_tile_s2(stage, b_frags, c_frags, a0_prefetch=None):
                s = fx.Index(stage)
                for kk in range_constexpr(WARP_K_STEPS):
                    for ii in range_constexpr(WARP_M_STEPS):
                        if a0_prefetch is not None and kk == 0 and ii == 0:
                            a0, a1 = a0_prefetch
                        else:
                            wm = warp_m_idx + ii * WMMA_M
                            wk = kk * WARP_ATOM_K
                            row = wm + ldmatrix_a_m_idx
                            col_bytes = (wk + ldmatrix_a_k_vec_idx) * DTYPE_BYTES
                            col_bytes = swizzle_xor16(row, col_bytes, k_blocks16)
                            a_frag = as_.vec_load((s, row, col_bytes // DTYPE_BYTES), 4 * MFMA_PER_WARP_K)
                            a0, a1 = extract_bf16_halves_s2(a_frag)
                        a0v = _i64_to_v4i16_s2(a0)
                        a1v = _i64_to_v4i16_s2(a1)
                        for jj in range_constexpr(WARP_N_STEPS):
                            b_idx = kk * WARP_N_STEPS + jj
                            c_idx = ii * WARP_N_STEPS + jj
                            b0, b1 = extract_bf16_halves_s2(b_frags[b_idx])
                            c_frags[c_idx] = mfma_k32_s2(a0v, a1v, b0, b1, c_frags[c_idx])

            # ---- Scheduling hints for S2 hot loop ----
            MFMA_GROUP_S2 = WARP_N_STEPS  # MFMAs per (kk, ii) sub-step (no gate/up split)
            MFMA_TOTAL_S2 = WARP_K_STEPS * WARP_M_STEPS * MFMA_GROUP_S2 * 2
            MFMA_PER_ITER_S2 = 2 * MFMA_GROUP_S2
            SCHED_ITERS_S2 = MFMA_TOTAL_S2 // MFMA_PER_ITER_S2 if MFMA_PER_ITER_S2 > 0 else 0

            def hot_loop_scheduler_s2():
                rocdl.sched_dsrd(2)
                rocdl.sched_mfma(2)
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(1)
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(1)
                dswr_tail = LDG_REG_A_COUNT
                if dswr_tail > SCHED_ITERS_S2:
                    dswr_tail = SCHED_ITERS_S2
                dswr_start = SCHED_ITERS_S2 - dswr_tail
                for _si in range_constexpr(SCHED_ITERS_S2):
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(MFMA_GROUP_S2)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(MFMA_GROUP_S2)
                    if _si >= dswr_start - 1:
                        rocdl.sched_dswr(1)
                rocdl.sched_barrier(0)

            # ---- Pipeline ----
            sts_a(ldg_a(0), 0)
            b_frags = load_b_ps(0, b_base)
            gpu.barrier()
            rocdl.sched_barrier(0)
            # V15: Prefetch first A fragment after initial barrier
            a0_pf_s2 = prefetch_a0_from_lds_s2(0)

            for bki in range_constexpr(BLOCK_K_LOOPS - 1):
                ns = (bki + 1) % 2
                a_regs = ldg_a((bki + 1) * tile_k)
                b_next = load_b_ps(bki + 1, b_base)
                compute_tile_s2(bki % 2, b_frags, c_frags, a0_prefetch=a0_pf_s2)
                sts_a(a_regs, ns)
                hot_loop_scheduler_s2()  # V15: moved before barrier
                gpu.barrier()
                # V15: Prefetch A0 for next tile
                a0_pf_s2 = prefetch_a0_from_lds_s2(ns)
                b_frags = b_next

            compute_tile_s2((BLOCK_K_LOOPS - 1) % 2, b_frags, c_frags, a0_prefetch=a0_pf_s2)

            # ---- V17 Epilogue: weight-multiply + atomic f32 add to output ----
            # MFMA 16x16x16 output layout: thread w_tid owns 4 elements at
            # M positions: (w_tid//16)*4 + {0,1,2,3}, N position: w_tid%16
            stmatrix_c_m_vec_idx = w_tid // WMMA_N * WMMA_C_FRAG
            stmatrix_c_n_idx = w_tid % WMMA_N
            hidden_size_i32 = fx.Int32(hidden_size)
            c4_i32 = fx.Int32(4)  # 4 bytes per f32
            zero_i32 = fx.Int32(0)

            for ii in range_constexpr(WARP_M_STEPS):
                wm = warp_m_idx + ii * WMMA_M
                # Pre-load token_id and weight for 4 M-rows (reuse across jj)
                token_ids_ii = []
                weights_ii = []
                for kk in range_constexpr(WMMA_C_FRAG):
                    m_global_idx = m_offset + fx.Index(wm + stmatrix_c_m_vec_idx + kk)
                    m_global_i32 = arith.index_cast(T.i32, m_global_idx)
                    tid_val = buffer_ops.buffer_load(sorted_tids_rsrc, m_global_idx,
                                                      vec_width=1, dtype=T.i32)
                    wt_val = buffer_ops.buffer_load(sorted_wts_rsrc, m_global_idx,
                                                     vec_width=1, dtype=T.f32)
                    token_ids_ii.append(tid_val)
                    weights_ii.append(wt_val)

                for jj in range_constexpr(WARP_N_STEPS):
                    wn = warp_n_idx + jj * WMMA_N
                    c_idx = ii * WARP_N_STEPS + jj
                    c_vec = c_frags[c_idx]
                    n_global_i32 = arith.index_cast(T.i32,
                        n_offset + fx.Index(wn + stmatrix_c_n_idx))
                    for kk in range_constexpr(WMMA_C_FRAG):
                        val_f32 = vector.extract(c_vec, static_position=[kk],
                                                 dynamic_position=[])
                        # Weight multiply
                        weighted_val = val_f32 * weights_ii[kk]
                        # Byte offset: (token_id * hidden_size + n_global) * 4
                        byte_off = (token_ids_ii[kk] * hidden_size_i32 + n_global_i32) * c4_i32
                        # Atomic f32 add via buffer descriptor
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            weighted_val, out_rsrc, byte_off, zero_i32, zero_i32)

    @flyc.jit
    def launch_stage2(
        arg_out: fx.Tensor,
        arg_activated: fx.Tensor,
        arg_w2_ps: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_tiles: fx.Tensor,
        num_m_tiles: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bn = hidden_size // tile_n
        moe_stage2_kernel(arg_out, arg_activated, arg_w2_ps, arg_expert_ids,
                          arg_sorted_token_ids, arg_sorted_weights,
                      arg_num_valid_tiles).launch(
            grid=(bn, num_m_tiles, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_stage2


# ============================================================
# Main fused_moe function
# ============================================================
def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_ps: torch.Tensor = None,
    w2_ps: torch.Tensor = None,
    activated_buf: torch.Tensor = None,
    output_buf: torch.Tensor = None,
) -> torch.Tensor:
    token_count = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    num_experts = w1.shape[0]
    inter_dim = w1.shape[1] // 2

    TILE_M = 32 if token_count <= 64 else 64
    TILE_K = 128
    TILE_N = 128

    # V27: Sort returns max-size tensors + GPU num_m_tiles tensor (zero-sync)
    sorted_token_ids, sorted_weights, expert_ids, num_m_tiles_or_tensor = \
        moe_sort_tokens(topk_ids, topk_weights, num_experts, TILE_M)

    total_sorted = sorted_token_ids.shape[0]
    max_tiles = expert_ids.shape[0]

    if w1_ps is None:
        w1_flat = w1.reshape(num_experts * 2 * inter_dim, hidden_size).contiguous()
        w1_ps = preshuffle_weight(w1_flat)
    if w2_ps is None:
        w2_flat = w2.reshape(num_experts * hidden_size, inter_dim).contiguous()
        w2_ps = preshuffle_weight(w2_flat)

    # V21: Zero output BEFORE S1
    if output_buf is not None and output_buf.shape[0] >= token_count:
        output = output_buf[:token_count]
        output.zero_()
    else:
        output = torch.zeros(token_count, hidden_size, dtype=torch.float32, device=hidden_states.device)

    # V27: Zero-sync for fused path (GPU tensor), CPU int for fallback path
    if isinstance(num_m_tiles_or_tensor, torch.Tensor):
        # Zero-sync: launch with max_tiles, kernel early-exits invalid tiles
        num_m_tiles_gpu = num_m_tiles_or_tensor
        num_m_tiles = max_tiles
    else:
        # Fallback: CPU int (already synced)
        num_m_tiles_gpu = torch.tensor([num_m_tiles_or_tensor], dtype=torch.int32,
                                        device=hidden_states.device)
        num_m_tiles = num_m_tiles_or_tensor

    # Allocate activated at max size (padding tiles: S1 writes, S2 discards via weight=0)
    if activated_buf is not None and activated_buf.shape[0] >= total_sorted:
        activated = activated_buf[:total_sorted]
    else:
        activated = torch.empty(total_sorted, inter_dim, dtype=torch.bfloat16, device=hidden_states.device)

    stage1_fn = compile_moe_stage1(
        inter_dim=inter_dim, hidden_size=hidden_size, num_experts=num_experts,
        tile_m=TILE_M, tile_n=TILE_N, tile_k=TILE_K,
    )
    stage1_fn(activated, hidden_states, w1_ps, expert_ids, sorted_token_ids,
              num_m_tiles_gpu, num_m_tiles, stream=torch.cuda.current_stream())

    stage2_fn = compile_moe_stage2(
        inter_dim=inter_dim, hidden_size=hidden_size, num_experts=num_experts,
        tile_m=TILE_M, tile_n=TILE_N, tile_k=TILE_K,
    )
    stage2_fn(output, activated, w2_ps, expert_ids,
              sorted_token_ids, sorted_weights,
              num_m_tiles_gpu, num_m_tiles, stream=torch.cuda.current_stream())

    return output
