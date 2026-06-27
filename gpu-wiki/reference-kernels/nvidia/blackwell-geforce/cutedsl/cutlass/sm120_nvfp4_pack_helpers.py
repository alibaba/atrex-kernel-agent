"""CuTeDSL NVFP4 GEMM kernels for sm_120 (RTX PRO 5000 Blackwell-Geforce).

Build progression:
  (1) single m16n8k64 atom (validated in 06_self_quant_demo.py)
  (2) per-warp K-loop accumulating into the same m16n8k64 register tile
  (3) per-block multi-warp tile of size (TILE_M, TILE_N, TILE_K)
       — M = TILE_M * NUM_WARPS_M, N = TILE_N * NUM_WARPS_N
  (4) multi-block grid: (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
  (5) grouped GEMM: each block consults expert_offsets to find its expert id
       and per-expert M.
  (6) prologue: gather + bf16→nvfp4 quant on the fly
  (7) epilogue: silu_and_mul + nvfp4 quant on the fly
"""
import os
os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")

import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.core import _pack_shape
from cutlass.cute.atom import make_atom
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu.warp.mma import MmaMXF4NVF4Trait, MmaSM120BlockScaledOp
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir


AB_DTYPE = cutlass.Float4E2M1FN
SF_DTYPE = cutlass.Float8E4M3FN
ACC_DTYPE = cutlass.Float32

ATOM_M, ATOM_N, ATOM_K = 16, 8, 64
SF_VEC_SIZE = 16


# ---------------------------------------------------------------------------
# inline-PTX wrapper for the m16n8k64 NVFP4 mma atom
# ---------------------------------------------------------------------------

class MmaMXF4NVF4InlineOp(MmaSM120BlockScaledOp):
    descriptive_name = "sm120 nvfp4 inline-ptx mma op"
    def __init__(self):
        super().__init__(AB_DTYPE, ACC_DTYPE, (ATOM_M, ATOM_N, ATOM_K), SF_DTYPE, SF_VEC_SIZE)
    def _make_trait(self, *, loc=None, ip=None, **kwargs):
        shape_mnk = _pack_shape(self.shape_mnk, loc=loc, ip=ip)
        ty = _cute_nvgpu_ir.MmaAtomSM120BlockScaledType.get(
            shape_mnk.type.attribute, SF_VEC_SIZE, False,
            self.ab_dtype.mlir_type, self.ab_dtype.mlir_type,
            self.acc_dtype.mlir_type, self.sf_type.mlir_type,
        )
        return MmaMXF4NVF4Trait(make_atom(ty, loc=loc, ip=ip))
    def _verify_fragment_A(self, x, *, loc=None, ip=None): return None
    def _verify_fragment_B(self, x, *, loc=None, ip=None): return None


def _ir(v, loc=None, ip=None):
    return v.ir_value(loc=loc, ip=ip) if hasattr(v, "ir_value") else v
def _ext(v, i, ty, *, loc=None, ip=None):
    return llvm.extractelement(_ir(v, loc, ip), _ir(cutlass.Int32(i), loc, ip), loc=loc, ip=ip)


@dsl_user_op
def mma_atom(a_regs, b_regs, c_regs, sfa, sfb, *, loc=None, ip=None):
    """One m16n8k64 NVFP4 MMA. Returns updated 4-element fp32 accumulator vector."""
    a = [_ext(a_regs, i, T.i32(), loc=loc, ip=ip) for i in range(4)]
    b = [_ext(b_regs, i, T.i32(), loc=loc, ip=ip) for i in range(2)]
    c = [_ext(c_regs, i, T.f32(), loc=loc, ip=ip) for i in range(4)]
    rst = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()]*4),
        [*a, *b, *c,
         _ir(sfa, loc, ip), _ir(cutlass.Int16(0), loc, ip), _ir(cutlass.Int16(0), loc, ip),
         _ir(sfb, loc, ip), _ir(cutlass.Int16(0), loc, ip), _ir(cutlass.Int16(0), loc, ip)],
        """mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
           {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9},
           {$10, $11, $12, $13}, {$14}, {$15, $16}, {$17}, {$18, $19};""",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,h,h,r,h,h",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    out = cute.make_rmem_tensor((4,), ACC_DTYPE)
    for i in range(4):
        out[i] = cutlass.Float32(llvm.extractvalue(T.f32(), rst, [i]))
    return out.load()


# ---------------------------------------------------------------------------
# operand fetch helpers — load nibbles from gmem into the warp's regs
# ---------------------------------------------------------------------------

def _load_nibble(t, row, k):
    bv = cutlass.Int32(t[row, k // cutlass.Int32(2)])
    sh = (k & cutlass.Int32(1)) * cutlass.Int32(4)
    return (bv >> sh) & cutlass.Int32(0xF)


def _build_gemm_v2(M, N, K):
    """Closure-bound kernel: M, N, K become Python compile-time constants."""
    K_HALF = K // 2
    SF_BLOCK_BYTES = 128 * 4
    SF_TILES_K = K // ATOM_K

    @cute.kernel
    def gemm_v2_kernel(
        a_q,          # (M, K // 2) uint8
        b_q,          # (N, K // 2) uint8
        a_sf_pack,    # (M_blocks * SF_TILES_K * 128, 4) uint8
        b_sf_pack,    # (N_blocks * SF_TILES_K * 128, 4) uint8
        alpha,        # (1,) fp32
        out,          # (M, N) fp32
    ):
        """One warp = one (16, 8) tile, looping over K in steps of ATOM_K."""
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        perm = utils.blackwell_helpers.get_permutation_mnk((ATOM_M, ATOM_N, ATOM_K), SF_VEC_SIZE, False)
        cute.make_tiled_mma(MmaMXF4NVF4InlineOp(), permutation_mnk=perm)

        m0 = by * cutlass.Int32(ATOM_M)
        n0 = bx * cutlass.Int32(ATOM_N)

        gA = cute.make_tensor(a_q.iterator + m0 * cutlass.Int32(K_HALF),
                              cute.make_layout((ATOM_M, K_HALF), stride=(K_HALF, 1)))
        gB = cute.make_tensor(b_q.iterator + n0 * cutlass.Int32(K_HALF),
                              cute.make_layout((ATOM_N, K_HALF), stride=(K_HALF, 1)))
        a_sf_off = by * cutlass.Int32(SF_BLOCK_BYTES * SF_TILES_K)
        b_sf_off = bx * cutlass.Int32(SF_BLOCK_BYTES * SF_TILES_K)

        acc = cute.make_rmem_tensor((4,), ACC_DTYPE); acc.fill(0.0)

        lane_g = tidx % cutlass.Int32(4)
        lane_r = tidx // cutlass.Int32(4)
        sfa_row = (tidx // cutlass.Int32(4)) + (tidx % cutlass.Int32(2)) * cutlass.Int32(8)
        sfb_row = tidx // cutlass.Int32(4)
        sfa_p = sfa_row * cutlass.Int32(4)
        sfb_p = sfb_row * cutlass.Int32(4)

        for kt in cutlass.range_constexpr(SF_TILES_K):
            k_off_b = kt * (ATOM_K // 2)
            gA_k = cute.make_tensor(
                gA.iterator + cutlass.Int32(k_off_b),
                cute.make_layout((ATOM_M, ATOM_K // 2), stride=(K_HALF, 1)),
            )
            gB_k = cute.make_tensor(
                gB.iterator + cutlass.Int32(k_off_b),
                cute.make_layout((ATOM_N, ATOM_K // 2), stride=(K_HALF, 1)),
            )
            gSFA_k = cute.make_tensor(
                a_sf_pack.iterator + a_sf_off + cutlass.Int32(kt * SF_BLOCK_BYTES),
                cute.make_layout((128, 4), stride=(4, 1)),
            )
            gSFB_k = cute.make_tensor(
                b_sf_pack.iterator + b_sf_off + cutlass.Int32(kt * SF_BLOCK_BYTES),
                cute.make_layout((128, 4), stride=(4, 1)),
            )

            rA = cute.make_rmem_tensor((4,), cutlass.Int32)
            rB = cute.make_rmem_tensor((2,), cutlass.Int32)
            for i in cutlass.range_constexpr(4): rA[i] = cutlass.Int32(0)
            for i in cutlass.range_constexpr(2): rB[i] = cutlass.Int32(0)

            for ln in cutlass.range_constexpr(32):
                v0, v1, v2 = ln % 8, (ln // 8) % 2, ln // 16
                lr = lane_r + cutlass.Int32(8 * v1)
                lk = lane_g * cutlass.Int32(8) + cutlass.Int32(v0) + cutlass.Int32(32 * v2)
                nib = _load_nibble(gA_k, lr, lk)
                ri = cutlass.Int32(ln // 8); ni = cutlass.Int32(ln % 8)
                rA[ri] = rA[ri] | (nib << (cutlass.Int32(4) * ni))
            for ln in cutlass.range_constexpr(16):
                v0, v1 = ln % 8, ln // 8
                lk = lane_g * cutlass.Int32(8) + cutlass.Int32(v0) + cutlass.Int32(32 * v1)
                nib = _load_nibble(gB_k, lane_r, lk)
                ri = cutlass.Int32(ln // 8); ni = cutlass.Int32(ln % 8)
                rB[ri] = rB[ri] | (nib << (cutlass.Int32(4) * ni))

            sfa = (cutlass.Int32(gSFA_k[sfa_p, 0])
                   | (cutlass.Int32(gSFA_k[sfa_p, 1]) << cutlass.Int32(8))
                   | (cutlass.Int32(gSFA_k[sfa_p, 2]) << cutlass.Int32(16))
                   | (cutlass.Int32(gSFA_k[sfa_p, 3]) << cutlass.Int32(24)))
            sfb = (cutlass.Int32(gSFB_k[sfb_p, 0])
                   | (cutlass.Int32(gSFB_k[sfb_p, 1]) << cutlass.Int32(8))
                   | (cutlass.Int32(gSFB_k[sfb_p, 2]) << cutlass.Int32(16))
                   | (cutlass.Int32(gSFB_k[sfb_p, 3]) << cutlass.Int32(24)))

            acc.store(mma_atom(rA.load(), rB.load(), acc.load(), sfa, sfb))

        av = alpha[0]
        for i in cutlass.range_constexpr(cute.size(acc)): acc[i] = acc[i] * av

        gO = cute.make_tensor(
            out.iterator + m0 * cutlass.Int32(N) + n0,
            cute.make_layout((ATOM_M, ATOM_N), stride=(N, 1)),
        )
        nb = lane_g * cutlass.Int32(2)
        gO[lane_r + cutlass.Int32(0), nb + cutlass.Int32(0)] = acc[0]
        gO[lane_r + cutlass.Int32(0), nb + cutlass.Int32(1)] = acc[1]
        gO[lane_r + cutlass.Int32(8), nb + cutlass.Int32(0)] = acc[2]
        gO[lane_r + cutlass.Int32(8), nb + cutlass.Int32(1)] = acc[3]

    return gemm_v2_kernel


# Cache of compiled kernels keyed by (M, N, K)
_KERN_CACHE = {}

def _build_launcher(M, N, K):
    """Build a @cute.jit launcher that closes over (M, N, K) and the kernel."""
    kern = _build_gemm_v2(M, N, K)
    grid_x = (N + ATOM_N - 1) // ATOM_N
    grid_y = (M + ATOM_M - 1) // ATOM_M

    @cute.jit
    def launcher(a_q, b_q, a_sf, b_sf, alpha, out, stream):
        kern(a_q, b_q, a_sf, b_sf, alpha, out).launch(
            grid=(grid_x, grid_y, 1), block=(32, 1, 1), stream=stream
        )
    return launcher


def launch_gemm_v2(a_q, b_q, a_sf, b_sf, alpha, out, M, N, K, stream):
    key = (int(M), int(N), int(K))
    if key not in _KERN_CACHE:
        _KERN_CACHE[key] = _build_launcher(*key)
    _KERN_CACHE[key](a_q, b_q, a_sf, b_sf, alpha, out, stream)


# ---------------------------------------------------------------------------
# Host-side helpers: build the SF buffer the kernel expects.
# Layout: for each M-tile (block of ATOM_M rows), pack the SF for each K-tile
# (ATOM_K cols) into a (128, 4) chunk; concat along K. Total SF buffer is
# (M_blocks * SF_TILES_K * 128, 4) bytes.
# ---------------------------------------------------------------------------

def pack_sf_per_atom(sf_raw: torch.Tensor, atom_dim: int) -> torch.Tensor:
    """sf_raw shape (M, K // 16) raw e4m3 bytes.
       returns (M_blocks * SF_TILES_K * 128, 4) for the kernel.
    """
    M, KSF = sf_raw.shape
    K = KSF * SF_VEC_SIZE
    assert M % atom_dim == 0 and K % ATOM_K == 0
    M_blocks = M // atom_dim
    SF_TILES_K = K // ATOM_K          # SF bytes per K-tile = 4
    out = torch.zeros((M_blocks, SF_TILES_K, 128, 4), dtype=torch.uint8, device=sf_raw.device)
    rows = torch.arange(atom_dim, device=sf_raw.device) * 4
    for mb in range(M_blocks):
        for kt in range(SF_TILES_K):
            chunk = sf_raw[mb*atom_dim:(mb+1)*atom_dim, kt*4:(kt+1)*4]   # (atom_dim, 4)
            out[mb, kt, rows] = chunk
    return out.reshape(M_blocks * SF_TILES_K * 128, 4).contiguous()


def pack_sf_per_atom_compressed(sf_raw: torch.Tensor, atom_dim: int) -> torch.Tensor:
    """Tight version of pack_sf_per_atom — no zero padding.
    Returns (M_blocks * SF_TILES_K * atom_dim, 4). 8x smaller than pack_sf_per_atom
    when atom_dim=16. Read pattern: thread `lane` reads bytes at row=sfa_logical
    (no *4 multiplication).
    """
    M, KSF = sf_raw.shape
    K = KSF * SF_VEC_SIZE
    assert M % atom_dim == 0 and K % ATOM_K == 0
    M_blocks = M // atom_dim
    SF_TILES_K = K // ATOM_K
    out = torch.empty((M_blocks, SF_TILES_K, atom_dim, 4), dtype=torch.uint8, device=sf_raw.device)
    for mb in range(M_blocks):
        for kt in range(SF_TILES_K):
            out[mb, kt] = sf_raw[mb*atom_dim:(mb+1)*atom_dim, kt*4:(kt+1)*4]
    return out.reshape(M_blocks * SF_TILES_K * atom_dim, 4).contiguous()


def pack_sf_per_block(sf_raw: torch.Tensor, atom_dim: int, atoms_per_block: int) -> torch.Tensor:
    """Block-aware compressed pack.
    Groups SF as (M_blocks_outer, SF_TILES_K, atoms_per_block, atom_dim, 4) so that
    all atoms for one CTA tile's K-block are contiguous (atoms_per_block × atom_dim × 4
    = 512 bytes per chunk for SFA when atoms_per_block=8). Enables 1 cp.async per
    CTA tile chunk.
    """
    M, KSF = sf_raw.shape
    K = KSF * SF_VEC_SIZE
    block_dim = atoms_per_block * atom_dim
    assert M % block_dim == 0 and K % ATOM_K == 0
    M_blocks_outer = M // block_dim
    SF_TILES_K = K // ATOM_K
    out = torch.empty((M_blocks_outer, SF_TILES_K, atoms_per_block, atom_dim, 4),
                      dtype=torch.uint8, device=sf_raw.device)
    for mbo in range(M_blocks_outer):
        for kt in range(SF_TILES_K):
            for ai in range(atoms_per_block):
                m_start = (mbo * atoms_per_block + ai) * atom_dim
                out[mbo, kt, ai] = sf_raw[m_start:m_start+atom_dim, kt*4:(kt+1)*4]
    return out.reshape(M_blocks_outer * SF_TILES_K * atoms_per_block * atom_dim, 4).contiguous()
