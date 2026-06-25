#!/usr/bin/env python3
"""Correctness tests for the standalone FlyDSL Chunk-GDN megakernel wrapper.

The tests do not import RTP-LLM.  GPU tests compare the FlyDSL megakernel
against a PyTorch reference for the fused recompute_w_u + fwd_h + fwd_o
portion, using precomputed `a` and `g_cumsum` inputs.
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from chunk_gdn_flydsl_operator import (  # noqa: E402
    CHUNK_SIZE,
    SUPPORTED_CHUNK_GDN_SHAPES,
    chunk_gdn_flydsl_fwd,
    is_supported_shape,
    make_chunk_offsets,
)


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def _make_inputs(
    *,
    b: int,
    t: int,
    hg: int,
    h: int,
    k_dim: int = 128,
    v_dim: int = 128,
    device: str = "cuda",
    cu_seqlens: torch.Tensor | None = None,
):
    gen = torch.Generator(device=device)
    gen.manual_seed(20260509 + t + hg * 17 + h * 31)

    q = (torch.randn(b, t, hg, k_dim, device=device, generator=gen) * 0.10).to(torch.bfloat16)
    k = F.normalize(
        torch.randn(b, t, hg, k_dim, device=device, generator=gen),
        p=2,
        dim=-1,
    ).to(torch.bfloat16)
    v = (torch.randn(b, t, h, v_dim, device=device, generator=gen) * 0.10).to(torch.bfloat16)
    beta = torch.rand(b, t, h, device=device, generator=gen).sigmoid().to(torch.bfloat16)

    if cu_seqlens is None:
        g_step = -(torch.rand(b, t, h, device=device, generator=gen) * 0.015 + 0.001)
        g_cumsum = torch.cumsum(g_step, dim=1).float()
    else:
        g_cumsum = torch.empty(b, t, h, device=device, dtype=torch.float32)
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            g_step = -(torch.rand(end - start, h, device=device, generator=gen) * 0.015 + 0.001)
            g_cumsum[0, start:end] = torch.cumsum(g_step, dim=0)

    a = torch.zeros(b, t, h, CHUNK_SIZE, device=device, dtype=torch.bfloat16)
    seq_ranges = [(0, 0, t)]
    if cu_seqlens is not None:
        cu_list = cu_seqlens.detach().cpu().tolist()
        seq_ranges = [(0, start, end) for start, end in zip(cu_list[:-1], cu_list[1:])]
    for batch, start, end in seq_ranges:
        for chunk_start in range(start, end, CHUNK_SIZE):
            chunk_len = min(CHUNK_SIZE, end - chunk_start)
            for row in range(chunk_len):
                vals = torch.randn(h, row + 1, device=device, generator=gen) * 0.03
                a[batch, chunk_start + row, :, : row + 1] = vals.to(torch.bfloat16)

    return q, k, v, a, g_cumsum, beta


def chunk_gdn_torch_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
):
    bsz, total_t, hg, k_dim = q.shape
    _, _, h, v_dim = v.shape
    head_group = h // hg
    out = torch.empty(bsz, total_t, h, v_dim, device=q.device, dtype=torch.float32)
    n_state = len(cu_seqlens) - 1 if cu_seqlens is not None else bsz
    final_state = torch.empty(n_state, h, k_dim, v_dim, device=q.device, dtype=torch.float32)

    if cu_seqlens is None:
        seq_ranges = [(batch, batch, 0, total_t) for batch in range(bsz)]
    else:
        seq_ranges = [
            (0, seq_idx, int(cu_seqlens[seq_idx].item()), int(cu_seqlens[seq_idx + 1].item()))
            for seq_idx in range(n_state)
        ]

    causal_cache = {}
    for batch, state_idx, start, end in seq_ranges:
        for h_idx in range(h):
            k_head = h_idx // head_group
            if initial_state is None:
                h_state = torch.zeros(k_dim, v_dim, device=q.device, dtype=torch.float32)
            else:
                h_state = initial_state[state_idx, h_idx].float().clone()

            for chunk_start in range(start, end, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, end)
                chunk_len = chunk_end - chunk_start
                token_slice = slice(chunk_start, chunk_end)

                q_chunk = q[batch, token_slice, k_head].float()
                k_chunk = k[batch, token_slice, k_head].float()
                v_chunk = v[batch, token_slice, h_idx].float()
                g_chunk = g_cumsum[batch, token_slice, h_idx].float()
                beta_chunk = beta[batch, token_slice, h_idx].float()
                a_chunk = a[batch, token_slice, h_idx, :chunk_len].float()

                if chunk_len not in causal_cache:
                    causal_cache[chunk_len] = torch.tril(
                        torch.ones(chunk_len, chunk_len, device=q.device, dtype=torch.bool)
                    )
                causal = causal_cache[chunk_len]

                h_shared = _bf16(h_state)
                u = k_chunk @ h_shared
                w = _bf16(v_chunk - torch.exp2(g_chunk).unsqueeze(1) * u)

                o_inter = (q_chunk @ h_shared) * (scale * torch.exp2(g_chunk).unsqueeze(1))
                p = q_chunk @ k_chunk.T
                gates = torch.exp2(g_chunk.unsqueeze(1) - g_chunk.unsqueeze(0)).masked_fill(~causal, 0.0)
                pg = _bf16(scale * gates * p)

                ag = _bf16((a_chunk * beta_chunk.unsqueeze(0)).masked_fill(~causal, 0.0))
                vd = ag @ w
                vd_shared = _bf16(vd)
                out[batch, token_slice, h_idx] = o_inter + pg @ vd_shared

                vn = _bf16(torch.exp2(g_chunk[-1] - g_chunk).unsqueeze(1) * vd)
                h_state = h_state * torch.exp2(g_chunk[-1]) + k_chunk.T @ vn

            final_state[state_idx, h_idx] = h_state

    return out.to(torch.bfloat16), final_state if output_final_state else None


def _assert_similar(testcase: unittest.TestCase, actual: torch.Tensor, expected: torch.Tensor, label: str):
    actual_f = actual.float().flatten()
    expected_f = expected.float().flatten()
    cos = F.cosine_similarity(actual_f, expected_f, dim=0).item()
    max_abs = (actual_f - expected_f).abs().max().item()
    mean_abs = (actual_f - expected_f).abs().mean().item()
    testcase.assertGreater(cos, 0.999, f"{label}: cos={cos:.6f} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}")
    testcase.assertLess(mean_abs, 5e-3, f"{label}: cos={cos:.6f} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}")


def _has_flydsl() -> bool:
    return importlib.util.find_spec("flydsl") is not None


GPU_TESTS_ENABLED = os.environ.get("CHUNK_GDN_RUN_GPU_TESTS", "1") == "1"
GPU_SKIP_REASON = "requires CUDA, FlyDSL, and CHUNK_GDN_RUN_GPU_TESTS=1"


class ChunkGdnStandaloneOperatorTest(unittest.TestCase):
    def test_supported_shape_registry(self):
        self.assertIn((8, 32, 128, 128), SUPPORTED_CHUNK_GDN_SHAPES)
        self.assertIn((2, 8, 128, 128), SUPPORTED_CHUNK_GDN_SHAPES)
        q = torch.empty(1, 1, 8, 128, dtype=torch.bfloat16)
        v = torch.empty(1, 1, 32, 128, dtype=torch.bfloat16)
        self.assertTrue(is_supported_shape(q, v))
        self.assertFalse(is_supported_shape(q[:, :, :3], v[:, :, :12]))

    def test_make_chunk_offsets(self):
        cu = torch.tensor([0, 1, 64, 65, 129], dtype=torch.long)
        self.assertEqual(make_chunk_offsets(cu).tolist(), [0, 1, 2, 3, 4])
        cu = torch.tensor([0, 64, 128, 200], dtype=torch.long)
        self.assertEqual(make_chunk_offsets(cu).tolist(), [0, 1, 2, 4])

    def test_validation_rejects_missing_precomputed_inputs(self):
        q = torch.empty(1, 8, 8, 128, dtype=torch.bfloat16)
        k = torch.empty_like(q)
        v = torch.empty(1, 8, 32, 128, dtype=torch.bfloat16)
        a = torch.empty(1, 8, 32, CHUNK_SIZE, dtype=torch.bfloat16)
        bad_g = torch.empty(1, 8, 32, dtype=torch.bfloat16)
        beta = torch.empty(1, 8, 32, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "g_cumsum must be fp32"):
            chunk_gdn_flydsl_fwd(q, k, v, a, bad_g, beta)

    @unittest.skipUnless(GPU_TESTS_ENABLED and torch.cuda.is_available() and _has_flydsl(), GPU_SKIP_REASON)
    def test_dense_aligned_hot_path_matches_torch_reference(self):
        q, k, v, a, g, beta = _make_inputs(b=1, t=64, hg=8, h=32)
        scale = k.shape[-1] ** -0.5
        actual_o, actual_h = chunk_gdn_flydsl_fwd(q, k, v, a, g, beta, scale=scale)
        expected_o, expected_h = chunk_gdn_torch_reference(q, k, v, a, g, beta, scale=scale)
        torch.cuda.synchronize()
        _assert_similar(self, actual_o, expected_o, "dense-hot-o")
        _assert_similar(self, actual_h, expected_h, "dense-hot-h")

    @unittest.skipUnless(GPU_TESTS_ENABLED and torch.cuda.is_available() and _has_flydsl(), GPU_SKIP_REASON)
    def test_small_h_bdv32_path_matches_torch_reference(self):
        q, k, v, a, g, beta = _make_inputs(b=1, t=64, hg=2, h=8)
        scale = k.shape[-1] ** -0.5
        actual_o, actual_h = chunk_gdn_flydsl_fwd(q, k, v, a, g, beta, scale=scale)
        expected_o, expected_h = chunk_gdn_torch_reference(q, k, v, a, g, beta, scale=scale)
        torch.cuda.synchronize()
        _assert_similar(self, actual_o, expected_o, "small-h-o")
        _assert_similar(self, actual_h, expected_h, "small-h-h")

    @unittest.skipUnless(GPU_TESTS_ENABLED and torch.cuda.is_available() and _has_flydsl(), GPU_SKIP_REASON)
    def test_tail_path_with_initial_state_matches_torch_reference(self):
        q, k, v, a, g, beta = _make_inputs(b=1, t=65, hg=8, h=32)
        initial_state = (torch.randn(1, 32, 128, 128, device="cuda") * 0.02).float()
        scale = k.shape[-1] ** -0.5
        actual_o, actual_h = chunk_gdn_flydsl_fwd(
            q, k, v, a, g, beta, scale=scale, initial_state=initial_state
        )
        expected_o, expected_h = chunk_gdn_torch_reference(
            q, k, v, a, g, beta, scale=scale, initial_state=initial_state
        )
        torch.cuda.synchronize()
        _assert_similar(self, actual_o, expected_o, "tail-o")
        _assert_similar(self, actual_h, expected_h, "tail-h")

    @unittest.skipUnless(GPU_TESTS_ENABLED and torch.cuda.is_available() and _has_flydsl(), GPU_SKIP_REASON)
    def test_varlen_matches_torch_reference(self):
        cu = torch.tensor([0, 64, 128], device="cuda", dtype=torch.long)
        q, k, v, a, g, beta = _make_inputs(b=1, t=128, hg=2, h=8, cu_seqlens=cu)
        initial_state = (torch.randn(2, 8, 128, 128, device="cuda") * 0.02).float()
        scale = k.shape[-1] ** -0.5
        actual_o, actual_h = chunk_gdn_flydsl_fwd(
            q, k, v, a, g, beta, scale=scale, initial_state=initial_state, cu_seqlens=cu
        )
        expected_o, expected_h = chunk_gdn_torch_reference(
            q, k, v, a, g, beta, scale=scale, initial_state=initial_state, cu_seqlens=cu
        )
        torch.cuda.synchronize()
        _assert_similar(self, actual_o, expected_o, "varlen-o")
        _assert_similar(self, actual_h, expected_h, "varlen-h")


if __name__ == "__main__":
    unittest.main()
