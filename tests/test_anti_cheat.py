#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Anti-cheat test suite for tools/validate_solution.py.

Runs as pytest (``pytest tests/test_anti_cheat.py``) or standalone
(``python tests/test_anti_cheat.py``). Pure-stdlib: needs no torch/triton/GPU.

Fixtures are inline (so the suite is portable wherever AKA is installed) and are
distilled from the two real SOL-ExecBench cheating incidents:
  * 067_flash_attention_gqa_ultralong -- whole fused op delegated to
    F.linear (cuBLAS) + F.scaled_dot_product_attention (cuDNN/flash).
  * 015_gqa_paged_prefill -- flashinfer wrapper / dead @cute.kernel "for
    language classification" + (total_q, num_batches) shape-keyed memo.
The genuine control is distilled from kernel_opt_post_norm_residual (real
Triton). Real on-disk artifacts are also checked opportunistically if present.
"""
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "tools"))

import validate_solution as V  # noqa: E402

REQUIRE = V.POLICY_REQUIRE_SELF_WRITTEN
ALLOW = V.POLICY_ALLOW_LIBS


def _sol(languages, entry="kernel.py::run"):
    return {"spec": {"languages": list(languages), "entry_point": entry}}


def _codes(findings, severity=None):
    return {f.code for f in findings if severity is None or f.severity == severity}


# --------------------------------------------------------------------------- #
# Inline fixtures
# --------------------------------------------------------------------------- #

CHEAT_067 = '''
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(hidden_states, cos, sin, q_w, k_w, v_w, o_w):
    q = F.linear(hidden_states, q_w)
    k = F.linear(hidden_states, k_w)
    v = F.linear(hidden_states, v_w)
    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    return F.linear(attn, o_w)
'''

CHEAT_015 = '''
import torch
import triton
import triton.language as tl
import cutlass
import cutlass.cute as cute

@triton.jit
def _gqa_kernel(Q, O, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(O + off, tl.load(Q + off, mask=off < n), mask=off < n)

# dead CuteDSL kernels "for language classification"
@cute.kernel
def _cute_rescale(gA, gB, gC):
    pass

@cute.jit
def _cute_add_jit(mA, mB, mC):
    _cute_rescale(mA, mB, mC).launch(grid=[1, 1, 1], block=[1, 1, 1])

_tile_cache_key = None
_tile_cache_val = None

def _get_tile_mapping(qo_indptr, total_q, block_m, device):
    global _tile_cache_key, _tile_cache_val
    key = (total_q, qo_indptr.shape[0])
    if _tile_cache_key == key:
        return _tile_cache_val
    num_batches = qo_indptr.shape[0] - 1
    cpu = qo_indptr.cpu()
    ids = []
    for b in range(num_batches):
        ids.append(b)
    val = torch.tensor(ids, dtype=torch.int32, device=device)
    _tile_cache_key = key
    _tile_cache_val = val
    return val

@torch.no_grad()
def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale, output, lse):
    total_q = q.shape[0]
    mapping = _get_tile_mapping(qo_indptr, total_q, 128, q.device)
    _gqa_kernel[(total_q,)](q, output, total_q, BLOCK=128)
'''

CHEAT_FLASHINFER = '''
import torch
import flashinfer

_w = None

@torch.no_grad()
def run(q, k, v, output):
    global _w
    if _w is None:
        _w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(1))
    output.copy_(_w.run(q, (k, v)))
'''

GENUINE_TRITON = '''
import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm(x_ptr, r_ptr, o_ptr, w_ptr, eps, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    res = tl.load(r_ptr + row * n_cols + cols, mask=mask, other=0.0)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    var = tl.sum(x * x, axis=0) / n_cols
    out = res + w * (x * tl.rsqrt(var + eps))
    tl.store(o_ptr + row * n_cols + cols, out, mask=mask)

@torch.no_grad()
def run(sublayer, residual, weight, eps):
    n_rows = sublayer.numel() // sublayer.shape[-1]
    hidden = sublayer.shape[-1]
    out = torch.empty_like(residual)
    _rmsnorm[(n_rows,)](sublayer, residual, out, weight, eps, hidden,
                        BLOCK=triton.next_power_of_2(hidden))
    return out
'''

GENUINE_WITH_PROJECTION = '''
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _act(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(o_ptr + off, tl.load(x_ptr + off, mask=off < n), mask=off < n)

@torch.no_grad()
def run(x, w, out):
    h = F.linear(x, w)          # glue projection alongside a real kernel
    _act[(1,)](h, out, h.numel(), BLOCK=128)
'''


# --------------------------------------------------------------------------- #
# MUST-BLOCK
# --------------------------------------------------------------------------- #

def test_067_library_delegation_blocked():
    f = V.analyze(CHEAT_067, _sol(["triton"]), REQUIRE)
    assert V.verdict(f) == V.SEV_FAIL
    assert "C1" in _codes(f, V.SEV_FAIL)   # no self-written kernel / SDPA on path
    assert "C2" in _codes(f, V.SEV_FAIL)   # declared triton, zero triton launches
    assert not V.has_self_written_kernel(CHEAT_067)


def test_015_camouflage_and_memo_blocked():
    f = V.analyze(CHEAT_015, _sol(["cute_dsl"]), REQUIRE)
    assert V.verdict(f) == V.SEV_FAIL
    fail = _codes(f, V.SEV_FAIL)
    assert "C2" in fail   # declared cute_dsl but only triton runs + dead @cute kernels
    assert "C3" in fail   # (total_q, qo_indptr.shape[0]) memo gating H2D
    # it DOES contain a real (triton) kernel, so C1 must NOT fire here
    assert V.has_self_written_kernel(CHEAT_015)


def test_flashinfer_wrapper_blocked():
    f = V.analyze(CHEAT_FLASHINFER, _sol(["cute_dsl"]), REQUIRE)
    assert V.verdict(f) == V.SEV_FAIL
    assert "C1" in _codes(f, V.SEV_FAIL)


def test_aliased_sdpa_import_blocked():
    src = ("import torch\nimport torch.nn.functional as G\n"
           "def run(q, k, v):\n    return G.scaled_dot_product_attention(q, k, v)\n")
    f = V.analyze(src, _sol(["triton"]), REQUIRE)
    assert V.verdict(f) == V.SEV_FAIL
    assert "C1" in _codes(f, V.SEV_FAIL)


def test_dynamic_dispatch_blocked():
    src = ("def run(x, o):\n    fn = __import__('flashinfer')\n"
           "    o.copy_(fn.run(x))\n")
    f = V.analyze(src, _sol(["triton"]), REQUIRE)
    assert "CX" in _codes(f, V.SEV_FAIL)


# --------------------------------------------------------------------------- #
# MUST-PASS (genuine kernels / honest labelling)
# --------------------------------------------------------------------------- #

def test_genuine_triton_passes():
    f = V.analyze(GENUINE_TRITON, _sol(["triton"]), REQUIRE)
    assert V.verdict(f) == "OK", [x.message for x in f]
    assert V.has_self_written_kernel(GENUINE_TRITON)


def test_genuine_with_glue_projection_is_warn_not_fail():
    # a real kernel + one F.linear glue -> WARN (needs justification), not FAIL
    f = V.analyze(GENUINE_WITH_PROJECTION, _sol(["triton"]), REQUIRE)
    assert V.verdict(f) == V.SEV_WARN
    assert "C1" not in _codes(f, V.SEV_FAIL)


def test_allow_libs_downgrades_honest_pytorch():
    src = ("import torch\nimport torch.nn.functional as F\n"
           "def run(x, w):\n    return F.linear(x, w)\n")
    f = V.analyze(src, _sol(["pytorch"]), ALLOW)
    assert V.verdict(f) == V.SEV_WARN   # libs allowed + honest tag -> not a hard fail


# --------------------------------------------------------------------------- #
# Opportunistic checks against the real on-disk artifacts
# --------------------------------------------------------------------------- #

REAL = {
    "067": ("/home/admin/SOL-ExecBench/kernel_opt_flash_attn_gqa/kernel.py",
            "/home/admin/SOL-ExecBench/data/benchmark/L1/067_flash_attention_gqa_ultralong/solution_triton.json"),
    "015": ("/home/admin/kernel_opt_gqa_paged_prefill/kernel.py", None),
    "control": ("/home/admin/SOL-ExecBench/kernel_opt_post_norm_residual/kernel.py", None),
}


def _maybe(path):
    return Path(path).read_text() if path and Path(path).exists() else None


def test_real_artifacts_if_present():
    import json
    k067 = _maybe(REAL["067"][0])
    if k067 is not None:
        sol = _maybe(REAL["067"][1])
        sol = json.loads(sol) if sol else _sol(["triton"])
        assert V.verdict(V.analyze(k067, sol, REQUIRE)) == V.SEV_FAIL
    k015 = _maybe(REAL["015"][0])
    if k015 is not None:
        f = V.analyze(k015, _sol(["cute_dsl"]), REQUIRE)
        assert V.verdict(f) == V.SEV_FAIL
        assert "C3" in _codes(f, V.SEV_FAIL)
    ctrl = _maybe(REAL["control"][0])
    if ctrl is not None:
        assert V.verdict(V.analyze(ctrl, _sol(["triton"]), REQUIRE)) == "OK"


# --------------------------------------------------------------------------- #
# Standalone runner
# --------------------------------------------------------------------------- #

def _run_standalone():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
