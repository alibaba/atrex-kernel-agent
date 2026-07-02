"""Setup-time anchor bench: measure the optimized-PyTorch layer baseline (Tb_layer) across
the full shape set and write the per-shape SOL-score weights into the layer manifest.

    w[s] = 1 / (Tb_layer[s] - SOL_layer[s])          (SOL-score sensitivity, per shape)

Tb_layer  = the official optimized-PyTorch solution (`solution.py`), benched here per shape.
SOL_layer = the fused-layer SOL, approximated as Σ_boundary SOL[b,s] from the manifest
            roofline bodies (compute is additive; the boundaries here are compute-bound so
            the intermediate-HBM over-count is negligible).

The scheduler (`optimize.py:_priority`) reads `manifest["shape_weights"]` and ranks each
boundary by mean_s w[s]·max(0, Tk[b,s]-SOL[b,s]). This makes priority track the single
official layer score instead of raw wall-clock ms (which the largest shape dominates).

Usage:
    python anchor_bench.py --op-dir <benchmark_op_dir> --manifest <boundaries.json> \
        --solution <solution.py> --platform B200
Inputs are built generically from <op-dir>/definition.json (axes + per-input shapes/dtypes).
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path

_TORCH_DTYPE = {"float32": "float32", "float16": "float16", "bfloat16": "bfloat16",
                "fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}


def _resolve_axes(defn: dict, shape_axes: dict) -> dict:
    """Resolve every axis name to a concrete int for one workload shape."""
    vals = dict(shape_axes)
    for name, spec in (defn.get("axes") or {}).items():
        t = spec.get("type")
        if t == "const":
            vals[name] = spec["value"]
        elif t == "var" and name not in vals:
            raise KeyError(f"var axis '{name}' missing from shape {shape_axes}")
    # expr axes may depend on const/var — resolve in a second pass
    for name, spec in (defn.get("axes") or {}).items():
        if spec.get("type") == "expr":
            vals[name] = int(eval(spec["expression"], {"__builtins__": {}}, dict(vals)))
    return vals


def _build_inputs(defn: dict, shape_axes: dict, device):
    import torch
    ax = _resolve_axes(defn, shape_axes)
    tdt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    args = []
    for name, spec in defn["inputs"].items():
        dims = [ax[d] if isinstance(d, str) else int(d) for d in spec["shape"]]
        dt = tdt[_TORCH_DTYPE.get(spec.get("dtype", "float32"), "float32")]
        t = torch.randn(*dims, device=device, dtype=dt) if dt.is_floating_point \
            else torch.zeros(*dims, device=device, dtype=dt)
        # keep projection weights small so fp32 matmuls stay numerically sane
        if "weight" in name and dt.is_floating_point:
            t = t * 0.01
        args.append(t)
    return tuple(args)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--solution", default="")
    ap.add_argument("--platform", required=True)
    # Match atrex-bench eval/performance.py canonical bench: do_bench(warmup, rep) where
    # warmup/rep are MILLISECOND budgets (not iteration counts). Defaults 10ms / 100ms.
    ap.add_argument("--warmup", type=int, default=10, help="do_bench warmup budget (ms)")
    ap.add_argument("--rep", type=int, default=100, help="do_bench measure budget (ms)")
    a = ap.parse_args()

    import torch
    op_dir = Path(a.op_dir)
    defn = json.loads((op_dir / "definition.json").read_text())
    workload = [json.loads(l) for l in (op_dir / "workload.jsonl").read_text().splitlines() if l.strip()]
    manifest_path = Path(a.manifest)
    manifest = json.loads(manifest_path.read_text())

    sol_path = Path(a.solution) if a.solution else (op_dir / "solution.py")
    spec = importlib.util.spec_from_file_location("anchor_solution", sol_path)
    sol = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(sol_path.parent))
    spec.loader.exec_module(sol)

    device = torch.device("cuda")
    from triton.testing import do_bench

    # SOL_layer[sid] = Σ_boundary roofline SOL_time_ms[platform]
    sol_layer: dict[str, float] = {}
    for b in manifest["boundaries"]:
        for sid, entry in ((b.get("roofline") or {}).get("shapes") or {}).items():
            ms = (entry.get("SOL_time_ms") or {}).get(a.platform)
            if isinstance(ms, (int, float)):
                sol_layer[sid] = sol_layer.get(sid, 0.0) + float(ms)

    weights, tb_layer = {}, {}
    for i, wl in enumerate(workload):
        sid = str(i)
        inputs = _build_inputs(defn, wl["axes"], device)
        # Same methodology as atrex-bench eval/performance.py: do_bench does the warmup
        # (warmup/rep are ms budgets), under torch.inference_mode(), fixed cloned inputs.
        with torch.inference_mode():
            tb = do_bench(lambda: sol.run(*inputs), warmup=a.warmup, rep=a.rep)  # ms
        tb_layer[sid] = tb
        denom = tb - sol_layer.get(sid, 0.0)
        weights[sid] = round(1.0 / denom, 6) if denom > 0 else 0.0  # official denom<=0 guard
        del inputs
        torch.cuda.empty_cache()
        print(f"sid {sid:>2} Tb_layer={tb:8.4f}ms SOL_layer={sol_layer.get(sid,0.0):7.4f}ms "
              f"w={weights[sid]:.4f}", flush=True)

    manifest["shape_weights"] = weights
    manifest["anchor_tb_layer_ms"] = {k: round(v, 6) for k, v in tb_layer.items()}
    manifest["sol_layer_ms"] = {k: round(v, 6) for k, v in sol_layer.items()}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[anchor] wrote shape_weights for {len(weights)} shapes -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
