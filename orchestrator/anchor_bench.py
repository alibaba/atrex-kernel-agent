"""Setup-time anchor weights: produce the per-shape SOL-score weights used by the layer
scheduler to rank boundaries by their gradient on the single official layer score:

    w[s] = 1 / (Tb_layer[s] - SOL_layer[s])

Both terms are **prefer-read, else self-compute** — the atrex-bench native op dir already
ships most of it, so we read it and only fall back to measuring/summing when a value is absent:

  SOL_layer[s]  ← <op-dir>/roofline.json shapes[s].SOL_time_ms[<platform>]   (fused-layer SOL)
                  else Σ_boundary roofline SOL[b,s] from the manifest.
  Tb_layer[s]   ← <op-dir>/metadata.json shapes[s].production_performance.performance_us
                  else bench the reference (input.py + reference.Model) with the atrex-bench
                  bench.py methodology: median-of-3 × do_bench(warmup=25, rep=100), no_grad.

The op dir is passed in (never hardcoded). It is the atrex-bench native layout:
shapes.json / roofline.json / metadata.json / input.py / reference.py.

Usage:
    python anchor_bench.py --op-dir <native_op_dir> --manifest <boundaries.json> --platform B200
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path

# Mirror optimize.py: roofline/metadata may key by a verbose SKU string; try aliases too.
_SOL_PLATFORM_KEY_ALIASES = {
    "B200": ["NVIDIA B200 (SM100)"], "B300": ["NVIDIA B300 (SM100)"],
    "H20": ["NVIDIA H20"], "A100": ["NVIDIA A100"], "MI308X": ["AMD MI308X"],
}


def _keys(platform: str):
    return [platform, *_SOL_PLATFORM_KEY_ALIASES.get(platform, [])]


def _pick(block: dict, platform: str):
    for k in _keys(platform):
        v = (block or {}).get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(m)
    return m


def _median_us(fn) -> float:
    """atrex-bench bench.py methodology: median of 3 × do_bench(warmup=25, rep=100)."""
    import torch
    from triton.testing import do_bench
    fn(); torch.cuda.synchronize()
    return float(sorted(do_bench(fn, warmup=25, rep=100) for _ in range(3))[1]) * 1e3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op-dir", required=True, help="atrex-bench native op dir (shapes.json/roofline.json/"
                                                    "metadata.json/input.py/reference.py)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--platform", required=True)
    a = ap.parse_args()

    op = Path(a.op_dir)
    shapes = json.loads((op / "shapes.json").read_text())
    manifest_path = Path(a.manifest)
    manifest = json.loads(manifest_path.read_text())

    # ── SOL_layer[sid]: prefer op-dir roofline.json, else Σ boundary SOL from manifest ──
    op_rf = {}
    if (op / "roofline.json").exists():
        op_rf = (json.loads((op / "roofline.json").read_text()).get("shapes") or {})
    boundary_sol_sum: dict[str, float] = {}
    for b in manifest["boundaries"]:
        for sid, e in ((b.get("roofline") or {}).get("shapes") or {}).items():
            ms = _pick(e.get("SOL_time_ms") or {}, a.platform)
            if ms is not None:
                boundary_sol_sum[sid] = boundary_sol_sum.get(sid, 0.0) + ms

    def sol_layer(sid: str):
        v = _pick((op_rf.get(sid) or {}).get("SOL_time_ms") or {}, a.platform)
        return v if v is not None else boundary_sol_sum.get(sid)

    # ── Tb_layer[sid]: prefer metadata production_performance, else bench the reference ──
    meta_shapes = {}
    if (op / "metadata.json").exists():
        meta_shapes = (json.loads((op / "metadata.json").read_text()).get("shapes") or {})

    def tb_from_metadata(sid: str):
        pp = (meta_shapes.get(sid) or {}).get("production_performance") or {}
        v = pp.get("performance_us")
        return (float(v) / 1000.0) if isinstance(v, (int, float)) else None  # us -> ms

    need_bench = any(tb_from_metadata(sid) is None for sid in shapes)
    model = mk = None
    if need_bench:
        import torch
        inp = _load(op / "input.py", "_anchor_inp")
        ref = _load(op / "reference.py", "_anchor_ref")
        model = ref.Model().to("cuda").eval()
        mk = inp._make_inputs

    weights, tb_layer, src = {}, {}, {}
    for sid in shapes:
        tb = tb_from_metadata(sid)
        if tb is not None:
            src[sid] = "metadata"
        else:
            import torch
            try:
                ins = mk(**shapes[sid]["input_kwargs"])
                with torch.no_grad():
                    tb = _median_us(lambda: model(**ins))
                src[sid] = "benched"
                del ins
            except Exception as e:  # e.g. OOM benching the eager reference at large shapes
                tb = None
                src[sid] = f"skip({type(e).__name__})"
            torch.cuda.empty_cache()
        if tb is None:
            weights[sid] = 1.0  # no anchor -> unweighted for this shape
            tb_layer[sid] = None
            print(f"sid {sid:>2} Tb=  n/a    ({src[sid]}) -> w=1.0", flush=True)
            continue
        sl = sol_layer(sid)
        tb_layer[sid] = round(tb, 6)
        if sl is None:
            weights[sid] = 1.0  # no SOL_layer available -> unweighted
        else:
            denom = tb - sl
            weights[sid] = round(1.0 / denom, 6) if denom > 0 else 0.0  # official denom<=0 guard
        print(f"sid {sid:>2} Tb={tb:8.4f}ms ({src[sid]:8s}) SOL_layer="
              f"{(sl if sl is not None else float('nan')):7.4f}ms w={weights[sid]:.4f}", flush=True)

    manifest["shape_weights"] = weights
    manifest["anchor_tb_layer_ms"] = tb_layer
    manifest["anchor_source"] = src
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[anchor] wrote shape_weights for {len(weights)} shapes -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
