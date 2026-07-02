"""Setup-time SOL-score weights for the layer scheduler:

    w[s] = 1 / (Tb_layer[s] - SOL_layer[s])

Tb_layer[s] must be a REAL optimized baseline — read ONLY from
`metadata.production_performance.performance_us`. We deliberately do NOT fabricate it:
benching the eager `reference.py` would give an O(S^2) time of tens of seconds at long
sequence lengths, which is not an "optimized baseline" and produces meaningless (~1e-5)
weights. So if any shape lacks a production baseline (e.g. not yet measured on the target
GPU), we skip weighting ENTIRELY — no `shape_weights` is written and the orchestrator
falls back to the unweighted raw ms-gap priority (`mean_s max(0, lat[s]-sol[s])`).

SOL_layer[s]: prefer `<op-dir>/roofline.json`, else Σ_boundary SOL from the manifest.

This is a pure JSON transform (no GPU / no torch); run once at setup.

Usage:
    python anchor_bench.py --op-dir <native_op_dir> --manifest <boundaries.json>
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def _pick(block):
    """SOL (ms) from a SOL_time_ms block, ignoring the platform label (single-platform campaign)."""
    if isinstance(block, (int, float)):
        return float(block)
    vals = [v for v in (block or {}).values() if isinstance(v, (int, float))]
    return float(vals[0]) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op-dir", required=True)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()

    op = Path(a.op_dir)
    shapes = json.loads((op / "shapes.json").read_text())
    manifest_path = Path(a.manifest)
    manifest = json.loads(manifest_path.read_text())

    # SOL_layer[sid]: prefer op-dir roofline.json, else Σ boundary SOL from the manifest.
    op_rf = {}
    if (op / "roofline.json").exists():
        op_rf = (json.loads((op / "roofline.json").read_text()).get("shapes") or {})
    boundary_sol_sum: dict[str, float] = {}
    for b in manifest["boundaries"]:
        for sid, e in ((b.get("roofline") or {}).get("shapes") or {}).items():
            ms = _pick(e.get("SOL_time_ms") or e.get("sol_time_ms"))
            if ms is not None:
                boundary_sol_sum[sid] = boundary_sol_sum.get(sid, 0.0) + ms

    def sol_layer(sid):
        v = _pick((op_rf.get(sid) or {}).get("SOL_time_ms") or (op_rf.get(sid) or {}).get("sol_time_ms"))
        return v if v is not None else boundary_sol_sum.get(sid)

    # Tb_layer[sid]: ONLY from metadata production baseline (us -> ms). No benching.
    meta_shapes = {}
    if (op / "metadata.json").exists():
        meta_shapes = (json.loads((op / "metadata.json").read_text()).get("shapes") or {})

    def tb(sid):
        pp = (meta_shapes.get(sid) or {}).get("production_performance") or {}
        v = pp.get("performance_us")
        return (float(v) / 1000.0) if isinstance(v, (int, float)) else None

    # All-or-nothing: weight only if EVERY shape has both a production Tb and a SOL_layer.
    weights, missing = {}, []
    for sid in shapes:
        t, sl = tb(sid), sol_layer(sid)
        if t is None or sl is None:
            missing.append(sid)
            continue
        denom = t - sl
        weights[sid] = round(1.0 / denom, 6) if denom > 0 else 0.0  # official denom<=0 guard

    if missing:
        for k in ("shape_weights", "anchor_tb_layer_ms", "anchor_source"):
            manifest.pop(k, None)  # clear any stale weights
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[anchor] no production baseline for {len(missing)}/{len(shapes)} shapes "
              f"(e.g. sids {missing[:5]}) -> UNWEIGHTED priority (raw ms-gap). "
              f"Populate metadata.production_performance to enable SOL-score weighting.", flush=True)
        return 0

    manifest["shape_weights"] = weights
    manifest["anchor_tb_layer_ms"] = {sid: round(tb(sid), 6) for sid in shapes}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[anchor] SOL-score weights for {len(weights)} shapes (Tb from metadata) -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
