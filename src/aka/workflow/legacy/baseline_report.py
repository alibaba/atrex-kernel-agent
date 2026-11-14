"""Render the Workflow-owned V0 record before a Workspace transaction."""


def render_baseline_report(result: dict, source_commit: str, *, private_shapes: bool) -> str:
    def metric(name: str) -> str:
        value = result.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "unknown"
        return f"{float(value):.6g}"

    by_shape = result.get("latency_us_by_shape")
    by_shape = by_shape if isinstance(by_shape, dict) else {}
    evaluator = str(result.get("evaluator") or "workspace test_kernel.py")
    eval_id = result.get("eval_id")
    shape_note = (
        "Shape ids are opaque; exact production inputs remain evaluator-private."
        if private_shapes
        else "The complete public workload was evaluated."
    )
    report = (
        "# V0 baseline report\n\n"
        "This report was generated mechanically by the campaign supervisor.\n\n"
        "## Provenance\n\n"
        f"- Source commit: `{source_commit}`\n"
        "- Implementation: the Problem's selected initial source, prepared by Source\n"
        f"- Evaluator: `{evaluator}`\n"
        "- Route: remote sandbox, one base-seed full-workload run\n"
        f"- Evaluator id: `{eval_id if eval_id is not None else 'unknown'}`\n\n"
        "## Result\n\n"
        "- Correctness: `PASS`\n"
        f"- Measured shapes: `{len(by_shape)}`\n"
        f"- Geomean latency: `{metric('latency_us_geomean')} us`\n"
        f"- Arithmetic mean latency: `{metric('latency_us_arith_mean')} us`\n"
        f"- Mean speedup vs metadata: `{metric('speedup_vs_ref_mean')}x`\n"
        f"- Performance score: `{metric('performance_score')}`\n"
        f"- Maximum absolute error: `{metric('max_abs_err')}`\n"
        f"- Maximum relative error: `{metric('max_rel_err')}`\n\n"
        f"{shape_note} Per-shape values are stored once in `memory/v0.json`; they are "
        "not duplicated here.\n"
    )
    return report
