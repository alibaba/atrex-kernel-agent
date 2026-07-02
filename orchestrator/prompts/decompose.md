# Layer decomposition (clean session, run once)

You are the **decomposition session** for a composite-operator optimization campaign. The input is a composite
of more than one fused op — a whole LLM layer, or a smaller multi-op composite such as `rope+attention` or
`attention+moe`. Your job is to carve it into fused-operator **boundaries** and emit one basic fused kernel per
boundary — then stop. The orchestrator fans each boundary out into its own workspace afterwards.

Read and follow **`{{DECOMPOSE_DOC}}`** (the fusion-boundary decomposition rules) exactly — the gate, the
fuse/split rules, the canonical boundary catalog, and the output contract.

Inputs:
- `layer_logic`: the composite op / layer PyTorch module to decompose — `{{LAYER_DEMO}}`
- `layer_dir` (your cwd): `{{LAYER_DIR}}` — write all outputs here
- `platform`: `{{PLATFORM}}`
- `roofline_py` (per-boundary SOL): `{{ROOFLINE_PY}}`
- `workload` (the full shape set — every shape the op is scored on; convert each entry to an integer-sid
  `shapes.json` entry): `{{WORKLOAD}}`
- gpu-wiki (operator knowledge only): `{{GPU_WIKI}}`
- additional_notes: `{{NOTES}}`

{{HARDWARE}}

Do exactly this, then STOP:

1. **Apply the gate** (§0 of the rules). If the input is really a single operator, emit a `boundaries.json`
   with a **single** boundary equal to the whole input (the orchestrator will then run it as an ordinary
   single-op campaign). Otherwise decompose.
2. **Draw the boundaries** per the catalog (§1–§2), adjusted to the actual module dataflow.
3. **Emit the deliverables** (§4) into `{{LAYER_DIR}}`:
   - `reference.py` — the full-layer PyTorch reference (for the final end-to-end recombine validation).
   - `<boundary>/kernel_demo.py` — one basic, correct, runnable PyTorch reference per boundary.
   - `shapes.json` — the layer's full shape set in **atrex-bench format**: `{"0": {"init_kwargs": null,
     "input_kwargs": {…axes…}}, "1": {…}, …}`, integer string sids `"0","1",…` (convert every entry of
     `{{WORKLOAD}}` — do not hand-pick). This is the ground-truth bench set for every boundary.
   - per boundary, a **`roofline.json`** in atrex-bench format:
     `{"shapes": {"<sid>": {"semantic_W_flops": {"<dtype>": W}, "semantic_Q_read_bytes": …,
     "semantic_Q_write_bytes": …, "SOL_time_ms": {"{{PLATFORM}}": ms}}}}` — run `{{ROOFLINE_PY}}` on that
     boundary for **every** sid in `shapes.json`. SOL is **per-shape** (there is no single "representative"
     shape and no scalar `sol_time_ms`) because op cost varies with the axes — attention ∝ B·S², so one
     shape's SOL is meaningless for the rest. Use the operator's **declared dtype**, and for **causal**
     attention count causal FLOPs (~½ the dense S×S), not the full matrix.
   - `boundaries.json` — the manifest: dataflow-ordered boundaries (`name`, `op_type`, `kernel_demo`,
     `dtype`, `bound`, `ceiling` per §5, and each boundary's `roofline` body), plus the layer-level
     `shapes` (the shapes.json body). The orchestrator materializes `shapes.json` + `roofline.json` into
     each boundary workspace from this manifest; sids are the join key across shapes.json, roofline.json,
     and each version's `performance.latency_us_by_shape`.
   - `decomposition.md` — the evidence chain for each fuse/split decision (`op(s) -> decision -> why`).
4. Sanity-check that every `kernel_demo.py` runs and matches the corresponding slice of `reference.py`.

Then **STOP**. Do not create workspaces, do not implement framework kernels, do not optimize — the orchestrator
handles all of that. Exit once `boundaries.json` exists and lists at least one boundary.
