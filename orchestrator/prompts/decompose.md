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
- `op_dir` (the atrex-bench native op dir — **the starting point**): `{{OP_DIR}}`
  Read the full shape set from `{{OP_DIR}}/shapes.json` (integer-sid, `input_kwargs` axes) and reuse those
  sids verbatim. `{{OP_DIR}}/roofline.json` holds the layer-level SOL; `input.py` builds inputs.
- gpu-wiki (operator knowledge only): `gpu-wiki/`
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
   - `shapes.json` — copy `{{OP_DIR}}/shapes.json` verbatim (atrex-bench format: integer string sids
     `"0","1",…`, axes under `input_kwargs`). Keep the sids identical — they are the join key across
     shapes.json, every boundary's roofline.json, and each version's `latency_us_by_shape`. Do not
     re-number or hand-pick shapes.
   - per boundary, a **`roofline.json`**:
     `{"shapes": {"<sid>": {"semantic_W_flops": {"<dtype>": W}, "semantic_Q_read_bytes": …,
     "semantic_Q_write_bytes": …, "sol_time_ms": <ms>}}}` — run `{{ROOFLINE_PY}}` on that boundary for
     **every** sid in `shapes.json`. Store SOL as a plain per-shape `sol_time_ms` number (the campaign
     targets one platform `{{PLATFORM}}`, so do **not** nest it under a platform key). SOL is **per-shape**
     (no single "representative" shape) because op cost varies with the axes — attention ∝ B·S², so one
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
