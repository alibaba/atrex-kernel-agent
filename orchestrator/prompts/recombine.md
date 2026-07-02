# Layer recombine + validate (clean session, run once at the end)

You are the **recombine session**. Every boundary of this LLM layer has been optimized in its own workspace;
each workspace's **git HEAD `kernel.py` is that boundary's best kernel**. Your job is to stitch them back into
one full-layer kernel and validate end-to-end — then stop.

Inputs:
- `layer_dir` (your cwd): `{{LAYER_DIR}}` — holds `boundaries.json` and `reference.py`.
- `boundaries.json` — dataflow-ordered boundaries; each entry's `workspace` field is the absolute path to that
  boundary's optimized workspace (its HEAD `kernel.py` is the best kernel).

{{HARDWARE}}

Do exactly this, then STOP:

1. Read `boundaries.json` and `reference.py`.
2. For each boundary in dataflow order, take its workspace's HEAD `kernel.py` (the best optimized kernel).
3. Assemble `{{LAYER_DIR}}/kernel.py`: a single full-layer entry point (`run(...)` / `Model`) that wires the
   per-boundary kernels together following `reference.py`'s dataflow. Do **not** re-fuse across boundaries or
   re-optimize — just compose the fixed boundaries.
4. **Validate end-to-end**: correctness of `{{LAYER_DIR}}/kernel.py` vs `reference.py` (use the same tolerance
   the boundaries used, bf16 default `rel_err < 0.01`), with a timeout guard. Then measure full-layer latency.
5. Write `{{LAYER_DIR}}/recombine_report.md`: per-boundary source workspace + version, full-layer correctness
   (max `rel_err`, PASS/FAIL), and full-layer latency.

Then **STOP**. Print one line: `recombine: PASS (<latency>)` or `recombine: FAIL (<reason>)`.
