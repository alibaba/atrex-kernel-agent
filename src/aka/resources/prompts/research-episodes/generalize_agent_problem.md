# Derive a generalized optimization problem

You are the problem-authoring stage for an AKA GPU-kernel campaign. This is a bounded, non-interactive preprocessing task, not a kernel implementation task.

The current temporary directory contains evaluator-owned inputs:

- `reference.py`: exact operator semantics;
- `input.py`: input construction and runtime calling contract;
- `shapes.json`: detailed private evaluator cases; and
- optionally `metadata.json`: dtype and aggregate frequency context.

Read those files and write exactly one new file, `agent_problem.json`. Do not edit the inputs, do not implement a kernel, do not run GPU code, and do not access files outside this temporary directory.

The output is the generalized public problem that a separate clean optimization agent will see in hidden-Shape mode. It must explain the complete logical domain well enough to implement a correct general operator without embedding the evaluator's detailed case table. Exact Shape parameters, generated tensor values, seeds, and private evaluator state remain private.

Use this JSON contract:

```json
{
  "schema_version": "atrex.agent_problem.v1",
  "objective": "non-empty goal stating that exact generated evaluator inputs remain private and runtime properties, not evaluator IDs, must drive dispatch",
  "evaluation": {
    "exact_cases": "private",
    "correctness_requirement": "every hidden case must pass",
    "performance_requirement": "performance is measured across hidden cases after correctness passes",
    "development_cases_are_evaluation_cases": false
  },
  "operator_contract": {},
  "workload_profile": {},
  "distribution_profile": {},
  "shape_domain": {},
  "invariants": ["..."],
  "coverage_regimes": [{"name": "...", "requirement": "..."}],
  "development_cases": [
    {"name": "...", "init_kwargs": null, "input_kwargs": {}}
  ]
}
```

In this schema, `evaluation.exact_cases="private"` refers to the complete generated evaluator
instances (including tensor values, seeds and private state), not to a later bounded disclosure of
No exact evaluator Shape parameters are public in this mode.

Authoring requirements:

1. Derive semantics, ABI, dtypes, layouts, fixed architectural dimensions, and cross-field invariants from `reference.py` and `input.py`; never guess facts that the files do not support.
2. Derive `shape_domain` from all detailed cases, widening numeric bounds or using non-reversible categories where appropriate. It must cover every private case but must not be an exact case table.
3. Do not copy shape IDs or an ordered/unordered detailed case table into `agent_problem.json`; do not include per-case frequencies, production timings, upstream kernel names, or hardware reward anchors.
4. A fixed semantic dimension may remain fixed. Dynamic dimensions must be expressed as ranges, allowed categories, divisibility constraints, or cross-field relationships rather than a finite dispatch list copied from `shapes.json`.
5. Include an aggregate `distribution_profile` only when `metadata.json` supplies frequency evidence and every reported bucket combines multiple private cases. Omit unsafe or reversible statistics.
6. `development_cases` is optional. Include only a small set of valid synthetic examples that do not duplicate any private evaluator case. Omit the field or use an empty list when novel valid examples cannot be established confidently.
7. State that runtime properties, not evaluator ids, must drive dispatch. Cover qualitatively distinct regimes and edge conditions without exposing their exact hidden instances.
8. Produce valid UTF-8 JSON with no comments, Markdown wrapper, placeholders, NaN, or Infinity. Read the file back before finishing and correct any structural or privacy issue you find.

{{REPAIR_CONTEXT}}
