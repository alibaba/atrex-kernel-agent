# Framework baseline

Implement this operator in self-contained `{{FRAMEWORK}}` code in `kernel.py`.
V0 is the reference wrapper; the Supervisor registers the accepted implementation as v{{N}}.
Preserve the full public contract. Correctness and implementation compliance are required;
speedup is evaluated in later Episodes. Work autonomously through measured defects.

## Context

- Workspace: `{{WORKSPACE}}`
- Baseline: v{{PREV}}; candidate: v{{N}}
- Platform: `{{PLATFORM}}`; framework: `{{FRAMEWORK}}`
- Constraints: {{NOTES}}
{{AGENT_RUNTIME}}
{{PLUGINS}}
{{HARDWARE}}
{{SANDBOX}}
{{EVALUATOR}}
{{CORRECTNESS_GUIDANCE}}

## Read, implement, measure

1. Read `README.md`, the current `kernel.py`, `solution.json` when present, and the public
   contract/reference files named by the task. For native problems these are `reference.py`,
   `input.py` and `agent_problem.json`; for SOL, `definition.json` and `workload.jsonl`.
   Use `memory/v0.json` only for a missing baseline fact. On resume, inspect preserved work
   and continue from its concrete next step.
2. Choose one implementation covering the public domain, including layouts, padding, mutation
   and numerical requirements. Then edit the candidate. Do not repeatedly reread unchanged
   files, explore evaluator internals, narrate future work, or create planning artifacts.
3. Run the prescribed smoke after the implementation edit. Let its measured diagnostics
   drive focused repairs; keep working code and the tested loader intact.

{{CUDA_STARTER}}

Do not build a local NumPy/Python replica, generic simulator, extra benchmark or diagnostic
`__main__` in the candidate. A small targeted calculation is appropriate only when a measured
counterexample leaves a specific indexing/numerical question unresolved; it never replaces
the evaluator. Do not install host GPU libraries or inspect the host toolchain to infer the
remote worker's capabilities. No profiling, ABBA or optimization sweep in V1.

## Source contract

Implement the math yourself in the selected framework. Dependencies may provide allocation,
compiler/ABI/launch plumbing, not prebuilt operator math or PyTorch compute fallbacks. For CUDA,
embed source and its in-process loader in `kernel.py`. Keep `solution.json` consistent while
editing. For Triton use plain Triton; Gluon conversion is a later explicit request.

Candidate source must not perform file/network/process access, dynamic imports, reflection,
or environment inspection. In particular, avoid `os`, `sys`, `ctypes`, `cffi`, `subprocess`,
`pathlib`, `importlib`, `open`, `eval`, `exec`, `getattr`, `hasattr`, `torch.load/save`,
`torch.cuda.set_device`, and a `__main__` block. Do not hide forbidden behavior with encoding
or alternate spellings. Keep examples, probes and debugging helpers out of the candidate.

## Resolve only missing facts

{{KNOWLEDGE_RESEARCH}}

## Smoke and finish

```bash
{{SMOKE_COMMAND}}
```
{{SMOKE_SCOPE}}

Follow an explicit Supervisor repair scope when supplied; otherwise keep this smoke scope.
Do not add multi-seed, a separate benchmark or an alternate Dev path.
On failure, inspect the saved `result.actionable_diagnostics` / `result.error`, fix the specific
defect and rerun this command with changed candidate bytes. If details are missing, read the
saved record once; report an unresolved diagnostic with its Record ID instead of guessing API
signatures, rewriting the loader repeatedly or bisecting source filters. Runtime owns
infrastructure retries. Unknown outcomes require reconciliation, not another submission.

After smoke passes, keep those exact candidate bytes, print
`v{{N}}: framework candidate smoke-passed ({{FRAMEWORK}})`, and stop. Do not add another static
audit, edit the measured candidate, commit, write canonical memory, or submit `episode-report`.

The Supervisor still runs full-domain multi-seed correctness, performance measurement,
implementation-policy review and production supplemental numerical validation. Smoke is not
acceptance. Repair any measured counterexample without changing references, probes, thresholds
or coverage; the Supervisor repeats the required gates before accepting v{{N}}.
