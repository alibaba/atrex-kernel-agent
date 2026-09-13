# Kernel engineering constraints

The injected session Prompt defines the task, implementation constraints, permitted operations,
and finish procedure. Do not assume another DSL, library, or validation procedure is allowed.

## Correctness and measurement integrity

- Preserve the evaluator-facing entry point, arguments, outputs, dtypes, layouts, and declared
  side effects. For SOL destination-passing `run()`, arguments are definition inputs followed
  by definition outputs.
- Compute each call's outputs from its current arguments. Inputs may have fresh values and
  addresses: no answer caching, input-value or pointer-identity caching, benchmark detection,
  or input-dependent work amortized across repeated calls.
- Reusable compilation and launch plans may depend on shape, dtype, layout, and device, not
  pointer identity. Reused scratch must remain valid for the current call. CUDA graphs must
  rebind current tensor pointers rather than replay captured stale addresses.
- Never modify timing, RNG, input generation, correctness checks, or evaluator behavior.
  Custom probes are diagnostic evidence, not replacements for official evaluation; Profile
  kernel duration is not end-to-end Evaluate latency.
- Keep computation on the default stream. Do not create or synchronize extra streams to
  overlap work or hide it from timing.
- Tolerances are safety margins, not optimization targets. Investigate results near a tolerance
  boundary and record unresolved numerical risks. Follow the session's validation procedure;
  do not add tests forbidden in that phase. Repair an invalid inherited Kernel instead of
  preserving an apparently fast result; record the finding when the phase uses the Journal.
- Use the injected hardware identity and architecture. Do not transfer architecture-specific
  assumptions to another device without checking applicability.

## Workspace and evidence

- The execution boundary defines writable paths. Keep the executable candidate in `kernel.py`;
  update `solution.json` only where the phase permits. Do not edit public contracts or canonical
  memory, or use Git for rollback.
- Cover the entire public input domain. Opaque Shape IDs identify results, not input parameters;
  do not reconstruct hidden cases or search outside the workspace for private inputs.
- Use `scratch/` for requests, diagnostic probes, and saved exact Kernel copies. `tools/` is
  Episode-local and writable. The session Prompt defines fresh-session and recovery behavior.
- Gateway Records contain measured facts; Journal entries contain hypotheses, changes, and
  conclusions. Past Agent analysis may be wrong. Optimize the evaluator's `performance_score`;
  use per-Shape latency for diagnosis and obey the current phase's acceptance criteria.

## Runtime tools

Use `python3 tools/sandbox.py` for GPU, Wiki, history, Journal, and report requests. The session
Prompt lists mounted resources. These Skills are always available:

- `skills/gpu-measurement/SKILL.md`: GPU operations and request examples.
- `skills/runtime-records/SKILL.md`: historical results, Kernel source, Journal, and report formats.
- `skills/KernelWiki/SKILL.md`: knowledge queries.

Reuse saved results and exact source when applicable. Runtime persists measurements, retries
infrastructure failures, and returns concise results; do not implement another retry loop or
modify its private records. Editing the HTTP client does not change Runtime authority.

On failure, follow `error.next_action`. `repairable: true` means fix the request or Journal state,
not repeat an unchanged call. If the outcome is unknown, check records before resubmitting;
private-state or dependency failures may need operator help. Correct rejected `episode-report`
requests and retry until accepted, then stop. A Baseline session has its own smoke-and-exit
procedure and does not submit an Episode report.
