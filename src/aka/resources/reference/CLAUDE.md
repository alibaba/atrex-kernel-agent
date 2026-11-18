# Kernel engineering constraints

The Session Prompt defines the current goal and finish condition. `WORKSPACE.md` defines the
selected SourceProvider layout and editable candidate paths. Installed Skills define Runtime
operations and request schemas. Do not infer a different workflow from unrelated files.

## Candidate correctness

- Preserve the evaluator-facing entry point, arguments, outputs, dtypes, layouts, and declared side
  effects. Compute every call from its current inputs; do not cache answers, pointer identities, or
  benchmark-specific results, and do not embed generated tensor values. Reusable plans may depend on shape, dtype, layout, and device only.
- Keep computation on the default stream. Do not alter timing, RNG, input generation, tolerances,
  metadata, correctness checks, or evaluator behavior.
- Read `metadata.json` when present. Every configured correctness condition must pass; one passing
  metric never overrides another failure. Tolerances are safety margins, not optimization targets.
- Use the injected hardware identity. Verify architecture-specific assumptions rather than
  transferring them from another device.

## Workspace and Git

- Edit only the SourceProvider-declared candidate. The default source tree uses `kernel.py` plus
  optional files under `impl/`; other providers may declare a single file or a native manifest.
  Keep evaluator inputs, public contracts, `CLAUDE.md`, and `README.md` unchanged. Edit
  `solution.json` only when `WORKSPACE.md` declares it candidate metadata.
- Use the trajectory's local Git repository for checkpoints, branches, comparison, and rollback.
  Before every Kernel-bound Runtime request, commit the complete candidate and leave the worktree
  clean. Runtime never commits Agent work on your behalf. The full HEAD is `kernel_commit_id`; every
  Kernel commit ID shown by Runtime is from this repository or imported peer Git, never Supervisor Git.
  A commit is source identity, not measurement evidence.
- In exposed mode, `shapes.json` is the authoritative public evaluator workload. In hidden mode,
  exact Shape parameters are private and Runtime result Shape IDs are opaque. Never infer or search
  for private cases. Shape-subset measurements are diagnostic, not full-contract evidence.
- Use `scratch/` for temporary probes and outputs. Installed `tools/` and `skills/`, Runtime
  projections, public inputs, and historical evidence are Supervisor-owned; do not edit them.
  Do not create extra plan, profile, or benchmark documents unless the Prompt or a Skill requires
  one.

## Runtime and evidence

- The local Agent workspace is CPU-only. Do not run or import the candidate locally or probe for a
  local GPU/toolchain with `torch`, `cuda`, `nvidia-smi`, `nvcc`, `ptxas`, `ncu`, or equivalents.
  Use `python3 tools/sandbox.py` and `skills/gpu-measurement/SKILL.md` for compilation, execution,
  evaluation, profiling, and diagnostics. Do not invoke Agate directly or obtain its credentials.
- `python3 tools/sandbox.py --help` lists the enabled operations. Typed Evaluate selects managed
  inputs automatically. Use an enabled Command for custom remote probes; it uploads the engineering
  workspace and returns scratch artifacts. Remote jobs are fresh and stateless, and custom probes
  do not replace official evaluation evidence.
- Follow the Session Prompt's Runtime execution policy. When Agent suspension is disabled, independent
  work may continue while a request is pending. Keep its input files unchanged and collect the same
  invocation's receipt before using its results; use the backend's task/output facility for background
  calls. Do not add shell timeouts or duplicate requests. A lost client does not prove that no job ran.
- Runtime measurement calls return the complete canonical public result and persist the same result as
  `records/<gateway_record_id>.json`. The file is durable evidence, not a second source that must be
  opened to recover fields omitted from the Runtime response.
  Completion does not imply correctness. Reuse matching Records, follow `error.next_action`, and
  do not add independent retry loops or resubmit an unknown outcome.
- Env is read-only discovery. Its complete redacted capability data uses the same canonical
  Runtime result contract and is archived in the central Store; it is not correctness/performance evidence.
- Root `PROGRESS.md` is a compact read-only index of the incumbent, measured frontier, and enabled
  research state that Runtime regenerates after accepted state changes. Direction and Experiment files contain fallible hypotheses and interpretations;
  Gateway Records remain measurement authority.

## Research and commit review

- Read `skills/runtime-records/SKILL.md` for the research modules and request schemas enabled in this
  Episode. Do not call disabled Direction or Experiment operations or edit their projected files.
- The persistent optimization Session starts directly from V0. Bring-up, diagnostics, correctness
  work, implementation and optimization all happen in this one conversation and Agent Git history.
- `submit-kernel` reviews whether an evaluated commit satisfies the selected final-source policy.
  Fast permits declared reproducible compute libraries; Performance requires self-authored core
  compute with Agent-selected DSL; Ultra additionally enforces its configured DSL. Repair and
  evaluate a new commit when the review returns violations.
