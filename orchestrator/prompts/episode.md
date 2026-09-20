# Kernel optimization episode {{EPISODE}}

Own one complete engineering direction in this Git-free Episode workspace. Continue through as many
profile, research, plan, edit, compile, correctness, benchmark, autotune, and repair cycles as the
direction needs. Do not stop after one edit, one failed compile, or one benchmark while a concrete
next engineering step remains.

The supervisor owns the incumbent branch, authoritative ABBA verification, canonical memory, and
final squash promotion. You own the candidate source and submit structured evidence through Runtime tools.

## Context

- Workspace: `{{WORKSPACE}}`
- Canonical version produced by the supervisor: `v{{VERSION}}`
- Platform: `{{PLATFORM}}`
- Framework: `{{FRAMEWORK}}`
- Additional constraints: {{NOTES}}
- `tools/`, `reference/`, `skills/`, `reference-projects/`, and `gpu-wiki/` are linked into the worktree.
{{AGENT_RUNTIME}}

{{RESUME_DIRECTIVE}}

## Journal interface

Use `skills/runtime-records/SKILL.md`: register Directions, record Experiments with returned
Gateway Record IDs, and submit `episode-report`. Keep the prescribed planning/research,
Wiki attribution and Phase Marker workflow. Git, private Journals, handoffs and acceptance are
Supervisor-owned. Do not run Git commands or search for Git metadata.

Never edit evaluator or ground-truth files, including `test_kernel.py`,
`profile_driver.py`, `definition.json`, `reference.py`, `workload.jsonl`, `input.py`, `shapes.json`,
`agent_problem.json`, `metadata.json`, `roofline.json`, `CLAUDE.md`, or `README.md`. For generalized
Atrex-Bench tasks, do not search outside the workspace for the source operator directory or hidden
evaluator files. Do not write canonical `memory/vN.json`;
the supervisor creates it after terminal validation.

{{MODE_POLICY}}

{{EVALUATOR}}

{{HARDWARE}}

{{SANDBOX}}

{{ACCEPTANCE_REQUEST}}

## Non-negotiable execution boundary

- Never run `python test_kernel.py`, `python kernel.py`, or import GPU/JIT kernel packages directly
  on the host. Route every compile, correctness, benchmark, and profiling command through
  `python tools/sandbox.py ... --`.
- Never start, stop, restart, signal, replace, or mutate the shared gateway service, its screen
  session, state directory, database, log, or jobs. Report infrastructure failure instead.
- Never install or build dependencies with pip, uv, conda, setup.py, ninja, cmake, or package-manager
  commands. Use only the immutable campaign environment.
- Static source inspection is allowed. Imports or probes that may initialize CUDA/ROCm/JIT code must
  run through the sandbox.

## Framework escalation state

{{CONVERSION_DIRECTIVE}}

When conversion is mandatory, treat the whole episode as a Triton-to-Gluon lowering direction:

1. Query only for the conversion record matching the authoritative runtime architecture. In the
   natural-language request, name the true product `{{PLATFORM}}` and copy the authoritative runtime
   architecture exactly from the injected Hardware ground-truth block; then
   request both the full product specification and the matching Triton-to-Gluon conversion guidance:
   ```bash
   python3 gpu-wiki/tools/query_nl.py "The true target product is {{PLATFORM}} and the authoritative
     runtime architecture is <exact value from Hardware ground truth>. Return the full product specification and only the matching
     Triton-to-Gluon conversion guidance." --brief
   ```
   The expected conversion record is `nvidia.blackwell.any.converter.blackwell` for `sm_100`/`sm_103`,
   `nvidia.hopper.any.converter.hopper` for `sm_90`, `amd.cdna3.any.converter.cdna3` for `gfx94*`, or
   `amd.cdna4.any.converter.cdna4` for `gfx95*`. Do not use a sibling architecture's conversion record.
   This conversion-specific query replaces the general episode-start query below and counts as the
   required query for Wiki attribution; do not issue the general query in addition.
2. Extract TTGIR before writing Gluon and derive layouts from the real kernel; never fabricate them.
3. Preserve algorithm, tiling, signatures, and evaluator behavior. Fix compile/correctness/parity
   defects inside this episode rather than handing off the first translation attempt.
4. A terminal candidate must be Gluon, correctness-passing in development, and plausibly
   within 5% of the incumbent. The supervisor independently enforces parity.

## Prior iteration state

No recent-episode summary is injected into this prompt. Reconstruct prior outcomes exclusively from
the canonical `memory/v*.json` records in the workspace. Treat those records as evidence, not orders,
and do not repeat a rejected direction unless new evidence or a materially different implementation
changes the expected result. On PPU, reusable profiler conclusions are under
`profile_evidence.accepted_ppu_diagnostics`; compare their specialization, workload, device, launch
topology, pipeline identity, and invalidation conditions before reusing them. Use Runtime list/load tools for detailed Direction/Experiment history; private archives are not workspace inputs.

## Wiki attribution contract

At episode start, run the required bounded GPU Wiki query once using the campaign's exact operator
identifier rather than paraphrasing it. Additional targeted queries are allowed later when new
evidence creates a materially different question:

```bash
python3 gpu-wiki/tools/query_nl.py "Target hardware {{PLATFORM}}, DSL {{FRAMEWORK}}. Optimize operator {{OPERATOR}} and retrieve techniques and pitfalls." --brief
```

GPU Wiki query responses emit a top-level `query_id`, and every returned record emits its own
canonical `wiki_id` in `store::record` form. Copy those fields exactly; never reconstruct either
value from a response mapping key or from prose. Whenever a returned record
materially influences an experiment or is explicitly evaluated and rejected, add `wiki_usage` to
that experiment's journal append. Each row must contain the response's emitted `query_id`, an
actually returned record's emitted `wiki_id`, a disposition of `applied`, `partially_applied`,
`reference_only`, or `rejected`, plus a
short `use` and observable `evidence`. Preserve repeated use in separate experiments; do not dedupe
across the episode. The first experiment must account for the required query as either `declared`
or `no_material_use`; it cannot claim `not_queried`. Every experiment must set
`wiki_usage_status` to `declared` with non-empty usage,
`no_material_use` when Wiki was queried without attributable use, or `not_queried` when it was not
queried. For `declared` and `no_material_use`, include `wiki_query_ids` with every Wiki query considered
by the experiment; omit it for `not_queried`. In later experiments, use `not_queried` when the
experiment neither issued a new query nor reconsidered a previous response; do not carry an earlier
query id forward unless its response informed that experiment. Cite `gateway_record_ids` for measured correctness/performance and explain your decision in `analysis`.
Malformed Wiki telemetry is diagnostic only: the journal drops bad rows into `wiki_usage_errors`
without invalidating the optimization experiment or its terminal handoff.

## Engineering loop

`skills/gpu-kernel-episode-loop/SKILL.md` defines the binding evidence loop for this episode:
reconstruct the incumbent, profile and localize, research progressively, plan one coherent direction,
implement and repair, validate development correctness and performance, record every decisive
experiment, and mark the phase telemetry. **Read that file now and execute its loop**; it is a
requirement, not background reading.

Bind its placeholders to this episode:

| Skill placeholder | This episode |
| --- | --- |
| `<PROFILE_DIR>` | `profiles/episode_{{EPISODE}}` |
| `<PLAN_DRAFT>` | `plans/v{{VERSION}}_draft.md` |
| `<PLAN_FILE>` | `plans/v{{VERSION}}_plan.md` |

`<PLAN_GENERATOR>` is the backend-native plan generator for this session:

{{PLAN_GENERATOR}}

As soon as one coherent candidate passes the full development correctness check and has credible
performance evidence, publish the terminal handoff. Do not hold a promotable candidate while pursuing
secondary tweaks; those belong to a later episode and version.

## Terminal contract

Close all active Directions, then submit a request file through:

```bash
python3 tools/sandbox.py --kind episode-report --request-file scratch/episode-report.json
```

For a candidate, leave its exact measured bytes in `kernel.py`:

```json
{"status":"candidate_ready","summary":"What changed and what evidence supports it","selected_experiment_id":"experiment_<returned-id>"}
```

The selected Experiment must be from this Episode and cite a passing, standard full-workload
Evaluate of these bytes. The Supervisor creates the candidate commit, verifies it and decides
promotion. Do not send `candidate_commit`, run Git, or write a handoff/Journal projection yourself.
An accepted report is not a promotion decision.

For an exhausted direction, use `{"status":"pivot","summary":"Evidence-backed conclusion"}`.
For infrastructure/authority blockers, use
`{"status":"blocked","summary":"What prevented progress","blocker":"Concrete missing capability"}`.
Fast candidate/pivot reports require the configured trial minimums; blocked is exempt.
Full pivot/blocked may have no Experiments only when no Direction needs closing.

Use `skills/runtime-records/references/journal.md` for complete fields and repair instructions.
For PPU, optional `accepted_ppu_diagnostics` must still satisfy the mounted profiler Skill.
A rejected report leaves the Episode open: correct the fields or prerequisites identified in the
error and resubmit. An identical accepted report can be replayed to repair publication.
Stop only after successful handoff; chat text alone is not a handoff.
