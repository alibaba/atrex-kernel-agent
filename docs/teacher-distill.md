# Hidden-Teacher Distillation Campaigns

## Purpose

`teacher-distill` is an opt-in offline workflow for producing an auditable 1→10 optimization trajectory. It does not replace the standard optimizer and is not the recommended online path for users who already have a strong kernel.

```text
Offline distillation:
PyTorch/reference V0
→ Agent-authored naive framework V1
→ profile-driven V2…VN
→ Teacher parity
→ evidence-backed review drafts

Online frontier optimization:
validated strong seed
→ separate standard warm-start campaign
→ 10→100
```

The campaign stops after verified Teacher parity. It never automatically continues into frontier optimization and never edits canonical `gpu-wiki/`.

## Threat model

The first release is **hidden-audited**, not a security sandbox.

- Teacher source and provenance remain in supervisor-private state outside the Candidate workspace.
- Optimization sessions see an opaque Teacher ID and measured target latencies only.
- `gpu-wiki/` is replaced by a physical, content-addressed sanitized view.
- Current-operator implementations, journeys, pitfalls, Teacher source projects, `reference-projects/`, KernelWiki, public web, downloads, and remote Git operations are unavailable.
- Common forbidden path/network actions are terminated and recorded in a private JSONL audit.
- An audited violation changes the terminal state to `TEACHER_LEAKAGE_VIOLATION`; no distillation drafts are generated.

Because coding Agents run as the local user, this does not defend against a deliberately adversarial process using an unobserved side channel. A future security-grade design requires a remote Teacher registry and OS-level filesystem/network isolation.

## Teacher solution contract

Pass a self-contained directory:

```text
teacher_solution/
├── kernel.py
├── solution.json
├── provenance.json
└── helpers/                 # optional; every source must be declared
```

No symlinks, undeclared files, absolute paths, parent traversal, dynamic external source, or missing source files are accepted. The full tree is content-hashed. Any mutation invalidates resume.

`solution.json` must use the same framework as the campaign and one evaluator-compatible entry point:

- SOL-ExecBench: `kernel.py::run`
- Native Atrex-Bench: `kernel.py::Model`

`provenance.json` requires:

```json
{
  "schema_version": 1,
  "operator": {
    "canonical_id": "gdn_decode",
    "aliases": ["gdn", "gated_delta_rule"]
  },
  "source": {
    "project": "upstream-project",
    "revision": "upstream-commit",
    "license": "Apache-2.0"
  },
  "target": {
    "framework": "CuteDSL",
    "architecture": "sm90"
  },
  "knowledge_deny": {
    "sources": ["upstream-project"],
    "paths": [],
    "tags": ["gdn", "gdn_decode"]
  }
}
```

See [`../teacher_distill/examples/`](../teacher_distill/examples/) for a non-proprietary structural example.

## Command

```bash
python3 orchestrator/optimize.py \
  --campaign-mode teacher-distill \
  --op-dir /path/to/operator \
  --teacher-solution /path/to/teacher_solution \
  --platform H20 \
  --arch sm_90 \
  --framework CuteDSL \
  --no-workload-bucketing \
  --sandbox-hardware REMOTE_GPU \
  --agent-cli pi \
  --max-iters 30 \
  --max-stall 5
```

Teacher mode requires:

- explicit single framework;
- production policy;
- single operator and single workspace;
- `--no-workload-bucketing`;
- no `--layer`;
- same-framework Teacher and Candidate.

Omitting `--optimization-mode` implies production. Explicit `--optimization-mode leaderboard` is rejected.

Default target and escalation controls:

```text
--teacher-geomean-ratio 1.05
--teacher-shape-ratio 1.10
--teacher-stall-before-episode 3
--teacher-partial-restarts 1
```

## Lifecycle

### 1. Teacher validation and measurement

Before an optimization Agent runs, the supervisor:

1. validates bundle/provenance/framework/architecture;
2. applies the production dependency/framework gate;
3. materializes a private evaluator-faithful workspace;
4. runs full single-seed correctness;
5. runs five additional correctness seeds;
6. records a full-workload benchmark;
7. locks workload, evaluator, measurement, bundle, and knowledge-view hashes.

### 2. Candidate baselines

- V0 is the immutable PyTorch/reference implementation.
- V1 is authored by the Agent in the selected framework and has no performance gate.
- V2+ uses the ordinary profile → research → one lever → correctness → benchmark loop.

For Native Atrex-Bench, V0 is materialized mechanically so the setup Agent cannot rewrite it into a framework kernel. For SOL, the existing evaluator-faithful PyTorch wrapper remains V0.

### 3. Target and ABBA

A Candidate reaches the provisional gate only when:

```text
Candidate geomean / Teacher geomean <= 1.05
and
max per-shape Candidate / Teacher <= 1.10
```

The supervisor then runs Teacher A → Candidate B → Candidate B → Teacher A in one GPU allocation. Only full correctness and both ratio gates produce `SUCCESS`. Performance failure continues optimization; verifier infrastructure failure is retryable and does not count as an optimization stall.

### 4. Stall escalation

After three ordinary no-promotion rounds:

1. one isolated long-horizon episode may use private checkpoint regressions;
2. the incumbent changes only after its normal ABBA improvement gate;
3. if no promotion occurs, one deterministic partial-memory restart masks half of V2…V(N−1), preserving V0, V1, and latest;
4. escalation and restart counters persist across resume and cannot repeat beyond configured limits.

### 5. Distillation

Completion runs:

```text
deterministic Evidence Builder
→ post-run Teacher gap analysis (hypothesis only)
→ clean Distillation Agent
→ deterministic draft validator
```

Outputs live below supervisor-private state:

```text
<private-root>/<campaign-id>/distillation/drafts/
├── evidence/
├── teacher_gap_analysis.md
├── teacher_gap_analysis.json
├── journey.md
├── pitfalls.md
├── optimization_cards/
├── promotion_checklist.md
├── draft_manifest.json
├── validation_report.json
└── reference_kernel/         # SUCCESS only; final Candidate, never Teacher
```

Every performance number and verified causal claim must cite an evidence ID. Masked, reverted, exploratory, Teacher-gap, and policy-violation evidence cannot prove a verified optimization. Canonical wiki promotion always requires human review and gpu-wiki CI.

## Resume and fork

`campaign_lock.json` and private state lock:

- Teacher bundle hash;
- operator/workload hash;
- evaluator hash;
- measurement configuration hash;
- platform, architecture, and framework;
- target ratios;
- sanitized knowledge-view hash.

Any mismatch fails with `RESUME_CONFIG_MISMATCH`. Changing Teacher, workloads, thresholds, framework, evaluator, or knowledge must start a new campaign/fork; targets never drift in place.

## Terminal states

| State | Meaning |
|---|---|
| `SUCCESS` | Provisional ratios and authoritative ABBA passed |
| `PLATEAU` | Configured post-escalation stall limit reached |
| `BUDGET_EXHAUSTED` | Iteration/token budget ended before parity |
| `INFRA_ERROR` | Setup, gateway, verifier, or distillation infrastructure failed |
| `TEACHER_LEAKAGE_VIOLATION` | Forbidden source/path/network access was audited |

`SUCCESS`, `PLATEAU`, and `BUDGET_EXHAUSTED` return a normal CLI exit after writing state. Infrastructure and leakage failures return nonzero.
