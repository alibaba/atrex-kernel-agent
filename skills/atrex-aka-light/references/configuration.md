# Light configuration reference

Run the matching checkout's `aka optimize --repo-root /path/to/aka --help` first.
The CLI creates a typed `WorkflowLaunch`. All flags use one parser; no `--` separator
is required. Defaults below are implementation fallbacks, not user consent: a new
Skill-driven launch still requires explicit GPU, mode, Agent and isolation choices.

## Launch and identity

| Option | Default | Meaning |
|---|---|---|
| `--repo-root PATH` | required | Matching installed resources and plugins |
| `--profile NAME` | `light` | `light`, `light-sandbox`, or an explicit composition file |
| `--patch FILE` | none | Repeatable composition patch |
| `--problem PATH` | required | Native Atrex-Bench or SOL problem |
| `--workspace-base PATH` | cwd | Parent of named Campaign containers |
| `--workspace-suffix NAME` | mode/DSL/GPU | Distinguish repeated Campaigns; lowercase letters, digits and underscores |
| `--platform TOKEN` | required | User-selected GPU product |
| `--hardware TOKEN` | platform | Agate hardware route, if different |
| `--optimization-mode MODE` | `fast` | `fast`, `performance`, `ultra` |
| `--framework DSL` | empty | Required for Ultra; Fast/Performance are open DSL |
| `--agent-cli NAME` | `claude` | `claude`, `codex`, `qodercli`, `pi`, `dsh` |
| `--max-iters N` | 20 | Maximum persistent Session audit segments |
| `--token-budget N` | 0 | Main Agent token budget; zero means no token stop |
| `--agent-reference-paths [NAME=]DIR` | none | Repeat per directory; projected as `references/NAME`, default basename |
| `--[no-]expose-evaluator-shapes` | exposed | Hidden requests fail if a supported hidden projection cannot be created |
| `--arch NAME` | queried | Explicit architecture override |
| `--notes TEXT` | `none` | Task constraints; never put credentials here |

The Campaign root is not Git. Only `lineages/tNNNN/agent-workspace` owns Agent Git.
A new Campaign creates initial source without mandatory V0 Evaluate. Agent Sessions
and Reviews have no active-time limit. Continuation restores the recorded launch
selection; do not replace its profile or patch while resuming.

## Agent-visible Agate tools

`--agate-tools env evaluate command` is the default. Supply a replacement list, for example
`--agate-tools env evaluate command profile`, or `--agate-tools all` for all seven kinds:
`env`, `evaluate`, `command`, `profile`, `dev`, `check`, `disassemble`.
The typed JSON equivalent is `--runtime '{"agate_tools":["env","evaluate","command"]}'`;
an empty array exposes no Agate tools. `all` must appear alone. Unknown and duplicate names
are rejected at launch. The selection persists in WorkflowLaunch and applies to every lineage.

Agent help and Prompt identify the enabled kinds. Runtime rejects disabled kinds before staging
or submitting a remote job. ABBA remains part of `evaluate`; internal evaluation implementation
and Supervisor-owned checks are unaffected. Wiki, Direction/Experiment, and `submit-kernel`
retain their independent enablement. Agent-side inspection uses `python3 tools/sandbox.py --help`.

## Measurement defaults

| Option | Default | Meaning |
|---|---|---|
| `--warmup-iters N` | 10 | Warmup iterations; zero allowed |
| `--bench-iters N` | 20 | Positive benchmark iterations |
| `--evaluation-step-timeout SECONDS` | 60 | Candidate and performance step budget |
| `--shape-batch-size N` | 10 | Shapes per outer request |
| `--verify-repeats N` | 2 | Explicit ABBA measurement repeats per side |
| `--timeout SECONDS` | derived | Outer execution budget; separate from Session/Review lifetime |

Explicit Agent request controls are preserved. Invalid budgets return their concrete
reason rather than being silently clipped. All Agate calls use the SDK. Runtime
polls a known Job with short reads within the budget; it does not blindly resubmit
an ambiguous request without a Job ID or idempotency guarantee.

Evaluate and ABBA use Ray Data with `shape_shards=1`. Both retain the
full reference and pass selected `options.shape_ids`, without rebuilding batch-local references. Requests set `lock_clocks=true` without specific MHz.
Aggregation follows `metadata.performance_objective` after all batches complete.

## Composition patches

Each patch replaces that module's configuration, then provider defaults apply.
Research and pooled search belong to `workflow`:

```json
{
  "api_version": 1,
  "patches": [{"id": "workflow", "config": {
    "tool_modules": ["directions", "experiments"],
    "search": {
      "epochs": 5,
      "trajectories": 3,
      "episodes_per_trajectory": 1,
      "max_parallel_episodes": 3,
      "synchronization": "asynchronous",
      "role_specialization": false
    }
  }}]
}
```

`search: {}` is serial. Research defaults to `tool_modules: []`. For pooled runs,
choose synchronous Epoch barriers or asynchronous publication. Role specialization
is off by default. The old terminal numerical-planner option is not supported.
`request_timeout` and `queue_timeout` control Runtime requests, not Reviewer active time.

Agent isolation is independent:

```json
{
  "api_version": 1,
  "patches": [{"id": "agent", "config": {
    "sandbox": "bwrap", "proc_mode": "host", "bwrap_executable": "bwrap"
  }}]
}
```

Aone Sandbox requires `host`; ordinary Linux/Lima can use `private`. Capture is built
into the shared Agent execution path and is not a `capture` patch field. Claim
Review follows the selected Agent sandbox mode. Bubblewrap makes supplied evidence read-only; native Review has a separate workspace and Home but no OS isolation boundary.

## Environment and Runtime options

Inject `AGATE_AK` / `AGATE_SK` securely into Supervisor environment. Keep credentials
out of JSON patches, notes and argv. `AGATE_URL` or the SDK profile selects routing.
Light leaves the Atrex-Bench worker version to Agate; it does not send a version pin.
The resolved version remains in the archived Agate response.
Explicit SSH is configured by the public Execution selection and its documented
`ATREX_SANDBOX_SSH*` settings; see [SSH backend](../../../docs/ssh-backend.md).

Runtime discovers supported safe options from the installed Agate SDK. Consult
`tools/sandbox.py --help` in the live workspace for Profile, Check, Disassemble,
Env, Dev and Evaluate options. Credentials, endpoint authority and source selection
remain Supervisor-owned; an option supported by one kind is not automatically
valid for another.

### Explicitly permitted supplied Bootstrap

For a new Campaign, `workflow.config.source_tree` can select a pinned initial
source using `{ "manifest": "/path/manifest.json", "directory": "/path/source" }`.
This changes the initial candidate, not the evaluator Reference or Shapes.
Imported helper files are placed under `impl/`; the adapter must import from that
location.

Ultra/Performance normally require self-authored compute. A launch owner may
explicitly grant `workflow.config.provided_bootstrap` with `name`, `provenance`
(repository and exact upstream revision), and `files` (the complete normalized
`kernel.py` / `impl/**` path-to-SHA256 map). Startup rejects a hash or file-set
mismatch. This permission is frozen in the Campaign launch composition and is
shown to both Agent and final-source Reviewer. The Reviewer also receives the
original source snapshot for comparison and reports `provided_source_compute`
when applicable. Without that launch-owner permission the category is invalid.

The exception covers only this supplied source and in-candidate modifications.
It does not permit installed prebuilt operator libraries, other downloaded
compute implementations, PyTorch/ATen compute fallback, or a different Ultra DSL.
Full Evaluate and correctness requirements are unchanged. Do not enable this
exception without explicit user authorization; do not edit an existing Campaign's
source contract or permission to retrofit it.
