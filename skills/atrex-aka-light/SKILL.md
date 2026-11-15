---
name: atrex-aka-light
description: Use AKA Light to validate accepted operator inputs and to install, launch, resume and inspect supervised GPU-kernel optimization campaigns. Covers Native Atrex-Bench and SOL contracts, Fast/Performance/Ultra final-source policy, persistent Agent Sessions and Git, Runtime evaluation, submit-kernel review, pooled search, observability, diagnosis and evidence export. Use for Light, not the Legacy workflow or standalone Aone Sandbox lifecycle.
---

# AKA Light

AKA Light runs persistent GPU-kernel optimization Sessions and sends GPU work to Agate. The coordinator needs matching AKA packages, an Agent CLI and Agate access; it does not need the target GPU locally. Treat the selected checkout as authoritative and run `aka optimize --repo-root REPO -- --help` before constructing a non-trivial launch.

## Route the request

| Goal | Action |
|---|---|
| Install/update Light | Run `scripts/install_light.sh AKA_REPO_ROOT`; read [installation](references/installation.md) |
| Validate an operator | Check [problem inputs](references/problem-inputs.md) |
| Start/resume/stop | Follow [operations](references/operations.md) |
| Run a minimal functional launch | Follow the [Native Vector Add example](examples/native-atrex-bench-vector-add/README.md) after completing the required launch intake below |
| Configure CLI or composition | Read [configuration](references/configuration.md) |
| Inspect a live campaign or export evidence | Prefer the dedicated `atrex-aka-light-observability` skill; otherwise read [observability](references/observability.md) |

Do not use this skill for the Legacy `workflow` Profile or standalone Aone Sandbox lifecycle.

## Required launch intake

Before constructing or executing a new Light launch, explain the optimization modes and Agent execution boundaries, then explicitly ask the user for all five inputs below. Do not infer them from an older Campaign, shell history, another Sandbox, or a nearby checkout:

1. **Agate AK/SK availability and safe injection source.** Ask the user to make `AGATE_AK` and `AGATE_SK` available to the Supervisor process, or to identify an already configured secure credential source. Never ask the user to paste either secret into chat, and never print, copy, log, commit, or place them in argv, notes, patches, workspace files, or manifests. Verify only that both values are present.
2. **Canonical Atrex-Bench checkout root.** This is the path containing `scripts/run_eval.py` and `src/atrex_bench/`; it is not merely the operator directory.
3. **Target GPU/platform token.** Require the user to select an explicit value such as `L20N` or
   `L20D`. Never infer it from the operator, an older Campaign, a currently available Sandbox or
   queue availability. If the Agate worker alias differs, ask the user to confirm that token
   separately and pass it through `--sandbox-hardware`.
4. **Optimization mode.** Present the following choices before asking the user to select one:
   - **Fast:** the Agent chooses any evaluator-supported DSL or mixture; explicitly declared reproducible libraries or prebuilt implementations may perform core computation. Use it for the fastest path to a working/high-performance result when external implementation reuse is acceptable.
   - **Performance:** the Agent chooses the DSL using expertise and measurements, but the final core computation must be self-authored; prebuilt or third-party core compute is not allowed. Use it to pursue strong performance without pinning a DSL.
   - **Ultra:** the final core computation must be self-authored and one user-selected DSL is mandatory and enforced, such as `Cuda`, `Triton`, or `CuteDSL`. If the user selects Ultra, also require the DSL; refuse launch when it is missing.
   Every new launch must pass the selected `--optimization-mode` explicitly. Never treat the CLI's
   `fast` fallback as user intent.
5. **Agent Sandbox choice.** Explain that this controls the local Agent process boundary, not whether Agate uses remote GPUs and not whether an Aone Sandbox is created. Require an explicit choice even though the CLI has a native-process fallback:
   - **With Sandbox (`--profile light-sandbox`):** run the Agent inside a Bubblewrap filesystem/process boundary while retaining Light capture, persistent Session, workspace and Agent Git. Supervisor-private credentials and state stay outside the Agent view. This requires Linux with Bubblewrap; on macOS use a Lima Linux VM. When the user selects this option, require one further choice:
     - **`private`:** create a private PID namespace and mount a new procfs. The Agent cannot see unrelated coordinator processes. This is available on ordinary Linux and Lima.
     - **`host`:** share the outer environment's PID namespace and bind its complete `/proc`. Filesystem allowlisting and dropped capabilities remain, but process visibility is not isolated. Use a dedicated outer environment. **Aone Sandbox can only use `host`; reject `private` there because Aone does not permit the nested procfs mount required by `private`.**
   - **Without Sandbox (`--profile light`):** run the Agent as a normal process under the coordinator's current OS user, without Bubblewrap isolation. Light capture, persistent Session, workspace and Agent Git still work. This is compatible with native macOS but gives the Agent the host access permitted to that user, so use it only when that trust boundary is acceptable.

When any required value is missing, pause the launch and ask one concise question containing the missing fields. A suitable prompt is:

> AKA Light 提供三种优化模式：Fast（Agent 自选 DSL，允许声明清楚的库或预制实现承担核心计算）、Performance（Agent 自选 DSL，但核心计算必须自行实现）、Ultra（核心计算自行实现，并强制使用用户指定 DSL）。Agent 也有两种运行边界：使用 Sandbox（`light-sandbox`，Linux Bubblewrap 隔离，不等于创建 Aone Sandbox）和不使用 Sandbox（`light`，Agent 以当前系统用户直接运行，兼容原生 macOS但隔离较弱）。Sandbox 又分为 `private`（私有 PID namespace 和 procfs）与 `host`（共享外层 PID namespace 和完整 `/proc`，不提供进程可见性隔离）；如果运行在 Aone Sandbox 中，只能选择 `host`。启动前请主动选择并提供：① Agate AK/SK 的安全注入方式（请将 `AGATE_AK`、`AGATE_SK` 预先放入 Supervisor 环境，不要在聊天中粘贴明文）；② canonical `atrex-bench` 根目录；③ 目标 GPU，例如 `L20N` 或 `L20D`；④ `fast`、`performance` 或 `ultra`，如选 `ultra` 请同时指定 DSL；⑤ 使用 Sandbox 还是不使用 Sandbox，如使用请再选择 `private` 或 `host`。未明确选择时不会使用默认值启动。

After the user answers, validate the checkout markers, supported operator input, non-empty credential presence, target platform, explicitly selected optimization mode, any Ultra DSL, explicitly selected Agent boundary, and whether the selected host supports the chosen proc mode before creating remote work or a Campaign workspace. Never start `private` proc mode inside an Aone Sandbox. In Aone, ask the user to accept `host` or choose no inner Agent Sandbox; do not silently select `host`.

## Supported problem inputs

Light accepts exactly two problem families:

| Family | Required operator files | Boundary |
|---|---|---|
| Native Atrex-Bench | `reference.py`, `input.py`, non-empty `shapes.json` | Operator must be inside a canonical checkout containing `scripts/run_eval.py` and `src/atrex_bench/` |
| SOL-ExecBench | `definition.json`, `workload.jsonl`, `reference.py` | Uses the SOL definition/workload contract |

An arbitrary `reference.py`, custom evaluator script, or setup command is not a Light problem. Convert it to a supported layout or intentionally use Legacy. The Native V0 is derived from `reference.py`; an adjacent candidate tree is not treated as the initial source.

## Install matching packages

In addition to the three matching AKA distributions, install
`atrex-gateway-client==0.14.31` from the internal ArtLab release index in the Supervisor
environment. Obtain the canonical Atrex-Bench source checkout from
`https://github.com/alibaba/atrex-bench`; do not substitute an Atrex-Bench wheel for that checkout.
Read [installation](references/installation.md) for the exact commands.

```bash
python3 -m venv /path/to/aka-venv
source /path/to/aka-venv/bin/activate
bash ~/.codex/skills/atrex-aka-light/scripts/install_light.sh \
  /path/to/atrex-kernel-agent
```

The checkout later supplied through `--repo-root` must match the installed wheels. Package version strings alone are insufficient; rebuild after source changes.

## Understand the current workflow

1. Light prepares and full-workload measures deterministic V0.
2. Each trajectory receives one persistent workspace, Agent conversation and Agent-owned Git repository.
3. An Episode is only an audit segment created when the Agent stops; the next Episode inherits the same conversation, workspace and Git history.
4. The Agent may probe, edit, commit, Evaluate, Profile, Check, Disassemble, use Dev/Command, and manage optional Direction/Experiment records throughout the Session.
5. The Agent decides when an exact commit is ready to serve as a final implementation for the current mode. It may then call `submit-kernel`; this is not required after every Evaluate and requires a passing standard full Evaluate with complete per-Shape performance for the clean commit.
6. `submit-kernel` reviews the exact commit against the selected mode's final-source policy. Its first receipt returns a stable Submission ID and immutable `v1`; repaired, fully evaluated commits may revise that identity as `v2`, `v3`, and so on. It does not end the Session, run automatic terminal ABBA, compare against a promotion threshold, roll back source, or modify Agent Git.
7. Accepted submissions with eligible full-Evaluate evidence determine lineage winners, optional global winners and the accepted-submission Pareto frontier.

There is no Framework Baseline stage, `episode-report`, candidate sealing, terminal handoff, automatic terminal ABBA, or Supervisor promotion threshold in the current Light workflow. Explicit ABBA remains available as an Agent-requested measurement operation.

Public Kernel identities are full Agent Git commit IDs. `--workspace` names a non-Git Campaign container; only `lineages/tNNNN/agent-workspace` repositories own Kernel history. Campaign-level V0 authority and audit storage live directly below that root. Peer Git is imported through read-only refs and never overwrites local HEAD.

## Choose mode and profile

`--profile` selects the workflow/Agent boundary; `--optimization-mode` selects final-source policy.
For a new Campaign, both are user decisions and must be passed explicitly. Parser or composition
defaults document implementation behavior only; they are not permission for the Skill to choose.

| Profile | Meaning |
|---|---|
| `light` | Current Light workflow with native Agent execution and capture |
| `light-sandbox` | Current Light workflow with Bubblewrap Agent isolation and capture |
| `workflow` | Legacy workflow, not Light |
| `observed` / `sandbox` | Legacy workflow with an alternate Agent boundary, not Light |

Bubblewrap profiles require Linux. On macOS the user must choose native `light` or run Linux through
Lima and choose `light-sandbox`. When `light-sandbox` is selected, require an explicit proc-mode
choice. Aone Sandbox supports only `host`, but the user must still accept `host` rather than having
the Skill select it automatically.

| Mode | Final submitted source |
|---|---|
| `fast` | Agent-selected DSL; explicitly declared reproducible libraries may perform core computation |
| `performance` | Agent selects a DSL using expert guidance; final core computation must be self-authored |
| `ultra` | One DSL must be explicitly configured and is enforced; final core computation must be self-authored |

All modes require a passing standard full Evaluate with complete per-Shape performance before `submit-kernel` can accept a commit. No mode requires automatic terminal ABBA or a minimum speedup. `leaderboard` and `production` remain compatibility aliases for existing workspaces.

Evaluator Shapes are exposed by default. A hidden request is valid only for the supported Performance/Ultra generalized flow and only when Light actually creates/loads the hidden `agent_problem.json`; otherwise launch fails instead of silently claiming Shapes are hidden.

## Candidate layout

- Native Atrex-Bench defaults to `source_tree`: editable `kernel.py` plus optional editable `impl/`.
- SOL uses the configured single-file contract.
- Native CUDA without a Python loader may use the configured `cuda_manifest` layout.
- Layout is fixed for a campaign and must match on resume.

`impl/` is optional. Do not create or require `impl/__init__.py` unless the implementation actually needs it.

## Search and budgets

Current Light defaults to pooled search:

```json
{
  "epochs": 10,
  "trajectories": 3,
  "episodes_per_trajectory": 1,
  "max_parallel_episodes": 3,
  "synchronization": "asynchronous",
  "role_specialization": false
}
```

Set `workflow.config.search` to an explicit empty mapping to select serial mode. In serial mode `--max-iters N` is the number of Episode audit segments. Pooled mode uses `epochs × trajectories × episodes_per_trajectory` and does not use `--max-iters` as its segment count.

Episode active time has no limit by default; set `workflow.config.episode_timeout_seconds` only when a limit is wanted. `--token-budget 0` means no token cap. Agate request, queue and per-Shape execution budgets are separate.

Pooled search keeps Campaign-wide scheduling facts in root `coordinator.json`, serialized by
`coordinator.lock`. Each immutable synchronization event is self-contained under
`publications/pNNNNNNNN/`: `publication.json` stores global metadata, `sources/tNNNN/` stores
producer Git history and `imports/tNNNN.json` stores consumer receipts. Each lineage keeps its
HEAD, Session, completed Episode, lineage winner, persistent Agent Git repository, Research queue,
Session capture and Runtime audit under `lineages/tNNNN/`; there are no Episode Journals. The
common Agent V0 Git root lives once under `bootstrap/agent-v0.{bundle,json}`; there is no
Campaign-level `peer-git/` or lineage `git-sharing/` wrapper. Immutable
Gateway Records live once in the Campaign Measurement Store,
Direction/Experiment versions live once in the Campaign Research Library, and every submit-kernel
decision lives once under Campaign `submissions/`. Each Agent workspace receives ordinary
visibility-filtered Library files. Bubblewrap overlays that subtree read-only; native-mode changes
are disposable and repaired on refresh. At synchronization boundaries Light publishes every
archived Direction/Experiment identity with its actual review quality, relevant Gateway Record
IDs, peer Agent Git refs, accepted-submission winners and the aggregated Direction queue. Review
quality controls evidentiary weight, not visibility. Light does not merge scratch files or
conversations or rewrite another trajectory's checkout.

The Campaign root has no `.git`, `.atrex-supervisor-runtime`, path-hash wrapper, `search/`, `peer-git/` or `git-sharing/`. Campaign scheduling and all synchronization artifacts are represented by `coordinator.json`, `coordinator.lock` and self-contained `publications/pNNNNNNNN/` directories; common V0 Git bootstrap state lives in `bootstrap/`; lineage-owned mutable state and Agent workspaces live in `lineages/`. V0 is a private full-Evaluate Gateway Record in `measurements/`; there is no separate baseline store. It is reference evidence, not an accepted submission or Pareto/winner candidate.

## Launch

Launch options precede `--`; campaign options follow it. The command below is only an Ultra/CUDA
syntax example and is valid only after the user explicitly selects `light-sandbox`, `L20D`,
`ultra`, and `Cuda`; substitute the user's actual choices:

```bash
aka optimize --repo-root /path/to/atrex-kernel-agent \
  --profile light-sandbox -- \
  --op-dir /path/to/atrex-bench/data/operator \
  --platform L20D \
  --framework Cuda \
  --agent-cli claude \
  --optimization-mode ultra \
  --workspace /path/to/campaign-root
```

For an end-to-end Native input and launch command based on Vector Add, read
[the Vector Add launch example](examples/native-atrex-bench-vector-add/README.md). It uses the
user-provided Atrex-Bench root and platform and assumes `AGATE_AK`/`AGATE_SK` are already present
in the Supervisor environment.

Pass `--profile light` explicitly when the user chooses native Light. Pass
`--profile light-sandbox` only when the user chooses Bubblewrap isolation; it does not provision an
Aone Sandbox.

Resolve before launch:

- matching checkout and installed wheels;
- supported problem input;
- logical platform and any distinct Agate worker alias;
- explicit framework for Ultra, or intentional Agent-selected DSL for Fast/Performance;
- authenticated Agent and user-confirmed Agate AK/SK injection;
- mode, Shape visibility and candidate layout;
- durable workspace, search topology and budgets.

Do not print credentials. Preserve uncommitted user changes before syncing or installing code.

## Runtime evaluation defaults

For standard Evaluate and explicit ABBA, current Light batches at most 10 Shapes per outer Ray Data request and sets `shape_shards=1`. Standard Evaluate retains the complete operator bundle and selects the batch with native `shape_ids`. As a temporary Agate compatibility layer, ABBA instead rebuilds each batch as a complete shard-local operator reference and sends no `shape_ids`, because current Agate ABBA requires every Shape in its submitted reference. The compatibility is isolated in `aka.light.evaluation.abba_shape_compat` so it can be removed when Agate accepts native ABBA Shape selection. Light uses 20 warmups, 100 timing iterations, deterministic correctness cases, and best-effort `lock_clocks=true` without a fixed MHz. Aggregation follows the benchmark's declared performance objective after all batches are merged.

Use Runtime tool help for exact operation-specific Agate fields. Endpoint, credentials, source selection and other authority-bearing options remain Supervisor-owned.

## Resume and inspect

Resume with the same checkout behavior, workspace, profile composition, problem identity, mode, Shape visibility, framework and candidate layout. Do not clear a workspace merely because the process exited; inspect Session, Runtime and Agate state first.

Start inspection with:

```bash
python3 "$AKA_REPO/observability/light_lineage_status.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_kernel_history.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_kernel_best.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_runtime_calls.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_agate_calls.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_gateway_jobs.py" "$CAMPAIGN" \
  --gateway-record-id gateway-0123456789abcdef0123456789abcdef --json
```

For performance authority, inspect `PROGRESS.md`, `frontier.json`, Campaign `submissions/`, MeasurementStore and immutable Gateway Records. Do not interpret old promotion-oriented fields or scripts as current winner authority. Export evidence by Gateway Record ID and source by Agent Git commit ID before destructive cleanup.

Current Runtime receipts are created before execution and finalized in place; their request IDs
are retained by Gateway Records and physical Job checkpoints. Kernel history, winner and export
views expose `evidence_health`. In Agate timing, `queue_seconds` is queue time and
`total_seconds` is Agate submitted-to-terminal time; Runtime end-to-end duration is separate.

## Safety boundaries

- A read-only status request does not authorize restart, resubmission, cleanup or deletion.
- A timeout, queue failure, missing receipt or transport error is not a correctness failure.
- Stop through the owning Supervisor/workload and verify Agent, Runtime and children exit.
- Workspace or Sandbox deletion requires explicit authorization.
- Do not edit private Runtime audit, Research Store, Submission Store, MeasurementStore, checkpoint, lock or Git state to force recovery.

## Read as needed

| Scenario | Reference |
|---|---|
| Installation, Bubblewrap, Lima or Aone host-proc mode | [Installation](references/installation.md) |
| Native/SOL contracts, Shapes, hidden flow or layouts | [Problem inputs](references/problem-inputs.md) |
| CLI flags, defaults, patches and Legacy-only parser flags | [Configuration](references/configuration.md) |
| Launch, modes, V0, persistent Episodes, resume and cleanup | [Operations](references/operations.md) |
| Live state, Agate, tokens, winners and evidence export | [Observability](references/observability.md) |
