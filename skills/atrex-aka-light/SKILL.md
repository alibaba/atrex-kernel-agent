---
name: atrex-aka-light
description: Use AKA Light to validate accepted operator inputs and to install, launch, resume and inspect supervised GPU-kernel optimization campaigns. Covers Native Atrex-Bench and SOL contracts, Fast/Performance/Ultra final-source policy, persistent Agent Sessions and Git, Runtime evaluation, submit-kernel review, pooled search, observability, diagnosis and evidence export. Use for Light; standalone Aone Sandbox lifecycle is covered by its dedicated skill.
---

# AKA Light

AKA Light runs persistent GPU-kernel optimization Sessions and sends GPU work to Agate. The coordinator needs matching AKA packages, an Agent CLI and Agate access; it does not need the target GPU locally. Treat the selected checkout as authoritative and run `aka optimize --repo-root REPO --help` before constructing a non-trivial launch.

## Route the request

| Goal | Action |
|---|---|
| Install/update Light | Run `scripts/install_light.sh AKA_REPO_ROOT`; read [installation](references/installation.md) |
| Validate an operator | Check [problem inputs](references/problem-inputs.md) |
| Start/resume/stop | Follow [operations](references/operations.md) |
| Run a minimal functional launch | Follow the [Native Vector Add example](examples/native-atrex-bench-vector-add/README.md) after completing the required launch intake below |
| Configure CLI or composition | Read [configuration](references/configuration.md) |
| Inspect a live campaign or export evidence | Prefer the dedicated `atrex-aka-light-observability` skill; otherwise read [observability](references/observability.md) |
| Open or capture a local Campaign Dashboard | Use `atrex-aka-light-dashboard`; this is inspection and does not require the launch intake below |

For standalone Aone Sandbox lifecycle, use its dedicated skill.

## Required launch intake

Before constructing or executing a new Light launch, explain every choice below and explicitly ask
the user for all eight inputs. Do not recommend or preselect a combination for a minimal smoke run,
and do not infer values from an older Campaign, shell history, another Sandbox, the host OS, or a
nearby checkout:

1. **Agate AK/SK availability and safe injection source.** Ask the user to make `AGATE_AK` and `AGATE_SK` available to the Supervisor process, or to identify an already configured secure credential source. Never ask the user to paste either secret into chat, and never print, copy, log, commit, or place them in argv, notes, patches, workspace files, or manifests. Verify only that both values are present.
2. **Canonical Atrex-Bench checkout root.** Ask the user either to provide an existing checkout path
   containing `scripts/run_eval.py` and `src/atrex_bench/`, or to choose a destination where
   `https://github.com/alibaba/atrex-bench` should be cloned. Do not assume
   `/path/to/atrex-bench` or any nearby checkout.
3. **Target GPU/platform token.** Require the user to select an explicit value such as `L20N` or
   `L20D`. Never infer it from the operator, an older Campaign, a currently available Sandbox or
   queue availability. If the Agate worker alias differs, ask the user to confirm that token
   separately and pass it through `--hardware`.
4. **Agent CLI.** Present all supported choices and require one: `claude` (Claude Code), `codex`
   (OpenAI Codex), `qodercli`, `pi`, or `dsh`. Check that the selected CLI is installed and authenticated;
   do not default to Claude Code or Codex.
5. **Optimization mode.** Present the following choices before asking the user to select one:
   - **Fast:** the Agent chooses any evaluator-supported DSL or mixture; explicitly declared reproducible libraries or prebuilt implementations may perform core computation. Use it for the fastest path to a working/high-performance result when external implementation reuse is acceptable.
   - **Performance:** the Agent chooses the DSL using expertise and measurements, but the final core computation must be self-authored; prebuilt or third-party core compute is not allowed. Use it to pursue strong performance without pinning a DSL.
   - **Ultra:** the final core computation must be self-authored and one user-selected DSL is mandatory and enforced, such as `Cuda`, `Triton`, or `CuteDSL`. If the user selects Ultra, also require the DSL; refuse launch when it is missing.
   Every new launch must pass the selected `--optimization-mode` explicitly. Never treat the CLI's
   `fast` fallback as user intent.
6. **Agent Sandbox choice.** Explain that this controls the local Agent process boundary, not whether Agate uses remote GPUs and not whether an Aone Sandbox is created. Require an explicit choice even though the CLI has a native-process fallback:
   - **With Sandbox (`--profile light-sandbox`):** run the Agent inside a Bubblewrap filesystem/process boundary while retaining Light capture, persistent Session, workspace and Agent Git. Supervisor-private credentials and state stay outside the Agent view. This requires Linux with Bubblewrap; on macOS use a Lima Linux VM. When the user selects this option, require one further choice:
     - **`private`:** create a private PID namespace and mount a new procfs. The Agent cannot see unrelated coordinator processes. This is available on ordinary Linux and Lima.
     - **`host`:** share the outer environment's PID namespace and bind its complete `/proc`. Filesystem allowlisting and dropped capabilities remain, but process visibility is not isolated. Use a dedicated outer environment. **Aone Sandbox can only use `host`; reject `private` there because Aone does not permit the nested procfs mount required by `private`.**
   - **Without Sandbox (`--profile light`):** run the Agent as a normal process under the coordinator's current OS user, without Bubblewrap isolation. Light capture, persistent Session, workspace and Agent Git still work. This is compatible with native macOS but gives the Agent the host access permitted to that user, so use it only when that trust boundary is acceptable.
7. **Run length.** The ordinary launch is one independent trajectory, not three-line pooled
   search. Require the user to choose a positive `--max-iters N`; `N` is the number of Episode
   audit segments, not the number of Kernel versions or Evaluates. Do not preselect `1`, `5`, `20`,
   or any other value for a smoke run. Direction and Experiment annotations remain disabled and
   are not part of ordinary launch intake. Only when the user explicitly requests advanced pooled
   search or Research annotations should you expose and collect those additional settings. An
   ordinary launch creates one Campaign repeat; ask for a repeat count only when the user requests
   repeated independent Campaigns.
8. **Optional reference knowledge.** Ask whether the user wants to expose one or more existing
   knowledge directories through `--agent-reference-paths`; `none` is a valid explicit answer. Each
   directory appears at `references/<basename>/`, and basenames must be unique. With
   `light-sandbox`, Bubblewrap mounts the directories read-only. With native `light`, Light creates
   workspace links but cannot prevent the Agent from modifying the original directories because it
   runs as the coordinator OS user. Warn native users to retain an independent copy or backup of
   every supplied knowledge directory before launch.

When any required value is missing, pause the launch and ask one concise question containing the missing fields. A suitable prompt is:

> 启动前请逐项选择：① Agate AK/SK 的安全注入方式（将 `AGATE_AK`、`AGATE_SK` 预先放入 Supervisor 环境，不要在聊天中粘贴明文）；② 已有 canonical `atrex-bench` 路径，或从 `https://github.com/alibaba/atrex-bench` clone 到哪个自选路径；③ Agate 的确切目标 GPU token，例如 `L20N` 或 `L20D`，示例不是默认值；④ Agent CLI：`claude`（Claude Code）、`codex`、`qodercli`、`pi` 或 `dsh`；⑤ 优化模式：`fast`（允许声明清楚的外部核心实现）、`performance`（核心计算自行实现，Agent 选 DSL）或 `ultra`（核心计算自行实现并固定用户指定 DSL），Ultra 还需选择 DSL；⑥ Agent 边界：`light`（原生进程）或 `light-sandbox`（Bubblewrap），后者还需选择 `private` 或 `host`，Aone Sandbox 只支持 `host`；⑦ 单线独立运行多少个 Episode audit segment，即明确给出正整数 `max-iters`；⑧ 是否通过 `--agent-reference-paths` 提供一个或多个参考知识目录，或明确选择 `none`。Bubblewrap 下为只读挂载；原生 `light` 无法阻止 Agent 修改原目录，使用前必须自行保留原始副本。Direction/Experiment 默认关闭，普通启动不询问；只有明确要求高级 Research 或多线搜索时才展开相关选项。未明确选择时不会根据最小试跑、Mac 环境或 CLI 默认值启动。

After the user answers, validate the checkout markers, supported operator input, non-empty credential presence, target platform, selected and authenticated Agent CLI, explicitly selected optimization mode, any Ultra DSL, explicitly selected Agent boundary, whether the selected host supports the chosen proc mode, optional reference directories, and the complete explicit search schedule before creating remote work or a Campaign workspace. Never start `private` proc mode inside an Aone Sandbox. In Aone, ask the user to accept `host` or choose no inner Agent Sandbox; do not silently select `host`. For native `light`, confirm that the reference-directory warning was shown; Light cannot make those originals read-only.

## Supported problem inputs

Light accepts exactly two problem families:

| Family | Required operator files | Boundary |
|---|---|---|
| Native Atrex-Bench | `reference.py`, `input.py`, non-empty `shapes.json` | Operator must be inside a canonical checkout containing `scripts/run_eval.py` and `src/atrex_bench/` |
| SOL-ExecBench | `definition.json`, `workload.jsonl`, `reference.py` | Uses the SOL definition/workload contract |

An arbitrary `reference.py`, custom evaluator script, or setup command is not a Light problem. Convert it to a supported layout. The Native V0 is derived from `reference.py`; an adjacent candidate tree is not treated as the initial source.

## Install matching packages

In addition to the two matching AKA distributions (`atrex-aka-framework` and `atrex-aka-light`), install
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

Claim Review follows the selected Agent boundary: Bubblewrap in sandbox mode, or a native
process in its independent review workspace without sandbox mode.

The checkout later supplied through `--repo-root` must match the installed wheels. Package version strings alone are insufficient; rebuild after source changes.

## Understand the current workflow

1. Light prepares initial source and a common Agent Git root; initial-source evaluation is optional and Agent-requested.
2. Each trajectory receives one persistent workspace, Agent conversation and Agent-owned Git repository.
3. An Episode is only an audit segment created when the Agent stops; the next Episode inherits the same conversation, workspace and Git history.
4. The Agent may probe, edit, commit, Evaluate, Profile, Check, Disassemble, use Dev/Command, and manage optional Direction/Experiment records throughout the Session.
5. The Agent decides when an exact commit is ready to serve as a final implementation for the current mode. It may then call `submit-kernel`; this is not required after every Evaluate and requires a passing standard full Evaluate with complete per-Shape performance for the clean commit.
6. `submit-kernel` reviews the exact commit against the selected mode's final-source policy. Its first receipt returns a stable Submission ID and immutable `v1`; repaired, fully evaluated commits may revise that identity as `v2`, `v3`, and so on. It does not end the Session, run automatic terminal ABBA, compare against a promotion threshold, roll back source, or modify Agent Git.
7. Accepted submissions with eligible full-Evaluate evidence determine lineage winners, optional global winners and the accepted-submission Pareto frontier.

There is no Framework Baseline stage, `episode-report`, candidate sealing, terminal handoff, automatic terminal ABBA, or Supervisor promotion threshold in the current Light workflow. Explicit ABBA remains available as an Agent-requested measurement operation.

Public Kernel identities are full Agent Git commit IDs. `--workspace-base` selects the parent of named non-Git Campaign containers; only `lineages/tNNNN/agent-workspace` repositories own Kernel history. Campaign-level V0 authority and audit storage live directly below that root. Peer Git is imported through read-only refs and never overwrites local HEAD.

## Choose mode and profile

`--profile` selects the workflow/Agent boundary; `--optimization-mode` selects final-source policy.
For a new Campaign, both are user decisions and must be passed explicitly. Parser or composition
defaults document implementation behavior only; they are not permission for the Skill to choose.

| Profile | Meaning |
|---|---|
| `light` | Current Light workflow with native Agent execution and capture |
| `light-sandbox` | Current Light workflow with Bubblewrap Agent isolation and capture |

Bubblewrap profiles require Linux. On macOS the user must choose native `light` or run Linux through
Lima and choose `light-sandbox`. When `light-sandbox` is selected, require an explicit proc-mode
choice. Aone Sandbox supports only `host`, but the user must still accept `host` rather than having
the Skill select it automatically.

| Mode | Final submitted source |
|---|---|
| `fast` | Agent-selected DSL; explicitly declared reproducible libraries may perform core computation |
| `performance` | Agent selects a DSL using expert guidance; final core computation must be self-authored |
| `ultra` | One DSL must be explicitly configured and is enforced; final core computation must be self-authored |

All modes require a passing standard full Evaluate with complete per-Shape performance before `submit-kernel` can accept a commit. No mode requires automatic terminal ABBA or a minimum speedup. Only `fast`, `performance` and `ultra` are supported; old mode names are not aliases.

Evaluator Shapes are exposed by default. A hidden request is valid only for the supported Performance/Ultra generalized flow and only when Light actually creates/loads the hidden `agent_problem.json`; otherwise launch fails instead of silently claiming Shapes are hidden.

## Candidate layout

- Native Atrex-Bench defaults to `source_tree`: editable `kernel.py` plus optional editable `impl/`.
- SOL uses the configured single-file contract.
- Native CUDA without a Python loader may use the configured `cuda_manifest` layout.
- Layout is fixed for a campaign and must match on resume.

`impl/` is optional. Do not create or require `impl/__init__.py` unless the implementation actually needs it.

## Search and budgets

Current Light defaults to one independent serial trajectory. Pooled search is an explicit advanced
configuration with these available fields; the values below are a three-trajectory example:

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

The default empty `workflow.config.search` selects serial mode. In serial mode `--max-iters N` is
the user-selected number of Episode audit segments. Pooled mode uses
`epochs × trajectories × episodes_per_trajectory` and does not use `--max-iters` as its segment
count. Direction and Experiment annotations default to disabled.

Episode active time is intentionally unbounded. There is no `workflow.config.episode_timeout_seconds` option. The Agent stops naturally or through operator cancellation/session revocation. A positive Campaign token budget is checked at invocation boundaries; it is not a wall-clock watchdog. `--token-budget 0` disables that cap. An optional `agent.config.timeout_seconds` bounds individual provider invocations; it is unset by default. Agate request, queue and per-Shape execution budgets are separate.
Runtime requests leave the Agent runnable by default (`agent.config.pause_agent_during_requests=false`). Background calls can overlap independent work; each uses frozen source/input bytes. Waiting counts toward an explicitly configured provider invocation limit. Set the option to `true` only when process suspension is desired.


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

The Campaign root has no `.git`, `.atrex-supervisor-runtime`, path-hash wrapper, `search/`, `peer-git/` or `git-sharing/`. Campaign scheduling and all synchronization artifacts are represented by `coordinator.json`, `coordinator.lock` and self-contained `publications/pNNNNNNNN/` directories; common V0 Git bootstrap state lives in `bootstrap/`; lineage-owned mutable state and Agent workspaces live in `lineages/`. The initial source is archived in `measurements/kernels/` without an automatic V0 Evaluate. The Agent decides which measurements to request. Unsubmitted measurements do not enter the accepted-submission winner or Pareto set.

## Launch

Launch options precede `--`; campaign options follow it. The uppercase values below are required
user selections, not defaults. Include `--framework USER_SELECTED_ULTRA_DSL` only for Ultra:

```bash
aka optimize --repo-root /path/to/atrex-kernel-agent \
  --profile USER_SELECTED_PROFILE \
  --patch /path/to/user-selected-search-schedule.json \
  --problem /path/to/atrex-bench/data/operator \
  --platform USER_SELECTED_GPU \
  --agent-cli USER_SELECTED_AGENT_CLI \
  --optimization-mode USER_SELECTED_MODE \
  USER_SELECTED_AGENT_REFERENCE_DIRS_IF_ANY \
  USER_SELECTED_SERIAL_MAX_ITERS_IF_APPLICABLE \
  --workspace-base /path/to/campaign-root
```

The search patch is mandatory for an ordinary Skill-constructed new launch and contains explicit
serial `search: {}`. Replace the serial placeholder with the user's `--max-iters N`. This creates
one independent trajectory. Do not expose Direction/Experiment or pooled-search settings unless
the user explicitly asks for those advanced features.

Run a new Campaign as a durable background task by default unless the user explicitly asks for an
interactive foreground run. Before detaching, validate the complete command. Persist and report:

- the Campaign root;
- the background process or Sandbox command ID;
- the bounded launcher-log path;
- the exact checkout commit and selected launch configuration.

On a normal host, use an environment-appropriate durable process supervisor or a background launch
script with redirected logs; do not leave the Campaign coupled to an interactive terminal. In an
Aone Sandbox, use the Sandbox's durable `command submit` API rather than shell `&`, and retain the
returned command ID. A local wait timeout or client disconnect does not mean the Campaign stopped.

After launch, inspect progress with the dedicated `atrex-aka-light-observability` skill. Start with
`light_lineage_status.py` for lineage/session/Research/token state, then use Kernel, Runtime, Agate,
or evidence-export views only when the question requires them. Do not repeatedly scrape raw logs,
walk the whole Campaign tree, or infer health from the launcher process alone when the supported
observability views are available.

When the user asks what the Agent is optimizing, use lineage status to select the active lineage
and obtain its exact `conversation_path`. Read only the last 100 complete JSONL events by default,
then summarize the current hypothesis, latest source change and measurement, present wait/operation,
next step, and any repeated friction. Current Episode captures normally live under
`lineages/tNNNN/segments/episodes/eNNNNNNNN/sessions/run-NNNNNNNN/conversation.jsonl`; always prefer
the path returned by observability because startup captures use a different private location.

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
- explicit `none` or validated optional reference directories, with the native-write warning;
- mode, Shape visibility and candidate layout;
- durable workspace, explicit search topology, explicit run length and budgets.

Do not print credentials. Preserve uncommitted user changes before syncing or installing code.

## Runtime evaluation defaults

For standard Evaluate and explicit ABBA, current Light batches at most 10 Shapes per outer Ray Data request and sets `shape_shards=1`. Standard Evaluate and ABBA retain the complete operator bundle and select each batch with native `options.shape_ids`; neither rewrites the reference into a batch-local problem. Light uses 10 warmups, 20 timing iterations, deterministic correctness cases, and best-effort `lock_clocks=true` without a fixed MHz. Runtime reports measured per-Shape latency, geometric mean latency and arithmetic mean latency after all batches are merged. Accepted-submission winners minimize geometric mean latency; Runtime does not derive reference-relative speedups or normalization scores.

Use Runtime tool help for exact operation-specific Agate fields. Endpoint, credentials, source selection and other authority-bearing options remain Supervisor-owned.

## Resume and inspect

Resume with the same checkout behavior, workspace, profile composition, problem identity, mode, Shape visibility, framework and candidate layout. Do not clear a workspace merely because the process exited; inspect Session, Runtime and Agate state first.

For “状态如何” or “跑得怎么样” requests, use the `atrex-aka-light-observability` skill rather
than restarting the launch or reading every file. Distinguish the background launcher/command state
from Campaign lineage state and from Agate Job state.

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
| CLI flags, defaults and patches | [Configuration](references/configuration.md) |
| Launch, modes, V0, persistent Episodes, resume and cleanup | [Operations](references/operations.md) |
| Live state, Agate, tokens, winners and evidence export | [Observability](references/observability.md) |
