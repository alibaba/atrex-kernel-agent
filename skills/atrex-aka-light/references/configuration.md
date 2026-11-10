# AKA Light configuration

Read this file when constructing or reviewing an AKA Light launch. It describes the public controls on the current upstream `main`; always confirm the selected checkout with `aka optimize --repo-root REPO -- --help` before using a newer revision.

## Two argument layers

`aka optimize` has an explicit `--` boundary:

```bash
aka optimize \
  --repo-root /path/to/atrex-kernel-agent \
  --profile light-sandbox \
  --patch /path/to/light-options.json \
  -- \
  --op-dir /path/to/operator \
  --platform L20D \
  --framework CuteDSL \
  --max-iters 20
```

Options before `--` select installed components. Options after `--` configure the optimization Campaign.

### Launch options before `--`

| Option | Default | Purpose |
|---|---:|---|
| `--repo-root PATH` | required | Checkout whose resources, profiles and packaged skills match the installed AKA wheels. It is not the Campaign workspace. |
| `--profile PROFILE` | `light` | Select `light`, `light-sandbox`, a Legacy-compatible profile, or an explicit Profile JSON. This chooses the Workflow and Agent boundary, not Leaderboard versus Production policy. |
| `--patch FILE` | none | Apply a composition patch. Repeat to apply multiple patches in order. Use this for module options such as Episode time, Wiki, generated memory or pooled search. |
| `--execution-profile PROFILE` | selected default | Advanced: select the separately composed Execution provider for prepared controller calls. This does not select the Light Workflow or change acceptance policy. |
| `--execution-patch FILE` | none | Advanced: patch that independent Execution selection. Repeatable. |

Do not add `--profile` or `--patch` when resuming through recorded continuation metadata; the frozen launch selection is restored and overrides are rejected.

## Campaign inputs, identity and policy

These options appear after `--`.

| Option | Default | Purpose and effect |
|---|---:|---|
| `--op-dir PATH` | required | A [supported Native Atrex-Bench or SOL operator directory](problem-inputs.md). AKA snapshots the problem contract and derives the initial candidate from it; candidate layouts are not alternate problem formats. |
| `--platform TOKEN` | required | Logical target product shown to the Agent and Wiki and, unless overridden, sent to Agate as the worker token. Examples include `L20D`, `H20`, `B200` and PPU product tokens. |
| `--framework DSL` | auto-dispatch | Run one child Campaign in the named DSL. If omitted, AKA starts one independent child per framework supported by the detected vendor. In Production it is an acceptance constraint; in Leaderboard it is a preferred direction. |
| `--optimization-mode MODE` | `leaderboard` | `leaderboard` permits evidence-backed mixed implementations; `production` requires a deployable implementation in the selected framework and adds fail-closed source review. |
| `--[no-]expose-evaluator-shapes` | exposed | Expose `shapes.json` and detailed evaluator metadata, or request the generalized hidden-shape Production flow. Visibility is part of workspace identity. |
| `--workspace PATH` | current directory | Base directory under which AKA creates the child `kernel_opt_*` workspace. Use an absolute path for durable launches. |
| `--notes TEXT` | `none` | Extra operator constraints or known bottlenecks injected into Agent instructions. Do not put credentials here. |
| `--arch ARCH` | auto-probed | Override runtime architecture such as `sm_89`, `sm_100` or `gfx942`. Use only when probing fails or an intentional override is required; it is not the same as `--platform`. |

Mode, shape visibility, framework, candidate layout, composition and problem snapshot identity must match on resume. Start a fresh workspace when intentionally changing them.

## Agent, Framework Baseline and review

| Option | Default | Purpose and effect |
|---|---:|---|
| `--agent-cli CLI` | `claude` | Coding provider used by Framework Baseline and optimization Episodes: `claude`, `qodercli`, `codex` or `pi`. Authentication comes from the provider's existing configuration. |
| `--framework-baseline MODE` | `auto` | Controls V1: `auto` normally runs or pins V1 in Production, `always` requests it in Leaderboard too, and `never` starts Episodes from measured V0. It does not skip V0. |
| `--framework-baseline-timeout SECONDS` | `28800` | Wall-clock ceiling for the V1 Framework Baseline session. This is separate from optimization Episode active time and Agate timeouts. |
| `--setup-timeout SECONDS` | `7200` | In current Light, the per-attempt timeout for automatically authoring `agent_problem.json` when hidden-shape Production lacks one; at most two attempts are made. It is not the V1 timeout. |
| `--handoff-resumes N` | `2` | Same-session repair turns allowed after an optimization Episode produces an incomplete terminal handoff. `0` disables these repairs. |
| `--production-review-timeout SECONDS` | `600` | Timeout for each independent read-only Production policy review session. The review is separate from evaluator correctness. |
| `--[no-]v1-ask-codex` | off | Ask Codex for bounded, read-only correctness-first guidance during V1. It does not edit or approve the candidate. |
| `--[no-]v1-ask-qoder` | off | Ask Qoder for the same V1 guidance. Both reviewers may be enabled. |

Optimization Episode active time is configured through the Workflow composition patch, not a campaign CLI flag; see [Composition options](#composition-options).

## Search budgets and stop conditions

| Option | Default | Purpose and effect |
|---|---:|---|
| `--max-iters N` | `20` | Inclusive canonical-version ceiling. `20` permits versions through `v20`; it is not a count of successful promotions. See [Operations: Budgets](operations.md#budgets). |
| `--token-budget N` | `0` | Admission cap for reported optimization-Episode tokens; `0` disables it. In-flight work may overshoot, Bootstrap/auxiliary usage is excluded, and a provider that reports no tokens cannot be fully enforced. |
| `--target-util PCT` | `90` | Stop after a promoted candidate's recorded peak utilization reaches this percentage. Set `0` to disable. This is a post-promotion stop, not an acceptance substitute. |
| `--max-stall N` | `0` | Stop after `N` consecutive unpromoted Episodes; `0` disables it. Mandatory conversion and optional Goal Episode policy take precedence when applicable. |
| `--convert-after N` | `3` | For a Triton-family serial Campaign with source checks, require Triton-to-Gluon conversion after `N` consecutive stalls. Failed conversions remain latched until a Gluon candidate passes; `0` disables conversion. |

Pooled multi-trajectory search uses its own `epochs × trajectories × episodes_per_trajectory` schedule instead of the serial `--max-iters` budget. `--token-budget` remains a shared admission cap and already-running Episodes may overshoot it.

## Evaluation, ABBA and promotion

| Option | Default | Purpose and effect |
|---|---:|---|
| `--sandbox-hardware TOKEN` | `--platform` | Agate worker token. Set it when the logical product name shown to the Agent differs from the executor alias. |
| `--sandbox-profile {pre,prod}` | normal Agate config | Select an Agate endpoint profile. Mutually exclusive with `--sandbox-url` and `--sandbox-ssh`. |
| `--sandbox-url URL` | normal Agate config | Explicit Agate endpoint, including a local gateway. Mutually exclusive with endpoint profile and SSH transport. |
| `--sandbox-timeout SECONDS` | derived | Advanced enclosing Eval/ABBA execution allocation. By default AKA derives it from candidate, performance and ABBA run limits; queue grace is separate. An explicit value must contain the full ABBA schedule. |
| `--verify-repeats N` | `2` | Native ABBA measurements per side. Light requires an even value from 2 through 16; `2` represents one A/B/B/A block. |
| `--verify-run-timeout SECONDS` | `300` | Execution budget for each individual A or B measurement. The enclosing allocation is derived from this and `--verify-repeats` unless explicitly overridden. |
| `--min-improvement-pct PCT` | `0` | Candidate GeoMean latency improvement must be strictly greater than this percentage to promote. Correctness and complete evidence remain mandatory. |

Current Light measurement defaults are one Shape per batch, up to 16 batches in parallel, 600 seconds for candidate initialization/correctness, 300 seconds for steady-state performance, five warmups and 20 timing iterations. These are measurement-policy defaults rather than all being public Campaign flags.

## SSH execution options

Use SSH only when GPU work must run in a mandatory remote Bubblewrap namespace instead of Agate.

| Option | Default | Purpose and effect |
|---|---:|---|
| `--sandbox-ssh [USER@]HOST` | off | Select OpenSSH transport. It is mutually exclusive with Agate URL/profile and requires an explicit `--framework`. |
| `--sandbox-ssh-init COMMAND` | environment or empty | Activation command run before each remote job and health probe. |
| `--sandbox-ssh-gpu INDEX` | environment | Physical NVIDIA GPU index, 0–31. Required for SSH; MIG/UUID selectors are not supported. |
| `--sandbox-ssh-runtime-bind REMOTE[=SANDBOX]` | environment or none | Repeatable read-only runtime directory mounted into the remote Bubblewrap namespace. |
| `--sandbox-health-command COMMAND` | built-in SSH probe | GPU readiness command used for failure classification and recovery. |
| `--environment-poll-interval SECONDS` | `60` | Delay between probes while a blocked SSH environment is recovering. |

SSH Light does not provide native Atrex-Bench ABBA for every problem/layout; verify support before starting a long campaign.

## Composition options

Composition patches are JSON files passed with launch-level `--patch` before `--`. A patch supplies the whole configuration for the named module; omitted fields receive that provider's defaults, not values from the selected Profile. In a Workflow patch used by `aka optimize`, retain `"startup": true` or the Campaign has no startup service.

```json
{
  "api_version": 1,
  "patches": [
    {
      "id": "workflow",
      "config": {
        "startup": true,
        "numerical_validation": false,
        "goal_episodes": false,
        "generate_memory": false,
        "output_limit_recovery_retries": 2,
        "episode_timeout_seconds": 14400
      }
    },
    {
      "id": "tool_runtime",
      "config": {
        "measurement_repetitions": 1,
        "pause_agent_during_requests": true,
        "wiki": true,
        "queue_timeout": 60
      }
    }
  ]
}
```

| Module | Configurable behavior |
|---|---|
| `agent` | `sandbox` (`none`/`bwrap`), session `capture`, provider `timeout_seconds` (default 28800), Bubblewrap executable, `proc_mode` (`private`/`projected`) and additional absolute read-only paths. The named `light-sandbox` Profile already selects the intended sandbox/capture combination. |
| `tool_runtime` | Measurement aggregation repetitions (`1` or `3`), whether to pause the Agent during authenticated Runtime requests (default true), GPU/Wiki request queue and request timeouts, Wiki enablement, and Wiki model/effort/timeouts/concurrency. These do not replace Agate's own execution limits. |
| `workflow` | Optimization Episode active-time limit (default 14400 seconds), optional supplemental numerical validation, Goal Episodes, generated memory summaries, output-limit recovery count (default 2), pooled search topology, and advanced source-tree import. |
| `candidate` / `workspace` | Candidate filesystem (`source_tree`, `single_file`, or `cuda_manifest`) and whether custom reusable state is inherited across Episodes. Layout is fixed for the Campaign. |
| `light_journal` | `research_records`: enable or disable Direction/Experiment research records. Evaluator facts, candidate reports and promotion evidence remain required either way. |
| `wiki` | Knowledge root or a replacement provider. The Tool Runtime separately decides whether Wiki is enabled and how queries execute. |

For pooled search, `workflow.config.search` accepts:

| Field | Default | Meaning |
|---|---:|---|
| `epochs` | `10` | Number of barriered search Epochs. |
| `trajectories` | `2` | Number of private trajectory workspaces. |
| `episodes_per_trajectory` | `1` | Sequential Episodes each trajectory runs during one Epoch. |
| `max_parallel_episodes` | `2` | Maximum simultaneously running trajectory Episodes; it cannot exceed `trajectories`. |

Completed Epoch winners publish only verified Kernel bytes and performance facts. Live drafts, Skills, tools, scratch, Conversation history and research state are not merged between trajectories.

## Related environment controls

| Variable | Default | Purpose |
|---|---:|---|
| `ATREX_SANDBOX_QUEUE_WAIT_GRACE` | `120` in Light | Additional remote queue-wait allowance. It is separate from execution time and may be explicitly overridden. |
| `ATREX_CODEX_SESSION_SETTINGS` | unset | JSON settings forwarded to Codex sessions; falls back to `ATREX_SESSION_SETTINGS`. |
| `ATREX_PI_SESSION_SETTINGS` | unset | JSON settings forwarded to Pi sessions; falls back to `ATREX_SESSION_SETTINGS`. |
| `ATREX_SANDBOX_SSH*` | unset | Environment equivalents for SSH target, init command, GPU and runtime binds. Explicit CLI values take precedence. |

Keep credentials in provider/Agate configuration or secret injection, not command lines, notes or composition JSON.

## Shared-parser options that do not configure Light

The optimization CLI is shared with Legacy. On current `main`, the following accepted flags are not copied into the Light `ManagedCampaign` and should not be used to claim a Light behavior change:

| Retained flag | Legacy purpose / Light behavior |
|---|---|
| `--fast-episodes`, `--fast-trials` | Configure Legacy fast Episode structure. Light uses unified Episodes; these flags do not configure it. |
| `--fast-episode-ask-*`, `--full-episode-ask-*` | Configure Legacy Episode reviewers. Light currently consumes only the V1 reviewer flags. |
| `--long-reviewer-session` | Reuses a Legacy reviewer session; it is not a Light Campaign control. |
| `--review-trace-mode` | Controls Legacy reviewer trace retention; it is not wired into Light. |
| `--gateway-candidate-recovery` | Legacy Framework Baseline candidate-repair policy; Light uses its own typed feedback and recovery path. |
| `--resume-framework-candidate` | Legacy retained-candidate adoption switch; Light restores V1 from its own durable state. |
| `--conversion-budget-action` | Legacy behavior at a conversion budget boundary. Light fails explicitly when mandatory conversion remains incomplete at its budget boundary. |
| `--verify-shape-batch-size` | Legacy ABBA batch control. Current Light uses one Shape per batch with bounded parallelism and does not consume this CLI field. |

If a later `main` wires one of these settings into `ManagedCampaign`, update this reference from the implementation rather than inferring support from parser acceptance alone.

## Agate-native Runtime options

Inside a Light Episode, Runtime tools discover safe, non-authority Agate options from the installed `atrex-gateway-client` and expose them through the tool schema. Updating the client and restarting the Supervisor refreshes that catalog. Credentials, endpoints, uploads, source selection, timeouts, evaluator version and other authority-bearing options remain Supervisor-owned and are deliberately blocked from this dynamic pass-through.

Use the installed Runtime tool help/schema for the exact operation-specific fields. Do not assume an option supported by Agate `eval` is also valid for Profile, Check, Disassemble or Dev.
