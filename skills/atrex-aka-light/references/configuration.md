# AKA Light configuration

Read this file when constructing or reviewing a Light launch. The selected checkout and `aka optimize --repo-root REPO -- --help` are authoritative when a newer revision differs.

## Argument boundary

```bash
aka optimize \
  --repo-root /path/to/atrex-kernel-agent \
  --profile light-sandbox \
  --patch /path/to/light-options.json \
  -- \
  --op-dir /path/to/operator \
  --platform L20D \
  --framework Cuda
```

Options before `--` choose installed components. Options after `--` configure the campaign.

### Launch options

| Option | Default | Meaning |
|---|---:|---|
| `--repo-root PATH` | required | Checkout whose profiles, resources and packaged Skills match the installed wheels |
| `--profile PROFILE` | `light` | `light`, `light-sandbox`, a Legacy profile, or an explicit Profile JSON |
| `--patch FILE` | none | Repeatable composition patch; each module entry replaces that module's whole configuration |
| `--execution-profile PROFILE` | selected default | Advanced independent prepared-Execution selection |
| `--execution-patch FILE` | none | Repeatable patch for that independent Execution selection |

Recorded continuations restore their frozen launch selection. Do not add a new profile or patch while resuming a recorded continuation.

## Campaign identity and policy

| Option | Default | Meaning |
|---|---:|---|
| `--op-dir PATH` | required | Supported Native Atrex-Bench or SOL operator directory |
| `--platform TOKEN` | required | Logical target product shown to the Agent and normally used as the Agate worker token |
| `--framework DSL` | none | Required only by Ultra; Fast and Performance leave DSL selection to the Agent |
| `--[no-]specify-dsl` | derived | Fast/Performance are open DSL; Ultra forces a pinned DSL |
| `--optimization-mode MODE` | `fast` | `fast`, `performance`, or `ultra`; historical names remain resume aliases |
| `--[no-]expose-evaluator-shapes` | exposed | Expose evaluator Shapes, or request the supported hidden Performance/Ultra flow |
| `--workspace PATH` | current directory | Base for durable child campaign workspaces |
| `--notes TEXT` | `none` | Additional operator constraints; never put credentials here |
| `--arch ARCH` | probed | Intentional runtime-architecture override such as `sm_103a` or `gfx942` |
| `--agent-cli CLI` | `claude` | Agent provider such as `claude`, `qodercli`, `codex`, or `pi` |
| `--agent-reference-dir DIR...` | none | Repeatable read-only mounts at `references/<basename>/`; requires Bubblewrap and unique basenames |
| `--setup-timeout SECONDS` | `7200` | Per-attempt authoring limit for a missing hidden-flow `agent_problem.json`; not an Episode limit |

Mode, Shape visibility, framework, candidate layout, composition and problem snapshot are resume identity. A hidden request that cannot actually produce the hidden projection fails; it is not recorded as hidden while exposing the original inputs.

## Search and stop conditions

Current Light defaults to pooled search. Configure it through `workflow.config.search`:

| Field | Default | Meaning |
|---|---:|---|
| `epochs` | `10` | Search Epochs |
| `trajectories` | `3` | Persistent private Agent trajectories |
| `episodes_per_trajectory` | `1` | Audit segments per trajectory per Epoch |
| `max_parallel_episodes` | `3` | Maximum concurrent trajectory segments; cannot exceed trajectories |
| `synchronization` | `synchronous` | `synchronous` barrier publication or `asynchronous` publication after completed segments |
| `role_specialization` | `false` | Add distinct trajectory role prompts |

An explicit empty mapping selects serial mode:

```json
{
  "api_version": 1,
  "patches": [
    {"id": "workflow", "config": {"startup": true, "search": {}}}
  ]
}
```

| Option | Default | Meaning |
|---|---:|---|
| `--max-iters N` | `20` | Number of Episode audit segments in serial mode; pooled search uses its own schedule |
| `--token-budget N` | `0` | Shared admission cap for reported main-Agent tokens; Review tokens are counted separately and do not consume this cap |

Episode active time is unlimited unless `workflow.config.episode_timeout_seconds` is a finite positive value. Agate queue, request and per-Shape execution budgets are independent.

## Evaluation and explicit ABBA

| Option | Default | Meaning |
|---|---:|---|
| `--sandbox-hardware TOKEN` | `--platform` | Agate worker token when it differs from the logical platform |
| `--sandbox-profile {pre,prod}` | configured Agate route | Endpoint profile; mutually exclusive with URL and SSH |
| `--sandbox-url URL` | configured Agate route | Explicit endpoint; mutually exclusive with profile and SSH |
| `--sandbox-timeout SECONDS` | derived | Enclosing request allocation; queue grace is separate |
| `--verify-repeats N` | `2` | Explicit ABBA measurements per side; even value from 2 through 16 |
| `--evaluation-step-timeout SECONDS` | `60` | Runtime-owned per-Shape candidate/correctness, performance and individual ABBA-run budget |
| `--verify-shape-batch-size N` | `10` | Maximum Shape IDs in one outer Evaluate or ABBA batch |
| `--gpu-clock-mhz MHZ` | compatibility only | Current Agate requests send `lock_clocks=true` without a fixed MHz |

Standard Evaluate and explicit ABBA use outer batches of at most 10 Shapes, Ray Data with `shape_shards=1`, 20 warmups, 100 timing iterations and deterministic correctness cases. Standard Evaluate uploads the complete operator and uses native `shape_ids`. Until Agate ABBA supports Shape selectors, Light rebuilds each ABBA batch as a complete operator reference containing exactly that batch and omits `shape_ids`; this removable adapter lives in `aka.light.evaluation.abba_shape_compat`. Light merges all batches before applying `metadata.performance_objective`.

Explicit ABBA is an Agent tool, not an automatic Episode terminal gate. `submit-kernel` requires a passing standard full Evaluate with complete per-Shape performance for the exact commit; it does not require ABBA or a speedup threshold.

When the Agent supplies an operation-specific timeout, Runtime honors it if the enclosing schedule can represent it. Otherwise Runtime returns the exact schedule conflict instead of silently replacing it. When omitted, Runtime derives a compatible value and its help must describe that behavior.

## SSH execution

Use SSH only when GPU work must run in a managed remote namespace rather than Agate.

| Option | Meaning |
|---|---|
| `--sandbox-ssh [USER@]HOST` | Select OpenSSH transport; mutually exclusive with Agate profile/URL and requires explicit framework |
| `--sandbox-ssh-init COMMAND` | Activation command before each remote job and health probe |
| `--sandbox-ssh-gpu INDEX` | Assigned physical NVIDIA GPU index |
| `--sandbox-ssh-runtime-bind REMOTE[=SANDBOX]` | Repeatable read-only runtime bind |
| `--sandbox-health-command COMMAND` | Remote GPU readiness probe |
| `--environment-poll-interval SECONDS` | Recovery-probe interval |

The current SSH path supports Native Evaluate, explicit ABBA, Profile, Check, Disassemble and Env for supported candidate layouts. Verify the selected checkout before relying on SOL or multi-GPU behavior.

## Composition options

Composition patches are JSON files passed before `--`. A module entry supplies the whole configuration for that module; omitted fields receive provider defaults. In CLI Workflow patches retain `"startup": true`.

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
        "output_limit_recovery_retries": 5,
        "episode_timeout_seconds": null,
        "search": {
          "epochs": 10,
          "trajectories": 3,
          "episodes_per_trajectory": 1,
          "max_parallel_episodes": 3,
          "synchronization": "synchronous",
          "role_specialization": false
        }
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

| Module | Current Light behavior |
|---|---|
| `agent` | `sandbox`, capture, provider wall timeout, Bubblewrap binary, `proc_mode: private\|host`, and extra read-only paths. The provider-wide default is `private`; Aone Sandbox launch configuration defaults to explicit `host`. |
| `tool_runtime` | Measurement repetitions, Agent pause during Runtime calls, GPU/Wiki queues and timeouts, Wiki configuration |
| `workflow` | Optional active-time limit, supplemental numerical validation, goal segmentation, generated memory, output-limit recovery, pooled search and source import |
| `candidate` / `workspace` | `source_tree`, `single_file`, or `cuda_manifest`, plus persistent state handling |
| `light_journal` | `tool_modules`: any subset of `directions` and `experiments`; empty disables both annotations |
| `wiki` | Knowledge provider/root; Tool Runtime separately decides whether Wiki is exposed |

Direction/Experiment annotations are optional. Standard Evaluate evidence, exact Agent Git commits, `submit-kernel` receipts and accepted-submission winner selection remain available when both modules are disabled.

Pooled publication shares eligible research records, relevant Gateway Records, peer Agent Git refs and accepted-submission winner facts. Scratch and conversation histories remain private. A peer ref is for inspection/cherry-pick and never changes local HEAD automatically.

## Environment controls

| Variable | Default | Meaning |
|---|---:|---|
| `ATREX_SANDBOX_QUEUE_WAIT_GRACE` | `120` in Light | Extra remote queue-wait allowance |
| `ATREX_CODEX_SESSION_SETTINGS` | unset | Codex settings JSON, falling back to `ATREX_SESSION_SETTINGS` |
| `ATREX_PI_SESSION_SETTINGS` | unset | Pi settings JSON, falling back to `ATREX_SESSION_SETTINGS` |
| `ATREX_SANDBOX_SSH*` | unset | Environment equivalents for SSH controls; explicit CLI wins |

Keep credentials in provider/Agate configuration or secret injection, not notes or composition JSON.

## Legacy-only shared-parser flags

The optimization CLI is shared with Legacy. Parser acceptance does not prove a flag configures Light. Treat these old controls as Legacy-only unless the selected checkout explicitly wires them into `ManagedCampaign`:

- `--framework-baseline`, `--framework-baseline-timeout`;
- `--handoff-resumes`, `--v1-ask-codex`, `--v1-ask-qoder`;
- `--min-improvement-pct`, `--target-util`, `--max-stall`, `--convert-after`;
- `--fast-episodes`, `--fast-trials`, and Fast/Full reviewer flags;
- `--gateway-candidate-recovery`, `--resume-framework-candidate`, and conversion-budget controls.

Do not describe any of these as current Light behavior merely because `--help` still lists it.

## Agate-native Runtime options

Runtime tools discover safe non-authority options from the installed gateway client and expose them in operation-specific help/schema. Updating the client and restarting the Supervisor refreshes that catalog. Credentials, endpoint, uploads, source selection, evaluator version and Supervisor-owned deadlines are blocked from Agent pass-through.

An option accepted by Agate Evaluate is not automatically valid for Profile, Check, Disassemble, Env, Dev or Command. Use the installed Runtime help for the exact operation.
