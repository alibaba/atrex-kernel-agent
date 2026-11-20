# Task interface and implementation lineage

## Parent-facing interface

The primary parent API is `aka.submit(...)`, returning a reconnectable Job.
The JSON CLI and the Python API call the same TaskService and worker. A parent
Agent with shell access can use the CLI directly.
There is no parent-facing network daemon, MCP server, webhook or remote upload
API. Callers on another machine need their own command bridge and file transport.

```python
import aka
job = aka.submit(
    contract=module_contract,
    baseline=current_source, seed=library_source,
    workload=recorded_workload, validation=precision_profile,
    execution={"backend": "agate", "gpu": "H100"},
    objective=optimization_goal, budget=budget,
    notes="Profile the bottleneck with NCU via aka execution; retain and explain the report.",
    key="module-revision-1",
)
events = job.events(after=cursor)
results = job.results()
job.export(candidate_id, destination)
job.feedback(candidate_id, {"adopted": True, "integration_latency_us": 123})
# Increase total admission limits and continue the same session, even after exhaustion:
job.resume(budget={"episodes": 4, "evaluations": 60})
# Reconnect without resubmitting or rebuilding a contract:
job = aka.job(job.id)
```

Published candidates can be exported immediately while research continues. An
external caller may test an export in its own application and send `feedback`;
this is optional caller policy, not another AKA acceptance stage. AKA does not
automatically launch a parent-side process or require a second benchmark.
The task's frozen contract defines formal acceptance. The caller decides when
to inspect candidates, perform integration tests and adopt a result.

Concurrent external measurement requests receive `TaskBusy` (HTTP 409 from the
session tool host) without changing task or evaluation state. Recovery workers
wait for the current measurement owner, then reuse the saved evaluation and GPU
receipt. Use task start/resume to recover interrupted work rather than launching
independent review or measurement recovery loops for the same task.

`contract` is either a reusable contract-package directory or
`{"binding": "team.module.v1", "source": domain_context}`. A directory contains
`contract.json` (adapter/config) and its trusted verifier/reference files; a custom
`python:file.py:Class` adapter can live in that package, without installation.
`baseline` is the current implementation; `seed` defaults to baseline.
`workload` and `validation` are JSON objects or JSON files frozen into the package.
`agent` defaults to a Claude persistent session; `agent={}` selects external
source authoring. `root`, `key`, `parent_task`, `start` are lifecycle options.
A generated key means a new task; provide a stable key for idempotency.

`notes` is optional caller-authored text for research priorities, hypotheses,
requested experiments and reporting. It is included in every research invocation
and saved in task/export metadata. The JSON submission accepts the same field.
Changing notes requires a new submission key but does not change contract identity;
notes cannot override the frozen validator, implementation rules or budget. A
requested NCU analysis remains an Agent instruction, not an automatically enforced
candidate publication gate. Make mandatory deliverables executable contract
checks if they must be required for acceptance. Later candidate-specific direction
can be sent through `feedback` and read on the next Agent invocation.

Installed `aka.task_bindings` are optional conveniences that translate domain
objects into a package once at submission. They receive `staging, source,
baseline, seed, workload, validation`. The built-in `module.v1` accepts the older
`module.json` plus `contract/` layout; `atrex-bench.v1` accepts a reference directory,
with workload replacing shapes and validation supplying runner overrides.
The earlier `adapter=..., source=...` Python spelling remains supported.
The binding is not another task, executor or research loop.

All commands take `aka task --root ROOT ...`. The default root is
`~/.atrex/aka-tasks`. Output is JSON; handled errors go to stderr and exit 1.

| Command | Arguments | Behavior |
| --- | --- | --- |
| `submit` | `SPEC --key KEY [--no-start]` | Freeze input, create/reuse an idempotent task, normally launch background worker |
| `status` | `TASK_ID` | State, control, counts, budgets consumed, workspace and contract ID |
| `start` | `TASK_ID` | Start/recover a worker; paused tasks need explicit resume |
| `events` | `TASK_ID [--after SEQ]` | Up to 1000 events after a cursor; advance to last returned seq |
| `results` | `TASK_ID` | Evaluations, accepted candidates, objective and frontier |
| `export` | `TASK_ID CANDIDATE_ID DESTINATION` | Export exact source, contract, evidence and delivery; destination must not exist |
| `feedback` | `TASK_ID CANDIDATE_ID JSON_FILE` | Store caller-defined integration feedback for an accepted candidate |
| `control` | `TASK_ID pause\|resume\|cancel [--budget JSON_FILE]` | Request control; resume can patch admission limits; cancellation completion is separately observed |
| `measure` | `TASK_ID [--role candidate\|baseline] [--scope acceptance] [--evaluation-id ID]` | Evaluate committed source or recover an existing evaluation; blocks until result/interruption |
| `submit-candidate` | `TASK_ID EVALUATION_ID` | Publish an eligible acceptance evaluation |

`work TASK_ID` is the foreground worker entry used by detached execution.
`invoke` is the coding Agent's tool client, not the parent task-management API.

### Submission manifest

The direct-package JSON uses the same fields as the Python call: `contract`,
`baseline`, optional `seed/workload/validation`, `execution`, `objective`, optional
`budget/agent`. Both direct entry forms default to Claude; `agent={}` selects
external authoring. Only the earlier normalized manifest keeps its omitted-agent
external-authoring default for compatibility. The table below describes the normalized
manifest; both entry forms feed this same representation.

Paths resolve relative to SPEC, not the caller's working directory. A reused key
requires identical content. The returned task ID identifies the long-running
task; submission does not wait for research completion.

| Field | Required / default | Meaning |
| --- | --- | --- |
| `schema_version` | required, `1` | Manifest version |
| `adapter` | required | `command`, `atrex-bench` or `python:file.py:Class` |
| `contract` | required | Directory of trusted semantic definition, workload, validation and adapter files |
| `seed` | required | Initial candidate source directory |
| `baseline` | defaults to seed | Current implementation source directory; separate from the semantic oracle |
| `target` | required | Execution backend and target configuration |
| `objective` | required | Metric ordering/constraints, or custom adapter objective configuration |
| `config` | adapter-defined | Instructions, harness argv, delivery and other domain inputs |
| `agent` | omitted = external authoring | Runtime configuration; a nonempty section enables the coding session |
| `budget` | optional | Admission limits below |
| `name`, `parent_task` | optional | Caller metadata; parent_task does not schedule dependencies |

Target fields: `backend` is `agate`, `remote`, `local`, or an installed
`aka.task_executors` entry point name; `config` carries backend-specific options;
`gpu` is mandatory for Agate; `profile` defaults to `prod` and accepts `prod/pre`;
`url` optionally overrides gateway URL; `num_gpus` defaults to 1;
`timeout_s` defaults to 600 and limits each execution request.
`profile` here is an environment selector, not a profiling operation.

Agent fields: `runtime` defaults to `claude`; `reasoning_effort` defaults to
`high`; `sandbox` defaults to `none`. `model` selects a model, `provider` selects
a pi provider, and `profile` selects host configuration from `~/.atrex/aka-agents.json`
(or `AKA_AGENT_PROFILES`). Task options override profile defaults. Supported native
persistent runtimes include Claude, Codex and pi. Unsupported continuation or a
missing executable is detected before GPU admission. Pi uses its current
`--print --mode json --session ID` protocol and task-owned session storage.

A host profile can set `runtime`, `model`, `provider`, `environment`, and
`secret_environment` mapping process variable names to existing host variable
names. For example:

```json
{"team": {"runtime": "claude", "model": "configured-model",
          "environment": {"ANTHROPIC_BASE_URL": "https://your-provider.example"},
          "secret_environment": {"ANTHROPIC_AUTH_TOKEN": "TEAM_API_KEY"}}}
```

The task stores only `{"profile":"team"}` and explicit nonsecret options.
Credentials stay in the host environment; use the same profile on worker recovery.
Existing per-runtime session settings are preserved, with explicit model/provider
options overriding them.

Budget fields: `evaluations` defaults to 100 and includes baseline/seed;
`executions` defaults to 100 and independently bounds research commands;
`episodes` defaults to 1 and counts native Agent invocations;
`agent_seconds` defaults to 3600 per invocation; `tokens`, when specified, controls
future admission using reported research and independent-review usage, not an in-flight hard stop. Unknown usage
blocks further budgeted admission. All explicit budget values are positive ints.
Resume budget values replace total limits, not the amount remaining. Existing
usage, source, candidates, contract ID and native session are retained. Limits
are read before the next admission; `agent_seconds` takes effect when a worker
opens its session and never changes an already running invocation's deadline.
The ledger holds the current budget; task.json in the task directory preserves
the original submission, while export includes the current manifest.

The default objective accepts `ordering=lexicographic|pareto` and nonempty
`metrics` with `name`, `unit`, `direction=minimize|maximize`, optional `context`
and absolute `tolerance`. Optional `constraints` have `name`, `unit`, optional
`context`, `operator=le|ge` and numeric `limit`.

For `command`, config requires `command` (argv with `{contract}`, `{candidate}`,
optional `{scope}` placeholders), `instructions`, and `delivery`. An optional
`result_prefix` changes `AKA_TASK_RESULT=`. The caller-owned harness expresses
precision, state, aliasing, workload and dependency semantics; these are not
separate universal top-level fields that AKA can infer.

## Coding-Agent tools and GPU operations

The worker starts a loopback HTTP endpoint with a random task-scoped token. Its
client is `aka task invoke OPERATION ...`, using injected `AKA_TASK_ENDPOINT`
and `AKA_TASK_TOKEN`. Allowed operations:

| Operation | Arguments |
| --- | --- |
| `measure` | optional `--scope`, `--evaluation-id` |
| `submit-candidate` | required `--evaluation-id` |
| `results` | none |
| `events` | optional `--after` |
| `execute` | internal transport for `aka execution`, described below |

The coding Agent uses the existing `aka execution` entry point inside its bound
session. The task selects the backend; the Agent selects commands and experiments:

```sh
aka execution --input probe.py=scratch/probe.py --output report.txt -- python3 probe.py
aka execution -- bash -lc 'nvcc experiment.cu -o experiment && ./experiment'
aka execution --execution-id EVALUATION_ID
```

The interface has `--input REMOTE=LOCAL`, `--output PATH`, `--env KEY=VALUE`,
and a command argv after `--`. The task timeout applies. Inputs support text and binary;
the current workspace, including scratch, is snapshotted without requiring a
commit. Each call is isolated, so dependent build/run commands belong in one call.
Local outputs return base64 content; Agate outputs use its OSS artifact references.
Output representations live in the execution receipt. Research is recorded in
the common ledger with role=research and cannot become candidate acceptance.

Agate executes research through dev; no profile/compile/disassemble operations
are exposed. Formal command harnesses use the same Executor through a command
plan. The native Atrex-Bench adapter retains eval as a compatibility capability.
Executors for other targets contribute `aka.task_executors` entry points. The
factory receives target, directory, execution_id, previous_job, on_job, control;
its object implements the existing Executor protocol. `on_job` checkpoints a
remote identity before waiting. JobResult can carry the raw ExecutionReceipt at
`details['receipt']`; otherwise stdout/stderr/exit_code are normalized by the host.

## Existing Kubernetes Pods

Install the `atrex-aka-light` wheel or run `python -m pip install .` in the source checkout. The optional Agate adapter also requires a compatible gateway SDK from your gateway operator. The host
runs `kubectl` with its existing credentials. The target container needs Linux
and Python 3. Single-card and distributed GPU measurements additionally need the
driver and the workload dependencies appropriate to that experiment. The local
backend is available with `execution={"backend": "local"}`.

```python
execution = {
    "backend": "remote",
    "config": {
        "context": "my-context",
        "namespace": "research",
        "pod": "gpu-worker",
        "container": "worker",
        "uid": pod_uid,
        "scheduling": {"gpu_ids": [0, 1, 2, 3], "gpu_count": 2, "replay_safe": True},
        "distributed": {"world_size": 2},
        "cache": {"mode": "cas"},
    },
    "timeout_s": 1200,
}
```

Obtain the Pod UID with `kubectl get pod gpu-worker -n research -o jsonpath='{.metadata.uid}'`.
For UID-pinned commands, expose the UID inside the target container using its
Pod specification. The transport checks it in the container immediately before
executing each command, including retries.

```yaml
env:
  - name: AKA_POD_UID
    valueFrom:
      fieldRef:
        fieldPath: metadata.uid
```

Choose an explicit container for Pods with multiple containers. The host needs
permission to read the Pod and execute in the target container. Pool admission
waits for every requested GPU. Enable replay only for repeatable commands.
Each attempt creates a fresh workspace. The content store reuses verified
input blobs and build artifacts across executions. Candidate evaluation,
publication and export use the frozen contract and its exact source revision.

A host may select a different transport executable with `AKA_REMOTE_TRANSPORT`.
It receives `identity --pod NAME [--context X] [--namespace N] [--container C] [--uid U]`
or `exec` with the same arguments followed by `-- COMMAND ARG...`. Identity
returns one JSON object containing Kubernetes-compatible `metadata.uid` and
`status.containerStatuses`. Exec must preserve argv, stdout, stderr, and exit
status and enforce every supplied UID at the target before forwarding the
command. The executable and its credentials are host configuration.
