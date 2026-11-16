# Runtime measurement requests

## Discover enabled operations

Run `python3 tools/sandbox.py --help` for this Campaign's available Agate tools. The default is
`env`, `evaluate`, and `command`; the launch configuration can select a different set. Only use
listed operations. Wiki, Direction/Experiment and submit-kernel have separate configuration.
Read `python3 tools/sandbox.py --kind OPERATION --help` before using unfamiliar options.
The Supervisor discovers safe native options from its installed Agate SDK. The returned schema
is authoritative for that operation; never transfer flags between kinds. Credentials, hardware,
evaluator inputs, retry policy and execution authority belong to the Supervisor.

## Evaluate and compare

When Evaluate is enabled:

```bash
python3 tools/sandbox.py --kind evaluate --no-sync
# Atrex-Bench only; diagnostic rather than full submission evidence:
python3 tools/sandbox.py --kind evaluate --mode correctness_only --no-sync
# ABBA is part of Evaluate; baseline must be reachable in Agent Git:
python3 tools/sandbox.py --kind evaluate --baseline-commit FULL_GIT_COMMIT --comparison-repeats 2 --no-sync
```

Commit the complete candidate and keep Git clean for Kernel-bound requests. Full passing standard
Evaluate with complete per-Shape performance supports submit-kernel. Custom inputs, subsets and
comparisons do not replace it. SOL uses its official workload through a private compatibility path;
Atrex-Bench-specific typed input/mode options do not apply to SOL. Omitted timing options inherit
Campaign defaults; explicit options are validated without silent clipping. Query the operation's help
for effective warmup, benchmark and timeout settings.

## Custom remote work and environment discovery

When the corresponding kind is enabled:

```bash
python3 tools/sandbox.py --kind env
python3 tools/sandbox.py --kind command -- python3 -m scratch.probe
```

Command uploads the engineering workspace, including candidate, scratch and safe tools, and
synchronizes scratch artifacts back. Git internals, control files, Skills and bulk records are
excluded. Use module form for Python scripts so the project root stays importable. Jobs are fresh
and stateless; packages, processes and other remote state are not retained between calls.
Custom code may compile, profile or test hypotheses, but cannot replace official full evaluation.
Env observes the remote target, not the local Agent host. Its task_worker_requirements identifies
the Task's dependency pins; match these when selecting imports and APIs.

## Waiting and results

Run calls in the foreground without shell timeouts or parallel duplicates. Wait for the same call;
use the Agent backend's wait/output facility if it backgrounds a long command. Do not edit the
candidate while measurement is pending. `--no-sync` disables artifact synchronization, not waiting.

Runtime returns the complete canonical operation/status/result receipt and record_path. The same
result is saved at records/<gateway_record_id>.json; inspect it with ordinary file/JSON tools when
needed later. Inspect correctness and inner errors even when the transport completed successfully.
Kernel identity is an Agent Git commit. Read historical source with Git and saved results directly.

Follow error.next_action and upstream diagnostics. Runtime owns infrastructure retries. A lost
connection does not prove that no remote job ran; do not add a blind retry loop. Compile, import,
correctness and candidate policy errors require code repair. Preserve uncertainty in interpretations;
profiler classifications and custom timings are diagnostic clues, not proofs of causal mechanisms.

## Optional Wiki

Wiki availability is campaign-configured. Only use these commands when the session mounts
`skills/KernelWiki/SKILL.md`; read that Skill for query scope and hardware identity.
Do not depend on a `gpu-wiki/` source directory in the Agent workspace. When Wiki is explicitly disabled,
use supplied public knowledge and measured evidence instead.

```bash
python3 tools/sandbox.py --kind wiki-query "Triton reduction on sm_120: measured bandwidth bottleneck"
python3 tools/sandbox.py --kind wiki-search --arch sm_120 --dsl triton --coverage
python3 tools/sandbox.py --kind wiki-hardware --product B200
```
