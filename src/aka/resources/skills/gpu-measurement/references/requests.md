# Runtime measurement requests

Run `python3 tools/sandbox.py --help` for this Campaign's enabled tools.
Read `python3 tools/sandbox.py --kind OPERATION --help` before using unfamiliar options.
The returned schema is authoritative for that Runtime tool, including its supported native Agate
options. Hardware, credentials, evaluator inputs and infrastructure retries belong to the Supervisor.
Wiki, research records and submission use their separately installed Skills.

## evaluate

```bash
python3 tools/sandbox.py --kind evaluate --no-sync
# Atrex-Bench only; diagnostic rather than full submission evidence:
python3 tools/sandbox.py --kind evaluate --mode correctness_only --no-sync
# ABBA comparison; baseline must be reachable in Agent Git:
python3 tools/sandbox.py --kind evaluate --baseline-commit FULL_GIT_COMMIT --comparison-repeats 2 --no-sync
```

`--comparison-repeats` is the comparison sample count: use an even integer from 2 to 16.
Each native ABBA block supplies two samples per candidate.

Commit the complete candidate and keep Git clean. Standard full Evaluate with passing correctness
and complete per-Shape performance supports submit-kernel. Subsets, custom inputs and ABBA are
diagnostic and do not replace it. ABBA compares the current candidate with the specified baseline
commit; Runtime exports both source trees. There is no separate comparison tool.
SOL uses its official workload; Atrex-Bench-specific input/mode options do not apply to SOL.
Evaluate returns correctness and measured `latency_us_geomean`, `latency_us_arith_mean`, and
`latency_us_by_shape` (microseconds), plus measurement scope and diagnostics. ABBA returns these
latencies separately for baseline and candidate. No reference-relative speedup, normalization score
or improvement percentage is generated. `gateway_record_id` and `record_path` locate the saved
result; Kernel identities are Agent Git commit IDs. Raw Agate evidence remains in the private archive.
Omitted timing options inherit Campaign policy; explicit options are validated without silent
clipping. Query this tool's help for effective warmup, benchmark and timeout settings.

## command

```bash
python3 tools/sandbox.py --kind command -- python3 -m scratch.probe
```

Uploads the engineering workspace, including candidate, scratch and safe tools, and synchronizes
scratch artifacts back. Git internals, control files, Skills and bulk records are excluded.
Use module form for Python scripts so the workspace root stays importable. Custom code may compile,
profile or test hypotheses, but does not replace official evaluation. Packages, processes and other
remote state are not retained between calls.
The receipt includes complete stdout, stderr and the exit code, including failed commands.

## env

```bash
python3 tools/sandbox.py --kind env
```

Observes the remote target, not the local Agent host. Hardware defaults to the Campaign selection.
The complete redacted capability result is returned and archived, without becoming correctness
evidence. Its task_worker_requirements identifies Task dependency pins; match these when selecting
imports and APIs. Read this tool's help for optional capability queries.

## profile

```bash
python3 tools/sandbox.py --kind profile --help
python3 tools/sandbox.py --kind profile
```

Profiles the current Kernel with managed inputs. Use this tool's help for supported profilers,
levels, counters and Shape selection. Counters and profiler classifications are diagnostic clues,
not correctness evidence or proof of a causal mechanism.

## dev

```bash
python3 tools/sandbox.py --kind dev -- python3 -c 'print("probe")'
```

Runs self-contained probes without uploading the candidate workspace automatically. Declare every
extra workspace input with --input and returned files with --sync. This is diagnostic evidence;
remote state is not retained between calls.
The receipt includes complete stdout, stderr and the exit code, including failed probes.

## check

```bash
python3 tools/sandbox.py --kind check --help
python3 tools/sandbox.py --kind check
```

Requests compilation diagnostics for the current Kernel. A successful check does not establish
full workload correctness or performance, and may not exercise kernels compiled on first launch.

## disassemble

```bash
python3 tools/sandbox.py --kind disassemble --help
python3 tools/sandbox.py --kind disassemble
```

Requests assembly diagnostics for the current Kernel. Query this tool's help for supported output
formats. Assembly is diagnostic evidence, not a correctness or performance measurement.

## Waiting and results

With Agent suspension disabled, independent work may continue while a call is pending. Collect the
same call's receipt before using its results; use the Agent backend's task/output facility for
background calls. Keep in-flight request inputs unchanged, and do not add shell timeouts or parallel
duplicates. --no-sync disables artifact synchronization; it does not make the Runtime client return early.

Runtime returns the canonical operation/status/result receipt and record_path. The same result is
saved at records/<gateway_record_id>.json; inspect it with ordinary file/JSON tools when needed.
Inspect correctness and inner errors even when transport completed successfully. Kernel identity
is an Agent Git commit. Read historical source with Git and saved results directly.

Follow error.next_action and upstream diagnostics. Runtime owns infrastructure retries. A lost
connection does not prove that no remote job ran; do not add a blind retry loop. Compile, import,
correctness and candidate policy errors require code repair.
