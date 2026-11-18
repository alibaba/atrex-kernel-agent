# Native Atrex-Bench Vector Add smoke input

This operator fixture is copied from the Vector Add package used by the live Aone
Sandbox functional campaigns. It is not a separately invented toy schema:

- `reference.py` defines the authoritative `torch.nn.Module` semantics;
- `input.py` defines `_make_inputs`, which turns one Shape's `input_kwargs` into CUDA
  tensors passed to `Model.forward`;
- `shapes.json` maps the integer-form string IDs `0` through `9` to
  `Model.__init__` and `_make_inputs` arguments;
- `metadata.json`, `roofline.json`, `shape_range.json`, and `agent_problem.json`
  preserve the same workload, scoring, domain, and hidden-Shape metadata used by
  that fixture.

Do not replace the Shape IDs with descriptive labels such as `small`, `medium`, or
`large`: the current Native Agate evaluation contract requires integer-form Shape
IDs. The Sandbox-local `task.json` is intentionally not copied because it pins an
old hardware target and Atrex-Bench build; launch hardware and checkout are explicit
user selections below.

## Required launch inputs

Before preparing or launching this example, explicitly obtain from the user:

1. a safe injection source for Agate `AGATE_AK` and `AGATE_SK` (the values must already be present
   in the Supervisor environment; never ask the user to paste them into chat);
2. an existing canonical Atrex-Bench checkout root, or a user-selected destination where
   `https://github.com/alibaba/atrex-bench` should be cloned; never assume
   `/path/to/atrex-bench`;
3. the exact target GPU/platform token accepted by Agate, such as `L20N` or `L20D`; examples are
   not defaults;
4. the Agent CLI: `claude` (Claude Code), `codex`, `qodercli`, or `pi`;
5. the optimization mode after explaining that Fast permits declared external core implementations,
   Performance requires self-authored core computation with an Agent-selected DSL, and Ultra also
   requires a user-pinned DSL.
6. whether to use the Bubblewrap Agent Sandbox (`light-sandbox`) or run the Agent natively without
   Bubblewrap (`light`). Clarify that this does not create or select an Aone Sandbox. When Sandbox
   is selected, also ask for `private` (private PID namespace/procfs) or `host` (shared outer PID
   namespace and `/proc`). Aone Sandbox supports only `host`.
7. the positive `max-iters` for the ordinary one-trajectory run. This is the number of Episode
   audit segments; the example does not choose it for the user.
8. whether to expose optional knowledge directories through `--agent-reference-paths`, or use none.
   Bubblewrap mounts them read-only. Native `light` links the originals without write protection;
   the user must retain an independent copy before exposing them.

Present the choices explicitly; the example has no preselected row:

| Required input | Choices to present |
|---|---|
| Agate credentials | Existing `AGATE_AK`/`AGATE_SK` in the Supervisor environment, or another already configured secure injection source; never paste values into chat |
| Atrex-Bench | Existing canonical checkout path, or clone `https://github.com/alibaba/atrex-bench` into a user-selected destination |
| GPU | Exact Agate platform/worker token selected by the user; `L20N` and `L20D` are examples, not an exhaustive list or default |
| Agent CLI | `claude` (Claude Code), `codex`, `qodercli`, or `pi` |
| Optimization mode | `fast`, `performance`, or `ultra` |
| Ultra DSL | Required only for Ultra; user selects a supported DSL such as `Cuda`, `Triton`, or `CuteDSL` |
| Agent boundary | `light` (native process) or `light-sandbox` (Bubblewrap on Linux/Lima) |
| Bubblewrap proc mode | Required only for `light-sandbox`: `private` or `host`; Aone Sandbox permits only `host` |
| Run length | Positive `max-iters`, the user-selected number of Episode audit segments for one independent trajectory |
| Reference knowledge | `none`, or one or more existing directories with unique basenames; native Light requires an independent backup because originals remain writable |

Native macOS does not silently select `light`: the user may choose native `light`, or choose
`light-sandbox` and provide/use a Lima Linux environment. Do not launch until every applicable row
has an explicit answer.

The commands below use these placeholders:

```bash
AKA_REPO=/path/to/atrex-kernel-agent
ATREX_BENCH_ROOT=/path/to/atrex-bench
CAMPAIGN_ROOT=/path/to/vector-add-campaign
AGENT_PATCH=/path/to/agent-boundary.json
SEARCH_PATCH=/path/to/search-schedule.json

: "${TARGET_PLATFORM:?set the user-selected GPU token}"
: "${AGENT_CLI:?set claude, codex, qodercli, or pi from the user selection}"
: "${OPTIMIZATION_MODE:?set fast, performance, or ultra from the user selection}"
: "${AGENT_PROFILE:?set light or light-sandbox from the user selection}"
: "${MAX_ITERS:?set the user-selected Episode count}"
# Also require AGENT_PROC_MODE=private|host when AGENT_PROFILE=light-sandbox.
# Also require ULTRA_FRAMEWORK when OPTIMIZATION_MODE=ultra.
# Set AGENT_REFERENCE_DIRS to an array of existing directories, or an empty array for none.
```

Set every required environment variable from the user's explicit selection; this example does not
choose an Atrex-Bench path, GPU, Agent CLI, optimization mode, Agent boundary, or run length. The
ordinary example explicitly uses one independent trajectory. Direction and Experiment annotations
are disabled by default and are not launch questions. For Ultra, require `ULTRA_FRAMEWORK`; do not
launch Ultra without the user's DSL choice. `light-sandbox` requires Linux/Bubblewrap; on native macOS the user must choose
`light` or run the sandboxed profile inside Lima. The command below materializes the selected proc
mode in a complete Agent composition patch. Inside Aone Sandbox, `host` must still be accepted by
the user; never launch `private` there. Proc mode is not applicable when
`AGENT_PROFILE=light`. See [installation](../../references/installation.md) for the boundary.

Select one Agent explicitly; none is preferred by this smoke example:

```bash
AGENT_CLI=claude   # Claude Code
# or: AGENT_CLI=codex
# or: AGENT_CLI=qodercli
# or: AGENT_CLI=pi
```

This directory is stored in the Skill, not under a canonical Atrex-Bench runtime. Copy
it into the `data/` tree of the user-provided Atrex-Bench checkout that will evaluate it:

```bash
cp -R \
  ~/.codex/skills/atrex-aka-light/examples/native-atrex-bench-vector-add \
  "$ATREX_BENCH_ROOT/data/native-atrex-bench-vector-add"
```

The resulting layout must have both runtime markers above the operator:

```text
/path/to/atrex-bench/
├── scripts/run_eval.py
├── src/atrex_bench/
└── data/native-atrex-bench-vector-add/
    ├── agent_problem.json
    ├── metadata.json
    ├── reference.py
    ├── input.py
    ├── roofline.json
    ├── shape_range.json
    └── shapes.json
```

Before launch, verify structure and credential presence without printing secrets:

```bash
test -f "$ATREX_BENCH_ROOT/scripts/run_eval.py"
test -d "$ATREX_BENCH_ROOT/src/atrex_bench"
test -f "$ATREX_BENCH_ROOT/data/native-atrex-bench-vector-add/reference.py"
test -f "$ATREX_BENCH_ROOT/data/native-atrex-bench-vector-add/input.py"
test -s "$ATREX_BENCH_ROOT/data/native-atrex-bench-vector-add/shapes.json"
python3 - "$ATREX_BENCH_ROOT/data/native-atrex-bench-vector-add/shapes.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    shapes = json.load(handle)
assert shapes
assert all(isinstance(shape_id, str) and shape_id.isdecimal() for shape_id in shapes)
assert sorted(map(int, shapes)) == list(range(len(shapes)))
PY
test -n "${AGATE_AK:-}"
test -n "${AGATE_SK:-}"
```

Do not print or interpolate `AGATE_AK` or `AGATE_SK` into the launch command. They are inherited by
the Supervisor process. Construct the mode arguments from the explicit selection:

```bash
case "$OPTIMIZATION_MODE" in
  fast|performance)
    MODE_ARGS=(--optimization-mode "$OPTIMIZATION_MODE" --no-specify-dsl)
    ;;
  ultra)
    test -n "$ULTRA_FRAMEWORK"
    MODE_ARGS=(--optimization-mode ultra --framework "$ULTRA_FRAMEWORK" --specify-dsl)
    ;;
  *)
    echo "Select fast, performance, or ultra" >&2
    exit 2
    ;;
esac

case "$AGENT_CLI" in
  claude|codex|qodercli|pi|dsh) ;;
  *) echo "Select claude, codex, qodercli, pi, or dsh" >&2; exit 2 ;;
esac

case "$AGENT_PROFILE" in
  light)
    PROFILE_ARGS=(--profile light)
    ;;
  light-sandbox)
    case "${AGENT_PROC_MODE:-}" in
      private|host) ;;
      *) echo "Select private or host" >&2; exit 2 ;;
    esac
    printf '{"api_version":1,"patches":[{"id":"agent","config":{"sandbox":"bwrap","proc_mode":"%s"}}]}\n' \
      "$AGENT_PROC_MODE" > "$AGENT_PATCH"
    PROFILE_ARGS=(--profile light-sandbox --patch "$AGENT_PATCH")
    ;;
  *)
    echo "Select light or light-sandbox" >&2
    exit 2
    ;;
esac

require_positive_integer() {
  case "$2" in
    ''|*[!0-9]*|0) echo "$1 must be a positive integer" >&2; exit 2 ;;
  esac
}

require_positive_integer MAX_ITERS "$MAX_ITERS"
printf '%s\n' \
  '{"api_version":1,"patches":[{"id":"workflow","config":{"startup":true,"search":{},"tool_modules":[]}}]}' \
  > "$SEARCH_PATCH"
SEARCH_ARGS=(--patch "$SEARCH_PATCH")
RUN_LENGTH_ARGS=(--max-iters "$MAX_ITERS")
REFERENCE_ARGS=()
if ((${#AGENT_REFERENCE_DIRS[@]})); then
  for directory in "${AGENT_REFERENCE_DIRS[@]}"; do
    REFERENCE_ARGS+=(--agent-reference-paths "$directory")
  done
fi

aka optimize --repo-root "$AKA_REPO" "${PROFILE_ARGS[@]}" "${SEARCH_ARGS[@]}" \
  --problem "$ATREX_BENCH_ROOT/data/native-atrex-bench-vector-add" \
  --platform "$TARGET_PLATFORM" \
  --agent-cli "$AGENT_CLI" \
  "${MODE_ARGS[@]}" \
  "${REFERENCE_ARGS[@]}" \
  "${RUN_LENGTH_ARGS[@]}" \
  --workspace-base "$CAMPAIGN_ROOT"
```

Treat that block as the validated launch argv, not as a requirement to keep an interactive terminal
open. Run it as a durable background task by default and save its launcher log and process identity.
Inside Aone Sandbox, submit it as one durable Sandbox command and retain the returned command ID;
do not append shell `&`. Use a foreground invocation only when the user explicitly requests one.

After submission, use the `atrex-aka-light-observability` skill to inspect the Campaign. Begin with
`light_lineage_status.py /path/to/created/kernel_opt_campaign --json`; use the other observability views for Kernel
history/best, Runtime calls, Agate calls, Gateway-to-Job tracing, or evidence export instead of
recursively scanning the workspace or relying only on the launcher log.

If the Agate worker token differs from `TARGET_PLATFORM`, add `--hardware WORKER_TOKEN`. Use the
installed revision's `aka optimize --repo-root "$AKA_REPO" --help` before launch.
The command explicitly runs one Campaign repeat with one independent trajectory, disables
Direction and Experiment annotations, and uses the run length selected by the user; it does not use Light's pooled schedule
or a preselected smoke-test iteration count. `MAX_ITERS` counts Episode audit segments rather than
Kernel versions. Advanced Research annotations and multi-trajectory search are separate opt-in
configurations, not choices shown during this ordinary launch. A Performance/Ultra run with
`--no-expose-evaluator-shapes` may generate `agent_problem.json` before setup.
