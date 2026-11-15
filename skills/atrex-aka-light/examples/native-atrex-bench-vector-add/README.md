# Minimal Native Atrex-Bench input

This directory shows the smallest useful Native Atrex-Bench problem accepted by AKA
Light:

- `reference.py` defines the authoritative `torch.nn.Module` semantics;
- `input.py` defines `_make_inputs`, which turns one Shape's `input_kwargs` into CUDA
  tensors passed to `Model.forward`;
- `shapes.json` maps stable Shape IDs to `Model.__init__` and `_make_inputs` arguments.

The example intentionally omits optional `metadata.json`, `roofline.json`, `valid.py`,
and `agent_problem.json`.

## Required launch inputs

Before preparing or launching this example, explicitly obtain from the user:

1. a safe injection source for Agate `AGATE_AK` and `AGATE_SK` (the values must already be present
   in the Supervisor environment; never ask the user to paste them into chat);
2. the canonical Atrex-Bench checkout root;
3. the target GPU/platform token, such as `L20N` or `L20D`.
4. the optimization mode after explaining that Fast permits declared external core implementations,
   Performance requires self-authored core computation with an Agent-selected DSL, and Ultra also
   requires a user-pinned DSL.
5. whether to use the Bubblewrap Agent Sandbox (`light-sandbox`) or run the Agent natively without
   Bubblewrap (`light`). Clarify that this does not create or select an Aone Sandbox. When Sandbox
   is selected, also ask for `private` (private PID namespace/procfs) or `host` (shared outer PID
   namespace and `/proc`). Aone Sandbox supports only `host`.

The commands below use these placeholders:

```bash
AKA_REPO=/path/to/atrex-kernel-agent
ATREX_BENCH_ROOT=/path/to/atrex-bench
CAMPAIGN_ROOT=/path/to/vector-add-campaign
AGENT_PATCH=/path/to/agent-boundary.json

: "${TARGET_PLATFORM:?set the user-selected GPU token}"
: "${OPTIMIZATION_MODE:?set fast, performance, or ultra from the user selection}"
: "${AGENT_PROFILE:?set light or light-sandbox from the user selection}"
# Also require AGENT_PROC_MODE=private|host when AGENT_PROFILE=light-sandbox.
# Also require ULTRA_FRAMEWORK when OPTIMIZATION_MODE=ultra.
```

Set every required environment variable from the user's explicit selection; this example does not
choose a GPU, optimization mode or Agent boundary. For Ultra, require `ULTRA_FRAMEWORK`; do not
launch Ultra without the user's DSL choice. `light-sandbox` requires Linux/Bubblewrap; on native macOS the user must choose
`light` or run the sandboxed profile inside Lima. The command below materializes the selected proc
mode in a complete Agent composition patch. Inside Aone Sandbox, `host` must still be accepted by
the user; never launch `private` there. Proc mode is not applicable when
`AGENT_PROFILE=light`. See [installation](../../references/installation.md) for the boundary.

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
    ├── reference.py
    ├── input.py
    └── shapes.json
```

Before launch, verify structure and credential presence without printing secrets:

```bash
test -f "$ATREX_BENCH_ROOT/scripts/run_eval.py"
test -d "$ATREX_BENCH_ROOT/src/atrex_bench"
test -f "$ATREX_BENCH_ROOT/data/native-atrex-bench-vector-add/reference.py"
test -f "$ATREX_BENCH_ROOT/data/native-atrex-bench-vector-add/input.py"
test -s "$ATREX_BENCH_ROOT/data/native-atrex-bench-vector-add/shapes.json"
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

case "$AGENT_PROFILE" in
  light)
    PROFILE_ARGS=(--profile light)
    ;;
  light-sandbox)
    case "${AGENT_PROC_MODE:-}" in
      private|host) ;;
      *) echo "Select private or host" >&2; exit 2 ;;
    esac
    printf '{"api_version":1,"patches":[{"id":"agent","config":{"sandbox":"bwrap","capture":true,"proc_mode":"%s"}}]}\n' \
      "$AGENT_PROC_MODE" > "$AGENT_PATCH"
    PROFILE_ARGS=(--profile light-sandbox --patch "$AGENT_PATCH")
    ;;
  *)
    echo "Select light or light-sandbox" >&2
    exit 2
    ;;
esac

aka optimize --repo-root "$AKA_REPO" "${PROFILE_ARGS[@]}" -- \
  --op-dir "$ATREX_BENCH_ROOT/data/native-atrex-bench-vector-add" \
  --platform "$TARGET_PLATFORM" \
  --agent-cli claude \
  "${MODE_ARGS[@]}" \
  --workspace "$CAMPAIGN_ROOT"
```

If the Agate worker token differs from `TARGET_PLATFORM`, add `--sandbox-hardware WORKER_TOKEN`. Use the
installed revision's `aka optimize --repo-root "$AKA_REPO" -- --help` before launch.
The command uses the default pooled three-trajectory search and exposed-Shape flow.
To run a bounded serial functional test, pass a launch-level composition patch with
`workflow.config.search` set to an empty mapping, then use `--max-iters 5`; in current
Light that value counts Episode audit segments rather than Kernel versions. A Performance/Ultra run with
`--no-expose-evaluator-shapes` may generate `agent_problem.json` before setup.
