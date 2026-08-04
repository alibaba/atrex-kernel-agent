# Long-horizon supervisor

This package is a standalone optimization entry point. It deliberately leaves the existing
`orchestrator/optimize.py`, prompts, sandbox wrapper, and memory manager unchanged.

```bash
python -m long_horizon \
  --op-dir /path/to/operator \
  --platform B200 \
  --sandbox-hardware REMOTE_GPU \
  --framework CuteDSL \
  --max-episodes 8
```

Each episode runs one persistent Claude session in an isolated Git worktree. The session may perform
many experiments and checkpoint commits before publishing `candidate_ready`, `pivot`, or `blocked`.
The incumbent worktree is untouched during exploration. A candidate is promoted only after an exact
same-allocation ABBA schedule passes correctness and beats the incumbent; promotion is a single squash
commit with canonical `memory/vN.json` evidence.

Runtime state is stored below `.atrex_long_horizon/` in the generated campaign workspace and excluded
through `.git/info/exclude`. Verification payloads live temporarily below
`aggregate_kernels/.atrex_long_horizon_verify/`, which lets the current sandbox's evaluator payload
route carry the ABBA driver without changing `tools/sandbox.py`.

The first version intentionally supports one explicit framework and Claude. Workload bucketing, layer
decomposition, Qoder session persistence, and Codex resume are not silently emulated; use the existing
clean-session orchestrator for those modes until dedicated adapters are implemented.
