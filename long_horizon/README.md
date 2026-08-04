# Long-horizon supervisor

This package is a long-horizon entry point layered directly on the current main orchestrator. It
deliberately leaves `orchestrator/optimize.py`, its prompts, `tools/sandbox.py`, reference files, and
the memory manager unchanged.

```bash
python -m long_horizon \
  --op-dir /path/to/operator \
  --platform B200 \
  --sandbox-hardware REMOTE_GPU \
  --framework CuteDSL \
  --max-iters 8 \
  --iter-timeout 18000
```

`python -m long_horizon` delegates argument parsing and orchestration to
`orchestrator.optimize.main()`. Consequently it inherits main's operator resolution, framework
auto-dispatch, V0 setup, framework baseline, production policy, workload bucketing and aggregation,
layer decomposition and ROI scheduling, workspace naming, sandbox selection, and final packaging.
Run `python -m long_horizon --help` for the exact current-main CLI plus four long-horizon options.

Only the optimization-round mechanism changes. Each main iteration becomes one long coding-agent
episode in an isolated Git worktree. The episode uses the current main iteration playbook but may run
many related profile/research/edit/validate/benchmark cycles and private checkpoint commits before
publishing `candidate_ready`, `pivot`, or `blocked`. Claude can resume the same session to repair an
incomplete handoff; Qoder and Codex run a single long invocation because current main does not expose
a persistent resume seam for them.

The incumbent worktree is untouched during exploration. A candidate is promoted only after an exact
same-allocation ABBA schedule passes correctness and beats the incumbent; promotion is a single squash
commit with canonical `memory/vN.json` evidence. A rejected/pivoted/blocked episode commits only its
failed `memory/vN.json` record, preserving main's version budgets, bucket v10 aggregation barrier, and
layer plateau accounting without changing the incumbent kernel.

Runtime state is stored below `.atrex_long_horizon/` in the generated campaign workspace and excluded
through `.git/info/exclude`. Verification payloads live temporarily below
`aggregate_kernels/.atrex_long_horizon_verify/`, which lets the current sandbox's evaluator payload
route carry the ABBA driver without changing `tools/sandbox.py`. The verifier invokes main's
`_sandbox_command`, so endpoint/profile, sync, queue-wait, and timeout behavior stay aligned with main.
