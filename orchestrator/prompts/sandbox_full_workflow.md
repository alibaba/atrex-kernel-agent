## Full-mode sandbox workflow

- Run every correctness or performance test through the sandbox's `run` interface. Gateway transports use the typed evaluator when available; OpenSSH uses the mandatory Bubblewrap-isolated command runner. Always pass `--kind run --no-memory`; read the emitted `[test_kernel] RESULT_JSON=...` line, then update `memory/v<N>.json` locally.
  For the V0 baseline, run exactly one full-workload base-seed measurement:
  ```bash
  python tools/sandbox.py --kind run --no-sync -- python test_kernel.py --version v0 --no-memory
  ```
  Use that run's performance result and accompanying correctness status to create `memory/v0.json`. Do not pass `--multi-seed` and do not launch a separate robustness run for V0.
  For every optimized version after V0, run the base-seed measurement and the five-seed correctness gate:
  ```bash
  python tools/sandbox.py --kind run --no-sync -- python test_kernel.py --version v<N> --no-memory
  python tools/sandbox.py --kind run --no-sync -- python test_kernel.py --version v<N> --multi-seed 5 --no-memory
  ```
  The harness must benchmark only the base seed. The additional `--multi-seed` run is correctness-only (no warmup/timing/reference benchmark repetition).
  Native Atrex-Bench run requests with large shape sets are tested automatically in concurrent four-shape batches and merged into one result.
- Run NVIDIA/AMD profiling through the `profile` kind. Gateways use the typed profile interface when available; OpenSSH executes the supplied wrapper and synchronizes its requested artifacts:
  ```bash
  python tools/sandbox.py --kind profile --profile-level sol --sync profiles/v<N> -- bash tools/profile_nvidia.sh profiles/v<N>/harness/profile_driver.py --output-dir profiles/v<N>
  python tools/sandbox.py --kind profile --profile-level sol --sync profiles/v<N> -- bash tools/profile_kernel.sh profiles/v<N>/harness/profile_driver.py --output-dir profiles/v<N>
  ```
  Use `--profile-level deep --kernel-regex '^<exact kernel name>$'` for a focused typed profile. If source-line correlation or another typed-profile gap is specifically required, the sandbox may use the supplied wrapper through `dev`; do not choose dev merely for convenience.
- Any import/API probe/benchmark that may initialize GPU code must be routed through `tools/sandbox.py`; static source inspection remains allowed on the host.
