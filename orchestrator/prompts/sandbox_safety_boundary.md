## Execution boundary

- Target remote hardware: **{{HARDWARE}}**{{ENDPOINT}}. All GPU execution goes through
  `python3 tools/sandbox.py`. Do not bypass the scoped Runtime or obtain Agate credentials.
  Local static inspection is allowed; local GPU/JIT imports, evaluators, timers, and profilers are not.
- Typed operations such as `--kind run` and `--kind profile` select their inputs automatically and
  take no command after `--`. For custom Dev commands, use `--kind dev --input <path> ... -- <command>`;
  declare every extra local file/directory with repeatable `--input`. Uploads are allowlist-only.
  A trivial Dev probe does not establish evaluator health, and diagnostics/custom-input checks do
  not replace official correctness/performance evidence. The current phase may further restrict GPU use.
- Git is Supervisor-only: do not run Git commands or access `.git`. Keep evaluator/public-contract
  inputs (`definition.json`, `reference.py`, `workload.jsonl`, `input.py`, `shapes.json`,
  `agent_problem.json`, `metadata.json`, `roofline.json`), `CLAUDE.md`, `README.md`, and canonical
  `memory/v*.json` unchanged. Never search outside the workspace for private cases or source operators,
  recreate private evaluator drivers, or delete/move existing tracked files and historical evidence
  to shrink uploads. Do not upload `memory/` as worker state. The Supervisor records measurements,
  writes canonical memory, commits source, and manages acceptance.
- Use preinstalled dependencies only. Do not install packages or build third-party libraries locally
  or remotely (pip, uv, conda, setup.py, ninja, cmake, or package managers). GPU packages may JIT on
  import, so inspect their source locally but run imports/probes only through permitted GPU operations.
  If required tooling is unavailable, choose supported tooling or report a blocker.
- Shared Gateway infrastructure is Supervisor-owned. Do not start, stop, signal, reconfigure, or
  replace its services, screen/SSH sessions, state, database, logs, or jobs. Report infrastructure
  failure; do not repair the service yourself.
