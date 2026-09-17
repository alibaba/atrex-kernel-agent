---
name: gpu-measurement
description: Run Kernel evaluation, ABBA comparison, profiling and GPU probes through the session's Supervisor HTTP Runtime.
---

# GPU measurement

Use `python3 tools/sandbox.py`. The Supervisor chooses the Gateway/SSH endpoint,
hardware, credentials and timeout, snapshots the current workspace, and supplies
the evaluator and private inputs. Do not set endpoint/credential flags or run GPU
code locally. Every remote job starts fresh; no remote filesystem is retained.

This skill changes transport, not the Episode workflow: follow the current
Setup/Framework Baseline/Fast/Full instructions, including their measurement,
Journal, plan, profile and Git requirements. An Agent measurement or comparison
does not grant promotion authority.

Read [requests.md](references/requests.md) for request examples. Use
`--kind OPERATION --help` for full CLI parameters. Evaluate keeps
`[test_kernel] RESULT_JSON=`; Profile returns `[sandbox] PROFILE_JSON=`.
Numbers in those results are measured facts; interpretation is your responsibility.

Missing/revoked capability or transport failure is an infrastructure blocker.
Do not bypass the Runtime or blindly retry an operation whose outcome is unknown.
Gateway retry policy is owned by the Supervisor; do not add client-side retries.
