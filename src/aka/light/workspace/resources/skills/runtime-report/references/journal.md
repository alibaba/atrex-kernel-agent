# Exact Kernel commit review

Direction and Experiment writes are disabled. GPU measurements are still saved automatically.

Commit the complete SourceProvider candidate, then write:

```json
{"kernel_commit_id":"0123456789abcdef0123456789abcdef01234567"}
```

Submit it with:

```bash
python3 tools/sandbox.py --kind submit-kernel --request-file scratch/kernel-submit.json
```

The commit must be reachable from current Agent or imported peer Git refs, contain a complete
candidate source tree, and already have a passing standard full Evaluate for that exact commit.
The receipt contains only checks applicable to the selected policy: Fast audits declared
dependencies, external loading and manifest consistency; Performance additionally requires
self-authored core compute; Ultra additionally enforces the configured DSL. Inapplicable checks
are omitted rather than reported as `not_applicable`. An accepted commit enters winner and Pareto
selection.

A rejected review returns actionable diagnostics and leaves Git unchanged. Request files are
limited to 2,093,056 bytes and the encoded HTTP body must fit within 2 MiB.
