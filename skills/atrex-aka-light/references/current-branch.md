# Current module branch entrypoints

Applies to the 0.2.0 module distributions integrating upstream main `9b369fcd`.
Build matching wheels with `scripts/install_light.sh` from this Skill. The
profile names `light`, `light-sandbox`, `workflow`, `observed`, and `sandbox`
are retained. Inspect `aka workflow --help` and installed module configuration
before choosing non-default values.

Launch a new Light campaign without the old optimizer argument separator:

```sh
aka workflow --repo-root /path/to/atrex-kernel-agent \
  --profile light-sandbox --problem /path/to/operator \
  --workspace-base /path/to/campaigns --platform L20D \
  --framework CuteDSL --agent-cli claude --optimization-mode production
```

`aka optimize` remains an alias for this public request CLI. `--problem` replaces
the older `--op-dir`; `--workspace-base` replaces the old workspace base flag.
WorkflowSettings and CampaignRuntimeSettings are exposed as native flags or
`--settings`/`--runtime` JSON objects. A frozen continuation uses its recorded
`--workflow-request`; changing its profile, patches or request fields is rejected.

DSH is selected by `--agent-cli dsh`. It uses the configured headless profile,
private session Home, JSON events and checkpoints. Provider authentication is
configured by the operator. Native child-attempt usage is counted without
recounting seeded or resumed history. Linux Bubblewrap/DSH execution requires
the matching DSH installation and remains a separate real-runtime check.

The Agent's `tools/sandbox.py` exposes the installed safe Agate options. A
request with unknown submission or explicit replay prohibition is recovered by
its original Job identity; it must not be blindly submitted again.

Shapes are exposed by default for new requests. Old serialized requests without
the visibility field preserve historical hidden-shape behavior. New source-tree
candidates have editable `kernel.py` and optional `impl/`; historical immutable
adapters remain immutable under their pinned source policy.

Observability commands remain in the checkout's `observability/` directory.
They read current shared Experiment records and public Agent session facts,
and retain read-only support for upstream v1 records. They never start GPU work.
Agent transport uses the managed Unix socket. The HTTP transport descriptions
and private module paths in the inherited main reference pages are historical.
