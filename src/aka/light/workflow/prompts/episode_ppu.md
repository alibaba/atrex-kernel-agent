## Reusable PPU diagnostics

When memory summaries are enabled, prior conclusions are in
`memory/v*.json.profile_evidence.accepted_ppu_diagnostics`; otherwise consult relevant saved
Profile records and historical Conversations. Do not assume summaries exist. Check their
specialization, workload, device, launch topology, pipeline identity, and invalidation conditions
before reuse. Follow `skills/ppu-acu-joint-profile/SKILL.md` for optional `accepted_ppu_diagnostics`
in `episode-report`: include only evidence applicable to the terminal probe-free Kernel, bound to
accepted decision-grade artifacts by path, SHA-256, schema, and evidence ID. The Supervisor validates
these bindings. Keep the artifacts and all transitive evidence under `scratch/`, synchronize remote
outputs there before reporting, and use workspace-relative `scratch/...` artifact paths. Leave cited
files intact for terminal revalidation. Omit the field or use an empty list when no reusable evidence exists.
