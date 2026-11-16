# FlyDSL 0.3.4 reference

Selected resource bundle: `flydsl-0.3.4`.
Agent workspace location: `reference/dsl/flydsl/0.3.4/`.

Before choosing FlyDSL imports or repairing a candidate, obtain the actual per-Job dependency
version from the selected runtime's environment query (`task_worker_requirements`).
For 0.3.4, read [the programming guide](guide.md), starting with its installation scope and
the section for the failing interface. Legacy and Light use this same reference even when
Wiki is disabled. For another version, use the corresponding installed or pinned source reference.

The guide is based on ROCm/FlyDSL v0.3.4, commit
`6bb37ba4e23c3a2729e3af244251b55fc5938237`, and the public CPython 3.12 Linux wheel
with SHA-256 `280d9fb8830c0ccfb65e1cad3a6bb57f8b9c8a74b5352dafeebd3b4041425ed1`.
Existence in the upstream repository does not imply inclusion in that wheel.
Imports, first-launch JIT and numerical/performance experiments run through the selected
sandbox; a documentation example is not proof that the current Task passed.

This directory is runtime reference content. Campaign receipts and project audit records
are maintained separately from this installed resource.
