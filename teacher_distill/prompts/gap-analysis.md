# TEACHER_GAP_ANALYSIS

Compare the read-only `teacher/` and `candidate/` implementations after the optimization campaign.

Write exactly:

- `teacher_gap_analysis.md` — structural differences and possible remaining performance explanations.
- `teacher_gap_analysis.json` — schema version 1, `status: "hypothesis"`,
  `promotion_eligible: false`, and a `findings` list whose entries all have
  `status: "hypothesis"`.

Rules:

- Do not copy Teacher source code or long source fragments into either output.
- Do not claim a difference is causal or verified merely because the Teacher uses it.
- Do not modify `teacher/`, `candidate/`, or any input artifact.
- Do not access the public web, upstream repositories, or files outside this workspace.
