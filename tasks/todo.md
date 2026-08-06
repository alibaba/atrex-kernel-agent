# Hidden-Teacher Distillation Campaign — Task Checklist

Implementation must follow TDD: add a failing behavior test before each production change.

## Phase 1 — Contracts and isolation artifacts

- [x] **Task 1:** Define Teacher campaign schemas and immutable domain models.
  - Verified: `python3 -m unittest teacher_distill.tests.test_models -v`
- [x] **Task 2:** Validate and fingerprint self-contained Teacher bundles.
  - Depends on: Task 1
  - Verified: `python3.12 -m unittest teacher_distill.tests.test_bundle -v`
- [x] **Task 3:** Build a physical, content-addressed sanitized gpu-wiki view.
  - Depends on: Tasks 1–2
  - Verified: `python3.12 -m unittest teacher_distill.tests.test_knowledge_view -v`

### Checkpoint A

- [x] Tasks 1–3 tests pass.
- [x] Valid/invalid bundle fixtures behave fail-closed.
- [x] Sanitized view is deterministic, queryable, self-contained, and contains no original-wiki symlinks.
- [x] Reviewed the generated inclusion/exclusion report on the repository gpu-wiki.

## Phase 2 — Backward-compatible orchestration seams

- [x] **Task 4:** Add optional `teacher_progress` memory/schema/summary support.
  - Depends on: Task 1
  - Verified: `python3.12 -m unittest tests.test_teacher_memory -v`
- [x] **Task 5:** Add `StopPolicy`/`DefaultStopPolicy` without changing standard campaigns.
  - Depends on: Task 1
  - Verified: `python3.12 -m unittest tests.test_stop_policy tests.test_framework_baseline long_horizon.tests.test_main_adapter -v`
- [x] **Task 6:** Add Teacher-mode CLI validation and lazy dispatch.
  - Depends on: Tasks 1–2, 5
  - Verified: `python3.12 -m unittest tests.test_teacher_cli` plus standard dispatch characterization cases.

### Checkpoint B

- [x] Tasks 4–6 focused tests pass.
- [ ] Existing `tests/` suite is green (blocked by pre-existing `test_production_explicit_framework_disables_conversion` and local environment dependencies).
- [x] Standard CLI defaults and auto-dispatch behavior are unchanged in characterization tests.
- [x] Unsupported Teacher-mode combinations fail before workspace/GPU/Agent work.

## Phase 3 — Hidden-audited execution and Teacher measurement

- [x] **Task 7:** Enforce sanitized runtime links, search restrictions, forbidden-access audit, and no-public-web policy.
  - Depends on: Tasks 3, 6
  - Verified: `python3.12 -m unittest tests.test_teacher_session_policy tests.test_agent_runtime_interface tests.test_agent_runtime_characterization tests.test_campaign_runtime_binding -v`
- [x] **Task 8:** Materialize and validate the private Teacher benchmark workspace.
  - Depends on: Tasks 1–2, 6
  - Verified: `python3.12 -m unittest teacher_distill.tests.test_bundle teacher_distill.tests.test_teacher_benchmark -v`
- [x] **Task 9:** Generalize same-allocation ABBA for private Teacher vs Git Candidate.
  - Depends on: Task 8
  - Verified: `python3.12 -m unittest teacher_distill.tests.test_abba long_horizon.tests.test_verifier -v`
- [x] **Task 10:** Implement `TeacherStopPolicy` and Teacher-progress recording.
  - Depends on: Tasks 4–5, 9
  - Verified: `python3.12 -m unittest teacher_distill.tests.test_stop_policy -v`

### Checkpoint C

- [x] Tasks 7–10 pass with fake sandbox responses.
- [x] All four Agent backends receive the same opaque Teacher policy environment.
- [x] Teacher source/path never appears in public workspace or prompts in policy tests.
- [x] Mocked provisional PASS → ABBA FAIL/INFRA → continue and ABBA PASS → stop behavior works.

## Phase 4 — Complete Teacher campaign and bounded exploration

- [x] **Task 11:** Implement `TeacherDistillCampaign` setup, resume locks, loop wiring, and terminal statuses.
  - Depends on: Tasks 6–10
  - Verified: `python3.12 -m unittest teacher_distill.tests.test_campaign teacher_distill.tests.test_stop_policy tests.test_campaign_runtime_binding tests.test_framework_baseline -v`
- [x] **Task 12:** Add one bounded long-horizon episode after stalls and one partial restart.
  - Depends on: Task 11
  - Verified: `python3.12 -m unittest teacher_distill.tests.test_escalation long_horizon.tests.test_campaign long_horizon.tests.test_git_episode long_horizon.tests.test_session -v`

### Checkpoint D

- [x] SUCCESS, PLATEAU, BUDGET_EXHAUSTED, INFRA_ERROR, leakage, and resume mismatch are covered.
- [x] Candidate HEAD remains monotonically best; long-horizon promotion stays ABBA-gated.
- [x] Resume restores escalation/restart counters and deterministic next action.
- [x] Existing long-horizon tests remain green.

## Phase 5 — Evidence-backed distillation

- [x] **Task 13:** Build deterministic evidence and performance-trajectory manifests.
  - Depends on: Tasks 11–12
  - Verified: `python3.12 -m unittest teacher_distill.tests.test_evidence -v`
- [x] **Task 14:** Generate hypothesis-only Teacher gap analysis and evidence-cited drafts.
  - Depends on: Task 13
  - Verified: `python3.12 -m unittest teacher_distill.tests.test_distillation teacher_distill.tests.test_campaign -v`
- [x] **Task 15:** Validate drafts and prohibit automatic canonical wiki promotion.
  - Depends on: Task 14
  - Verified: `python3.12 -m unittest teacher_distill.tests.test_distillation teacher_distill.tests.test_draft_validator -v`

## Phase 6 — Documentation and release gate

- [x] **Task 16:** Add docs, CLI examples, minimal fixtures, and mocked end-to-end test.
  - Depends on: Tasks 1–15
- [ ] Run: `python -m unittest discover -s tests -v` — 166 tests ran; the feature tests pass, while upstream/main independently reproduces the two assertion failures (`test_local_gateway`, `test_optimize_dispatch`) and this interpreter lacks Torch for `test_aggregate_dispatch`.
- [x] Run: `python3.12 -m unittest discover -s long_horizon/tests -v` — 57 passed after rebasing onto upstream/main.
- [x] Run: `python3.12 -m unittest discover -s teacher_distill/tests -v` — 94 passed after final isolation and verification hardening.
- [x] Run: `python3.12 -m unittest gpu-wiki/scripts/test_query.py -v` — 48 passed.
- [x] Run: `python3.12 -m unittest gpu-wiki/scripts/test_check_self_contained.py -v` — 47 passed.
- [x] Run: `git diff --check`.

### Checkpoint E — Release readiness

- [x] Standard campaign behavior remains unchanged in focused characterization/regression tests.
- [x] Mocked Teacher campaign passes end to end through the real CLI and supervisor.
- [ ] One internal real-GPU smoke campaign passes the checklist in `tasks/plan.md` — no configured GPU/gateway is available in this environment.
- [x] Campaign execution leaves canonical `gpu-wiki/` untouched by construction and validator tests.
- [x] Threat-model wording clearly says `hidden-audited`, not security-grade isolation.
- [ ] Human review approves the implementation and generated draft format.
