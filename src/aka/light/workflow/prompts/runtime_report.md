## Episode handoff

Leave the selected measured source in place: `kernel.py` for a single-file task,
or the entire contracted source tree for a tree task. Submit `status`, `summary`, and
`gateway_record_ids` for `candidate_ready`; at least one cited record must be a passing standard
full Evaluate for that exact source. Profile, ABBA, and correctness-only checks alone are insufficient.
Do not submit `selected_experiment_id` or manage Git. The Supervisor validates and seals the source.

For `pivot`, submit only `status` and `summary`; for `blocked`, also supply `blocker`.
Chat text is not a handoff. Follow `skills/runtime-records/references/journal.md` and run:

```bash
python3 tools/sandbox.py --kind episode-report --request-file scratch/episode-report.json
```

Correct rejected fields or evidence and resubmit. A rejected report leaves the Episode open;
stop after acceptance. Acceptance checks and promotion policy are unchanged by research recording.
