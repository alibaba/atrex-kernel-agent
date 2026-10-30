# Episode Report with research recording disabled

Direction and research Experiment writes are disabled for this Episode. GPU measurement records
are still saved automatically. Do not call legacy Journal append/finalize or write control handoffs.
Do not manage Git or supply a Commit ID: the Supervisor seals the validated candidate.

## Candidate

Run the ordinary evaluator prescribed by the current phase:

```bash
python3 tools/sandbox.py --kind run --no-sync
```

Inspect the result using `record-read` if necessary. Leave the exact measured source in place:
`kernel.py` for single-file tasks, or the contracted source tree described by `WORKSPACE.md`.
Save `scratch/episode-report.json`, replacing the example ID with the returned `gateway_record_id`:

```json
{
  "status": "candidate_ready",
  "summary": "The candidate passes full evaluation and is ready for independent verification",
  "gateway_record_ids": ["gateway-11111111111111111111111111111111"]
}
```

```bash
python3 tools/sandbox.py --kind episode-report --request-file scratch/episode-report.json
```

The list must be nonempty and contain unique, real Kernel-bound Gateway Record IDs. At least one
must be a completed, passing standard full Evaluate for the byte-identical current candidate.
Matching historical results are allowed. Profile, Dev, ABBA, and correctness-only checks alone
cannot qualify the candidate. Do not supply `selected_experiment_id`.

The Supervisor performs or reuses its mandatory acceptance checks before committing. In production
this includes the configured numerical validation. Acceptance still precedes independent ABBA
verification and does not guarantee promotion. Success returns `status: "accepted"`.

## Pivot or blocker

An exhausted approach can finish with:

```json
{"status": "pivot", "summary": "The tested approach did not provide a useful improvement"}
```

An infrastructure or authority blocker additionally requires a nonempty `blocker`:

```json
{"status": "blocked", "summary": "Cannot validate the candidate", "blocker": "GPU service unavailable"}
```

Neither state accepts `gateway_record_ids`. Both allow the existing optional
`accepted_ppu_diagnostics` field when the PPU Skill requires it; referenced files must be in `scratch/`.

## Repair and replay

For an invalid report, follow `error.next_action`: correct its fields or evidence, repair and measure
the Kernel if necessary, then resubmit. Rejection leaves the Episode open. If publication was
interrupted after acceptance, repeat the identical report; this does not repeat acceptance or create
another Commit. Never blindly repeat a GPU request with an unknown outcome.

Request files are limited to 2,093,056 bytes (2 MiB minus 4 KiB); the encoded HTTP body must fit
within 2 MiB. Shorten text or lists before retrying a size rejection.
