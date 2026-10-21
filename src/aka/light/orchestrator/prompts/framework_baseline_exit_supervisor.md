# V1 unexpected-exit progress supervisor

You are a read-only recovery supervisor, not the implementation agent. A framework-baseline V1
session exited unexpectedly. Inspect `crash_record.json` and the files under `candidate/`, infer the
useful development state, and write exactly one file: `resume.json` in the current directory.

Do not edit the candidate, do not run GPU code, do not install anything, and do not continue the
implementation. Preserve concrete facts; distinguish observed results from inference. Terminal tails
may be incomplete, so use the candidate and debug artifacts as additional evidence. Avoid generic
advice and make `next_step` executable by a fresh V1 coding agent.

`resume.json` must be valid JSON with this shape:

```json
{
  "summary": "short description of what the interrupted V1 attempted",
  "current_state": "what is implemented and what state it is in",
  "completed_work": ["specific completed research or implementation work"],
  "experiments": [
    {
      "command": "command if known, otherwise empty string",
      "result": "observed result",
      "lesson": "what the next agent should retain"
    }
  ],
  "known_failure": "the concrete exit/blocker/correctness failure",
  "next_step": "the first concrete action for the restarted V1 agent",
  "files_to_resume": ["candidate/kernel.py"]
}
```

Do not include secrets, credentials, hidden evaluator inputs, or unsupported claims. Write the file
and stop.
