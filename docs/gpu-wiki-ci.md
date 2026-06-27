# GPU Wiki CI Gate

`gpu-wiki/` is source knowledge. It should stay independent from Atrex Server,
CLI, RayJob, OSS, task runtime paths, and local agent workflows.

## Blocking Gate

Use this command as the blocking AoneCI job for CRs that touch `gpu-wiki/`:

```bash
bash scripts/ci/check-gpu-wiki.sh
```

The gate runs:

- `python3 gpu-wiki/scripts/check-self-contained.py --root gpu-wiki`
- `python3 -m unittest gpu-wiki/scripts/test_check_self_contained.py -v`
- `git diff --check`
- `openspec validate wikify-gpu-wiki-knowledge --strict`, when `openspec` is available

The script skips itself when the CR does not change `gpu-wiki/`.

## AoneCI Agentic Review

The AoneCI Agentic review prompt lives at:

`/.aoneci/agentic/cr-comment-fix-suggestion.md`

Recommended policy:

- Keep `gpu-wiki-self-contained` blocking.
- Keep the Agentic review advisory until its findings are stable.
- Promote repeated LLM findings into deterministic checker rules and tests.

The Agentic review should flag semantic boundary leaks: runtime/task protocol
language, mandatory download steps, backend-boundary pollution, Markdown
navigability regressions, and skill-package or agent-workflow language in source
wiki content.
