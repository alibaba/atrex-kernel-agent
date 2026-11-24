## Wiki attribution contract

Query Wiki only when a missing hardware or implementation fact can change the next engineering decision. Use the actual architecture returned by `python3 tools/sandbox.py --kind env`, the exact operator, and the current DSL. Replace the architecture placeholder below with the Env result:

```bash
python3 tools/sandbox.py --kind wiki-query "Runtime architecture <architecture from Env>, DSL {{FRAMEWORK}}. Optimize operator {{OPERATOR}} and retrieve applicable techniques, hardware facts, and pitfalls."
```

For related implementations, name the mechanism and missing fact in `wiki-query`; inspect the returned `source_matches` paths and line numbers instead of starting with a manual reference-directory scan. Source hits are unverified references.

Follow `.agents/skills/kernel-wiki/SKILL.md` for result interpretation and optional Experiment attribution. Request a product specification only when the public product identity is known and relevant.
