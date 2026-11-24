## Wiki attribution contract

Query Wiki only when a missing hardware or implementation fact can change the next engineering decision. Use the actual architecture returned by `python3 tools/sandbox.py --kind env`, the exact operator, and the current DSL. Replace the architecture placeholder below with the Env result:

```bash
python3 tools/sandbox.py --kind wiki-query "Runtime architecture <architecture from Env>, DSL {{FRAMEWORK}}. Optimize operator {{OPERATOR}} and retrieve applicable techniques, hardware facts, and pitfalls."
```

Follow `.agents/skills/kernel-wiki/SKILL.md` for result interpretation and optional Experiment attribution. Request a product specification only when the public product identity is known and relevant.
