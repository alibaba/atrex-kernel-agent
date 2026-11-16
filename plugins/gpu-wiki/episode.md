## Wiki attribution contract

Query Wiki only when a missing hardware or implementation fact can change the next engineering decision. Use the exact operator, true product, authoritative runtime architecture, and current DSL:

```bash
python3 tools/sandbox.py --kind wiki-query "Target product {{PLATFORM}}, runtime architecture {{ARCH}}, DSL {{FRAMEWORK}}. Optimize operator {{OPERATOR}} and retrieve applicable techniques, hardware facts, and pitfalls."
```

Follow `skills/KernelWiki/SKILL.md` for result interpretation and optional Experiment attribution. When conversion is mandatory, ask only for the exact product specification and matching conversion guidance; do not issue an additional generic query without a separate measured uncertainty.
