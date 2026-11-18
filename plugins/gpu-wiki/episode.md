## Wiki attribution contract

Query Wiki only when a missing hardware or implementation fact can change the next engineering decision. Use the exact operator, target hardware and, if known, public product, authoritative runtime architecture, and current DSL:

```bash
python3 tools/sandbox.py --kind wiki-query "Target hardware {{PLATFORM}}, runtime architecture {{ARCH}}, DSL {{FRAMEWORK}}. Optimize operator {{OPERATOR}} and retrieve applicable techniques, hardware facts, and pitfalls."
```

Follow `.agents/skills/kernel-wiki/SKILL.md` for result interpretation and optional Experiment attribution. Deployment names such as L20N/L20D are not public product identities; use the runtime architecture for architecture facts and device measurements for product-specific numbers.
