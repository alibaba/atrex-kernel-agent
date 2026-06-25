# Real-time Learning Guide

This document supplements the learning methods in SKILL.md with detailed step-by-step procedures.

---

## Process 1: Querying Unknown APIs

```1. Confirm the problem: "What does tl.xxx correspond to in Gluon?"
2. Check references/porting_rules.md (complete API reference, including Gluon-specific APIs)
3. Check Gluon official source code:
   - https://github.com/triton-lang/triton/tree/main/python/triton/experimental/gluon
4. Experimental verification: Write minimal test code and compile to see if it passes
5. Record to api_mapping.md and mark verification status```

## Process 2: Resolving Compilation Errors

```
1. Read error message → Locate problematic code line and used API
2. Check references/common_pitfalls.md (common errors and solutions)
3. Search GitHub Issues: https://github.com/triton-lang/triton/issues (keywords: gluon + error message)
4. Apply fix → Recompile
5. Record new errors/solutions to common_pitfalls.md
```

## Process 3: Verifying Uncertain Knowledge

```
1. Propose hypothesis: "tl.xxx may correspond to gl.yyy"
2. Design comparative test: Write Triton version and Gluon version
3. Run test: Compare functional correctness and output consistency
4. Draw conclusion: ✅ Hypothesis valid / ❌ Hypothesis invalid
5. api_mapping.md, verification
```

---

## Learning Resources

### 1. Official Documentation (Highest Priority)

- GitHub: https://github.com/triton-lang/triton/tree/main/python/triton/experimental/gluon
- API Definitions: Check function definitions in `language.py`

### 2. Internal References

- `references/porting_rules.md` — Complete API reference + conversion rules
- `references/common_pitfalls.md` — Known errors and solutions
- `examples/` — Verified conversion examples

### 3. Community Resources (Supplementary Reference)

- Triton Issues: https://github.com/triton-lang/triton/issues (Search: `gluon`, `AMD`, `ROCm`)

---

## Knowledge Update Guidelines

Record after each successful conversion:

```markdown
| date | Triton | Gluon | verification | source |
|------|--------|-------|---------|------|
| 2026-03-03 | `tl.xxx` | `gl.yyy` | ✅ verification | official documentation + experiment |
```

Verification Status:
- ✅ **Verified**: Passed functional tests + compilation tests
- ⚠️ **Pending Verification**: Inferred from documentation only
- ❌ **Falsified**: Experiment failed, mapping is incorrect
