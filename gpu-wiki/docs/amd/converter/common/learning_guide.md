# Real-time Learning Guide

This document supplements the learning methods in conversion-guide.md with detailed step-by-step procedures.


**Last updated**: 2026-06-30

---

## Process 1: Querying Unknown APIs

```
1. Confirm the problem: "What does tl.xxx correspond to in Gluon?"
2. Check `porting_rules.md` (complete API reference, including Gluon-specific APIs)
3. Check Gluon official source code:
   - https://github.com/triton-lang/triton/tree/main/python/triton/experimental/gluon
4. Experimental verification: Write minimal test code and compile to see if it passes
5. Record the finding in the relevant architecture-specific API mapping notes and mark verification status
```

## Process 2: Resolving Compilation Errors

```
1. Read error message → Locate problematic code line and used API
2. Check the architecture-specific `common_pitfalls.md` page for the target backend (for example, `../cdna3/common_pitfalls.md` or `../cdna4/common_pitfalls.md`)
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
5. Update the relevant architecture-specific API mapping notes with the verification result
```

---

## Learning Resources

### 1. Official Documentation (Highest Priority)

- GitHub: https://github.com/triton-lang/triton/tree/main/python/triton/experimental/gluon
- API Definitions: Check function definitions in `language.py`

### 2. Internal References

- `porting_rules.md` — Complete API reference + conversion rules
- Architecture-specific `common_pitfalls.md` pages — Known errors and solutions
- Architecture-specific conversion examples from the relevant backend guide or reference material

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


## Related

- [Triton → Gluon Conversion Guide (NVIDIA Hopper)](hopper-conversion-guide.md)
- [HSACO Offline Launcher Guide](hsaco-offline-launcher.md)
- [General Gluon Operations Porting Rules (Applicability of each section is annotated: [Common] = General, [AMD CDNA3] = AMD-specific, [Hopper] = NVIDIA Hopper-specific)](porting_rules.md)
- [Verification Guide](verification_guide.md)
- [Triton Embraces Tile IR: Beyond SIMT](../../../nvidia/common/triton/triton-tile-ir-beyond-simt.md)
- [Composable Kernel (CK) Architecture Overview](../../common/ck-architecture-overview.md)
- [Gluon Tutorial 07: Persistent Kernels and Pipeline Optimization](../../../nvidia/common/gluon/gluon-07-persistent-kernel-pipeline.md)
- [Document Relationship Diagram](../../../RELATIONS.md)
