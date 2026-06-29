# GPU Kernel Research Subagent

Evidence-driven knowledge search and plan writing for GPU kernel optimization iterations.

## When to Use

This skill is invoked as a subagent by gpu-kernel-profile-optimizer Stage 2. It performs read-only research — searching knowledge sources, extracting bottleneck-relevant optimization knowledge, and writing the iteration plan. The main agent must not perform evidence search or write the plan directly.

## Input Contract

The calling agent provides these parameters at subagent launch:

| Parameter | Description |
|-----------|-------------|
| `workspace_path` | Workspace absolute path (kernel_opt_<name>/) |
| `version` | Current iteration `V<N>` |
| `platform` | Target platform: nvidia / amd |
| `framework` | DSL/framework: triton / cutedsl / flydsl / gluon |
| `kernel_type` | Kernel type (gemm, attention, moe, norm, etc.) |
| `profiles_dir` | `profiles/v<N>/` path with current profiling artifacts |
| `memory_dir` | `memory/` directory path |
| `historical_plans` | All `plans/v*_plan.md` paths |
| `stop_conditions` | Optimization stop criteria |
| `gpu_wiki_path` | gpu-wiki root path |

## Mandatory Reads

Starting from V1, read before any search:

1. `kernel_opt_<name>/README.md`
2. `<gpu-wiki>/README.md`
3. All unmasked `kernel_opt_<name>/memory/v*.json` files (skip `masked: true`)
4. Current `profiles/v<N>/` artifacts
5. Previous `kernel_opt_<name>/memory/v<N-1>.json` (if unmasked)
6. Historical `plans/v*_plan.md`

## Knowledge Base Search

Translate Stage 1 profiler symptoms into gpu-wiki search keywords using the
**Symptom-Driven Retrieval (NVIDIA vs AMD)** guidance in `<gpu-wiki>/README.md` —
NVIDIA and AMD use different vocabularies and sub-trees, and that vendor-split
mapping is maintained there, not in this skill.

---

## Progressive Search Strategy

The core search behavior follows a three-layer progressive expansion model. Each invocation MUST return new knowledge — if the current layer yields nothing new, escalate to the next layer.

### Historical Search Log Parsing

At the start of each invocation, BEFORE initiating any search:

1. Read all historical `plans/v*_plan.md` files from the `historical_plans` input
2. Parse each file's Search Log table (the `| Source | Layer | Query | Finding | New? | Actionability |` table)
3. Extract every `(Source, Query, Finding)` triple to form the **used knowledge set**
4. This set serves as the deduplication reference for the current invocation

**First iteration (V1)**: No historical plans exist → used knowledge set is empty → all findings are automatically new.

**Parsing rules**:
- Match triples by semantic similarity, not exact string equality — "bank conflict mitigation" and "LDS bank conflict reduction" are the same knowledge
- A finding is "used" if its (Source, Finding) pair substantially overlaps with any historical entry, regardless of query wording differences

### Three-Layer Search Space

| Layer | Scope | Sources | Search Method |
|-------|-------|---------|---------------|
| **L1** (gpu-wiki) | Local curated knowledge | `gpu-wiki/docs/` (kernel-opt, ref-docs, pitfalls, hardware-specs, converter), `gpu-wiki/3rdparty/`, `gpu-wiki/reference-kernels/` | Navigate via README hierarchy; grep by keyword; read targeted files |
| **L2** (reference-projects) | Local code repositories | `reference-projects/` — cloned upstream frameworks and implementations (cutlass, flash-attention, flashinfer, DeepGEMM, triton, etc.) | Search source code for implementation patterns; read specific modules by kernel type |
| **L3** (public net) | Internet resources | Papers, blog posts, vendor official docs, GitHub issues, community forums | Web search by targeted query; findings provide optimization ideas only; hardware specs still require gpu-wiki or explicit confirmation |

**Layer ordering is strict**: always attempt L1 → L2 → L3 in sequence. Never skip a layer.

### Layer Exhaustion Detection

A layer is "exhausted" when it cannot yield new actionable knowledge for the current optimization context:

**Layer 1 (gpu-wiki) exhausted when ALL of**:
- All README-navigable paths relevant to the current (platform, framework, architecture) scope have appeared in the historical Search Log
- The current invocation's L1 search yields no new finding not already in the used knowledge set
- Note: The exact path scope for exhaustion is defined by the DSL+architecture triple (see `dsl-arch-scoped-exhaustion` change for precise path enumeration)

**Layer 2 (reference-projects) exhausted when ALL of**:
- All reference-project modules relevant to the current kernel type and framework have been searched in historical plans
- The current invocation's L2 search yields no new implementation pattern

**Layer 3 (public net) — NEVER exhausted**:
- Internet is unbounded; Layer 3 cannot be marked as exhausted
- However, if L3 search produces no actionable result after good-faith effort, report `"search space effectively exhausted"` — no new actionable knowledge found across all layers

### Progressive Escalation Rule

Each invocation follows this flow:

```
1. Parse historical Search Logs → build used knowledge set
2. Search Layer 1 (gpu-wiki)
   ├── New finding found? → Record it (New? = Yes) → MAY stop if sufficient for plan
   └── No new finding? → Mark "L1 exhausted for this invocation" → Continue to step 3
3. Search Layer 2 (reference-projects)
   ├── New finding found? → Record it (New? = Yes) → MAY stop if sufficient for plan
   └── No new finding? → Mark "L2 exhausted for this invocation" → Continue to step 4
4. Search Layer 3 (public net)
   ├── New finding found? → Record it (New? = Yes) → Write plan
   └── No new finding? → Report "search space exhausted" → Return exhaustion status
```

**Key rules**:
- **Always start from L1**: Even if previous invocations reached L3, the current invocation MUST begin at L1 (gpu-wiki may have been updated between iterations)
- **Cannot stop without new knowledge**: The invocation MUST NOT return a plan unless the Search Log contains at least one New? = Yes entry
- **Multi-layer findings are allowed**: A single invocation may record findings from multiple layers (e.g., one from L1 and one from L2) — this is fine as long as novelty constraint is met
- **Escalation within a layer**: Within L1, follow the Five-Tier Priority (P0→P1→P2→P3→P4→P5) defined in gpu-wiki README before declaring L1 exhausted

### Novelty Constraint

The fundamental invariant of the progressive search strategy:

1. **Search Log minimum**: The `plans/v<N>_plan.md` Search Log table MUST contain at least one row with `New? = Yes`
2. **Action derivation**: The single optimization action chosen for the plan MUST be derived from or supported by at least one `New? = Yes` entry — it SHALL NOT be based solely on previously-used findings
3. **Exhaustion exception**: If all three layers fail to produce any new finding, the subagent SHALL:
   - Report status: `"search space exhausted — no new actionable knowledge"`
   - NOT write a speculative plan
   - Return this status to the calling agent for escalation handling (e.g., stop optimization, change strategy, or accept current performance)

---

## Plan Writing

Output file: `plans/v<N>_plan.md`  
Format: Follow the template defined in `reference/plan.md`.

The plan must contain:
- Input Evidence (from profile artifacts)
- Search Log (with Layer and New? columns populated)
- Single Optimization Action (derived from new knowledge)
- Expected Impact
- Risks and Rollback

## Output Contract

The subagent returns:

| Field | Description |
|-------|-------------|
| `plan_path` | Absolute path of written `plans/v<N>_plan.md` |
| `evidence_summary` | Extracted bottleneck evidence from profiles |
| `search_sources` | List of sources searched, with new/used annotation |
| `optimization_action` | The single optimization action chosen |
| `expected_impact` | How the action addresses the current bottleneck |
| `risks` | Risk assessment and rollback strategy |

## Constraints

- Do NOT modify `kernel.py` or any implementation files
- Do NOT perform Stage 3 (implementation)
- Do NOT output multiple parallel optimization actions — exactly one action per plan
- Do NOT skip gpu-wiki (always start from L1)
- Do NOT fabricate hardware specs — use gpu-wiki values or request explicit confirmation
- Do NOT repeat a plan that already exists in historical `plans/v*_plan.md`
- Do NOT read `masked: true` memory files as active data
- Do NOT write a plan without at least one New? = Yes entry in the Search Log
