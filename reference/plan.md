# Iteration V<N> Plan

## Input Evidence
<profile evidence from Stage 1, compared with the previous iteration report when available>

## Search Log

Each row records one search action. Rules:
- **Layer**: L1 = gpu-wiki (docs/ + 3rdparty/ + reference-kernels/), L2 = reference-projects/, L3 = public net
- **New?**: Yes if this (Source, Query, Finding) triple has NOT appeared in any prior v*_plan.md Search Log; No otherwise
- The table MUST contain at least one row with New? = Yes (novelty constraint)
- Optimization actions MUST be derived from at least one New? = Yes entry
- Each action MUST address a different aspect or use a fundamentally different technique (diversity requirement)

| Source | Layer | Query | Finding | New? | Actionability |
|--------|-------|-------|---------|------|---------------|
| | L1 | | | | |
| | L2 | | | | |
| | L3 | | | | |

## Ranked Optimization Actions
<target 3 actions, minimum 1. Ranked by expected impact: primary / secondary / fallback>

### Action 1 (Primary): <title>
<optimization description and category>

#### Expected Impact
<how this action improves the current bottleneck and ISA targets>

#### Risks and Rollback
<why it might fail and how to roll back>

### Action 2 (Secondary): <title>
<optimization description and category>

#### Expected Impact
<how this action improves the current bottleneck and ISA targets>

#### Risks and Rollback
<why it might fail and how to roll back>

### Action 3 (Fallback): <title>
<optimization description and category>

#### Expected Impact
<how this action improves the current bottleneck and ISA targets>

#### Risks and Rollback
<why it might fail and how to roll back>