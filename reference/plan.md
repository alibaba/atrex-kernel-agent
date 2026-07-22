# Iteration V<N> Plan

## Input Evidence
<profile evidence from Stage 1, compared with the previous iteration report when available>

## Stall Context
- Consecutive reverted/no-improvement iterations: <stall_count>
- Search mode: <normal | forced expansion>
- Novel finding required: <No | Yes>

## Search Log

Each row records one search action. Rules:
- **Layer**: L1 = gpu-wiki (docs/ + 3rdparty/ + reference-kernels/), L2 = reference-projects/, L3 = public net
- **New?**: Yes if this (Source, Query, Finding) triple has NOT appeared in any prior v*_plan.md Search Log; No otherwise
- In normal mode, an untried historical finding may be reused with `New? = No`
- In forced-expansion mode, the table MUST contain `New? = Yes` and the action must derive from it

| Source | Layer | Query | Finding | New? | Actionability |
|--------|-------|-------|---------|------|---------------|
| | L1 | | | | |
| | L2 | | | | |
| | L3 | | | | |

## This Iteration's Optimization Action
<choose exactly one optimization category>

## Expected Impact
<how the action improves the current bottleneck and ISA targets>

## Performance Expectation and ISA Escalation
<state a measurable post-change expectation; inspect PTX/SASS only if measured behavior misses that expectation and compiler lowering may explain the mismatch>

## Risks and Rollback
<why it might fail and how to roll back>
