# Candidate Proposal

## Evidence-to-Action Chain

- Measured evidence: <profile, benchmark, source, or prior-episode observation>
- Inference: <what the evidence implies about the bottleneck>
- Optimization category: <exactly one category>
- Proposed action: <smallest coherent implementation direction>
- Expected mechanism: <why the proposed action should affect the measured bottleneck>

## Proposed Changes

| Path and symbol | Intended change | Evidence link | Expected effect |
| --- | --- | --- | --- |
| `<path>:<symbol>` | <concrete change> | <packet evidence> | <metric and direction> |

## Scope and Constraints

- Must preserve: <correctness, compatibility, and campaign constraints>
- Allowed choices: <permitted implementation choices>
- Prohibited choices: <forbidden fallbacks, dependencies, or evaluator coupling>
- Rejected directions: <direction and evidence-based reason>

## Validation and Falsification

- Correctness: <full-workload and multi-seed checks>
- Performance: <comparable measurement and attribution checks>
- Falsification condition: <result that disproves the hypothesis>
- Rollback condition: <result that requires reverting the experiment>
- Direction stop condition: <evidence that exhausts this direction>

## Questions and Assumptions

- <`None`, or a material unresolved question or conservative assumption>
