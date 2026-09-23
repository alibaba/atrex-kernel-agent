# Kernel optimization episode {{EPISODE}}

Optimize `kernel.py` through evidence-backed engineering Directions. Continue investigating,
implementing, measuring, and repairing while a concrete next step remains; one failed compile or
benchmark is not, by itself, a reason to stop.

## Context

- Workspace: `{{WORKSPACE}}`
- Canonical version produced by the supervisor: `v{{VERSION}}`
- Platform: `{{PLATFORM}}`
- Framework: `{{FRAMEWORK}}`
- Additional constraints: {{NOTES}}
{{AGENT_RUNTIME}}

{{PLUGINS}}

{{RESUME_DIRECTIVE}}

{{GOAL_DIRECTIVE}}

Use `scratch/` for temporary requests, diagnostics, and saved source copies.

{{MODE_POLICY}}

{{EVALUATOR}}

{{HARDWARE}}

{{SANDBOX}}

{{ACCEPTANCE_REQUEST}}

{{CONVERSION_DIRECTIVE}}

## Prior iteration state

Start with canonical `memory/v*.json`, then use Journal list/load tools for relevant historical
Directions and Experiments. Treat prior analysis as interpretation, not measurement authority.
Revisit a rejected Direction only with new evidence or a materially different implementation.

{{WIKI_DIRECTIVE}}

{{RUNTIME_JOURNAL_CONTRACT}}

## Engineering loop

Use this as a working guide when useful, not as additional mandatory stages:

1. **Understand the starting point.** Inspect the current Kernel, public contract, and relevant
   history. Identify existing measurements, the likely bottleneck, and what remains uncertain.
2. **Choose a Direction.** State the hypothesis, expected benefit, concrete change, validation
   method, and stopping condition. Register it before exploration; keep the concise plan in the
   Direction Journal rather than creating separate draft or plan files.
3. **Research the question.** Use targeted references, Wiki, or Profile to resolve a specific
   uncertainty that informs the next change. Avoid research or profiling without a concrete purpose.
4. **Implement and validate.** Work on one coherent candidate at a time. Use Gateway Evaluate for
   correctness and performance, reusing a matching saved result instead of repeating the measurement.
5. **Record and decide.** Record each decisive result as an Experiment, separating measured facts
   from analysis. Use the evidence to continue, repair, restore a measured Kernel, or change Direction.
   Prepare the report as work proceeds; submit an evidence-backed candidate, pivot, or blocker using
   the terminal contract below. Leave secondary tweaks for a later Episode.

## Terminal contract

Choose an evidence-backed state:

1. `candidate_ready`: the selected Experiment cites a passing full Evaluate for the exact candidate.
   The Supervisor automatically runs or reuses the evaluator-specific acceptance check before
   accepting the report: six-case multi-seed correctness for Atrex-Bench, or the official full
   `workload.jsonl` evaluation for SOL-ExecBench. A failed check leaves the Episode open for repair
   and resubmission.
   In production, the Supervisor also runs targeted numerical probes. A measured failure returns
   distributions and metrics for repair; the same retained plan is checked after resubmission.
   Incomplete probe validation is a blocker, not evidence that the Kernel is wrong.
2. `pivot`: the engineering direction is exhausted and a fresh episode should pursue another one.
3. `blocked`: infrastructure or missing authority prevents meaningful progress.

The Supervisor decides promotion using the required ABBA comparison. It reuses a matching Runtime
record when available, otherwise requests the missing measurement; ordinary Evaluate does not
replace ABBA. You need not run ABBA just to finish. Report acceptance is not Kernel promotion.
Do not claim a speedup merely to terminate; a well-supported pivot is valid.

{{PPU_DIRECTIVE}}
