"""Persistent research sessions over task contracts; no benchmark dispatch here."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import sys

from .content import document
from .service import TaskService, exclusive


def prompt(service, task_id, spec, goal):
    root = service.directory(task_id)
    command = shlex.join([sys.executable, "-m", "aka.light.tasks.cli", "invoke"])
    execution = shlex.join([sys.executable, "-m", "aka.light.execution.command_cli"])
    return f"""Continue this GPU module research task in the same source workspace and Session.

{spec['instructions']}

Caller notes for this research task:
{spec.get('notes', '')}
Follow these research and reporting instructions within the frozen contract.
Notes do not change acceptance rules, precision, implementation policy or budgets.

Implementation requirements checked independently on the published source:
{document({k: v for k, v in spec.get('config', {}).get('implementation', {}).items() if k != 'reviewer'})}

Objective (used by both candidate selection and reports):
{goal.describe()}

The task contract is frozen at {root / 'contract'}. Read it; never modify it.
Only edit the current candidate workspace. Keep reference, workload, validation,
execution configuration, task ledger and saved evidence unchanged. Commit all candidate
changes before measurement. Do not run GPU code on this host or access gateway credentials.

Use these host-owned commands:
  {command} measure
  {command} submit-candidate --evaluation-id EVALUATION_ID
  {command} results
  {command} events

For your own research experiments use the task-bound execution tool:
  {execution} [--input REMOTE=LOCAL] [--output PATH] [--env KEY=VALUE] -- COMMAND ARG...
This is the task-bound `aka execution` entry point.
It uses the same execution backend as acceptance. Choose compilation, profiling,
disassembly and experimental commands yourself; there are no separate tools for
those operations. Agate uses dev internally. To run a shell script, pass
`-- bash -lc '...'`. The workspace (including scratch/) is snapshotted for each
call; you may experiment before committing. Each call starts in that snapshot,
so combine dependent build/run commands in one call and request output files.
Keep execution IDs and recover interrupted calls with `aka execution --execution-id ID`.
Research output is not acceptance evidence. Commit a candidate and measure it
through the frozen task contract before publishing.

An acceptance evaluation must pass before submission. Diagnostic evidence is not acceptance.
Submissions publish immutable candidates without changing your HEAD. Continue exploring
after useful submissions; retain research notes under scratch/. The parent owns adoption.
Declared adapter capabilities: {', '.join(spec['capabilities'])}.
Delivery requirements: {document(spec['delivery'])}
"""


def run(root, task_id, *, agent_factory=None):
    service = TaskService(root)
    directory = service.directory(task_id)
    with exclusive(directory / "worker.lock"):
        task, adapter, goal = service.bind(task_id)
        spec = task["manifest"]
        if task["control"] == "run":
            service.store.state(task_id, "running")
        try:
            if spec.get("agent") and agent_factory is None:
                from .agents import preflight
                preflight(spec["agent"])
            policy = spec.get("config", {}).get("implementation", {})
            if policy.get("framework") or policy.get("self_authored"):
                from .agents import preflight
                preflight(policy.get("reviewer") or spec.get("agent") or {"runtime": "claude"})
            # Recover known remote jobs by ID. Never silently launch a second evaluation.
            with service.store.connect() as db:
                pending = [dict(r) for r in db.execute(
                    "SELECT * FROM evaluations WHERE task_id=? AND state!='finished' ORDER BY created", (task_id,))]
            for row in pending:
                service.measure(task_id, evaluation_id=row["id"], _wait=True)
            control = service.store.task(task_id)["control"]
            if control != "run":
                service.store.state(task_id, "paused" if control == "pause" else "cancelled")
                return
            with service.store.connect() as db:
                baseline = db.execute("SELECT id FROM evaluations WHERE task_id=? AND role='baseline' ORDER BY created LIMIT 1", (task_id,)).fetchone()
                initial = db.execute("SELECT id FROM evaluations WHERE task_id=? AND role='candidate' ORDER BY created LIMIT 1", (task_id,)).fetchone()
            if baseline is None:
                baseline_result = service.measure(task_id, role="baseline", _wait=True)
            else:
                baseline_result = service.store.evaluation(baseline["id"])
            control = service.store.task(task_id)["control"]
            if control != "run":
                service.store.state(task_id, "paused" if control == "pause" else "cancelled")
                return
            if spec.get("config", {}).get("baseline_required", False) and (
                    baseline_result["assessment"]["verdict"] != "pass" or not baseline_result["assessment"]["complete"]):
                service.store.state(task_id, "needs_input", "Baseline failed the frozen task contract")
                return
            if initial is None:
                measured = service.measure(task_id, _wait=True)
                from .service import assessment
                if goal.eligible(assessment(measured["assessment"])):
                    service.submit_candidate(task_id, measured["id"])
            if not spec.get("agent"):
                control = service.store.task(task_id)["control"]
                service.store.state(task_id, "paused" if control == "pause" else "cancelled" if control == "cancel" else
                                    "awaiting_candidate", "Task prepared; external authoring enabled")
                return
            from .runtime import TaskRuntime
            with TaskRuntime(service, task_id) as runtime:
                run_sessions(service, task_id, spec, goal, runtime.environment, agent_factory=agent_factory)
        except BaseException as error:
            control = service.store.task(task_id)["control"]
            with service.store.connect() as db:
                pending = db.execute("SELECT count(*) FROM evaluations WHERE task_id=? AND state!='finished'", (task_id,)).fetchone()[0]
            state = (("pausing" if pending else "paused") if control == "pause" else
                     ("cancelling" if pending else "cancelled") if control == "cancel" else "interrupted")
            service.store.state(task_id, state, f"{type(error).__name__}: {error}")
            raise


def run_sessions(service, task_id, spec, goal, runtime_environment, *, agent_factory=None):
    from aka.contracts.agent import AgentAccess
    from aka.agent.provider import Agent, AgentOptions
    directory = service.directory(task_id)
    budget = spec.get("budget", {})
    from .agents import resolve
    agent_config, agent_environment = resolve(spec["agent"])
    owner = (agent_factory() if agent_factory else Agent(AgentOptions(
        sandbox=agent_config.get("sandbox", "none"),
        timeout_seconds=budget.get("agent_seconds", 3600))))
    task = service.store.task(task_id)
    session = owner.open_session(agent_config.get("runtime", "claude"), directory / "workspace",
                                 **({} if agent_factory else {"access": AgentAccess(read_only_paths=tuple(directory / name for name in ("contract", "baseline", "seed"))), "reopen": bool(task["session"])}),
                                 session_id=task["session"], output_directory=directory / "sessions",
                                 environment={**agent_environment,
                                              "PYTHONPATH": str(Path(__import__("aka").__path__[0]).parent),
                                              "ATREX_AGENT_SESSION_KEY": task_id,
                                              "PI_CODING_AGENT_SESSION_DIR": str(directory / "sessions/pi"),
                                              **runtime_environment})
    try:
        resume = bool(task["session"])
        while True:
            task = service.store.task(task_id)
            budget = task["manifest"].get("budget", {})
            if task["control"] != "run":
                service.store.state(task_id, "paused" if task["control"] == "pause" else "cancelled")
                return
            if "tokens" in budget:
                with service.store.connect() as db:
                    last = db.execute("SELECT kind,data FROM events WHERE task_id=? AND kind IN ('episode_started','episode_finished') ORDER BY seq DESC LIMIT 1", (task_id,)).fetchone()
                    unknown_review = db.execute("SELECT 1 FROM events WHERE task_id=? AND kind='implementation_review' AND json_extract(data,'$.usage.measurement') IS NOT NULL AND json_extract(data,'$.usage.total_tokens') IS NULL LIMIT 1", (task_id,)).fetchone()
                if unknown_review:
                    service.store.state(task_id, "needs_input", "Implementation review has unknown token accounting")
                    return
                if last and (last["kind"] == "episode_started" or json.loads(last["data"])["usage"]["total_tokens"] is None):
                    service.store.state(task_id, "needs_input", "Interrupted invocation has no token accounting; create a new budgeted task")
                    return
            if task["episodes"] >= budget.get("episodes", 1) or task["tokens"] >= budget.get("tokens", 2**63 - 1):
                service.store.state(task_id, "finished", "Agent admission budget exhausted")
                return
            if resume and not session.supports_resume:
                raise ValueError("Selected Agent runtime cannot resume a persistent session")
            message = prompt(service, task_id, spec, goal)
            with service.store.connect() as db:
                feedback = [json.loads(row[0]) for row in db.execute(
                    "SELECT data FROM events WHERE task_id=? AND kind='integration_feedback' ORDER BY seq DESC LIMIT 10", (task_id,))][::-1]
            if feedback:
                message += "\nParent integration feedback:\n" + document(feedback[-10:])
            # Reserve before invocation: a crashed process must not reset admission.
            with service.store.transaction() as db:
                db.execute("UPDATE tasks SET episodes=episodes+1,session=? WHERE id=?",
                           (getattr(session, "session_id", "") or task["session"], task_id))
                service.store.emit(db, task_id, "episode_started", {"episode": task["episodes"] + 1})
            observation = session.invoke(message, resume=resume, reasoning_effort=agent_config.get("reasoning_effort", "high"))
            with service.store.transaction() as db:
                db.execute("UPDATE tasks SET session=?,tokens=tokens+? WHERE id=?",
                           (observation.session_id, observation.terminal_usage.total_tokens or 0, task_id))
                service.store.emit(db, task_id, "episode_finished", {
                    "exit_status": observation.exit_status, "timed_out": observation.timed_out,
                    "usage": asdict(observation.terminal_usage), "session_id": observation.session_id,
                    "errors": list(getattr(observation, "observation_errors", ()))})
            if observation.exit_status != 0 or observation.timed_out:
                raise RuntimeError("Agent invocation interrupted; source and Session retained")
            if "tokens" in budget and observation.terminal_usage.total_tokens is None:
                service.store.state(task_id, "needs_input", "Agent did not report usage required by the token budget")
                return
            resume = True
    finally:
        session.close()
        owner.close()
