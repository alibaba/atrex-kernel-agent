"""Project only Workflow public snapshots and their declared output content."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
import re
from pathlib import Path
from threading import RLock

from aka.contracts.observation import OBSERVATION_VERSION, Campaign, Campaigns, Episode
from aka.contracts.workflow_export import WorkflowSnapshotReader
from .artifacts import SCOPE as ARTIFACT_SCOPE, capture_artifacts
from .portable import PortableWorkflowReader

PHASES = ("profile", "research", "planning", "implementation", "correctness", "benchmark", "recording")
PHASE_LABELS = {"profile": "Profiling", "research": "Research", "planning": "Planning",
                "implementation": "Implementation", "correctness": "Correctness",
                "benchmark": "Benchmarking", "recording": "Recording",
                "orchestration": "Orchestration", "unattributed": "Unattributed"}
STAGES = ({"id": "prepare", "label": "Prepare"}, {"id": "explore", "label": "Explore"},
          {"id": "verify", "label": "Verify"}, {"id": "record", "label": "Record"})
PROGRESS_STAGES = {
    "preparing": ("Preparing", "prepare"), "exploring": ("Exploring", "explore"),
    "awaiting_verification": ("Awaiting verification", "verify"),
    "checking_fast_evaluator": ("Evaluator verification", "verify"),
    "verifying": ("ABBA verification", "verify"), "promoting": ("Promoting", "record"),
    "promoted": ("Promoted", "record"), "recording": ("Recording", "record"),
    "recorded": ("Recorded", "record"), "completed": ("Completed", "record"),
    "failed": ("Failed", None), "interrupted": ("Interrupted", None),
    "v1_interrupted": ("Baseline interrupted", "prepare"), "waiting": ("Waiting for records", "prepare"),
}
TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "total_tokens")
SCOPE = "Settled Episode Agent usage. Excludes setup, V1, independent reviewers and auxiliary CLIs. Current invocation usage settles when the call returns."


def obj(value):
    return value if isinstance(value, Mapping) else {}


def string(value):
    return value if isinstance(value, str) else None


def label(value):
    value = value.replace("_", " ")
    return value[:1].upper() + value[1:]


def display_name(name, framework):
    name = name.removeprefix("kernel_opt_")
    if framework and name.lower().endswith("_" + framework.lower()):
        name = name[:-(len(framework) + 1)]
    return label(name)


def sequence(value):
    return value if isinstance(value, (list, tuple)) else []


def integer(value):
    return value if type(value) is int and value >= 0 else None


def version(value):
    if integer(value) is not None:
        return value
    match = re.fullmatch(r"v([0-9]+)", value) if isinstance(value, str) else None
    return int(match[1]) if match else None


def positive(value):
    return (value if type(value) in (int, float) and math.isfinite(value) and value > 0 else None)


def usage(value):
    value = obj(value)
    measurement = value.get("measurement", "unavailable")
    if measurement not in {"exact", "partial", "unavailable"}:
        measurement = "unavailable"
    return {**{key: integer(value.get(key)) if measurement != "unavailable" else None
               for key in TOKEN_FIELDS}, "measurement": measurement}


def experiment(value):
    """Only task-level journal text and scalar evaluation, never private shape inputs."""
    value = obj(value)
    result = {key: value[key] for key in
              ("name", "timestamp", "hypothesis", "change", "result", "conclusion", "summary")
              if isinstance(value.get(key), str)}
    evaluation = obj(value.get("evaluation"))
    result["evaluation"] = {key: evaluation[key] for key in
                            ("correctness", "performance", "latency_us", "kernel_hash")
                            if isinstance(evaluation.get(key), (str, int, float))}
    return result


def episode_tokens(fact):
    value = obj(fact.usage)
    return {"total": integer(value.get("total")), "control_tokens": integer(value.get("control_tokens")),
            "raw_control_tokens": integer(value.get("raw_control_tokens")),
            "measurement": value.get("measurement", "unavailable"),
            "terminal_usage": usage(value.get("terminal_usage")),
            "attribution_measurement": value.get("attribution_measurement", "unavailable"),
            "reason_codes": list(sequence(value.get("reason_codes")))}


def outcome_status(fact):
    if fact.accepted:
        return "accepted"
    return "rejected" if fact.status == "candidate_ready" else fact.status


class WorkspaceReader:
    def __init__(self, workspace: Path, workflow_reader: WorkflowSnapshotReader | None = None):
        self.workspace = Path(workspace).resolve()
        self.workflow_reader = workflow_reader or PortableWorkflowReader()
        self._paths = {}
        self._snapshots = {}
        self._previous_progress = {}
        self._artifact_contents = {}
        self._lock = RLock()

    def _discover(self):
        references = self.workflow_reader.discover(self.workspace)
        if any(not re.fullmatch(r'[A-Za-z0-9_-]+', ref.id) for ref in references):
            raise ValueError('invalid public observation campaign identity')
        if len({ref.id for ref in references}) != len(references):
            raise ValueError('duplicate public observation campaign identity')
        self._paths = {ref.id: ref for ref in references}

    def _read(self, identifier):
        ref = self._paths[identifier]
        try:
            facts = self.workflow_reader.read(ref.workspace)
            if facts.campaign.id != identifier:
                raise ValueError("public Workflow campaign identity changed")
        except (OSError, ValueError):
            previous = self._snapshots.get(identifier)
            if previous is None:
                raise
            return previous, [{"file": "public-snapshot", "code": "record_updating"}]
        self._snapshots[identifier] = facts
        return facts, []

    def _history(self, facts, issues):
        history, incumbent, prior_key, series = [], None, None, 0
        declared_incumbent = None
        available = True
        for candidate in facts.candidates:
            measured = candidate.result
            baseline = candidate.version == 0
            valid = measured.state == "pass" and measured.complete and candidate.promoted
            if valid:
                if measured.objective == "shape_speedup_arithmetic_mean":
                    value = positive(measured.score) or positive(measured.metrics.get("speedup_vs_ref_mean"))
                    objective, unit = measured.objective, "×"
                else:
                    value, objective, unit = positive(measured.latency_us), "latency_us_geomean", "μs"
                if value is not None:
                    key = (measured.objective or objective, measured.expected_case_ids,
                           candidate.comparability or ('unqualified', candidate.version))
                    if key != prior_key:
                        series += 1
                    prior_key = key
                    incumbent = {"version": candidate.version, "value": value, "unit": unit,
                        "label": "Mean speedup across shapes" if unit == "×" else "Geometric mean latency",
                        "objective": objective, "series": series, "source": f"candidate:v{candidate.version}",
                        "revision": candidate.revision}
                    available = True
                    if candidate == facts.incumbent:
                        declared_incumbent = dict(incumbent)
            missing = candidate.promoted and not valid and not baseline
            if missing:
                issues.append({"file": f"candidate:v{candidate.version}", "code": "promotion_measurement_unavailable"})
                prior_key, available = None, False
            history.append({"version": candidate.version, "promoted": bool(valid and not baseline),
                            "episode": candidate.episode,
                            "incumbent": dict(incumbent) if incumbent and available else None})
        return {"latest_version": facts.candidates[-1].version if facts.candidates else None,
                "incumbent": declared_incumbent, "history": history}

    def _snapshot(self, identifier):
        facts, issues = self._read(identifier)
        issues += [{"file": "public-warning", "code": row["code"]} for row in facts.warnings]
        value = facts.progress
        progress = {"phase": value.phase, "episode": value.episode, "version": value.version,
                    "mode": value.mode, "experiment_count": value.experiment_count,
                    "latest_experiment": experiment(value.latest_experiment), "updated_at": value.updated_at,
                    "record_updating": value.record_updating or bool(issues and issues[0]["code"] == "record_updating"),
                    "pending_tokens": value.pending_tokens,
                    "last_terminal": {"status": value.last_terminal.get("status"),
                        "updated_at": value.last_terminal.get("updated_at"), "stale": value.last_terminal.get("stale", False)}}
        if progress["record_updating"] and identifier in self._previous_progress:
            progress = {**deepcopy(self._previous_progress[identifier]), "record_updating": True}
        elif not progress["record_updating"]:
            self._previous_progress[identifier] = deepcopy(progress)
        phase_label, stage_id = PROGRESS_STAGES.get(progress["phase"], (label(progress["phase"]), progress["phase"]))
        progress.update(phase_label=phase_label, stage_id=stage_id)
        stages = [dict(stage) for stage in STAGES]
        if stage_id and not any(stage["id"] == stage_id for stage in stages):
            stages.append({"id": stage_id, "label": phase_label})
        token_summary = facts.token_summary
        measurement = token_summary.get("measurement", "unavailable")
        tokens = {"total": integer(token_summary.get("total")) if measurement != "unavailable" else None,
                  "control_total": integer(token_summary.get("control_total")), "measurement": measurement,
                  "receipt_count": integer(token_summary.get("receipt_count")),
                  "reason_codes": list(sequence(token_summary.get("reason_codes"))),
                  "pending": progress["pending_tokens"], "scope": token_summary.get("scope", SCOPE)}
        rows = [{"episode": row.episode, "version": row.version, "mode": row.mode,
                 "status": outcome_status(row), "accepted": row.accepted, "summary": row.summary,
                 "violation": row.violation, "verification_gate": row.gate, "tokens": episode_tokens(row)}
                for row in facts.episodes]
        snapshot = {"observation_version": OBSERVATION_VERSION, "id": identifier,
                    "name": facts.campaign.name, "display_name": display_name(facts.campaign.name, facts.framework),
                    "workspace": str(self._paths[identifier].workspace), "stages": stages,
                    "artifact_scope": ARTIFACT_SCOPE, "framework": facts.framework or None,
                    "platform": facts.hardware.get("platform"), "arch": facts.hardware.get("arch"),
                    "updated_at": value.updated_at, "progress": progress, "canonical": self._history(facts, issues),
                    "counts": {key: integer(facts.counts.get(key)) for key in ("episodes", "accepted", "rejected")},
                    "tokens": tokens, "episodes": rows, "issues": issues}
        descriptions, contents = capture_artifacts(facts, snapshot)
        snapshot["artifacts"] = descriptions
        cache = self._artifact_contents.setdefault(identifier, {})
        cache.update(contents)
        while len(cache) > 128 or sum(len(content) for content in cache.values()) > 64 * 1024 * 1024:
            oldest = next(iter(cache))
            if oldest in contents:
                break
            del cache[oldest]
        return snapshot

    def campaigns(self) -> Campaigns:
        with self._lock:
            self._discover()
            campaigns = []
            for identifier in self._paths:
                value = self._snapshot(identifier)
                campaigns.append({key: value[key] for key in
                    ("id", "name", "display_name", "framework", "platform", "updated_at", "progress", "tokens", "counts")})
            known = [row["tokens"]["total"] for row in campaigns if row["tokens"]["total"] is not None]
            partial = len(known) != len(campaigns) or any(row["tokens"]["measurement"] != "exact" for row in campaigns)
            return {"observation_version": OBSERVATION_VERSION, "campaigns": campaigns,
                    "empty_message": "No public campaigns yet. Waiting for Workflow snapshots.",
                    "tokens": {"total": sum(known) if known else None,
                               "measurement": "partial" if partial else "exact", "scope": SCOPE}}

    def campaign(self, identifier: str) -> Campaign:
        with self._lock:
            self._discover()
            return self._snapshot(identifier)

    def artifact(self, identifier: str, artifact_id: str) -> bytes:
        with self._lock:
            try:
                return self._artifact_contents[identifier][artifact_id]
            except KeyError as error:
                raise KeyError(artifact_id) from error

    def episode(self, identifier: str, number: int) -> Episode:
        with self._lock:
            self._discover()
            snapshot = self._snapshot(identifier)
            facts = self._snapshots[identifier]
            fact = next((row for row in facts.episodes if row.episode == number), None)
            if fact is None:
                raise KeyError(number)
            phases = {row["phase"]: row for row in fact.phases}
            keys = dict.fromkeys((*PHASES, *phases))
            rows = [{"phase": key, "label": PHASE_LABELS.get(key, label(key)),
                     "usage": usage(obj(phases.get(key)).get("usage"))} for key in keys]
            return {"observation_version": OBSERVATION_VERSION, "episode": fact.episode,
                    "version": fact.version, "mode": fact.mode, "status": outcome_status(fact),
                    "accepted": fact.accepted, "summary": fact.summary, "violation": fact.violation,
                    "verification": dict(fact.verification), "tokens": episode_tokens(fact), "phases": rows,
                    "experiments": [experiment(row) for row in fact.experiments],
                    "latest_experiment": snapshot["progress"]["latest_experiment"] if facts.progress.episode == number else None,
                    "detail_available": fact.detail_available, "issues": snapshot["issues"],
                    "evidence": [artifact.id for artifact in facts.artifacts if artifact.filename.startswith(f"episodes/e{number:04d}/")]}
