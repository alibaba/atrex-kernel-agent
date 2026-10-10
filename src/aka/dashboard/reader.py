"""Project durable optimizer records without importing the optimizer or writing files."""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

RUNTIME = ".atrex_long_horizon"
PHASES = ("profile", "research", "planning", "implementation", "correctness", "benchmark", "recording")
TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "total_tokens")
SCOPE = "Settled Episode Agent usage. Excludes setup, V1, independent reviewers and auxiliary CLIs. Current invocation usage settles when the call returns."


def obj(value):
    return value if isinstance(value, dict) else {}


def sequence(value):
    return value if isinstance(value, list) else []


def integer(value):
    return value if type(value) is int and value >= 0 else None


def version(value):
    if integer(value) is not None:
        return value
    match = re.fullmatch(r"v([0-9]+)", value) if isinstance(value, str) else None
    return int(match[1]) if match else None


def positive(value):
    return (value if type(value) in (int, float) and math.isfinite(value) and value > 0 else None)


def stamp(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, OverflowError):
        return 0.0


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat() if value else None


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


def outcome_status(attempt):
    if attempt.get("accepted") is True:
        return "accepted"
    return "rejected" if attempt.get("status") == "candidate_ready" else attempt.get("status")


def episode_tokens(attempt, telemetry):
    phases = obj(telemetry.get("phase_tokens"))
    terminal = usage(phases.get("terminal_usage"))
    reasons = sorted({str(r) for r in sequence(telemetry.get("reason_codes"))
                      + sequence(phases.get("reason_codes"))
                      + sequence(obj(attempt.get("telemetry")).get("reason_codes"))})
    control = integer(telemetry.get("control_tokens"))
    if control is None:
        control = integer(attempt.get("tokens"))
    total = terminal["total_tokens"]
    measurement = terminal["measurement"] if total is not None else "unavailable"
    if total is not None and control is not None and total != control:
        reasons = sorted(set(reasons) | {"control_token_total_mismatch"})
    if ({"same_session_resume_usage_semantics_unqualified", "control_token_total_mismatch"}
            & set(reasons)) and measurement == "exact":
        measurement = "partial"
    if total == 0 and measurement != "exact" and "structured_usage_unavailable" in reasons:
        total, measurement = None, "unavailable"
    if total is None and control:
        # Legacy control totals remain useful, but cannot certify structured usage.
        total, measurement = control, "partial"
        reasons = sorted(set(reasons) | {"structured_usage_unavailable"})
    displayed_control = control if control or total is not None else None
    return {"total": total, "control_tokens": displayed_control, "raw_control_tokens": control,
            "measurement": measurement, "terminal_usage": terminal,
            "attribution_measurement": phases.get("measurement", "unavailable"),
            "reason_codes": reasons}


class WorkspaceReader:
    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()
        self._paths = {}
        self._cache = {}
        self._previous_progress = {}
        # ThreadingHTTPServer can request list and detail concurrently.
        self._lock = RLock()

    def _discover(self):
        root = self.workspace
        if (root.name.startswith("kernel_opt_") or (root / RUNTIME).is_dir()
                or (root / "memory").is_dir() or (root / "framework_baseline.json").is_file()):
            paths = [root]
        else:
            paths = sorted(p for p in root.iterdir() if p.name.startswith("kernel_opt_")
                           and p.is_dir() and not p.is_symlink())
        self._paths = {hashlib.sha256(str(p).encode()).hexdigest()[:16]: p for p in paths}

    def _read(self, path, issues, label):
        previous = self._cache.get(path)
        try:
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if previous and previous[0] == signature:
                return previous[1]
            def invalid_constant(value):
                raise ValueError(f"non-finite JSON number: {value}")
            value = json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid_constant)
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
        except FileNotFoundError:
            self._cache.pop(path, None)
            return {}
        except (OSError, UnicodeError, ValueError) as exc:
            issues.append({"file": label, "code": "record_updating" if previous else "record_invalid",
                           "detail": str(exc)})
            return previous[1] if previous else {}
        self._cache[path] = (signature, value)
        return value

    def _mtime(self, path):
        cached = self._cache.get(path)
        return cached[0][0] / 1e9 if cached else 0.0

    def _records(self, path, state, issues):
        records = {a["episode"]: a for a in sequence(state.get("attempts"))
                   if isinstance(a, dict) and integer(a.get("episode"))}
        for source in sorted((path / RUNTIME / "episodes").glob("e*/attempt.json")):
            match = re.fullmatch(r"e([0-9]+)", source.parent.name)
            if match is None:
                continue
            value = self._read(source, issues, str(source.relative_to(path)))
            if not value:
                continue
            if value.get("episode") != int(match[1]):
                issues.append({"file": str(source.relative_to(path)), "code": "episode_identity_mismatch"})
                continue
            records[int(match[1])] = value
        return records

    def _telemetry(self, path, number, attempt, issues):
        relative = f"{RUNTIME}/episodes/e{number:04d}/telemetry.summary.json"
        value = self._read(path / relative, issues, relative)
        if value and (value.get("episode") != number or
                      version(value.get("version")) != version(attempt.get("version"))):
            issues.append({"file": relative, "code": "telemetry_identity_mismatch"})
            return {}
        return value

    def _history(self, path, baseline, records, issues):
        accepted = {version(a.get("version")): a for a in records.values() if a.get("accepted") is True}
        history, incumbent, prior_key, series = [], None, None, 0
        curve_available = True
        versions = []
        for source in (path / "memory").glob("v*.json"):
            n = version(source.stem)
            if n is not None:
                versions.append((n, source))
        for n, source in sorted(versions):
            memory = self._read(source, issues, f"memory/v{n}.json")
            if not memory or version(memory.get("version")) != n:
                if n in accepted:
                    curve_available, prior_key = False, None
                if memory:
                    issues.append({"file": f"memory/v{n}.json", "code": "version_identity_mismatch"})
                continue
            perf = obj(memory.get("performance"))
            shapes = obj(perf.get("latency_us_by_shape"))
            expected = integer(perf.get("expected_shape_count"))
            complete = (perf.get("measurement_status") in (None, "complete")
                        and all(positive(v) is not None for v in shapes.values())
                        and (bool(shapes) or perf.get("performance_objective") != "shape_speedup_arithmetic_mean")
                        and (expected is None or len(shapes) == expected))
            valid = (complete and obj(memory.get("correctness")).get("status") == "PASS"
                     and obj(memory.get("quality_gate")).get("result") == "PASS")
            is_baseline = n == 0 or n == version(baseline.get("version"))
            promotion = accepted.get(n)
            verification = obj(obj(promotion).get("verification"))
            valid_promotion = (promotion is not None and verification.get("gate") == "PASS"
                               and not verification.get("error")
                               and perf.get("measurement_subject") in (None, "candidate")
                               and (not memory.get("git_commit_hash") or not promotion.get("candidate_commit")
                                    or memory["git_commit_hash"] == promotion["candidate_commit"]))
            promoted = False
            recorded_objective = perf.get("performance_objective")
            if valid and (is_baseline or valid_promotion):
                objective = perf.get("performance_objective")
                if objective == "shape_speedup_arithmetic_mean":
                    value = positive(perf.get("performance_score")) or positive(perf.get("speedup_vs_ref_mean"))
                    unit = "×"
                else:
                    value = positive(perf.get("latency_us_geomean")) or positive(perf.get("latency_us"))
                    objective, unit = "latency_us_geomean", "μs"
                if value is not None:
                    key = (recorded_objective or objective, tuple(sorted(shapes)), perf.get("device_identity"),
                           baseline.get("platform"), baseline.get("arch"))
                    if key != prior_key:
                        series += 1
                    prior_key = key
                    incumbent = {"version": n, "value": value, "unit": unit,
                                 "objective": objective, "series": series,
                                 "source": f"memory/v{n}.json"}
                    curve_available = True
                    promoted = not is_baseline
            missing_promotion = promotion is not None and not promoted and not is_baseline
            if missing_promotion:
                issues.append({"file": f"memory/v{n}.json", "code": "promotion_measurement_unavailable"})
                # Keep the last certified incumbent in the headline, leave a gap in the curve.
                prior_key = None
                curve_available = False
            history.append({"version": n, "promoted": promoted,
                            "episode": obj(promotion).get("episode"),
                            "incumbent": dict(incumbent) if incumbent and curve_available else None})
        return {"latest_version": history[-1]["version"] if history else None,
                "incumbent": incumbent, "history": history}

    def _snapshot(self, identifier):
        path = self._paths[identifier]
        issues = []
        def read(relative):
            return self._read(path / relative, issues, relative)
        state = read(f"{RUNTIME}/state.json")
        active = read(f"{RUNTIME}/active_episode.json")
        live = read("memory/live.json")
        baseline = read("framework_baseline.json")
        manifest = read("trace-retention-manifest.json")
        crash = read(f"{RUNTIME}/framework_baseline/progress.json")
        records = self._records(path, state, issues)
        if integer(state.get("episodes")) is not None and max(records, default=0) > state["episodes"]:
            issues.append({"file": f"{RUNTIME}/state.json", "code": "record_updating"})
        rows = []
        for number, attempt in sorted(records.items()):
            telemetry = self._telemetry(path, number, attempt, issues)
            rows.append({"episode": number, "version": version(attempt.get("version")),
                         "mode": attempt.get("mode"), "status": outcome_status(attempt),
                         "accepted": attempt.get("accepted") is True,
                         "summary": attempt.get("summary"), "violation": attempt.get("violation"),
                         "verification_gate": obj(attempt.get("verification")).get("gate"),
                         "tokens": episode_tokens(attempt, telemetry)})

        associated = (not active or not live or (active.get("episode") == live.get("episode")
                      and version(active.get("memory_version")) == version(live.get("version"))))
        if not associated:
            issues.append({"file": "memory/live.json", "code": "active_live_identity_mismatch"})
        matched_live = live if associated else {}
        phase = active.get("phase") or matched_live.get("phase") or "waiting"
        episode = active.get("episode", matched_live.get("episode"))
        settled = episode in records or matched_live.get("phase") == "recorded"
        progress = {"phase": "recorded" if settled else phase, "episode": episode,
                    "version": version(active.get("memory_version", matched_live.get("version"))),
                    "mode": active.get("mode") or obj(records.get(episode)).get("mode"),
                    "experiment_count": integer(matched_live.get("experiment_count")),
                    "latest_experiment": experiment(matched_live.get("latest_experiment")),
                    "updated_at": iso(max(stamp(matched_live.get("updated_at")),
                                          self._mtime(path / RUNTIME / "active_episode.json"))),
                    "process_status": "unknown", "record_updating": not associated,
                    "pending_tokens": bool(episode and not settled and phase in {"preparing", "exploring"}),
                    "v1_crash": {"status": crash.get("status"), "updated_at": crash.get("updated_at")}}
        if not associated and identifier in self._previous_progress:
            progress = {**self._previous_progress[identifier], "record_updating": True}
        elif associated:
            self._previous_progress[identifier] = progress.copy()

        observed_time = max(stamp(live.get("updated_at")), self._mtime(path / RUNTIME / "active_episode.json"),
                            self._mtime(path / RUNTIME / "state.json"))
        terminal_time = stamp(manifest.get("generated_at"))
        stale_terminal = bool(manifest and observed_time > terminal_time)
        if stale_terminal:
            issues.append({"file": "trace-retention-manifest.json", "code": "older_terminal_record"})
        progress["last_terminal"] = {"status": manifest.get("status"), "updated_at": manifest.get("generated_at"),
                                     "stale": stale_terminal, "current_run_confirmed": False}
        if not active and not live and crash.get("status") == "interrupted":
            progress["phase"] = "v1_interrupted"
        if not active and not stale_terminal and manifest.get("status") in {"completed", "failed", "interrupted"}:
            progress["phase"] = manifest["status"]

        raw_tokens = integer(state.get("tokens"))
        unavailable = any(r["tokens"]["measurement"] == "unavailable" for r in rows)
        observed_sum = sum(r["tokens"]["raw_control_tokens"] or 0 for r in rows)
        reasons = sorted({reason for r in rows for reason in r["tokens"]["reason_codes"]})
        if unavailable:
            reasons = sorted(set(reasons) | {"episode_usage_unavailable"})
        if raw_tokens is not None and observed_sum != raw_tokens:
            reasons = sorted(set(reasons) | {"control_usage_not_fully_archived"})
        if raw_tokens is None:
            measurement = "unavailable"
        elif unavailable or any(r["tokens"]["measurement"] != "exact" for r in rows) or observed_sum != raw_tokens:
            measurement = "partial" if raw_tokens else "unavailable"
        else:
            measurement = "exact"
        total = None if measurement == "unavailable" else raw_tokens
        canonical = self._history(path, baseline, records, issues)
        hardware = obj(manifest.get("hardware"))
        memory_time = max((self._mtime(source) for source in self._cache if source.parent == path / "memory"), default=0)
        return {"id": identifier, "name": path.name, "workspace": str(path),
                "framework": baseline.get("framework"), "platform": baseline.get("platform") or hardware.get("platform"),
                "arch": baseline.get("arch") or hardware.get("arch"),
                "updated_at": iso(max(observed_time, terminal_time, stamp(baseline.get("recorded_at")), memory_time)),
                "progress": progress, "canonical": canonical,
                "counts": {key: integer(state.get(key)) for key in
                           ("episodes", "accepted", "rejected", "pivoted", "blocked", "interrupted", "protocol_failures")},
                "tokens": {"total": total, "control_total": raw_tokens, "measurement": measurement,
                           "receipt_count": len(obj(state.get("usage_receipts"))), "reason_codes": reasons,
                           "pending": progress["pending_tokens"], "scope": SCOPE},
                "budget": {"token_limit": None, "version_limit": None, "status": "not_recorded"},
                "episodes": rows, "issues": issues}

    def campaigns(self):
        with self._lock:
            self._discover()
            campaigns = []
            for identifier in self._paths:
                value = self._snapshot(identifier)
                campaigns.append({key: value[key] for key in
                                  ("id", "name", "framework", "platform", "updated_at", "progress", "tokens", "counts")})
            known = [c["tokens"]["total"] for c in campaigns if c["tokens"]["total"] is not None]
            partial = len(known) != len(campaigns) or any(c["tokens"]["measurement"] != "exact" for c in campaigns)
            return {"campaigns": campaigns, "tokens": {"total": sum(known) if known else None,
                    "measurement": "partial" if partial else "exact", "scope": SCOPE}}

    def campaign(self, identifier):
        with self._lock:
            self._discover()
            return self._snapshot(identifier)

    def episode(self, identifier, number):
        with self._lock:
            self._discover()
            path = self._paths[identifier]
            snapshot = self._snapshot(identifier)
            issues = []
            state = self._read(path / RUNTIME / "state.json", issues, f"{RUNTIME}/state.json")
            records = self._records(path, state, issues)
            attempt = records.get(number)
            active = self._read(path / RUNTIME / "active_episode.json", issues, f"{RUNTIME}/active_episode.json")
            if attempt is None and active.get("episode") != number:
                raise KeyError(number)
            attempt = attempt or {"episode": number, "version": active.get("memory_version"), "mode": active.get("mode")}
            telemetry = self._telemetry(path, number, attempt, issues)
            relative = f"{RUNTIME}/episodes/e{number:04d}/episode_runtime/journal.json"
            journal = self._read(path / relative, issues, relative)
            if not journal and active.get("episode") == number and active.get("worktree"):
                # Resolve only the supervisor-registered active worktree, never an HTTP path.
                journal = self._read(Path(active["worktree"]) / RUNTIME / "journal.json", issues, "active_worktree/journal.json")
            if journal and (journal.get("episode") != number or
                            (journal.get("memory_version") is not None and
                             version(journal["memory_version"]) != version(attempt.get("version"))) or
                            (active.get("episode") == number and any(
                                active.get(key) and journal.get(key) != active[key]
                                for key in ("base_commit", "episode_branch")))):
                issues.append({"file": relative, "code": "journal_identity_mismatch"})
                journal = {}
            phases = obj(telemetry.get("phase_tokens"))
            rows = [{"phase": key, "usage": usage(obj(obj(phases.get("phases")).get(key)).get("usage"))}
                    for key in PHASES]
            rows += [{"phase": key, "usage": usage(phases.get(key))} for key in ("orchestration", "unattributed")]
            verification = obj(attempt.get("verification"))
            summary = {key: verification.get(key) for key in
                       ("gate", "candidate_latency_us", "incumbent_latency_us", "improvement_pct", "error")}
            return {"episode": number, "version": version(attempt.get("version")), "mode": attempt.get("mode"),
                    "status": outcome_status(attempt) or "observed_active", "accepted": attempt.get("accepted"),
                    "summary": attempt.get("summary") or obj(journal.get("outcome")).get("summary"),
                    "violation": attempt.get("violation"), "verification": summary,
                    "tokens": episode_tokens(attempt, telemetry), "phases": rows,
                    "experiments": [experiment(e) for e in sequence(journal.get("experiments"))],
                    "latest_experiment": snapshot["progress"]["latest_experiment"] if active.get("episode") == number else None,
                    "detail_available": bool(journal), "issues": snapshot["issues"] + issues,
                    "evidence": [f"{RUNTIME}/episodes/e{number:04d}/attempt.json",
                                 f"{RUNTIME}/episodes/e{number:04d}/telemetry.summary.json",
                                 relative] if number in records else ["memory/live.json"]}
