#!/usr/bin/env python3
# Copyright 2026 Alibaba Group.
# Licensed under the Apache License, Version 2.0.

"""Inventory long-horizon experiments and their Git attribution.

This report is deliberately descriptive. It exposes structured attempts,
legacy experiments, candidate lineages, and exact commit mentions so the
distiller can select useful records without assigning an episode's total gain
to every probe.
"""
import json
import re

import config as c


def load_json(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


class CommitLineageIndex:
    """Serve every episode range from two batched Git history walks."""

    def __init__(self, ranges):
        tokens = sorted({value for pair in ranges for value in pair if value})
        self._resolved = self._resolve_commits(tokens)
        heads = sorted({
            self._resolved[value]
            for pair in ranges for value in pair if value in self._resolved
        })
        self._parents = {}
        self._kernel_rows = []
        self._ancestor_cache = {}
        if not heads:
            return
        raw = c.git("rev-list", "--parents", *heads)
        for line in raw.splitlines():
            fields = line.split()
            if fields:
                self._parents[fields[0]] = tuple(fields[1:])
        raw = c.git(
            "log", "--topo-order", "--reverse", "--format=%H%x09%s",
            *heads, "--", "kernel.py",
        )
        for line in raw.splitlines():
            sha, sep, subject = line.partition("\t")
            if sep:
                self._kernel_rows.append((sha, subject))

    @staticmethod
    def _resolve_commits(tokens):
        """Resolve and type-check all journal commit tokens in one process."""
        if not tokens:
            return {}
        expressions = ["%s^{commit}" % token for token in tokens]
        raw = c.git(
            "cat-file", "--batch-check=%(objectname) %(objecttype)",
            input_text="\n".join(expressions) + "\n",
        )
        resolved = {}
        for token, line in zip(tokens, raw.splitlines()):
            sha, sep, kind = line.partition(" ")
            if sep and kind == "commit":
                resolved[token] = sha
        return resolved

    def _ancestors(self, sha):
        cached = self._ancestor_cache.get(sha)
        if cached is not None:
            return cached
        seen, pending = set(), [sha]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(self._parents.get(current, ()))
        self._ancestor_cache[sha] = seen
        return seen

    def rows(self, base, candidate):
        base_sha = self._resolved.get(base)
        candidate_sha = self._resolved.get(candidate)
        if not base_sha or not candidate_sha:
            return []
        selected = self._ancestors(candidate_sha) - self._ancestors(base_sha)
        return [row for row in self._kernel_rows if row[0] in selected]


def experiment_id(experiment, index):
    if not isinstance(experiment, dict):
        return "experiment-%d" % index
    return str(experiment.get("id") or experiment.get("name")
               or "experiment-%d" % index)


def mentioned_experiments(experiments, sha, subject):
    token_match = re.match(r"^(v\d+(?:-[A-Za-z0-9]+)?)", subject, re.I)
    token = token_match.group(1) if token_match else ""
    exact, labelled = [], []
    for index, experiment in enumerate(experiments, 1):
        text = (json.dumps(experiment, ensure_ascii=False)
                if isinstance(experiment, dict) else str(experiment))
        if sha[:7].lower() in text.lower():
            exact.append(experiment_id(experiment, index))
        elif token and re.search(
            r"(?<![A-Za-z0-9])%s(?![A-Za-z0-9])" % re.escape(token),
            text, re.I,
        ):
            labelled.append(experiment_id(experiment, index))
    return exact or labelled


def episode_sources():
    """Yield committed episodes, then runtime-only episodes not yet archived."""
    seen = set()
    for path in sorted((c.TRACE / "memory").glob("long_horizon_e*.json")):
        value = load_json(path)
        journal = value.get("journal") if isinstance(value.get("journal"), dict) else {}
        key = (journal.get("episode"), journal.get("candidate_commit"))
        seen.add(key)
        yield path, value, journal, "committed"
    runtime = c.TRACE / ".atrex_long_horizon" / "episodes"
    for path in sorted(runtime.glob("e*/episode_runtime/journal.json")):
        journal = load_json(path)
        key = (journal.get("episode"), journal.get("candidate_commit"))
        if key in seen:
            continue
        yield path, {}, journal, "runtime"


def main():
    c.require_trace()
    c.ensure_dirs()
    lines = ["# Long-horizon recon: %s" % c.TRACE.name, ""]
    total_experiments = total_commits = structured_attempts = 0

    # Scan lightweight addresses first, then render episode payloads. Git
    # history is fetched once for the complete trace.
    ranges = []
    for _path, wrapper, journal, _durability in episode_sources():
        ranges.append((
            str(journal.get("base_commit") or wrapper.get("base_commit") or ""),
            str(journal.get("candidate_commit")
                or wrapper.get("candidate_commit") or ""),
        ))
    lineages = CommitLineageIndex(ranges)

    for path, wrapper, journal, durability in episode_sources():
        experiments = journal.get("experiments") or []
        plan = journal.get("attempt_plan") or {}
        attempts = plan.get("attempts") if isinstance(plan, dict) else []
        attempts = attempts if isinstance(attempts, list) else []
        structured_attempts += len(attempts)
        total_experiments += len(experiments)
        base = str(journal.get("base_commit") or wrapper.get("base_commit") or "")
        candidate = str(journal.get("candidate_commit")
                        or wrapper.get("candidate_commit") or "")
        commits = lineages.rows(base, candidate)
        total_commits += len(commits)
        lines += [
            "## episode %s (%s)" % (journal.get("episode", "?"), durability),
            "",
            "- state: `%s`" % (journal.get("state")
                                or wrapper.get("status") or "unknown"),
            "- experiments: %d" % len(experiments),
            "- structured attempts: %d" % len(attempts),
            "- code commits in candidate lineage: %d" % len(commits),
            "- evidence: `%s`" % path.relative_to(c.TRACE),
            "",
        ]
        if attempts:
            lines += ["### Structured attempts", "",
                      "| id | status | candidate | evidence summary |",
                      "|---|---|---|---|"]
            for attempt in attempts:
                evidence = attempt.get("evidence") or {}
                evidence_summary = (evidence.get("summary") or ""
                                    if isinstance(evidence, dict) else evidence)
                lines.append("| `%s` | %s | `%s` | %s |" % (
                    attempt.get("id", "?"), attempt.get("status", "?"),
                    str(attempt.get("candidate_commit") or "")[:12],
                    str(evidence_summary).replace("|", "\\|")[:240],
                ))
            lines.append("")
        if commits:
            lines += ["### Candidate lineage", "",
                      "| commit | subject | matching experiments |",
                      "|---|---|---|"]
            for sha, subject in commits:
                matches = mentioned_experiments(experiments, sha, subject)
                lines.append("| `%s` | %s | %s |" % (
                    sha[:12], subject.replace("|", "\\|"),
                    ", ".join("`%s`" % value for value in matches) or "none",
                ))
            lines.append("")
    lines += [
        "## Summary", "",
        "- experiments: %d" % total_experiments,
        "- structured attempts: %d" % structured_attempts,
        "- code commits in candidate lineages: %d" % total_commits,
        "",
        "Only a structured accepted attempt with its own candidate commit and measurement may be",
        "split automatically. Legacy free-form experiments require manual review; never assign an",
        "episode-level gain to each commit.",
    ]
    out = c.REPORTS / "long-horizon.md"
    out.write_text("\n".join(lines) + "\n")
    print("wrote %s" % out)
    print("  experiments=%d structured_attempts=%d lineage_commits=%d"
          % (total_experiments, structured_attempts, total_commits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
