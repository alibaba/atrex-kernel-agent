#!/usr/bin/env python3
"""Which versions earned a record, and which latency numbers may be believed.

A run's latency series cannot be read as a progress curve. Measured on these
corpora, improving and regressing steps are close to equal in number, and the
large excursions are measurement-environment effects the runs themselves
diagnosed as such (a cold start, a re-baselined harness, a machine outage). So a
per-step delta means little, and only a **best-so-far ratchet** is defensible: a
version earns a milestone when it sets a new record against everything before it.

Self-contained on purpose. An earlier version imported this logic from another
tree. That coupling was wrong in both directions: a change to its thresholds would
silently redraw this store's record set, and this file's needs (an A/B corpus with
no ladder at all) would push back on a module that has no business knowing about
them. The three thresholds are pinned here with the corpus fact behind each.
"""
import re

# A step must clear this to be worth a record. Below it the runs' own vocabulary
# is "flat within noise", and both archives use that phrase for sub-percent moves.
MILESTONE_PCT = 1.5
# A step this large restructured something; it also earns a reference-kernel
# snapshot where the transcript holds a whole file.
MEGA_PCT = 20.0
# A jump of this factor is the harness changing what it measures, not the kernel
# getting that much slower or faster. Guarding on it stops one re-baselining from
# turning every later version into a "milestone" against the wrong floor.
REBASELINE_FACTOR = 5.0

# A version that says it changed nothing cannot own a speedup: any move across it
# is re-baselining or noise. These are the exact phrases the runs use.
NO_CHANGE_RE = re.compile(
    r"NO[_ ]CHANGE|NO[_ ]OP|falsified|flat[- ]within[- ]noise|within noise"
    r"|no actionable lever|no improvement", re.I)

# A re-measurement of an existing version is not a version. Admitting these puts
# the same version on the ladder twice with different numbers.
RERUN_RE = re.compile(r"(_rerun\d*|_baseline|_base)$", re.I)

# The run could not measure anything, so neither its number nor its absence means
# anything about the code.
INFRA_RE = re.compile(
    r"infra|sandbox|outage|blocked|not_run|unavailable|no_gpu|machine_down", re.I)

# Long enough to carry a transferable lesson. A bare "reverted" does not.
MIN_DEADEND_CHARS = 80


def text_of(v):
    """Everything the run said about this version, in one string."""
    return " ".join(str(x) for x in (
        v.get("subject") or "", v.get("body") or "",
        v.get("action_description") or "", v.get("gate_failure") or "") if x)


def declares_no_change(v):
    return bool(NO_CHANGE_RE.search(text_of(v)))


def noise_reason(v, outcome_matters=True):
    """Why this version cannot be reasoned about, or None.

    Two different questions hide here. Infrastructure noise and re-measurements
    mean the version is not a data point at all. A failed correctness check or
    quality gate is different: it only disqualifies the version from *claiming a
    speedup*. For a reverted version that failure is the lesson, so applying the
    outcome filters there would discard exactly the records worth having.
    """
    if RERUN_RE.search(v.get("version") or ""):
        return "remeasurement"
    if INFRA_RE.search(text_of(v)):
        return "infrastructure"
    if not outcome_matters:
        return None
    status = v.get("correctness_status")
    if status and str(status).upper() != "PASS":
        return "incorrect"
    if str(v.get("gate_result") or "").upper() == "FAIL":
        return "gate-fail"
    return None


def select(versions):
    """Return (milestones, deadends, rejected).

    `milestones` carry `improve_pct` against the best-so-far and a `kind` of
    baseline / milestone / mega / final. `deadends` are the reverted versions
    worth mining. `rejected` records why everything else was dropped, so the
    report can be audited instead of trusted.
    """
    milestones, deadends, rejected = [], [], []
    best = None

    for v in versions:
        if v.get("has_commit") and v.get("reverted"):
            # Judged before the outcome filters: this version's failure is the
            # knowledge, so a failed gate must not disqualify it.
            reason = noise_reason(v, outcome_matters=False)
            if reason:
                rejected.append((v["version"], reason))
            elif len(text_of(v)) > MIN_DEADEND_CHARS:
                deadends.append(v)
            else:
                rejected.append((v["version"], "reverted, nothing said"))
            continue

        reason = noise_reason(v)
        if reason:
            rejected.append((v["version"], reason))
            continue

        geo = v.get("geomean_us")
        if not geo or geo <= 0:
            rejected.append((v["version"], "no usable geomean"))
            continue

        if not v.get("sha"):
            # A strategy record is schema-required to carry an implementation and
            # the snippet must be verbatim, so a milestone whose code is not
            # reachable cannot be evidenced at all. The `git_commit_hash` a memory
            # file reports about itself is not a usable substitute: it is written
            # independently of the commit and disagrees with git in this corpus.
            rejected.append((v["version"], "no commit, code not reachable"))
            continue

        if best is None:
            milestones.append(dict(v, kind="baseline", improve_pct=None))
            best = geo
            continue

        if geo > best * REBASELINE_FACTOR:
            rejected.append((v["version"], "harness re-baselining (%.0fx worse)"
                             % (geo / best)))
            continue
        if geo * REBASELINE_FACTOR < best and not text_of(v).strip():
            # An unexplained 5x win is an artifact; take the new floor but claim
            # nothing for it.
            best = geo
            rejected.append((v["version"], "unexplained %.0fx win, no text"
                             % (best / geo)))
            continue

        improve = (best - geo) / best * 100.0
        if improve >= MILESTONE_PCT and not declares_no_change(v):
            milestones.append(dict(
                v, kind="mega" if improve >= MEGA_PCT else "milestone",
                improve_pct=round(improve, 2)))
            best = geo
        else:
            if improve > 0:
                best = geo
            rejected.append((v["version"], "under %.1f%% (%.2f%%)"
                             % (MILESTONE_PCT, improve)))

    if milestones and best is not None:
        # Only a version that actually holds the floor may be called the end
        # state. The last version by number is often a measurement excursion, and
        # labelling that "final" would present an artifact as the run's outcome.
        holders = [v for v in versions
                   if v.get("geomean_us") and v["geomean_us"] <= best * 1.0001]
        if holders:
            last = max(holders, key=lambda v: v.get("n") or 0)
            if last["version"] not in {m["version"] for m in milestones}:
                milestones.append(dict(last, kind="final", improve_pct=None))

    return milestones, deadends, rejected


def running_min(versions):
    """The best-so-far series, for the report. Not a selection."""
    out, best = [], None
    for v in versions:
        geo = v.get("geomean_us")
        if not geo or geo <= 0:
            continue
        if best is None or geo < best:
            best = geo
            out.append((v["version"], geo))
    return out
