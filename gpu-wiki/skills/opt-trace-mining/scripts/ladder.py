#!/usr/bin/env python3
"""Which versions earned a record, and which latency numbers may be believed.

A trace's latency series cannot be read as a progress curve. Steps improve and
regress in roughly equal numbers, and the large excursions are usually
measurement environment -- a cold vendor-library autotune, a shared machine --
which the run itself often diagnoses in prose. So a per-step delta is meaningless
and only a best-so-far ratchet is defensible: a version earns a milestone when it
sets a new record against everything before it.

The thresholds live here rather than in another tree. They decide this store's
record set, so a change to them has to show up as a diff in this file; the
predecessor imported them from a module elsewhere, where a change would silently
alter which versions became records. `self_test()` pins the ratchet's behaviour
on a synthetic non-monotonic series -- run `python3 ladder.py`.
"""
import re

# A step must beat the best-so-far by this much to be worth a record. Below it,
# the "improvement" is within run-to-run spread on every harness measured so far.
MILESTONE_PCT = 1.5
# A step this large is a different kind of event: it usually replaced the
# algorithm rather than tuned it, so it also earns a code snapshot.
MEGA_PCT = 20.0
# A jump larger than this in either direction is the harness re-baselining, not
# the kernel changing.
REBASELINE_FACTOR = 5.0

# A version that says it changed nothing cannot own a speedup: any move across it
# is re-baselining or noise.
NO_CHANGE_RE = re.compile(
    r"NO[_ ]CHANGE|NO[_ ]OP|falsified|flat[- ]within[- ]noise|within noise"
    r"|no actionable lever|no improvement", re.I)

RERUN_RE = re.compile(r"(_rerun\d*|_baseline|_base)$", re.I)
INFRA_RE = re.compile(
    r"infra|sandbox|outage|blocked|not_run|unavailable|no_gpu|machine_down", re.I)


def text_of(v):
    """Everything the run said about this version, in one string."""
    return " ".join(str(x) for x in (
        v.get("subject") or "", v.get("body") or "",
        v.get("action_description") or "", v.get("gate_failure") or "") if x)


def declares_no_change(v):
    return bool(NO_CHANGE_RE.search(text_of(v)))


def owns_kernel_change(v):
    """Whether this version can own a code delta, with legacy compatibility."""
    if "kernel_changed" in v:
        return bool(v.get("kernel_changed"))
    return not declares_no_change(v)


def noise_reason(v, outcome_matters=True):
    """Why this version cannot be reasoned about, or None.

    Two different questions hide here. Infrastructure noise and re-measurements
    mean the version is not a data point at all. A failed correctness check or
    quality gate is different: it only disqualifies the version from *claiming a
    speedup*. For a reverted version that failure is the lesson, so applying the
    outcome filters there would discard exactly the records worth having.
    """
    if RERUN_RE.search(v["version"]):
        return "remeasurement"
    long_horizon_status = v.get("long_horizon_status")
    if long_horizon_status in {"interrupted", "blocked", "invalid_handoff"}:
        return "infrastructure"
    authoritative = (
        v.get("measurement_source") == "authoritative_verification"
        and v.get("measurement_subject") == "candidate"
        and v.get("gate_result") == "PASS"
    )
    passed = (
        v.get("gate_result") == "PASS"
        and v.get("correctness_status") == "PASS"
    )
    if (long_horizon_status is None and not authoritative and not passed
            and INFRA_RE.search(text_of(v))):
        return "infrastructure"
    if not outcome_matters:
        return None
    status = v.get("correctness_status")
    if status and status != "PASS":
        return "incorrect"
    if v.get("gate_result") == "FAIL":
        return "gate-fail"
    return None


def authoritative_gain(v):
    """Same-allocation supervisor gain, when its origin is explicit."""
    value = v.get("authoritative_improvement_pct")
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and v.get("measurement_source") == "authoritative_verification"
        and v.get("measurement_subject") == "candidate"
        and v.get("gate_result") == "PASS"
    ):
        return float(value)
    return None


def select(versions):
    """Return (milestones, deadends, rejected).

    milestones carry `improve_pct` against the best-so-far and a `kind` of
    baseline / milestone / mega / final. deadends are the reverted versions worth
    mining. rejected records why everything else was dropped, so the report can
    be audited instead of trusted.
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
            elif len(text_of(v)) > 80:
                deadends.append(v)
            else:
                # Too short a subject carries no transferable lesson.
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
            # A strategy record must carry payload.implementation, and the
            # snippet must be verbatim from the code, so a milestone with no
            # commit cannot be evidenced at all. The commit hash a step record
            # states about itself is not a usable fallback: it is written
            # independently of the commit and disagrees with git in practice.
            rejected.append((v["version"], "no commit, code not reachable"))
            continue

        authoritative = authoritative_gain(v)
        if authoritative is not None:
            if best is None or geo < best:
                best = geo
            if owns_kernel_change(v) and authoritative >= MILESTONE_PCT:
                milestones.append(dict(
                    v,
                    kind="mega" if authoritative >= MEGA_PCT else "milestone",
                    improve_pct=round(authoritative, 2),
                ))
            elif not owns_kernel_change(v):
                rejected.append((v["version"], "commit does not change kernel.py"))
            else:
                rejected.append((
                    v["version"],
                    "under %.1f%% authoritative verification (%.2f%%)"
                    % (MILESTONE_PCT, authoritative),
                ))
            continue

        if best is None:
            best = geo
            if owns_kernel_change(v):
                milestones.append(dict(v, kind="baseline", improve_pct=None))
            else:
                rejected.append((v["version"], "commit does not change kernel.py"))
            continue

        if geo > best * REBASELINE_FACTOR:
            rejected.append((v["version"], "harness re-baselining (%.0fx worse)"
                             % (geo / best)))
            continue
        if geo * REBASELINE_FACTOR < best and not text_of(v).strip():
            # An unexplained 5x win is an artifact: take the new floor but claim
            # nothing for it.
            best = geo
            rejected.append((v["version"], "unexplained %.0fx win, no text"
                             % (best / geo)))
            continue

        improve = (best - geo) / best * 100.0
        if improve >= MILESTONE_PCT and owns_kernel_change(v):
            milestones.append(dict(
                v, kind="mega" if improve >= MEGA_PCT else "milestone",
                improve_pct=round(improve, 2)))
            best = geo
        else:
            if improve > 0:
                best = geo
            if "kernel_changed" in v and not owns_kernel_change(v):
                rejected.append((v["version"], "commit does not change kernel.py"))
            else:
                rejected.append((v["version"], "under %.1f%% (%.2f%%)"
                                 % (MILESTONE_PCT, improve)))

    if milestones and best is not None:
        # Only a version that actually holds the floor may be called the end
        # state. The last version by number is often a measurement excursion, and
        # labelling that "final" would present it as the run's outcome.
        candidates = [
            v for v in versions
            if v.get("geomean_us")
            and v["geomean_us"] <= best * 1.0001
            and noise_reason(v) is None
            and v.get("sha")
            and owns_kernel_change(v)
            and (authoritative_gain(v) is None
                 or authoritative_gain(v) >= MILESTONE_PCT)
        ]
        if candidates:
            last = max(candidates, key=lambda v: v["n"])
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


def self_test():
    """Pin the ratchet on a synthetic, deliberately non-monotonic series."""
    def ver(n, geo=None, **kw):
        row = {"version": "v%d" % n, "n": n, "geomean_us": geo,
               "has_commit": True, "sha": "%040x" % n, "dsl": "triton"}
        row.update(kw)
        return row

    series = [
        ver(1, 100.0),                                   # baseline
        ver(2, 99.9),                                    # under threshold
        ver(3, 75.0),                                    # mega (>= 20%)
        ver(4, 500.0),                                   # harness re-baselining
        ver(5, 79.0, subject="v5: retune (+1.2%)"),      # under threshold
        ver(6, 60.0, subject="v6: flat-within-noise"),   # claims no change
        ver(7, 58.0),                                    # milestone off 60.0
        ver(8, 95.0, reverted=True,
            subject="v8: reverted (dead-end recorded: shared-memory staging is "
                    "slower because the tile no longer fits in one bank set)"),
        ver(9, None, subject="v9: infra outage, not_run"),
    ]
    milestones, deadends, rejected = select(series)
    got = [(m["version"], m["kind"]) for m in milestones]
    want = [("v1", "baseline"), ("v3", "mega"), ("v7", "milestone")]
    bad = []
    if got != want:
        bad.append("milestones %s, expected %s" % (got, want))
    if [d["version"] for d in deadends] != ["v8"]:
        bad.append("deadends %s, expected ['v8']"
                   % [d["version"] for d in deadends])
    why = dict(rejected)
    for version, fragment in (("v2", "under"), ("v4", "re-baselining"),
                              ("v6", "under"), ("v9", "infrastructure")):
        if fragment not in why.get(version, ""):
            bad.append("%s rejected as %r, expected %r"
                       % (version, why.get(version), fragment))
    # v6 sets the floor without owning it, which is what lets v7 be a milestone
    # against 60.0 rather than against 79.0.
    if not any(m["version"] == "v7" and m["improve_pct"] and m["improve_pct"] < 5
               for m in milestones):
        bad.append("v7 was not measured against the floor v6 set")
    return bad, 4 + len(want)


if __name__ == "__main__":
    import sys

    failures, n = self_test()
    for line in failures:
        print("FAIL %s" % line)
    print("ladder: %d checks, %d failures (MILESTONE_PCT=%.1f MEGA_PCT=%.0f "
          "REBASELINE_FACTOR=%.0f)"
          % (n, len(failures), MILESTONE_PCT, MEGA_PCT, REBASELINE_FACTOR))
    sys.exit(1 if failures else 0)
