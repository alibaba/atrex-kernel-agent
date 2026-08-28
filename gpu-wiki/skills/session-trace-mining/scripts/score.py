#!/usr/bin/env python3
"""The ranking model: how a record's `worth.rank.score` is computed.

Self-contained on purpose. An earlier version imported the wiki's shared ranking
model (`tools/wiki_score.py`), which was the right instinct -- scores should be
comparable across stores -- but the wrong mechanism: it made this skill's output
depend on a module in another tree, and a change there would silently re-rank this
store with no diff to show for it.

So the formula is implemented here, and **reproduces the shared model exactly for
the inputs this corpus produces**. That is checkable rather than asserted:
`self_test()` pins the published curve at three points, and the constants below
carry the same values `tools/wiki_score.py` publishes in its own formula string.

What this corpus actually has is narrow, and the implementation says so rather
than carrying dead branches:

  strategy        one number -- the step gain in percent
  anti-strategy   negative evidence, so a mid-range base rather than a gain curve
  counters        all zero: a fresh store has no feedback and no telemetry yet

The feedback and recency terms are implemented because they are three lines and
they are what makes a score move once `ingest_feedback` starts reporting outcomes;
everything the shared model derives from corpus-wide statistics we do not have
(`corpus_kept_ratio`, `rediscovered`, `n_operators`) is deliberately absent, and a
prior carrying those keys is reported rather than half-used.
"""
import math

# Published by the shared model in its own formula string, copied so a reader can
# check them against it:
#   clamp(prior + 0.35*signed_feedback*confidence + usage_bonus) * recency(90d)
FEEDBACK_BAND = 0.35
HALF_LIFE_DAYS = 90.0
USAGE_CEILING = 0.05          # usage is evidence of relevance, not of correctness
USAGE_SATURATION = 50.0       # queries at which the usage bonus is fully earned
DAY = 86400.0

# A gain curve rather than a linear scale: the difference between 2% and 4% matters
# more than between 60% and 62%. log1p(gain)/log1p(100) puts 4% at 0.35, 20% at
# 0.66 and 99% at 1.0.
GAIN_SATURATION_PCT = 100.0

# Negative evidence starts mid-range: a trap exactly one run hit is still real
# knowledge and must not rank near zero.
ANTI_BASE = 0.45
# Nothing to go on: mid-low, still reachable.
UNKNOWN_BASE = 0.30

# Feedback needs to be believed before it moves anything, so the band is scaled by
# how many outcomes have been observed.
CONFIDENCE_N = 8.0

KNOWN_PRIOR_KEYS = {"step_gain_pct", "n_independent_runs", "best_speedup_x"}


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def prior_score(prior, record_type):
    """Belief derived from the corpus, in [0, 1].

    Branching on type rather than sharing one formula, because the same field
    means different things: a low gain makes a strategy less credible and says
    nothing about whether an anti-strategy's warning is worth reading.
    """
    prior = prior or {}
    if record_type == "anti-strategy":
        return ANTI_BASE
    if record_type == "reference-kernel":
        # The operator's speedup belongs to the operator, not to this file, so it
        # only nudges. Reference code is steadily useful, never top-ranked.
        speedup = prior.get("best_speedup_x")
        bonus = 0.0
        if speedup and speedup > 1.0:
            bonus = 0.15 * _clamp(math.log(speedup) / math.log(300.0))
        return _clamp(0.45 + bonus)

    gain = prior.get("step_gain_pct")
    if gain is None:
        return UNKNOWN_BASE
    # A regression is not a gain. It reaches here only when a record was typed as
    # a strategy while reporting a negative delta, which the gates forbid, so
    # treat it as no information rather than as a large negative.
    if gain <= 0:
        return UNKNOWN_BASE
    weighted = [(_clamp(math.log1p(gain) / math.log1p(GAIN_SATURATION_PCT)), 1.5)]
    runs = prior.get("n_independent_runs")
    if runs and runs > 1:
        weighted.append((_clamp(math.log(runs) / math.log(4.0)), 0.5))
    return _clamp(sum(v * w for v, w in weighted)
                  / sum(w for _v, w in weighted))


def feedback_term(counters):
    """(signed, confidence): which way the evidence points, and how much there is.

    Deliberately symmetric and bounded. `verified_effective` means "this record
    was believed and it held up", whichever direction the record argues, so an
    anti-strategy confirmed twice moves the same way a strategy adopted twice
    does; the writer guarantees that meaning and the formula does not flip signs.

    `fallback_served` is excluded from the observation count on purpose: it is the
    exploration channel, and letting it build confidence would make a record that
    was only ever shown as a fallback reinforce itself.
    """
    counters = counters or {}
    good = float(counters.get("verified_effective") or 0)
    bad = float(counters.get("verified_ineffective") or 0)
    n = good + bad
    if n <= 0:
        return 0.0, 0.0
    signed = (good - bad) / n
    confidence = _clamp(n / CONFIDENCE_N)
    return signed, confidence


def usage_bonus(counters):
    counters = counters or {}
    used = max(float(counters.get("served_count") or 0),
               float(counters.get("query_count") or 0))
    return USAGE_CEILING * _clamp(math.log1p(used)
                                  / math.log(USAGE_SATURATION))


def recency(counters, now_ts=None):
    """Multiplier in (0.5, 1]. Never zero: old knowledge is not wrong."""
    counters = counters or {}
    last = counters.get("last_used_ts")
    if not last or not now_ts or now_ts <= last:
        return 1.0
    age_days = (now_ts - last) / DAY
    return 0.5 + 0.5 * math.pow(0.5, age_days / HALF_LIFE_DAYS)


def compute(prior, counters, record_type, now_ts=None):
    """Full score with its decomposition, so a ranking can be justified."""
    base = prior_score(prior, record_type)
    signed, confidence = feedback_term(counters)
    adjust = FEEDBACK_BAND * signed * confidence
    usage = usage_bonus(counters)
    decay = recency(counters, now_ts)
    value = _clamp((base + adjust + usage) * decay)
    unknown = sorted(set(prior or {}) - KNOWN_PRIOR_KEYS)
    return {
        "value": round(value, 4),
        "formula": ("clamp(prior + %.2f*signed_feedback*confidence + usage_bonus)"
                    " * recency_decay(half_life=%dd)"
                    % (FEEDBACK_BAND, int(HALF_LIFE_DAYS))),
        "components": {
            "prior": round(base, 4),
            "feedback_signed": round(signed, 4),
            "feedback_confidence": round(confidence, 4),
            "feedback_adjust": round(adjust, 4),
            "usage_bonus": round(usage, 4),
            "recency": round(decay, 4),
        },
        # Reported rather than silently ignored: a prior key this model does not
        # understand means the ingest side learned something the ranking has not.
        "unused_prior_keys": unknown,
    }


def tier_for(score, record_type, basis):
    """Credibility class. Same four names the committed stores use.

    A session-derived record cannot reach `proven` on its own numbers: that
    requires a result clearly better than the technique's own norm across
    independent runs, which one archive cannot establish. Anti-strategies are
    `cautionary` by construction -- they are negative evidence.
    """
    if record_type == "anti-strategy":
        return "cautionary"
    if basis == "measured" and score >= 0.45:
        return "promising"
    if basis in ("measured", "reported"):
        return "promising" if score >= 0.35 else "provisional"
    return "provisional"


def self_test():
    """Pin the published curve. Run by `python3 score.py`.

    The three strategy points are the shared model's own documented values, so a
    drift between this implementation and it shows up here rather than as a
    silently re-ranked store.
    """
    checks = [
        ({"step_gain_pct": 4.0}, "strategy", 0.35),
        ({"step_gain_pct": 20.0}, "strategy", 0.66),
        ({"step_gain_pct": 99.0}, "strategy", 1.00),
        ({"step_gain_pct": 71.23}, "strategy", 0.9274),
        ({"step_gain_pct": 11.13}, "strategy", 0.5408),
        ({"step_gain_pct": None}, "anti-strategy", 0.45),
    ]
    bad = []
    for prior, rtype, want in checks:
        got = compute(prior, {}, rtype)["value"]
        if abs(got - want) > 0.01:
            bad.append("%s %s -> %.4f, expected %.4f"
                       % (rtype, prior, got, want))
    # Feedback must be bounded by the band, whatever the counters say.
    loud = compute({"step_gain_pct": 50.0},
                   {"verified_effective": 99, "verified_ineffective": 0},
                   "strategy")["value"]
    quiet = compute({"step_gain_pct": 50.0}, {}, "strategy")["value"]
    if loud - quiet > FEEDBACK_BAND + 0.06:
        bad.append("feedback moved the score by %.3f, band is %.2f"
                   % (loud - quiet, FEEDBACK_BAND))
    return bad


if __name__ == "__main__":
    import sys
    failures = self_test()
    for line in failures:
        print("FAIL %s" % line)
    print("score: 7 checks, %d failures" % len(failures))
    sys.exit(1 if failures else 0)
