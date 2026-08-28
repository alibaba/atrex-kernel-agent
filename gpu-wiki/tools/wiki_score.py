#!/usr/bin/env python3
"""Importance scoring for the kernel-experience store.

Shared by build_kernel_records.py (cold start) and rebuild_importance.py (after
feedback). Kept in one place so a record's score can never depend on which
script last touched it.

Three properties the ranking must have, and how they are obtained:

* cold start -- a record nobody has queried yet must not sink. The prior is
  built from the corpus itself (how often the technique survived, how large the
  win was, how many independent runs reproduced it), so a fresh record starts
  from what the evidence already says about it.
* no rich-get-richer -- feedback moves the score inside a bounded band around
  the prior. A record cannot climb forever just because it keeps being served.
* failure is informative -- for an anti-strategy, "steered the agent away" is
  the success event, so the same counters are read with the opposite sign.
"""
from __future__ import annotations

import math

BUILDER_VERSION = "1.1"

HALF_LIFE_DAYS = 90.0
DAY = 86400.0

# How far feedback may move a record away from its prior. 0.35 means a record
# with a strong prior and terrible feedback still outranks a weak-prior record
# with no feedback, which is the intended conservatism: the corpus measured
# these numbers, a handful of online trials did not.
FEEDBACK_BAND = 0.35

# Below this many adoption outcomes the feedback term is faded in linearly, so
# one lucky success cannot promote a record.
CONFIDENCE_N = 8.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _stretch_kept(kept: float) -> float:
    """Map an observed success rate onto [0, 1].

    The corpus range is roughly 0.16 (cache modifiers) to 0.75 (compile-time
    division), so a raw ratio would compress everything into the lower half.
    The divisor is deliberately loose enough that the best technique lands near
    0.9 rather than pinned at the ceiling.
    """
    return _clamp((kept - 0.10) / 0.70)


def prior_score(prior: dict, record_type: str) -> float:
    """Belief derived from the corpus, in [0, 1].

    Branching on type rather than sharing one formula, because the same field
    means different things: a low success rate makes a strategy less credible
    and an anti-strategy more credible, and a large operator-level speedup says
    nothing about how important one reference source file is.
    """
    kept = prior.get("corpus_kept_ratio")
    parts: list[tuple[float, float]] = []      # (value, weight)

    if record_type == "anti-strategy":
        # Negative evidence is the point of the pitfalls corpus, so the baseline
        # is mid-range and the extra signals only add. A trap that exactly one
        # run hit is still real knowledge and must not be scored near zero.
        base = 0.45
        redisc = prior.get("rediscovered") or 1
        if redisc > 1:
            # rediscovered 2 -> +0.10, 5 -> +0.22, 10+ -> +0.30
            parts.append((_clamp(math.log(redisc) / math.log(10.0)), 0.30))
        if (prior.get("n_operators") or 1) > 1:
            parts.append((_clamp(math.log(prior["n_operators"]) / math.log(5.0)), 0.20))
        if kept is not None:
            # A lever that usually fails makes the warning more valuable.
            parts.append((_clamp(1.0 - _stretch_kept(kept)), 0.25))
        return _clamp(base + sum(v * w for v, w in parts))

    if record_type == "technique-card":
        # A card is worth reading regardless of whether the lever works: "this
        # fails 84% of the time" is as actionable as "this usually works", so the
        # bonus tracks decisiveness (distance from a coin flip) rather than
        # success. Scoring it by success rate would confuse how good the lever is
        # with how useful the card is.
        if kept is None:
            return 0.55
        return _clamp(0.55 + 0.30 * _clamp(abs(kept - 0.5) / 0.35))

    if record_type == "symptom-card":
        # RELATIONS.md defines patterns/ as the entry point of the whole wiki, so
        # a symptom card starts high; the count of operators where this really
        # was the measured bottleneck is the evidence on top.
        seen = prior.get("n_operators") or 0
        return _clamp(0.55 + 0.25 * _clamp(math.log1p(seen) / math.log(11.0)))

    if record_type == "reference-kernel":
        # The operator's speedup belongs to the operator, not to this file, so it
        # only nudges. Reference code is steadily useful, never top-ranked.
        speedup = prior.get("best_speedup_x")
        bonus = 0.0
        if speedup and speedup > 1.0:
            bonus = 0.15 * _clamp(math.log(speedup) / math.log(300.0))
        return _clamp(0.45 + bonus)

    if record_type == "doc":
        # Unmeasured reference prose: useful, but it claims no result. It rises
        # once telemetry shows agents actually reading it.
        return 0.40

    # strategy, numerics-rule, dispatch-rule
    gain = prior.get("step_gain_pct")
    if gain is not None:
        # 4% -> 0.35, 20% -> 0.66, 99% -> 1.0
        parts.append((_clamp(math.log1p(max(0.0, gain)) / math.log1p(100.0)), 1.5))
    if kept is not None:
        parts.append((_stretch_kept(kept), 1.0))
    runs = prior.get("n_independent_runs")
    if runs:
        parts.append((_clamp(math.log(max(1, runs)) / math.log(4.0)), 0.5))
    if not parts:
        return 0.30    # nothing to go on: mid-low, still reachable
    return _clamp(sum(v * w for v, w in parts) / sum(w for _v, w in parts))


def decay(last_used_ts: float | None, now_ts: float | None) -> float:
    """Recency multiplier in (0, 1]. Never zero: old knowledge is not wrong."""
    if not last_used_ts or not now_ts or now_ts <= last_used_ts:
        return 1.0
    age_days = (now_ts - last_used_ts) / DAY
    return 0.5 + 0.5 * math.pow(0.5, age_days / HALF_LIFE_DAYS)


def reference_rate(prior: dict, record_type: str) -> float | None:
    """The neutral point for reading this record's counters.

    Must share denominators with the counters, so the ladder ratio wins over the
    corpus-wide success rate when it is available. For an anti-strategy the
    counters mean the opposite thing, so the reference flips: how often this
    lever normally fails.
    """
    reference = prior.get("corpus_ladder_ratio")
    if reference is None:
        reference = prior.get("corpus_kept_ratio")
    if reference is None:
        return None
    if record_type == "anti-strategy":
        return 1.0 - reference
    return reference


def feedback_term(counters: dict, record_type: str,
                  base_rate: float | None = None) -> tuple[float, float]:
    """Return (signed_strength in [-1, 1], confidence in [0, 1]).

    Orientation is the writer's responsibility, not this function's:
    verified_effective always means "believing this record was the right call".
    For a strategy that is "adopted and the version was kept"; for an
    anti-strategy it is "the warning held, the direction really was a dead end".
    ingest_feedback.py maps corpus outcomes accordingly, so both types share one
    formula here without a hidden sign flip.

    The neutral point is base_rate, NOT 0.5. In this corpus roughly twenty
    attempts are rejected for every one retained, so scoring a raw success ratio
    against 0.5 would push every warning to the ceiling and every strategy to
    the floor purely because of that base rate. Measuring the deviation from how
    the same lever usually behaves is the actual signal.
    """
    good = float(counters.get("verified_effective") or 0)
    bad = float(counters.get("verified_ineffective") or 0)
    n = good + bad
    if n <= 0:
        return 0.0, 0.0
    posterior = (good + 1.0) / (n + 2.0)          # Beta(1,1) prior
    reference = 0.5 if base_rate is None else _clamp(base_rate, 0.05, 0.95)
    # Normalize by the room available on the side the deviation falls, so a
    # lever with a 16% base rate can still register a strong positive.
    room = (1.0 - reference) if posterior >= reference else reference
    signed = (posterior - reference) / max(room, 1e-6)
    confidence = _clamp(n / CONFIDENCE_N)
    return _clamp(signed, -1.0, 1.0), confidence


def compute(prior: dict, counters: dict, record_type: str,
            now_ts: float | None = None) -> dict:
    """Full score with a decomposition, so --explain can justify a ranking."""
    base = prior_score(prior, record_type)
    base_rate = reference_rate(prior, record_type)
    signed, confidence = feedback_term(counters, record_type, base_rate)
    adjust = FEEDBACK_BAND * signed * confidence
    recency = decay(counters.get("last_used_ts"), now_ts)

    # Usage is evidence of relevance, not of correctness, so it gets a small
    # bounded bonus and is deliberately sub-linear. served_count is the better
    # signal but page-level telemetry can only supply query_count, so take
    # whichever is available.
    used = max(float(counters.get("served_count") or 0),
               float(counters.get("query_count") or 0))
    usage = 0.05 * _clamp(math.log1p(used) / math.log(50.0))

    value = _clamp((base + adjust + usage) * recency)
    return {
        "value": round(value, 4),
        "formula": ("clamp(prior + %.2f*signed_feedback*confidence + usage_bonus) "
                    "* recency_decay(half_life=%dd)" % (FEEDBACK_BAND, int(HALF_LIFE_DAYS))),
        "components": {
            "prior": round(base, 4),
            "base_rate": round(base_rate, 4) if base_rate is not None else -1.0,
            "feedback_signed": round(signed, 4),
            "feedback_confidence": round(confidence, 4),
            "feedback_adjust": round(adjust, 4),
            "usage_bonus": round(usage, 4),
            "recency": round(recency, 4),
        },
        "computed_at": None,
        "builder_version": BUILDER_VERSION,
    }


TIERS = ("proven", "promising", "provisional", "cautionary")

# Thresholds on feedback_term's output, not on raw counts. Comparing good against
# bad directly would be the base-rate trap all over again: this corpus rejects
# roughly twenty attempts per retained step, so "reverted more often than kept"
# is true of almost every lever and says nothing about this record. What matters
# is the deviation from how the same lever normally behaves.
TIER_SIGNAL = 0.5        # half the room available on that side of the reference
TIER_CONFIDENCE = 0.5    # half of CONFIDENCE_N outcomes, so 4 observations


def tier(record_type: str, gain: dict | None, counters: dict,
         base_rate: float | None = None) -> str:
    """Coarse standing, in one word.

    Exists because worth.track is not served: the agent has no counters to read,
    so "how far should I trust this" has to be answered here rather than left to
    it to re-derive. The score alone cannot answer it -- 0.66 is a well-evidenced
    anti-strategy and a mediocre strategy.
    """
    if record_type == "anti-strategy":
        # Negative evidence is never something to adopt, however well confirmed.
        return "cautionary"
    signed, confidence = feedback_term(counters, record_type, base_rate)
    decided = confidence >= TIER_CONFIDENCE
    if decided and signed <= -TIER_SIGNAL:
        return "cautionary"
    if (gain or {}).get("basis") != "measured":
        return "provisional"
    if decided and signed >= TIER_SIGNAL:
        return "proven"
    return "promising"


def rank(prior: dict, counters: dict, record_type: str, gain: dict | None,
         now_ts: float | None = None) -> dict:
    """The worth.rank block: the sort key, the tier, and the audit trail."""
    scored = compute(prior, counters, record_type, now_ts)
    return {
        "score": scored.pop("value"),
        "tier": tier(record_type, gain, counters,
                     reference_rate(prior, record_type)),
        **scored,
    }


def empty_counters() -> dict:
    return {
        "query_count": 0,
        "served_count": 0,
        "applied_count": 0,
        "verified_effective": 0,
        "verified_ineffective": 0,
        "fallback_served": 0,
        "first_seen_ts": None,
        "last_used_ts": None,
    }
