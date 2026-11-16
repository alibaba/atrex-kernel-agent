"""Summarize observed scalar correctness facts without exposing case inputs."""
from __future__ import annotations

import math


def finite_number(value):
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None



def correctness_summary(value, *, passed: bool) -> dict:
    """Runtime's worst-case scalar summary, never case inputs or raw outputs."""
    aliases = {"rel_err": {"rel_err", "relative_l2"},
               "max_abs_err": {"max_abs_err", "max_elementwise_abs_diff"},
               "max_rel_err": {"max_rel_err", "max_elementwise_rel_diff"}}
    metrics = dict.fromkeys(aliases)
    pending = [value.get("correctness", value) if isinstance(value, dict) else value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if "correctness" in item:
                pending.append(item["correctness"])
                continue
            for key, child in item.items():
                number = finite_number(child)
                if number is not None and number >= 0:
                    for name, names in aliases.items():
                        if key in names:
                            metrics[name] = number if metrics[name] is None else max(metrics[name], number)
                if isinstance(child, (dict, list, tuple)):
                    pending.append(child)
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
    return {"status": "PASS" if passed else "FAIL", **metrics}
