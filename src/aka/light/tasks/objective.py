"""One objective drives admission, comparison and Agent instructions."""
from __future__ import annotations
import math
from aka.contracts.task import Assessment


class MetricObjective:
    def __init__(self, config):
        if set(config) - {"metrics", "constraints", "ordering"}:
            raise ValueError("Unknown objective fields")
        self.metrics = config.get("metrics", [])
        self.constraints = config.get("constraints", [])
        self.ordering = config.get("ordering", "lexicographic")
        if self.ordering not in {"lexicographic", "pareto"} or not self.metrics:
            raise ValueError("Objective requires metrics and lexicographic or pareto ordering")
        for item in [*self.metrics, *self.constraints]:
            if set(item) - {"name", "unit", "context", "direction", "limit", "operator", "tolerance"} or not item.get("name") or not item.get("unit"):
                raise ValueError("Objective metrics require explicit names and units")
            for key in ("limit", "tolerance"):
                if key in item and (type(item[key]) not in (int, float) or not math.isfinite(item[key])):
                    raise ValueError("Objective bounds must be finite numbers")
        for item in self.metrics:
            if item.get("direction") not in {"minimize", "maximize"} or item.get("tolerance", 0) < 0:
                raise ValueError("Objective direction/tolerance is invalid")
        for item in self.constraints:
            if item.get("operator") not in {"le", "ge"} or "limit" not in item:
                raise ValueError("Constraints require le/ge and a limit")

    @staticmethod
    def value(assessment, item):
        for metric in assessment.metrics:
            if metric.name == item["name"] and metric.context == item.get("context", ""):
                if metric.unit != item["unit"]:
                    raise ValueError("Objective metric unit mismatch")
                return metric.value
        raise ValueError(f"Missing objective metric: {item['name']}")

    def eligible(self, assessment: Assessment):
        if assessment.verdict != "pass" or not assessment.complete:
            return False
        try:
            for item in self.metrics:
                self.value(assessment, item)
            for item in self.constraints:
                value = self.value(assessment, item)
                if item["operator"] == "le" and value > item["limit"] or item["operator"] == "ge" and value < item["limit"]:
                    return False
        except ValueError:
            return False
        return True

    def compare(self, candidate, incumbent):
        if (not self.eligible(candidate) or not self.eligible(incumbent) or not candidate.environment
                or candidate.environment != incumbent.environment):
            return "incomparable"
        signs = []
        for item in self.metrics:
            delta = self.value(candidate, item) - self.value(incumbent, item)
            if item["direction"] == "maximize":
                delta = -delta
            signs.append(0 if abs(delta) <= item.get("tolerance", 0) else (1 if delta > 0 else -1))
        if self.ordering == "lexicographic":
            sign = next((s for s in signs if s), 0)
        elif min(signs) < 0 < max(signs):
            return "incomparable"
        else:
            sign = next((s for s in signs if s), 0)
        return {-1: "better", 0: "equivalent", 1: "worse"}[sign]

    def describe(self):
        from .content import document
        return "Optimize " + document({"ordering": self.ordering, "metrics": self.metrics, "constraints": self.constraints})
