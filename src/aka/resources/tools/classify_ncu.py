#!/usr/bin/env python3
"""Installed CSV-profile entry for Execution-owned symptom classification."""
from aka.execution.profiling.nvidia.ncu_classification import PATTERNS, _get, classify
from aka.execution.profiling.nvidia.classify_ncu import format_summary as _format_summary, main as _main

from aka.execution.profiling.nvidia.classify_ncu import CSV_LOCALIZE_BY_SYMPTOM as LOCALIZE_BY_SYMPTOM


def format_summary(metrics, findings):
    return _format_summary(metrics, findings, localize_by_symptom=LOCALIZE_BY_SYMPTOM)


def main():
    return _main(localize_by_symptom=LOCALIZE_BY_SYMPTOM)


if __name__ == "__main__":
    main()
