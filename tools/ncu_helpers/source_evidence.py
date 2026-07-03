#!/usr/bin/env python3
# Copyright 2026 Alibaba Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One entry point for atrex's source-level NCU evidence.

profile_iter_nvidia.sh calls this once (on a `--source` run) instead of invoking the
individual disasm / warp-stalls / source-metrics tools itself. It runs them,
reuses each tool's own writer, and emits `analysis/source_evidence_manifest.json`
— a single index telling the agent which evidence exists and what each file is
for.

All of this is *independent evidence*: it never feeds classify_ncu.py and never
changes summary.txt. The controlled-vocabulary diagnosis still comes only from
summary.txt; these artifacts answer "which source line / SASS address" once a
symptom is known.

Usage:
    python3 source_evidence.py --run-dir profile/run \\
        --report profile/run/ncu.ncu-rep \\
        --source-report profile/run/ncu_source.ncu-rep --tag run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import disasm  # noqa: E402
import warp_stalls  # noqa: E402
import source_metrics  # noqa: E402


def _jobs(run_dir, report, source_report, tag):
    """Return (main_fn, argv, stem, purpose) for each evidence artifact.

    disasm + warp-stalls read the full report; source-metrics needs the
    per-PC instances from the `--set source` report and is skipped without it.
    """
    common = ["--run-dir", str(run_dir), "--tag", tag]
    jobs = [
        (disasm.main, common + ["--report", str(report)],
         f"disasm_{tag}",
         "structured source-correlated SASS (+PTX when nvdisasm/cuobjdump present)"),
        (warp_stalls.main, common + ["--report", str(report), "--by", "reason", "--self-check"],
         f"warp_stalls_reason_{tag}",
         "warp-stall sample counts by reason"),
        (warp_stalls.main, common + ["--report", str(report), "--by", "line", "--self-check"],
         f"warp_stalls_line_{tag}",
         "warp-stall sample counts by source line"),
    ]
    if source_report is not None:
        jobs += [
            (source_metrics.main, common + ["--report", str(source_report), "--by", "line"],
             f"source_metrics_line_{tag}",
             "per-source-line metric attribution (loads/stores/sectors/atomics)"),
            (source_metrics.main, common + ["--report", str(source_report), "--by", "sass"],
             f"source_metrics_sass_{tag}",
             "per-SASS-address metric attribution"),
        ]
    return jobs


def run(run_dir, report, source_report, tag):
    """Run every evidence job and return the manifest dict."""
    analysis = Path(run_dir) / "analysis"
    evidence = []
    for main_fn, argv, stem, purpose in _jobs(run_dir, report, source_report, tag):
        note = ""
        try:
            rc = main_fn(argv)
        except Exception as e:  # one bad tool must not sink the rest
            rc, note = 1, f"{type(e).__name__}: {e}"[:200]
        json_path = analysis / f"{stem}.json"
        if json_path.exists():
            status = "ok"
        elif rc == 0:
            status = "missing"  # claimed success but wrote nothing
        else:
            status = "error" if note else "skipped"
        evidence.append({
            "name": stem,
            "json": f"analysis/{stem}.json",
            "txt": f"analysis/{stem}.txt",
            "purpose": purpose,
            "status": status,
            "note": note,
        })
    return {
        "schema": "v1",
        "tag": tag,
        "report": str(report),
        "source_report": str(source_report) if source_report else None,
        "evidence": evidence,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True,
                    help="full report (.ncu-rep) for disasm + warp-stalls")
    ap.add_argument("--source-report", type=Path, default=None,
                    help="--set source report for per-line/SASS metric attribution")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args(argv)

    if not args.report.exists():
        print(f"[skip] {args.report} does not exist", file=sys.stderr)
        return 1
    source_report = (args.source_report
                     if args.source_report and args.source_report.exists() else None)

    manifest = run(args.run_dir, args.report, source_report, args.tag)
    out = args.run_dir / "analysis" / "source_evidence_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))

    ok = sum(1 for e in manifest["evidence"] if e["status"] == "ok")
    print(f"source-evidence -> {out} ({ok}/{len(manifest['evidence'])} artifacts ok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
