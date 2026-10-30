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

"""Minimal v1 JSON envelope, byte-compatible with VeloQ's `ncu` contract.

The atrex source-metrics / warp-stalls / disasm tools emit this shape so the
artifacts are self-describing and so a future swap to the `veloq` binary (if it
ever lands on a host) is a drop-in replacement rather than a reparse.

Envelope shape (mirrors crates/veloq-core/src/envelope.rs):

    {
      "schema": "v1",
      "source": {"kind": "ncu", "version": "v1"},
      "command": "ncu.source-metrics",
      "trace": {"kind": "ncu", "path": "<report>"},
      "data": {
        "count": <int>,            # rows after --limit
        "total_matched": <int>,    # rows before --limit
        "rows": [ ... ],
        "auxiliary": { ... }
      }
    }

This module deliberately has no third-party deps — stdlib json only.
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = "v1"


def envelope(command, report_path, *, count, total_matched, rows, auxiliary,
             axis=None):
    """Build a v1 envelope dict.

    `command` is "<source>.<verb>" e.g. "ncu.source-metrics". `axis`, when
    given, is surfaced inside `data` (VeloQ puts the axis on the response for
    source-metrics / warp-stalls).
    """
    data = {
        "count": int(count),
        "total_matched": int(total_matched),
        "rows": rows,
        "auxiliary": auxiliary,
    }
    if axis is not None:
        # match VeloQ: axis sits alongside count/rows in `data`
        data = {"axis": axis, **data}
    return {
        "schema": SCHEMA_VERSION,
        "source": {"kind": "ncu", "version": "v1"},
        "command": command,
        "trace": {"kind": "ncu", "path": str(report_path)},
        "data": data,
    }


def sorted_counter_map(d):
    """Return a dict ordered by key ascending, mirroring Rust BTreeMap JSON.

    Values may be None (VeloQ `Option<f64>` -> JSON null): a counter present in
    the matched set but absent on this row is null, distinct from 0.0.
    """
    return {k: d[k] for k in sorted(d)}


def write_json(path, obj):
    """Write `obj` as pretty JSON with keys NOT reordered (callers pre-sort the
    maps that must mirror BTreeMap). Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str))
    return p
