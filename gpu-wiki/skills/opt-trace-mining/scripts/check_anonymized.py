#!/usr/bin/env python3
"""Read-side mirror of the store's anonymisation gate.

`anonymize.py` removes leak shapes; this reports the ones that survived. The
patterns are IMPORTED from `tools/check_kernel_wiki.py` rather than restated,
because a pre-flight that disagrees with the gate is worse than no pre-flight:
it declares clean what the store will reject, and the packet is regenerated only
after a whole distillation batch has already been written against it.

Two entry points, both used by the pipeline:

  scan_text(text)     any prose the agent will be allowed to read (packets)
  scan_record(record) exactly the layers the gate scans (payload, retrieval,
                      evidence.summary, worth.gain), via the checker's own
                      projection, so `evidence.raw` is correctly exempt

Usage:
  python3 check_anonymized.py <file-or-dir> [...]      # .json records, or text
"""
import json
import re
import sys
from pathlib import Path

import config as c

_checker = None


def checker():
    global _checker
    if _checker is None:
        _checker = c.load_store_checker()
    return _checker


def patterns():
    """The gate's own leak patterns, plus the private denylist if one is set."""
    ck = checker()
    out = dict(ck.LEAK_PATTERNS)
    terms = sorted(ck.private_denylist(), key=len, reverse=True)
    if terms:
        out["private denylist"] = re.compile(
            "|".join(re.escape(t) for t in terms), re.I)
    return out


def scan_text(text, allow_version_widths=True):
    """Return [(label, token)] for every leak shape in `text`."""
    hits = []
    if not text:
        return hits
    for label, rx in patterns().items():
        for m in rx.finditer(text):
            token = m.group(0)
            # The gate makes this one allowance, so the pre-flight must too:
            # `v4` beside `vec`/`float`/`.f16` is a vector width.
            if allow_version_widths and label == "version id" and re.match(
                    r"^[vV](?:2|4|8|16|32|64|100)$", token):
                ctx = text[max(0, m.start() - 12):m.end() + 12]
                if re.search(r"vec|\.f16|\.bf16|x2|float", ctx, re.I):
                    continue
            hits.append((label, token))
    return hits


def scan_record(record):
    """Scan one record exactly the way the store's gate does."""
    ck = checker()
    prose, code = ck.agent_text(record)
    return scan_text(prose + "\n" + ck.comments_only(code))


def _scan_path(path, out):
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            out.append((str(path), [("unreadable", str(exc)[:60])]))
            return
        if isinstance(data, dict) and "payload" in data and "evidence" in data:
            hits = scan_record(data)
        else:
            hits = scan_text(json.dumps(data, ensure_ascii=False))
    else:
        hits = scan_text(path.read_text(errors="replace"))
    if hits:
        out.append((str(path), hits))


def main(argv):
    if not argv:
        print(__doc__.strip())
        return 2
    findings, n = [], 0
    for arg in argv:
        root = Path(arg)
        paths = ([p for p in sorted(root.rglob("*")) if p.is_file()]
                 if root.is_dir() else [root])
        for path in paths:
            if path.name == "index.json":
                continue
            n += 1
            _scan_path(path, findings)

    print("scanned %d file(s) with %d pattern(s)" % (n, len(patterns())))
    for path, hits in findings:
        print("  %s" % path)
        for label, token in hits[:6]:
            print("      %-18s %s" % (label, token[:80]))
        if len(hits) > 6:
            print("      ... %d more" % (len(hits) - 6))
    if findings:
        print("LEAKS in %d file(s)" % len(findings))
        return 1
    print("clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
