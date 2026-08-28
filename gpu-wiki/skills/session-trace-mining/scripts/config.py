#!/usr/bin/env python3
"""The corpora under study, scratch space, and where the records land.

Three distinct places:

  corpus    a *set* of session transcripts, read-only. Named, not passed as a
            path, because a record cites its set by name and the archive can be
            copied anywhere without invalidating a single record.
  scratch   STM_WORKSPACE: parsed events, candidates, packets. All reproducible
            from the corpus, so none of it belongs in the wiki.
  product   kernel_wiki/session_trace/<set>/: records plus the reports that
            justify which candidates were selected. This is what gets reviewed
            and committed.

  STM_SET         which set to work on (required by most scripts)
  STM_ROOT        the archive root that SETS paths are resolved against. There
                  is no useful default: point it at your own transcript archive.
  STM_WORKSPACE   scratch root (default /tmp/session-trace-mining/<set>)
  STM_STORE       override the product root

No transcripts ship with this repository, so `SETS` below holds a single worked
EXAMPLE entry with placeholder values. Replace it with your own sets before
running the pipeline.
"""
import os
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
# The skill lives at <wiki-root>/skills/<name>, so the wiki root is two levels
# up. Self-locating rather than hard-coded, so the tree can be moved.
#
# This is the only path outside the skill that anything here needs, and it is used
# for *data* reads plus one write, never to import logic: the product goes to
# `<wiki-root>/kernel_wiki/session_trace/<set>/`, and the dedup scan reads the
# committed record store to find work it already covers. Naming, milestone
# selection and scoring are implemented in `families.py`, `ladder.py` and
# `score.py` so that no module in another tree can silently change this store's
# output.
WIKI_ROOT = SKILL_DIR.parent.parent

# The committed kernel-experience store, read-only. Data only: the overlap gate
# resolves record ids, episode keys and (operator, version) pairs against it. A
# scan that finds no records is reported as a gate failure rather than passing
# quietly -- an overlap gate with an empty index is worse than no gate at all.
COMMITTED_STORES = (WIKI_ROOT / "kernel_wiki" / "records",)

SCHEMA_DIR = SKILL_DIR / "assets" / "schema"
BASE_NAME = "clean-1.3"
BASE_SCHEMA = SCHEMA_DIR / "clean-1.3.frozen.json"
DERIVED_NAME = "session-trace-1.0"
DERIVED_SCHEMA = SCHEMA_DIR / "session-trace-1.0.schema.json"
# The pinned base is a byte copy of this repository's kernel schema. Set
# STM_SCHEMA_BASE to re-pin from somewhere else.
UPSTREAM_SCHEMA = Path(os.environ["STM_SCHEMA_BASE"]) \
    if os.environ.get("STM_SCHEMA_BASE") else \
    WIKI_ROOT / "schema" / "kernel" / "schema.json"

# Deliberately not a path that could resolve by accident: a run against the wrong
# tree would produce records citing files nobody else can re-resolve.
ARCHIVE_ROOT = Path(os.environ.get("STM_ROOT")
                    or "/set-STM_ROOT-to-your-transcript-archive")

# ----------------------------------------------------------------------- sets

# `unit` is the decision from the Phase-0 probe (references/lessons.md #1, #2)
# and is not negotiable per-run: it is what ids and pairing key on.
#
#   version-ladder  the run keeps a numbered ladder (memory/vN.json + `vN:`
#                   commit subjects). One candidate per version, assembled
#                   across the whole set.
#   ab-comparison   no ladder. One candidate per measured A/B inside one
#                   monotone region.
#
# `product`/`dsl` are defaults, not assertions: ingest detects both from the
# transcripts and only falls back to these, recording which happened in
# `arch_basis`/`dsl_basis`. A set whose hardware cannot be detected and has no
# default fails loudly rather than being filed under a guess.
#
# ======================= EXAMPLE ONLY -- NOTHING REAL =======================
# Every value below is a placeholder and resolves to nothing. To use this skill:
#   1. point STM_ROOT at the directory holding your own transcript archive;
#   2. replace this entry with one entry per set you actually have, `path`
#      being relative to STM_ROOT (a directory, or a single .jsonl file);
#   3. run the three probes in `scripts/probe_*.py` to settle each set's `unit`
#      before distilling anything (SKILL.md, "Porting to another corpus").
# ===========================================================================
SETS = {
    "example-set": {
        # Directory or single transcript file under STM_ROOT. Placeholder.
        "path": "example-attention-sessions",
        # "codex" for rollout-*.jsonl logs, "claude-code" for a project dir.
        "format": "codex",
        # "ab-comparison" or "version-ladder" -- decide it with the probes.
        "unit": "ab-comparison",
        # Fallbacks, used only where detection finds nothing in the transcripts.
        "arch": "blackwell-ultra", "product": "b300", "dsl": "cutedsl",
        "workload_family": "attention",
        "note": "EXAMPLE placeholder set. Describe your run here: operator, "
                "dtype, what the sessions were trying to make faster, and any "
                "quirk a reader of reports/recon.md needs to know -- a device "
                "that reports itself under a different name, a DSL migration "
                "mid-run, two operators sharing one workspace.",
    },
}

# Archive entries that are verified duplicates of a registered set, typically the
# zip or tarball a set was extracted from. Naming them here means a future reader
# does not have to re-derive that they are redundant, and a glob over the archive
# root can never silently double-count a corpus. Empty by default; add your own:
#
#   "example-attention-sessions.zip": "zip of the example-set directory",
EXCLUDED = {}

SET = os.environ.get("STM_SET") or ""


def set_config(name=None):
    name = name or SET
    if not name:
        raise SystemExit("set STM_SET to one of: %s" % ", ".join(sorted(SETS)))
    if name not in SETS:
        raise SystemExit("unknown set %r; known: %s"
                         % (name, ", ".join(sorted(SETS))))
    return SETS[name]


def set_root(name=None):
    """Absolute root of one set. The only place an absolute path is formed."""
    return (ARCHIVE_ROOT / set_config(name)["path"]).resolve()


def require_set(name=None):
    name = name or SET
    cfg = set_config(name)
    root = set_root(name)
    if not root.exists():
        raise SystemExit(
            "set %r resolves to %s which does not exist; point STM_ROOT at your "
            "transcript archive and register the set in config.SETS"
            % (name, root))
    return name, cfg, root


def workspace(name=None):
    name = name or SET
    return Path(os.environ.get("STM_WORKSPACE")
                or Path("/tmp/session-trace-mining") / (name or "unset")).resolve()


def work(name=None):
    return workspace(name) / "work"


def packets(name=None):
    return workspace(name) / "packets"


def store(name=None):
    """Where the product lands.

    A subtree of `kernel_wiki/` beside the committed store it feeds, but *not*
    inside `kernel_wiki/records/`: these records use the derived
    `session-trace-1.0` schema, so filing them into the committed store would
    make that store fail its own schema gate. Promoting a record means rewriting
    it against the kernel schema, deliberately and one at a time.
    """
    name = name or SET
    return Path(os.environ.get("STM_STORE")
                or WIKI_ROOT / "kernel_wiki" / "session_trace"
                / (name or "unset")).resolve()


def records(name=None):
    return store(name) / "records"


def reports(name=None):
    return store(name) / "reports"


def ensure_dirs(name=None):
    for d in (work(name), packets(name), records(name), reports(name)):
        d.mkdir(parents=True, exist_ok=True)
