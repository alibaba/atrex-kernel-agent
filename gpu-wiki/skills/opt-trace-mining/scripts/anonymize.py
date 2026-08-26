#!/usr/bin/env python3
"""Write-side scrubber for everything an agent is allowed to read.

The store's rule is that a served record may not contain a reference the reader
cannot resolve. `tools/check_kernel_wiki.py` enforces it as three leak SHAPES --
absolute `/root|/home|/Users` paths, e-mail addresses, markdown page paths --
plus an optional private denylist supplied through `ATREX_WIKI_DENYLIST`. This
module is the write side of the same rule: it removes those shapes from packet
text *before* an agent ever sees them, so a distiller that quotes its input
verbatim cannot produce a rejected record.

Scrubbing by shape rather than by a list of names is deliberate. A committed
denylist would publish the very identifiers it is supposed to hide, so the only
name-based rule here reads from a file outside the repository:

    export ATREX_WIKI_DENYLIST=/path/to/private-substrings.txt   # one per line

Two things are rewritten beyond the gate's own shapes, because they are dangling
references too even though the gate cannot see them generically:

  version ids     `v83` means nothing outside the trace, so it becomes
                  "an earlier step". The trace's own version number lives in
                  `evidence.raw.version`, which is never served.
  workload hashes an 8-hex id identifies a benchmarked shape, and dropping it
                  would destroy a per-shape latency table. It is replaced by a
                  stable, packet-local `shape-N` label instead.

Run `python3 anonymize.py` for the self-test that pins this behaviour.
"""
import os
import re

# ---------------------------------------------------------------- leak shapes

# Kept in the same order the store's gate lists them, and applied before
# anything else: a path may end in `.md` and an e-mail may sit inside a path.
ABS_PATH_RE = re.compile(r"(?:file://)?/(?:root|home|Users)/[\w.+-]+"
                         r"(?:/[\w.+-]*)*")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
MD_PATH_RE = re.compile(r"\b[\w][\w/.-]*\.md\b")

LEAK_SUBS = (
    (ABS_PATH_RE, "a local path"),
    (EMAIL_RE, "an address"),
    (MD_PATH_RE, "a document"),
)

# ------------------------------------------------------- trace-shaped subjects

# Paths and commands into the trace repository. The agent has no access to the
# trace, so each of these is something it cannot open.
TRACE_SUBS = (
    (re.compile(r"`?\bgit\s+(?:log|show|diff|blame)[^`\n]*`?"), "the history"),
    (re.compile(r"\bprofiles?/[\w./*-]+"), "the profile"),
    (re.compile(r"\bmemory/[vV]?\d+[\w.-]*\.json\b"), "the step record"),
    (re.compile(r"\bversions/kernel_[vV]\d+\.py\b"), "that snapshot"),
    (re.compile(r"\bkernel_[vV]\d+\.py\b"), "that snapshot"),
    (re.compile(r"\b(?:kernel|reference|test_kernel|bench)\.py\b"),
     "the kernel source"),
    (re.compile(r"\b(?:definition|solution|workload)\.jsonl?\b"),
     "the trace metadata"),
    (re.compile(r"\bkernel_opt_\d+_[\w-]+"), "this operator"),
)

# Version identifiers, longest form first so `v7_rerun` does not decay into
# "an earlier step_rerun".
VERSION_SUBS = (
    (re.compile(r"^\s*[vV]\d+\w*\s*[:\-\u2013\u2014]\s*"), ""),
    (re.compile(r"\b[vV]\d+_(?:rerun\d*|baseline|base)\b"), "a re-measurement"),
    (re.compile(r"\((?:from\s+|see\s+|cf\.?\s+|per\s+)?[vV]\d+"
                r"(?:\s*[,/&]\s*[vV]\d+)*\)"), "(an earlier step)"),
    (re.compile(r"\b[vV]\d+(?:\s*(?:[,/&]|and|vs\.?)\s*[vV]\d+)+\b"),
     "earlier steps"),
    (re.compile(r"\b[vV]\d+\b"), "an earlier step"),
)

# A width, not a version: `v4` next to `vec`/`float`/`.f16` is a vector width,
# and the store's gate makes the same allowance.
WIDTH_TOKEN_RE = re.compile(r"^[vV](?:2|4|8|16|32|64|100)$")
WIDTH_CONTEXT_RE = re.compile(r"vec|\.f16|\.bf16|x2|float", re.I)

# Git references appear as short or full object ids. Require either a seven
# digit short hash or at least one a-f character so ordinary measurements are
# not mistaken for hashes.
HASH_RE = re.compile(
    r"\b(?:\d{7}|(?=[0-9a-f]{7,40}\b)(?=[0-9a-f]*[a-f])[0-9a-f]{7,40})\b"
)
MULTI_WS_RE = re.compile(r"\s{2,}")

DENYLIST_REPLACEMENT = "a private identifier"


def private_denylist():
    """Substrings from `ATREX_WIKI_DENYLIST`, or [] when it is unset.

    The same environment variable the store's gate reads, so the write side and
    the read side always agree about what is private.
    """
    path = os.environ.get("ATREX_WIKI_DENYLIST")
    if not path or not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [ln.strip() for ln in handle
                if ln.strip() and not ln.startswith("#")]


def _denylist_re():
    terms = sorted(private_denylist(), key=len, reverse=True)
    if not terms:
        return None
    return re.compile("|".join(re.escape(t) for t in terms), re.I)


class Scrubber:
    """Packet-local scrubber.

    Packet-local because the `shape-N` labels have to be stable within the unit
    a reader sees and must not imply anything across records: two packets that
    both say `shape-2` are not talking about the same shape, and a global
    counter would suggest they were.
    """

    def __init__(self):
        self._shapes = {}
        self._denylist = _denylist_re()

    # -- shape labels ------------------------------------------------------
    def shape_label(self, token):
        token = str(token)
        if token not in self._shapes:
            self._shapes[token] = "shape-%d" % (len(self._shapes) + 1)
        return self._shapes[token]

    @property
    def n_shapes(self):
        return len(self._shapes)

    # -- text --------------------------------------------------------------
    def __call__(self, text):
        if text is None or text == "":
            return ""
        out = str(text)
        if self._denylist is not None:
            out = self._denylist.sub(DENYLIST_REPLACEMENT, out)
        for rx, repl in LEAK_SUBS:
            out = rx.sub(repl, out)
        out = HASH_RE.sub(lambda m: self.shape_label(m.group(0)), out)
        for rx, repl in TRACE_SUBS:
            out = rx.sub(repl, out)
        out = self._versions(out)
        return _tidy(out)

    def _versions(self, text):
        for rx, repl in VERSION_SUBS:
            if rx is VERSION_SUBS[-1][0]:
                text = rx.sub(self._one_version, text)
            else:
                text = rx.sub(repl, text)
        return text

    @staticmethod
    def _one_version(match):
        token = match.group(0)
        if WIDTH_TOKEN_RE.match(token):
            context = match.string[max(0, match.start() - 12):match.end() + 12]
            if WIDTH_CONTEXT_RE.search(context):
                return token
        return "an earlier step"


def _tidy(text):
    """Repair the grammar the substitutions break.

    Without this the output reads as "the an earlier step kernel" and an agent
    quoting it produces a sentence a reviewer cannot parse -- which is how a
    scrubbed field ends up being rewritten by hand, and hand-rewriting is where
    fabrication enters.
    """
    text = re.sub(r"\ban earlier step(?:[\s,]+an earlier step)+\b",
                  "earlier steps", text)
    text = re.sub(r"\b([Tt]he|[Aa]n?|[Tt]his|[Tt]hat)\s+an earlier step\b",
                  lambda m: ("An earlier step" if m.group(1)[0].isupper()
                             else "an earlier step"), text)
    text = re.sub(r"\ban earlier step\s+(?:HEAD|head)\b", "an earlier state",
                  text)
    text = re.sub(r"[(\[]\s*[)\]]", "", text)
    text = re.sub(r"\s+([,;:%])", r"\1", text)
    # Only a sentence-ending period: `a .cg cache modifier` must survive, or the
    # text a record has to quote verbatim comes back subtly wrong.
    text = re.sub(r"\s+\.(?=\s|$)", ".", text)
    text = MULTI_WS_RE.sub(" ", text).strip()
    return re.sub(r"^[:;,.\-\u2013\u2014\s]+", "", text)


# ---------------------------------------------------------------------- code

TRIPLE_RE = re.compile(r"('''|\"\"\")(.*?)\1", re.S)
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)


def _scrub_comment(text, scrubber):
    """Substitutions only, without _tidy: a comment keeps its own line breaks."""
    out = text
    if scrubber._denylist is not None:                          # noqa: SLF001
        out = scrubber._denylist.sub(DENYLIST_REPLACEMENT, out)  # noqa: SLF001
    for rx, repl in LEAK_SUBS:
        out = rx.sub(repl, out)
    out = HASH_RE.sub(lambda m: scrubber.shape_label(m.group(0)), out)
    for rx, repl in TRACE_SUBS:
        out = rx.sub(repl, out)
    return out


def scrub_code(text, scrubber=None):
    """Scrub comments and docstrings only; executable code is left untouched.

    A `verbatim` gate compares a record's snippet against this output, and an
    identifier such as a local named `v0` must survive for the code to still
    compile, so substitution is confined to comment and string-literal regions.
    Version ids are deliberately NOT rewritten in code for the same reason.
    """
    if not text:
        return ""
    sc = scrubber or Scrubber()
    text = TRIPLE_RE.sub(
        lambda m: m.group(1) + _scrub_comment(m.group(2), sc) + m.group(1), text)
    text = BLOCK_COMMENT_RE.sub(lambda m: _scrub_comment(m.group(0), sc), text)

    lines = []
    for line in text.split("\n"):
        for marker in ("#", "//"):
            idx = line.find(marker)
            if idx < 0:
                continue
            prefix = line[:idx]
            # Crude but sufficient: skip a marker that sits inside a string.
            if prefix.count('"') % 2 or prefix.count("'") % 2:
                continue
            line = prefix + _scrub_comment(line[idx:], sc)
            break
        lines.append(line)
    return "\n".join(lines)


# ----------------------------------------------------------------- self-test

def self_test():
    """The behaviours the anonymisation gate depends on."""
    sc = Scrubber()
    cases = [
        # (input, must NOT contain, must contain)
        ("ran /root/someone/traces/kernel_opt_007_demo/kernel.py", "/root/", "a local path"),
        ("reported by dev.person@example.com", "@", "an address"),
        ("see docs/nvidia/blackwell/b200/ref-docs/demo.md for the page",
         ".md", "a document"),
        ("v83: padded the fft size (reverted)", "v83", "padded"),
        ("regression against v7 and v9", "v7", "earlier steps"),
        ("the v4 kernel was slower", "the an earlier step", "an earlier step"),
        ("vec4 loads with v4 float lanes", "an earlier step", "v4"),
        ("shape 3f2a91bc regressed", "3f2a91bc", "shape-1"),
        ("candidate 5f078e26f6bd1bc1d807918d7f8103423bd2e03a passed",
         "5f078e26f6bd1bc1d807918d7f8103423bd2e03a", "shape-2"),
        ("committed kernel source only (5962444)", "5962444", "shape-3"),
        ("check memory/v12.json for the numbers", "memory/", "the step record"),
        ("git log --oneline shows the revert", "git log", "the history"),
    ]
    bad = []
    for text, forbidden, wanted in cases:
        got = sc(text)
        if forbidden and forbidden in got:
            bad.append("%r still contains %r -> %r" % (text, forbidden, got))
        elif wanted not in got:
            bad.append("%r lost %r -> %r" % (text, wanted, got))

    # A stable label per shape within one scrubber, and a fresh namespace in the
    # next one.
    if sc("3f2a91bc again") != "shape-1 again":
        bad.append("shape label is not stable within one scrubber")
    if Scrubber()("ff00aa11 first") != "shape-1 first":
        bad.append("shape labels are not packet-local")

    code = ("v0 = 1  # see /home/me/notes.md\n"
            "def f():\n"
            "    '''kernel_opt_003_demo timing'''\n"
            "    return v0\n")
    scrubbed = scrub_code(code)
    if "v0 = 1" not in scrubbed or "return v0" not in scrubbed:
        bad.append("scrub_code rewrote executable code")
    if "/home/me" in scrubbed or "notes.md" in scrubbed:
        bad.append("scrub_code left a path in a comment")
    if "kernel_opt_003_demo" in scrubbed:
        bad.append("scrub_code left a corpus id in a docstring")
    return bad, len(cases) + 5


if __name__ == "__main__":
    import sys

    failures, n = self_test()
    for line in failures:
        print("FAIL %s" % line)
    print("anonymize: %d checks, %d failures" % (n, len(failures)))
    sys.exit(1 if failures else 0)
