#!/usr/bin/env python3
"""Parse one session set into work/versions.jsonl + candidates.jsonl + meta.json.

The row this writes is deliberately the same shape a miner of a checked-out
optimisation run would write, so `ladder.select()` in this skill can consume
either. The difference is what is available: a checked-out run has the code and
the benchmark harness, while this one has only the transcripts, so every field has
to be recovered from what the session happened to print.

Three sources, each authoritative about one thing (lessons.md #1):

  `git log --oneline` echoes   the ladder skeleton and the kept/reverted verdict.
                               Present for versions whose session is not even in
                               the archive, so it is the only complete list.
  memory/vN.json documents     the measurements. Only some versions have one
                               in-transcript, but where present it is complete.
  the session that owns a       the code changes and the raw benchmark output.
  version                      Found by which session wrote that vN.json or made
                               that `vN:` commit.

Sets with no ladder at all are cut a different way: one candidate per measured
A/B inside one monotone region. Same row shape, `unit` says which.

Usage: STM_SET=<name> python3 ingest.py
"""
import json
import os
import re
import sys
from collections import Counter, OrderedDict

import config as c
import metrics as M
import transcripts as T

# ------------------------------------------------------------------ patterns

# `2ae4b64 v54: reverted (spill-elim reorder DEAD-END: ...)` in a git log echo.
GITLOG_RE = re.compile(r"^[ \t>|*\\/]*([0-9a-f]{7,40})\s+v(\d+)\s*:\s*(.+)$",
                       re.M)
# `[master 2ab1a19] v15: flat-prefill in-kernel qo_indptr scan`
COMMIT_ECHO_RE = re.compile(r"^\[[^\]]*?\s([0-9a-f]{7,40})\]\s*v(\d+)\s*:\s*(.+)$",
                            re.M)
# `### [v1/S1] 2026-07-07 — ...` in a hand-kept iteration log.
NOTES_HEAD_RE = re.compile(r"^#{2,4}\s*\[v(\d+)(?:/[^\]]*)?\]\s*(.*)$", re.M)

REVERT_RE = re.compile(r"\breverted\b|\bdead-end recorded\b|\brevert\b", re.I)

OWN_VN_JSON_RE = re.compile(r"memory/v(\d+)\.json")
OWN_COMMIT_RE = re.compile(r"git\s+commit[^\n]*?\bv(\d+)\s*:", re.I)
OWN_MM_RE = re.compile(r"memory_manager\.py\s+(?:create|update)\s+v?(\d+)", re.I)

# GPU identity as the harness prints it. Only the public part names are matched:
# a deployment that relabels its parts with site-local SKU aliases should add
# them here, since such a mapping is fleet-specific and cannot be inferred.
GPU_ALIAS = {"b300": ("b300", "blackwell-ultra"),
             "b200": ("b200", "blackwell"),
             "gb200": ("b200", "blackwell")}
GPU_RE = re.compile(r"NVIDIA\s+(B[23]00|GB200)", re.I)
SM_RE = re.compile(r"\bsm[_-]?(\d{2,3})a?\b", re.I)
SM_ARCH = {"100": ("b200", "blackwell"), "103": ("b300", "blackwell-ultra")}

# Which DSL a kernel is written in. Gluon first: it is imported *from* Triton, so
# a Gluon kernel always also mentions Triton and the cheaper test would swallow it.
def detect_dsl(text):
    if not text:
        return None
    if re.search(r"\bgluon\b|triton\.experimental\.gluon", text):
        return "gluon"
    if re.search(r"\bcutlass\b|\bcute\b|cutlass\.cute|CuTeDSL|cute_dsl", text,
                 re.I):
        # Spelled as the wiki schema's `scope.dsl` vocabulary spells it: a record
        # whose dsl is not in that enum fails the schema gate, and the layout gate
        # derives its directory from the same string.
        return "cutedsl"
    if re.search(r"load_inline|__global__|cpp_extension|\.cu\b", text):
        return "cuda"
    # A bare mention of triton is the *harness*, not the kernel: this corpus
    # benchmarks CuTeDSL kernels with `triton.testing.do_bench`, and matching the
    # word alone filed one of them under dsl=triton. Require a real Triton kernel
    # signal instead.
    stripped = re.sub(r"triton\.testing[\w.]*", " ", text)
    if re.search(r"@triton\.jit|\btl\.\w+|from\s+triton\s+import"
                 r"|triton\.language|\btriton\.\w+", stripped):
        return "triton"
    return None


# ------------------------------------------------------------ per-transcript

def own_version(events):
    """Which version this session produced, and how we know.

    Writing `memory/vN.json` is the strongest signal (the run writes it once per
    cycle, for its own version); a `vN:` commit is next; a memory_manager
    create/update is weakest because a session sometimes patches an older record.
    """
    votes, how = Counter(), {}
    for e in events:
        if e.kind == "edit":
            for path in (e.meta.get("files") or {}):
                m = OWN_VN_JSON_RE.search(path)
                if m:
                    v = "v" + m.group(1)
                    votes[v] += 5
                    how.setdefault(v, "wrote memory/vN.json")
        elif e.kind == "tool-call":
            for m in OWN_COMMIT_RE.finditer(e.text):
                v = "v" + m.group(1)
                votes[v] += 3
                how.setdefault(v, "git commit vN:")
            for m in OWN_MM_RE.finditer(e.text):
                v = "v" + m.group(1)
                votes[v] += 2
                how.setdefault(v, "memory_manager create/update")
    if not votes:
        return None, None
    top = max(votes, key=lambda v: (votes[v], int(v[1:])))
    return top, how.get(top)


def regions(events):
    """Split at every discontinuity: across one, line order is not history."""
    out, cur = [], []
    for e in events:
        if e.kind == "discontinuity":
            if cur:
                out.append(cur)
            cur = []
            continue
        cur.append(e)
    if cur:
        out.append(cur)
    return out


def aborted_turns(events):
    """Line ranges of turns that were abandoned.

    `turn_aborted` means the edit landed but the conclusion never did, so
    anything inside is not evidence of an outcome.
    """
    spans, start = [], None
    for e in events:
        if e.kind == "turn-start":
            start = e.line_no
        elif e.kind == "turn-abort":
            spans.append((start or 0, e.line_no))
            start = None
    return spans


def in_spans(line_no, spans):
    return any(a <= line_no <= b for a, b in spans)


def scan_transcript(path, rel_path, cfg):
    """Everything one transcript can contribute, with provenance on each item."""
    fmt, events = T.parse(path)
    if not fmt:
        return None
    meta = {"rel_path": rel_path, "format": fmt, "session_id": None,
            "cwd": None, "git_branch": None, "cli_version": None}
    for e in events:
        if e.kind == "session-meta":
            for k in ("session_id", "cwd", "git_branch", "cli_version"):
                if not meta.get(k) and e.meta.get(k):
                    meta[k] = e.meta[k]
    if not meta["session_id"]:
        meta["session_id"] = path.stem

    aborted = aborted_turns(events)
    regs = regions(events)
    reg_of = {}
    for ri, reg in enumerate(regs):
        for e in reg:
            reg_of[e.line_no] = ri

    # tool_use_id / call_id -> the command, so an output can be identified by the
    # benchmark that produced it. Needed for A/B pairing: two runs of the *same*
    # command are comparable, two different commands are not.
    cmd_of = {}
    poll_of = {}
    for e in events:
        if e.kind == "tool-call":
            key = e.meta.get("call_id") or e.meta.get("tool_use_id")
            if key:
                cmd_of[key] = e.text
                if e.meta.get("polls_shell"):
                    poll_of[key] = e.meta["polls_shell"]

    # A long-running benchmark is launched once and drained by many polls, so the
    # poll itself names no benchmark. The launching command echoes `SESSION_ID=N`
    # in its output; that is the only link back to what is being measured, and
    # without it every poll looks like a distinct command and nothing ever pairs.
    shell_cmd = {}
    for e in events:
        if e.kind == "tool-output" and e.meta.get("opens_shell"):
            cmd = cmd_of.get(e.meta.get("call_id"))
            if cmd:
                shell_cmd[e.meta["opens_shell"]] = cmd

    def bench_cmd(e):
        key = e.meta.get("call_id") or e.meta.get("tool_use_id")
        shell = poll_of.get(key)
        if shell and shell in shell_cmd:
            return shell_cmd[shell]
        return cmd_of.get(key) or e.meta.get("cmd")

    ladder, notes, docs = OrderedDict(), OrderedDict(), []
    for e in events:
        if e.kind != "tool-output":
            continue
        for sha, n, subj in GITLOG_RE.findall(e.text):
            _remember_ladder(ladder, n, sha, subj, e.line_no, e.digest)
        for sha, n, subj in COMMIT_ECHO_RE.findall(e.text):
            _remember_ladder(ladder, n, sha, subj, e.line_no, e.digest)
        for n, head in NOTES_HEAD_RE.findall(e.text):
            notes.setdefault("v" + n, {"text": head.strip(),
                                       "line_no": e.line_no,
                                       "digest": e.digest})
        for doc in M.extract_version_docs(e.text):
            docs.append({"doc": doc, "line_no": e.line_no, "digest": e.digest,
                         "tier": e.tier or T.TIER_T2})

    # Version documents also arrive as Write inputs, which is the only place a
    # document appears for the version being created right now.
    for e in events:
        if e.kind != "edit":
            continue
        for fpath, f in (e.meta.get("files") or {}).items():
            if not OWN_VN_JSON_RE.search(fpath):
                continue
            for doc in M.extract_version_docs(f.get("new_text") or ""):
                docs.append({"doc": doc, "line_no": e.line_no,
                             "digest": e.digest, "tier": T.TIER_T3})

    edits = [e for e in events
             if e.kind == "edit" and e.meta.get("code_files")
             and not in_spans(e.line_no, aborted)]
    outputs = [e for e in events
               if e.kind == "tool-output" and e.tier in (T.TIER_T1, T.TIER_T2)
               and not in_spans(e.line_no, aborted)]

    ver, how = own_version(events)
    scope = detect_scope(events, cfg)
    return {"meta": meta, "events": events, "regions": regs, "reg_of": reg_of,
            "cmd_of": cmd_of, "bench_cmd": bench_cmd, "shell_cmd": shell_cmd,
            "ladder": ladder, "notes": notes, "docs": docs,
            "edits": edits, "outputs": outputs, "own_version": ver,
            "own_how": how, "scope": scope, "aborted": aborted}


def _remember_ladder(ladder, n, sha, subj, line_no, digest):
    """Keep the longest subject seen for a version.

    The same commit is echoed many times across a set (`cc74a76 v43 ...` appears
    four times in one file), sometimes truncated by the terminal width. The
    longest sighting is the one that still has its numbers.
    """
    v = "v" + n
    subj = subj.strip()
    prev = ladder.get(v)
    if prev and len(prev["subject"]) >= len(subj):
        return
    ladder[v] = {"version": v, "sha": sha, "subject": subj,
                 "line_no": line_no, "digest": digest}


def detect_scope(events, cfg):
    """product / arch / dsl, from what the session printed. Defaults are last.

    `arch_basis` records which happened, because "detected from the device query"
    and "assumed from the set config" are different levels of certainty and a
    reader of `reports/recon.md` needs to see which one a record rests on.
    """
    product = arch = None
    basis = "set-default"
    for e in events:
        if e.kind != "tool-output":
            continue
        m = GPU_RE.search(e.text)
        if m:
            product, arch = GPU_ALIAS[m.group(1).lower()]
            basis = "device-query"
            break
    if product is None:
        for e in events:
            text = e.text if e.kind in ("tool-call", "tool-output") else ""
            m = SM_RE.search(text or "")
            if m and m.group(1) in SM_ARCH:
                product, arch = SM_ARCH[m.group(1)]
                basis = "sm-target"
                break
    dsl, dsl_basis = None, "set-default"
    # A vote over every code edit, not the first one that says anything. Reading
    # only the first edit made one set's v2 flip from triton to gluon when the
    # citation set changed and a different edit happened to come first -- the
    # kernel's language is a property of the session, not of an edit ordering.
    votes = Counter()
    for e in events:
        if e.kind != "edit":
            continue
        for fpath, f in (e.meta.get("files") or {}).items():
            if not T.is_code_path(fpath):
                continue
            got = detect_dsl((f.get("new_text") or "") + " "
                             + (f.get("diff") or ""))
            if got:
                votes[got] += 1
    if votes:
        dsl, dsl_basis = votes.most_common(1)[0][0], "kernel-source"
    return {"product": product or cfg.get("product"),
            "arch": arch or cfg.get("arch"),
            "arch_basis": basis,
            "dsl": dsl or cfg.get("dsl") or "any",
            "dsl_basis": dsl_basis}


# ------------------------------------------------------------------ dedup

def wiki_index():
    """(operator, version) already covered by the committed store.

    The user's decision is to skip colliding versions outright, so this index is
    consulted during ingest rather than only at gate time -- a skipped version
    should not even produce a packet.
    """
    idx = {}
    for root in c.COMMITTED_STORES:
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            if path.name == "index.json":
                continue
            try:
                rec = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            raw = ((rec.get("evidence") or {}).get("raw") or {})
            repo, ver = raw.get("source_repo"), raw.get("version")
            if not repo or not ver:
                continue
            key = "%s|%s" % (os.path.basename(str(repo).rstrip("/")),
                             str(ver).lower())
            idx.setdefault(key, []).append(rec.get("id"))
    return idx


def operator_id(cwd, fallback):
    """The operator, from the working directory the session ran in.

    `cwd` is the authoritative identity in these corpora (present on every claude
    line, in session_meta for codex) and it is what the main store's
    `source_repo` basename is derived from, so dedup keys line up.
    """
    if not cwd:
        return fallback
    return os.path.basename(str(cwd).rstrip("/")) or fallback


# ------------------------------------------------------- version-ladder build

def build_versions(scans, cfg, set_name, wiki):
    """One row per version, merged across the whole set."""
    ladder, notes = OrderedDict(), OrderedDict()
    for s in scans:
        for v, row in s["ladder"].items():
            prev = ladder.get(v)
            if prev is None or len(row["subject"]) > len(prev["subject"]):
                ladder[v] = dict(row, rel_path=s["meta"]["rel_path"])
        for v, row in s["notes"].items():
            notes.setdefault(v, dict(row, rel_path=s["meta"]["rel_path"]))

    docs_by_ver = {}
    for s in scans:
        for d in s["docs"]:
            ver = str(d["doc"].get("version") or "").lower()
            if not re.fullmatch(r"v\d+", ver):
                continue
            # Highest trust wins: a document written by this session (T3) over one
            # read back later (T2), and a longer document over a truncated one.
            prev = docs_by_ver.get(ver)
            cand = dict(d, rel_path=s["meta"]["rel_path"],
                        session_id=s["meta"]["session_id"])
            if prev is None or (cand["tier"] == T.TIER_T3
                                and prev["tier"] != T.TIER_T3):
                docs_by_ver[ver] = cand

    session_of_path = {s["meta"]["rel_path"]: s["meta"]["session_id"]
                       for s in scans}
    owner = {}
    for s in scans:
        v = s["own_version"]
        if not v:
            continue
        prev = owner.get(v)
        score = (len(s["edits"]), len(s["outputs"]))
        if prev is None or score > (len(prev["edits"]), len(prev["outputs"])):
            owner[v] = s

    rows, skipped = [], []
    all_vers = sorted(set(ladder) | set(docs_by_ver) | set(notes) | set(owner),
                      key=lambda v: int(re.sub(r"\D", "", v) or 0))
    for ver in all_vers:
        lad = ladder.get(ver) or {}
        doc = docs_by_ver.get(ver)
        note = notes.get(ver) or {}
        own = owner.get(ver)
        summary = M.summarise_version_doc(doc["doc"]) if doc else {}
        subject = lad.get("subject") or note.get("text") or ""

        op = operator_id(own["meta"]["cwd"] if own else None,
                         c.set_config(set_name)["path"].split("/")[-1])
        dedup = "%s|%s" % (op, ver)
        if dedup in wiki:
            skipped.append({"version": ver, "dedup_key": dedup,
                            "covered_by": wiki[dedup]})
            continue

        # geomean: the document is authoritative; a subject arrow is the fallback
        # that keeps versions without an in-transcript document usable.
        geo = summary.get("geomean_us")
        geo_basis = "version-doc" if geo else None
        if not geo and subject:
            _before, after = M.arrow_pair(subject)
            if after:
                geo, geo_basis = after, "subject-arrow"

        subj_pct, subj_pct_basis = (M.delta_from_text(subject)
                                    if subject else (None, None))
        scope = own["scope"] if own else {"product": cfg.get("product"),
                                          "arch": cfg.get("arch"),
                                          "arch_basis": "set-default",
                                          "dsl": cfg.get("dsl") or "any",
                                          "dsl_basis": "set-default"}
        # Citations, kept per transcript. A version's evidence can come from three
        # different files -- the git-log echo in one session, the memory document
        # in another, the code edits in the owning session -- and a single
        # rel_path with one flat line list checks a digest against the wrong file.
        # A distilling agent found four such citations in one ladder set.
        cited = []          # (rel_path, line_no, digest)
        tiers = {}
        for src in (lad, note):
            if src.get("line_no") and src.get("rel_path"):
                cited.append((src["rel_path"], src["line_no"], src["digest"]))
        if doc:
            cited.append((doc["rel_path"], doc["line_no"], doc["digest"]))
            tiers["geomean_us"] = doc["tier"]
        owner_path = own["meta"]["rel_path"] if own else None
        if own:
            # Verbatim channels first (a snippet may only come from one), most
            # recent last, capped so a citation stays reviewable rather than
            # listing an entire session.
            ranked = sorted(
                own["edits"],
                key=lambda e: (any(f.get("verbatim")
                                   for f in e.meta["files"].values()),
                               e.line_no))
            for e in ranked[-6:]:
                cited.append((owner_path, e.line_no, e.digest))

        # One primary transcript: the owner when it contributed anything, since
        # that is where the code is; otherwise whoever supplied the numbers.
        paths = [p for p, _l, _d in cited]
        primary = None
        for cand in (owner_path, doc["rel_path"] if doc else None,
                     lad.get("rel_path"), note.get("rel_path")):
            if cand and cand in paths:
                primary = cand
                break
        cites = sorted({l for p, l, _d in cited if p == primary})
        digests = {str(l): d for p, l, d in cited if p == primary}
        siblings = sorted({p for p in paths if p and p != primary})

        rows.append({
            "version": ver,
            "n": int(re.sub(r"\D", "", ver) or 0),
            "unit": "version-ladder",
            "has_commit": bool(lad.get("sha")),
            "has_memory": bool(doc),
            "sha": lad.get("sha"),
            "parent": None,
            "date": None,
            "subject": subject,
            "body": "",
            "reverted": bool(REVERT_RE.search(subject))
                        or str(summary.get("gate_result") or "").upper()
                        in ("REVERT", "REVERTED"),
            "subject_pct": subj_pct,
            "subject_pct_basis": subj_pct_basis,
            "geomean_us": geo,
            "geomean_basis": geo_basis,
            "dsl": scope["dsl"],
            "scope": scope,
            "operator_id": op,
            "dedup_key": dedup,
            # The session id must name the transcript that rel_path points
            # at, not the owning session: after the primary-transcript fix
            # those can differ, and a mismatch made the provenance gate
            # reject a record for citing an id from a file it does not read.
            "owner_session": session_of_path.get(primary)
                             or (own["meta"]["session_id"] if own else None),
            "owner_rel_path": primary or (own["meta"]["rel_path"] if own else None),
            "sibling_paths": siblings,
            "after_us": geo,
            "owner_how": own["own_how"] if own else None,
            "n_edits": len(own["edits"]) if own else 0,
            "n_verbatim_edits": (
                sum(1 for e in own["edits"]
                    for f in e.meta["files"].values() if f.get("verbatim"))
                if own else 0),
            "n_metric_outputs": len(own["outputs"]) if own else 0,
            # A strategy record must carry a verbatim implementation, so how the
            # code arrived matters as much as whether it did: `full` means the
            # transcript holds a real patch, `partial` means edits were made but
            # only as reconstructed strings, `blind` means the change is not
            # recoverable at all (no owning session, or it edited nothing).
            "diff_coverage": (
                "blind" if not own or not own["edits"]
                else ("full" if any(f.get("verbatim")
                                    for e in own["edits"]
                                    for f in e.meta["files"].values())
                      else "partial")),
            "doc_rel_path": doc["rel_path"] if doc else None,
            "ladder_rel_path": lad.get("rel_path"),
            "cite_lines": sorted(set(cites)),
            "cite_digests": digests,
            "number_tiers": tiers,
            "profile_dirs": [], "ncu_usable_dirs": [], "report_dirs": [],
            **{k: v for k, v in summary.items() if k != "geomean_us"},
        })
    return rows, skipped, ladder, skipped


# ------------------------------------------------------ ab-comparison build

# Two runs are comparable only when the same command produced both. Volatile
# parts (a temp path, a device index, a timestamped output dir) are normalized
# out; everything else must match exactly.
VOLATILE_RE = re.compile(
    r"(/tmp/[\w./-]+|CUDA_VISIBLE_DEVICES=\d+|--device[= ]\d+"
    r"|\d{8}T\d{6}Z?|core-node-[\w-]+|--out(?:put)?[= ]\S+)")


def bench_identity(cmd):
    if not cmd:
        return None
    ident = VOLATILE_RE.sub("~", cmd)
    ident = re.sub(r"\s+", " ", ident).strip()
    return ident or None


def build_ab(scans, cfg, set_name, wiki):
    """One row per measured A/B.

    Two sources, in order of how well the corpus supports them:

      variant comparison   both sides printed in one output (`static=`/`clc=` on
                           one line, one table column per kernel variant, or a
                           columnar NCU digest). This is what the codex sets
                           actually contain, and it is stronger evidence than a
                           re-run: same script, same GPU, same moment.
      edit-bracketed pair  the same benchmark run before and after an edit. Kept
                           because other corpora do re-run a fixed command, but it
                           finds nothing in a codex set: measured 61 distinct
                           benchmark identities in one transcript and 0 repeated,
                           because every run is a freshly written inline script.
    """
    rows = []
    # One sequence across the whole set: resetting it per transcript made `ab01`
    # repeat across the files of one set, and the packet writer keys on the candidate
    # name, so later transcripts silently overwrote earlier ones.
    seq = 0
    for s in scans:
        seen_cmp = {}
        for reg_i, reg in enumerate(s["regions"]):
            reg_edits = [e for e in reg if e in s["edits"]]
            reg_out = [e for e in reg if e in s["outputs"]]

            # ---- source 1: comparisons complete within one output
            for e in reg_out:
                for cmp_ in M.variant_ab(e.text):
                    key = (cmp_["baseline"], cmp_["candidate"],
                           round(cmp_["delta_pct"], 2), cmp_["n_rows"])
                    if key in seen_cmp:
                        row = seen_cmp[key]
                        if e.line_no not in row["cite_lines"]:
                            row["cite_lines"] = sorted(row["cite_lines"]
                                                       + [e.line_no])
                            row["cite_digests"][str(e.line_no)] = e.digest
                        continue
                    linked = _link_edits(reg_edits, cmp_,
                                         before_line=e.line_no)
                    files = {}
                    for ed in linked:
                        files.update(ed.meta.get("files") or {})
                    verbatim = any(f.get("verbatim") for f in files.values())
                    seq += 1
                    cites = {e.line_no}
                    digests = {str(e.line_no): e.digest}
                    for ed in linked:
                        cites.add(ed.line_no)
                        digests[str(ed.line_no)] = ed.digest
                    row = _ab_row(
                        s, cfg, set_name, seq,
                        before_us=cmp_["baseline_us"],
                        after_us=cmp_["candidate_us"],
                        delta=cmp_["delta_pct"],
                        delta_basis=cmp_["kind"],
                        bench_identity="%s vs %s" % (cmp_["baseline"],
                                                     cmp_["candidate"]),
                        files=files, verbatim=verbatim, cites=cites,
                        digests=digests, tier=e.tier or T.TIER_T1,
                        n_rows=cmp_["n_rows"], date=e.ts,
                        extra={"baseline_label": cmp_["baseline"],
                               "candidate_label": cmp_["candidate"],
                               "side_basis": cmp_["side_basis"],
                               "quote": cmp_["quote"],
                               "linked_by": [str(ed.line_no) for ed in linked],
                               "region": reg_i})
                    seen_cmp[key] = row
                    rows.append(row)

            # ---- source 2: the same benchmark either side of an edit
            runs = {}
            for e in reg_out:
                ident = bench_identity(s["bench_cmd"](e))
                if not ident:
                    continue
                ts = M.timings(e.text)
                if not ts:
                    continue
                runs.setdefault(ident, []).append((e, ts))
            for ident, seen in runs.items():
                seen = sorted(seen, key=lambda t: t[0].line_no)
                # Consecutive runs of the same benchmark bracket a change. A pair
                # is a candidate only when code was edited in between; several
                # edits between one pair are one change, not several, because the
                # measurement cannot attribute the delta to any one of them.
                for (b_ev, b_ts), (a_ev, a_ts) in zip(seen, seen[1:]):
                    between = [e for e in reg_edits
                               if b_ev.line_no < e.line_no < a_ev.line_no]
                    if not between:
                        continue
                    b_us, a_us = _headline(b_ts), _headline(a_ts)
                    delta = M.delta_from_pair(b_us, a_us)
                    if delta is None:
                        continue
                    seq += 1
                    files = {}
                    for ed in between:
                        files.update(ed.meta.get("files") or {})
                    cites = {b_ev.line_no, a_ev.line_no}
                    digests = {str(b_ev.line_no): b_ev.digest,
                               str(a_ev.line_no): a_ev.digest}
                    for ed in between:
                        cites.add(ed.line_no)
                        digests[str(ed.line_no)] = ed.digest
                    rows.append(_ab_row(
                        s, cfg, set_name, seq, before_us=b_us, after_us=a_us,
                        delta=delta, delta_basis="before-after",
                        bench_identity=ident[:300], files=files,
                        verbatim=any(f.get("verbatim") for f in files.values()),
                        cites=cites, digests=digests,
                        tier=b_ev.tier or T.TIER_T1, n_rows=1,
                        date=between[-1].ts,
                        extra={"region": reg_i,
                               "linked_by": [str(e.line_no) for e in between]}))
    return rows


# A label names a variant; the code that implements it usually mentions one of
# the label's own words. `1CTA+CLC (us)` -> {1cta, clc}.
def _label_tokens(label):
    return {t.lower() for t in re.split(r"[^\w]+", label or "")
            if len(t) >= 3 and not re.fullmatch(r"\d+", t)}


# Words that name a column's role, not a mechanism. They appear in almost every
# diff, so linking on them attached 22 edits to one comparison -- which is not a
# link, it is the whole session.
LINK_STOPWORDS = {"static", "baseline", "original", "best", "median", "cases",
                  "wins", "total", "decode", "prefill", "kernel", "official",
                  "gen", "cosine", "batch", "seqlen", "block", "split", "p50"}


def _link_edits(edits, cmp_, before_line=None, limit=4):
    """Edits that plausibly produced the candidate side of a comparison.

    The link is lexical: a diff that mentions `use_clc_scheduler` is evidence for
    the `1CTA+CLC` column. Only the candidate's own distinctive tokens are used --
    the baseline's words describe what was already there -- and the nearest few
    edits win, because a comparison is explained by the change just before it and
    a list of twenty is not a citation. Recorded as `linked_by` so a reviewer can
    check which lines it rested on.
    """
    want = _label_tokens(cmp_["candidate"]) - LINK_STOPWORDS
    if not want:
        return []
    hits = []
    for ed in edits:
        if before_line is not None and ed.line_no > before_line:
            continue
        blob = " ".join((f.get("diff") or "")
                        for f in (ed.meta.get("files") or {}).values()).lower()
        blob += " " + " ".join(ed.meta.get("files") or {}).lower()
        if any(tok in blob for tok in want):
            hits.append(ed)
    return hits[-limit:]


def _ab_row(s, cfg, set_name, seq, before_us, after_us, delta, delta_basis,
            bench_identity, files, verbatim, cites, digests, tier, n_rows, date,
            extra=None):
    row = {
        "version": "ab%02d" % seq,
        "n": seq,
        "unit": "ab-comparison",
        "has_commit": False,
        "has_memory": False,
        "sha": None, "parent": None, "date": date,
        "subject": "", "body": "",
        "reverted": delta < 0,
        "bench_identity": bench_identity,
        "before_us": before_us, "after_us": after_us,
        "geomean_us": after_us,
        "geomean_basis": delta_basis,
        "improve_pct_raw": round(delta, 3),
        "delta_basis": delta_basis,
        # A delta this large is, in this corpus, always two columns that do not
        # measure the same thing -- a split-K variant paired against a
        # whole-kernel baseline, or a column whose header lies about its unit.
        # Genuine multi-x slowdowns do exist in this domain (paged KV at
        # page_size=1 is ~4.6x slower than at 128), so this is a flag for review
        # rather than a filter inside the parser: partition keeps flagged rows out
        # of the record set and recon reports how many there were.
        "implausible": abs(delta) > 300.0,
        "n_shapes": n_rows,
        "dsl": s["scope"]["dsl"],
        "scope": s["scope"],
        "operator_id": operator_id(
            s["meta"]["cwd"], c.set_config(set_name)["path"].split("/")[-1]),
        "dedup_key": None,
        "owner_session": s["meta"]["session_id"],
        "owner_rel_path": s["meta"]["rel_path"],
        "owner_how": delta_basis,
        "edit_files": sorted(files),
        "diff_coverage": "full" if verbatim else ("partial" if files else "blind"),
        "n_edits": len(files),
        "n_verbatim_edits": sum(1 for f in files.values() if f.get("verbatim")),
        "n_metric_outputs": 1,
        "cite_lines": sorted(cites),
        "cite_digests": digests,
        "number_tiers": {"latency": tier},
        "correctness_status": None,
        "gate_result": None, "gate_failure": None,
        "action_category": None, "action_description": None,
        "expected_impact": None, "pitfalls": [],
        "open_directions": [], "search_log": [],
        "profile_evidence": {},
        "profile_dirs": [], "ncu_usable_dirs": [], "report_dirs": [],
    }
    row.update(extra or {})
    return row


def _headline(ts):
    """The number that represents a run.

    The minimum rather than the mean: a benchmark prints several shapes and often
    a warm-up, and the minimum is the one reading the harness itself reports as
    the kernel time. Taking a mean would mix shapes of different sizes.
    """
    vals = [t["value_us"] for t in ts if t.get("value_us")]
    return min(vals) if vals else None


# ---------------------------------------------------------------------- main

def main():
    set_name, cfg, root = c.require_set()
    c.ensure_dirs(set_name)
    files = [p for p in T.iter_transcripts(root) if "subagents" not in p.parts]
    sub = [p for p in T.iter_transcripts(root) if "subagents" in p.parts]

    scans, unparsed = [], []
    for p in files:
        rel = str(p.relative_to(root)) if p != root else p.name
        s = scan_transcript(p, rel, cfg)
        if s is None:
            unparsed.append(rel)
            continue
        scans.append(s)

    wiki = wiki_index()
    if cfg["unit"] == "version-ladder":
        rows, skipped, ladder, _ = build_versions(scans, cfg, set_name, wiki)
    else:
        rows = build_ab(scans, cfg, set_name, wiki)
        skipped, ladder = [], {}

    fmts = Counter(s["meta"]["format"] for s in scans)
    products = Counter(s["scope"]["product"] for s in scans)
    dsls = Counter(s["scope"]["dsl"] for s in scans)
    operators = Counter(operator_id(s["meta"]["cwd"], "?") for s in scans)

    meta = {
        "set": set_name,
        "set_root": str(root),
        "unit": cfg["unit"],
        "declared": {k: cfg.get(k) for k in
                     ("format", "arch", "product", "dsl", "workload_family")},
        "note": cfg.get("note"),
        "n_transcripts": len(files),
        "n_subagent_transcripts": len(sub),
        "n_unparsed": len(unparsed),
        "unparsed": unparsed[:20],
        "formats": dict(fmts),
        "products_detected": {str(k): v for k, v in products.items()},
        "dsls_detected": {str(k): v for k, v in dsls.items()},
        "operators": dict(operators),
        "ladder_versions": len(ladder),
        "sessions_owning_a_version": sum(1 for s in scans if s["own_version"]),
        "sessions_without_version": sum(1 for s in scans if not s["own_version"]),
        "wiki_skipped": skipped,
        "excluded_from_archive": c.EXCLUDED,
    }

    work = c.work(set_name)
    _write_jsonl(work / "versions.jsonl", rows)
    (work / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    # The set registry, so the provenance gate can resolve `set` -> root without
    # any absolute path having been stored on a record.
    (work / "sets.json").write_text(json.dumps(
        {name: str(c.set_root(name)) for name in c.SETS},
        ensure_ascii=False, indent=1) + "\n")

    print("set        %s  (%s, unit=%s)" % (set_name, dict(fmts), cfg["unit"]))
    print("transcripts %d main, %d subagent, %d unparsed"
          % (len(files), len(sub), len(unparsed)))
    print("scope      product=%s dsl=%s"
          % (dict(products), dict(dsls)))
    print("operators  %s" % dict(operators))
    if cfg["unit"] == "version-ladder":
        print("ladder     %d versions from echoes; %d sessions own a version, "
              "%d do not" % (len(ladder), meta["sessions_owning_a_version"],
                             meta["sessions_without_version"]))
        print("rows       %d  (with geomean %d, with commit %d, reverted %d)"
              % (len(rows), sum(1 for r in rows if r.get("geomean_us")),
                 sum(1 for r in rows if r.get("sha")),
                 sum(1 for r in rows if r.get("reverted"))))
        print("skipped    %d versions already covered by the main stores"
              % len(skipped))
        for s in skipped[:8]:
            print("             %s -> %s" % (s["dedup_key"], s["covered_by"][:2]))
    else:
        print("rows       %d A/B candidates (improving %d, regressing %d)"
              % (len(rows), sum(1 for r in rows if r["improve_pct_raw"] > 0),
                 sum(1 for r in rows if r["improve_pct_raw"] <= 0)))
        print("           blind diff coverage: %d"
              % sum(1 for r in rows if r.get("diff_coverage") == "blind"))
    print("-> %s" % (work / "versions.jsonl"))
    return 0


def _write_jsonl(path, rows):
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
