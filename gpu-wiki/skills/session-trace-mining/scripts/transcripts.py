#!/usr/bin/env python3
"""Two transcript formats, one event stream.

This is the only file that knows how a session log is shaped. Everything
downstream sees `Event` rows and never a raw jsonl line, so porting to a third
agent product means editing here and nowhere else.

The two formats disagree about almost everything, including where the truth is:

  codex        `event_msg/patch_apply_end` carries verbatim `@@` hunks, so code
               changes are exact. But `turn_id` is absent from every response
               item, so turns can only be recovered from line position.
  claude-code  every line carries `sessionId`/`uuid`/`timestamp`, so grouping is
               free. Code changes arrive as `Edit` old/new strings, and the
               paired result line carries a pre-computed `structuredPatch` that
               is preferred over reconstructing one.

Both are read with `errors="replace"` and a per-line try: a corpus that has been
copied through a Mac and a zip has lines that do not decode, and losing one line
must not lose the file.
"""
import hashlib
import json
import re
from pathlib import Path

# ---------------------------------------------------------------- event model

# The kinds every downstream script may rely on. A parser that cannot classify a
# line emits nothing rather than guessing a kind.
KINDS = (
    "session-meta",     # provenance header: session_id, cwd, git branch
    "turn-start",       # a new turn begins here (pairing boundary)
    "turn-abort",       # turn abandoned: edits may have landed, verdict did not
    "discontinuity",    # compaction or rollback: line order stops being history
    "prompt",           # human or orchestrator instruction  (tier T5)
    "agent-text",       # assistant prose / thinking          (tier T4)
    "tool-call",        # the command that was run
    "tool-output",      # what it printed                     (tier T1/T2)
    "edit",             # a code change, with a diff when recoverable
)

# Where a number found in this event may be believed. The gate that audits
# fabrication only admits T1/T2/T3, so this table is load-bearing rather than
# documentation.
TIER_T1 = "T1"  # tool output: bench / profiler stdout
TIER_T2 = "T2"  # tool output, but of something the agent itself authored
TIER_T3 = "T3"  # agent-authored structured field (memory/vN.json performance.*)
TIER_T4 = "T4"  # agent prose -- excluded from evidence
TIER_T5 = "T5"  # orchestrator prompt -- excluded from evidence


class Event:
    """One thing that happened, with enough provenance to cite it.

    `line_no` is 1-based so it matches `sed -n '<n>p'`, which is how a human
    checks a citation.
    """

    __slots__ = ("kind", "line_no", "digest", "ts", "tier", "text", "meta")

    def __init__(self, kind, line_no, digest, ts=None, tier=None, text="",
                 meta=None):
        self.kind = kind
        self.line_no = line_no
        self.digest = digest
        self.ts = ts
        self.tier = tier
        self.text = text or ""
        self.meta = meta or {}

    def __repr__(self):
        return "<%s L%d %s %r>" % (self.kind, self.line_no, self.tier or "-",
                                   self.text[:60])


def digest_of(raw_line):
    """sha256 of the raw line bytes, newline stripped.

    The line rather than the parsed object: dict ordering and float repr are not
    stable across json round-trips, and a digest that changes when nothing
    changed is a gate that gets switched off.
    """
    if isinstance(raw_line, str):
        raw_line = raw_line.encode("utf-8", "replace")
    return hashlib.sha256(raw_line.rstrip(b"\r\n")).hexdigest()[:12]


# --------------------------------------------------------------- shared bits

# Codex wraps every exec result in a fixed preamble. Stripping it keeps the
# `Wall time 8.1 seconds` out of the number pool, where it would otherwise be a
# citable magnitude and let an agent "source" an 8.1% claim.
CODEX_PREAMBLE = re.compile(
    r"\AScript completed\s*\nWall time [\d.]+ seconds\s*\nOutput:\s*\n?", re.M)
TRUNCATED_RE = re.compile(r"Warning: truncated output \(original token count: "
                          r"(\d+)\)")

# Reading back something the agent wrote itself demotes the whole output to T2.
# Matched against the command, not the output, because the command names the
# file: `sed -n '1,260p' agent_space/hd256_2cta_fp8/NOTES.md`.
SELF_READ_RE = re.compile(
    r"\b(cat|sed|head|tail|less|bat|rg|grep)\b[^|;&]*"
    r"(NOTES\.md|PLAN\.md|opt_logs|memory/v\d+\.json|/v\d+\.json)"
    r"|memory_manager\.py\s+(read|summary)"
    r"|\bgit\s+log\b", re.I)

CODE_SUFFIXES = (".py", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".sh",
                 ".ptx", ".cuda")


def is_code_path(path):
    return str(path).endswith(CODE_SUFFIXES)


def _join_text(blocks, key="text"):
    out = []
    for b in blocks or []:
        if isinstance(b, dict):
            v = b.get(key)
            if isinstance(v, str):
                out.append(v)
        elif isinstance(b, str):
            out.append(b)
    return "".join(out)


def synth_diff_from_strings(path, old, new):
    """A unified diff for an edit that only gave us the two versions.

    Hunk header line counts are real (the strings are complete), so the result
    is a legal diff rather than a decoration. It is marked `synthesized` by its
    caller because the verbatim gate must know that the `+` lines were assembled
    here and not copied out of a patch the agent applied.
    """
    old_lines = old.splitlines() if old else []
    new_lines = new.splitlines() if new else []
    head = "--- a/%s\n+++ b/%s\n@@ -1,%d +1,%d @@\n" % (
        path, path, len(old_lines), len(new_lines))
    body = "".join("-%s\n" % l for l in old_lines)
    body += "".join("+%s\n" % l for l in new_lines)
    return head + body


def render_structured_patch(path, hunks):
    """Render claude's `structuredPatch` back into text.

    Its `lines` already carry the ` `/`+`/`-` prefixes, so they are copied
    verbatim -- this is the one channel where a claude snippet can be as exact
    as a codex one.
    """
    out = ["--- a/%s\n+++ b/%s\n" % (path, path)]
    for h in hunks or []:
        if not isinstance(h, dict):
            continue
        out.append("@@ -%s,%s +%s,%s @@\n" % (
            h.get("oldStart", 0), h.get("oldLines", 0),
            h.get("newStart", 0), h.get("newLines", 0)))
        for line in h.get("lines") or []:
            out.append(line if line.endswith("\n") else line + "\n")
    return "".join(out)


# ------------------------------------------------------------------ codex

def _codex_output_text(payload):
    out = payload.get("output")
    if isinstance(out, str):
        text = out
    elif isinstance(out, list):
        text = _join_text(out)
    else:
        text = ""
    # Some outputs are double-wrapped in a chunk envelope that carries the real
    # payload under "output".
    if text.lstrip().startswith("{") and '"chunk_id"' in text[:200]:
        try:
            inner = json.loads(text)
            if isinstance(inner.get("output"), str):
                text = inner["output"]
        except (json.JSONDecodeError, AttributeError):
            pass
    return text


EXEC_CMD_RE = re.compile(r'"cmd"\s*:\s*"((?:[^"\\]|\\.)*)"')
# A benchmark that outlives one call is started in a shell whose id the launching
# command echoes, and is then drained by polls that carry only that id:
#   exec_command(... echo SESSION_ID=$! ...)  ->  "SESSION_ID=89171"
#   write_stdin({"session_id":89171, "chars":""})  ->  the benchmark's output
# Without this handshake every poll looks like a distinct command and no two runs
# of the same benchmark can ever be compared.
OPENS_SHELL_RE = re.compile(r"SESSION_ID=(\d+)")
POLL_SESSION_RE = re.compile(r'"session_id"\s*:\s*(\d+)')


def _codex_cmd(payload):
    """The shell command inside the JS the model wrote.

    `input` is a JS program calling `tools.exec_command({"cmd": ...})`, so the
    command is a JSON string embedded in source. Pulling it out with a regex and
    unescaping through json is enough: we only need it to decide the trust tier
    and to label the span for a human.
    """
    src = payload.get("input") or payload.get("arguments") or ""
    if not isinstance(src, str):
        return ""
    m = EXEC_CMD_RE.search(src)
    if not m:
        return src[:400]
    try:
        return json.loads('"%s"' % m.group(1))
    except json.JSONDecodeError:
        return m.group(1)[:400]


def parse_codex(path):
    """Yield Events for one `rollout-*.jsonl`."""
    session_id = None
    pending_cmd = {}          # call_id -> (cmd, line_no)
    with open(path, "rb") as fh:
        for i, raw in enumerate(fh, 1):
            digest = digest_of(raw)
            try:
                obj = json.loads(raw.decode("utf-8", "replace"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(obj, dict):
                continue
            top = obj.get("type")
            p = obj.get("payload")
            p = p if isinstance(p, dict) else {}
            pt = p.get("type")
            ts = obj.get("timestamp") or p.get("timestamp")

            if top == "session_meta":
                session_id = p.get("session_id") or p.get("id")
                git = p.get("git") or {}
                yield Event("session-meta", i, digest, ts, meta={
                    "session_id": session_id,
                    "cwd": p.get("cwd"),
                    "cli_version": p.get("cli_version"),
                    "originator": p.get("originator"),
                    "git_branch": git.get("branch"),
                    "git_commit": git.get("commit_hash"),
                    "repository_url": git.get("repository_url"),
                })
                continue

            if top == "turn_context":
                yield Event("turn-start", i, digest, ts,
                            meta={"turn_id": p.get("turn_id"),
                                  "cwd": p.get("cwd"), "source": "turn_context"})
                continue

            if top == "compacted" or pt == "context_compacted":
                yield Event("discontinuity", i, digest, ts,
                            meta={"why": "compaction"})
                continue
            if pt == "thread_rolled_back":
                yield Event("discontinuity", i, digest, ts,
                            meta={"why": "rollback"})
                continue
            if pt == "turn_aborted":
                yield Event("turn-abort", i, digest, ts,
                            meta={"turn_id": p.get("turn_id")})
                continue
            if pt == "task_started":
                yield Event("turn-start", i, digest, ts,
                            meta={"turn_id": p.get("turn_id"),
                                  "source": "task_started"})
                continue

            if pt == "patch_apply_end":
                if not p.get("success"):
                    continue
                changes = p.get("changes") or {}
                files = {}
                for fpath, ch in changes.items():
                    if not isinstance(ch, dict):
                        continue
                    if ch.get("type") == "update":
                        body = ch.get("unified_diff") or ""
                        files[fpath] = {"channel": "patch-apply",
                                        "diff": _with_header(fpath, body),
                                        "verbatim": True}
                    elif ch.get("type") == "add":
                        content = ch.get("content") or ""
                        files[fpath] = {
                            "channel": "patch-apply-add",
                            "diff": synth_diff_from_strings(fpath, "", content),
                            "verbatim": False}
                    elif ch.get("type") == "delete":
                        files[fpath] = {"channel": "patch-apply-delete",
                                        "diff": "", "verbatim": False}
                if files:
                    yield Event("edit", i, digest, ts, meta={
                        "turn_id": p.get("turn_id"), "files": files,
                        "code_files": [f for f in files if is_code_path(f)]})
                continue

            if pt in ("custom_tool_call", "function_call"):
                cmd = _codex_cmd(p)
                cid = p.get("call_id")
                if cid:
                    pending_cmd[cid] = (cmd, i)
                src = p.get("input") or ""
                poll = None
                if isinstance(src, str) and "write_stdin" in src:
                    m = POLL_SESSION_RE.search(src)
                    poll = int(m.group(1)) if m else None
                yield Event("tool-call", i, digest, ts, text=cmd,
                            meta={"call_id": cid, "name": p.get("name"),
                                  "polls_shell": poll})
                continue

            if pt in ("custom_tool_call_output", "function_call_output"):
                text = _codex_output_text(p)
                cmd, _cmd_line = pending_cmd.get(p.get("call_id"), ("", None))
                trunc = bool(TRUNCATED_RE.search(text))
                body = CODEX_PREAMBLE.sub("", text)
                opened = OPENS_SHELL_RE.search(body)
                yield Event("tool-output", i, digest, ts,
                            tier=TIER_T2 if SELF_READ_RE.search(cmd or "")
                            else TIER_T1,
                            text=body, meta={"call_id": p.get("call_id"),
                                             "cmd": cmd, "truncated": trunc,
                                             "opens_shell": int(opened.group(1))
                                             if opened else None})
                continue

            if pt == "message":
                role = p.get("role")
                text = _join_text(p.get("content"))
                if role == "assistant":
                    yield Event("agent-text", i, digest, ts, tier=TIER_T4,
                                text=text)
                elif role in ("user", "developer"):
                    yield Event("prompt", i, digest, ts, tier=TIER_T5,
                                text=text, meta={"role": role})
                continue

            if pt == "agent_message":
                yield Event("agent-text", i, digest, ts, tier=TIER_T4,
                            text=p.get("message") or "")
            elif pt == "user_message":
                yield Event("prompt", i, digest, ts, tier=TIER_T5,
                            text=p.get("message") or "", meta={"role": "user"})


def _with_header(path, body):
    """Give a bare hunk body the file header the diff format needs.

    codex stores only the `@@` hunks, so a snippet checked against it would have
    no way to know which file it came from.
    """
    if body.lstrip().startswith(("---", "diff ")):
        return body
    return "--- a/%s\n+++ b/%s\n%s" % (path, path, body)


# --------------------------------------------------------------- claude-code

EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
# Reading back the run's own bookkeeping demotes the output to T2, same rule as
# codex but keyed on the tool call instead of a shell string.
SELF_READ_TOOLS = {"Read"}


def parse_claude(path):
    """Yield Events for one claude-code session jsonl (main or subagent)."""
    pending_edit = {}      # tool_use_id -> (line_no, file_path, old, new, tool)
    with open(path, "rb") as fh:
        for i, raw in enumerate(fh, 1):
            digest = digest_of(raw)
            try:
                obj = json.loads(raw.decode("utf-8", "replace"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(obj, dict):
                continue
            typ = obj.get("type")
            ts = obj.get("timestamp")
            base = {"uuid": obj.get("uuid"), "parent": obj.get("parentUuid"),
                    "session_id": obj.get("sessionId") or obj.get("session_id"),
                    "agent_id": obj.get("agentId"),
                    "sidechain": obj.get("isSidechain")}

            if typ in ("queue-operation", "last-prompt"):
                text = obj.get("content") or obj.get("lastPrompt") or ""
                yield Event("prompt", i, digest, ts, tier=TIER_T5,
                            text=text if isinstance(text, str) else "",
                            meta=dict(base, role="orchestrator"))
                continue

            if typ == "file-history-snapshot" or typ == "file-history-delta":
                yield Event("discontinuity", i, digest, ts,
                            meta=dict(base, why="file-history"))
                continue

            msg = obj.get("message")
            msg = msg if isinstance(msg, dict) else {}
            content = msg.get("content")
            blocks = content if isinstance(content, list) else []

            if typ == "assistant":
                # A session_meta equivalent: claude repeats cwd/gitBranch on
                # every line, so the first one we see is the header.
                if obj.get("cwd"):
                    yield Event("session-meta", i, digest, ts, meta=dict(
                        base, cwd=obj.get("cwd"),
                        git_branch=obj.get("gitBranch"),
                        cli_version=obj.get("version"),
                        model=msg.get("model")))
                for b in blocks:
                    if not isinstance(b, dict):
                        continue
                    bt = b.get("type")
                    if bt in ("text", "thinking"):
                        yield Event("agent-text", i, digest, ts, tier=TIER_T4,
                                    text=b.get(bt) or "", meta=dict(base))
                    elif bt == "tool_use":
                        name = b.get("name")
                        inp = b.get("input") or {}
                        inp = inp if isinstance(inp, dict) else {}
                        cmd = (inp.get("command") or inp.get("file_path")
                               or json.dumps(inp, ensure_ascii=False)[:400])
                        yield Event("tool-call", i, digest, ts, text=str(cmd),
                                    meta=dict(base, name=name,
                                              tool_use_id=b.get("id"),
                                              input=inp))
                        if name in EDIT_TOOLS:
                            pending_edit[b.get("id")] = (
                                i, inp.get("file_path") or "", name, inp)
                continue

            if typ == "user":
                res = obj.get("toolUseResult")
                for b in blocks:
                    if not isinstance(b, dict) or b.get("type") != "tool_result":
                        continue
                    tid = b.get("tool_use_id")
                    text = b.get("content")
                    text = text if isinstance(text, str) else _join_text(text)
                    edit = pending_edit.pop(tid, None)
                    if edit:
                        yield _claude_edit_event(i, digest, ts, base, edit, res)
                        continue
                    tier = TIER_T1
                    meta = dict(base, tool_use_id=tid)
                    if isinstance(res, dict):
                        pop = res.get("persistedOutputPath")
                        if pop:
                            meta["persisted_output"] = pop
                        if res.get("filePath"):
                            # A Read: the agent is looking at something it or the
                            # harness wrote, so the numbers inside are echoes.
                            meta["read_path"] = res["filePath"]
                            tier = TIER_T2
                    if SELF_READ_RE.search(text[:300]):
                        tier = TIER_T2
                    yield Event("tool-output", i, digest, ts, tier=tier,
                                text=text, meta=meta)
                if not blocks:
                    text = content if isinstance(content, str) else ""
                    if text:
                        yield Event("prompt", i, digest, ts, tier=TIER_T5,
                                    text=text, meta=dict(base, role="user"))


def _claude_edit_event(line_no, digest, ts, base, edit, result):
    """Build an edit Event, preferring the harness's own patch over ours."""
    edit_line, fpath, tool, inp = edit
    hunks = None
    if isinstance(result, dict):
        sp = result.get("structuredPatch")
        if isinstance(sp, list) and sp:
            hunks = sp
    # The post-edit text, kept verbatim beside the diff: the version documents
    # (`memory/vN.json`) arrive through this channel and a caller that had to
    # un-prefix them out of a synthesized diff would be parsing our own
    # reconstruction instead of what was written.
    new_text = inp.get("content") if tool == "Write" else inp.get("new_string")
    if hunks:
        diff, channel, verbatim = (render_structured_patch(fpath, hunks),
                                   "structured-patch", True)
    elif tool == "Write":
        diff, channel, verbatim = (
            synth_diff_from_strings(fpath, "", inp.get("content") or ""),
            "write", False)
    else:
        diff, channel, verbatim = (
            synth_diff_from_strings(fpath, inp.get("old_string") or "",
                                    inp.get("new_string") or ""),
            "edit-strings", False)
    files = {fpath: {"channel": channel, "diff": diff, "verbatim": verbatim,
                     "new_text": new_text or ""}}
    return Event("edit", line_no, digest, ts, meta=dict(
        base, files=files, tool=tool,
        code_files=[f for f in files if is_code_path(f)],
        call_line=edit_line))


# ------------------------------------------------------------------ dispatch

def detect_format(path):
    """codex vs claude-code, from the first line that parses.

    A filename test would be wrong: the claude corpora name files by session
    uuid, and one codex rollout sits loose in the same tree.
    """
    with open(path, "rb") as fh:
        for _ in range(20):
            raw = fh.readline()
            if not raw:
                break
            try:
                obj = json.loads(raw.decode("utf-8", "replace"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(obj, dict):
                continue
            if "payload" in obj and "type" in obj:
                return "codex"
            if obj.get("type") in ("assistant", "user", "queue-operation",
                                   "last-prompt", "attachment", "system",
                                   "mode", "permission-mode", "ai-title"):
                return "claude-code"
    return None


def parse(path):
    """(format, [Event, ...]) for one transcript."""
    fmt = detect_format(path)
    if fmt == "codex":
        return fmt, list(parse_codex(path))
    if fmt == "claude-code":
        return fmt, list(parse_claude(path))
    return None, []


def iter_transcripts(root):
    """Every transcript under `root`, AppleDouble and .DS_Store excluded.

    `._*` files are Mac resource forks: they parse as garbage and would double
    the apparent corpus size.
    """
    root = Path(root)
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*.jsonl")):
        if p.name.startswith("._") or p.name == ".DS_Store":
            continue
        yield p
