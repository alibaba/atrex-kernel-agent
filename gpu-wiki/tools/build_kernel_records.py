#!/usr/bin/env python3
"""Seed the kernel-experience store (schema clean-1.3) from the curated markdown.

ONE-SHOT MIGRATION TOOL. It projects the curated markdown wiki that used to live
in this repository into ``kernel_wiki/records/``. Once seeding is done the
markdown is removed, so this script has no input left: from then on records are
mined from optimization traces and agent sessions by the skills under
``skills/``, and admitted by the wiki-gate skill. It is kept for provenance and
so the seed can be replayed against an external checkout via ``--docs-root``.

Mapping (kernel store only):

    ref-docs / converter   -> doc             (whole page, one record)
    kernel-opt             -> technique-card  (one per H2 section)
    pitfalls               -> anti-strategy   (one per trap)
    (aggregated)           -> symptom-card    (one per profiler symptom, per scope)

Hardware FACTS are a separate store with a separate schema and are not built
here: ``hardware-specs/``, ``kernel-opt/hardware/`` and the PTX ISA page are
excluded and handled by ``build_hardware_records.py``.

Nothing is invented. Scope comes from the path and from ``manifest.json``; prose
and code are copied verbatim. A field the page does not support is left out
rather than guessed.

Usage:
    python3 build_kernel_records.py --sample         # preview, nothing written
    python3 build_kernel_records.py --all --clean     # full seed
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GPU_WIKI = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(GPU_WIKI)
DOCS_ROOT = os.path.join(REPO_ROOT, "gpu-wiki-md", "docs")
RECORDS_ROOT = os.path.join(GPU_WIKI, "kernel_wiki", "records")
SCHEMA_VERSION = "clean-1.3"
SCHEMA = SCHEMA_VERSION  # alias for tools that index this store

ROLES = {"hardware-specs", "kernel-opt", "ref-docs", "pitfalls", "converter"}
TRUE_ARCH = {"ampere", "hopper", "blackwell", "blackwell-ultra",
             "blackwell-geforce", "cdna3", "cdna4", "rdna4"}
DSLS = {"cuda", "cutedsl", "flydsl", "gluon", "triton", "aiter"}
VENDORS = {"nvidia", "amd"}

ROLE_TO_TYPE = {
    "ref-docs": "doc",
    "converter": "doc",
    "kernel-opt": "technique-card",
    "pitfalls": "anti-strategy",
}

# Pages whose content is hardware FACT, owned by the hardware store.
HARDWARE_ONLY = re.compile(
    r"(^|/)hardware-specs/|(^|/)kernel-opt/hardware/|(^|/)languages/ptx-sm100\.md$")

# H2 headings that are boilerplate rather than a technique.
BOILERPLATE_HEADINGS = re.compile(
    r"^\s*(end[\s-]?user\s+licen[cs]e(\s+agreement)?|licen[cs]e(\s+agreement)?|eula|"
    r"description|usage|overview|introduction|"
    r"references?(\s+document(s|ation)?)?|"
    r"related(\s+(docs?|document(s|ation)?|reading|links?))?|"
    r"further\s+reading|cross[\s-]?references?|see\s+also|"
    r"table\s+of\s+contents|contents|changelog|acknowledge?ments?|credits|"
    r"disclaimer|copyright)\s*$", re.IGNORECASE)

HW_LABEL = {
    "ampere": "NVIDIA Ampere (A100 / SM80)",
    "hopper": "NVIDIA Hopper (H20 / H100 / H200 / SM90)",
    "blackwell": "NVIDIA Blackwell (SM100)",
    "blackwell-ultra": "NVIDIA Blackwell Ultra (B300 / SM103)",
    "blackwell-geforce": "NVIDIA Blackwell GeForce/workstation (RTX PRO 5000 / SM120)",
    "cdna3": "AMD CDNA3 (gfx942)",
    "cdna4": "AMD CDNA4 (gfx950 / MI355X)",
    "rdna4": "AMD RDNA4 (gfx1250)",
    "generic": "GPU (vendor/architecture-general)",
}
PRODUCT_LABEL = {
    "b200": "NVIDIA B200 (GB200 / SM100)",
    "b300": "NVIDIA B300 (GB300 / SM103)",
    "mi300x": "AMD MI300X (CDNA3 / gfx942)",
    "mi308x": "AMD MI308X (CDNA3 / gfx942)",
    "mi355x": "AMD MI355X (CDNA4 / gfx950)",
    "sm120": "NVIDIA RTX PRO 5000 (SM120)",
}

# --------------------------------------------------------------------------- #
# classification vocabularies, carried over from the markdown-era retrieval tool
# so the JSON store keeps the same hard filters the wiki used to support
# --------------------------------------------------------------------------- #
SYMPTOMS = {
    "compute-bound": {"compute bound", "compute-bound", "tensor core throughput"},
    "low-sm-utilization": {"low sm utilization", "low-sm-utilization",
                           "persistent kernel", "occupancy tuning"},
    "memory-bound": {"memory bound", "memory-bound", "memory bandwidth",
                     "coalesced access"},
    "moe-load-imbalance": {"moe load imbalance", "moe-load-imbalance",
                           "expert load imbalance"},
    "pipeline-stalls": {"pipeline stalls", "pipeline-stalls", "software pipeline",
                        "software pipelining", "pipeline depth"},
    "register-pressure": {"register pressure", "register-pressure",
                          "register spill", "vgpr spill"},
    "tail-effect": {"tail effect", "tail-effect", "wave quantization"},
}
KERNEL_TYPES = {
    "attention": {"attention", "flash attention", "flash-attention", "flashmla", "mla"},
    "gemm": {"gemm", "matmul", "matrix multiplication"},
    "gemv": {"gemv", "matrix vector"},
    "moe": {"moe", "mixture of experts"},
    "norm": {"norm", "rmsnorm", "layernorm"},
    "reduction": {"reduction", "softmax"},
}
OPERATORS = {
    "activation": {"activation", "silu", "gelu"},
    "allreduce": {"allreduce", "all reduce"},
    "conv": {"conv", "convolution"},
    "cross-entropy": {"cross entropy", "cross-entropy"},
    "elementwise": {"elementwise", "vector add", "vectoradd"},
    "flash-attention": {"flash attention", "flash-attention", "flashattention",
                        "flash attn", "fmha"},
    "gdn": {"gdn", "gated delta net", "gated-delta-net"},
    "gemm": {"gemm", "matmul"},
    "gemv": {"gemv"},
    "grouped-gemm": {"grouped gemm", "grouped-gemm"},
    "mamba": {"mamba", "state space model", "ssm"},
    "mla": {"mla", "flashmla", "multi head latent attention"},
    "moe": {"moe", "mixture of experts"},
    "norm": {"rmsnorm", "layernorm", "rms norm", "layer norm"},
    "paged-attention": {"paged attention", "paged-attention"},
    "quantization": {"quantization", "quantize", "quant"},
    "rope": {"rope", "rotary"},
    "softmax": {"softmax"},
    "sort": {"sort", "sorting"},
    "topk": {"topk", "top k"},
}
KERNEL_TYPE_TO_WORKLOAD = {
    "attention": "attention", "gemm": "gemm-projection", "gemv": "gemm-projection",
    "moe": "moe", "norm": "norm", "reduction": "misc",
}
OPERATOR_TO_WORKLOAD = {
    "activation": "mlp-activation", "conv": "conv-vision",
    "flash-attention": "attention", "gdn": "ssm-linear-attention",
    "gemm": "gemm-projection", "gemv": "gemm-projection",
    "grouped-gemm": "gemm-projection", "mamba": "ssm-linear-attention",
    "mla": "attention", "moe": "moe", "norm": "norm",
    "paged-attention": "attention", "rope": "rope", "softmax": "misc",
    "topk": "mask-index", "sort": "mask-index",
}

# The schema's verdict vocabulary has no "unknown", so a documented trap must be
# classified. Priority order: a correctness failure outranks an API limit, which
# outranks a performance ceiling.
VERDICT_RULES = (
    ("accuracy-gate/ceiling", re.compile(
        r"correctness|accuracy|numerical|nan\b|inf\b|wrong|incorrect|"
        r"silently (?:drops|computes)|all zeros|mismatch|precision loss|tolerance",
        re.I)),
    ("api-limitation", re.compile(
        r"not supported|unsupported|no-op|ignored by|codegen|compiler|importerror|"
        r"incompatible|api limitation|not available|cannot be expressed|no way to",
        re.I)),
    ("performance-ceiling", re.compile(
        r"slower|regress|no speedup|does not improve|ceiling|saturat|bound by|"
        r"no gain|worse|degrad", re.I)),
)


def is_hardware_page(relpath: str) -> bool:
    return bool(HARDWARE_ONLY.search(relpath))


# --------------------------------------------------------------------------- #
# path / scope
# --------------------------------------------------------------------------- #
def parse_path(relpath: str) -> dict | None:
    parts = relpath.split("/")
    if len(parts) < 2:
        return None
    vendor = parts[0] if parts[0] in VENDORS else "generic"

    role_idx = next((i for i, seg in enumerate(parts) if seg in ROLES), None)
    if role_idx is None:
        return None
    role = parts[role_idx]

    arch, product = "generic", None
    for seg in parts[1:role_idx]:
        if seg in TRUE_ARCH:
            arch = seg
        elif seg == "common":
            arch = "generic"
        else:
            product = seg

    after = parts[role_idx + 1:-1]
    dsl = next((seg for seg in after if seg in DSLS), "any")
    topic = next((seg for seg in after if seg != dsl), "any")

    return {"vendor": vendor, "arch": arch, "product": product, "dsl": dsl,
            "role": role, "slug": os.path.splitext(parts[-1])[0],
            "subpath": after, "topic": topic}


def load_docs_manifest(docs_root: str) -> tuple[list[dict], dict]:
    """Scope manifest for the seed pages.

    It travels with the markdown, so look beside the docs root first; the
    in-repo copy is only there while the retired tree is still present.
    """
    for path in (os.path.join(os.path.dirname(docs_root), "manifest.json"),
                 os.path.join(GPU_WIKI, "manifest.json")):
        if os.path.isfile(path):
            docs = json.load(open(path, encoding="utf-8")).get("docs", {})
            return docs.get("defaults", []), docs.get("entries", {})
    return [], {}


DOCS_DEFAULTS, DOCS_ENTRIES = load_docs_manifest(DOCS_ROOT)


def manifest_architectures(relpath: str) -> list[str]:
    """Explicit architecture scope for a page; [] when path inference suffices.

    An exact entry wins over prefix defaults. This is what carries the
    cross-architecture pages that path inference alone would flatten into the
    vendor-general bucket.
    """
    entry = DOCS_ENTRIES.get(relpath)
    if entry is not None and entry.get("architectures"):
        return sorted(entry["architectures"])
    best, best_len = [], -1
    for default in DOCS_DEFAULTS:
        prefix = default.get("prefix", "")
        if relpath.startswith(prefix) and len(prefix) > best_len:
            best, best_len = sorted(default.get("architectures") or []), len(prefix)
    return best


def target_label(meta: dict) -> str:
    if meta["product"] and meta["product"] in PRODUCT_LABEL:
        return PRODUCT_LABEL[meta["product"]]
    return HW_LABEL.get(meta["arch"], HW_LABEL["generic"])


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-") or "section"


def sanitize_id(text: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", text.lower()).strip("-._") or "record"


def classify(stable_text: str, vocabulary: dict) -> list[str]:
    return sorted(name for name, markers in vocabulary.items()
                  if any(marker in stable_text for marker in markers))


def stable_text_of(relpath: str, title: str) -> str:
    """Title plus path words -- the same classification input the wiki used."""
    return "%s %s" % (title.lower(),
                      " ".join(re.findall(r"[a-z0-9]+", relpath.lower())))


# --------------------------------------------------------------------------- #
# markdown helpers
# --------------------------------------------------------------------------- #
def read(relpath: str) -> str:
    with open(os.path.join(DOCS_ROOT, relpath), encoding="utf-8") as fh:
        return fh.read()


def first_h1(md: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def strip_code_blocks(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def first_paragraph(body: str) -> str:
    para: list[str] = []
    for line in strip_code_blocks(body).splitlines():
        s = line.strip()
        if not s:
            if para:
                break
            continue
        if s.startswith("#"):
            if para:
                break
            continue
        para.append(s)
    return " ".join(para).strip()


def first_sentence(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    m = re.search(r"(.+?[.!?])(\s|$)", text)
    out = m.group(1) if m else text
    return out if len(out) <= limit else out[:limit - 1] + "\u2026"


_META_LINE = re.compile(r"^\s*(\*\*[^*]+\*\*\s*:?|last\s+updated|hardware\s*:|"
                        r"stack\s*:|distilled\s+from|source\s*:)", re.IGNORECASE)


def _is_table_line(line: str) -> bool:
    return line.strip().startswith("|")


def table_summary(body: str) -> str:
    for line in strip_code_blocks(body).splitlines():
        if _is_table_line(line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            cells = [c for c in cells if c and not set(c) <= set("-: ")]
            if cells:
                return "table columns: " + ", ".join(cells[:8])
    return ""


def doc_summary(md: str) -> str:
    """First substantive prose paragraph, skipping metadata lines and tables."""
    para: list[str] = []
    for line in strip_code_blocks(md).splitlines():
        s = line.strip()
        if not s:
            if para:
                break
            continue
        if (s.startswith("#") or _META_LINE.match(s) or _is_table_line(s)
                or s == "---"):
            if para:
                break
            continue
        para.append(s)
    return " ".join(para).strip()


def section_gist(body: str) -> str:
    para = first_paragraph(body)
    if para.startswith("|"):
        return table_summary(body) or first_sentence(para)
    return first_sentence(para) if para else ""


def split_h2(md: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    head: str | None = None
    body: list[str] = []
    in_fence = False
    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and line.startswith("## "):
            if head is not None:
                sections.append((head, "\n".join(body).strip()))
            head, body = line[3:].strip(), []
        elif head is not None:
            body.append(line)
    if head is not None:
        sections.append((head, "\n".join(body).strip()))
    return sections


def extract_bullets(body: str) -> list[str]:
    out = []
    for line in strip_code_blocks(body).splitlines():
        s = line.strip()
        if s.startswith(("- ", "* ")):
            out.append(s[2:].strip())
    return out


def labelled_block(body: str, label: str) -> str:
    """Text following a bold '**Label**:' marker inside a section."""
    body = strip_code_blocks(body)
    pat = re.compile(r"\*\*%s\*\*\s*:?\s*(.*)" % re.escape(label), re.IGNORECASE)
    lines = body.splitlines()
    for i, line in enumerate(lines):
        m = pat.search(line)
        if m:
            chunk = [m.group(1).strip()]
            for nxt in lines[i + 1:]:
                s = nxt.strip()
                if not s or re.match(r"\*\*[A-Za-z]", s) or s.startswith("#"):
                    break
                chunk.append(s)
            return " ".join(c for c in chunk if c).strip()
    return ""


def implementation_of(text: str) -> dict | None:
    """Verbatim code from a page or section, as a schema implementation object."""
    blocks = re.findall(r"(```.*?```)", text, flags=re.DOTALL)
    if not blocks:
        return None
    return {"snippet": "\n\n".join(b.strip() for b in blocks),
            "format": "source (verbatim)"}


def classify_verdict(*texts: str) -> str:
    blob = " ".join(t for t in texts if t)
    for verdict, pattern in VERDICT_RULES:
        if pattern.search(blob):
            return verdict
    return "not-worth-it-here"


_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# Sentence truncation can cut a link before its closing paren, so the tail form
# has to be handled too, along with the bare relative/tree paths left behind.
_MD_LINK_CUT = re.compile(r"\[([^\]]+)\]\([^)\s]*")
_MD_FILE = re.compile(r"\(?\b[\w][\w/.-]*\.md\b\)?")
_TREE_PATH = re.compile(r"(?:\.{1,2}/)+[\w./-]*|\b(?:docs|records)/[\w./-]*")


def strip_page_refs(text: str) -> str:
    """Remove cross-page references from anything the agent will read.

    The markdown tree is deleted once seeding completes, so a page path in a
    served field is a citation the agent can never resolve. Link text is kept;
    only the unusable target goes away.
    """
    if not text:
        return text
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_LINK_CUT.sub(r"\1", text)
    text = _MD_FILE.sub("", text)
    text = _TREE_PATH.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip(" ,;")


def qualitative_gain() -> dict:
    """Documentation states no measured benefit, so say exactly that."""
    return {"basis": "qualitative", "kind": "none", "pct": None, "metrics": []}


def heuristic_score(meta: dict, code_chars: int, prose_chars: int) -> float:
    role_w = {"ref-docs": 0.20, "converter": 0.15, "kernel-opt": 0.25,
              "pitfalls": 0.25}
    score = 0.35 + role_w.get(meta["role"], 0.1)
    if meta["product"]:
        score += 0.15
    elif meta["arch"] != "generic":
        score += 0.08
    if code_chars:
        score += min(0.12, code_chars / 8000.0)
    if prose_chars > 400:
        score += 0.05
    return round(min(score, 0.98), 3)


# --------------------------------------------------------------------------- #
# record envelope
# --------------------------------------------------------------------------- #
def base_record(meta: dict, rtype: str, id_suffix: str, family: str | None,
                title: str = "") -> dict:
    arch = meta["arch"]
    level = "generic" if arch == "generic" else "cross-operator"
    rid = sanitize_id(".".join([meta["vendor"], meta["product"] or arch,
                                meta["dsl"], meta["role"], meta["slug"]]))
    if id_suffix:
        rid = sanitize_id(rid + "." + id_suffix)

    stable = stable_text_of(meta["relpath"], title or meta["slug"])
    operators = classify(stable, OPERATORS)
    kernel_types = classify(stable, KERNEL_TYPES)
    symptoms = classify(stable, SYMPTOMS)

    if family is None:
        if operators:
            family = operators[0]
        elif meta["topic"] != "any":
            family = meta["topic"]

    workload = "any"
    for op in operators:
        if op in OPERATOR_TO_WORKLOAD:
            workload = OPERATOR_TO_WORKLOAD[op]
            break
    else:
        for kt in kernel_types:
            if kt in KERNEL_TYPE_TO_WORKLOAD:
                workload = KERNEL_TYPE_TO_WORKLOAD[kt]
                break

    archs = manifest_architectures(meta["relpath"])
    if not archs and arch != "generic":
        archs = [arch]

    return {
        "schema": SCHEMA_VERSION,
        "id": rid,
        "type": rtype,
        "level": level,
        "status": "active",
        "episode_key": "|".join([
            arch, meta["dsl"], family or meta["role"],
            meta["slug"] + (("#" + id_suffix) if id_suffix else ""), level]),
        "retrieval": {
            "scope": {
                "vendor": meta["vendor"],
                "arch": arch,
                "architectures": archs,
                "product": meta["product"] or "any",
                "dsl": meta["dsl"],
                "operator_family": family,
                "operators": operators,
                "shape_signature": {},
            },
            "generality": {
                "arch": "any" if arch == "generic" else arch,
                "language": meta["dsl"],
                "workload_family": workload,
            },
            "signals": {
                "metrics": {},
                "shape_regime": {"predicate": "*", "var_axes": []},
                "symptoms": symptoms,
            },
            "technique_tags": sorted(set(operators) | set(kernel_types)),
            "triggers": [],
        },
        "payload": {},
        "evidence": {
            "summary": {"confidence": "documented"},
            "raw": {
                "page_path": meta["relpath"],
                "seed": "curated markdown wiki, removed after the JSON migration",
            },
        },
        "worth": {"rank": {"score": 0.0, "tier": "provisional"},
                  "gain": qualitative_gain()},
    }


# --------------------------------------------------------------------------- #
# per-type builders
# --------------------------------------------------------------------------- #
def build_doc(meta: dict, md: str) -> list[dict]:
    title = first_h1(md) or meta["slug"]
    summary = doc_summary(md) or table_summary(md) or title
    sections = [{"heading": h, "anchor": slugify(h),
                 "gist": strip_page_refs(section_gist(b)) if b else ""}
                for h, b in split_h2(md)]

    rec = base_record(meta, "doc", "", None, title=title)
    tgt = target_label(meta)
    rec["payload"] = {
        "goal": "Reference for %s: %s" % (tgt, title),
        "problem": {"target": tgt, "statement": title},
        "title": title,
        "summary": strip_page_refs(summary),
        "sections": sections,
    }
    impl = implementation_of(md)
    if impl:
        rec["payload"]["implementation"] = impl
    rec["retrieval"]["triggers"] = [
        "%s reference for %s is needed" % (meta["role"], tgt)]
    code = sum(len(m) for m in re.findall(r"```.*?```", md, flags=re.DOTALL))
    rec["worth"]["rank"]["score"] = heuristic_score(meta, code, len(md))
    return [rec]


def build_technique_cards(meta: dict, md: str) -> list[dict]:
    title = first_h1(md) or meta["slug"]
    tgt = target_label(meta)
    sections = split_h2(md) or [(title, md.split("\n", 1)[-1])]
    recs = []
    for head, body in sections:
        if BOILERPLATE_HEADINGS.match(head):
            continue
        what = first_paragraph(body)
        if not what:
            m = re.search(r"```[a-zA-Z0-9]*\n(.*?)```", body, re.DOTALL)
            if not m:
                continue
            comment = next((ln.strip(" /#*") for ln in m.group(1).splitlines()
                            if ln.strip().startswith(("//", "#", "/*", "*"))), "")
            what = comment or ("Code pattern: %s" % head)
        caveats = [b for b in extract_bullets(body)
                   if re.search(r"caveat|but |not |avoid|don't|warning|note", b, re.I)]

        rec = base_record(meta, "technique-card", slugify(head), None, title=title)
        rec["payload"] = {
            "goal": "Optimization technique for %s: %s" % (tgt, head),
            "problem": {"target": tgt,
                        "statement": "%s -- %s" % (title, head)},
            "technique": head,
            "what": strip_page_refs(what),
            "when": strip_page_refs(first_sentence(what)),
            "caveats": [strip_page_refs(c) for c in caveats],
            "success_rate_pct": None,
            "typical_gain_pct": None,
        }
        impl = implementation_of(body)
        if impl:
            rec["payload"]["implementation"] = impl
        rec["retrieval"]["triggers"] = [
            "the '%s' technique is being considered on %s" % (head, tgt)]
        rec["retrieval"]["technique_tags"] = sorted(
            set(rec["retrieval"]["technique_tags"]) | {slugify(head)})
        code = sum(len(m) for m in re.findall(r"```.*?```", body, flags=re.DOTALL))
        rec["worth"]["rank"]["score"] = heuristic_score(meta, code, len(body))
        recs.append(rec)
    return recs


ARCH_TO_SM = {"ampere": "sm_80", "hopper": "sm_90", "blackwell": "sm_100",
              "blackwell-ultra": "sm_103", "blackwell-geforce": "sm_120"}

MIN_MECHANISM_CHARS = 40
# Mirrors the established-fact gate: these phrasings report a measurement rather
# than name a cause, and a measurement must not enter the store as a hard limit.
NON_MECHANISM = re.compile(
    r"no improvement (?:found|over)|flat[- ]within[- ]noise|within noise|"
    r"all (?:\w+\s+){0,3}(?:approaches|trials|variants|attempts) "
    r"(?:tested|were|failed|flat)|tested \d+ approaches|reverted to an earlier",
    re.I)


def established_fact(meta: dict, root_cause: str, observed: str,
                     lesson: str) -> dict | None:
    """State the trap as a fact, or return None when no cause is documented.

    A negative record is only worth keeping when the failure is a law under a
    checkable condition. Without that, one unmeasured run reads like a hard limit
    and steers later agents away from a direction that may well work.
    """
    condition = {}
    if meta["arch"] in ARCH_TO_SM:
        condition["sm_arch"] = ARCH_TO_SM[meta["arch"]]
    if meta["dsl"] != "any":
        condition["toolchain"] = meta["dsl"]
    if not condition:
        return None
    for candidate in (root_cause, observed, lesson):
        text = strip_page_refs(candidate or "")
        if len(text) >= MIN_MECHANISM_CHARS and not NON_MECHANISM.search(text):
            return {"condition": condition, "mechanism": text}
    return None


def _anti_record(meta: dict, title: str, head: str, id_suffix: str, tgt: str,
                 attempted: str, observed: str, root_cause: str, lesson: str,
                 body: str) -> dict | None:
    fact = established_fact(meta, root_cause, observed, lesson)
    if fact is None:
        return None
    rec = base_record(meta, "anti-strategy", id_suffix, None, title=title)
    payload = {
        "goal": "Avoid a known trap on %s: %s" % (tgt, head),
        "problem": {"target": tgt,
                    "statement": "%s -- %s" % (title, head)
                                 if head != title else title},
        "attempted": strip_page_refs(attempted),
        "verdict": classify_verdict(head, attempted, observed, root_cause, lesson),
        "lesson": strip_page_refs(lesson),
        "established_fact": fact,
    }
    if observed:
        payload["observed"] = strip_page_refs(observed)
    if root_cause:
        payload["root_cause"] = strip_page_refs(root_cause)
    impl = implementation_of(body)
    if impl:
        payload["implementation"] = impl
    rec["payload"] = payload
    rec["retrieval"]["triggers"] = [
        "a kernel on %s risks the '%s' trap" % (tgt, head)]
    rec["worth"]["rank"]["tier"] = "cautionary"
    code = sum(len(m) for m in re.findall(r"```.*?```", body, flags=re.DOTALL))
    rec["worth"]["rank"]["score"] = heuristic_score(meta, code, len(body))
    return rec


def build_anti_strategies(meta: dict, md: str) -> list[dict]:
    title = first_h1(md) or meta["slug"]
    tgt = target_label(meta)
    h2 = split_h2(md)

    numbered = [(h, b) for h, b in h2 if re.match(r"^\d+\.", h)]
    if numbered:
        recs = []
        for head, body in numbered:
            attempted = labelled_block(body, "Trap") or first_paragraph(body)
            if not attempted:
                continue
            rec = _anti_record(
                meta, title, head, slugify(head), tgt, attempted,
                labelled_block(body, "Result"), labelled_block(body, "Why"),
                labelled_block(body, "Lesson") or head, body)
            if rec:
                recs.append(rec)
        return recs

    # Single-trap page using labelled headings (trap / symptom / why / lesson).
    secs = {slugify(h): b for h, b in h2}

    def pick(*keys) -> str:
        for k in keys:
            if k in secs and first_paragraph(secs[k]):
                return first_paragraph(secs[k])
        return ""

    attempted = pick("trap") or doc_summary(md) or first_paragraph(md)
    if not attempted:
        return []
    rec = _anti_record(
        meta, title, title, "", tgt, attempted,
        pick("symptom", "reality", "result"), pick("why", "root-cause", "cause"),
        pick("lesson", "takeaway") or title, md)
    return [rec] if rec else []


BUILDERS = {"doc": build_doc, "technique-card": build_technique_cards,
            "anti-strategy": build_anti_strategies}


def synthesize_symptom_cards(records: list[dict]) -> list[dict]:
    """Turn the built records into symptom -> candidate-lever entry points.

    An agent profiles first and then retrieves by symptom, so it needs one record
    that maps that symptom to an ordered list of levers. Nothing is measured
    here: documentation states no success rates, so those stay null and the order
    falls back to the documentation-derived rank.
    """
    buckets: dict[tuple[str, str, str], list[dict]] = {}
    for rec in records:
        scope = rec["retrieval"]["scope"]
        for symptom in rec["retrieval"]["signals"].get("symptoms") or []:
            buckets.setdefault((scope["vendor"], scope["arch"], symptom), []).append(rec)

    cards = []
    for (vendor, arch, symptom), members in sorted(buckets.items()):
        tags: dict[str, float] = {}
        for rec in members:
            score = (rec.get("worth", {}).get("rank", {}) or {}).get("score", 0.0)
            for tag in rec["retrieval"].get("technique_tags") or []:
                tags[tag] = max(tags.get(tag, 0.0), score)
        if len(tags) < 2:
            continue
        ordered = sorted(tags.items(), key=lambda kv: (-kv[1], kv[0]))

        meta = {"vendor": vendor, "arch": arch, "product": None, "dsl": "any",
                "role": "symptom", "slug": symptom, "topic": "any", "subpath": [],
                "relpath": members[0]["evidence"]["raw"]["page_path"]}
        rec = base_record(meta, "symptom-card", "", symptom, title=symptom)
        tgt = HW_LABEL.get(arch, HW_LABEL["generic"])
        rec["payload"] = {
            "goal": "Choose a lever after profiling shows %s on %s" % (symptom, tgt),
            "problem": {
                "target": tgt,
                "statement": "The profile points at %s; which documented "
                             "techniques address it on %s?" % (symptom, tgt),
                "bottleneck": symptom,
            },
            "likely_causes": [],
            "candidate_techniques": [
                {"technique": tag, "success_rate_pct": None,
                 "typical_gain_pct": None} for tag, _ in ordered],
        }
        rec["retrieval"]["signals"]["symptoms"] = [symptom]
        rec["retrieval"]["technique_tags"] = [t for t, _ in ordered]
        rec["retrieval"]["triggers"] = [
            "the profile shows %s and a candidate lever is needed" % symptom]
        rec["worth"]["rank"]["score"] = round(min(0.9, 0.4 + 0.05 * len(ordered)), 3)
        cards.append(rec)
    return cards


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def iter_docs() -> list[str]:
    out = []
    for dirpath, _dirs, files in os.walk(DOCS_ROOT):
        for name in files:
            if not name.endswith(".md") or name in ("README.md", "RELATIONS.md"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), DOCS_ROOT)
            if not is_hardware_page(rel):
                out.append(rel)
    return sorted(out)


def build_records(relpaths: list[str]) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    skipped: list[str] = []
    for rel in relpaths:
        meta = parse_path(rel)
        if meta is None:
            skipped.append(rel)
            continue
        meta["relpath"] = rel
        rtype = ROLE_TO_TYPE.get(meta["role"])
        if rtype is None:
            skipped.append(rel)
            continue
        md = read(rel)
        built = BUILDERS[rtype](meta, md)
        # A page whose every section is boilerplate is still knowledge: keep it
        # whole as a doc rather than dropping it.
        if not built and rtype != "doc":
            built = build_doc(meta, md)
        records.extend(built)
    return records, skipped


def dedupe_ids(records: list[dict]) -> None:
    seen: dict[str, int] = {}
    for rec in records:
        rid = rec["id"]
        if rid in seen:
            seen[rid] += 1
            rec["id"] = "%s.%d" % (rid, seen[rid])
        else:
            seen[rid] = 0


def record_relpath(rec: dict) -> str:
    """<type>/<vendor>/<arch>/<dsl>/<operator-family-or-topic>/<id>.json.

    Derivable from the record alone, so the layout gate can re-derive and compare.
    """
    scope = rec["retrieval"]["scope"]
    page = (rec.get("evidence", {}).get("raw", {}) or {}).get("page_path", "")
    topic = "any"
    if page:
        meta = parse_path(page)
        if meta:
            topic = meta["topic"]
    return os.path.join(rec["type"], scope["vendor"], scope["arch"],
                        scope["dsl"], topic, rec["id"] + ".json")


def index_title(rec: dict) -> str:
    """The one-line label an index entry shows.

    Shared with tools/reindex_kernel_wiki.py so a record inserted by wiki-gate is
    indexed exactly the way seeding indexed it.
    """
    payload = rec["payload"]
    return (payload.get("title") or payload.get("technique")
            or payload["problem"]["statement"])


def search_text(rec: dict) -> str:
    """Precomputed ranking blob; the retrieval tool matches terms against this."""
    p = rec["payload"]
    parts = [rec["id"], p.get("title", ""), p.get("goal", ""),
             p.get("technique", ""), p.get("what", ""), p.get("summary", ""),
             p.get("lesson", ""), p.get("attempted", "")]
    problem = p.get("problem") or {}
    parts += [problem.get("statement", ""), problem.get("operator") or ""]
    parts += [s.get("heading", "") for s in (p.get("sections") or [])]
    parts += rec["retrieval"].get("triggers") or []
    parts += rec["retrieval"].get("technique_tags") or []
    parts += rec["retrieval"]["signals"].get("symptoms") or []
    return " ".join(x for x in parts if x).lower()


def write_records(records: list[dict], clean: bool) -> None:
    if clean:
        for rtype in ("doc", "technique-card", "anti-strategy", "symptom-card"):
            d = os.path.join(RECORDS_ROOT, rtype)
            if os.path.isdir(d):
                shutil.rmtree(d)
    index = []
    for rec in records:
        rel = record_relpath(rec)
        path = os.path.join(RECORDS_ROOT, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=1)
        rank = rec["worth"]["rank"]
        index.append({
            "id": rec["id"],
            "type": rec["type"],
            "level": rec["level"],
            "status": rec["status"],
            "episode_key": rec["episode_key"],
            "path": os.path.join("records", rel),
            "title": index_title(rec),
            "retrieval": rec["retrieval"],
            "worth_score": rank["score"],
            "tier": rank["tier"],
            "gain_pct": (rec["worth"].get("gain") or {}).get("pct"),
            "page_path": rec["evidence"]["raw"].get("page_path"),
            "search_text": search_text(rec),
        })
    os.makedirs(RECORDS_ROOT, exist_ok=True)
    with open(os.path.join(RECORDS_ROOT, "index.json"), "w", encoding="utf-8") as fh:
        json.dump({"schema": SCHEMA_VERSION,
                   "generated_by": "build_kernel_records.py",
                   "builder_version": "1.0",
                   "count": len(index),
                   "records": index}, fh, ensure_ascii=False, indent=1)


def main() -> int:
    global DOCS_ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="seed every eligible page")
    ap.add_argument("--sample", action="store_true", help="preview, write nothing")
    ap.add_argument("--clean", action="store_true",
                    help="wipe the four kernel record roots first")
    ap.add_argument("--docs-root", default=None,
                    help="markdown docs root; the seed tree is removed after "
                         "migration, so pass an external checkout to replay")
    args = ap.parse_args()

    if args.docs_root:
        DOCS_ROOT = os.path.abspath(args.docs_root)
        global DOCS_DEFAULTS, DOCS_ENTRIES
        DOCS_DEFAULTS, DOCS_ENTRIES = load_docs_manifest(DOCS_ROOT)
    if not os.path.isdir(DOCS_ROOT):
        print("no markdown docs root at %s -- this is a one-shot seeding tool; "
              "pass --docs-root <checkout of the pre-migration docs tree>"
              % DOCS_ROOT, file=sys.stderr)
        return 3

    paths = iter_docs()
    records, skipped = build_records(paths)
    records.extend(synthesize_symptom_cards(records))
    dedupe_ids(records)

    if args.sample:
        print(json.dumps(records[:2], ensure_ascii=False, indent=1))
        print("\n%d pages -> %d records (nothing written)" % (len(paths), len(records)),
              file=sys.stderr)
        return 0
    if not args.all:
        ap.error("pass --all to seed or --sample to preview")

    write_records(records, clean=args.clean)
    counts: dict[str, int] = {}
    for rec in records:
        counts[rec["type"]] = counts.get(rec["type"], 0) + 1
    print("seeded %d records from %d pages: %s" % (len(records), len(paths), counts))
    if skipped:
        print("skipped %d pages with no kernel-side role" % len(skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
