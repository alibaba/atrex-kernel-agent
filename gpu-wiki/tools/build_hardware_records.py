#!/usr/bin/env python3
"""Seed the hardware-facts store (schema hw-1.0) from the curated markdown.

ONE-SHOT MIGRATION TOOL, the hardware half of the seeding pair. See
``build_kernel_records.py`` for the other half and for why both are one-shot.

This store holds FACTS, not experience: vendor and ISA definitions that our
benchmarks cannot falsify. The split follows one test -- can a measurement prove
this sentence wrong?

    cannot be falsified -> fact  -> here (peak numbers, syntax, capacities)
    can be falsified    -> experience -> kernel_wiki (what was tried, what won)

So only the fact-bearing pages are read here:

    hardware-specs/                    -> spec-sheet   (one per page)
    kernel-opt/hardware/               -> arch-feature (one per page)
    kernel-opt/languages/ptx-sm100.md  -> instruction  (one per ISA family)

Every number keeps its evidence class. These pages are third-party curated
documentation rather than vendor datasheets, so everything is recorded as
``architecture-analysis`` -- treated as provisional, with runtime device
attributes preferred. Nothing is upgraded to ``vendor-published`` on our word.

A field the page does not state is OMITTED rather than set to null, because the
fabrication gate requires every null to carry an explanation of how to obtain it,
and a silently invented peak would poison every utilization computed from it.

Usage:
    python3 build_hardware_records.py --sample
    python3 build_hardware_records.py --all --clean
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
RECORDS_ROOT = os.path.join(GPU_WIKI, "hardware_wiki", "records")
SCHEMA_VERSION = "hw-1.0"
EVIDENCE_CLASS = "architecture-analysis"

VENDORS = {"nvidia", "amd"}
TRUE_ARCH = {"ampere", "hopper", "blackwell", "blackwell-ultra",
             "blackwell-geforce", "cdna3", "cdna4", "rdna4"}
PRODUCTS = {"b200", "b300", "mi300x", "mi308x", "mi355x", "sm120"}

# Same denylist the no-advice gate enforces; applied before we copy any prose.
ADVICE = re.compile(
    r"\b(?:usually faster|is faster than|we recommend|you should use|"
    r"best choice|outperforms|prefer(?:red)? (?:to use|using)|"
    r"success rate|retained in \d+%)\b", re.I)

ARCH_TO_SM = {"ampere": "sm_80", "hopper": "sm_90", "blackwell": "sm_100",
              "blackwell-ultra": "sm_103", "blackwell-geforce": "sm_120"}
ARCH_TO_CC = {"ampere": "8.0", "hopper": "9.0", "blackwell": "10.0",
              "blackwell-ultra": "10.3", "blackwell-geforce": "12.0"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def scrub_advice(text: str) -> str:
    """Drop sentences that read as a recommendation.

    Comparative guidance is measured knowledge and belongs in the experience
    store; letting it in here would both fail the no-advice gate and blur the
    fact/experience boundary the two stores exist to keep apart.
    """
    kept = [s for s in re.split(r"(?<=[.;])\s+", text) if not ADVICE.search(s)]
    return " ".join(kept).strip()


def read(relpath: str) -> str:
    with open(os.path.join(DOCS_ROOT, relpath), encoding="utf-8") as fh:
        return fh.read()


def first_h1(md: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def strip_code(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def split_sections(md: str, level: int) -> list[tuple[str, str]]:
    marker = "#" * level + " "
    out: list[tuple[str, str]] = []
    head: str | None = None
    body: list[str] = []
    in_fence = False
    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and line.startswith(marker):
            if head is not None:
                out.append((head, "\n".join(body).strip()))
            head, body = line[len(marker):].strip(), []
        elif head is not None:
            body.append(line)
    if head is not None:
        out.append((head, "\n".join(body).strip()))
    return out


def paragraphs(body: str) -> list[str]:
    out, cur = [], []
    for line in strip_code(body).splitlines():
        s = line.strip()
        if not s or s.startswith(("#", "|", ">", "-", "*")):
            if cur:
                out.append(" ".join(cur))
                cur = []
            continue
        cur.append(s)
    if cur:
        out.append(" ".join(cur))
    return out


def parse_tables(body: str) -> list[list[list[str]]]:
    """Every markdown table in a body, as rows of trimmed cells."""
    tables, rows = [], []
    for line in strip_code(body).splitlines():
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if cells and all(set(c) <= set("-: ") for c in cells if c):
                continue                      # separator row
            rows.append(cells)
        elif rows:
            tables.append(rows)
            rows = []
    if rows:
        tables.append(rows)
    return tables


def kv_from_table(table: list[list[str]]) -> dict[str, str]:
    """A two-column Parameter/Value table as a mapping."""
    out = {}
    for row in table[1:] if len(table) > 1 else []:
        if len(row) >= 2 and row[0]:
            out[re.sub(r"\*+", "", row[0]).strip()] = row[1].strip()
    return out


def num(text: str) -> float | None:
    m = re.search(r"(\d[\d,]*\.?\d*)", text.replace(",", "") if text else "")
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    return int(value) if value.is_integer() else value


def clean_label(text: str) -> str:
    return re.sub(r"\*+", "", text).strip()


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-") or "item"


def path_identity(relpath: str) -> dict:
    parts = relpath.split("/")
    vendor = parts[0] if parts[0] in VENDORS else "generic"
    arch, product = "generic", None
    for seg in parts[1:-1]:
        if seg in TRUE_ARCH:
            arch = seg
        elif seg in PRODUCTS:
            product = seg
    if product is None:
        # Not every product page sits in a product overlay directory; several
        # name the part in the filename instead (hardware_specs_mi355x.md), and
        # that is what a --product lookup will ask for.
        stem = os.path.splitext(parts[-1])[0].lower()
        product = next((name for name in PRODUCTS if name in stem), None)
    return {"vendor": vendor, "arch": arch, "product": product}


def provenance(relpath: str, title: str) -> dict:
    """Schema forbids extra keys here, so the caveat lives in facts, not a note."""
    return {
        "evidence_class": EVIDENCE_CLASS,
        "sources": [{"title": title or relpath,
                     "kind": "third-party-analysis",
                     "ref": relpath}],
    }


# --------------------------------------------------------------------------- #
# spec-sheet
# --------------------------------------------------------------------------- #
DTYPE_ALIASES = (
    ("fp64", r"fp64"), ("fp32", r"fp32"), ("tf32", r"tf32"),
    ("bf16", r"bf16|fp16\s*/\s*bf16"), ("fp16", r"fp16"),
    ("fp8", r"fp8"), ("fp4", r"fp4"), ("fp6", r"fp6"),
    ("int8", r"int8"), ("int4", r"int4"),
)

# A value cell can state that the number is unavailable. Parsing a digit out of
# such a sentence is the worst failure mode this store has: it silently poisons
# every utilization computed from it, so these cells become an `unavailable`
# entry instead of a value.
UNAVAILABLE = re.compile(
    r"not published|not disclosed|unpublished|not available|n/?a\b|unknown|"
    r"query the .*at runtime|verify with|read it from|must not be .*reused|"
    r"do not (?:infer|assume)", re.I)


MEMORY_FIELDS = (
    ("capacity_gb", r"^(vram|memory(\s+size|\s+capacity)?|hbm)"),
    ("bandwidth_tb_s", r"bandwidth"),
    ("l2_cache_mb", r"l2\s*cache"),
    ("bus_width_bits", r"bus\s*width|interface\s*width"),
)
COMPUTE_FIELDS = (
    ("sm_count", r"^(streaming multiprocessors|sms?\b|compute units|cus?\b)"),
    ("gpcs", r"graphics processing clusters|gpcs?"),
    ("cuda_cores_per_sm", r"cuda cores per sm|shader cores per"),
    ("cuda_cores", r"^cuda cores|^shader cores|^stream processors"),
    ("tensor_cores_per_sm", r"tensor cores per sm|matrix cores per"),
    ("tensor_cores", r"^tensor cores|^matrix cores"),
    ("rt_cores_per_sm", r"rt cores per sm"),
    ("rt_cores", r"^rt cores"),
    ("texture_units", r"texture units"),
    ("register_file_kb_per_sm", r"register file per sm"),
    ("register_file_kb_total", r"total register file"),
    ("shared_memory_kb_per_sm", r"shared memory capacity|shared memory per"),
    ("warp_size", r"warp size|wavefront size"),
)


def match_field(label: str, table: tuple) -> str | None:
    low = label.lower()
    for field, pattern in table:
        if re.search(pattern, low):
            return field
    return None


def row_dtype(label: str) -> str | None:
    """The precision a peak-table row is about, or None if the row is not one.

    Handles three label styles seen across the pages: a bare "FP4", an
    accumulate-qualified "FP4 (Tensor Core, FP32 Accumulate)" where the
    parenthetical must NOT decide the dtype, and "Peak Compute (BF16/FP16
    Matrix)" where it must.
    """
    low = clean_label(label).lower()
    primary = low.split("(")[0].strip() or low
    if primary.startswith("peak compute") and "(" in low:
        primary = low.split("(", 1)[1].rstrip(") ")
    if "tf32" in low:
        return "tf32"
    return next((name for name, pat in DTYPE_ALIASES
                 if re.search(pat, primary)), None)


def parse_peak_compute(md: str) -> list[dict]:
    """Peak throughput rows, one entry per precision the page states."""
    entries: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def dtype_rows(table: list[list[str]]) -> int:
        return sum(1 for row in table[1:]
                   if len(row) >= 2 and row_dtype(row[0])
                   and num(row[1]) is not None)

    tables = [tb for tb in parse_tables(md) if tb]
    # Prefer tables that declare themselves a throughput table. Unrelated tables
    # (feature matrices, ISA path notes) also list dtype names, and reading a
    # peak out of one of those silently poisons every utilization derived from it.
    declared = [tb for tb in tables
                if re.search(r"precision|data type|peak|tflops|tops",
                             " ".join(tb[0]).lower())]
    candidates = declared or ([max(tables, key=dtype_rows)] if tables else [])
    for table in candidates:
        if dtype_rows(table) < 2:
            continue
        header = " ".join(table[0]).lower()
        tops_col = "tops" in header
        for row in table[1:]:
            if len(row) < 2:
                continue
            label = clean_label(row[0])
            low = label.lower()
            # Match the PRIMARY dtype only: a label such as "FP4 (Tensor Core,
            # FP32 Accumulate)" also names its accumulate type, and matching on
            # any substring would file the FP4 peak under fp32.
            dtype = row_dtype(label)
            if dtype is None:
                continue
            value = row[1]
            dense = num(value)
            if dense is None:
                continue
            # Normalize peta-scale statements to the tera-scale unit the rest of
            # the store uses, or cross-part comparison silently breaks.
            scale = 1000 if re.search(r"\bp(?:flops|ops)\b", value, re.I) else 1
            dense *= scale
            kind = ("tensor-core" if ("tensor" in low or "matrix" in low)
                    else "cuda-core")
            # A page may state the same dtype for both pipelines (Hopper lists
            # TF32 tensor throughput and CUDA-core FP32 separately); keep both.
            if (dtype, kind) in seen:
                continue
            seen.add((dtype, kind))
            entry = {
                "dtype": dtype,
                "unit": "TOPS" if ("tops" in value.lower()
                                   or "pops" in value.lower()
                                   or (tops_col and dtype.startswith("int"))
                                   or dtype in ("fp4", "int8", "int4")) else "TFLOPS",
                "unit_kind": kind,
                "dense": dense,
                "provenance": EVIDENCE_CLASS,
            }
            sparse = None
            if len(row) >= 3:
                sparse = num(row[2])
                if sparse is not None:
                    sparse *= (1000 if re.search(r"\bp(?:flops|ops)\b", row[2],
                                                 re.I) else 1)
            elif "sparsit" in value.lower():
                # "2.5 PFLOPS dense; 5 PFLOPS with structured sparsity"
                tail = value.lower().split(";", 1)[-1]
                sparse = num(tail)
                if sparse is not None:
                    sparse *= scale
            if sparse is not None:
                entry["sparse_2to4"] = sparse
            if len(row) >= 4 and row[3] and row[3] != "—":
                use_case = scrub_advice(clean_label(row[3]))
                if use_case:
                    entry["use_case"] = use_case
            entries.append(entry)
    return entries


def parse_block(md: str, heading_re: str,
                fields: tuple) -> tuple[dict, dict]:
    """One facts sub-block (memory / compute_units), plus its unavailable notes."""
    block: dict = {}
    unavailable: dict = {}
    for level in (3, 2):
        for head, body in split_sections(md, level):
            if not re.search(heading_re, head, re.I):
                continue
            for table in parse_tables(body):
                for label, value in kv_from_table(table).items():
                    field = match_field(label, fields)
                    if not field or field in block or field in unavailable:
                        continue
                    if UNAVAILABLE.search(value):
                        # Keep the field as null AND explain it: an agent asking
                        # for this number must get the disposition, not an error.
                        block[field] = None
                        unavailable[field] = value.strip()
                        continue
                    parsed = num(value)
                    if parsed is not None:
                        block[field] = parsed
                    elif field == "kind":
                        block[field] = value
            if block or unavailable:
                break
        if block or unavailable:
            break
    return block, unavailable


def build_spec_sheet(relpath: str, md: str) -> dict | None:
    peak = parse_peak_compute(md)
    if not peak:
        return None                                # no numbers -> no record
    identity = path_identity(relpath)
    arch = identity["arch"]
    # Required by the schema even when unknown: AMD parts have no SM arch or CUDA
    # compute capability, and inventing one would be a fabricated fact.
    identity["sm_arch"] = ARCH_TO_SM.get(arch)
    identity["compute_capability"] = ARCH_TO_CC.get(arch)

    memory, mem_missing = parse_block(md, r"memory (specification|hierarchy)",
                                     MEMORY_FIELDS)
    compute, cu_missing = parse_block(md, r"(compute|execution) units",
                                      COMPUTE_FIELDS)
    memory["provenance"] = EVIDENCE_CLASS
    compute["provenance"] = EVIDENCE_CLASS
    unavailable = {**mem_missing, **cu_missing}

    title = first_h1(md) or relpath
    intro = next((p for p in paragraphs(md.split("\n## ", 1)[0]) if len(p) > 40), "")
    summary = scrub_advice(intro) or title

    slug = identity.get("product") or arch
    return {
        "schema": SCHEMA_VERSION,
        "id": ".".join([identity["vendor"], arch, "spec-sheet", slug]),
        "type": "spec-sheet",
        "status": "current",
        "identity": identity,
        "facts": {
            "summary": summary,
            "peak_compute": peak,
            "memory": memory,
            "compute_units": compute,
            **({"unavailable": unavailable} if unavailable else {}),
            "usage_notes": [
                "utilization = measured throughput / the peak that matches the "
                "kernel's dominant compute type.",
                "Values are architecture-analysis: verify against cudaDeviceProp "
                "or the vendor tool on the deployed GPU before hard-coding "
                "launch geometry.",
            ],
        },
        "provenance": provenance(relpath, title),
    }


# --------------------------------------------------------------------------- #
# arch-feature
# --------------------------------------------------------------------------- #
def build_arch_feature(relpath: str, md: str) -> dict | None:
    identity = path_identity(relpath)
    title = first_h1(md) or relpath
    sections = split_sections(md, 2)
    body_by_head = {slugify(h): b for h, b in sections}

    overview = body_by_head.get("overview", "")
    paras = paragraphs(overview) or paragraphs(md.split("\n## ", 1)[0])
    what = scrub_advice(paras[0]) if paras else ""
    if not what:
        return None
    why = scrub_advice(" ".join(paras[1:3])) if len(paras) > 1 else ""
    if not why:
        why = ("Defines a capability of %s that a kernel must target explicitly; "
               "the constraints below are hardware-imposed, not tuning choices."
               % identity["arch"])

    parameters: dict = {}
    for head, body in sections:
        if re.search(r"key propert|parameters|specification", head, re.I):
            for table in parse_tables(body):
                for k, v in kv_from_table(table).items():
                    cleaned = scrub_advice(v)
                    if cleaned:
                        parameters[slugify(k).replace("-", "_")] = cleaned
            break

    constraints: list[str] = []
    for head, body in sections:
        if re.search(r"requirement|constraint|limitation|rule", head, re.I):
            for line in strip_code(body).splitlines():
                s = line.strip()
                if s.startswith(("- ", "* ")):
                    item = scrub_advice(s[2:].strip())
                    if item:
                        constraints.append(item)

    mnemonics = sorted(set(re.findall(
        r"\b(?:tcgen05|cp\.async|mbarrier|clusterlaunchcontrol|cvt)[\w.]*", md)))

    facts = {"what": what, "why_it_matters": why}
    if parameters:
        facts["parameters"] = parameters
    if constraints:
        facts["constraints"] = constraints[:12]
    if mnemonics:
        facts["related_instructions"] = mnemonics[:12]

    feature = slugify(os.path.splitext(os.path.basename(relpath))[0])
    identity["feature"] = feature
    arch = identity["arch"]
    identity["sm_arch"] = ARCH_TO_SM.get(arch)
    identity["availability"] = {
        "sm_arch": [ARCH_TO_SM[arch]] if arch in ARCH_TO_SM else [],
        "products": [identity["product"]] if identity.get("product") else [],
    }
    return {
        "schema": SCHEMA_VERSION,
        "id": ".".join([identity["vendor"], arch, "arch-feature", feature]),
        "type": "arch-feature",
        "status": "current",
        "identity": identity,
        "facts": facts,
        "provenance": provenance(relpath, title),
    }


# --------------------------------------------------------------------------- #
# instruction
# --------------------------------------------------------------------------- #
PTX_LINE = re.compile(
    r"^\s*(?:@\S+\s+)?((?:tcgen05|cp|mbarrier|clusterlaunchcontrol|cvt|ld|st|"
    r"fence|wgmma|setmaxnreg)\.[\w.:]+)")


def build_instructions(relpath: str, md: str) -> list[dict]:
    """One record per ISA instruction family, keyed by its mnemonic stem.

    These sections open with a code block rather than prose, so semantics comes
    from the comment that documents each form in the listing.
    """
    identity_base = path_identity(relpath)
    families: dict[str, dict] = {}

    for head, body in split_sections(md, 2):
        for block in re.findall(r"```[a-zA-Z0-9]*\n(.*?)```", body, re.DOTALL):
            pending = ""
            for line in block.splitlines():
                bare = line.strip()
                if bare.startswith("//"):
                    pending = bare.lstrip("/ ").strip()
                    continue
                m = PTX_LINE.match(line)
                if not m:
                    continue
                parts = m.group(1).split(".")
                family = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
                slot = families.setdefault(
                    family, {"syntax": [], "notes": [], "section": head})
                if bare not in slot["syntax"]:
                    slot["syntax"].append(bare)
                    if pending:
                        slot["notes"].append(pending)
                pending = ""

    out = []
    for family, slot in sorted(families.items()):
        semantics = scrub_advice("; ".join(slot["notes"][:4])) or (
            "PTX forms of the %s instruction family, as documented under %s."
            % (family, slot["section"]))
        identity = dict(identity_base)
        identity["isa"] = "ptx"
        identity["mnemonic"] = family
        arch = identity["arch"]
        identity["sm_arch"] = ARCH_TO_SM.get(arch)
        identity["availability"] = {
            "sm_arch": [ARCH_TO_SM[arch]] if arch in ARCH_TO_SM else [],
            "products": [],
        }
        facts = {"syntax": slot["syntax"][:12], "semantics": semantics}
        extra = [scrub_advice(n) for n in slot["notes"][4:8]]
        extra = [n for n in extra if n]
        if extra:
            facts["caveats"] = extra
        out.append({
            "schema": SCHEMA_VERSION,
            "id": ".".join([identity["vendor"], arch, "instruction",
                            slugify(family)]),
            "type": "instruction",
            "status": "current",
            "identity": identity,
            "facts": facts,
            "provenance": provenance(relpath, slot["section"]),
        })
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def iter_pages() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"spec-sheet": [], "arch-feature": [],
                                    "instruction": []}
    for dirpath, _dirs, files in os.walk(DOCS_ROOT):
        for name in sorted(files):
            if not name.endswith(".md") or name == "README.md":
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), DOCS_ROOT)
            if "/hardware-specs/" in "/" + rel:
                groups["spec-sheet"].append(rel)
            elif "/kernel-opt/hardware/" in "/" + rel:
                groups["arch-feature"].append(rel)
            elif rel.endswith("languages/ptx-sm100.md"):
                groups["instruction"].append(rel)
    return groups


def build_all() -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    empty: list[str] = []
    groups = iter_pages()
    for rel in groups["spec-sheet"]:
        rec = build_spec_sheet(rel, read(rel))
        records.append(rec) if rec else empty.append(rel)
    for rel in groups["arch-feature"]:
        rec = build_arch_feature(rel, read(rel))
        records.append(rec) if rec else empty.append(rel)
    for rel in groups["instruction"]:
        got = build_instructions(rel, read(rel))
        records.extend(got) or (empty.append(rel) if not got else None)
    # ids must be unique; a page pair could collide on the same slug
    seen: dict[str, int] = {}
    for rec in records:
        rid = rec["id"]
        if rid in seen:
            seen[rid] += 1
            rec["id"] = "%s.%d" % (rid, seen[rid])
        else:
            seen[rid] = 0
    return records, empty


def record_relpath(rec: dict) -> str:
    ident = rec["identity"]
    slug = rec["id"].rsplit(".", 1)[-1]
    return os.path.join(rec["type"], ident["vendor"], ident["arch"],
                        slug + ".json")


def write_records(records: list[dict], clean: bool) -> None:
    if clean:
        for rtype in ("spec-sheet", "arch-feature", "instruction"):
            d = os.path.join(RECORDS_ROOT, rtype)
            if os.path.isdir(d):
                shutil.rmtree(d)
    for rec in records:
        path = os.path.join(RECORDS_ROOT, record_relpath(rec))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=1)
    # index.json is owned by build_hardware_index.py so its shape stays canonical.
    os.makedirs(RECORDS_ROOT, exist_ok=True)


def main() -> int:
    global DOCS_ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--docs-root", default=None)
    args = ap.parse_args()

    if args.docs_root:
        DOCS_ROOT = os.path.abspath(args.docs_root)
    if not os.path.isdir(DOCS_ROOT):
        print("no markdown docs root at %s -- one-shot seeding tool; pass "
              "--docs-root <checkout of the pre-migration docs tree>" % DOCS_ROOT,
              file=sys.stderr)
        return 3

    records, empty = build_all()
    if args.sample:
        print(json.dumps(records[:2], ensure_ascii=False, indent=1))
        print("\n%d records (nothing written)" % len(records), file=sys.stderr)
        return 0
    if not args.all:
        ap.error("pass --all to seed or --sample to preview")

    write_records(records, clean=args.clean)
    counts: dict[str, int] = {}
    for rec in records:
        counts[rec["type"]] = counts.get(rec["type"], 0) + 1
    print("seeded %d hardware records: %s" % (len(records), counts))
    for rel in empty:
        print("  no extractable facts: %s" % rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
