# Schema — `hardware_wiki` (the hardware-fact store)

A store that is **a peer of, and independent from, `kernel_wiki` (the experience
store)**. It holds **facts** about hardware and ISAs, not optimization experience.

## Why it has to be independent

Not because "the content is different", but because **the retrieval problem is a
different kind of problem**:

| | Experience store `kernel_wiki` | This store `hardware_wiki` |
|---|---|---|
| Nature of the knowledge | Someone tried it and this is what happened — falsifiable, has a success rate | Vendor and ISA definitions — not falsifiable |
| Retrieval semantics | Fuzzy ranking: symptom → top-k candidate techniques | Exact lookup: given an address, one answer |
| Ranking | Needs `worth` (importance, tier, feedback) | **No ranking**, and no `worth` layer at all |
| Zero hits | Returns a labelled random fallback sample | **Must raise an error** — answering "BF16 peak" with a random sample is worse than not answering, because that number ends up as the denominator of a roofline |
| Feedback loop | Required; it drives the ranking | Meaningless; only errata and generational replacement apply |

So this store's `tools/query_hardware.py` is a **lookup**, not a ranked search: no
ranking, no fallback, and an unrecognized address fails with an error.

## The boundary: what belongs here, what stays in the experience store

**There is exactly one criterion: can our benchmark falsify this sentence?**

- Cannot be falsified → a fact → this store. Examples: the syntax and modifier
  vocabulary of `tcgen05.ld.red`; TMEM is 256 KB per SM; the B300 INT8 peak is
  187.5 TOPS.
- Can be falsified → experience → the experience store. Examples: "LDGSTS
  saturates around 32 KiB in flight", "CLC is not always faster on balanced
  GEMMs", "that technique has a 26% retention rate" — these are measurements, and
  they vary with device and workload.

When a source page mixes the two kinds, it **must be split**. For instance the
§12 microbenchmarks on the B200 tensor-core analysis page, and the Performance
Impact subsection of the CLC page, both say explicitly that "these are B200
observations, not architectural guarantees" — those stay in the experience store.

The **no-advice** gate in `check_hardware_wiki.py` catches recommendation language
that slips in ("usually faster", "we recommend", "retention rate N%").

## The three record types

| type | One record = | Address |
|---|---|---|
| `spec-sheet` | Every number for one chip | `--product b300` |
| `instruction` | One ISA instruction family | `--instruction tcgen05.ld.red` |
| `arch-feature` | One architectural capability | `--feature fp4-k96-2cta` |

A single envelope: `identity` (the address) + `facts` (the content, for the agent)
+ `provenance` (the evidence) + `status`. **There is no worth.**

## Evidence grading is mandatory

Every number has to be able to answer "who says so". `provenance.evidence_class`
has three levels:

- `vendor-published` — a value the vendor printed for this exact chip
- `derived-from-system-total` — computed from an n-card system-level number (**the
  divisor must be stated**, which the provenance gate enforces)
- `architecture-analysis` — third-party or inferred, **treated as provisional**;
  prefer runtime device attributes

One vendor page often mixes several levels, so `memory` and `compute_units`
support `provenance_overrides` for **field-level exceptions**. B300 is the case in
point: capacity and bandwidth are vendor-published, while the 126 MB L2 is
architecture analysis.

**Any unpublished field stays `null`, with `facts.unavailable` stating how to
obtain it.** Inventing a peak is far more dangerous than missing one — it silently
pollutes every utilization figure derived from it. The fabrication gate enforces
this: a `null` must come with an explanation, and a field that has an explanation
may not also carry a value.

## Usage

```bash
# Fetch a roofline denominator, with its evidence class
python3 tools/query_hardware.py --product b300 --field peak_compute.bf16.dense
# → {"value": 2250, "provenance": "architecture-analysis", ...}

# An unpublished field: you get a disposition, not a substitute value
python3 tools/query_hardware.py --product b300 --field compute_units.shared_memory_kb_per_sm
# → {"value": null, "unavailable": "...Query the CUDA device attribute at runtime..."}

# Cross-generation comparison: guards against assuming the newer part is stronger
python3 tools/query_hardware.py --product b300 --vs b200
# → int8 −95.8%, fp64 −96.6%, fp4 +50%

# Instructions and features
python3 tools/query_hardware.py --instruction tcgen05.ld.red
python3 tools/query_hardware.py --feature fp4-k96-2cta
python3 tools/query_hardware.py --capability sm_103 --list features
```

The convention matches the experience store: **stdout carries exactly one JSON
document** and diagnostics go to stderr, so you can `json.load` it directly.

## Maintenance

```bash
python3 tools/build_hardware_index.py         # rebuild records/index.json
python3 tools/build_hardware_index.py --check # check the index agrees with the records
python3 tools/check_hardware_wiki.py          # six gates
python3 -m unittest discover -s tools         # query contract tests
```

Each of the six gates prevents one class of **silent** corruption: `schema`
(record drift), `ids` (id does not match path, so references do not resolve),
`index` (a record is unreachable), `provenance` (a number with no source),
`no-advice` (experience mixed into facts), and `fabrication` (an unpublished
number invented).

## Current state

30 records, all projected from this repository's curated documentation:

| type | Count |
|---|---:|
| `spec-sheet` | 9 |
| `arch-feature` | 10 |
| `instruction` | 11 |

Products covered: b200, b300, mi300x, mi308x, mi355x, sm120.

### Product name mapping

`tools/hardware_identity.py` is the fixed identity table shared by every query
entry point. The "internal address" in that table is exactly the `product` value
in `hardware_wiki/records/index.json`; case, vendor prefixes, spaces, underscores
and `GPU` / `accelerator` suffixes only affect how the input is written and never
produce a new internal address.

| Internal address | vendor | arch | Examples of permitted purely-formatting variation |
|---|---|---|---|
| `b200` | nvidia | blackwell | `B200`, `NVIDIA B-200 GPU` |
| `b300` | nvidia | blackwell-ultra | `B300`, `nvidia-b300` |
| `mi300x` | amd | cdna3 | `MI300X`, `AMD Instinct MI-300X GPU` |
| `mi308x` | amd | cdna3 | `MI308X`, `amd_mi308x` |
| `mi355x` | amd | cdna4 | `MI355X`, `AMD MI-355X accelerator` |
| `sm120` | nvidia | blackwell-geforce | `SM120`, `sm_120`, `SM-120` |

For example `NVIDIA B300 GPU` → `b300` and `AMD Instinct MI308X GPU` → `mi308x`.
A100/A800/A30/A10, L20/L40S/L4, H100/H200/H800/H20/GH200, B100/GB200/GB300,
RTX PRO 5000/RTX 5090/5080, and MI300A/MI350X are in the same identity table; they
currently have no product spec sheet, so they return a `not-recorded` disposition
and never borrow another product's numbers.

An architecture name is never force-mapped onto a particular product: `gfx942`,
for instance, covers several CDNA3 SKUs, so it cannot be used to pick `mi300x`
over `mi308x`, and `GB200` likewise stays a distinct product rather than borrowing
the B200 spec sheet. The query side only eliminates case, spaces, `-`, `_` and
unambiguous vendor wrapper words; it never translates one identity into another.
Any non-product identifier yields `unknown-product` rather than being mapped onto
some spec sheet. Run `python3 tools/query_hardware.py --list products` to see the
internal addresses, the formatting rules, and the products that are recognized but
have no spec sheet yet.

**A note on evidence grading**: these pages are third-party curated documentation,
not vendor datasheets, so `provenance.evidence_class` is `architecture-analysis`
for every record here — which by the schema's definition makes them
**provisional**, to be checked against runtime device attributes first. This store
does not vouch on a vendor's behalf: no number here is ever marked
`vendor-published`.
