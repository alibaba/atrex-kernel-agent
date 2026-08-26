# GPU Wiki Agent Entry

Read `README.md` first. This wiki is **two independent JSON record stores**, and
which one to ask depends on whether a benchmark could prove the answer wrong.

**Default door — describe your situation, do not compose a query.**
`python3 gpu-wiki/tools/query_nl.py "<prose>"` parses AKA's standard request
format deterministically; other prose gets one store-blind intent extraction by
`query_bridge_agent`. Deterministic code then resolves operator aliases and
components, queries isolated lanes, safely widens, and returns payloads keyed by
stable record id. The bridge cannot invoke either query tool or carry a record.

Every call independently attempts both `gpu_wiki` and `internal_gpu_wiki`. A missing
directory, the tracked `SOURCE.txt` placeholder alone, an incomplete interface, or a
zero-result lookup makes only that module empty. Internal results use
`internal_gpu_wiki::<stable-id>` and each record declares its `store`, preserving
strict store and payload isolation.

```bash
python3 gpu-wiki/tools/query_nl.py "fused RMSNorm in triton on sm_100. ncu says 75% of
    DRAM peak so it is bandwidth bound. Is fusing the passes a known dead end here, and
    how does triton express the row reduction on this part?" --brief
python3 gpu-wiki/tools/query_nl.py --file request.txt --max-bytes 20000
```

Say which parts of your description are measured and which are still guesses — that
is what decides whether the bridge spends the symptom axis. Give the architecture the
runtime reported. Also state the separate true public product and explicitly request its
full hardware spec plus relevant architecture/ISA facts, so the same call can return both stores.
Do not pre-compress into keywords.

Read each record's `source`, `type`, `match.arch`, and isolated `payload`, plus
deterministic `notes`. Evidence and bridge commentary are not served.

When a returned record materially informs an optimization decision, preserve its
stable mapping key in that iteration's `memory/vN.json` under
`wiki_references.kernel_wiki` or `wiki_references.hardware_wiki`. Retrieval alone
does not count as adoption: do not record ids for payloads that were only read and
rejected, and never copy payload text into this provenance field.

**Experience, addressed directly** (`kernel_wiki/records/`) — ranked, scoped search,
for when you already know the exact address. Query it with
`python3 gpu-wiki/tools/query_wiki.py` using explicit `--arch` / `--vendor` /
`--dsl` filters before broad grep. `--arch` takes whatever the runtime
reported (`sm_90`, `sm_100`, `gfx942`, `h20`, `b300`, `mi300x`) or the family
name (`hopper`, `blackwell`, `cdna3`, ...).

**Check `--coverage` before filtering on `--type`.** The type axis looks like the one
that expresses intent, and it is the one that most often returns zero on a subject
the store covers well: in one store `strategy` is 81% of the records for an
architecture while the types the docs name are about 4%, and callers who led with
`--type anti-strategy` concluded the store was empty on operators it documents
thoroughly. A "known dead end" is frequently prose inside a `strategy` record.

```bash
python3 gpu-wiki/tools/query_wiki.py --arch sm_100 --dsl triton --coverage
python3 gpu-wiki/tools/query_wiki.py --symptom register-pressure --arch blackwell --brief --limit 2
python3 gpu-wiki/tools/query_wiki.py --list-arch          # also --list-dsl --list-type --list-symptoms
python3 gpu-wiki/tools/query_wiki.py --list-family --like gemm   # filter a long vocabulary
```

Scope is a hard boundary and unknown filter values fail closed. A zero-match
query returns a **labelled random sample** of the scoped pool — never read it as
advice about what you asked, and never drop `--arch` to make an empty result
look successful.

**Facts** (`hardware_wiki/records/`) — exact lookup, fail-loud. Peaks,
capacities, ISA and feature definitions:

```bash
python3 gpu-wiki/tools/query_hardware.py --product b200 --field peak_compute.bf16.dense
python3 gpu-wiki/tools/query_hardware.py --list products   # b200 b300 mi300x mi308x mi355x sm120
```

A recognized but unrecorded part (`h20`, `h100`, `a100`) exits **4** with a
disposition: obtain the number from runtime device attributes or the vendor
datasheet for that exact part. Never substitute another part's numbers — a wrong
peak silently rescales every utilization figure derived from it.

After changing records, the index, or the tools, run:

```bash
python3 gpu-wiki/tools/check_kernel_wiki.py --full     # 9 gates
python3 gpu-wiki/tools/check_hardware_wiki.py          # 6 gates
python3 -m unittest discover -s gpu-wiki/tools
```
