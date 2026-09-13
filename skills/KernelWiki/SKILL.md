---
name: KernelWiki
description: Retrieve GPU kernel optimization experience and hardware facts through the Supervisor Runtime.
allowed-tools: "Bash Read"
---

# GPU Wiki

Run from the Agent workspace. All queries use `python3 tools/sandbox.py`; the
Supervisor owns the Wiki stores, credentials, retrieval, and bounded results.

## Natural-language query

```bash
python3 tools/sandbox.py --kind wiki-query "Target B200, runtime sm_100. A BF16 Triton RMSNorm reaches 75% DRAM peak. Which fusion strategies apply? Include the product specification and relevant architecture facts." --brief
python3 tools/sandbox.py --kind wiki-query --file scratch/research-request.txt --max-records 6
```

State the actual product and runtime architecture, DSL, operator, relevant
shapes/dtypes, measured symptoms, previous attempts, and the decision you need
to make. Separate measurements from guesses; do not replace product identity
with a scheduler label. `--file` must stay inside this workspace. Later queries
can use `--exclude <ids-already-read>` and a smaller `--max-bytes` budget.

## Structured experience search

```bash
python3 tools/sandbox.py --kind wiki-search --arch sm_100 --dsl triton --coverage
python3 tools/sandbox.py --kind wiki-search rmsnorm --arch sm_100 --dsl triton --emit-json --limit 5
python3 tools/sandbox.py --kind wiki-search --list-symptoms
```

Keep architecture scope. Check `--coverage` before restricting record type:
dead ends may appear inside strategy records. A labelled fallback sample is
not an exact match or a recommendation for your case.

## Exact hardware facts

```bash
python3 tools/sandbox.py --kind wiki-hardware --product b200 --field peak_compute.bf16.dense
python3 tools/sandbox.py --kind wiki-hardware --list products
```

Unknown or unrecorded hardware remains unknown; never substitute a sibling
product's peak throughput or bandwidth. Use `--kind <operation> --help` for
the selected query's options. Store overrides and retaining private query
workspaces are not permitted.

## Read and use results

Natural-language results contain `query_id`, `records`, and `notes`. Each record
has a canonical `wiki_id`, store/source, applicability, and isolated payload;
inspect architecture scope and store-gap/truncation notes before using it.
Structured search and hardware lookup preserve their respective result formats.
Retrieval is not proof: verify advice against the current Kernel and Gateway
measurements. Explain materially used knowledge in the Experiment's `evidence`
and `analysis`; do not add unsupported Wiki metadata fields to Journal requests.
