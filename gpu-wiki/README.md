# GPU Wiki

GPU Wiki is the structured knowledge system used by ATREX Kernel Agent (AKA).
It turns optimization traces and agent sessions into validated JSON knowledge,
serves that knowledge through a lightweight query path, and mines completed AKA
optimization traces for the next generation of reusable knowledge.

![GPU Wiki knowledge flywheel: skills distill traces into structured knowledge, AKA queries it for optimization, and complete traces feed new knowledge back into the Wiki.](gpu-wiki-knowledge-flywheel.png)

## The knowledge flywheel

The diagram is the operating model of this directory:

```text
optimization traces + agent sessions
                ↓
deterministic extraction → semantic distillation → Wiki Gate
                ↓
       structured GPU Wiki records
                ↓
natural-language query → deterministic retrieval → AKA optimization
                ↓
          complete optimization trace
                └──────────────→ mined again as new candidate knowledge
```

The loop is intentionally asymmetric: retrieval is lightweight and read-only;
knowledge admission is slower and guarded. A completed optimization does not
write directly into the Wiki. It becomes input to the mining skills, and only a
candidate that passes the Wiki Gate can become a record.

## 1. Inputs become candidate knowledge

The flywheel accepts two primary inputs:

- **Optimization traces**: the version history of a kernel, including attempts,
  failures, measurements, and the final result.
- **Agent sessions**: the reasoning and tool-use history that explains why an
  optimization path was selected, rejected, or revised.

The Skills Workshop in the diagram represents three distinct responsibilities:

1. **Extract** — deterministic scripts reconstruct versions, measurements,
   events, and source relationships without interpreting them.
2. **Distill** — an agent turns the extracted packet into a self-contained
   candidate: a technique, failure mode, symptom, or reference.
3. **Validate** — `wiki-gate` checks schema, scope, provenance, relations, and
   conflicts before admitting anything.

| Skill | Input | Output |
|---|---|---|
| [`opt-trace-mining`](skills/opt-trace-mining/) | Git history of successive kernel versions | Candidate optimization experience |
| [`session-trace-mining`](skills/session-trace-mining/) | AI coding-agent session transcripts | Candidate agent-discovered knowledge |
| [`wiki-gate`](skills/wiki-gate/) | One candidate record | Insert, confirm, conflict, or reject decision |

`wiki-gate` is the only writer. Scripts establish what happened, the distillation
agent explains what can be reused, and the gate prevents unsupported conclusions
from entering the store.

## 2. Validated knowledge enters two isolated stores

GPU Wiki separates benchmark-falsifiable experience from vendor-defined facts.
The boundary question is: **can a measurement prove this statement wrong?**

| | `kernel_wiki` — experience | `hardware_wiki` — facts |
|---|---|---|
| Contains | Techniques, failures, symptoms, reference knowledge | Products, capacities, architecture features, ISA |
| Record types | `technique-card`, `anti-strategy`, `symptom-card`, `doc` | `spec-sheet`, `arch-feature`, `instruction` |
| Retrieval | Ranked and scope-aware | Exact and fail-loud |
| Ranking | Engine-side `worth` | None |
| Zero matches | Labelled fallback sample within scope | Explicit error or missing-data procedure |

The stores are isolated because their failure modes differ. Returning an
unrelated optimization example can still be useful when clearly labelled;
returning an unrelated peak-FLOPS number would silently corrupt every utilization
calculation derived from it.

```text
gpu-wiki/
├── kernel_wiki/
│   └── records/       <type>/<vendor>/<arch>/<dsl>/<family>/<id>.json + index.json
├── hardware_wiki/
│   └── records/       <type>/<vendor>/<arch>/<slug>.json + index.json
├── schema/
│   ├── kernel/        clean-1.3 contract and generated template
│   └── hardware/      hw-1.0 contract
├── skills/            trace mining, session mining, and admission gate
├── tools/             query, validation, indexing, migration, and ranking tools
└── query_intent.json  typed bridge-agent output contract
```

The distributed knowledge base is JSON-only. Seed records retain historical
source paths for provenance, but the source Markdown corpus is not distributed
in this repository. The builders remain available to replay that migration
against a compatible external checkout.

## 3. AKA describes the optimization problem

The default entry point is natural language. AKA should describe the real
optimization context instead of guessing Wiki flags:

- true public GPU product, such as `B200`, `B300`, `H20`, or `MI308X`;
- runtime architecture exactly as reported, such as `sm_100` or `gfx942`;
- operator, framework, shapes, and dtypes;
- measured symptoms with numbers, separated from unverified assumptions;
- previous attempts, failures, and relevant error text;
- the decision or stopping condition the retrieved knowledge should inform.

```bash
python3 gpu-wiki/tools/query_nl.py \
  "Target B200, runtime sm_100. A BF16 Triton fused RMSNorm is memory-bound; \
   profiling shows 75% of DRAM peak. Which fusion and launch strategies apply, \
   and what hardware facts should constrain the decision?" \
  --brief --max-records 6
```

Do not substitute a scheduler label for the product name. Product normalization
only repairs formatting differences such as case, separators, or vendor wrappers;
it does not translate one hardware identity into another.

## 4. The bridge stays small; scripts perform retrieval

`query_bridge_agent` has one job: convert prose into typed semantic intent. It
cannot query either Wiki store, read records, rank results, or write files.

After the bridge returns plain JSON, deterministic code owns the rest:

1. strictly validate the intent;
2. normalize product, architecture, DSL, operator, and symptom vocabulary;
3. construct isolated kernel and hardware lookups;
4. widen safely without dropping the architecture boundary;
5. execute the store-specific query tools;
6. deduplicate, project, budget, and return the records.

The Claude success path uses one tool-free bridge call. Invalid bridge JSON gets
at most one fresh repair attempt. `claude` is the default agent CLI; `codex` and
`qodercli` are also supported.

The consuming agent receives only two top-level fields:

```json
{
  "records": {
    "stable.record.id": {
      "store": "gpu_wiki",
      "source": "kernel_wiki",
      "type": "technique-card",
      "applies_to": {},
      "match": {},
      "payload": {}
    }
  },
  "notes": []
}
```

Record IDs are stable mapping keys, and every payload remains an independent JSON
value. Evidence, retrieval metadata, rank decomposition, bridge commentary, and
other engine-side fields are deliberately not served, so they cannot anchor AKA's
judgement.

### Direct structured queries

Use the lower-level tools when the address is already known or a script needs a
deterministic interface.

```bash
# Ranked kernel experience. Architecture is a hard isolation boundary.
python3 gpu-wiki/tools/query_wiki.py "rmsnorm" \
  --arch sm_100 --dsl triton --emit-json --brief --limit 5
python3 gpu-wiki/tools/query_wiki.py \
  --symptom register-pressure --arch blackwell --brief --limit 2
python3 gpu-wiki/tools/query_wiki.py --arch sm_100 --dsl triton --coverage

# Exact hardware facts. Unknown addresses fail loudly and never return samples.
python3 gpu-wiki/tools/query_hardware.py \
  --product b200 --field peak_compute.bf16.dense
python3 gpu-wiki/tools/query_hardware.py --product b300 --vs b200
python3 gpu-wiki/tools/query_hardware.py --instruction tcgen05.mma
python3 gpu-wiki/tools/query_hardware.py --feature tma
```

For `kernel_wiki`, a zero-text-match fallback is explicitly labelled and stays
inside the resolved scope. Do not remove `--arch` merely to obtain results. For
`hardware_wiki`, every number carries an evidence class; provisional
`architecture-analysis` values should be verified with runtime attributes or the
vendor tool before launch geometry is hard-coded.

## 5. AKA optimizes with traceable Wiki references

AKA reads the isolated payloads and decides whether any record materially informs
the next optimization step. Retrieval alone is not adoption.

When a record changes an optimization decision, AKA preserves its stable record ID
with that iteration's memory. This creates a traceable link between:

- the Wiki knowledge that was actually used;
- the code version produced from it;
- correctness and performance measurements;
- the decision to keep, revise, or reject the attempt.

Rejected records are not recorded as adopted knowledge, and payload text is not
copied into the provenance field. The stable ID is sufficient to reconnect the
optimization trace to the exact structured record.

## 6. The complete trace closes the loop

After optimization finishes, the complete trace contains more than the winning
kernel. It preserves:

- every significant version and code change;
- successful, neutral, and regressed attempts;
- correctness gates and performance measurements;
- Wiki record IDs that informed specific decisions;
- new insights the agent derived while reconciling knowledge with measurements.

That complete trace returns to `opt-trace-mining` and
`session-trace-mining`. Deterministic scripts reconstruct the evidence, an agent
distills reusable lessons, and `wiki-gate` decides whether each candidate is safe
to admit. In this way AKA can bootstrap new knowledge from real optimization work
without allowing the optimization agent to write unverified conclusions directly
into the Wiki.

The next AKA run can then retrieve the validated lesson, completing the flywheel:

```text
GPU Wiki → query → optimize → complete trace → mine → validate → GPU Wiki
```

## Validation and maintenance

Run all integrity gates after changing records, indexes, schemas, or retrieval
tools:

```bash
# Record integrity.
python3 gpu-wiki/tools/check_kernel_wiki.py --full
python3 gpu-wiki/tools/check_hardware_wiki.py

# Generated artifacts and indexes.
python3 gpu-wiki/tools/build_hardware_index.py --check
python3 gpu-wiki/schema/kernel/render_template.py --check

# Retrieval and bridge contracts.
python3 -m unittest discover -s gpu-wiki/tools
```

The gates cover schema conformance, deterministic IDs, index consistency,
anonymization, payload isolation, relation resolution, self-containment,
provenance, fact/advice separation, and fabrication checks.

### Migration replay

The seed builders are retained for provenance and reproducibility. They require a
compatible external Markdown checkout:

```bash
python3 gpu-wiki/tools/build_kernel_records.py \
  --docs-root /path/to/docs --sample
python3 gpu-wiki/tools/build_hardware_records.py \
  --docs-root /path/to/docs --sample
```

Full `--all --clean` rebuilds overwrite generated record trees. Run them only in a
disposable or clean checkout and review the complete diff.

## Tool map

| Flywheel stage | Main tools |
|---|---|
| Extract and distill | `skills/opt-trace-mining/`, `skills/session-trace-mining/` |
| Validate and admit | `skills/wiki-gate/` |
| Natural-language query | `tools/query_nl.py`, `skills/query_bridge_agent/` |
| Kernel retrieval | `tools/query_wiki.py` |
| Hardware lookup | `tools/query_hardware.py` |
| Store validation | `tools/check_kernel_wiki.py`, `tools/check_hardware_wiki.py` |
| Index maintenance | `tools/reindex_kernel_wiki.py`, `tools/build_hardware_index.py` |
| Ranking maintenance | `tools/wiki_score.py`, `tools/rebuild_importance.py` |
| Historical migration | `tools/build_kernel_records.py`, `tools/build_hardware_records.py` |

For the full record contracts, see
[`schema/kernel/README.md`](schema/kernel/README.md) and
[`schema/hardware/README.md`](schema/hardware/README.md).
