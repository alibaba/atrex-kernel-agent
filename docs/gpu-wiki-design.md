# GPU Wiki Knowledge Base Design Notes

## 1. Project Positioning

GPU Wiki is a GPU kernel programming and optimization knowledge base designed for AI Agents. It is not a conventional documentation site for human readers; it is a structured technical memory that helps Agents make accurate decisions during GPU kernel development, optimization, profiling, and cross-platform migration.

Core usage:

1. The Agent enters through `README.md`, `CLAUDE.md`, and directory-level `README.md` files.
2. The Agent selects the target **vendor** (nvidia / amd / generic), then the **topic/DSL**, then the **architecture** (sm90 / sm100 / sm120 / gfx942 / gfx950).
3. Within a topic the Agent reads reference articles and optimization cards directly, consults `pitfalls/` for negative knowledge, and follows links to runnable examples in `reference-kernels/` or upstream source under `reference-projects/` when deeper evidence is needed.

The repository is optimized for Agent retrieval: all of a vendor's knowledge lives under one subtree, directory names encode vendor/topic/architecture, and a relationship document calls out conflicts between architectures.

## 2. Current Repository Layout

```text
gpu-wiki/
├── AGENTS.md                  # Agent instruction: read README.md first
├── CLAUDE.md                  # Agent entry point and quick navigation
├── README.md                  # Project overview, usage policy, vendor index, source list
├── docs/                      # Curated markdown knowledge base (vendor-first)
│   ├── README.md              # docs/ vendor index
│   ├── RELATIONS.md           # Reading paths, cross-architecture relations and conflicts
│   ├── generic/               # Vendor-agnostic theory + hands-on/ + converter/
│   ├── nvidia/                # common, ptx, profiling, cutedsl, gluon, cuda, triton, converter, hardware-specs
│   └── amd/                   # common, flydsl, gluon, converter, hardware-specs
├── reference-kernels/         # Runnable/reference kernel implementations
├── reference-projects/        # Upstream framework source (git submodules)
└── 3rdparty/                  # Community GPU kernel wikis (git submodules)
```

Current scale (working tree at the time of this rewrite):

| Category | Count |
|----------|------:|
| `docs/` markdown files | 402 |
| `docs/nvidia/` markdown files | 246 |
| `docs/amd/` markdown files | 131 |
| `docs/generic/` markdown files | 23 |
| `reference-kernels/` Python files | 488 |

## 3. Organization Principles

### 3.1 Vendor is the First-Class Dimension

GPU kernel optimization is highly architecture-dependent: the same technique can help on one GPU and hurt on another. Keeping every vendor's knowledge in a single subtree (`docs/nvidia/`, `docs/amd/`, `docs/generic/`) prevents the Agent from drifting across vendors and misapplying experience. Within `docs/nvidia/` the second axis is **architecture** (`hopper/` sm90, `blackwell/` sm100, `blackwell-geforce/` sm120) plus a `common/` for cross-arch material; `docs/amd/` and `docs/generic/` stay topic-first.

### 3.2 Second Dimension: Architecture (NVIDIA) / Topic (AMD, generic)

For **NVIDIA**, the second level is GPU architecture — `hopper/` (sm90), `blackwell/` (sm100, B200/B300), `blackwell-geforce/` (sm120, RTX PRO) — because optimization is highly arch-dependent. Cross-arch material (PTX ISA, NCU profiling, CuTeDSL/CUTLASS fundamentals, general theory, hardware-specs) lives in `nvidia/common/`. For **AMD** and **generic**, the second level stays topic/DSL (`common`, `flydsl`, `gluon`, `converter`, …) with architecture as `gfx942/`/`gfx950/` subfolders. Each leaf folder consolidates reference articles, cards, and hands-on together.

### 3.3 Architecture and Pitfalls are Sub-Layers

- Architecture-specific material lives in `sm90/` (Hopper), `sm100/` (Blackwell datacenter), `sm120/` (Blackwell GeForce / RTX PRO), `gfx942/` (CDNA3), `gfx950/` (CDNA4) subfolders, often with a `hands-on/` folder of code-derived patterns. NVIDIA Blackwell also has a richer `common/blackwell/` knowledge-card tree.
- Negative knowledge (counter-intuitive traps, reverted approaches) lives in a `pitfalls/` subfolder inside the relevant topic, following the format: trap → symptom → real cause → why → lesson.

### 3.4 One File, One Focus

Most topic files cover a single technique, pitfall group, conversion rule set, hardware table, or framework topic. Broad overviews live in directory-level `README.md` files; details live in focused child documents.

## 4. Knowledge Domains (per vendor)

### 4.1 `docs/nvidia/` (architecture-first)

| Dir | Role |
|-----|------|
| `common/` | Cross-arch: `ptx/` (ISA), `profiling/` (NCU/Nsight), `cutedsl/` (CUTLASS/CuTe fundamentals), `gluon/` `triton/` general notes, `hardware-specs/`, and general optimization theory cards |
| `hopper/` (sm90) | `cutedsl/`, `gluon/`, `converter/` (Triton→Gluon), `hands-on/` (TMA/WGMMA/mbarrier/warp-spec) |
| `blackwell/` (sm100) | structured tree (`hardware/ techniques/ kernels/ patterns/ languages/ migration/`), `hands-on/` (CuTeDSL examples), `articles/` (FA4/B300 deep-dives), `cutedsl/`, `gluon/`, `triton/pitfalls/` |
| `blackwell-geforce/` (sm120) | `cutedsl/`, `cuda/`, `triton/` — each with `pitfalls/`: NVFP4 GEMM/GEMV, GDN chunk-fwd/decode, MoE data-prep, RMSNorm-MLP PDL |


### 4.2 `docs/amd/`

| Topic | Role |
|-------|------|
| `hardware-specs/` | MI300X / MI308X (CDNA3) and MI355X (CDNA4) spec tables + CDNA3↔CDNA4 comparison |
| `common/` | General optimization (MFMA, CK, rocprofv3, roofline, LDS, occupancy) + `hands-on/` + `gfx942/` + `gfx950/` |
| `flydsl/` | FlyDSL framework + MI300X/MI355X reports + `gfx942/` `gfx950/` + `pitfalls/` |
| `gluon/` | MI300X/MI355X Gluon (`gfx942/`, `gfx950/`) |
| `converter/` | Triton→Gluon: general rules (`common/`) + CDNA3 (`cdna3/`) + CDNA4 (`cdna4/`) |

### 4.3 `docs/generic/`

Vendor-agnostic GPU theory (memory hierarchy, execution model, instruction/application optimization, GEMM, operator cookbook, LLM inference, surveys), Triton optimization patterns under `hands-on/`, and PyTorch→Triton conversion under `converter/`.

### 4.4 `docs/RELATIONS.md`

Repository-level relationship graph: hierarchical reading paths, cross-architecture comparison tables, conflict/difference inventories, and document groups that should be read together. Consult it for tasks spanning multiple architectures or vendor migration.

## 5. Reference Code and Ground Truth

`reference-kernels/` stores Python/CUDA kernel implementations organized by hardware architecture → DSL/framework → source project. Upstream API/ISA ground truth lives in `reference-projects/` (git submodules of CUTLASS, flash-attention, FlashInfer, FlyDSL, Triton, etc.), and community wikis in `3rdparty/`. `docs/` answers "how is this used and why"; upstream projects answer "what exactly does this API/instruction mean".

## 6. Maintenance Rules

1. Keep each vendor's knowledge under its own subtree; do not recreate top-level category directories (`kernel-opt/`, `ref-docs/`, `pitfalls/`, `converter/`).
2. Within a topic, keep reference articles, cards, and `pitfalls/` together.
3. Keep directory `README.md` files current — Agents depend on them for navigation.
4. Record cross-architecture conflicts in `docs/RELATIONS.md` or a relevant pitfall file.
5. Prefer evidence-backed notes (profiling data, hardware constraints, source references).
6. Update counts and structure in this document after large imports or reorganizations.
</content>
