# GPU Wiki Design

## 1. Purpose

GPU Wiki provides architecture-safe knowledge retrieval for kernel research,
implementation, profiling, and conversion. Hardware behavior is the primary
isolation boundary because a valid optimization on one GPU may be ineffective
or incorrect on another.

The physical documentation hierarchy is therefore **architecture first**:

```text
scope → knowledge role → DSL/topic → document
```

The knowledge role remains important, but it is secondary to hardware scope.

## 2. Canonical layout

```text
gpu-wiki/
├── docs/
│   ├── generic/
│   │   ├── converter/
│   │   ├── kernel-opt/
│   │   └── ref-docs/
│   ├── nvidia/
│   │   ├── common/
│   │   ├── ampere/
│   │   ├── hopper/
│   │   ├── blackwell/
│   │   │   └── b200/
│   │   ├── blackwell-ultra/
│   │   └── blackwell-geforce/
│   └── amd/
│       ├── common/
│       ├── cdna3/
│       │   ├── mi300x/
│       │   └── mi308x/
│       ├── cdna4/
│       └── rdna4/
├── reference-kernels/
├── manifest.json
├── reference-projects/
├── scripts/
└── 3rdparty/
```

Architecture roots and aliases:

| Physical scope | Architecture and product aliases |
|---|---|
| `nvidia/ampere/` | SM80, A100 |
| `nvidia/hopper/` | SM90, H20, H100, H200 |
| `nvidia/blackwell/` | SM100 general knowledge |
| `nvidia/blackwell/b200/` | B200, GB200 product-only evidence |
| `nvidia/blackwell-ultra/` | SM103, B300, GB300 |
| `nvidia/blackwell-geforce/` | SM120, RTX PRO 5000, Pro5000 |
| `amd/cdna3/` | gfx942 general knowledge |
| `amd/cdna3/mi300x/` | MI300X product-only evidence |
| `amd/cdna3/mi308x/` | MI308X product-only evidence |
| `amd/cdna4/` | gfx950, MI355X |
| `amd/rdna4/` | gfx1250 |

`common/` means vendor-general or intentionally cross-architecture. It must not
be used as a dumping ground for documents whose target architecture is known.

## 3. Knowledge roles

Every content document sits below one of these role directories:

| Role | Purpose |
|---|---|
| `hardware-specs/` | Verified hardware facts, peak throughput, bandwidth, memory and execution resources |
| `kernel-opt/` | Short decision cards, reusable optimization patterns, and hands-on notes |
| `ref-docs/` | Full reports, framework/API references, profiling studies, and implementation journeys |
| `pitfalls/` | Failed approaches, misleading signals, architecture traps, and negative evidence |
| `converter/` | Cross-DSL and cross-architecture conversion guidance |

Current content inventory, excluding README indexes and `RELATIONS.md`:

| Role | Documents |
|---|---:|
| Hardware specs | 9 |
| Kernel optimization | 124 |
| Reference documents | 185 |
| Pitfalls | 21 |
| Converter | 5 |
| **Searchable total** | **344** |

The reference index adds 499 source files plus 23 manifest-selected substantive
guides. Navigation-only README files are not indexed unless explicitly selected.

## 4. Product overlays and inheritance

An architecture directory holds reusable family knowledge. A product overlay is
used only when a fact, benchmark, or pitfall is tied to one product.

Examples:

```text
nvidia/blackwell/kernel-opt/hardware/tma.md
nvidia/blackwell/b200/hardware-specs/hardware_specs_b200.md
amd/cdna3/ref-docs/flydsl/...
amd/cdna3/mi308x/pitfalls/flydsl/...
```

Retrieval inheritance is one-way:

- B200 includes Blackwell/SM100 general pages plus B200 pages.
- B300 includes applicable Blackwell general pages, but never B200-only pages.
- MI300X and MI308X each include CDNA3 general pages, but never the sibling
  product's pages.
- A family query such as `gfx942` may intentionally include both CDNA3 product
  overlays for cross-product research.

SM120 is a distinct architecture scope. RTX PRO 5000 pages must not inherit
SM100/B200 or SM103/B300 knowledge merely because all products use the
“Blackwell” brand.

## 5. Scoped query behavior

`gpu-wiki/scripts/query.py` loads explicit scope and classification metadata
from the top-level `manifest.json`, with architecture-first path inference as a
fallback for undeclared files. It applies hard scope filters before ranking
keyword matches.

```bash
python3 gpu-wiki/scripts/query.py "gemm" --arch h20
python3 gpu-wiki/scripts/query.py "gdn" --arch pro5000 --dsl cutedsl
python3 gpu-wiki/scripts/query.py --arch b200 \
  --area docs --section kernel-opt --symptom pipeline-stalls
python3 gpu-wiki/scripts/query.py gemm --arch b300 \
  --area reference-kernels --source cutlass --kind kernel
python3 gpu-wiki/scripts/query.py rms_nrom --arch h20 --fuzzy
```

Only `--arch` is needed to establish hardware isolation, and it implies its
vendor when `--vendor` is omitted. A keyword search covers docs, indexed
reference sources, and manifest-selected substantive guides by default.
`--area`, `--section`, `--symptom`, `--kernel-type`, `--operator`, `--dsl`,
`--source`, `--status`, and `--kind` are optional narrowing filters. Unknown
filter values fail closed; test/build/package references are excluded unless
`--include-auxiliary` is explicit.

Normal query terms normalize filename/path separators. `--fuzzy` adds
`SequenceMatcher` and trigram similarity for uncertain spelling after hard
scope isolation. `manifest.json` is authoritative for explicit entries and
prefix defaults; path inference remains the fallback rather than a competing
metadata source.

## 6. Placement rules

When adding or moving a document:

1. Determine whether it is vendor-independent, vendor-general,
   architecture-general, or product-only.
2. Choose the narrowest correct hardware scope.
3. Choose the role and then the DSL/topic subdirectory.
4. Add evidence links for hardware claims; prefer official vendor sources for
   specifications and API/ISA truth.
5. Update the nearest README index and any links from reference kernels.
6. Add or update a query isolation test when introducing a product scope.

Do not infer applicability from a brand name in prose. Physical scope and the
top-level manifest are authoritative.

## 7. Navigation and evidence

The recommended reading order is:

1. Hardware specifications for the exact target.
2. Vendor `common/` knowledge.
3. Architecture-general `kernel-opt/` cards.
4. Product overlays, if any.
5. Detailed `ref-docs/` and negative evidence in `pitfalls/`.
6. `reference-kernels/` and upstream source for implementation details.

`docs/RELATIONS.md` records cross-architecture relationships and conflicts.
`reference-kernels/` remains architecture-first and provides runnable or
near-runnable examples; it is not automatically safe to import unchanged.

## 8. Validation gates

A structural change is complete only when all of the following hold:

- no role-first root directories remain below `gpu-wiki/docs/`;
- every searchable document has a valid physical scope and role;
- all relative Markdown links stay inside the self-contained wiki and resolve;
- `manifest.json` is valid and covers every indexed reference source/guide;
- query unit tests cover aliases, inheritance, and sibling-product exclusion;
- representative A100, H20, B200, B300, Pro5000/SM120, MI300X, MI308X, and
  MI355X searches return the intended scope;
- `git diff --check` is clean.
