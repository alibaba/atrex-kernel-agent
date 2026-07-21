# GPU Wiki Documentation

The knowledge base is organized **architecture first**. Choose the vendor and
architecture before choosing a knowledge role or DSL. This prevents advice for
one GPU family or product from silently entering another GPU's search scope.

```text
docs/
├── generic/                         # vendor-independent knowledge
├── nvidia/
│   ├── common/                      # vendor-general / explicit cross-arch
│   ├── ampere/                      # SM80 / A100
│   ├── hopper/                      # SM90 / H20, H100, H200
│   ├── blackwell/                   # SM100 general
│   │   └── b200/                    # B200/GB200 product overlay
│   ├── blackwell-ultra/             # SM103 / B300, GB300
│   └── blackwell-geforce/           # SM120 / RTX PRO 5000
└── amd/
    ├── common/                      # vendor-general / explicit cross-arch
    ├── cdna3/                       # gfx942 general
    │   ├── mi300x/                  # MI300X product overlay
    │   └── mi308x/                  # MI308X product overlay
    ├── cdna4/                       # gfx950 / MI355X
    └── rdna4/                       # gfx1250
```

Within each scope, the second dimension is the knowledge role:

- `hardware-specs/`: verified hardware facts and roofline inputs.
- `kernel-opt/`: concise optimization cards and hands-on patterns.
- `ref-docs/`: full reports, API notes, and implementation journeys.
- `pitfalls/`: negative evidence and architecture-specific traps.
- `converter/`: code and DSL conversion guidance.

Product queries inherit their architecture's general knowledge while remaining
isolated from sibling products. For example, B200 includes `nvidia/blackwell/`
and `nvidia/blackwell/b200/`, whereas B300 excludes B200-only evidence.

## Entry points

- [Generic knowledge](generic/)
- [NVIDIA knowledge](nvidia/)
- [AMD knowledge](amd/)
- [Cross-architecture relationships](RELATIONS.md)

Use the scoped query instead of searching the entire tree when a target GPU is
known:

```bash
python3 gpu-wiki/scripts/query.py --arch h20 --section docs/
python3 gpu-wiki/scripts/query.py "gdn" --arch pro5000 --dsl cutedsl
python3 gpu-wiki/scripts/query.py "flash attention" --arch mi308x --dsl flydsl
```

`--section`, `--symptom`, `--kernel-type`, and `--operator` are optional
narrowing filters; `--arch` is the important isolation boundary.
