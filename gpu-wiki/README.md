# GPU Wiki

GPU Wiki is the curated knowledge and reference-code base for GPU kernel
development, optimization, profiling, and cross-platform migration.

The documentation is **architecture first**:

```text
docs/<vendor>/<architecture>/<role>/<dsl-or-topic>/...
```

Vendor-general knowledge lives in `<vendor>/common/`. Product-only knowledge
uses an overlay below its architecture, such as `nvidia/blackwell/b200/` or
`amd/cdna3/mi308x/`. See [the documentation index](docs/README.md) for the full
tree.

## Recommended workflow

1. Identify the target GPU or architecture.
2. Run `scripts/query.py --arch <target>` to establish the safe search scope.
3. Optionally narrow by role, DSL, symptom, kernel type, or operator.
4. Read concise `kernel-opt/` cards first, then detailed `ref-docs/` and
   `pitfalls/` evidence.
5. Use `reference-kernels/` for concrete implementation patterns and upstream
   sources or vendor documentation for API/ISA ground truth.

Examples:

```bash
# Hardware facts: section is optional but useful for narrowing.
python3 gpu-wiki/scripts/query.py --arch a100 --area docs --section hardware-specs
python3 gpu-wiki/scripts/query.py --arch h20 --area docs --section hardware-specs
python3 gpu-wiki/scripts/query.py --arch pro5000 --area docs --section hardware-specs

# Search an operator within an isolated architecture/DSL scope.
python3 gpu-wiki/scripts/query.py "gdn" --arch sm120 --dsl cutedsl
python3 gpu-wiki/scripts/query.py "flash attention" --arch mi308x --dsl flydsl

# Optional diagnostic filters.
python3 gpu-wiki/scripts/query.py --arch b200 \
  --section kernel-opt --symptom pipeline-stalls

# Typo-tolerant lookup after architecture/DSL scope isolation.
python3 gpu-wiki/scripts/query.py rms_nrom --arch h20 --fuzzy
python3 gpu-wiki/scripts/query.py flash_attenion --arch mi308x --dsl flydsl --fuzzy

# Search only concrete source implementations (docs + reference kernels are
# searched together by default when --area is omitted).
python3 gpu-wiki/scripts/query.py gemm --arch h20 --area reference-kernels

# Restrict references to an explicitly classified usability status.
python3 gpu-wiki/scripts/query.py gemm --arch sm120 \
  --area reference-kernels --status diagnostic-archive

# Restrict by upstream source or source role. Test/build/package files are
# omitted by default; add --include-auxiliary when they are specifically needed.
python3 gpu-wiki/scripts/query.py gemm --arch b300 --source cutlass --kind kernel

# A copied source filename or relative path works without fuzzy mode.
python3 gpu-wiki/scripts/query.py dense_blockscaled_gemm_sm103.py \
  --arch sm103 --area reference-kernels
```

`--section`, `--symptom`, `--kernel-type`, and `--operator` are optional. They
reduce results; they are not required for normal keyword or architecture
search. `--status` selects reference kernels with an explicit usability status
including the honest `unclassified` fallback. `--source` and `--kind` narrow by
upstream project and source role. Unknown filter values fail closed.

Normal keyword queries treat whitespace and filename/path separators (`_`,
`-`, `.`, `/`) as equivalent term boundaries. `--fuzzy` applies
`SequenceMatcher` plus trigram similarity to normalized titles, paths, and
summaries for actual spelling uncertainty. Architecture, vendor, DSL, and
section filters remain hard boundaries and are applied first. Adjust
false-positive tolerance with `--fuzzy-threshold` (default `0.78`). Fuzzy
matching does not make architecture-specific advice portable.

By default, keyword searches cover curated `docs/` pages, source files under
`reference-kernels/`, and a small manifest-selected set of substantive
reference guides. Use `--area docs` or
`--area reference-kernels` to isolate one area. A documentation-role filter
such as `--section kernel-opt` selects `docs/` pages because reference kernels
do not have a documentation role; omit `--section` or query the reference area
separately when concrete implementations are needed.

Reference-kernel scope comes from
[`manifest.json`](manifest.json) first,
with path inference retained as a fallback for files not yet declared. Search
results show an explicit status when one is available.

The same manifest also provides the architecture/vendor scope for `docs/`,
including exact cross-architecture overrides for selected common pages.
`--arch blackwell` is a family query covering B200 and B300; use `--arch sm100`
for the exact SM100/B200 scope and `--arch sm103` for B300.

Accepted card aliases include:

| Query input | Canonical scope |
|---|---|
| A100 / SM80 | NVIDIA Ampere |
| H20 / H100 / H200 / SM90 | NVIDIA Hopper |
| B200 / GB200 | NVIDIA Blackwell product overlay |
| B300 / GB300 / SM103 | NVIDIA Blackwell Ultra |
| RTX PRO 5000 / Pro5000 / SM120 | NVIDIA Blackwell GeForce/workstation |
| MI300X / MI308X / gfx942 | AMD CDNA3 and product overlays |
| MI355X / gfx950 | AMD CDNA4 |

## Hardware fact entry points

- [A100 / Ampere](docs/nvidia/ampere/hardware-specs/hardware_specs_ampere.md)
- [H20, H100, H200 / Hopper](docs/nvidia/hopper/hardware-specs/hardware_specs_hopper.md)
- [B200](docs/nvidia/blackwell/b200/hardware-specs/hardware_specs_b200.md)
- [B300](docs/nvidia/blackwell-ultra/hardware-specs/hardware_specs_b300.md)
- [RTX PRO 5000 / SM120](docs/nvidia/blackwell-geforce/hardware-specs/hardware_specs_sm120.md)
- [MI300X](docs/amd/cdna3/mi300x/hardware-specs/hardware_specs_mi300x.md)
- [MI308X](docs/amd/cdna3/mi308x/hardware-specs/hardware_specs_mi308x.md)
- [MI355X](docs/amd/cdna4/hardware-specs/hardware_specs_mi355x.md)

## Repository areas

- [`docs/`](docs/): curated architecture-scoped knowledge.
- [`reference-kernels/`](reference-kernels/): runnable or illustrative kernels,
  already organized by hardware architecture and framework.
- [`manifest.json`](manifest.json): explicit search metadata and exact-file
  overrides for indexed repository areas.
- `reference-projects/`: optional local upstream source snapshots.
- `3rdparty/`: supplementary external knowledge, used after local scoped search.
- `scripts/`: query, consistency, and maintenance checks.

The source wiki does not define downstream task-runtime filenames or submission
protocols. Examples must be adapted to the consuming benchmark harness.
