# Converter — framework-transition sheets

Each doc here is a **single sheet consumed by one framework-transition session**. A sheet is the
minimal API map + non-obvious pitfalls + **pointers to the local Triton source** (`reference-projects/
triton`) — the session reads one sheet and opens source only for the construct it converts. There is
no per-topic file to wander through.

## Router — pick the sheet by (transition, arch)

| Transition | Fires when | Session / consumer | Sheet |
|-----------|-----------|--------------------|-------|
| PyTorch → Triton | first optimization iteration (V0 is a PyTorch wrapper) | `kernel-optimize` | [pytorch-to-triton.md](pytorch-to-triton.md) |
| Triton → Gluon, **Blackwell sm_100** | Triton plateaus (`--convert-after`) | `gpu-kernel-convert` (`prompts/convert.md`) | [nvidia/blackwell.md](nvidia/blackwell.md) |
| Triton → Gluon, **Hopper sm_90** | ″ | ″ | [nvidia/hopper.md](nvidia/hopper.md) |
| Triton → Gluon, **CDNA3 gfx94x** | ″ | ″ | [amd/cdna3.md](amd/cdna3.md) |
| Triton → Gluon, **CDNA4 gfx95x** | ″ | ″ | [amd/cdna4.md](amd/cdna4.md) |

The convert session selects the row by the real runtime arch (from hardware ground truth), reads that
one sheet, and does not read the others.
