# GPU Wiki Documentation

Knowledge base for GPU kernel programming and optimization, organized **vendor-first**: pick the vendor, then the topic/DSL. Each topic folder consolidates everything for that topic — reference articles, optimization cards/hands-on, and pitfalls (under a `pitfalls/` subfolder). Architecture-specific material lives in `sm90/` `sm100/` `sm120/` `gfx942/` `gfx950/` subfolders.

---

| Directory | Description |
|-----------|-------------|
| [generic/](generic/) | Vendor-agnostic GPU optimization theory, Triton hands-on patterns, and PyTorch→Triton conversion |
| [nvidia/](nvidia/) | All NVIDIA content, **architecture-first**: `common/` (cross-arch: ptx, profiling, cutedsl fundamentals, theory, hardware-specs), `hopper/` (sm90), `blackwell/` (sm100), `blackwell-geforce/` (sm120) |
| [amd/](amd/) | All AMD content: `common` (theory + hands-on + gfx942/gfx950), `flydsl`, `gluon`, `converter`, `hardware-specs` |
| [RELATIONS.md](RELATIONS.md) | Cross-architecture relationships: reading paths, conflicts, and differences when migrating between vendors/architectures |

## How to navigate

1. Determine the **vendor** (nvidia / amd / generic) and the **topic/DSL** (cutedsl, ptx, gluon, flydsl, …).
2. Open `‹vendor›/‹topic›/README.md` for that topic's file index.
3. Within a topic: reference articles and optimization cards sit at the topic root; arch-specific material is under `sm90/` `sm100/` `sm120/` `gfx942/` `gfx950/`; negative knowledge is under `pitfalls/`.
4. For cross-vendor / cross-architecture migration, read [RELATIONS.md](RELATIONS.md) first.
</content>
