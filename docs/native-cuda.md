# Native CUDA on Agate

Light campaigns with `framework=cuda` author separate CUDA sources and a manifest.
The installed `atrex-gpu-gateway` Skill owns the format and usage instructions:
[AKA workspace contract](../src/aka/light/workspace/resources/skills/atrex-gpu-gateway/reference/aka-workspace.md)
and [Agate manifest v1](../src/aka/light/workspace/resources/skills/atrex-gpu-gateway/reference/eval-manifest.md).

Use `--sandbox-profile pre` to explicitly select Agate preproduction for integration tests.
This does not change global Agate settings or Diamond.

The Skill is copied verbatim from the gateway repository and included in the AKA
package and Agent Skill discovery. Update it with:

```bash
python3 tools/sync_agate_skill.py /path/to/atrex-gpu-gateway
python3 tools/sync_agate_skill.py /path/to/atrex-gpu-gateway --check
```

Validation (2026-10-27): 22 offline regressions cover archive digests, Git promotion,
header/declaration identity, crash recovery, measurement deduplication, Skill
installation, and numerical failure classification. Both Light and Legacy wheels
build from their source distributions; the bundled Agate Skill is byte-identical
to the gateway-owned source. L20N preproduction exercised native CUDA Evaluate,
multi-file CUTLASS, targeted numerical inputs, Profile, two-shape native ABBA and a
deliberately incorrect candidate. One ABBA allocation encountered repository networking
failure; the existing infrastructure retry recovered it and both shapes passed. Check/Disassemble and SOL/SSH do not accept native manifests; unsupported
native requests stop without a Dev fallback.
