#!/usr/bin/env python3
"""Refresh the reviewed Agate Skill from a local gateway checkout, or check drift."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aka.light.workspace.skill_manifest import SKILL_MANIFEST


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gateway", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = SKILL_MANIFEST["atrex-gpu-gateway"]
    source = args.gateway / "skills/atrex-gpu-gateway"
    target = ROOT / "src/aka/light/workspace/resources" / manifest.root
    actual = {p.relative_to(source).as_posix() for p in source.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if actual != set(manifest.files):
        parser.error("Skill file list changed; review and update SKILL_MANIFEST before syncing")
    translations = json.loads((ROOT / "tools/agate_skill_translations.json").read_text())
    for name, digests in translations.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digests["upstream_sha256"]:
            parser.error(f"Upstream Skill changed: {name}; review the English translation and refresh its digests")
        if hashlib.sha256((target / name).read_bytes()).hexdigest() != digests["english_sha256"]:
            parser.error(f"English translation changed: {name}; review it and refresh its digest")
    changed = []
    for name in manifest.files:
        if name == "reference/aka-workspace.md" or name in translations:
            continue  # Preserve AKA integration guidance and reviewed English translations.
        data = (source / name).read_bytes()
        destination = target / name
        if not destination.is_file() or destination.read_bytes() != data:
            changed.append(name)
            if not args.check:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
    print(f"Agate Skill: {len(manifest.files)} files; {len(changed)} differing files")
    return int(args.check and bool(changed))


if __name__ == "__main__":
    raise SystemExit(main())
