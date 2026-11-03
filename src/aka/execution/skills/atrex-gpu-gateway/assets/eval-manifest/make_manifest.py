"""Build inline or OSS-archive manifests for the accompanying add-one candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", choices=["triton", "cuda", "cuda-native", "cuda-cutlass"])
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--git-mirror", action="append", default=[], help="Preferred HTTPS mirror for cuda-cutlass; repeat for priority order")
    parser.add_argument("--output", type=Path, default=Path("solution.json"))
    args = parser.parse_args()
    root = Path(__file__).parent / args.candidate
    files = {f"impl/{path.name}": path.read_text(encoding="utf-8") for path in sorted(root.iterdir())
             if path.is_file() and path.suffix in {".py", ".cpp", ".cu"}}
    if args.candidate not in {"cuda-native", "cuda-cutlass"}:
        files["impl/__init__.py"] = ""
    source = {"type": "inline", "files": files}
    if args.archive:
        archive = args.output.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for name, content in files.items():
                bundle.writestr(name, content)
        source = {"type": "oss_archive", "archive": "archives/solution.zip",
                  "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                  "files": {name: hashlib.sha256(content.encode("utf-8")).hexdigest()
                            for name, content in files.items()}}
    dependencies = ([{"id": "triton", "kind": "pip", "requirement": "triton>=3"}]
                    if args.candidate == "triton" else [
                        {"id": "ninja", "kind": "pip", "requirement": "ninja>=1.11"},
                        {"id": "cuda", "kind": "system", "name": "CUDA toolkit", "version": ">=12"},
                        {"id": "extension", "kind": "build", "execution": "entry_point",
                         "toolchain": "torch.utils.cpp_extension.load", "depends_on": ["ninja", "cuda"],
                         "inputs": ["impl/binding.cpp", "impl/kernel.cu"], "outputs": ["build/extension.so"]},
                    ])
    manifest = {"schema_version": 1, "name": f"add-one-{args.candidate}", "version": "1",
                "entry_point": "impl/entry.py::Model", "source": source,
                "dependencies": dependencies, "targets": {"hardware": [args.gpu], "num_gpus": 1,
                "shape_ids": ["0", "1"]}}
    if args.candidate in {"cuda-native", "cuda-cutlass"}:
        manifest["dependencies"] = dependencies[:2]
        manifest["entry_point"] = {
            "type": "cuda", "symbol": "launch_add_one",
            "build": {"toolchain": "torch_extension", "sources": ["impl/kernel.cu"],
                      "cflags": ["-O3"], "cuda_cflags": ["-O3"]},
            "inputs": [{"kind": "tensor", "name": "x", "dtype": "float32"}],
            "outputs": [{"name": "out", "like": "x"}],
            "arguments": [{"ref": "x"}, {"ref": "out"}, {"numel": "x"}],
        }
    if args.candidate == "cuda-cutlass":
        manifest["dependencies"].append({
            "id": "cutlass", "kind": "git", "repository": "https://github.com/NVIDIA/cutlass.git",
            "ref": "v3.9.2", "mirrors": args.git_mirror, "include_dirs": ["include", "tools/util/include"],
        })
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
