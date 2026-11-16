#!/usr/bin/env python3
"""
TTGIR extraction tool.

Standalone function version, no class context required.

Usage:
    python extract_ttgir.py kernel.py -o kernel.ttgir

The input .py must LAUNCH the Triton kernel at import time (module-level call),
because Triton compiles lazily on first launch and only then writes the .ttgir
artifact into TRITON_CACHE_DIR. Layouts read from this TTGIR are the ground
truth for Triton->Gluon conversion (never fabricate layouts).
"""

import tempfile
import os
import sys
import importlib.util
import shutil
import argparse


def extract_ttgir(triton_code: str) -> str:
    """
    Extract TTGIR from Triton code.

    Args:
        triton_code: Triton kernel code (must launch the kernel at import time).

    Returns:
        ttgir_content: TTGIR text.

    Raises:
        FileNotFoundError: no .ttgir file was produced.
    """
    dump_dir = tempfile.mkdtemp()
    os.environ['TRITON_CACHE_DIR'] = dump_dir

    temp_file = None
    try:
        # Create a temporary module file.
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=dump_dir) as f:
            temp_file = f.name
            f.write(triton_code)

        # Import and execute it (triggers Triton compilation on kernel launch).
        module_name = os.path.splitext(os.path.basename(temp_file))[0]
        spec = importlib.util.spec_from_file_location(module_name, temp_file)
        if spec is None:
            raise ImportError(f"cannot load module spec: {module_name}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Scan for the .ttgir artifact in the cache dir.
        ttgir_content = None
        for root, dirs, files in os.walk(dump_dir):
            for file in files:
                if '.ttgir' in file:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        ttgir_content = f.read()
                    break
            if ttgir_content:
                break

        if not ttgir_content:
            raise FileNotFoundError("no .ttgir file found")

        return ttgir_content

    except Exception as e:
        print(f"extract failed: {e}", file=sys.stderr)
        raise

    finally:
        os.environ.pop('TRITON_CACHE_DIR', None)
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
        shutil.rmtree(dump_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Extract the TTGIR of a Triton kernel")
    parser.add_argument("input", help="input Triton code file")
    parser.add_argument("-o", "--output", help="output TTGIR file")
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        triton_code = f.read()

    try:
        ttgir = extract_ttgir(triton_code)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(ttgir)
            print(f"TTGIR saved to: {args.output}")
        else:
            print(ttgir)

    except Exception as e:
        print(f"extract failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
