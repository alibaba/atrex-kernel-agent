#!/usr/bin/env python3
"""ncu-compatible front end for the PPU asight profiler (acu).

Activate by making this reachable as `ncu` on the gateway's PATH; `/usr/local/cuda/bin/ncu`
is both on that PATH and the path `_find_profile_tool` already probes, so a symlink there
needs no code change and no gateway restart:

    ln -s <repo>/tools/profile_ppu.py /usr/local/cuda/bin/ncu

AKA's local gateway builds an NVIDIA `ncu` command line and then parses the CSV with
NVIDIA metric names. `acu` is option-compatible except for three things, all handled here:

  * `--log-file PATH`      -> `--csv-file PATH` (acu writes the CSV directly)
  * `--target-processes A`  -> unsupported, dropped
  * metric names           -> PPU uses ppu__/cu__ where NVIDIA uses gpu__/sm__/dram__

Known PPU metric names are translated back in the emitted CSV whether or not we translated
them on the way in, so the gateway's `_parse_ncu_csv` sees the names it looks for even for
`--set full`, which asks for no explicit metric list. Unknown names are left untouched.

We also disable the GPM stream before each session: the host amperf collector otherwise
holds the device PCM and acu aborts partway through with "Device is not ready for
profiling". acu re-enables the stream when it exits, so this has to run every time.
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

ACU = "/usr/local/PPU_SDK/asight/bin/acu"
ASIGHT_BIN = "/usr/local/PPU_SDK/asight/bin"
ASIGHT_LIB = "/usr/local/PPU_SDK/asight/lib"
PPU_SMI = "/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi"

NVIDIA_TO_PPU = {
    "gpu__time_duration.sum": "ppu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "cu__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "ppu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "ppu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "cu__warps_active.avg.pct_of_peak_sustained_active",
}

PPU_TO_NVIDIA = {ppu: nvidia for nvidia, ppu in NVIDIA_TO_PPU.items()}


def translate_argv(argv):
    out = [ACU]
    csv_path = None
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--log-file":
            csv_path = argv[index + 1]
            out += ["--csv-file", csv_path]
            index += 2
            continue
        if token.startswith("--log-file="):
            csv_path = token.split("=", 1)[1]
            out += ["--csv-file", csv_path]
            index += 1
            continue
        if token == "--target-processes":
            index += 2
            continue
        if token.startswith("--target-processes="):
            index += 1
            continue
        if token in ("--metrics", "-m") or token.startswith("--metrics="):
            if token.startswith("--metrics="):
                raw = token.split("=", 1)[1]
                index += 1
            else:
                raw = argv[index + 1]
                index += 2
            names = []
            for name in raw.split(","):
                name = name.strip()
                if not name:
                    continue
                mapped = NVIDIA_TO_PPU.get(name)
                if mapped:
                    names.append(mapped)
                else:
                    names.append(name)
            out += ["--metrics", ",".join(names)]
            continue
        out.append(token)
        index += 1
    return out, csv_path


def release_gpm_stream():
    """Force an Enabled->Disabled transition, which is what actually releases the device PCM.

    Issuing `-s 0` while the stream is already Disabled is a driver no-op and acu still aborts
    with "Device is not ready for profiling", so the stream has to be primed to Enabled first.
    """
    if not os.access(PPU_SMI, os.X_OK):
        print(f"[ncu-shim] {PPU_SMI} not executable; skipping GPM release", file=sys.stderr)
        return
    base = [PPU_SMI, "gpm"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        base += ["-i", visible]
    subprocess.run(base + ["-s", "1"], capture_output=True, text=True)
    result = subprocess.run(base + ["-s", "0"], capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        print(f"[ncu-shim] GPM release failed: {detail}", file=sys.stderr)


def restore_metric_names(csv_path):
    path = Path(csv_path)
    if not path.is_file():
        return
    rows = list(csv.reader(path.read_text(encoding="utf-8", errors="replace").splitlines()))
    header_index = next(
        (i for i, row in enumerate(rows) if "Kernel Name" in row and "Metric Name" in row), None
    )
    if header_index is None:
        return
    column = [c.strip() for c in rows[header_index]].index("Metric Name")
    for row in rows[header_index + 1:]:
        if len(row) > column:
            row[column] = PPU_TO_NVIDIA.get(row[column], row[column])
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, quoting=csv.QUOTE_ALL).writerows(rows)


def main():
    argv = sys.argv[1:]
    if argv and argv[0] in ("--version", "-v"):
        return subprocess.run([ACU, "--version"]).returncode

    command, csv_path = translate_argv(argv)
    if csv_path and "-f" not in command and "--force-overwrite" not in command:
        command.insert(1, "-f")

    env = dict(os.environ)
    env["PATH"] = ASIGHT_BIN + os.pathsep + env.get("PATH", "")
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ASIGHT_LIB + (os.pathsep + pythonpath if pythonpath else "")

    release_gpm_stream()
    code = subprocess.run(command, env=env).returncode
    if csv_path:
        restore_metric_names(csv_path)
    return code


if __name__ == "__main__":
    sys.exit(main())
