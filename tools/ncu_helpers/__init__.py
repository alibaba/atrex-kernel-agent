# Copyright 2026 Alibaba Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Nsight Compute (.ncu-rep) parsing helpers for the NVIDIA profiling toolchain.

Modules:
    ncu_utils              -- ncu_report loading + safe metric access + curated metric sets
    analyze_reports        -- dump key/all metrics from a .ncu-rep to JSON
    extract_stall_hotspots -- per-PC / per-source-line stall attribution (pcsamp metrics)
    envelope               -- v1 JSON envelope (VeloQ-compatible contract)
    row_key                -- stable per-row keys + cross-capture diff
    source_metrics         -- per-line / per-SASS metric attribution (VeloQ port)
    warp_stalls            -- warp-stall attribution from timed_warp_samples (VeloQ port)
    disasm                 -- structured source-correlated SASS (+PTX) (VeloQ port)
    source_evidence        -- one entry point that runs disasm/warp_stalls/
                              source_metrics and writes source_evidence_manifest.json

The envelope/row_key/source_metrics/warp_stalls/disasm modules are a dependency-
free Python port of VeloQ's `ncu` verbs onto the same ncu_report API. They emit
independent evidence and do NOT feed classify_ncu.py. profile_iter_nvidia.sh drives
them through source_evidence on `--source` runs; summary.txt's LOCALIZE line
points the agent at the relevant artifact per symptom.

These are also importable as top-level modules when this directory is placed on
sys.path (as profile_iter_nvidia.sh / extract_nvidia_asm.py do via NCU_HELPERS).
"""
