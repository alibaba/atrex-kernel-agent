#!/usr/bin/env python3
"""Installed compatibility entry for Execution-owned NVIDIA assembly analysis."""
from aka.execution.profiling.nvidia.extract_nvidia_asm import *  # noqa: F401,F403
from aka.execution.profiling.nvidia.extract_nvidia_asm import _find_ncu_helpers, main

if __name__ == "__main__":
    main()
