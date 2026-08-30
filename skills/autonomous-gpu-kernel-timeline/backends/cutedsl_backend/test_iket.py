#!/usr/bin/env python3
"""Small CuTe DSL kernel used to validate the IKeT capture adapter."""

import cutlass
import cutlass.cute as cute


@cute.kernel
def timeline_probe():
    thread, _, _ = cute.arch.thread_idx()
    cute.experimental.iket.mark("probe_start", thread)
    token = cute.experimental.iket.range_start("probe_body", thread)
    cute.experimental.iket.range_end(token, thread)
    cute.experimental.iket.mark("probe_end", thread)


@cute.jit
def launch_probe():
    timeline_probe().launch(grid=(1, 1, 1), block=(32, 1, 1))


def main() -> None:
    compiled = cute.compile(launch_probe)
    compiled()
    cutlass.cuda.stream_sync(cutlass.cuda.default_stream())
    print("cutedsl iket fixture: ok")


if __name__ == "__main__":
    main()
