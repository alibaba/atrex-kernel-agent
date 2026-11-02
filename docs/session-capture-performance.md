# Session capture performance evidence

This benchmark compares capture disabled (`ATREX_SESSION_CAPTURE=0`) with capture enabled (`=1`) on the same implementation. The disabled arm exercises the previous `Popen.communicate()` path; the enabled arm includes live capture and finalization. It measured the overhead introduced by capture in that implementation, not Kernel performance or an Agent's optimization quality.

![Capture-off/on latency and CPU curves at 1, 8, and 32 MiB](assets/session-capture-overhead.png)

Points are medians of five measured invocations; whiskers show the observed minimum and maximum, not confidence intervals. Lines connect measured points only.

## Results

All times are seconds. Peak RSS is MiB. Each arrow is **capture off → capture on**.

| Transcript per source (MiB) | Responses | Invocation wall time (s) | Supervisor CPU time (s) | Peak Supervisor RSS (MiB) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 256 | 0.070 → 0.194 | 0.004 → 0.092 | 31.2 → 41.5 |
| 8 | 2,048 | 0.094 → 0.689 | 0.014 → 0.593 | 55.7 → 141.6 |
| 32 | 8,192 | 0.127 → 2.324 | 0.043 → 2.186 | 139.9 → 471.8 |

At 32 MiB per source, capture adds about **2.20 seconds wall time**, **2.14 seconds Supervisor CPU**, and **332 MiB peak RSS** relative to the disabled path in this workload. This is a material cost, not evidence that capture is free. Archive construction and parsing produce in-memory objects as well as persisted bytes; the diagnostic byte cap is not a peak-memory cap. Concurrent sessions may amplify this resource cost.

All 36 invocations (30 measured plus six warm-ups) exited successfully, with no timeout or stderr. Returned stdout matched the fixture byte-for-byte. Every enabled invocation reported complete capture and the expected exact usage; no diagnostic limits were reached. The usage checks use synthetic counters and do not constitute a live-provider accounting audit.

[Raw results](assets/session-capture-performance.json) include every sample and warm-up, byte counts and stdout/fixture hashes, capture validity, the environment, script SHA-256, and hashes of the measured implementation files. No test module or plotting script is added to the repository.

## Interpretation for real Agent sessions

The benchmark emits hundreds to thousands of responses in a burst, without waiting for model generation. In the intended AKA workload, where an individual model response takes tens of seconds to minutes, the same transcript accumulates over a much longer interval. The approximately 2.20 seconds of additional wall time at 32 MiB is for the **entire 8,192-response synthetic invocation**, not for each response. Its large off/on ratio therefore should not be read as the slowdown of a real Agent session.

Capture reads and processes output concurrently with the CLI, so some of its work can overlap model generation and tool waits; final projection still adds synchronous work at completion. For model-latency-bound runs with sufficient CPU and memory, we therefore **expect Capture to contribute a small fraction of overall AKA wall time rather than be a major latency bottleneck**. This is a workload-based expectation, not an end-to-end result established by this benchmark. Seconds of burst-processing overhead do not prove negligible impact in every deployment: long-running polling, slow storage, and resource contention can add costs, and the measured memory increase still matters when many sessions run concurrently.

## Environment and measurement scope

- Measured on **2026-09-16**, Apple M5, 10 logical CPUs, 16 GiB RAM, macOS 26.5.2 arm64, CPython 3.14.6.
- Checkout base: `8d220f1875ef7ea29cd61e39498e613278b4c9b4`, with this change's Pi/Qoder accounting fixes applied. The raw file's `source_sha256` identifies the actual measured Python files independently of subsequent documentation edits.
- Workload sizes: **1, 8, 32 MiB per source**, respectively 256, 2,048, and 8,192 final assistant responses. Each response is exactly 4,096 bytes, followed by one small terminal event. Each invocation writes one identical native transcript and stdout: the 32 MiB point therefore presents approximately **64 MiB across the two sources**, not 32 MiB total.
- Both arms run the same real child process via `run_bounded()`, with a fresh workspace and provider home. The home contains 64 unrelated small transcript files to exercise scoped discovery. No real Agent CLI, model, GPU, credentials, or network is used; the child emits synthetic Claude-format events.
- One warm-up per size and mode is excluded. Five measured runs per size and mode use alternating off/on ordering. Every sample runs in a fresh Supervisor Python process.
- The wall timer surrounds the complete `run_bounded()` call: capture startup, process execution, pipe/native reads, discovery, writes, final native sync, and conversation projection. Python imports, fixture/home preparation, and post-run assertions are outside it.
- CPU uses `time.process_time()`: Supervisor CPU across its threads, excluding child-process CPU. Peak RSS uses `resource.getrusage(RUSAGE_SELF)` for the fresh Supervisor process, including its imports; it is not the RSS delta over the timed interval.
- Files are on the workstation's local temporary storage. Filesystem caches are warm; there is no cache flush or CPU isolation. The implementation polls native files at 1 Hz; short invocations may only reach the final native sync.

This burst-output benchmark exposes capture overhead without hiding it behind model/network wait time. Its off/on ratios are **not** production Agent slowdown ratios. It does not cover long-running discovery, large resumed histories, many subagents, all provider formats, Linux storage, or concurrent Campaigns. The three points cannot establish asymptotic complexity. Those conditions require additional measurements before claiming negligible production overhead. Operators can use [the capture configuration](session-observability.md#disable-capture-and-roll-back) if the extra resource cost is unacceptable.

## Read the archived samples

These measurements describe the 2026-09-16 implementation identified by hashes in the
raw JSON. They are retained as historical capture-cost evidence, not current Light
performance measurements. The old checkout-import benchmark no longer runs against the
installed module layout. Current capture selection is documented in
[Session observability](session-observability.md#disable-capture-and-roll-back).

Summarize the unchanged archived samples from the repository root:

```bash
python3 - docs/assets/session-capture-performance.json <<'PY'
import json
import statistics
import sys

with open(sys.argv[1]) as source:
    data = json.load(source)
for size in data["sizes_mib"]:
    for enabled in (False, True):
        rows = [r for r in data["samples"]
                if r["size_mib"] == size and r["capture"] == enabled]
        print(f"{size} MiB capture={'on' if enabled else 'off'} n={len(rows)}")
        for key in ("wall_seconds", "supervisor_cpu_seconds", "peak_rss_mib"):
            values = [row[key] for row in rows]
            print(f"  {key}: median={statistics.median(values):.6f}"
                  f" min={min(values):.6f} max={max(values):.6f}")
PY
```
