# Remote PPU timeline capture

Use this contract when the instrumented kernel is compiled and launched in a remote PPU environment.
The remote profiler does not rewrite source automatically: the agent preserves a clean snapshot,
creates one temporary instrumented snapshot, and uploads only the selected skill and attempt inputs.

## Attempt layout

Keep one self-contained attempt below the episode profile directory:

```text
profiles/episode_N/timeline/attempt-N/
├── clean-source/
├── instrumented-source/
├── events.json
├── manifest.json
├── correctness.json
├── harness/
│   └── capture_ppu.py
└── evidence/
```

The source entries may be files instead of directories for a single-file kernel. The harness must
open only declared attempt files and skill resources; do not depend on an undeclared local checkout.

## Sandbox transport

When the configured sandbox exposes the PPU target, upload the complete PPU skill because the
harness needs the adapter, header, decoder, and possibly ACU extraction:

```bash
ATTEMPT=profiles/episode_N/timeline/attempt-N
python tools/sandbox.py --kind profile --hardware <PPU_HARDWARE> \
  --input skills/ppu-acu-joint-profile \
  --sync "$ATTEMPT" -- \
  env PPU_DEVICE="${PPU_DEVICE:?set PPU_DEVICE}" \
      PPU_PROFILE_SKILL=skills/ppu-acu-joint-profile \
      python "$ATTEMPT/harness/capture_ppu.py"
```

Custom `--input` selects the sandbox's isolated dev-compatible transport. It packages the explicitly
named skill and attempt command inputs, chooses inline or OSS transport according to payload size,
and synchronizes only the declared attempt directory. Keep raw captures below the attempt; raise
`--max-output-file-mb` only when an expected evidence file exceeds the default bound.

If the PPU is available only through another authorized Pod executor, stage the same skill and
attempt inputs and return the same attempt outputs. Transport may change; source snapshots, capture
identity, decode requirements, and cleanup rules do not.

## Harness responsibilities

The attempt harness performs these operations inside the remote environment:

1. Select the physical PPU before compiling or allocating the buffer.
2. Compile through the target project's real PPU compiler/runtime and preserve its normal
   architecture flags, shared-library loading, and launch route.
3. Use `backends/ppu_backend/adapter.py` to initialize the exact host/device ABI. Either include
   `ppu_timeline.cuh` through the real project path or call `instrument_source` so HGGC receives the
   uploaded backend header through a native include.
4. Pass the device buffer through the existing launch interface without changing useful-work
   predicates, barriers, waits, or control flow.
5. Declare the `%globaltimer` nanosecond contract in the manifest. If an environment sanity check
   materially contradicts that documented unit, reject the attempt rather than fitting a scale.
6. Warm up and validate the representative output, write the accepted correctness artifact, then
   capture one identified target launch, synchronize, and save the entire raw buffer with
   `save_torch_buffer`.
7. Write manifest v4 and the event dictionary from actual grid, block, launch, device, runtime,
   workload, writer, source, timer, and correctness facts.
8. Run `scripts/timeline.py decode` before the remote job exits. A rejected capture is an attempt
   failure, not evidence to interpret.

The harness may copy or import adapter functions, but it must not replace the target project's real
compiler/runtime path with a standalone CUDA-SDK executable.

## Return and cleanup

Return the clean and instrumented snapshots, correctness evidence, raw capture, manifest, event
dictionary, canonical events, Perfetto trace, summary, and accepted receipt
in the attempt directory. Preserve failed attempt diagnostics under a new attempt number rather than
overwriting prior evidence.

Restore the probe-free source after the attempt. ACU collection remains a separate probe-free launch
and is not part of this remote timeline command unless a later, optional joint-analysis decision
requires both completed evidence sets.
