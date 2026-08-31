# Standalone CUDA backend

Use the standalone header from `backends/cuda_backend/atrex_timeline.cuh`. For embedded NVRTC source,
prepend the immutable header text and restore the candidate's line numbers with `#line`; do not rely
on a workspace-relative include path.

## Device contract

- One 16-byte record contains a 64-bit `%globaltimer` timestamp, 32-bit payload, and 32-bit tag.
- The tag contains 16-bit site id, 2-bit event kind, 10-bit physical SM, three user flag bits, and one
  backend-reserved committed bit.
- The buffer header owns ABI version, dimensions, launch id, capacity, owner count, records per owner,
  and status. Unknown layouts fail closed.
- A selected writer owns a deterministic contiguous slot range. There is no global event allocator.
- The host initializes exact capacity. A writer that exhausts its slots sets overflow and never writes
  out of bounds.
- The backend provides begin, end, instant, and counter. It does not insert CTA barriers, memory
  fences, or asynchronous completion waits. Marker placement determines the observed semantics.
- Disabled builds keep the call sites but compile the writer operations to no-ops.

## Owner layout

Describe writers per CTA in the manifest. Device code computes:

```text
owner = linear_cta * writers_per_cta + writer_ordinal
slot  = owner * records_per_owner + sequence
```

Construct one `Recorder` per selected owner outside the measured loop and reuse it for every event
owned by that writer. The claim table records the selected CTA/thread once; a second constructor for
the same owner fails the capture as `duplicate_owner`, even when it comes from the same thread.

The decoder derives 3D CTA, writer role, warp/group, and sequence from that formula. The manifest,
event dictionary, instrumented source, binary identity, and raw buffer belong to the same capture.

## Event dictionary

Each site has a stable numeric id and a name. Add boundary semantics, async domain, ordering, payload
meaning, and source anchor when they affect interpretation. These fields record AKA's declaration;
they do not prove that the chosen site is semantically correct. Set `kind` to `range`, `instant`, or
`counter` when known; the decoder rejects a runtime kind that contradicts the declaration.

## Capture lifecycle

Use `allocation_bytes` and `initialize_host_buffer` from the same header to allocate and initialize
the exact host-visible layout; the Python/Torch equivalents are in `adapter.py`. Pass the device
pointer through the candidate's existing launch path, synchronize after the target launch, and save
the bytes below the episode profile directory. Run `scripts/timeline.py decode` with the raw file,
manifest, event dictionary, clean source, instrumented source, workload identity, and optional
measurement artifact. For a multi-file kernel/wrapper change, point each source argument at a small
snapshot directory containing every changed source; directory hashes include relative names and file
contents and reject symlinks. The current sandbox evaluator binds correctness only to `kernel.py`, so
a multi-file snapshot remains diagnostic evidence; `decision_grade` requires a single-file source
snapshot until the evaluator records a canonical source-set digest.

An exploration receipt may use `--correctness passed` as an explicit agent report. A final
`decision_grade` receipt additionally requires `--correctness-evidence` pointing to the
sandbox-owned `.atrex_long_horizon/evaluations.jsonl` produced by the immutable evaluator for the
instrumented `kernel.py`.

Campaign workspaces expose `skills/` as a local symlink that is not uploaded implicitly. Add
`--input skills/autonomous-gpu-kernel-timeline/backends/cuda_backend` to a sandbox command that reads
the header or adapter at runtime. This deliberately selects the sandbox's custom dev-compatible
transport while preserving evaluator inputs; it does not authorize changing `profile_driver.py`.

Use the tool's structural verdict as a fact check. Interpret the resulting phases yourself.
`backends/cuda_backend/test_backend.cu` is the compact runnable example for allocation, launch,
synchronization, status handling, and raw-buffer export.
