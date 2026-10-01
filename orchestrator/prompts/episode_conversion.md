## Required Triton-to-Gluon conversion

Use this Episode for conversion, preserving algorithm, tiling, signatures, and evaluator behavior.
Do not submit another Triton kernel. Extract TTGIR before writing Gluon and derive layouts from the
real Kernel; never fabricate them. Repair compile/correctness/parity defects before handoff.
The candidate must pass correctness and be plausibly within {{CONVERSION_TOLERANCE}} of incumbent
latency; the Supervisor independently enforces parity.

At Episode start, make this conversion-specific Wiki query once; no separate general query is needed:

```bash
python3 tools/sandbox.py --kind wiki-query "Target product {{PLATFORM}}, runtime architecture {{ARCH}}, operator {{OPERATOR}}. Return the full product specification and only the matching Triton-to-Gluon conversion guidance." --brief
```

Resolve an unknown architecture through the Runtime before querying. Matching records are
`nvidia.blackwell.any.converter.blackwell` for `sm_100`/`sm_103`,
`nvidia.hopper.any.converter.hopper` for `sm_90`, `amd.cdna3.any.converter.cdna3` for `gfx94*`, and
`amd.cdna4.any.converter.cdna4` for `gfx95*`; never substitute a sibling architecture's record.
Treat guidance as hypotheses to verify; do not copy Wiki metadata into the Journal. Additional
targeted queries are allowed when new evidence raises a materially different question.
