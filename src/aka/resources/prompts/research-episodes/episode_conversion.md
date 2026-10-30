## Required Triton-to-Gluon conversion

Use this Episode for conversion, preserving algorithm, tiling, signatures, and evaluator behavior.
Do not submit another Triton kernel. Extract TTGIR before writing Gluon and derive layouts from the
real Kernel; never fabricate them. Repair compile/correctness/parity defects before handoff.
The candidate must pass correctness and be plausibly within {{CONVERSION_TOLERANCE}} of incumbent
latency; the Supervisor independently enforces parity.

{{CONVERSION_RESEARCH}}
