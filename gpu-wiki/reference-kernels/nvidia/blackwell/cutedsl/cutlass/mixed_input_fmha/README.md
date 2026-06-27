# Blackwell CUTLASS Mixed-Input FMHA Kernels

Reference implementations of mixed-precision FMHA (Fused Multi-Head Attention) using the CUTLASS framework on the Blackwell architecture.

---

| Kernel | Description |
|--------|-------------|
| [mixed_input_fmha_decode.py](mixed_input_fmha_decode.py) | Mixed-precision FMHA Decode |
| [mixed_input_fmha_prefill_d256.py](mixed_input_fmha_prefill_d256.py) | Mixed-precision FMHA Prefill (d=256) |
| [mixed_input_fmha_prefill_d512.py](mixed_input_fmha_prefill_d512.py) | Mixed-precision FMHA Prefill (d=512) |
| [prefill_helpers.py](prefill_helpers.py) | Prefill helper utility functions |
