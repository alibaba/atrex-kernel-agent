# Hopper CuTeDSL FlashInfer Kernels

Reference kernel implementations for the FlashInfer project on the Hopper (SM90) architecture using CuTeDSL.

---

| Kernel | Description |
|--------|-------------|
| [fused_add_rmsnorm.py](fused_add_rmsnorm.py) | Fused addition + RMSNorm |
| [gdn_decode_bf16_state.py](gdn_decode_bf16_state.py) | GDN Decode BF16 state management |
| [gdn_decode_mtp.py](gdn_decode_mtp.py) | GDN Decode MTP mode |
| [gdn_decode_nontranspose.py](gdn_decode_nontranspose.py) | GDN Decode non-transpose mode |
| [gdn_decode_pretranspose.py](gdn_decode_pretranspose.py) | GDN Decode pre-transpose mode |
| [layernorm.py](layernorm.py) | LayerNorm kernel |
| [norm_utils.py](norm_utils.py) | Normalization utility functions |
| [rmsnorm.py](rmsnorm.py) | RMSNorm kernel |
| [ssd_kernel.py](ssd_kernel.py) | SSD (State Space Duality) kernel |
