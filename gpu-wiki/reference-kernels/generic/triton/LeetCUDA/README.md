# LeetCUDA Project Triton Kernel Reference Implementation

Triton kernel reference implementations from the LeetCUDA project, including basic operator exercises: vector addition, softmax, layer norm, attention state merging.

---

| Kernel | Description |
|--------|------|
| [cuda_merge_attn_states.py](cuda_merge_attn_states.py) | CUDA version of Attention state merging |
| [test_merge_attn_states.py](test_merge_attn_states.py) | Attention state merging test |
| [triton_fused_softmax.py](triton_fused_softmax.py) | Triton fused Softmax kernel |
| [triton_layer_norm.py](triton_layer_norm.py) | Triton Layer Normalization kernel |
| [triton_merge_attn_states.py](triton_merge_attn_states.py) | Triton version of Attention state merging kernel |
| [triton_vector_add.py](triton_vector_add.py) | Triton vector addition kernel |
