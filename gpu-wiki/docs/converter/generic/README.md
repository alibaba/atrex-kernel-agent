# Vendor-Agnostic Conversion Tools

PyTorch→Triton conversion rules and model capability guidance for GPU-kernel work.

---

| File | Description |
|------|------|
| [PyTorch → Triton Conversion Guide](conversion-guide.md) | Reference guide for converting PyTorch code into optimized Triton kernels |
| [API Mapping: PyTorch → Triton](api_mapping.md) | Complete API mapping from PyTorch to Triton (element-wise, reduction, etc.) |
| [PyTorch → Triton Complete Conversion Rules](porting_rules.md) | Conversion principles, operator fusion priority, matmul handling |
| [Model Capability Selection Guide](model_config_guide.md) | Capability-tier and validation-depth guidance for GPU-kernel conversion and review |
