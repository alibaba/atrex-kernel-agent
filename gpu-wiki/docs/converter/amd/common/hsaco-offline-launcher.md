# HSACO Offline Launcher Guide

## Use Cases

When a remote AMD GPU environment **does not have Triton/Gluon installed**, but you need to run Triton/Gluon-compiled kernels:
1. Compile kernels into `.hsaco` files in an environment with Triton
2. Deploy the `.hsaco` file to the remote environment
3. The remote environment only needs PyTorch + ROCm to run

## Two Usage Modes

### Mode A: C++ Launcher Mode (Recommended for Production Deployment)

Compile a C++ `hsaco_launcher.so` extension, providing a `launch_with_kernarg()` interface via pybind11.

**Advantages**: Most stable performance, HIP calls are completed at the C++ layer
**Disadvantages**: Requires remote compilation of the C++ extension

### Mode B: Pure Python Single-File Embedding Mode (Recommended for Quick Testing)

The hsaco is embedded into the Python file as a base64 string, with HIP API calls made directly via ctypes.

**Advantages**: Single-file self-contained, no need to compile any .so
**Disadvantages**: ctypes calls to HIP require attention to subtle compatibility issues

## Usage Workflow

### Step 1: Compile HSACO Variants

Compile in an environment with Triton/Gluon:

```python
from submission_gluon_v4 import compile_stage1, compile_stage2

# Determine constexpr grid based on _get_split_config
# For example: batch=4 -> NUM_KV_SPLITS=16, batch=256 -> NUM_KV_SPLITS=4
compiled = compile_stage1(batch=4, num_heads=16, NUM_KV_SPLITS=16, ...)
hsaco_data = compiled.asm['hsaco']  # Extract hsaco binary
```

### Step 2: Extract Kernel Metadata

# Parse amdhsa.kernels segment to get arg layout
result = subprocess.run(['/opt/rocm/bin/llvm-readobj', '--notes', 'kernel.hsaco'],
                       capture_output=True, text=True)

Key information:
- `kernarg_segment_size`: Total kernarg size (bytes)
- Per-arg `offset`, `size`, `value_kind` (global_buffer/by_value)

### Step 3: Deploy and Load

#### Mode A: C++ Launcher

```bash
# Remote compilation
cd "${RTP_TOOLS_ROOT:?set RTP_TOOLS_ROOT}" && ROCM_PATH=/opt/rocm python3 setup.py build_ext --inplace
```

```python
import hsaco_launcher
hsaco_launcher.launch_with_kernarg(
    hsaco_path, kernel_name, kernarg_bytes,
    grid_x, grid_y, grid_z, block_x, block_y, block_z,
    shared_mem_bytes
)
```

#### Mode B: Pure Python Single-File

```python
import ctypes, base64, struct

hip = ctypes.CDLL("libamdhip64.so.7")  # Must call after torch import
hip.hipModuleLoadData.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
hip.hipModuleLoadData.restype = ctypes.c_int
hip.hipModuleGetFunction.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
hip.hipModuleGetFunction.restype = ctypes.c_int
hip.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p, ctypes.c_uint] * 3 + \
    [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, \
     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
hip.hipModuleLaunchKernel.restype = ctypes.c_int

# Load hsaco
binary = base64.b64decode(EMBEDDED_B64_STRING)
buf = ctypes.create_string_buffer(len(binary))
ctypes.memmove(buf, binary, len(binary))
mod = ctypes.c_void_p()
hip.hipModuleLoadData(ctypes.byref(mod), buf)
func = ctypes.c_void_p()
hip.hipModuleGetFunction(ctypes.byref(func), mod, kernel_name.encode())

# Build kernarg (see kernarg layout rules below)
# Launch
HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(0x02)
HIP_LAUNCH_PARAM_END = ctypes.c_void_p(0x03)
extra = (ctypes.c_void_p * 6)(
    HIP_LAUNCH_PARAM_BUFFER_POINTER, ctypes.c_void_p(ctypes.addressof(argbuf)),
    HIP_LAUNCH_PARAM_BUFFER_SIZE, ctypes.c_void_p(ctypes.addressof(buf_size)),
    HIP_LAUNCH_PARAM_END, None
)
hip.hipModuleLaunchKernel(func, gx, gy, gz, bx, by, bz, shared_mem, None, None, extra)
```

## Key Concepts

### Triton Kernel Parameter Classification

| Category | Description | In hsaco | Passing Method |
|------|------|------------|---------|
| **Tensor Pointers** | `tl.pointer_type` | Not baked | kernarg buffer (8-byte pointer) |
| **Regular Scalars** | `int`, `float`, etc. | Specialized (baked into hsaco) | No need to pass |
| **`do_not_specialize` Scalars** | Marked as not specialized | Not baked | kernarg buffer (by type size) |
| **`constexpr` Parameters** | `tl.constexpr` / `gl.constexpr` | Baked into hsaco | Different values compile to different hsaco |

### kernarg Buffer Layout Rules

ROCm kernel parameters are passed via `HIP_LAUNCH_PARAM_BUFFER`:

1. **Pointer parameters**: 8 bytes, 8-byte aligned
2. **float32 parameters**: 4 bytes, 4-byte aligned
3. **int32 parameters**: 4 bytes, 4-byte aligned
4. **float followed by pointer**: Requires 4 bytes of padding to align to 8

**Important**: All scalar parameters must be packed in 4-byte units (i32/f32), and cannot be packed in 8-byte units.
This is the most common source of precision bugs—packing int as an 8-byte pointer will cause incorrect offsets for all subsequent parameters.

### Gluon Kernel's Additional Pointer Parameters

The Gluon compiler adds **2 implicit pointers** at the end of kernel parameters (global_scratch and profile_scratch).
These are not visible in the Python function signature, but appear in the hsaco's `.amdhsa.kernels` metadata.

**Handling**: Append 2 null pointers (value 0) at the end of the kernarg buffer:
```python
# 5 explicit pointers + sm_scale + 6 stride + 2 implicit pointers
_build_kernarg([
    ptr(q.data_ptr()), ptr(kv_buffer_fp8.data_ptr()), ptr(kv_scale.data_ptr()),
    ptr(kv_indptr.data_ptr()), ptr(att_out.data_ptr()),
    f32(sm_scale),
    i32(stride_q_token), i32(stride_q_head), i32(stride_kv_token),
    i32(stride_mid_b), i32(stride_mid_h), i32(stride_mid_s),
    ptr(0), ptr(0),  # global_scratch + profile_scratch
])
```

### Dynamic Shared Memory

Gluon kernel uses `gl.allocate_shared_memory` to allocate dynamic shared memory.
The `group_segment_fixed_size: 0` in hsaco metadata indicates the use of dynamic shared memory.
**You must** retrieve the `shared` value from the compiled kernel metadata and pass it during launch.

```python
# Get from compiled kernel
shared_mem = compiled.metadata.shared  # For example 32768
# Pass to launch
_launch_kernel(func, grid, block, shared_mem, kernarg)
```

## ⚠️ Common Precision Pitfalls (Generalization Guide)

### 1. Improper Parameter Specialization Strategy (Most Common)

**Symptom**: Need to compile many hsaco variants, or different inputs produce different precision
**Root Cause**: Triton by default specializes all unannotated scalars (bakes into hsaco), causing:
- Each parameter value change requires a new variant
- Compile-time baked values may not match runtime actual values

**Generic Decision Framework**:
```
Does the parameter affect kernel control flow / loop boundaries / memory layout?
├─ YES → Must be constexpr or specialized (baked into hsaco)
│         For example: BLOCK_SIZE, num_warps, stride (when stride determines memory access pattern)
└─ NO  → Use do_not_specialize (pass at runtime)
          For example: scale coefficients, bias, alpha/beta and other mathematical operation parameters
```

**Implementation**:
```python
# Performance-insensitive mathematical parameters → do_not_specialize
@gluon.jit(do_not_specialize=["sm_scale", "alpha", "bias"])
def my_kernel(..., sm_scale, alpha, bias, ...):
    ...

# Structural parameters affecting compilation → keep specialized or constexpr
@gluon.jit
def my_kernel(..., stride_m: tl.int32, stride_n: tl.int32, BLOCK_M: gl.constexpr, ...):
    ...
```

### 2. kernarg Scalar Packing Errors

**Symptom**: Some CTA outputs are 0, or output values are completely wrong, but the kernel does not crash
**Root Cause**: When manually constructing kernarg, scalars are packed as 8-byte pointers (`struct.pack_into('Q', ...)`),
while i32/f32 scalars should be 4 bytes (`struct.pack_into('i'/'f', ...)`), causing incorrect offsets for subsequent parameters

**Fix**: Use type markers to distinguish pointers from scalars:
```python
class _PtrArg: ...  # 8 bytes
class _F32Arg: ...  # 4 bytes
class _I32Arg: ...  # 4 bytes
```

### 3. Dynamic Shared Memory Not Passed

**Symptom**: Kernel launches successfully but outputs are all 0 or NaN
**Root Cause**: `shared_mem_bytes` passes 0, but the kernel uses `gl.allocate_shared_memory` which requires dynamic shared memory
**Fix**: Retrieve the correct shared memory size from `compiled.metadata.shared`, do not rely on the `group_segment_fixed_size` in hsaco metadata (which is 0 when dynamic)

### 4. Compiler Implicit Parameter Omission

**Symptom**: Precision does not fully match, with minor deviations in some values
**Root Cause**: The Gluon/Triton compiler appends implicit pointers (e.g., global_scratch, profile_scratch) at the end of kernel parameters.
These are invisible in the Python signature but are recorded in the hsaco metadata.
**Fix**: Use `llvm-readobj --notes kernel.hsaco` to check the actual number of parameters, and pad null pointers at the end of the kernarg.

### 5. HIP Runtime Loading Timing

**Symptom**: `hipModuleGetFunction error 500` (HIP_ERROR_NOT_FOUND)
**Root Cause**: Inconsistent behavior when loading the HIP library before or after importing PyTorch/aiter, resulting in symbol resolution conflicts.
**Fix**:
- Pattern A (C++ launcher): Not affected
- Pattern B (Pure Python): Use `ctypes.CDLL("libamdhip64.so.7")` to load after importing torch

### 6. GPU Idle Frequency Causing Unstable Performance

**Symptom**: Significant performance variance across multiple runs of the same kernel
**Root Cause**: The GPU is in an idle state (SCLK ~95MHz), requiring a warmup on the first run to increase clock frequency.
**Fix**: Run a few matmul operations before benchmarking to allow the GPU to reach higher clock frequencies.

## Notes

1. **shared memory**: If the kernel uses dynamic shared memory (`group_segment_fixed_size: 0`),
   the `shared` value must be obtained from the compiled metadata and passed during launch.
2. **kernel function name**: The compiled function name of a Gluon kernel is the same as the Python function name (e.g., `_my_kernel`),
   while Triton kernels may have name mangling; use `llvm-readobj --symbols` to inspect them.
3. **HIP stream**: The current implementation uses `nullptr` (default stream). If synchronization with PyTorch is required,
   use `torch.cuda.current_stream().cuda_stream` instead.
4. **HIP_LAUNCH_PARAM_BUFFER**: Always pass kernarg using the extra parameter approach to avoid compatibility issues
   with `kernelParams` on certain ROCm versions.
5. **Pattern Selection**:
   - For production, use Pattern A (C++ launcher), as ctypes may have ABI compatibility issues in edge cases.
   - For quick testing, use Pattern B (Pure Python), which is self-contained in a single file and simplest to deploy.
