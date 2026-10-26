# CUDA starting point

Copy the plumbing from `skills/gpu-measurement/templates/cuda_nvrtc.py` for NVIDIA CUDA.
It is a complete **affine example**, not an implementation of your operator. Replace its
source literal, exported symbol, `Model` signature, allocations and arguments to implement
the public contract. Keep everything self-contained in `kernel.py`; do not import this file.

The template compiles its embedded source with `cuda.bindings.nvrtc`, loads the resulting
CUBIN through `cuda.bindings.driver`, and launches on the input device's current Torch stream.
It caches module/function handles per device, not tensor pointers or outputs. CPU tensors hold
pointer, int32 and float32 argument values, and a CPU int64 array holds their addresses for
`cuLaunchKernel`. Keep these buffers alive through the launch call and match the CUDA ABI
exactly; add int64/double storage explicitly when the kernel requires it.

Use the existing five-argument `nvrtcCreateProgram(source, name, 0, [], [])` and
`nvrtcCompileProgram(program, len(options), options)` calls. Architecture comes from the
runtime capability. Keep the single supported compilation path; do not guess overloads,
add compatibility ladders, or replace it with a filesystem/shared-library loader.
Compilation failures include the NVRTC log. The example needs no external headers or
dynamic shared memory; add either only when the operator requires it and verify on the worker.

Torch is used only for allocation, context/stream discovery and host argument packing.
There is no `ctypes`, NumPy, environment access, file I/O or prebuilt compute dependency.
Declare `torch` and `cuda-python` with these non-compute roles in `solution.json`.
The affine example assumes contiguous CUDA float32 input; implement all layouts and dtypes
allowed by your operator instead of silently narrowing its contract.

Validation: L20N, runtime `sm_120`, Torch 2.9.0+cu130, CUDA 13.0 and cuda.bindings 13.1.1,
through Agate's source gate and evaluator.
The unmodified example passed three correctness cases each for empty, singleton, tail-sized
and large inputs (0, 1, 513, 1048579 elements), including negative/fractional scale arguments.
Separate GPU checks passed non-default-stream execution, module reuse with new inputs,
CUDA Graph replay after input mutation, and preservation of NVRTC compile-error diagnostics.
This validates the compiler, loader and argument ABI, not a new operator's correctness.
Every adapted candidate still needs its prescribed smoke and independent full acceptance.
