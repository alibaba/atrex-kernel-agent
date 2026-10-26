"""CUDA/NVRTC starting point: replace the example math and Model contract."""
import torch
from cuda.bindings import driver, nvrtc

_CUDA_SOURCE = r'''
extern "C" __global__ void affine(const float* x, float* out, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * scale + 1.0f;
}
'''
_MODULES = {}


def _checked(result):
    if int(result[0]) != 0:
        raise RuntimeError(str(result[0]))
    return result[1:]


def _function(device):
    cached = _MODULES.get(device.index)
    if cached is not None:
        return cached[1]
    major, minor = torch.cuda.get_device_capability(device)
    program, = _checked(nvrtc.nvrtcCreateProgram(
        _CUDA_SOURCE.encode(), b"candidate.cu", 0, [], []))
    try:
        options = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"--std=c++17"]
        status, = nvrtc.nvrtcCompileProgram(program, len(options), options)
        if int(status) != 0:
            size, = _checked(nvrtc.nvrtcGetProgramLogSize(program))
            log = bytearray(size)
            _checked(nvrtc.nvrtcGetProgramLog(program, log))
            raise RuntimeError(log.decode(errors="replace"))
        size, = _checked(nvrtc.nvrtcGetCUBINSize(program))
        image = bytearray(size)
        _checked(nvrtc.nvrtcGetCUBIN(program, image))
    finally:
        _checked(nvrtc.nvrtcDestroyProgram(program))
    module, = _checked(driver.cuModuleLoadData(image))
    function, = _checked(driver.cuModuleGetFunction(module, b"affine"))
    _MODULES[device.index] = (module, function)
    return function


class Model(torch.nn.Module):
    def __init__(self, scale=2.0):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        # This example requires contiguous CUDA float32 input; adapt to the operator.
        out = torch.empty_like(x)
        n = x.numel()
        if n == 0:
            return out
        with torch.cuda.device(x.device):
            function = _function(x.device)
            # CPU buffers hold C argument values; kernelParams holds their addresses.
            pointers = torch.tensor([x.data_ptr(), out.data_ptr()], dtype=torch.int64, device="cpu")
            count = torch.tensor([n], dtype=torch.int32, device="cpu")
            scale = torch.tensor([self.scale], dtype=torch.float32, device="cpu")
            params = torch.tensor([pointers.data_ptr(), pointers.data_ptr() + 8,
                                   count.data_ptr(), scale.data_ptr()], dtype=torch.int64, device="cpu")
            _checked(driver.cuLaunchKernel(
                function, (n + 255) // 256, 1, 1, 256, 1, 1, 0,
                torch.cuda.current_stream(x.device).cuda_stream, params.data_ptr(), 0))
        return out
