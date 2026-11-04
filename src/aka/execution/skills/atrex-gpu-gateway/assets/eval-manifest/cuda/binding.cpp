#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

void launch_add_one(const float* x, float* out, int64_t n, cudaStream_t stream);

torch::Tensor add_one(torch::Tensor x) {
    const c10::cuda::CUDAGuard guard(x.device());
    auto input = x.contiguous();
    auto out = torch::empty_like(input);
    int64_t n = input.numel();
    if (n > 0) {
        launch_add_one(input.data_ptr<float>(), out.data_ptr<float>(), n,
                       at::cuda::getCurrentCUDAStream());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_one", &add_one);
}
