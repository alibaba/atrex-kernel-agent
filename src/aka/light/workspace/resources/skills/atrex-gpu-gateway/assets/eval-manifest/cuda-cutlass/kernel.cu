#include <cutlass/functional.h>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void add_one_kernel(const float* x, float* out, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = cutlass::plus<float>()(x[i], 1.0f);
}

extern "C" void launch_add_one(const void* x, void* out, int64_t n, cudaStream_t stream) {
    if (n > 0) {
        add_one_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
            static_cast<const float*>(x), static_cast<float*>(out), n);
    }
}
