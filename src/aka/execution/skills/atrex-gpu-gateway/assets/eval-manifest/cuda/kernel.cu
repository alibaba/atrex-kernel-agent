#include <cstdint>
#include <cuda_runtime.h>

__global__ void add_one_kernel(const float* x, float* out, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + 1.0f;
}

void launch_add_one(const float* x, float* out, int64_t n, cudaStream_t stream) {
    add_one_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, out, n);
}
