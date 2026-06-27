// gpu-wiki archive note:
// SM120 RMSNorm + NVFP4 input-quant source snapshot from the vLLM/proj4
// experiments. Contains the fused add+rmsnorm+fp4 quant path plus PDL and
// row-ready producer variants used to study RMSNorm -> MLP C1 handoff. The
// associated wiki conclusion is correctness-safe but not default-promoted.
//
/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <limits>

#include <torch/csrc/stable/tensor.h>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "libtorch_stable/torch_utils.h"
#include "libtorch_stable/dispatch_utils.h"
#include "cub_helpers.h"
#include "cuda_vec_utils.cuh"

#include "cuda_utils.h"
#include "launch_bounds_utils.h"

// Define before including nvfp4_utils.cuh so the header
// can use this macro during compilation.
#define NVFP4_ENABLE_ELTS16 1
#include "nvfp4_utils.cuh"

namespace vllm {

template <typename Type>
__device__ __forceinline__ float scalar_to_float(Type value);

template <>
__device__ __forceinline__ float scalar_to_float<half>(half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float scalar_to_float<__nv_bfloat16>(
    __nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename Type>
__device__ __forceinline__ Type scalar_from_float(float value);

template <>
__device__ __forceinline__ half scalar_from_float<half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 scalar_from_float<__nv_bfloat16>(
    float value) {
  return __float2bfloat16(value);
}

template <typename Type>
__device__ __forceinline__ typename PackedTypeConverter<Type>::Type
packed_pair_from_float(float x, float y);

template <>
__device__ __forceinline__ half2 packed_pair_from_float<half>(float x,
                                                             float y) {
  return __floats2half2_rn(x, y);
}

template <>
__device__ __forceinline__ __nv_bfloat162
packed_pair_from_float<__nv_bfloat16>(float x, float y) {
  return __float22bfloat162_rn(make_float2(x, y));
}

template <typename Type, int width>
struct alignas(16) RmsNormVec {
  static_assert(width > 0 && (width & (width - 1)) == 0);
  using Packed = typename PackedTypeConverter<Type>::Type;
  Type data[width];

  __device__ __forceinline__ void add_(RmsNormVec<Type, width> const& other) {
#pragma unroll
    for (int i = 0; i < width; i += 2) {
      Packed lhs{data[i], data[i + 1]};
      Packed rhs{other.data[i], other.data[i + 1]};
      lhs += rhs;
      data[i] = lhs.x;
      data[i + 1] = lhs.y;
    }
  }

  __device__ __forceinline__ float sum_squares() const {
    float result = 0.0f;
#pragma unroll
    for (int i = 0; i < width; i += 2) {
      float2 z = cast_to_float2(Packed{data[i], data[i + 1]});
      result += z.x * z.x + z.y * z.y;
    }
    return result;
  }
};

template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    fused_add_rms_norm_cvt_fp16_to_fp4(
        int32_t numRows, int32_t numCols, int32_t num_padded_cols,
        Type const* __restrict__ input, int64_t input_stride,
        Type* __restrict__ residual, Type const* __restrict__ weight,
        float epsilon, bool rms_weight_offset,
        float const* __restrict__ SFScale, uint32_t* __restrict__ out,
        uint32_t* __restrict__ SFout, bool pdl_trigger) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Type* row_smem = reinterpret_cast<Type*>(smem_raw);
  using PackedVec = vllm::PackedVec<Type, CVT_FP4_PACK16>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  __shared__ float s_variance;
  int32_t const rowIdx = blockIdx.x;
  float variance = 0.0f;

  if (rowIdx < numRows) {
    static constexpr int RMS_VEC_WIDTH = 8;
    int32_t const vec_num_cols = numCols / RMS_VEC_WIDTH;
    int64_t const input_vec_stride = input_stride / RMS_VEC_WIDTH;
    using Vec = RmsNormVec<Type, RMS_VEC_WIDTH>;
    auto input_v = reinterpret_cast<Vec const*>(input);
    auto residual_v = reinterpret_cast<Vec*>(residual);
    auto row_smem_v = reinterpret_cast<Vec*>(row_smem);
    for (int32_t idx = threadIdx.x; idx < vec_num_cols; idx += blockDim.x) {
      int64_t const input_offset =
          static_cast<int64_t>(rowIdx) * input_vec_stride + idx;
      int64_t const residual_offset =
          static_cast<int64_t>(rowIdx) * vec_num_cols + idx;
      Vec z = input_v[input_offset];
      z.add_(residual_v[residual_offset]);
      residual_v[residual_offset] = z;
      row_smem_v[idx] = z;
      variance += z.sum_squares();
    }
  }

  using BlockReduce = cub::BlockReduce<float, 512>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance =
        rowIdx < numRows ? rsqrtf(variance / numCols + epsilon) : 0.0f;
  }
  __syncthreads();

  float const global_scale = (SFScale == nullptr) ? 1.0f : SFScale[0];
  int32_t const numKTiles = (numCols + 63) / 64;
  for (int32_t colIdx = threadIdx.x; colIdx < num_padded_cols;
       colIdx += blockDim.x) {
    int32_t const elem_idx = colIdx * CVT_FP4_ELTS_PER_THREAD;
    PackedVec norm_vec;

#pragma unroll
    for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
      int32_t const elem0 = elem_idx + i * 2;
      int32_t const elem1 = elem0 + 1;
      float x0 = 0.0f;
      float x1 = 0.0f;
      if (rowIdx < numRows && elem1 < numCols) {
        float const weight_offset = rms_weight_offset ? 1.0f : 0.0f;
        x0 = scalar_to_float(row_smem[elem0]) * s_variance *
             (scalar_to_float(weight[elem0]) + weight_offset);
        x1 = scalar_to_float(row_smem[elem1]) * s_variance *
             (scalar_to_float(weight[elem1]) + weight_offset);
      }
      using PackedScalar = typename PackedTypeConverter<Type>::Type;
      norm_vec.elts[i] =
          PackedScalar{scalar_from_float<Type>(x0), scalar_from_float<Type>(x1)};
    }

    auto sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                           CVT_FP4_NUM_THREADS_PER_SF>(
            rowIdx, colIdx, numKTiles, SFout);
    auto out_val =
        cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
            norm_vec, global_scale, sf_out);

    bool valid = (rowIdx < numRows) && (elem_idx < numCols);
    if (valid) {
      if constexpr (CVT_FP4_PACK16) {
        int64_t outOffset = static_cast<int64_t>(rowIdx) * (numCols / 8) +
                            static_cast<int64_t>(colIdx) * 2;
        uint64_t packed64 =
            (uint64_t(out_val.hi) << 32) | uint64_t(out_val.lo);
        reinterpret_cast<uint64_t*>(out)[outOffset >> 1] = packed64;
      } else {
        int64_t outOffset =
            static_cast<int64_t>(rowIdx) *
                (numCols / CVT_FP4_ELTS_PER_THREAD) +
            colIdx;
        out[outOffset] = out_val;
      }
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (pdl_trigger) {
    __syncthreads();
    __threadfence();
    cudaTriggerProgrammaticLaunchCompletion();
  }
	#endif
	}

	template <class Type, bool UE8M0_SF = false>
	__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
	    fused_add_rms_norm_cvt_fp16_to_fp4_row_ready_persistent(
	        int32_t numRows, int32_t numCols, int32_t num_padded_cols,
	        Type const* __restrict__ input, int64_t input_stride,
	        Type* __restrict__ residual, Type const* __restrict__ weight,
	        float epsilon, bool rms_weight_offset,
	        float const* __restrict__ SFScale, uint32_t* __restrict__ out,
	        uint32_t* __restrict__ SFout, uint32_t* __restrict__ chunk_counts,
	        uint32_t* __restrict__ chunk_ready, int32_t chunk_rows,
	        int32_t pdl_trigger_chunk, bool publish_ready_flag) {
	  extern __shared__ __align__(16) unsigned char smem_raw[];
	  Type* row_smem = reinterpret_cast<Type*>(smem_raw);
	  using PackedVec = vllm::PackedVec<Type, CVT_FP4_PACK16>;
	  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
	      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
	  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
	                "Vec size is not matched.");

	#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
	  if (pdl_trigger_chunk < 0) {
	    __syncthreads();
	    if (threadIdx.x == 0) {
	      cudaTriggerProgrammaticLaunchCompletion();
	    }
	    __syncthreads();
	  }
	#endif

	  __shared__ float s_variance;
	  using BlockReduce = cub::BlockReduce<float, 512>;
	  __shared__ typename BlockReduce::TempStorage reduceStore;

	  for (int32_t rowIdx = blockIdx.x; rowIdx < numRows;
	       rowIdx += gridDim.x) {
	    float variance = 0.0f;
	    static constexpr int RMS_VEC_WIDTH = 8;
	    int32_t const vec_num_cols = numCols / RMS_VEC_WIDTH;
	    int64_t const input_vec_stride = input_stride / RMS_VEC_WIDTH;
	    using Vec = RmsNormVec<Type, RMS_VEC_WIDTH>;
	    auto input_v = reinterpret_cast<Vec const*>(input);
	    auto residual_v = reinterpret_cast<Vec*>(residual);
	    auto row_smem_v = reinterpret_cast<Vec*>(row_smem);
	    for (int32_t idx = threadIdx.x; idx < vec_num_cols; idx += blockDim.x) {
	      int64_t const input_offset =
	          static_cast<int64_t>(rowIdx) * input_vec_stride + idx;
	      int64_t const residual_offset =
	          static_cast<int64_t>(rowIdx) * vec_num_cols + idx;
	      Vec z = input_v[input_offset];
	      z.add_(residual_v[residual_offset]);
	      residual_v[residual_offset] = z;
	      row_smem_v[idx] = z;
	      variance += z.sum_squares();
	    }

	    variance =
	        BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

	    if (threadIdx.x == 0) {
	      s_variance = rsqrtf(variance / numCols + epsilon);
	    }
	    __syncthreads();

	    float const global_scale = (SFScale == nullptr) ? 1.0f : SFScale[0];
	    int32_t const numKTiles = (numCols + 63) / 64;
	    for (int32_t colIdx = threadIdx.x; colIdx < num_padded_cols;
	         colIdx += blockDim.x) {
	      int32_t const elem_idx = colIdx * CVT_FP4_ELTS_PER_THREAD;
	      PackedVec norm_vec;

	#pragma unroll
	      for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
	        int32_t const elem0 = elem_idx + i * 2;
	        int32_t const elem1 = elem0 + 1;
	        float x0 = 0.0f;
	        float x1 = 0.0f;
	        if (elem1 < numCols) {
	          float const weight_offset = rms_weight_offset ? 1.0f : 0.0f;
	          x0 = scalar_to_float(row_smem[elem0]) * s_variance *
	               (scalar_to_float(weight[elem0]) + weight_offset);
	          x1 = scalar_to_float(row_smem[elem1]) * s_variance *
	               (scalar_to_float(weight[elem1]) + weight_offset);
	        }
	        using PackedScalar = typename PackedTypeConverter<Type>::Type;
	        norm_vec.elts[i] = PackedScalar{scalar_from_float<Type>(x0),
	                                        scalar_from_float<Type>(x1)};
	      }

	      auto sf_out =
	          cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
	                                             CVT_FP4_NUM_THREADS_PER_SF>(
	              rowIdx, colIdx, numKTiles, SFout);
	      auto out_val =
	          cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
	              norm_vec, global_scale, sf_out);

	      if (elem_idx < numCols) {
	        if constexpr (CVT_FP4_PACK16) {
	          int64_t outOffset = static_cast<int64_t>(rowIdx) * (numCols / 8) +
	                              static_cast<int64_t>(colIdx) * 2;
	          uint64_t packed64 =
	              (uint64_t(out_val.hi) << 32) | uint64_t(out_val.lo);
	          reinterpret_cast<uint64_t*>(out)[outOffset >> 1] = packed64;
	        } else {
	          int64_t outOffset =
	              static_cast<int64_t>(rowIdx) *
	                  (numCols / CVT_FP4_ELTS_PER_THREAD) +
	              colIdx;
	          out[outOffset] = out_val;
	        }
	      }
	    }

	    __syncthreads();
	    if (threadIdx.x == 0) {
	      int32_t const chunk = rowIdx / chunk_rows;
	      int32_t const chunk_start = chunk * chunk_rows;
	      int32_t const rows_in_chunk =
	          min(chunk_rows, numRows - chunk_start);
	      uint32_t old = 0;
	#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
	      asm volatile("atom.global.release.gpu.add.u32 %0, [%1], %2;"
	                   : "=r"(old)
	                   : "l"(chunk_counts + chunk), "r"(1u)
	                   : "memory");
	#else
	      __threadfence();
	      old = atomicAdd(chunk_counts + chunk, 1u);
	#endif
	      bool const chunk_complete =
	          old + 1u == static_cast<uint32_t>(rows_in_chunk);
	      if (chunk_complete) {
	        if (publish_ready_flag) {
	          uint32_t const one = 1u;
	          asm volatile("st.global.release.gpu.u32 [%0], %1;"
	                       :
	                       : "l"(chunk_ready + chunk), "r"(one)
	                       : "memory");
	        }
	#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
	        if (chunk == pdl_trigger_chunk) {
	          cudaTriggerProgrammaticLaunchCompletion();
	        }
	#endif
	      }
	    }
	    __syncthreads();
	  }
	}

	// Use UE4M3 by default.
	template <class Type, bool UE8M0_SF = false>
	__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    cvt_fp16_to_fp4(int32_t numRows, int32_t numCols, int32_t num_padded_cols,
                    Type const* __restrict__ in,
                    float const* __restrict__ SFScale,
                    uint32_t* __restrict__ out, uint32_t* __restrict__ SFout) {
  using PackedVec = vllm::PackedVec<Type, CVT_FP4_PACK16>;

  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  // Precompute SF layout parameter (constant for entire kernel).
  int32_t const numKTiles = (numCols + 63) / 64;

  int sf_m = round_up<int>(numRows, 128);
  int32_t const colIdx = blockDim.x * blockIdx.y + threadIdx.x;
  int elem_idx = colIdx * CVT_FP4_ELTS_PER_THREAD;

  // Get the global scaling factor, which will be applied to the SF.
  // Note SFScale is the same as next GEMM's alpha, which is
  // (448.f / (Alpha_A / 6.f)).
  float const global_scale = (SFScale == nullptr) ? 1.0f : SFScale[0];

  // Iterate over all rows and cols including padded ones -
  //  ensures we visit every single scale factor address to initialize it.
  for (int rowIdx = blockIdx.x; rowIdx < sf_m; rowIdx += gridDim.x) {
    if (colIdx < num_padded_cols) {
      PackedVec in_vec;
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;

      // If we are outside valid rows OR outside valid columns -> Use Zeros
      bool valid = (rowIdx < numRows) && (elem_idx < numCols);
      if constexpr (CVT_FP4_PACK16) {
        ld256_cg_or_zero(reinterpret_cast<u32x8_t&>(in_vec),
                         &reinterpret_cast<const uint32_t*>(in)[inOffset * 8],
                         valid);
      } else {
        ld128_cg_or_zero(reinterpret_cast<uint4&>(in_vec),
                         &reinterpret_cast<const uint32_t*>(in)[inOffset * 4],
                         valid);
      }

      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                             CVT_FP4_NUM_THREADS_PER_SF>(
              rowIdx, colIdx, numKTiles, SFout);

      auto out_val =
          cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
              in_vec, global_scale, sf_out);

      // We do NOT write output for padding because the 'out' tensor is not
      // padded.
      if (valid) {
        if constexpr (CVT_FP4_PACK16) {
          int64_t outOffset = rowIdx * (numCols / 8) + colIdx * 2;
          uint64_t packed64 =
              (uint64_t(out_val.hi) << 32) | uint64_t(out_val.lo);
          reinterpret_cast<uint64_t*>(out)[outOffset >> 1] = packed64;
        } else {
          out[inOffset] = out_val;
        }
      }
    }
  }
}

// Use UE4M3 by default.
template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    cvt_fp16_to_fp4_sf_major(int32_t numRows, int32_t numCols,
                             int32_t sf_n_unpadded, int32_t num_packed_cols,
                             Type const* __restrict__ in,
                             float const* __restrict__ SFScale,
                             uint32_t* __restrict__ out,
                             uint32_t* __restrict__ SFout) {
  using PackedVec = PackedVec<Type, CVT_FP4_PACK16>;

  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  int32_t const colIdx = blockDim.x * blockIdx.y + threadIdx.x;
  int elem_idx = colIdx * CVT_FP4_ELTS_PER_THREAD;

  // Get the global scaling factor, which will be applied to the SF.
  // Note SFScale is the same as next GEMM's alpha, which is
  // (448.f / (Alpha_A / 6.f)).
  float const global_scale = (SFScale == nullptr) ? 1.0f : SFScale[0];

  // Iterate over all rows and cols including padded ones -
  //  ensures we visit every single scale factor address to initialize it.
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    if (colIdx < num_packed_cols) {
      PackedVec in_vec;
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;

      // If we are outside valid rows OR outside valid columns -> Use Zeros
      bool valid = (rowIdx < numRows) && (elem_idx < numCols);
      if constexpr (CVT_FP4_PACK16) {
        ld256_cg_or_zero(reinterpret_cast<u32x8_t&>(in_vec),
                         &reinterpret_cast<const uint32_t*>(in)[inOffset * 8],
                         valid);
      } else {
        ld128_cg_or_zero(reinterpret_cast<uint4&>(in_vec),
                         &reinterpret_cast<const uint32_t*>(in)[inOffset * 4],
                         valid);
      }

      auto sf_out =
          sf_out_rowmajor_u8<uint32_t>(rowIdx, colIdx, sf_n_unpadded, SFout);

      auto out_val =
          cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
              in_vec, global_scale, sf_out);

      // We do NOT write output for padding because the 'out' tensor is not
      // padded.
      if (valid) {
        if constexpr (CVT_FP4_PACK16) {
          int64_t outOffset = rowIdx * (numCols / 8) + colIdx * 2;
          uint64_t packed64 =
              (uint64_t(out_val.hi) << 32) | uint64_t(out_val.lo);
          reinterpret_cast<uint64_t*>(out)[outOffset >> 1] = packed64;
        } else {
          out[inOffset] = out_val;
        }
      }
    }
  }
}

}  // namespace vllm

void scaled_fp4_quant_sm1xxa(torch::stable::Tensor const& output,
                             torch::stable::Tensor const& input,
                             torch::stable::Tensor const& output_sf,
                             torch::stable::Tensor const& input_sf,
                             bool is_sf_swizzled_layout) {
  int32_t m = input.size(0);
  int32_t n = input.size(1);

  STD_TORCH_CHECK(n % 16 == 0, "The N dimension must be multiple of 16.");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Half ||
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "Unsupported input data type for quantize_to_fp4.");

  int multiProcessorCount =
      get_device_attribute(cudaDevAttrMultiProcessorCount, -1);

  auto input_sf_ptr = static_cast<float const*>(input_sf.data_ptr());
  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  auto stream = get_current_cuda_stream(input.get_device_index());

  int sf_n_unpadded = int(n / CVT_FP4_SF_VEC_SIZE);

  // Grid, Block size. Each thread converts 8 values.
  dim3 block(std::min(int(n / ELTS_PER_THREAD), 512));
  int const numBlocksPerSM =
      vllm_runtime_blocks_per_sm(static_cast<int>(block.x));

  if (is_sf_swizzled_layout) {
    int sf_n_int = int(vllm::round_up(sf_n_unpadded, 4) / 4);
    int32_t num_padded_cols =
        sf_n_int * 4 * CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;

    int grid_y = vllm::div_round_up(num_padded_cols, static_cast<int>(block.x));
    int grid_x =
        std::min(vllm::computeEffectiveRows(m),
                 std::max(1, (multiProcessorCount * numBlocksPerSM) / grid_y));
    dim3 grid(grid_x, grid_y);

    VLLM_STABLE_DISPATCH_HALF_TYPES(
        input.scalar_type(), "nvfp4_quant_kernel", [&] {
          using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
          auto input_ptr = static_cast<cuda_type const*>(input.data_ptr());
          vllm::cvt_fp16_to_fp4<cuda_type, false><<<grid, block, 0, stream>>>(
              m, n, num_padded_cols, input_ptr, input_sf_ptr,
              reinterpret_cast<uint32_t*>(output_ptr),
              reinterpret_cast<uint32_t*>(sf_out));
        });
  } else {
    int num_packed_cols = n / CVT_FP4_ELTS_PER_THREAD;
    int grid_y = vllm::div_round_up(num_packed_cols, static_cast<int>(block.x));
    int grid_x = std::min(
        m, std::max(1, (multiProcessorCount * numBlocksPerSM) / grid_y));
    dim3 grid(grid_x, grid_y);

    VLLM_STABLE_DISPATCH_HALF_TYPES(
        input.scalar_type(), "nvfp4_quant_kernel", [&] {
          using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
          auto input_ptr = static_cast<cuda_type const*>(input.data_ptr());
          vllm::cvt_fp16_to_fp4_sf_major<cuda_type, false>
              <<<grid, block, 0, stream>>>(
                  m, n, sf_n_unpadded, num_packed_cols, input_ptr, input_sf_ptr,
                  reinterpret_cast<uint32_t*>(output_ptr),
                  reinterpret_cast<uint32_t*>(sf_out));
        });
  }
}

static void fused_add_rms_norm_scaled_fp4_quant_sm1xxa_launch(
    torch::stable::Tensor const& output,
    torch::stable::Tensor const& output_sf,
    torch::stable::Tensor const& input,
    torch::stable::Tensor& residual,
    torch::stable::Tensor const& weight,
    torch::stable::Tensor const& input_sf,
    double epsilon,
    bool rms_weight_offset,
    bool is_sf_swizzled_layout,
    int64_t row_offset,
    int64_t rows,
    cudaStream_t stream,
    bool pdl_trigger) {
  STD_TORCH_CHECK(is_sf_swizzled_layout,
                  "task23 fused RMSNorm+FP4 quant currently supports only "
                  "swizzled scale layout");
  STD_TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor");
  STD_TORCH_CHECK(residual.dim() == 2, "residual must be a 2D tensor");
  STD_TORCH_CHECK(weight.dim() == 1, "weight must be a 1D tensor");

  int32_t full_m = input.size(0);
  int32_t n = input.size(1);
  STD_TORCH_CHECK(row_offset >= 0, "row_offset must be non-negative");
  STD_TORCH_CHECK(rows >= 0, "rows must be non-negative");
  STD_TORCH_CHECK(row_offset + rows <= full_m,
                  "row chunk exceeds input rows: row_offset=", row_offset,
                  ", rows=", rows, ", input rows=", full_m);
  int32_t m = static_cast<int32_t>(rows);
  STD_TORCH_CHECK(input.stride(1) == 1,
                  "input last dimension must be contiguous");
  STD_TORCH_CHECK(input.stride(0) % 8 == 0,
                  "input row stride must be a multiple of 8");
  STD_TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  STD_TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  STD_TORCH_CHECK(residual.size(0) == full_m && residual.size(1) == n,
                  "residual shape must match input");
  STD_TORCH_CHECK(weight.size(0) == n,
                  "weight hidden size must match input hidden size");
  STD_TORCH_CHECK(n % 16 == 0, "The N dimension must be multiple of 16.");
  STD_TORCH_CHECK(output.size(0) == m && output.size(1) == n / 2,
                  "output must have shape (M, N / 2)");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Half ||
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "Unsupported input data type for fused RMSNorm+FP4 quant.");
  STD_TORCH_CHECK(input.scalar_type() == residual.scalar_type(),
                  "input and residual dtypes must match");
  STD_TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
                  "input and weight dtypes must match");

  int sf_n_unpadded = int(n / CVT_FP4_SF_VEC_SIZE);
  int sf_n_int = int(vllm::round_up(sf_n_unpadded, 4) / 4);
  int sf_m = vllm::round_up<int>(m, 128);
  int32_t num_padded_cols =
      sf_n_int * 4 * CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;
  bool const output_sf_is_int32 =
      output_sf.scalar_type() == torch::headeronly::ScalarType::Int;
  bool const output_sf_is_fp8 =
      output_sf.scalar_type() ==
      torch::headeronly::ScalarType::Float8_e4m3fn;
  STD_TORCH_CHECK(output_sf_is_int32 || output_sf_is_fp8,
                  "output_sf must be int32 packed scales or fp8 view scales");
  int64_t const expected_sf_cols = output_sf_is_fp8 ? sf_n_int * 4 : sf_n_int;
  STD_TORCH_CHECK(
      output_sf.size(0) == sf_m && output_sf.size(1) == expected_sf_cols,
      "output_sf must be swizzled shape (", sf_m, "x", expected_sf_cols,
      "), got (", output_sf.size(0), "x", output_sf.size(1), ")");

  auto input_sf_ptr = static_cast<float const*>(input_sf.data_ptr());
  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  int64_t input_stride = input.stride(0);

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());

  dim3 block((m < 256) ? 512 : 256);
  dim3 grid(sf_m);
  VLLM_STABLE_DISPATCH_HALF_TYPES(
      input.scalar_type(), "fused_add_rms_norm_nvfp4_quant_kernel", [&] {
        using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
        auto input_ptr =
            static_cast<cuda_type const*>(input.data_ptr()) +
            row_offset * input_stride;
        auto residual_ptr =
            static_cast<cuda_type*>(residual.data_ptr()) +
            row_offset * n;
        auto weight_ptr = static_cast<cuda_type const*>(weight.data_ptr());
        vllm::fused_add_rms_norm_cvt_fp16_to_fp4<cuda_type, false>
            <<<grid, block, static_cast<size_t>(n) * sizeof(cuda_type),
               stream>>>(
                m, n, num_padded_cols, input_ptr, input_stride, residual_ptr,
                weight_ptr, static_cast<float>(epsilon), rms_weight_offset,
                input_sf_ptr,
                reinterpret_cast<uint32_t*>(output_ptr),
                reinterpret_cast<uint32_t*>(sf_out), pdl_trigger);
      });
}

void fused_add_rms_norm_scaled_fp4_quant_sm1xxa(
    torch::stable::Tensor const& output,
    torch::stable::Tensor const& output_sf,
    torch::stable::Tensor const& input,
    torch::stable::Tensor& residual,
    torch::stable::Tensor const& weight,
    torch::stable::Tensor const& input_sf,
    double epsilon,
    bool rms_weight_offset,
    bool is_sf_swizzled_layout,
    bool pdl_trigger) {
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  auto stream = get_current_cuda_stream(input.get_device_index());
  fused_add_rms_norm_scaled_fp4_quant_sm1xxa_launch(
      output, output_sf, input, residual, weight, input_sf, epsilon,
      rms_weight_offset, is_sf_swizzled_layout, 0, input.size(0), stream,
      pdl_trigger);
}

void fused_add_rms_norm_scaled_fp4_quant_sm1xxa_chunk_stream(
    torch::stable::Tensor const& output,
    torch::stable::Tensor const& output_sf,
    torch::stable::Tensor const& input,
    torch::stable::Tensor& residual,
    torch::stable::Tensor const& weight,
    torch::stable::Tensor const& input_sf,
    double epsilon,
    bool rms_weight_offset,
    bool is_sf_swizzled_layout,
    int64_t row_offset,
    int64_t rows,
    cudaStream_t stream,
    bool pdl_trigger) {
  fused_add_rms_norm_scaled_fp4_quant_sm1xxa_launch(
      output, output_sf, input, residual, weight, input_sf, epsilon,
      rms_weight_offset, is_sf_swizzled_layout, row_offset, rows, stream,
      pdl_trigger);
}

void fused_add_rms_norm_scaled_fp4_quant_sm1xxa_row_ready_stream(
    torch::stable::Tensor const& output,
    torch::stable::Tensor const& output_sf,
    torch::stable::Tensor const& input,
    torch::stable::Tensor& residual,
    torch::stable::Tensor const& weight,
    torch::stable::Tensor const& input_sf,
    torch::stable::Tensor const& chunk_counts,
    torch::stable::Tensor const& chunk_ready,
    double epsilon,
	    bool rms_weight_offset,
	    bool is_sf_swizzled_layout,
	    int64_t chunk_rows,
	    cudaStream_t stream,
	    bool pdl_trigger_at_start,
	    bool publish_ready_flag) {
  STD_TORCH_CHECK(is_sf_swizzled_layout,
                  "row-ready RMSNorm+FP4 quant supports only swizzled scale "
                  "layout");
  STD_TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor");
  STD_TORCH_CHECK(residual.dim() == 2, "residual must be a 2D tensor");
  STD_TORCH_CHECK(weight.dim() == 1, "weight must be a 1D tensor");
  STD_TORCH_CHECK(chunk_counts.is_cuda() && chunk_ready.is_cuda(),
                  "chunk counters must be CUDA tensors");
  STD_TORCH_CHECK(chunk_counts.is_contiguous() && chunk_ready.is_contiguous(),
                  "chunk counters must be contiguous");
  STD_TORCH_CHECK(chunk_counts.scalar_type() ==
                      torch::headeronly::ScalarType::Int,
                  "chunk_counts must be int32");
  STD_TORCH_CHECK(chunk_ready.scalar_type() ==
                      torch::headeronly::ScalarType::Int,
                  "chunk_ready must be int32");
  STD_TORCH_CHECK(chunk_rows > 0, "chunk_rows must be positive");

  int32_t const m = input.size(0);
  int32_t const n = input.size(1);
  int64_t const chunks = (static_cast<int64_t>(m) + chunk_rows - 1) / chunk_rows;
  STD_TORCH_CHECK(chunk_counts.numel() >= chunks && chunk_ready.numel() >= chunks,
                  "chunk counter tensors are too small");
  STD_TORCH_CHECK(input.stride(1) == 1,
                  "input last dimension must be contiguous");
  STD_TORCH_CHECK(input.stride(0) % 8 == 0,
                  "input row stride must be a multiple of 8");
  STD_TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  STD_TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  STD_TORCH_CHECK(residual.size(0) == m && residual.size(1) == n,
                  "residual shape must match input");
  STD_TORCH_CHECK(weight.size(0) == n,
                  "weight hidden size must match input hidden size");
  STD_TORCH_CHECK(n % 16 == 0, "The N dimension must be multiple of 16.");
  STD_TORCH_CHECK(output.size(0) == m && output.size(1) == n / 2,
                  "output must have shape (M, N / 2)");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Half ||
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "Unsupported input data type for row-ready fused RMSNorm+FP4 quant.");
  STD_TORCH_CHECK(input.scalar_type() == residual.scalar_type(),
                  "input and residual dtypes must match");
  STD_TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
                  "input and weight dtypes must match");

  int sf_n_unpadded = int(n / CVT_FP4_SF_VEC_SIZE);
  int sf_n_int = int(vllm::round_up(sf_n_unpadded, 4) / 4);
  int sf_m = vllm::round_up<int>(m, 128);
  int32_t num_padded_cols =
      sf_n_int * 4 * CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;
  bool const output_sf_is_int32 =
      output_sf.scalar_type() == torch::headeronly::ScalarType::Int;
  bool const output_sf_is_fp8 =
      output_sf.scalar_type() ==
      torch::headeronly::ScalarType::Float8_e4m3fn;
  STD_TORCH_CHECK(output_sf_is_int32 || output_sf_is_fp8,
                  "output_sf must be int32 packed scales or fp8 view scales");
  int64_t const expected_sf_cols = output_sf_is_fp8 ? sf_n_int * 4 : sf_n_int;
  STD_TORCH_CHECK(
      output_sf.size(0) == sf_m && output_sf.size(1) == expected_sf_cols,
      "output_sf must be swizzled shape (", sf_m, "x", expected_sf_cols,
      "), got (", output_sf.size(0), "x", output_sf.size(1), ")");

  // Keep the producer as close as possible to the original row-parallel
  // RMSNorm+quant kernel. Throttling this grid serializes row production and
  // costs more than the C1 overlap can hide for prefill-sized M.
  int persistent_blocks = std::max(1, m);
  if (const char* env_blocks =
          std::getenv("VLLM_PROJ4_ROW_READY_QUANT_BLOCKS")) {
    int parsed = std::atoi(env_blocks);
    if (parsed > 0) {
      persistent_blocks = std::min(parsed, m);
    }
  }

  int32_t pdl_trigger_chunk =
      pdl_trigger_at_start ? int32_t(-1) : std::numeric_limits<int32_t>::max();
  if (const char* env_trigger =
          std::getenv("VLLM_PROJ4_ROW_READY_PDL_TRIGGER")) {
    if (std::strcmp(env_trigger, "start") == 0) {
      pdl_trigger_chunk = -1;
    } else if (std::strcmp(env_trigger, "chunk0") == 0 ||
               std::strcmp(env_trigger, "first_chunk") == 0 ||
               std::strcmp(env_trigger, "first") == 0) {
      pdl_trigger_chunk = 0;
    } else if (std::strcmp(env_trigger, "none") == 0) {
      pdl_trigger_chunk = std::numeric_limits<int32_t>::max();
    } else {
      int parsed = std::atoi(env_trigger);
      if (parsed >= 0) {
        pdl_trigger_chunk = parsed;
      }
    }
  }

  dim3 block((m < 256) ? 512 : 256);
  dim3 grid(persistent_blocks);
  auto input_sf_ptr = static_cast<float const*>(input_sf.data_ptr());
  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  int64_t input_stride = input.stride(0);

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      input.scalar_type(), "row_ready_fused_add_rms_norm_nvfp4_quant_kernel", [&] {
        using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
        auto input_ptr = static_cast<cuda_type const*>(input.data_ptr());
        auto residual_ptr = static_cast<cuda_type*>(residual.data_ptr());
        auto weight_ptr = static_cast<cuda_type const*>(weight.data_ptr());
        vllm::fused_add_rms_norm_cvt_fp16_to_fp4_row_ready_persistent<
            cuda_type, false>
            <<<grid, block, static_cast<size_t>(n) * sizeof(cuda_type),
               stream>>>(
                m, n, num_padded_cols, input_ptr, input_stride, residual_ptr,
                weight_ptr, static_cast<float>(epsilon), rms_weight_offset,
                input_sf_ptr, reinterpret_cast<uint32_t*>(output_ptr),
	                reinterpret_cast<uint32_t*>(sf_out),
	                static_cast<uint32_t*>(chunk_counts.data_ptr()),
	                static_cast<uint32_t*>(chunk_ready.data_ptr()),
	                static_cast<int32_t>(chunk_rows), pdl_trigger_chunk,
	                publish_ready_flag);
	      });
	}
