// gpu-wiki archive note:
// SM120 vLLM entry points for RMSNorm-MLP parent fusion, whole-A PDL, row-chunk
// pipeline, and row-ready C1 experiments. Archive as integration/reference
// source only; the measured routes are neutral or negative in served gates.
//
/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

#include <algorithm>
#include <optional>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <torch/csrc/stable/tensor.h>

#include "cutlass_extensions/common.hpp"
#include "libtorch_stable/torch_utils.h"
#include "nvfp4_utils.cuh"

#if defined ENABLE_NVFP4_SM120 && ENABLE_NVFP4_SM120
void cutlass_nvfp4_mlp_c1_act_quant_sm120a(
    torch::stable::Tensor& act_payload,
    torch::stable::Tensor& interleaved_act_payload,
    torch::stable::Tensor& act_sf,
    torch::stable::Tensor const& A, torch::stable::Tensor const& B,
    torch::stable::Tensor const& A_sf, torch::stable::Tensor const& B_sf,
    torch::stable::Tensor const& c1_alpha,
    torch::stable::Tensor const& c2_input_global_scale_inv,
    int64_t logical_act_cols, bool use_bfloat16_intermediate,
    bool launch_with_pdl);

void cutlass_nvfp4_mlp_c1_act_quant_sm120a_stream(
    torch::stable::Tensor& act_payload,
    torch::stable::Tensor& interleaved_act_payload,
    torch::stable::Tensor& act_sf,
    torch::stable::Tensor const& A, torch::stable::Tensor const& B,
    torch::stable::Tensor const& A_sf, torch::stable::Tensor const& B_sf,
    torch::stable::Tensor const& c1_alpha,
    torch::stable::Tensor const& c2_input_global_scale_inv,
    int64_t logical_act_cols, bool use_bfloat16_intermediate,
    cudaStream_t stream, bool launch_with_pdl);

void cutlass_nvfp4_mlp_c1_act_quant_interleaved_sm120a(
    torch::stable::Tensor& act_payload,
    torch::stable::Tensor& interleaved_act_payload,
    torch::stable::Tensor& act_sf,
    torch::stable::Tensor const& A, torch::stable::Tensor const& B,
    torch::stable::Tensor const& A_sf, torch::stable::Tensor const& B_sf,
    torch::stable::Tensor const& c1_alpha,
    torch::stable::Tensor const& c2_input_global_scale_inv,
    int64_t logical_act_cols, bool use_bfloat16_intermediate,
    bool launch_with_pdl);

void cutlass_nvfp4_mlp_c1_act_quant_direct_store_sm120a(
    torch::stable::Tensor& act_payload,
    torch::stable::Tensor& act_sf,
    torch::stable::Tensor const& A, torch::stable::Tensor const& B,
    torch::stable::Tensor const& A_sf, torch::stable::Tensor const& B_sf,
    torch::stable::Tensor const& c1_alpha,
    torch::stable::Tensor const& c2_input_global_scale_inv,
    int64_t logical_act_cols, bool use_bfloat16_intermediate,
    int32_t store_mode, bool launch_with_pdl);

void cutlass_nvfp4_mlp_c1_act_quant_paired_output_sm120a(
    torch::stable::Tensor& act_payload,
    torch::stable::Tensor& act_sf,
    torch::stable::Tensor const& A, torch::stable::Tensor const& B,
    torch::stable::Tensor const& A_sf, torch::stable::Tensor const& B_sf,
    torch::stable::Tensor const& c1_alpha,
    torch::stable::Tensor const& c2_input_global_scale_inv,
    int64_t logical_act_cols, bool use_bfloat16_intermediate,
    bool launch_with_pdl);

void compact_interleaved_c1_fp4_payload_sm120a(
    torch::stable::Tensor& act_payload,
    torch::stable::Tensor const& interleaved_payload,
    int64_t logical_act_cols);

void cutlass_scaled_fp4_mm_interleaved_a_sm120a(
    torch::stable::Tensor& D, torch::stable::Tensor const& A,
    torch::stable::Tensor const& B, torch::stable::Tensor const& A_sf,
    torch::stable::Tensor const& B_sf, torch::stable::Tensor const& alpha,
    int64_t logical_k);

void cutlass_scaled_fp4_mm_sm120a_stream(
    torch::stable::Tensor& D, torch::stable::Tensor const& A,
    torch::stable::Tensor const& B, torch::stable::Tensor const& A_sf,
    torch::stable::Tensor const& B_sf, torch::stable::Tensor const& alpha,
    cudaStream_t stream);
#endif

void scaled_fp4_quant_out(torch::stable::Tensor const& input,
                          torch::stable::Tensor const& input_sf,
                          bool is_sf_swizzled_layout,
                          torch::stable::Tensor& output,
                          torch::stable::Tensor& output_sf);

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
    bool pdl_trigger);

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
    bool pdl_trigger);

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
	    bool publish_ready_flag);

		void proj4_c1_row_ready_set(unsigned int* ready_flags, int chunk_rows,
	                            int problem_m, int wait_mode,
	                            cudaStream_t stream);
void proj4_c1_row_ready_clear(cudaStream_t stream);

void cutlass_scaled_fp4_mm(torch::stable::Tensor& D,
                           torch::stable::Tensor const& A,
                           torch::stable::Tensor const& B,
                           torch::stable::Tensor const& A_sf,
                           torch::stable::Tensor const& B_sf,
                           torch::stable::Tensor const& alpha);

bool cutlass_scaled_mm_supports_fp4(int64_t cuda_device_capability);

namespace {

void check_cuda_matrix(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  STD_TORCH_CHECK(tensor.dim() == 2, name, " must be a 2D tensor");
}

void check_cuda_scale(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  STD_TORCH_CHECK(tensor.dim() >= 1, name, " must have at least 1 dimension");
}

torch::stable::Tensor empty_cuda_tensor(
    std::initializer_list<int64_t> shape,
    torch::headeronly::ScalarType dtype,
    const torch::stable::Tensor& like) {
  return torch::stable::empty(shape, dtype, std::nullopt, like.device());
}

torch::stable::Tensor allocate_fp4_payload(int64_t m, int64_t n,
                                           const torch::stable::Tensor& like) {
  STD_TORCH_CHECK(n % 2 == 0, "FP4 payload columns must be even");
  return empty_cuda_tensor({m, n / 2}, torch::headeronly::ScalarType::Byte,
                           like);
}

torch::stable::Tensor allocate_swizzled_fp4_scale(
    int64_t m, int64_t n, const torch::stable::Tensor& like) {
  auto [sf_m, sf_n_int32] = vllm::computeSwizzledSFShape(m, n);
  return empty_cuda_tensor({sf_m, sf_n_int32 * 4},
                           torch::headeronly::ScalarType::Float8_e4m3fn,
                           like);
}

void proj4_cuda_check(cudaError_t status, const char* expr) {
  STD_TORCH_CHECK(status == cudaSuccess, "CUDA call failed: ", expr, ": ",
                  cudaGetErrorString(status));
}

#define PROJ4_CUDA_CHECK(expr) proj4_cuda_check((expr), #expr)

	int64_t dtype_element_size_bytes(torch::headeronly::ScalarType dtype) {
	  if (dtype == torch::headeronly::ScalarType::Half ||
	      dtype == torch::headeronly::ScalarType::BFloat16) {
	    return 2;
	  }
	  STD_TORCH_CHECK(false, "unsupported dtype for row-chunk output: ", dtype);
	  return 0;
	}

	bool row_chunk_use_single_c2() {
	  const char* mode = std::getenv("VLLM_PROJ4_RMSNORM_MLP_ROW_CHUNK_MODE");
	  return mode != nullptr &&
	         (std::strcmp(mode, "single_c2") == 0 ||
	          std::strcmp(mode, "full_c2") == 0);
	}

		bool row_chunk_use_ready_wait() {
		  const char* mode = std::getenv("VLLM_PROJ4_RMSNORM_MLP_ROW_CHUNK_MODE");
		  return mode != nullptr &&
		         (std::strcmp(mode, "ready_wait") == 0 ||
		          std::strcmp(mode, "pdl_ready_wait") == 0);
		}

		int row_ready_wait_mode() {
		  const char* mode = std::getenv("VLLM_PROJ4_ROW_READY_WAIT_MODE");
		  if (mode != nullptr &&
		      (std::strcmp(mode, "count") == 0 ||
		       std::strcmp(mode, "counts") == 0 ||
		       std::strcmp(mode, "chunk_count") == 0)) {
		    return 1;
		  }
		  if (mode != nullptr &&
		      (std::strcmp(mode, "ready_nocache") == 0 ||
		       std::strcmp(mode, "flag_nocache") == 0)) {
		    return 2;
		  }
		  if (mode != nullptr &&
		      (std::strcmp(mode, "count_nocache") == 0 ||
		       std::strcmp(mode, "counts_nocache") == 0 ||
		       std::strcmp(mode, "chunk_count_nocache") == 0)) {
		    return 3;
		  }
		  if (mode != nullptr &&
		      (std::strcmp(mode, "count_prethrottle") == 0 ||
		       std::strcmp(mode, "counts_prethrottle") == 0 ||
		       std::strcmp(mode, "chunk_count_prethrottle") == 0)) {
		    return 4;
		  }
		  if (mode != nullptr &&
		      (std::strcmp(mode, "count_prethrottle_nocache") == 0 ||
		       std::strcmp(mode, "counts_prethrottle_nocache") == 0 ||
		       std::strcmp(mode, "chunk_count_prethrottle_nocache") == 0)) {
		    return 5;
		  }
		  return 0;
		}

		}  // namespace

torch::stable::Tensor cutlass_nvfp4_mlp_parent_fused(
    const torch::stable::Tensor& input,
    const torch::stable::Tensor& gate_up_weight,
    const torch::stable::Tensor& gate_up_weight_scale,
    const torch::stable::Tensor& gate_up_alpha,
    const torch::stable::Tensor& gate_up_input_global_scale_inv,
    const torch::stable::Tensor& down_weight,
    const torch::stable::Tensor& down_weight_scale,
    const torch::stable::Tensor& down_alpha,
    const torch::stable::Tensor& down_input_global_scale_inv,
    int64_t gate_up_output_size, int64_t down_output_size,
    int64_t gate_up_padding_cols, int64_t down_padding_cols,
    const std::optional<torch::stable::Tensor>& down_bias) {
  check_cuda_matrix(input, "input");
  check_cuda_matrix(gate_up_weight, "gate_up_weight");
  check_cuda_scale(gate_up_weight_scale, "gate_up_weight_scale");
  check_cuda_scale(gate_up_alpha, "gate_up_alpha");
  check_cuda_scale(gate_up_input_global_scale_inv,
                   "gate_up_input_global_scale_inv");
  check_cuda_matrix(down_weight, "down_weight");
  check_cuda_scale(down_weight_scale, "down_weight_scale");
  check_cuda_scale(down_alpha, "down_alpha");
  check_cuda_scale(down_input_global_scale_inv,
                   "down_input_global_scale_inv");

  STD_TORCH_CHECK(gate_up_output_size > 0,
                  "gate_up_output_size must be positive");
  STD_TORCH_CHECK(gate_up_output_size % 2 == 0,
                  "gate_up_output_size must be even for [gate | up]");
  STD_TORCH_CHECK(down_output_size > 0, "down_output_size must be positive");
  STD_TORCH_CHECK(gate_up_padding_cols >= 0,
                  "gate_up_padding_cols must be non-negative");
  STD_TORCH_CHECK(down_padding_cols >= 0,
                  "down_padding_cols must be non-negative");
  if (down_bias) {
    check_cuda_scale(*down_bias, "down_bias");
  }

  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      !down_bias.has_value(),
      "cutlass_nvfp4_mlp_parent_fused currently supports the bias-free Qwen "
      "MLP path only.");
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      gate_up_padding_cols == 0 && down_padding_cols == 0,
      "cutlass_nvfp4_mlp_parent_fused currently supports the no-K-padding "
      "NVFP4 target shapes only.");
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      gate_up_weight.size(0) == gate_up_output_size,
      "cutlass_nvfp4_mlp_parent_fused currently requires unpadded C1 N rows.");
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      down_weight.size(0) == down_output_size,
      "cutlass_nvfp4_mlp_parent_fused currently requires unpadded C2 N rows.");

  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Half ||
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "input must be fp16 or bf16");
  STD_TORCH_CHECK(gate_up_weight.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "gate_up_weight must be packed NVFP4 bytes");
  STD_TORCH_CHECK(down_weight.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "down_weight must be packed NVFP4 bytes");
  STD_TORCH_CHECK(gate_up_weight_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "gate_up_weight_scale must be fp8_e4m3fn");
  STD_TORCH_CHECK(down_weight_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "down_weight_scale must be fp8_e4m3fn");

  int64_t const m = input.size(0);
  int64_t const hidden = input.size(1);
  int64_t const act_cols = gate_up_output_size / 2;
  STD_TORCH_CHECK(hidden % 16 == 0,
                  "input hidden size must be divisible by 16");
  STD_TORCH_CHECK(act_cols % 16 == 0,
                  "C2 activation columns must be divisible by 16");
  STD_TORCH_CHECK(gate_up_weight.size(1) == hidden / 2,
                  "gate_up_weight K does not match input hidden size");
  STD_TORCH_CHECK(down_weight.size(1) == act_cols / 2,
                  "down_weight K does not match fused activation size");

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const int32_t sm = get_sm_version_num();

#if defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120
  if (sm >= 120 && sm < 130) {
    auto input_fp4 = allocate_fp4_payload(m, hidden, input);
    auto input_sf = allocate_swizzled_fp4_scale(m, hidden, input);
    auto act_payload = allocate_fp4_payload(m, act_cols, input);
    auto act_sf = allocate_swizzled_fp4_scale(m, act_cols, input);

    scaled_fp4_quant_out(input, gate_up_input_global_scale_inv, true,
                         input_fp4, input_sf);

    const char* c1_store_mode =
        std::getenv("VLLM_PROJ4_MLP_PARENT_C1_STORE_MODE");
    int32_t direct_store_mode = -1;
    if (c1_store_mode != nullptr) {
      if (std::strcmp(c1_store_mode, "atomic") == 0) {
        direct_store_mode = 0;
      } else if (std::strcmp(c1_store_mode, "byte") == 0) {
        direct_store_mode = 1;
      } else if (std::strcmp(c1_store_mode, "pair") == 0) {
        direct_store_mode = 2;
      } else if (std::strcmp(c1_store_mode, "interleaved_c2") == 0) {
        direct_store_mode = 3;
      } else if (std::strcmp(c1_store_mode, "nextpair_atomic") == 0) {
        direct_store_mode = 4;
      } else if (std::strcmp(c1_store_mode, "nextpair") == 0) {
        direct_store_mode = 5;
      } else if (std::strcmp(c1_store_mode, "paired_output") == 0) {
        direct_store_mode = 6;
      }
    }

    auto output = empty_cuda_tensor({m, down_weight.size(0)},
                                    input.scalar_type(), input);

    if (direct_store_mode == 3) {
      auto interleaved_act_payload =
          allocate_fp4_payload(m, gate_up_output_size, input);
      cutlass_nvfp4_mlp_c1_act_quant_interleaved_sm120a(
          act_payload, interleaved_act_payload, act_sf, input_fp4,
          gate_up_weight, input_sf, gate_up_weight_scale, gate_up_alpha,
          down_input_global_scale_inv, act_cols,
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
          false);
      cutlass_scaled_fp4_mm_interleaved_a_sm120a(
          output, interleaved_act_payload, down_weight, act_sf,
          down_weight_scale, down_alpha, act_cols);
      return output;
    }

    if (direct_store_mode >= 0) {
      if (direct_store_mode == 6) {
        cutlass_nvfp4_mlp_c1_act_quant_paired_output_sm120a(
            act_payload, act_sf, input_fp4, gate_up_weight, input_sf,
            gate_up_weight_scale, gate_up_alpha, down_input_global_scale_inv,
            act_cols,
            input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
            false);
      } else {
        cutlass_nvfp4_mlp_c1_act_quant_direct_store_sm120a(
            act_payload, act_sf, input_fp4, gate_up_weight, input_sf,
            gate_up_weight_scale, gate_up_alpha, down_input_global_scale_inv,
            act_cols,
            input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
            direct_store_mode, false);
      }
    } else {
      auto interleaved_act_payload =
          allocate_fp4_payload(m, gate_up_output_size, input);
      cutlass_nvfp4_mlp_c1_act_quant_sm120a(
          act_payload, interleaved_act_payload, act_sf, input_fp4,
          gate_up_weight, input_sf, gate_up_weight_scale, gate_up_alpha,
          down_input_global_scale_inv, act_cols,
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
          false);
    }

    cutlass_scaled_fp4_mm(output, act_payload, down_weight, act_sf,
                          down_weight_scale, down_alpha);
    return output;
  }
#endif

  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false, "No compiled cutlass_nvfp4_mlp_parent_fused kernel for SM ", sm,
      ". Recompile with CUDA >= 12.8 and CC >= 120.");
  return input;
}

static torch::stable::Tensor cutlass_nvfp4_mlp_parent_fused_from_input_fp4_impl(
    const torch::stable::Tensor& input_fp4,
    const torch::stable::Tensor& input_fp4_scale,
    const torch::stable::Tensor& gate_up_weight,
    const torch::stable::Tensor& gate_up_weight_scale,
    const torch::stable::Tensor& gate_up_alpha,
    const torch::stable::Tensor& down_weight,
    const torch::stable::Tensor& down_weight_scale,
    const torch::stable::Tensor& down_alpha,
    const torch::stable::Tensor& down_input_global_scale_inv,
    int64_t gate_up_output_size, int64_t down_output_size,
    int64_t gate_up_padding_cols, int64_t down_padding_cols,
    bool use_bfloat16_intermediate,
    bool launch_c1_with_pdl,
    const std::optional<torch::stable::Tensor>& down_bias) {
  check_cuda_matrix(input_fp4, "input_fp4");
  check_cuda_scale(input_fp4_scale, "input_fp4_scale");
  check_cuda_matrix(gate_up_weight, "gate_up_weight");
  check_cuda_scale(gate_up_weight_scale, "gate_up_weight_scale");
  check_cuda_scale(gate_up_alpha, "gate_up_alpha");
  check_cuda_matrix(down_weight, "down_weight");
  check_cuda_scale(down_weight_scale, "down_weight_scale");
  check_cuda_scale(down_alpha, "down_alpha");
  check_cuda_scale(down_input_global_scale_inv,
                   "down_input_global_scale_inv");

  STD_TORCH_CHECK(gate_up_output_size > 0,
                  "gate_up_output_size must be positive");
  STD_TORCH_CHECK(gate_up_output_size % 2 == 0,
                  "gate_up_output_size must be even for [gate | up]");
  STD_TORCH_CHECK(down_output_size > 0, "down_output_size must be positive");
  STD_TORCH_CHECK(gate_up_padding_cols >= 0,
                  "gate_up_padding_cols must be non-negative");
  STD_TORCH_CHECK(down_padding_cols >= 0,
                  "down_padding_cols must be non-negative");
  if (down_bias) {
    check_cuda_scale(*down_bias, "down_bias");
  }

  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      !down_bias.has_value(),
      "cutlass_nvfp4_mlp_parent_fused_from_input_fp4 currently supports the "
      "bias-free Qwen MLP path only.");
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      gate_up_padding_cols == 0 && down_padding_cols == 0,
      "cutlass_nvfp4_mlp_parent_fused_from_input_fp4 currently supports the "
      "no-K-padding NVFP4 target shapes only.");
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      gate_up_weight.size(0) == gate_up_output_size,
      "cutlass_nvfp4_mlp_parent_fused_from_input_fp4 currently requires "
      "unpadded C1 N rows.");
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      down_weight.size(0) == down_output_size,
      "cutlass_nvfp4_mlp_parent_fused_from_input_fp4 currently requires "
      "unpadded C2 N rows.");

  STD_TORCH_CHECK(input_fp4.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "input_fp4 must be packed NVFP4 bytes");
  STD_TORCH_CHECK(input_fp4_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "input_fp4_scale must be fp8_e4m3fn");
  STD_TORCH_CHECK(gate_up_weight.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "gate_up_weight must be packed NVFP4 bytes");
  STD_TORCH_CHECK(down_weight.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "down_weight must be packed NVFP4 bytes");
  STD_TORCH_CHECK(gate_up_weight_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "gate_up_weight_scale must be fp8_e4m3fn");
  STD_TORCH_CHECK(down_weight_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "down_weight_scale must be fp8_e4m3fn");

  int64_t const m = input_fp4.size(0);
  int64_t const hidden = input_fp4.size(1) * 2;
  int64_t const act_cols = gate_up_output_size / 2;
  STD_TORCH_CHECK(hidden % 16 == 0,
                  "input hidden size must be divisible by 16");
  STD_TORCH_CHECK(act_cols % 16 == 0,
                  "C2 activation columns must be divisible by 16");
  STD_TORCH_CHECK(gate_up_weight.size(1) == input_fp4.size(1),
                  "gate_up_weight K does not match input_fp4 hidden size");
  STD_TORCH_CHECK(down_weight.size(1) == act_cols / 2,
                  "down_weight K does not match fused activation size");

  const torch::stable::accelerator::DeviceGuard device_guard(
      input_fp4.get_device_index());
  const int32_t sm = get_sm_version_num();

#if defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120
  if (sm >= 120 && sm < 130) {
    auto act_payload = allocate_fp4_payload(m, act_cols, input_fp4);
    auto act_sf = allocate_swizzled_fp4_scale(m, act_cols, input_fp4);

    const char* c1_store_mode =
        std::getenv("VLLM_PROJ4_MLP_PARENT_C1_STORE_MODE");
    int32_t direct_store_mode = -1;
    if (c1_store_mode != nullptr) {
      if (std::strcmp(c1_store_mode, "atomic") == 0) {
        direct_store_mode = 0;
      } else if (std::strcmp(c1_store_mode, "byte") == 0) {
        direct_store_mode = 1;
      } else if (std::strcmp(c1_store_mode, "pair") == 0) {
        direct_store_mode = 2;
      } else if (std::strcmp(c1_store_mode, "interleaved_c2") == 0) {
        direct_store_mode = 3;
      } else if (std::strcmp(c1_store_mode, "nextpair_atomic") == 0) {
        direct_store_mode = 4;
      } else if (std::strcmp(c1_store_mode, "nextpair") == 0) {
        direct_store_mode = 5;
      } else if (std::strcmp(c1_store_mode, "paired_output") == 0) {
        direct_store_mode = 6;
      }
    }

    auto output_dtype = use_bfloat16_intermediate
                            ? torch::headeronly::ScalarType::BFloat16
                            : torch::headeronly::ScalarType::Half;
    auto output =
        empty_cuda_tensor({m, down_weight.size(0)}, output_dtype, input_fp4);

    if (direct_store_mode == 3) {
      auto interleaved_act_payload =
          allocate_fp4_payload(m, gate_up_output_size, input_fp4);
      cutlass_nvfp4_mlp_c1_act_quant_interleaved_sm120a(
          act_payload, interleaved_act_payload, act_sf, input_fp4,
          gate_up_weight, input_fp4_scale, gate_up_weight_scale, gate_up_alpha,
          down_input_global_scale_inv, act_cols, use_bfloat16_intermediate,
          launch_c1_with_pdl);
      cutlass_scaled_fp4_mm_interleaved_a_sm120a(
          output, interleaved_act_payload, down_weight, act_sf,
          down_weight_scale, down_alpha, act_cols);
      return output;
    }

    if (direct_store_mode >= 0) {
      if (direct_store_mode == 6) {
        cutlass_nvfp4_mlp_c1_act_quant_paired_output_sm120a(
            act_payload, act_sf, input_fp4, gate_up_weight, input_fp4_scale,
            gate_up_weight_scale, gate_up_alpha, down_input_global_scale_inv,
            act_cols, use_bfloat16_intermediate, launch_c1_with_pdl);
      } else {
        cutlass_nvfp4_mlp_c1_act_quant_direct_store_sm120a(
            act_payload, act_sf, input_fp4, gate_up_weight, input_fp4_scale,
            gate_up_weight_scale, gate_up_alpha, down_input_global_scale_inv,
            act_cols, use_bfloat16_intermediate, direct_store_mode,
            launch_c1_with_pdl);
      }
    } else {
      auto interleaved_act_payload =
          allocate_fp4_payload(m, gate_up_output_size, input_fp4);
      cutlass_nvfp4_mlp_c1_act_quant_sm120a(
          act_payload, interleaved_act_payload, act_sf, input_fp4,
          gate_up_weight, input_fp4_scale, gate_up_weight_scale, gate_up_alpha,
          down_input_global_scale_inv, act_cols, use_bfloat16_intermediate,
          launch_c1_with_pdl);
    }

    cutlass_scaled_fp4_mm(output, act_payload, down_weight, act_sf,
                          down_weight_scale, down_alpha);
    return output;
  }
#endif

  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_nvfp4_mlp_parent_fused_from_input_fp4 kernel for "
      "SM ",
      sm, ". Recompile with CUDA >= 12.8 and CC >= 120.");
  return input_fp4;
}

torch::stable::Tensor cutlass_nvfp4_mlp_parent_fused_from_input_fp4(
    const torch::stable::Tensor& input_fp4,
    const torch::stable::Tensor& input_fp4_scale,
    const torch::stable::Tensor& gate_up_weight,
    const torch::stable::Tensor& gate_up_weight_scale,
    const torch::stable::Tensor& gate_up_alpha,
    const torch::stable::Tensor& down_weight,
    const torch::stable::Tensor& down_weight_scale,
    const torch::stable::Tensor& down_alpha,
    const torch::stable::Tensor& down_input_global_scale_inv,
    int64_t gate_up_output_size, int64_t down_output_size,
    int64_t gate_up_padding_cols, int64_t down_padding_cols,
    bool use_bfloat16_intermediate,
    const std::optional<torch::stable::Tensor>& down_bias) {
  return cutlass_nvfp4_mlp_parent_fused_from_input_fp4_impl(
      input_fp4, input_fp4_scale, gate_up_weight, gate_up_weight_scale,
      gate_up_alpha, down_weight, down_weight_scale, down_alpha,
      down_input_global_scale_inv, gate_up_output_size, down_output_size,
      gate_up_padding_cols, down_padding_cols, use_bfloat16_intermediate,
      false, down_bias);
}

torch::stable::Tensor cutlass_nvfp4_rmsnorm_quant_mlp_parent_pdl(
    const torch::stable::Tensor& input,
    torch::stable::Tensor& residual,
    const torch::stable::Tensor& rms_weight,
    const torch::stable::Tensor& gate_up_weight,
    const torch::stable::Tensor& gate_up_weight_scale,
    const torch::stable::Tensor& gate_up_alpha,
    const torch::stable::Tensor& gate_up_input_global_scale_inv,
    const torch::stable::Tensor& down_weight,
    const torch::stable::Tensor& down_weight_scale,
    const torch::stable::Tensor& down_alpha,
    const torch::stable::Tensor& down_input_global_scale_inv,
    double epsilon,
    bool rms_weight_offset,
    int64_t gate_up_output_size, int64_t down_output_size,
    int64_t gate_up_padding_cols, int64_t down_padding_cols,
    bool use_bfloat16_intermediate,
    const std::optional<torch::stable::Tensor>& down_bias) {
  check_cuda_matrix(input, "input");
  check_cuda_matrix(residual, "residual");
  check_cuda_scale(rms_weight, "rms_weight");
  check_cuda_scale(gate_up_input_global_scale_inv,
                   "gate_up_input_global_scale_inv");
  STD_TORCH_CHECK(input.size(0) == residual.size(0) &&
                      input.size(1) == residual.size(1),
                  "input and residual shapes must match");
  STD_TORCH_CHECK(input.scalar_type() == residual.scalar_type(),
                  "input and residual dtypes must match");
  STD_TORCH_CHECK(input.scalar_type() == rms_weight.scalar_type(),
                  "input and rms_weight dtypes must match");

  int64_t const m = input.size(0);
  int64_t const hidden = input.size(1);
  auto input_fp4 = allocate_fp4_payload(m, hidden, input);
  auto input_fp4_scale = allocate_swizzled_fp4_scale(m, hidden, input);

  fused_add_rms_norm_scaled_fp4_quant_sm1xxa(
      input_fp4, input_fp4_scale, input, residual, rms_weight,
      gate_up_input_global_scale_inv, epsilon, rms_weight_offset, true, true);

  return cutlass_nvfp4_mlp_parent_fused_from_input_fp4_impl(
      input_fp4, input_fp4_scale, gate_up_weight, gate_up_weight_scale,
      gate_up_alpha, down_weight, down_weight_scale, down_alpha,
      down_input_global_scale_inv, gate_up_output_size, down_output_size,
      gate_up_padding_cols, down_padding_cols, use_bfloat16_intermediate, true,
      down_bias);
}

torch::stable::Tensor cutlass_nvfp4_rmsnorm_quant_mlp_parent_row_chunk(
    const torch::stable::Tensor& input,
    torch::stable::Tensor& residual,
    const torch::stable::Tensor& rms_weight,
    const torch::stable::Tensor& gate_up_weight,
    const torch::stable::Tensor& gate_up_weight_scale,
    const torch::stable::Tensor& gate_up_alpha,
    const torch::stable::Tensor& gate_up_input_global_scale_inv,
    const torch::stable::Tensor& down_weight,
    const torch::stable::Tensor& down_weight_scale,
    const torch::stable::Tensor& down_alpha,
    const torch::stable::Tensor& down_input_global_scale_inv,
    double epsilon,
    bool rms_weight_offset,
    int64_t gate_up_output_size, int64_t down_output_size,
    int64_t gate_up_padding_cols, int64_t down_padding_cols,
    bool use_bfloat16_intermediate,
    int64_t chunk_rows,
    const std::optional<torch::stable::Tensor>& down_bias) {
  check_cuda_matrix(input, "input");
  check_cuda_matrix(residual, "residual");
  check_cuda_scale(rms_weight, "rms_weight");
  check_cuda_matrix(gate_up_weight, "gate_up_weight");
  check_cuda_scale(gate_up_weight_scale, "gate_up_weight_scale");
  check_cuda_scale(gate_up_alpha, "gate_up_alpha");
  check_cuda_scale(gate_up_input_global_scale_inv,
                   "gate_up_input_global_scale_inv");
  check_cuda_matrix(down_weight, "down_weight");
  check_cuda_scale(down_weight_scale, "down_weight_scale");
  check_cuda_scale(down_alpha, "down_alpha");
  check_cuda_scale(down_input_global_scale_inv,
                   "down_input_global_scale_inv");

  STD_TORCH_CHECK(chunk_rows > 0, "chunk_rows must be positive");
  STD_TORCH_CHECK(input.size(0) == residual.size(0) &&
                      input.size(1) == residual.size(1),
                  "input and residual shapes must match");
  STD_TORCH_CHECK(input.scalar_type() == residual.scalar_type(),
                  "input and residual dtypes must match");
  STD_TORCH_CHECK(input.scalar_type() == rms_weight.scalar_type(),
                  "input and rms_weight dtypes must match");
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      !down_bias.has_value(),
      "row-chunk RMSNorm-MLP parent fusion currently supports the bias-free "
      "Qwen MLP path only.");
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      gate_up_padding_cols == 0 && down_padding_cols == 0,
      "row-chunk RMSNorm-MLP parent fusion currently supports the no-K-padding "
      "NVFP4 target shapes only.");
  STD_TORCH_CHECK(gate_up_output_size > 0 && gate_up_output_size % 2 == 0,
                  "gate_up_output_size must be positive and even");
  STD_TORCH_CHECK(down_output_size > 0, "down_output_size must be positive");
  STD_TORCH_CHECK(gate_up_weight.size(0) == gate_up_output_size,
                  "gate_up_weight rows must equal gate_up_output_size");
  STD_TORCH_CHECK(down_weight.size(0) == down_output_size,
                  "down_weight rows must equal down_output_size");
  STD_TORCH_CHECK(input.scalar_type() == torch::headeronly::ScalarType::Half ||
                      input.scalar_type() ==
                          torch::headeronly::ScalarType::BFloat16,
                  "input must be fp16 or bf16");
  STD_TORCH_CHECK(gate_up_weight.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "gate_up_weight must be packed NVFP4 bytes");
  STD_TORCH_CHECK(down_weight.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "down_weight must be packed NVFP4 bytes");
  STD_TORCH_CHECK(gate_up_weight_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "gate_up_weight_scale must be fp8_e4m3fn");
  STD_TORCH_CHECK(down_weight_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "down_weight_scale must be fp8_e4m3fn");

  int64_t const m = input.size(0);
  int64_t const hidden = input.size(1);
  int64_t const act_cols = gate_up_output_size / 2;
  STD_TORCH_CHECK(hidden % 16 == 0,
                  "input hidden size must be divisible by 16");
  STD_TORCH_CHECK(act_cols % 16 == 0,
                  "C2 activation columns must be divisible by 16");
  STD_TORCH_CHECK(gate_up_weight.size(1) == hidden / 2,
                  "gate_up_weight K does not match input hidden size");
  STD_TORCH_CHECK(down_weight.size(1) == act_cols / 2,
                  "down_weight K does not match fused activation size");

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const int32_t sm = get_sm_version_num();

#if defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120
	  if (sm >= 120 && sm < 130) {
	    auto output_dtype = use_bfloat16_intermediate
	                            ? torch::headeronly::ScalarType::BFloat16
	                            : torch::headeronly::ScalarType::Half;
	    auto output = empty_cuda_tensor({m, down_weight.size(0)}, output_dtype,
	                                    input);
	    if (row_chunk_use_ready_wait()) {
	      STD_TORCH_CHECK(
	          chunk_rows <= static_cast<int64_t>(std::numeric_limits<int>::max()),
	          "row-ready chunk_rows is too large");
	      auto input_fp4 = allocate_fp4_payload(m, hidden, input);
	      auto input_fp4_scale = allocate_swizzled_fp4_scale(m, hidden, input);
	      auto act_payload = allocate_fp4_payload(m, act_cols, input);
	      auto interleaved_act_payload =
	          allocate_fp4_payload(m, gate_up_output_size, input);
	      auto act_sf = allocate_swizzled_fp4_scale(m, act_cols, input);
	      int64_t const chunks = (m + chunk_rows - 1) / chunk_rows;
		      auto chunk_counts =
		          empty_cuda_tensor({chunks}, torch::headeronly::ScalarType::Int,
		                            input);
		      int const wait_mode = row_ready_wait_mode();
		      bool const publish_ready_flag = wait_mode == 0 || wait_mode == 2;
		      auto chunk_ready =
		          publish_ready_flag
		              ? empty_cuda_tensor({chunks}, torch::headeronly::ScalarType::Int,
		                                  input)
		              : chunk_counts;
		      cudaStream_t stream = get_current_cuda_stream(input.get_device_index());
		      PROJ4_CUDA_CHECK(
		          cudaMemsetAsync(chunk_counts.data_ptr(), 0,
		                          chunks * static_cast<int64_t>(sizeof(int)),
		                          stream));
		      if (publish_ready_flag) {
		        PROJ4_CUDA_CHECK(
		            cudaMemsetAsync(chunk_ready.data_ptr(), 0,
		                            chunks * static_cast<int64_t>(sizeof(int)),
		                            stream));
		      }
		      void* wait_ptr = publish_ready_flag ? chunk_ready.data_ptr()
		                                          : chunk_counts.data_ptr();
		      proj4_c1_row_ready_set(
		          static_cast<unsigned int*>(wait_ptr),
		          static_cast<int>(chunk_rows), static_cast<int>(m), wait_mode,
		          stream);

		      fused_add_rms_norm_scaled_fp4_quant_sm1xxa_row_ready_stream(
		          input_fp4, input_fp4_scale, input, residual, rms_weight,
		          gate_up_input_global_scale_inv, chunk_counts, chunk_ready, epsilon,
		          rms_weight_offset, true, chunk_rows, stream, true,
		          publish_ready_flag);

	      cutlass_nvfp4_mlp_c1_act_quant_sm120a_stream(
	          act_payload, interleaved_act_payload, act_sf, input_fp4,
	          gate_up_weight, input_fp4_scale, gate_up_weight_scale, gate_up_alpha,
	          down_input_global_scale_inv, act_cols, use_bfloat16_intermediate,
	          stream, true);
	      proj4_c1_row_ready_clear(stream);

	      cutlass_scaled_fp4_mm(output, act_payload, down_weight, act_sf,
	                            down_weight_scale, down_alpha);
	      return output;
	    }
	    if (row_chunk_use_single_c2()) {
	      STD_TORCH_CHECK(
	          chunk_rows % 128 == 0,
	          "VLLM_PROJ4_RMSNORM_MLP_ROW_CHUNK_MODE=single_c2 requires "
	          "chunk_rows to be a multiple of 128 so swizzled scale rows can be "
	          "copied into the full-M scale tensor without layout conversion.");
	      auto act_payload = allocate_fp4_payload(m, act_cols, input);
	      auto act_sf = allocate_swizzled_fp4_scale(m, act_cols, input);
	      cudaStream_t primary_stream =
	          get_current_cuda_stream(input.get_device_index());
	      cudaStream_t consumer_stream{};
	      PROJ4_CUDA_CHECK(cudaStreamCreateWithFlags(&consumer_stream,
	                                                 cudaStreamNonBlocking));

	      std::vector<cudaEvent_t> ready_events;
	      std::vector<torch::stable::Tensor> keepalive;
	      int64_t const chunks = (m + chunk_rows - 1) / chunk_rows;
	      ready_events.reserve(static_cast<size_t>(chunks));
	      keepalive.reserve(static_cast<size_t>(chunks) * 5);

	      int64_t const act_payload_row_bytes = act_payload.size(1);
	      int64_t const act_sf_row_bytes = act_sf.size(1);
	      for (int64_t row_start = 0; row_start < m; row_start += chunk_rows) {
	        int64_t const rows = std::min<int64_t>(chunk_rows, m - row_start);
	        auto input_fp4 = allocate_fp4_payload(rows, hidden, input);
	        auto input_fp4_scale = allocate_swizzled_fp4_scale(rows, hidden, input);
	        auto chunk_act_payload = allocate_fp4_payload(rows, act_cols, input);
	        auto chunk_interleaved_act_payload =
	            allocate_fp4_payload(rows, gate_up_output_size, input);
	        auto chunk_act_sf = allocate_swizzled_fp4_scale(rows, act_cols, input);

	        fused_add_rms_norm_scaled_fp4_quant_sm1xxa_chunk_stream(
	            input_fp4, input_fp4_scale, input, residual, rms_weight,
	            gate_up_input_global_scale_inv, epsilon, rms_weight_offset, true,
	            row_start, rows, primary_stream, false);

	        cudaEvent_t ready{};
	        PROJ4_CUDA_CHECK(cudaEventCreateWithFlags(&ready,
	                                                  cudaEventDisableTiming));
	        PROJ4_CUDA_CHECK(cudaEventRecord(ready, primary_stream));
	        PROJ4_CUDA_CHECK(cudaStreamWaitEvent(consumer_stream, ready, 0));
	        ready_events.push_back(ready);

	        cutlass_nvfp4_mlp_c1_act_quant_sm120a_stream(
	            chunk_act_payload, chunk_interleaved_act_payload, chunk_act_sf,
	            input_fp4, gate_up_weight, input_fp4_scale, gate_up_weight_scale,
	            gate_up_alpha, down_input_global_scale_inv, act_cols,
	            use_bfloat16_intermediate, consumer_stream, false);

	        auto* dst_payload =
	            static_cast<char*>(act_payload.data_ptr()) +
	            row_start * act_payload_row_bytes;
	        PROJ4_CUDA_CHECK(cudaMemcpyAsync(
	            dst_payload, chunk_act_payload.data_ptr(),
	            rows * act_payload_row_bytes, cudaMemcpyDeviceToDevice,
	            consumer_stream));

	        auto* dst_sf = static_cast<char*>(act_sf.data_ptr()) +
	                       row_start * act_sf_row_bytes;
	        PROJ4_CUDA_CHECK(cudaMemcpyAsync(
	            dst_sf, chunk_act_sf.data_ptr(),
	            chunk_act_sf.size(0) * act_sf_row_bytes,
	            cudaMemcpyDeviceToDevice, consumer_stream));

	        keepalive.push_back(input_fp4);
	        keepalive.push_back(input_fp4_scale);
	        keepalive.push_back(chunk_act_payload);
	        keepalive.push_back(chunk_interleaved_act_payload);
	        keepalive.push_back(chunk_act_sf);
	      }

	      cudaEvent_t done{};
	      PROJ4_CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
	      PROJ4_CUDA_CHECK(cudaEventRecord(done, consumer_stream));
	      PROJ4_CUDA_CHECK(cudaStreamWaitEvent(primary_stream, done, 0));
	      PROJ4_CUDA_CHECK(cudaStreamSynchronize(consumer_stream));
	      for (auto event : ready_events) {
	        PROJ4_CUDA_CHECK(cudaEventDestroy(event));
	      }
	      PROJ4_CUDA_CHECK(cudaEventDestroy(done));
	      PROJ4_CUDA_CHECK(cudaStreamDestroy(consumer_stream));

	      cutlass_scaled_fp4_mm(output, act_payload, down_weight, act_sf,
	                            down_weight_scale, down_alpha);
	      return output;
	    }
	    cudaStream_t primary_stream =
	        get_current_cuda_stream(input.get_device_index());
	    cudaStream_t consumer_stream{};
    PROJ4_CUDA_CHECK(cudaStreamCreateWithFlags(&consumer_stream,
                                               cudaStreamNonBlocking));

    std::vector<cudaEvent_t> ready_events;
    std::vector<torch::stable::Tensor> keepalive;
    int64_t const chunks = (m + chunk_rows - 1) / chunk_rows;
    ready_events.reserve(static_cast<size_t>(chunks));
    keepalive.reserve(static_cast<size_t>(chunks) * 6);

    for (int64_t row_start = 0; row_start < m; row_start += chunk_rows) {
      int64_t const rows = std::min<int64_t>(chunk_rows, m - row_start);
      auto input_fp4 = allocate_fp4_payload(rows, hidden, input);
      auto input_fp4_scale = allocate_swizzled_fp4_scale(rows, hidden, input);
      auto act_payload = allocate_fp4_payload(rows, act_cols, input);
      auto interleaved_act_payload =
          allocate_fp4_payload(rows, gate_up_output_size, input);
      auto act_sf = allocate_swizzled_fp4_scale(rows, act_cols, input);
      auto chunk_output =
          empty_cuda_tensor({rows, down_weight.size(0)}, output_dtype, input);

      fused_add_rms_norm_scaled_fp4_quant_sm1xxa_chunk_stream(
          input_fp4, input_fp4_scale, input, residual, rms_weight,
          gate_up_input_global_scale_inv, epsilon, rms_weight_offset, true,
          row_start, rows, primary_stream, false);

      cudaEvent_t ready{};
      PROJ4_CUDA_CHECK(cudaEventCreateWithFlags(&ready,
                                                cudaEventDisableTiming));
      PROJ4_CUDA_CHECK(cudaEventRecord(ready, primary_stream));
      PROJ4_CUDA_CHECK(cudaStreamWaitEvent(consumer_stream, ready, 0));
      ready_events.push_back(ready);

      cutlass_nvfp4_mlp_c1_act_quant_sm120a_stream(
          act_payload, interleaved_act_payload, act_sf, input_fp4,
          gate_up_weight, input_fp4_scale, gate_up_weight_scale, gate_up_alpha,
          down_input_global_scale_inv, act_cols, use_bfloat16_intermediate,
          consumer_stream, false);

      cutlass_scaled_fp4_mm_sm120a_stream(
          chunk_output, act_payload, down_weight, act_sf, down_weight_scale,
          down_alpha, consumer_stream);

      int64_t const row_bytes =
          down_weight.size(0) * dtype_element_size_bytes(output_dtype);
      auto* dst = static_cast<char*>(output.data_ptr()) + row_start * row_bytes;
      PROJ4_CUDA_CHECK(cudaMemcpyAsync(
          dst, chunk_output.data_ptr(), rows * row_bytes,
          cudaMemcpyDeviceToDevice, consumer_stream));

      keepalive.push_back(input_fp4);
      keepalive.push_back(input_fp4_scale);
      keepalive.push_back(act_payload);
      keepalive.push_back(interleaved_act_payload);
      keepalive.push_back(act_sf);
      keepalive.push_back(chunk_output);
    }

    cudaEvent_t done{};
    PROJ4_CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    PROJ4_CUDA_CHECK(cudaEventRecord(done, consumer_stream));
    PROJ4_CUDA_CHECK(cudaStreamWaitEvent(primary_stream, done, 0));
    PROJ4_CUDA_CHECK(cudaStreamSynchronize(consumer_stream));
    for (auto event : ready_events) {
      PROJ4_CUDA_CHECK(cudaEventDestroy(event));
    }
    PROJ4_CUDA_CHECK(cudaEventDestroy(done));
    PROJ4_CUDA_CHECK(cudaStreamDestroy(consumer_stream));
    return output;
  }
#endif

  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled row-chunk RMSNorm-MLP parent fusion kernel for SM ", sm,
      ". Recompile with CUDA >= 12.8 and CC >= 120.");
  return input;
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor>
cutlass_nvfp4_mlp_c1_act_quant(
    const torch::stable::Tensor& input_fp4,
    const torch::stable::Tensor& gate_up_weight,
    const torch::stable::Tensor& input_fp4_scale,
    const torch::stable::Tensor& gate_up_weight_scale,
    const torch::stable::Tensor& gate_up_alpha,
    const torch::stable::Tensor& c2_input_global_scale_inv,
    int64_t logical_act_cols, bool use_bfloat16_intermediate) {
  check_cuda_matrix(input_fp4, "input_fp4");
  check_cuda_matrix(gate_up_weight, "gate_up_weight");
  check_cuda_scale(input_fp4_scale, "input_fp4_scale");
  check_cuda_scale(gate_up_weight_scale, "gate_up_weight_scale");
  check_cuda_scale(gate_up_alpha, "gate_up_alpha");
  check_cuda_scale(c2_input_global_scale_inv, "c2_input_global_scale_inv");

  STD_TORCH_CHECK(input_fp4.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "input_fp4 must be packed NVFP4 bytes");
  STD_TORCH_CHECK(gate_up_weight.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "gate_up_weight must be packed NVFP4 bytes");
  STD_TORCH_CHECK(input_fp4_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "input_fp4_scale must be fp8_e4m3fn");
  STD_TORCH_CHECK(gate_up_weight_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "gate_up_weight_scale must be fp8_e4m3fn");
  STD_TORCH_CHECK(logical_act_cols > 0,
                  "logical_act_cols must be positive");

#if defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120
  int64_t const m = input_fp4.size(0);
  auto act_payload = allocate_fp4_payload(m, logical_act_cols, input_fp4);
  auto interleaved_act_payload =
      allocate_fp4_payload(m, logical_act_cols * 2, input_fp4);
  auto act_sf = allocate_swizzled_fp4_scale(m, logical_act_cols, input_fp4);

  cutlass_nvfp4_mlp_c1_act_quant_sm120a(
      act_payload, interleaved_act_payload, act_sf, input_fp4, gate_up_weight,
      input_fp4_scale, gate_up_weight_scale, gate_up_alpha,
      c2_input_global_scale_inv, logical_act_cols,
      use_bfloat16_intermediate, false);
  return {act_payload, act_sf};
#endif

  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false, "No compiled cutlass_nvfp4_mlp_c1_act_quant kernel.");
  return {input_fp4, input_fp4_scale};
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor>
cutlass_nvfp4_mlp_c1_act_quant_direct_store(
    const torch::stable::Tensor& input_fp4,
    const torch::stable::Tensor& gate_up_weight,
    const torch::stable::Tensor& input_fp4_scale,
    const torch::stable::Tensor& gate_up_weight_scale,
    const torch::stable::Tensor& gate_up_alpha,
    const torch::stable::Tensor& c2_input_global_scale_inv,
    int64_t logical_act_cols, bool use_bfloat16_intermediate) {
  check_cuda_matrix(input_fp4, "input_fp4");
  check_cuda_matrix(gate_up_weight, "gate_up_weight");
  check_cuda_scale(input_fp4_scale, "input_fp4_scale");
  check_cuda_scale(gate_up_weight_scale, "gate_up_weight_scale");
  check_cuda_scale(gate_up_alpha, "gate_up_alpha");
  check_cuda_scale(c2_input_global_scale_inv, "c2_input_global_scale_inv");

  STD_TORCH_CHECK(input_fp4.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "input_fp4 must be packed NVFP4 bytes");
  STD_TORCH_CHECK(gate_up_weight.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "gate_up_weight must be packed NVFP4 bytes");
  STD_TORCH_CHECK(input_fp4_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "input_fp4_scale must be fp8_e4m3fn");
  STD_TORCH_CHECK(gate_up_weight_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "gate_up_weight_scale must be fp8_e4m3fn");
  STD_TORCH_CHECK(logical_act_cols > 0,
                  "logical_act_cols must be positive");

#if defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120
  int64_t const m = input_fp4.size(0);
  auto act_payload = allocate_fp4_payload(m, logical_act_cols, input_fp4);
  auto act_sf = allocate_swizzled_fp4_scale(m, logical_act_cols, input_fp4);

  cutlass_nvfp4_mlp_c1_act_quant_direct_store_sm120a(
      act_payload, act_sf, input_fp4, gate_up_weight, input_fp4_scale,
      gate_up_weight_scale, gate_up_alpha, c2_input_global_scale_inv,
      logical_act_cols, use_bfloat16_intermediate, 0, false);
  return {act_payload, act_sf};
#endif

  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false, "No compiled cutlass_nvfp4_mlp_c1_act_quant_direct_store kernel.");
  return {input_fp4, input_fp4_scale};
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor>
cutlass_nvfp4_mlp_c1_act_quant_direct_store_byte(
    const torch::stable::Tensor& input_fp4,
    const torch::stable::Tensor& gate_up_weight,
    const torch::stable::Tensor& input_fp4_scale,
    const torch::stable::Tensor& gate_up_weight_scale,
    const torch::stable::Tensor& gate_up_alpha,
    const torch::stable::Tensor& c2_input_global_scale_inv,
    int64_t logical_act_cols, bool use_bfloat16_intermediate) {
  check_cuda_matrix(input_fp4, "input_fp4");
  check_cuda_matrix(gate_up_weight, "gate_up_weight");
  check_cuda_scale(input_fp4_scale, "input_fp4_scale");
  check_cuda_scale(gate_up_weight_scale, "gate_up_weight_scale");
  check_cuda_scale(gate_up_alpha, "gate_up_alpha");
  check_cuda_scale(c2_input_global_scale_inv, "c2_input_global_scale_inv");

  STD_TORCH_CHECK(input_fp4.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "input_fp4 must be packed NVFP4 bytes");
  STD_TORCH_CHECK(gate_up_weight.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "gate_up_weight must be packed NVFP4 bytes");
  STD_TORCH_CHECK(input_fp4_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "input_fp4_scale must be fp8_e4m3fn");
  STD_TORCH_CHECK(gate_up_weight_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "gate_up_weight_scale must be fp8_e4m3fn");
  STD_TORCH_CHECK(logical_act_cols > 0,
                  "logical_act_cols must be positive");

#if defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120
  int64_t const m = input_fp4.size(0);
  auto act_payload = allocate_fp4_payload(m, logical_act_cols, input_fp4);
  auto act_sf = allocate_swizzled_fp4_scale(m, logical_act_cols, input_fp4);

  cutlass_nvfp4_mlp_c1_act_quant_direct_store_sm120a(
      act_payload, act_sf, input_fp4, gate_up_weight, input_fp4_scale,
      gate_up_weight_scale, gate_up_alpha, c2_input_global_scale_inv,
      logical_act_cols, use_bfloat16_intermediate, 1, false);
  return {act_payload, act_sf};
#endif

  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_nvfp4_mlp_c1_act_quant_direct_store_byte kernel.");
  return {input_fp4, input_fp4_scale};
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor>
cutlass_nvfp4_mlp_c1_act_quant_direct_store_pair(
    const torch::stable::Tensor& input_fp4,
    const torch::stable::Tensor& gate_up_weight,
    const torch::stable::Tensor& input_fp4_scale,
    const torch::stable::Tensor& gate_up_weight_scale,
    const torch::stable::Tensor& gate_up_alpha,
    const torch::stable::Tensor& c2_input_global_scale_inv,
    int64_t logical_act_cols, bool use_bfloat16_intermediate) {
  check_cuda_matrix(input_fp4, "input_fp4");
  check_cuda_matrix(gate_up_weight, "gate_up_weight");
  check_cuda_scale(input_fp4_scale, "input_fp4_scale");
  check_cuda_scale(gate_up_weight_scale, "gate_up_weight_scale");
  check_cuda_scale(gate_up_alpha, "gate_up_alpha");
  check_cuda_scale(c2_input_global_scale_inv, "c2_input_global_scale_inv");

  STD_TORCH_CHECK(input_fp4.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "input_fp4 must be packed NVFP4 bytes");
  STD_TORCH_CHECK(gate_up_weight.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "gate_up_weight must be packed NVFP4 bytes");
  STD_TORCH_CHECK(input_fp4_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "input_fp4_scale must be fp8_e4m3fn");
  STD_TORCH_CHECK(gate_up_weight_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "gate_up_weight_scale must be fp8_e4m3fn");
  STD_TORCH_CHECK(logical_act_cols > 0,
                  "logical_act_cols must be positive");

#if defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120
  int64_t const m = input_fp4.size(0);
  auto act_payload = allocate_fp4_payload(m, logical_act_cols, input_fp4);
  auto act_sf = allocate_swizzled_fp4_scale(m, logical_act_cols, input_fp4);

  cutlass_nvfp4_mlp_c1_act_quant_direct_store_sm120a(
      act_payload, act_sf, input_fp4, gate_up_weight, input_fp4_scale,
      gate_up_weight_scale, gate_up_alpha, c2_input_global_scale_inv,
      logical_act_cols, use_bfloat16_intermediate, 2, false);
  return {act_payload, act_sf};
#endif

  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_nvfp4_mlp_c1_act_quant_direct_store_pair kernel.");
  return {input_fp4, input_fp4_scale};
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor>
cutlass_nvfp4_mlp_c1_act_quant_direct_store_nextpair_atomic(
    const torch::stable::Tensor& input_fp4,
    const torch::stable::Tensor& gate_up_weight,
    const torch::stable::Tensor& input_fp4_scale,
    const torch::stable::Tensor& gate_up_weight_scale,
    const torch::stable::Tensor& gate_up_alpha,
    const torch::stable::Tensor& c2_input_global_scale_inv,
    int64_t logical_act_cols, bool use_bfloat16_intermediate) {
  check_cuda_matrix(input_fp4, "input_fp4");
  check_cuda_matrix(gate_up_weight, "gate_up_weight");
  check_cuda_scale(input_fp4_scale, "input_fp4_scale");
  check_cuda_scale(gate_up_weight_scale, "gate_up_weight_scale");
  check_cuda_scale(gate_up_alpha, "gate_up_alpha");
  check_cuda_scale(c2_input_global_scale_inv, "c2_input_global_scale_inv");

  STD_TORCH_CHECK(input_fp4.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "input_fp4 must be packed NVFP4 bytes");
  STD_TORCH_CHECK(gate_up_weight.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "gate_up_weight must be packed NVFP4 bytes");
  STD_TORCH_CHECK(input_fp4_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "input_fp4_scale must be fp8_e4m3fn");
  STD_TORCH_CHECK(gate_up_weight_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "gate_up_weight_scale must be fp8_e4m3fn");
  STD_TORCH_CHECK(logical_act_cols > 0,
                  "logical_act_cols must be positive");

#if defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120
  int64_t const m = input_fp4.size(0);
  auto act_payload = allocate_fp4_payload(m, logical_act_cols, input_fp4);
  auto act_sf = allocate_swizzled_fp4_scale(m, logical_act_cols, input_fp4);

  cutlass_nvfp4_mlp_c1_act_quant_direct_store_sm120a(
      act_payload, act_sf, input_fp4, gate_up_weight, input_fp4_scale,
      gate_up_weight_scale, gate_up_alpha, c2_input_global_scale_inv,
      logical_act_cols, use_bfloat16_intermediate, 4, false);
  return {act_payload, act_sf};
#endif

  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_nvfp4_mlp_c1_act_quant_direct_store_nextpair_atomic kernel.");
  return {input_fp4, input_fp4_scale};
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor>
cutlass_nvfp4_mlp_c1_act_quant_direct_store_nextpair(
    const torch::stable::Tensor& input_fp4,
    const torch::stable::Tensor& gate_up_weight,
    const torch::stable::Tensor& input_fp4_scale,
    const torch::stable::Tensor& gate_up_weight_scale,
    const torch::stable::Tensor& gate_up_alpha,
    const torch::stable::Tensor& c2_input_global_scale_inv,
    int64_t logical_act_cols, bool use_bfloat16_intermediate) {
  check_cuda_matrix(input_fp4, "input_fp4");
  check_cuda_matrix(gate_up_weight, "gate_up_weight");
  check_cuda_scale(input_fp4_scale, "input_fp4_scale");
  check_cuda_scale(gate_up_weight_scale, "gate_up_weight_scale");
  check_cuda_scale(gate_up_alpha, "gate_up_alpha");
  check_cuda_scale(c2_input_global_scale_inv, "c2_input_global_scale_inv");

  STD_TORCH_CHECK(input_fp4.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "input_fp4 must be packed NVFP4 bytes");
  STD_TORCH_CHECK(gate_up_weight.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "gate_up_weight must be packed NVFP4 bytes");
  STD_TORCH_CHECK(input_fp4_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "input_fp4_scale must be fp8_e4m3fn");
  STD_TORCH_CHECK(gate_up_weight_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "gate_up_weight_scale must be fp8_e4m3fn");
  STD_TORCH_CHECK(logical_act_cols > 0,
                  "logical_act_cols must be positive");

#if defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120
  int64_t const m = input_fp4.size(0);
  auto act_payload = allocate_fp4_payload(m, logical_act_cols, input_fp4);
  auto act_sf = allocate_swizzled_fp4_scale(m, logical_act_cols, input_fp4);

  cutlass_nvfp4_mlp_c1_act_quant_direct_store_sm120a(
      act_payload, act_sf, input_fp4, gate_up_weight, input_fp4_scale,
      gate_up_weight_scale, gate_up_alpha, c2_input_global_scale_inv,
      logical_act_cols, use_bfloat16_intermediate, 5, false);
  return {act_payload, act_sf};
#endif

  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_nvfp4_mlp_c1_act_quant_direct_store_nextpair kernel.");
  return {input_fp4, input_fp4_scale};
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor>
cutlass_nvfp4_mlp_c1_act_quant_paired_output(
    const torch::stable::Tensor& input_fp4,
    const torch::stable::Tensor& gate_up_weight,
    const torch::stable::Tensor& input_fp4_scale,
    const torch::stable::Tensor& gate_up_weight_scale,
    const torch::stable::Tensor& gate_up_alpha,
    const torch::stable::Tensor& c2_input_global_scale_inv,
    int64_t logical_act_cols, bool use_bfloat16_intermediate) {
  check_cuda_matrix(input_fp4, "input_fp4");
  check_cuda_matrix(gate_up_weight, "gate_up_weight");
  check_cuda_scale(input_fp4_scale, "input_fp4_scale");
  check_cuda_scale(gate_up_weight_scale, "gate_up_weight_scale");
  check_cuda_scale(gate_up_alpha, "gate_up_alpha");
  check_cuda_scale(c2_input_global_scale_inv, "c2_input_global_scale_inv");

  STD_TORCH_CHECK(input_fp4.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "input_fp4 must be packed NVFP4 bytes");
  STD_TORCH_CHECK(gate_up_weight.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "gate_up_weight must be packed NVFP4 bytes");
  STD_TORCH_CHECK(input_fp4_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "input_fp4_scale must be fp8_e4m3fn");
  STD_TORCH_CHECK(gate_up_weight_scale.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "gate_up_weight_scale must be fp8_e4m3fn");
  STD_TORCH_CHECK(logical_act_cols > 0,
                  "logical_act_cols must be positive");

#if defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120
  int64_t const m = input_fp4.size(0);
  auto act_payload = allocate_fp4_payload(m, logical_act_cols, input_fp4);
  auto act_sf = allocate_swizzled_fp4_scale(m, logical_act_cols, input_fp4);

  cutlass_nvfp4_mlp_c1_act_quant_paired_output_sm120a(
      act_payload, act_sf, input_fp4, gate_up_weight, input_fp4_scale,
      gate_up_weight_scale, gate_up_alpha, c2_input_global_scale_inv,
      logical_act_cols, use_bfloat16_intermediate, false);
  return {act_payload, act_sf};
#endif

  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_nvfp4_mlp_c1_act_quant_paired_output kernel.");
  return {input_fp4, input_fp4_scale};
}

bool cutlass_nvfp4_mlp_parent_fused_supported(
    int64_t cuda_device_capability) {
#if defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120
  if (cuda_device_capability >= 120 && cuda_device_capability < 130) {
    return cutlass_scaled_mm_supports_fp4(cuda_device_capability);
  }
#endif
  return false;
}
