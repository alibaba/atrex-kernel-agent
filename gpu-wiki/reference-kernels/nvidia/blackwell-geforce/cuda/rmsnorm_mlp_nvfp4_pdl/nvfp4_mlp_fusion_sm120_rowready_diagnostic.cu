// gpu-wiki archive note:
// Diagnostic SM120 CUTLASS/vLLM source for RMSNorm-MLP C1 act-quant fusion,
// row-ready waits, and Warp1/LoadMN producer-consumer probes. It is archived
// for the PDL/fusion lessons and should not be treated as a promoted kernel.
//
/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstddef>

#include <cuda_runtime.h>
#include <torch/csrc/stable/tensor.h>

#include "libtorch_stable/torch_utils.h"

// Task28 row-ready experiment: only this translation unit enables the CUTLASS
// SM120 A-load wait hook. Other CUTLASS users compile the stock header path.
#define VLLM_PROJ4_C1_ROW_READY_WAIT_HOOK 1
#define VLLM_PROJ4_C1_ROW_READY_WAIT_HOOK_DEFINE_SYMBOLS 1
#define VLLM_PROJ4_C1_WARP1_CONSUMER_HOOK 1

#include "cutlass_extensions/common.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm120_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_conversion.h"
#include "cutlass/util/packed_stride.hpp"

#include "core/math.hpp"

using namespace cute;

namespace {

void proj4_c1_row_ready_cuda_check(cudaError_t status, const char* expr) {
  STD_TORCH_CHECK(status == cudaSuccess, "CUDA call failed: ", expr, ": ",
                  cudaGetErrorString(status));
}

#define PROJ4_C1_READY_CUDA_CHECK(expr) \
  proj4_c1_row_ready_cuda_check((expr), #expr)

}  // namespace

void proj4_c1_row_ready_set(unsigned int* ready_flags, int chunk_rows,
                            int problem_m, int wait_mode,
                            cudaStream_t stream) {
  VllmProj4C1RowReadyConfig config{ready_flags, 1, chunk_rows, problem_m,
                                   wait_mode};
  PROJ4_C1_READY_CUDA_CHECK(cudaMemcpyToSymbolAsync(
      vllm_proj4_c1_row_ready_config, &config, sizeof(config), 0,
      cudaMemcpyHostToDevice, stream));
}

void proj4_c1_row_ready_clear(cudaStream_t stream) {
  int const disabled = 0;
  PROJ4_C1_READY_CUDA_CHECK(cudaMemcpyToSymbolAsync(
      vllm_proj4_c1_row_ready_config, &disabled, sizeof(disabled),
      offsetof(VllmProj4C1RowReadyConfig, enabled), cudaMemcpyHostToDevice,
      stream));
}

#define CHECK_TYPE(x, st, m)             \
  STD_TORCH_CHECK(x.scalar_type() == st, \
                  ": Inconsistency of torch::stable::Tensor type:", m)
#define CHECK_TH_CUDA(x, m) \
  STD_TORCH_CHECK(x.is_cuda(), m, ": must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m) \
  STD_TORCH_CHECK(x.is_contiguous(), m, ": must be contiguous")
#define CHECK_INPUT(x, st, m) \
  CHECK_TH_CUDA(x, m);        \
  CHECK_CONTIGUOUS(x, m);     \
  CHECK_TYPE(x, st, m)

constexpr auto FLOAT4_E2M1X2 = torch::headeronly::ScalarType::Byte;
constexpr auto SF_DTYPE = torch::headeronly::ScalarType::Float8_e4m3fn;

namespace vllm::proj4 {

struct RowMajorEvenK {};

enum class C1ScheduleMode {
  kAuto = 0,
  kCooperative = 1,
  kPingpong = 2,
};

C1ScheduleMode get_c1_schedule_mode() {
  const char* mode = std::getenv("VLLM_PROJ4_MLP_PARENT_C1_SCHEDULE");
  if (mode == nullptr) {
    return C1ScheduleMode::kAuto;
  }
  if (std::strcmp(mode, "cooperative") == 0 ||
      std::strcmp(mode, "coop") == 0) {
    return C1ScheduleMode::kCooperative;
  }
  if (std::strcmp(mode, "pingpong") == 0) {
    return C1ScheduleMode::kPingpong;
  }
  return C1ScheduleMode::kAuto;
}

enum class C1PairedOutputKernelForkMode {
  kDefault = 0,
  kShallowQueue = 1,
  kRawPublishProbe = 2,
  kLoadMnConsumeProbe = 3,
  kWarp1ConsumeProbe = 4,
  kWarp1ValidProbe = 5,
  kWarp1ValidNoWaitProbe = 6,
  kWarp1LoadMnValidProbe = 7,
};

C1PairedOutputKernelForkMode get_c1_paired_output_kernel_fork_mode() {
  const char* mode =
      std::getenv("VLLM_PROJ4_MLP_PARENT_C1_KERNEL_FORK");
  if (mode == nullptr) {
    return C1PairedOutputKernelForkMode::kDefault;
  }
  if (std::strcmp(mode, "shallow_queue") == 0 ||
      std::strcmp(mode, "task22_shallow_queue") == 0) {
    return C1PairedOutputKernelForkMode::kShallowQueue;
  }
  if (std::strcmp(mode, "raw_publish_probe") == 0 ||
      std::strcmp(mode, "publish_probe") == 0 ||
      std::strcmp(mode, "task39_raw_publish_probe") == 0) {
    return C1PairedOutputKernelForkMode::kRawPublishProbe;
  }
  if (std::strcmp(mode, "loadmn_consume_probe") == 0 ||
      std::strcmp(mode, "loadmn_probe") == 0 ||
      std::strcmp(mode, "task39_loadmn_consume_probe") == 0) {
    return C1PairedOutputKernelForkMode::kLoadMnConsumeProbe;
  }
  if (std::strcmp(mode, "warp1_consume_probe") == 0 ||
      std::strcmp(mode, "warp1_probe") == 0 ||
      std::strcmp(mode, "task39_warp1_consume_probe") == 0) {
    return C1PairedOutputKernelForkMode::kWarp1ConsumeProbe;
  }
  if (std::strcmp(mode, "warp1_valid_probe") == 0 ||
      std::strcmp(mode, "warp1_payload_scale_probe") == 0 ||
      std::strcmp(mode, "task39_warp1_valid_probe") == 0) {
    return C1PairedOutputKernelForkMode::kWarp1ValidProbe;
  }
  if (std::strcmp(mode, "warp1_valid_nowait_probe") == 0 ||
      std::strcmp(mode, "warp1_valid_fullslots_probe") == 0 ||
      std::strcmp(mode, "task39_warp1_valid_nowait_probe") == 0) {
    return C1PairedOutputKernelForkMode::kWarp1ValidNoWaitProbe;
  }
  if (std::strcmp(mode, "warp1_loadmn_valid_probe") == 0 ||
      std::strcmp(mode, "warp1_loadmn_payload_scale_probe") == 0 ||
      std::strcmp(mode, "task39_warp1_loadmn_valid_probe") == 0) {
    return C1PairedOutputKernelForkMode::kWarp1LoadMnValidProbe;
  }
  return C1PairedOutputKernelForkMode::kDefault;
}

enum class C1PairedOutputDebugMode {
  kFull = 0,
  kSkipPayload = 1,
  kSkipScale = 2,
  kSkipScaleAndPayload = 3,
  kSkipScaleStore = 4,
  kScaleComputeOnly = 5,
};

int32_t get_c1_paired_output_debug_mode() {
  const char* mode =
      std::getenv("VLLM_PROJ4_MLP_PARENT_C1_PAIRED_DEBUG");
  if (mode == nullptr) {
    return static_cast<int32_t>(C1PairedOutputDebugMode::kFull);
  }
  if (std::strcmp(mode, "skip_payload") == 0 ||
      std::strcmp(mode, "scale_only") == 0 ||
      std::strcmp(mode, "1") == 0) {
    return static_cast<int32_t>(C1PairedOutputDebugMode::kSkipPayload);
  }
  if (std::strcmp(mode, "skip_scale") == 0 ||
      std::strcmp(mode, "payload_only") == 0 ||
      std::strcmp(mode, "2") == 0) {
    return static_cast<int32_t>(C1PairedOutputDebugMode::kSkipScale);
  }
  if (std::strcmp(mode, "skip_both") == 0 ||
      std::strcmp(mode, "none") == 0 ||
      std::strcmp(mode, "3") == 0) {
    return static_cast<int32_t>(
        C1PairedOutputDebugMode::kSkipScaleAndPayload);
  }
  if (std::strcmp(mode, "skip_scale_store") == 0 ||
      std::strcmp(mode, "4") == 0) {
    return static_cast<int32_t>(C1PairedOutputDebugMode::kSkipScaleStore);
  }
  if (std::strcmp(mode, "scale_compute_only") == 0 ||
      std::strcmp(mode, "5") == 0) {
    return static_cast<int32_t>(C1PairedOutputDebugMode::kScaleComputeOnly);
  }
  return static_cast<int32_t>(C1PairedOutputDebugMode::kFull);
}

template <class ElementCompute, cutlass::FloatRoundStyle RoundStyle>
struct C1PairSiluMul {
  using ElementAux = cutlass::float_e2m1_t;
  static constexpr int kCallbackEpilogueTileM = 64;
  static constexpr int kCallbackEpilogueTileN = 32;
  static constexpr int kCallbackLogicalColsPerEpilogueTile = kCallbackEpilogueTileN / 2;
  static constexpr int kCallbackPackedBytesPerEpilogueRow =
      kCallbackLogicalColsPerEpilogueTile / 2;

  struct SharedStorage {
    cute::array_aligned<uint8_t,
                        kCallbackEpilogueTileM * kCallbackLogicalColsPerEpilogueTile>
        smem_payload;
  };

  struct Arguments {
    ElementCompute alpha = ElementCompute(1);
    ElementCompute const* alpha_ptr = nullptr;
    int32_t intermediate_dtype = 0;  // 0: fp16, 1: bf16
  };
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const&, Arguments const& args, void*) {
    return args;
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const&) {
    return true;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const&,
                                              Arguments const&, void*,
                                              cudaStream_t,
                                              cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE C1PairSiluMul() {}

  CUTLASS_HOST_DEVICE
  C1PairSiluMul(Params const& params, SharedStorage const&)
      : params_ptr(&params) {}

  Params const* params_ptr = nullptr;

  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(
      cutlass::epilogue::fusion::detail::ProducerLoadArgs<Args...> const&) {
    return cutlass::epilogue::fusion::EmptyProducerLoadCallbacks{};
  }

  template <class CoordTensor, class ThrResidue>
  struct ConsumerStoreCallbacks
      : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(CoordTensor tC_cD, ThrResidue residue_tC_cD,
                           Params const* params_ptr)
        : tC_cD(tC_cD),
          residue_tC_cD(residue_tC_cD),
          params_ptr(params_ptr) {}

    CoordTensor tC_cD;
    ThrResidue residue_tC_cD;
    Params const* params_ptr;

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE cutlass::Array<ElementCompute, FragmentSize> visit(
        cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc,
        int epi_v, int epi_m, int epi_n) {
      static_assert(FragmentSize == 4,
                    "SM120 FP4 epilogue fragments are expected to be 2x2.");

      using ConvertAcc = cutlass::NumericArrayConverter<
          ElementCompute, ElementAccumulator, FragmentSize, RoundStyle>;
      ConvertAcc convert_acc{};
      auto acc = convert_acc(frg_acc);

      ElementCompute alpha = params_ptr->alpha_ptr != nullptr
                                 ? *(params_ptr->alpha_ptr)
                                 : params_ptr->alpha;
      int32_t intermediate_dtype = params_ptr->intermediate_dtype;

      cutlass::Array<ElementCompute, FragmentSize> out;
      CUTLASS_PRAGMA_UNROLL
      for (int lane = 0; lane < FragmentSize; ++lane) {
        out[lane] = ElementCompute(0);
      }

      ElementCompute act0;
      ElementCompute act1;
      if (intermediate_dtype == 1) {
        cutlass::NumericConverter<cutlass::bfloat16_t, ElementCompute,
                                  RoundStyle>
            convert{};
        cutlass::NumericConverter<ElementCompute, cutlass::bfloat16_t,
                                  RoundStyle>
            convert_back{};
        auto round_bf16 = [&](ElementCompute x) {
          return convert_back(convert(x));
        };
        ElementCompute gate0 = round_bf16(acc[0] * alpha);
        ElementCompute up0 = round_bf16(acc[1] * alpha);
        ElementCompute gate1 = round_bf16(acc[2] * alpha);
        ElementCompute up1 = round_bf16(acc[3] * alpha);
        act0 = round_bf16(__fdividef(gate0, ElementCompute(1) +
                                                __expf(-gate0)) *
                          up0);
        act1 = round_bf16(__fdividef(gate1, ElementCompute(1) +
                                                __expf(-gate1)) *
                          up1);
      } else {
        cutlass::NumericConverter<cutlass::half_t, ElementCompute, RoundStyle>
            convert{};
        cutlass::NumericConverter<ElementCompute, cutlass::half_t, RoundStyle>
            convert_back{};
        auto round_fp16 = [&](ElementCompute x) {
          return convert_back(convert(x));
        };
        ElementCompute gate0 = round_fp16(acc[0] * alpha);
        ElementCompute up0 = round_fp16(acc[1] * alpha);
        ElementCompute gate1 = round_fp16(acc[2] * alpha);
        ElementCompute up1 = round_fp16(acc[3] * alpha);
        act0 = round_fp16(__fdividef(gate0, ElementCompute(1) +
                                                __expf(-gate0)) *
                          up0);
        act1 = round_fp16(__fdividef(gate1, ElementCompute(1) +
                                                __expf(-gate1)) *
                          up1);
      }

      // SM120 row-blockscale visitor consumes two columns for one row followed
      // by two columns for the next row.
      // Duplicate each row's paired result across the source gate/up columns.
      out[0] = act0;
      out[1] = act0;
      out[2] = act1;
      out[3] = act1;
      return out;
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      cutlass::epilogue::fusion::detail::ConsumerStoreArgs<Args...> const&
          args) {
    return ConsumerStoreCallbacks(args.tCcD, args.residue_tCcD, params_ptr);
  }
};

template <class PairCompute, class BlockScaleStore>
struct C1ActQuantScaleFusion
    : cutlass::epilogue::fusion::Sm90EVT<BlockScaleStore, PairCompute> {
  using Base = cutlass::epilogue::fusion::Sm90EVT<BlockScaleStore, PairCompute>;
  using ElementAux = cutlass::float_e2m1_t;
  using SharedStorage = typename Base::SharedStorage;
  using Params = typename Base::Params;

  struct Arguments {
    typename PairCompute::Arguments pair{};
    typename BlockScaleStore::Arguments sf{};
  };

  CUTLASS_HOST_DEVICE C1ActQuantScaleFusion() {}

  CUTLASS_HOST_DEVICE
  C1ActQuantScaleFusion(Params const& params,
                        SharedStorage const& shared_storage)
      : Base(params, shared_storage) {}

  static constexpr typename Base::Arguments make_base_arguments(
      Arguments const& args) {
    return typename Base::Arguments{args.pair, args.sf};
  }

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const& problem_shape, Arguments const& args,
      void* workspace) {
    return Base::to_underlying_arguments(problem_shape,
                                         make_base_arguments(args), workspace);
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape,
                            Arguments const& args) {
    return Base::can_implement(problem_shape, make_base_arguments(args));
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape,
                                   Arguments const& args) {
    return Base::get_workspace_size(problem_shape, make_base_arguments(args));
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const& problem_shape, Arguments const& args,
      void* workspace, cudaStream_t stream,
      cutlass::CudaHostAdapter* cuda_adapter = nullptr) {
    return Base::initialize_workspace(problem_shape, make_base_arguments(args),
                                      workspace, stream, cuda_adapter);
  }

};

template <class ElementCompute, cutlass::FloatRoundStyle RoundStyle>
struct C1DirectPayloadStore {
  using ElementAux = cutlass::float_e2m1_t;
  static constexpr int kEpilogueTileM = 64;
  static constexpr int kEpilogueTileN = 32;
  static constexpr int kLogicalColsPerEpilogueTile = kEpilogueTileN / 2;
  static constexpr int kPackedBytesPerEpilogueRow =
      kLogicalColsPerEpilogueTile / 2;

  struct SharedStorage {
    cute::array_aligned<uint8_t,
                        kEpilogueTileM * kLogicalColsPerEpilogueTile>
        smem_payload;
  };

  struct Arguments {
    uint8_t* ptr_payload = nullptr;
    int64_t stride_bytes = 0;
    // 0: atomic nibble, 1: shared-memory full byte, 2: register-pair with
    // atomic fallback, 4: next-fragment pair with odd atomic, 5:
    // next-fragment pair and skip odd atomic.
    int32_t store_mode = 0;
  };
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const&, Arguments const& args, void*) {
    return args;
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const& args) {
    return args.ptr_payload != nullptr && args.stride_bytes > 0;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const&, Arguments const&, void*, cudaStream_t,
      cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE C1DirectPayloadStore() {}

  CUTLASS_HOST_DEVICE
  C1DirectPayloadStore(Params const& params,
                       SharedStorage const& shared_storage)
      : params_ptr(&params),
        smem_payload(
            const_cast<uint8_t*>(shared_storage.smem_payload.data())) {}

  Params const* params_ptr = nullptr;
  uint8_t* smem_payload = nullptr;

  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(
      cutlass::epilogue::fusion::detail::ProducerLoadArgs<Args...> const&) {
    return cutlass::epilogue::fusion::EmptyProducerLoadCallbacks{};
  }

  template <class CoordTensor, class ProblemShapeMN, class TiledCopy_>
  struct ConsumerStoreCallbacks
      : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(CoordTensor&& tC_cD_global_,
                           ProblemShapeMN problem_shape_mn_,
                           Params const* params_ptr_,
                           uint8_t* smem_payload_, int64_t tile_m_base_,
                           int64_t tile_n_base_, int thread_idx_)
        : tC_cD_global(cute::forward<CoordTensor>(tC_cD_global_)),
          problem_shape_mn(problem_shape_mn_),
          params_ptr(params_ptr_),
          smem_payload(smem_payload_),
          tile_m_base(tile_m_base_),
          tile_n_base(tile_n_base_),
          thread_idx(thread_idx_) {}

    CoordTensor tC_cD_global;
    ProblemShapeMN problem_shape_mn;
    Params const* params_ptr;
    uint8_t* smem_payload;
    int64_t tile_m_base;
    int64_t tile_n_base;
    int thread_idx;
    static constexpr int NumCollaboratingThreads =
        decltype(size(TiledCopy_{}))::value;
    static constexpr int kCallbackEpilogueTileM =
        C1DirectPayloadStore<ElementCompute, RoundStyle>::kEpilogueTileM;
    static constexpr int kCallbackEpilogueTileN =
        C1DirectPayloadStore<ElementCompute, RoundStyle>::kEpilogueTileN;
    static constexpr int kCallbackLogicalColsPerEpilogueTile =
        C1DirectPayloadStore<ElementCompute,
                             RoundStyle>::kLogicalColsPerEpilogueTile;
    static constexpr int kCallbackPackedBytesPerEpilogueRow =
        C1DirectPayloadStore<ElementCompute,
                             RoundStyle>::kPackedBytesPerEpilogueRow;

    template <typename ElementAccumulator, typename ElementInput,
              int FragmentSize>
    CUTLASS_DEVICE auto visit(
        cutlass::Array<ElementAccumulator, FragmentSize> const&,
        int, int, int,
        cutlass::Array<ElementInput, FragmentSize> const& frg_input) {
      return frg_input;
    }

    CUTLASS_DEVICE static void atomic_store_nibble(uint8_t* byte_ptr,
                                                   uint8_t nibble,
                                                   int shift) {
      uintptr_t const addr = reinterpret_cast<uintptr_t>(byte_ptr);
      auto* word_ptr =
          reinterpret_cast<unsigned int*>(addr & ~uintptr_t{0x3});
      unsigned int const word_shift =
          static_cast<unsigned int>((addr & uintptr_t{0x3}) * 8 + shift);
      unsigned int const mask = 0xFu << word_shift;
      unsigned int old = *word_ptr;
      unsigned int assumed;
      do {
        assumed = old;
        unsigned int desired =
            (assumed & ~mask) |
            ((static_cast<unsigned int>(nibble) & 0xFu) << word_shift);
        old = atomicCAS(word_ptr, assumed, desired);
      } while (old != assumed);
    }

    template <class SmemTensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void reduce(SmemTensor&&, SyncFn const& sync_fn, int epi_m,
                               int epi_n, bool, VTensor visit_results) {
      static constexpr int FragmentSize = 4;
      using ConvertFp4 = cutlass::NumericConverter<
          cutlass::float_e2m1_t, ElementCompute, RoundStyle>;
      ConvertFp4 convert_fp4{};

      auto coords = coalesce(tC_cD_global(_, _, _, epi_m, epi_n));
      auto values = coalesce(visit_results);

      if (params_ptr->store_mode == 4 || params_ptr->store_mode == 5) {
        bool const skip_odd_atomic = params_ptr->store_mode == 5;
        CUTLASS_PRAGMA_UNROLL
        for (int epi_v = 0; epi_v < size(values); ++epi_v) {
          auto frg = values(epi_v);
          CUTLASS_PRAGMA_UNROLL
          for (int lane = 0; lane < FragmentSize; ++lane) {
            int coord_idx = epi_v * FragmentSize + lane;
            auto coord = coords(coord_idx);
            if (!elem_less(coord, problem_shape_mn)) {
              continue;
            }

            int64_t const row = get<0>(coord);
            int64_t const c1_col = get<1>(coord);
            if ((c1_col & 1) != 0) {
              continue;
            }

            int64_t const logical_col = c1_col >> 1;
            int64_t const byte_col = logical_col >> 1;
            int const parity = static_cast<int>(logical_col & 1);
            auto fp4 = convert_fp4(frg[lane]);
            uint8_t const nibble = static_cast<uint8_t>(fp4.raw()) & 0x0f;
            uint8_t* byte_ptr =
                params_ptr->ptr_payload + row * params_ptr->stride_bytes +
                byte_col;

            if (parity == 0) {
              bool found_pair = false;
              if (epi_v + 1 < size(values)) {
                auto other_frg = values(epi_v + 1);
                int other_coord_idx = (epi_v + 1) * FragmentSize + lane;
                auto other_coord = coords(other_coord_idx);
                if (elem_less(other_coord, problem_shape_mn)) {
                  int64_t const other_row = get<0>(other_coord);
                  int64_t const other_c1_col = get<1>(other_coord);
                  int64_t const other_logical_col = other_c1_col >> 1;
                  if (other_row == row && (other_c1_col & 1) == 0 &&
                      (other_logical_col >> 1) == byte_col &&
                      (other_logical_col & 1) == 1) {
                    auto other_fp4 = convert_fp4(other_frg[lane]);
                    uint8_t const pair_nibble =
                        static_cast<uint8_t>(other_fp4.raw()) & 0x0f;
                    *byte_ptr =
                        static_cast<uint8_t>(nibble | (pair_nibble << 4));
                    found_pair = true;
                  }
                }
              }

              if (!found_pair) {
                atomic_store_nibble(byte_ptr, nibble, 0);
              }
            } else if (!skip_odd_atomic) {
              atomic_store_nibble(byte_ptr, nibble, 4);
            }
          }
        }
        return;
      }

      if (params_ptr->store_mode == 2) {
        CUTLASS_PRAGMA_UNROLL
        for (int epi_v = 0; epi_v < size(values); ++epi_v) {
          auto frg = values(epi_v);
          CUTLASS_PRAGMA_UNROLL
          for (int lane = 0; lane < FragmentSize; ++lane) {
            int coord_idx = epi_v * FragmentSize + lane;
            auto coord = coords(coord_idx);
            if (!elem_less(coord, problem_shape_mn)) {
              continue;
            }

            int64_t const row = get<0>(coord);
            int64_t const c1_col = get<1>(coord);
            if ((c1_col & 1) != 0) {
              continue;
            }

            int64_t const logical_col = c1_col >> 1;
            int64_t const byte_col = logical_col >> 1;
            int const parity = static_cast<int>(logical_col & 1);
            auto fp4 = convert_fp4(frg[lane]);
            uint8_t const nibble = static_cast<uint8_t>(fp4.raw()) & 0x0f;

            bool found_pair = false;
            uint8_t pair_nibble = 0;
            CUTLASS_PRAGMA_UNROLL
            for (int other_epi_v = 0; other_epi_v < size(values);
                 ++other_epi_v) {
              auto other_frg = values(other_epi_v);
              CUTLASS_PRAGMA_UNROLL
              for (int other_lane = 0; other_lane < FragmentSize;
                   ++other_lane) {
                int other_coord_idx = other_epi_v * FragmentSize + other_lane;
                auto other_coord = coords(other_coord_idx);
                if (!elem_less(other_coord, problem_shape_mn)) {
                  continue;
                }

                int64_t const other_row = get<0>(other_coord);
                int64_t const other_c1_col = get<1>(other_coord);
                if ((other_c1_col & 1) != 0 || other_row != row) {
                  continue;
                }

                int64_t const other_logical_col = other_c1_col >> 1;
                if ((other_logical_col >> 1) != byte_col ||
                    static_cast<int>(other_logical_col & 1) == parity) {
                  continue;
                }

                auto other_fp4 = convert_fp4(other_frg[other_lane]);
                pair_nibble =
                    static_cast<uint8_t>(other_fp4.raw()) & 0x0f;
                found_pair = true;
              }
            }

            uint8_t* byte_ptr =
                params_ptr->ptr_payload + row * params_ptr->stride_bytes +
                byte_col;
            if (found_pair) {
              if (parity == 0) {
                *byte_ptr = static_cast<uint8_t>(nibble | (pair_nibble << 4));
              }
              continue;
            }

            int const shift = parity * 4;
            atomic_store_nibble(byte_ptr, nibble, shift);
          }
        }
        return;
      }

      if (params_ptr->store_mode == 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int epi_v = 0; epi_v < size(values); ++epi_v) {
          auto frg = values(epi_v);
          CUTLASS_PRAGMA_UNROLL
          for (int lane = 0; lane < FragmentSize; ++lane) {
            int coord_idx = epi_v * FragmentSize + lane;
            auto coord = coords(coord_idx);
            if (elem_less(coord, problem_shape_mn)) {
              int64_t const row = get<0>(coord);
              int64_t const c1_col = get<1>(coord);
              if ((c1_col & 1) == 0) {
                int const local_row =
                    static_cast<int>(row - tile_m_base -
                                     epi_m * kCallbackEpilogueTileM);
                int const local_logical_col =
                    static_cast<int>((c1_col - tile_n_base -
                                      epi_n * kCallbackEpilogueTileN) >>
                                     1);
                if (local_row >= 0 && local_row < kCallbackEpilogueTileM &&
                    local_logical_col >= 0 &&
                    local_logical_col < kCallbackLogicalColsPerEpilogueTile) {
                  auto fp4 = convert_fp4(frg[lane]);
                  smem_payload[local_row * kCallbackLogicalColsPerEpilogueTile +
                               local_logical_col] =
                      static_cast<uint8_t>(fp4.raw()) & 0x0f;
                }
              }
            }
          }
        }

        sync_fn();

        constexpr int total_packed_bytes =
            kCallbackEpilogueTileM * kCallbackPackedBytesPerEpilogueRow;
        int64_t const row_base = tile_m_base + epi_m * kCallbackEpilogueTileM;
        int64_t const c1_col_base = tile_n_base + epi_n * kCallbackEpilogueTileN;
        int64_t const problem_m = get<0>(problem_shape_mn);
        int64_t const problem_n = get<1>(problem_shape_mn);
        for (int idx = thread_idx; idx < total_packed_bytes;
             idx += NumCollaboratingThreads) {
          int const local_row = idx / kCallbackPackedBytesPerEpilogueRow;
          int const byte_in_epi = idx - local_row * kCallbackPackedBytesPerEpilogueRow;
          int64_t const row = row_base + local_row;
          int64_t const c1_col0 = c1_col_base + byte_in_epi * 4;
          if (row < problem_m && c1_col0 + 2 < problem_n) {
            uint8_t const lo =
                smem_payload[local_row * kCallbackLogicalColsPerEpilogueTile +
                             byte_in_epi * 2] &
                0x0f;
            uint8_t const hi =
                smem_payload[local_row * kCallbackLogicalColsPerEpilogueTile +
                             byte_in_epi * 2 + 1] &
                0x0f;
            int64_t const byte_col = c1_col0 >> 2;
            params_ptr->ptr_payload[row * params_ptr->stride_bytes +
                                    byte_col] =
                static_cast<uint8_t>(lo | (hi << 4));
          }
        }

        sync_fn();
        return;
      }

      CUTLASS_PRAGMA_UNROLL
      for (int epi_v = 0; epi_v < size(values); ++epi_v) {
        auto frg = values(epi_v);
        CUTLASS_PRAGMA_UNROLL
        for (int lane = 0; lane < FragmentSize; ++lane) {
          int coord_idx = epi_v * FragmentSize + lane;
          auto coord = coords(coord_idx);
          if (elem_less(coord, problem_shape_mn)) {
            int64_t const row = get<0>(coord);
            int64_t const c1_col = get<1>(coord);
            if ((c1_col & 1) == 0) {
              int64_t const logical_col = c1_col >> 1;
              int64_t const byte_col = logical_col >> 1;
              int const shift = static_cast<int>((logical_col & 1) * 4);
              auto fp4 = convert_fp4(frg[lane]);
              uint8_t const nibble = static_cast<uint8_t>(fp4.raw()) & 0x0f;
              atomic_store_nibble(
                  params_ptr->ptr_payload + row * params_ptr->stride_bytes +
                      byte_col,
                  nibble, shift);
            }
          }
        }
      }
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      cutlass::epilogue::fusion::detail::ConsumerStoreArgs<Args...> const&
          args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    auto problem_shape_mn = make_shape(M, N);
    Tensor mD_crd = make_identity_tensor(problem_shape_mn);
    Tensor cD_mn = local_tile(mD_crd, take<0, 2>(args.tile_shape_mnk),
                              make_coord(m, n));
    Tensor tC_cD_global =
        cutlass::epilogue::fusion::sm90_partition_for_epilogue<ReferenceSrc>(
            cD_mn, args.epi_tile, args.tiled_copy, args.thread_idx);

    int64_t const tile_m_base =
        static_cast<int64_t>(m) *
        static_cast<int64_t>(size<0>(args.tile_shape_mnk));
    int64_t const tile_n_base =
        static_cast<int64_t>(n) *
        static_cast<int64_t>(size<1>(args.tile_shape_mnk));

    return ConsumerStoreCallbacks<decltype(tC_cD_global),
                                  decltype(problem_shape_mn),
                                  decltype(args.tiled_copy)>(
        cute::move(tC_cD_global), problem_shape_mn, params_ptr, smem_payload,
        tile_m_base, tile_n_base, args.thread_idx);
  }
};

template <class PairCompute, class BlockScaleStore, class DirectPayloadStore>
struct C1ActQuantDirectStoreFusion
    : cutlass::epilogue::fusion::Sm90EVT<
          DirectPayloadStore,
          cutlass::epilogue::fusion::Sm90EVT<BlockScaleStore, PairCompute>> {
  using ScaleTree =
      cutlass::epilogue::fusion::Sm90EVT<BlockScaleStore, PairCompute>;
  using Base =
      cutlass::epilogue::fusion::Sm90EVT<DirectPayloadStore, ScaleTree>;
  using ElementAux = cutlass::float_e2m1_t;
  using SharedStorage = typename Base::SharedStorage;
  using Params = typename Base::Params;

  struct Arguments {
    typename PairCompute::Arguments pair{};
    typename BlockScaleStore::Arguments sf{};
    typename DirectPayloadStore::Arguments direct{};
  };

  CUTLASS_HOST_DEVICE C1ActQuantDirectStoreFusion() {}

  CUTLASS_HOST_DEVICE
  C1ActQuantDirectStoreFusion(Params const& params,
                              SharedStorage const& shared_storage)
      : Base(params, shared_storage) {}

  static constexpr typename Base::Arguments make_base_arguments(
      Arguments const& args) {
    return typename Base::Arguments{
        typename ScaleTree::Arguments{args.pair, args.sf}, args.direct};
  }

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const& problem_shape, Arguments const& args,
      void* workspace) {
    return Base::to_underlying_arguments(problem_shape,
                                         make_base_arguments(args), workspace);
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape,
                            Arguments const& args) {
    return Base::can_implement(problem_shape, make_base_arguments(args));
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape,
                                   Arguments const& args) {
    return Base::get_workspace_size(problem_shape, make_base_arguments(args));
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const& problem_shape, Arguments const& args,
      void* workspace, cudaStream_t stream,
      cutlass::CudaHostAdapter* cuda_adapter = nullptr) {
    return Base::initialize_workspace(problem_shape, make_base_arguments(args),
                                      workspace, stream, cuda_adapter);
  }

};

template <int SFVecSize, class EpilogueTile, class CtaTileShapeMNK,
          int FragmentSize, class ElementOutput, class ElementCompute,
          class ElementBlockScaleFactor,
          cutlass::FloatRoundStyle RoundStyle =
              cutlass::FloatRoundStyle::round_to_nearest>
struct C1PairedOutputStore {
  static_assert(size<1>(EpilogueTile{}) % SFVecSize == 0,
                "EpilogueTileN should be divisible by SFVecSize");
  static_assert(size<1>(EpilogueTile{}) / SFVecSize == 1 ||
                    size<1>(EpilogueTile{}) / SFVecSize == 2 ||
                    size<1>(EpilogueTile{}) / SFVecSize == 4 ||
                    size<1>(EpilogueTile{}) / SFVecSize == 8,
                "Possible store in interleaved 4B aligned format");

  static constexpr int NumWarpgroups = 2;
  static constexpr int NumSyncWarps =
      cutlass::NumWarpsPerWarpGroup * NumWarpgroups;
  static constexpr int NumQuadsPerWarp = 8;
  static constexpr int NumSyncQuads = NumSyncWarps * NumQuadsPerWarp;
  static constexpr int kEpilogueTileM = 64;
  static constexpr int kEpilogueTileN = 32;
  static constexpr int kLogicalColsPerEpilogueTile = kEpilogueTileN / 2;
  static constexpr int kPackedBytesPerEpilogueRow =
      kLogicalColsPerEpilogueTile / 2;

  struct SharedStorage {
    cute::array_aligned<ElementCompute, NumSyncQuads> smem_aux;
    cute::array_aligned<uint8_t,
                        kEpilogueTileM * kLogicalColsPerEpilogueTile>
        smem_payload;
  };

  using NormalConstStrideMNL = cute::Stride<_0, _0, int64_t>;
  struct Arguments {
    ElementBlockScaleFactor* ptr_scale_factor = nullptr;
    ElementCompute const* norm_constant_ptr = nullptr;
    NormalConstStrideMNL norm_constant_stride = {};
    uint8_t* ptr_payload = nullptr;
    int64_t stride_bytes = 0;
    // Timing-only selector used by task22 to decompose paired-output visitor
    // cost. Nonzero modes intentionally do not preserve the output contract.
    int32_t debug_timing_mode = 0;
  };
  using Params = Arguments;

  using UnderlyingElementBlockScaleFactor =
      cute::remove_pointer_t<ElementBlockScaleFactor>;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const&, Arguments const& args, void*) {
    return args;
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape,
                            Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    return args.ptr_scale_factor != nullptr && args.norm_constant_ptr != nullptr &&
           args.ptr_payload != nullptr && args.stride_bytes > 0 &&
           (N % SFVecSize == 0);
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const&, Arguments const&, void*, cudaStream_t,
      cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE C1PairedOutputStore() {}

  CUTLASS_HOST_DEVICE
  C1PairedOutputStore(Params const& params,
                      SharedStorage const& shared_storage)
      : params_ptr(&params),
        smem_aux(const_cast<ElementCompute*>(
            shared_storage.smem_aux.data())),
        smem_payload(
            const_cast<uint8_t*>(shared_storage.smem_payload.data())) {}

  Params const* params_ptr = nullptr;
  ElementCompute* smem_aux = nullptr;
  uint8_t* smem_payload = nullptr;

  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(
      cutlass::epilogue::fusion::detail::ProducerLoadArgs<Args...> const&) {
    return cutlass::epilogue::fusion::EmptyProducerLoadCallbacks{};
  }

  template <class RTensor, class GTensor, class STensor, class CoordGTensor,
            class ThrResidue, class TileCoordMN, class TiledCopy_,
            class CoordTensor, class ProblemShapeMN>
  struct ConsumerStoreCallbacks
      : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(RTensor&& tC_rSFD_, GTensor&& tC_gSFD_,
                           STensor&& sAmaxs_, CoordGTensor tC_cSFD_,
                           ThrResidue residue_tC_cSFD_,
                           Params const* params_ptr_,
                           TileCoordMN tile_coord_mn_,
                           ElementCompute norm_constant_,
                           ElementCompute norm_constant_scaled_down_,
                           int thread_idx_, TiledCopy_ const&,
                           CoordTensor&& tC_cD_global_,
                           ProblemShapeMN problem_shape_mn_,
                           uint8_t* smem_payload_, int64_t tile_m_base_,
                           int64_t tile_n_base_)
        : tC_rSFD(cute::forward<RTensor>(tC_rSFD_)),
          tC_gSFD(cute::forward<GTensor>(tC_gSFD_)),
          sAmaxs(cute::forward<STensor>(sAmaxs_)),
          tC_cSFD(tC_cSFD_),
          residue_tC_cSFD(residue_tC_cSFD_),
          params_ptr(params_ptr_),
          norm_constant(norm_constant_),
          norm_constant_scaled_down(norm_constant_scaled_down_),
          tile_coord_mn(tile_coord_mn_),
          thread_idx(thread_idx_),
          tC_cD_global(cute::forward<CoordTensor>(tC_cD_global_)),
          problem_shape_mn(problem_shape_mn_),
          smem_payload(smem_payload_),
          tile_m_base(tile_m_base_),
          tile_n_base(tile_n_base_) {}

    RTensor tC_rSFD;
    GTensor tC_gSFD;
    STensor sAmaxs;
    CoordGTensor tC_cSFD;
    ThrResidue residue_tC_cSFD;
    Params const* params_ptr;
    ElementCompute norm_constant;
    ElementCompute norm_constant_scaled_down;
    TileCoordMN tile_coord_mn;
    int thread_idx;
    CoordTensor tC_cD_global;
    ProblemShapeMN problem_shape_mn;
    uint8_t* smem_payload;
    int64_t tile_m_base;
    int64_t tile_n_base;

    static constexpr int NumCollaboratingThreads =
        decltype(size(TiledCopy_{}))::value;
    static_assert(NumCollaboratingThreads % cutlass::NumThreadsPerWarpGroup ==
                  0);
    static constexpr int NumCollaboratingWarpGroups =
        NumCollaboratingThreads / cutlass::NumThreadsPerWarpGroup;
    static_assert(NumCollaboratingWarpGroups == 1 ||
                      NumCollaboratingWarpGroups == 2,
                  "SM120 epilogue currently only supports one or two warp "
                  "groups collaborating.");

    template <class ElementAccumulator, class ElementInput>
    CUTLASS_DEVICE auto visit(
        cutlass::Array<ElementAccumulator, FragmentSize> const&, int, int, int,
        cutlass::Array<ElementInput, FragmentSize> const& frg_input) {
      return frg_input;
    }

    template <class SmemTensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void reduce(SmemTensor&&, SyncFn const& sync_fn, int epi_m,
                               int epi_n, bool, VTensor visit_results) {
      static constexpr int ColsPerThreadAccFrag = 2;
      static constexpr int RowsPerThreadAccFrag = 2;
      static_assert(FragmentSize ==
                    (ColsPerThreadAccFrag * RowsPerThreadAccFrag));

      static constexpr int NumThreadsPerQuad = 4;
      static_assert(SFVecSize == 16 || SFVecSize == 32 || SFVecSize == 64,
                    "SF vector size must be either 16, 32 or 64.");
      constexpr int WarpsPerSF = SFVecSize / 16;
      static_assert(WarpsPerSF == 1 || WarpsPerSF == 2 || WarpsPerSF == 4,
                    "Only one, two or four warps are allowed in reduction.");
      constexpr bool IsInterWarpReductionNeeded = (WarpsPerSF != 1);
      static constexpr int AccFragsPerSF =
          SFVecSize / (ColsPerThreadAccFrag * NumThreadsPerQuad * WarpsPerSF);
      static_assert(size<2>(visit_results) % AccFragsPerSF == 0,
                    "Fragments along N mode must be a multiple of the number "
                    "of accumulator fragments needed per SF");

      int32_t const debug_timing_mode = params_ptr->debug_timing_mode;
      if (debug_timing_mode ==
          static_cast<int32_t>(
              C1PairedOutputDebugMode::kSkipScaleAndPayload)) {
        return;
      }
      bool const run_scale_compute =
          debug_timing_mode !=
          static_cast<int32_t>(C1PairedOutputDebugMode::kSkipScale);
      bool const write_scale_output =
          run_scale_compute &&
          debug_timing_mode !=
              static_cast<int32_t>(C1PairedOutputDebugMode::kSkipScaleStore) &&
          debug_timing_mode !=
              static_cast<int32_t>(C1PairedOutputDebugMode::kScaleComputeOnly);
      bool const run_payload =
          debug_timing_mode !=
              static_cast<int32_t>(C1PairedOutputDebugMode::kSkipPayload) &&
          debug_timing_mode !=
              static_cast<int32_t>(C1PairedOutputDebugMode::kScaleComputeOnly);

      auto warp_idx = thread_idx / cutlass::NumThreadsPerWarp;
      auto warpgroup_idx = thread_idx / cutlass::NumThreadsPerWarpGroup;
      auto quad_idx_in_warp =
          (thread_idx % cutlass::NumThreadsPerWarp) / NumThreadsPerQuad;
      auto thread_idx_in_quad = thread_idx % NumThreadsPerQuad;

      cutlass::maximum_absolute_value_reduction<ElementCompute, true> amax_op;
      cutlass::multiplies<ElementCompute> mul;

      auto synchronize = [&]() {
        cutlass::arch::NamedBarrier::sync(
            NumCollaboratingThreads,
            cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      };

      if (run_scale_compute) {
        Tensor tC_rSFD_flt = filter_zeros(tC_rSFD);

        CUTLASS_PRAGMA_UNROLL
        for (int sf_id = 0; sf_id < size(tC_rSFD_flt); ++sf_id) {
          auto coord = idx2crd(sf_id, tC_rSFD_flt.shape());
          auto row_in_acc = get<0, 1, 1>(coord);
          auto row = crd2idx(get<1>(coord), get<1>(tC_rSFD_flt.shape()));
          auto sf = crd2idx(get<2>(coord), get<2>(tC_rSFD_flt.shape()));

          ElementCompute amax{0};
          auto acc_frag_row = row_in_acc * RowsPerThreadAccFrag;
          auto acc_frag_start_for_sf = sf * AccFragsPerSF;
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < AccFragsPerSF; ++i) {
            auto acc_frg = visit_results(0, row, acc_frag_start_for_sf + i);
            amax = amax_op(amax, acc_frg[acc_frag_row]);
            amax = amax_op(amax, acc_frg[acc_frag_row + 1]);
          }

          CUTLASS_PRAGMA_UNROLL
          for (int i = 1; i < 3; ++i) {
            auto amax_other = __shfl_xor_sync(0xffffffff, amax, i);
            amax = amax_op(amax, amax_other);
          }

          if constexpr (IsInterWarpReductionNeeded) {
            if (thread_idx_in_quad == 0) {
              sAmaxs(quad_idx_in_warp, warp_idx) = amax;
            }
            synchronize();

            if constexpr (WarpsPerSF == 4) {
              if constexpr (NumCollaboratingWarpGroups == 2) {
                auto amax_other2 = sAmaxs(quad_idx_in_warp, warp_idx ^ 2);
                auto amax_other4 = sAmaxs(quad_idx_in_warp, warp_idx ^ 4);
                auto amax_other6 = sAmaxs(quad_idx_in_warp, warp_idx ^ 6);
                synchronize();
                amax = amax_op(amax, amax_other2);
                amax = amax_op(amax, amax_other4);
                amax = amax_op(amax, amax_other6);
              } else {
                static_assert(cutlass::detail::dependent_false<TiledCopy_>,
                              "Unsupported warp layout.");
              }
            } else if constexpr (WarpsPerSF == 2) {
              auto amax_other = sAmaxs(
                  quad_idx_in_warp,
                  warp_idx ^ (1 << NumCollaboratingWarpGroups));
              synchronize();
              amax = amax_op(amax, amax_other);
            }
          }

          ElementCompute pvscale = mul(amax, norm_constant_scaled_down);
          UnderlyingElementBlockScaleFactor qpvscale =
              cutlass::NumericConverter<UnderlyingElementBlockScaleFactor,
                                        ElementCompute>{}(pvscale);
          tC_rSFD_flt(coord) = qpvscale;

          ElementCompute qpvscale_rcp = [&]() {
            if constexpr (cute::is_same_v<UnderlyingElementBlockScaleFactor,
                                          cutlass::float_ue8m0_t>) {
              auto e8m0_qpvscale_rcp =
                  cutlass::reciprocal_approximate<
                      UnderlyingElementBlockScaleFactor>{}(qpvscale);
              return cutlass::NumericConverter<
                  ElementCompute, UnderlyingElementBlockScaleFactor>{}(
                  e8m0_qpvscale_rcp);
            } else {
              auto qpvscale_up = cutlass::NumericConverter<
                  ElementCompute, UnderlyingElementBlockScaleFactor>{}(
                  qpvscale);
              return cutlass::reciprocal_approximate_ftz<
                  decltype(qpvscale_up)>{}(qpvscale_up);
            }
          }();

          ElementCompute acc_scale = mul(norm_constant, qpvscale_rcp);
          acc_scale =
              cutlass::minimum_with_nan_propagation<ElementCompute>{}(
                  acc_scale,
                  cutlass::platform::numeric_limits<ElementCompute>::max());

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < AccFragsPerSF; ++i) {
            auto acc_frag = visit_results(0, row, acc_frag_start_for_sf + i);
            visit_results(0, row, acc_frag_start_for_sf + i)[acc_frag_row] =
                mul(acc_frag[acc_frag_row], acc_scale);
            visit_results(0, row, acc_frag_start_for_sf + i)[acc_frag_row + 1] =
                mul(acc_frag[acc_frag_row + 1], acc_scale);
          }
        }

        if (write_scale_output) {
          bool write_sf = (thread_idx_in_quad == 0);
          if constexpr (NumCollaboratingWarpGroups == 2) {
            if constexpr (IsInterWarpReductionNeeded) {
              write_sf &= warp_idx < cutlass::NumWarpsPerWarpGroup;
            }
          } else {
            if constexpr (IsInterWarpReductionNeeded) {
              write_sf &= ((warp_idx < 2) ||
                           (warpgroup_idx == 1 && warp_idx < 6));
            }
          }

          if (write_sf &&
              elem_less(tC_cSFD(_0{}, _0{}, _0{}, epi_m, epi_n),
                        residue_tC_cSFD)) {
            copy_aligned(tC_rSFD,
                         tC_gSFD(_, _, _, _0{}, _0{},
                                 get<0>(tile_coord_mn) + epi_m,
                                 get<1>(tile_coord_mn) + epi_n));
          }
        }
      }

      if (!run_payload) {
        return;
      }

      using ConvertFp4 =
          cutlass::NumericConverter<cutlass::float_e2m1_t, ElementCompute,
                                    RoundStyle>;
      ConvertFp4 convert_fp4{};
      auto coords = coalesce(tC_cD_global(_, _, _, epi_m, epi_n));
      auto values = coalesce(visit_results);

      CUTLASS_PRAGMA_UNROLL
      for (int epi_v = 0; epi_v < size(values); ++epi_v) {
        auto frg = values(epi_v);
        CUTLASS_PRAGMA_UNROLL
        for (int lane = 0; lane < FragmentSize; ++lane) {
          int coord_idx = epi_v * FragmentSize + lane;
          auto coord = coords(coord_idx);
          if (elem_less(coord, problem_shape_mn)) {
            int64_t const row = get<0>(coord);
            int64_t const c1_col = get<1>(coord);
            if ((c1_col & 1) == 0) {
              int const local_row =
                  static_cast<int>(row - tile_m_base -
                                   epi_m * kEpilogueTileM);
              int const local_logical_col =
                  static_cast<int>((c1_col - tile_n_base -
                                    epi_n * kEpilogueTileN) >>
                                   1);
              if (local_row >= 0 && local_row < kEpilogueTileM &&
                  local_logical_col >= 0 &&
                  local_logical_col < kLogicalColsPerEpilogueTile) {
                auto fp4 = convert_fp4(frg[lane]);
                smem_payload[local_row * kLogicalColsPerEpilogueTile +
                             local_logical_col] =
                    static_cast<uint8_t>(fp4.raw()) & 0x0f;
              }
            }
          }
        }
      }

      sync_fn();

      constexpr int total_packed_bytes =
          kEpilogueTileM * kPackedBytesPerEpilogueRow;
      int64_t const row_base = tile_m_base + epi_m * kEpilogueTileM;
      int64_t const c1_col_base = tile_n_base + epi_n * kEpilogueTileN;
      int64_t const problem_m = get<0>(problem_shape_mn);
      int64_t const problem_n = get<1>(problem_shape_mn);
      for (int idx = thread_idx; idx < total_packed_bytes;
           idx += NumCollaboratingThreads) {
        int const local_row = idx / kPackedBytesPerEpilogueRow;
        int const byte_in_epi =
            idx - local_row * kPackedBytesPerEpilogueRow;
        int64_t const row = row_base + local_row;
        int64_t const c1_col0 = c1_col_base + byte_in_epi * 4;
        if (row < problem_m && c1_col0 + 2 < problem_n) {
          uint8_t const lo =
              smem_payload[local_row * kLogicalColsPerEpilogueTile +
                           byte_in_epi * 2] &
              0x0f;
          uint8_t const hi =
              smem_payload[local_row * kLogicalColsPerEpilogueTile +
                           byte_in_epi * 2 + 1] &
              0x0f;
          int64_t const byte_col = c1_col0 >> 2;
          params_ptr->ptr_payload[row * params_ptr->stride_bytes + byte_col] =
              static_cast<uint8_t>(lo | (hi << 4));
        }
      }

      sync_fn();
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      cutlass::epilogue::fusion::detail::ConsumerStoreArgs<Args...> const&
          args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    using Sm1xxBlockScaledOutputConfig =
        cutlass::detail::Sm1xxBlockScaledOutputConfig<SFVecSize>;
    UnderlyingElementBlockScaleFactor* ptr_scale_factor = nullptr;
    if constexpr (!cute::is_same_v<UnderlyingElementBlockScaleFactor,
                                   ElementBlockScaleFactor>) {
      ptr_scale_factor = params_ptr->ptr_scale_factor[l];
      l = 0;
    } else {
      ptr_scale_factor = params_ptr->ptr_scale_factor;
    }

    auto epi_tile_mn =
        shape<1>(zipped_divide(make_layout(take<0, 2>(args.tile_shape_mnk)),
                               args.epi_tile));
    Tensor mSFD = make_tensor(
        make_gmem_ptr(ptr_scale_factor),
        Sm1xxBlockScaledOutputConfig::tile_atom_to_shape_SFD(
            args.problem_shape_mnkl));

    Tensor gSFD =
        local_tile(mSFD, args.epi_tile, make_coord(_, _, l));
    Tensor tCgSFD =
        cutlass::epilogue::fusion::sm90_partition_for_epilogue<ReferenceSrc>(
            gSFD, args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tCrSFD = make_tensor_like<UnderlyingElementBlockScaleFactor>(
        take<0, 3>(cute::layout(tCgSFD)));

    auto tile_coord_mn =
        make_coord(m * size<0>(epi_tile_mn), n * size<1>(epi_tile_mn));

    Tensor mNormConst = make_tensor(
        make_gmem_ptr(params_ptr->norm_constant_ptr),
        make_layout(make_shape(M, N, L), params_ptr->norm_constant_stride));
    ElementCompute norm_constant = mNormConst(_0{}, _0{}, l);
    ElementCompute fp_max =
        ElementCompute(cutlass::platform::numeric_limits<ElementOutput>::max());
    ElementCompute scale_down_factor =
        cutlass::reciprocal_approximate_ftz<ElementCompute>{}(fp_max);
    ElementCompute norm_constant_scaled_down =
        cutlass::multiplies<ElementCompute>{}(norm_constant,
                                              scale_down_factor);

    Tensor sAmaxs =
        make_tensor(make_smem_ptr(smem_aux),
                    make_layout(make_shape(Int<NumQuadsPerWarp>{},
                                           Int<NumSyncWarps>{})));

    auto problem_shape_mn = make_shape(M, N);
    Tensor mD_crd = make_identity_tensor(problem_shape_mn);
    Tensor cD_mn = local_tile(mD_crd, take<0, 2>(args.tile_shape_mnk),
                              make_coord(m, n));
    Tensor tC_cD_global =
        cutlass::epilogue::fusion::sm90_partition_for_epilogue<ReferenceSrc>(
            cD_mn, args.epi_tile, args.tiled_copy, args.thread_idx);
    int64_t const tile_m_base =
        static_cast<int64_t>(m) *
        static_cast<int64_t>(size<0>(args.tile_shape_mnk));
    int64_t const tile_n_base =
        static_cast<int64_t>(n) *
        static_cast<int64_t>(size<1>(args.tile_shape_mnk));

    return ConsumerStoreCallbacks<
        decltype(tCrSFD), decltype(tCgSFD), decltype(sAmaxs),
        decltype(args.tCcD), decltype(args.residue_tCcD),
        decltype(tile_coord_mn), decltype(args.tiled_copy),
        decltype(tC_cD_global), decltype(problem_shape_mn)>(
        cute::move(tCrSFD), cute::move(tCgSFD), cute::move(sAmaxs),
        args.tCcD, args.residue_tCcD, params_ptr, tile_coord_mn,
        norm_constant, norm_constant_scaled_down, args.thread_idx,
        args.tiled_copy, cute::move(tC_cD_global), problem_shape_mn,
        smem_payload, tile_m_base, tile_n_base);
  }
};

template <int SFVecSize, class EpilogueTile, class CtaTileShapeMNK,
          int FragmentSize, class ElementOutput, class ElementCompute,
          class ElementBlockScaleFactor,
          cutlass::FloatRoundStyle RoundStyle =
              cutlass::FloatRoundStyle::round_to_nearest>
struct C1RawPublishProbeStore {
  static_assert(size<1>(EpilogueTile{}) % SFVecSize == 0,
                "EpilogueTileN should be divisible by SFVecSize");

  static constexpr int kEpilogueTileM = 64;
  static constexpr int kEpilogueTileN = 32;
  static constexpr int kLogicalColsPerEpilogueTile = kEpilogueTileN / 2;

  struct SharedStorage {
    cute::array_aligned<ElementCompute,
                        kEpilogueTileM * kLogicalColsPerEpilogueTile>
        smem_raw;
  };

  using NormalConstStrideMNL = cute::Stride<_0, _0, int64_t>;
  struct Arguments {
    ElementBlockScaleFactor* ptr_scale_factor = nullptr;
    ElementCompute const* norm_constant_ptr = nullptr;
    NormalConstStrideMNL norm_constant_stride = {};
    uint8_t* ptr_payload = nullptr;
    int64_t stride_bytes = 0;
    int32_t debug_timing_mode = 0;
  };
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const&, Arguments const& args, void*) {
    return args;
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape,
                            Arguments const&) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    return (N % SFVecSize == 0);
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const&, Arguments const&, void*, cudaStream_t,
      cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE C1RawPublishProbeStore() {}

  CUTLASS_HOST_DEVICE
  C1RawPublishProbeStore(Params const& params,
                         SharedStorage const& shared_storage)
      : params_ptr(&params),
        smem_raw(const_cast<ElementCompute*>(
            shared_storage.smem_raw.data())) {}

  Params const* params_ptr = nullptr;
  ElementCompute* smem_raw = nullptr;

  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(
      cutlass::epilogue::fusion::detail::ProducerLoadArgs<Args...> const&) {
    return cutlass::epilogue::fusion::EmptyProducerLoadCallbacks{};
  }

  template <class CoordTensor, class ProblemShapeMN, class TiledCopy_>
  struct ConsumerStoreCallbacks
      : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(CoordTensor&& tC_cD_global_,
                           ProblemShapeMN problem_shape_mn_,
                           int thread_idx_, TiledCopy_ const&,
                           ElementCompute* smem_raw_,
                           int64_t tile_m_base_, int64_t tile_n_base_)
        : tC_cD_global(cute::forward<CoordTensor>(tC_cD_global_)),
          problem_shape_mn(problem_shape_mn_),
          thread_idx(thread_idx_),
          smem_raw(smem_raw_),
          tile_m_base(tile_m_base_),
          tile_n_base(tile_n_base_) {}

    CoordTensor tC_cD_global;
    ProblemShapeMN problem_shape_mn;
    int thread_idx;
    ElementCompute* smem_raw;
    int64_t tile_m_base;
    int64_t tile_n_base;

    static constexpr int NumCollaboratingThreads =
        decltype(size(TiledCopy_{}))::value;

    template <class ElementAccumulator, class ElementInput>
    CUTLASS_DEVICE auto visit(
        cutlass::Array<ElementAccumulator, FragmentSize> const&, int, int, int,
        cutlass::Array<ElementInput, FragmentSize> const& frg_input) {
      return frg_input;
    }

    template <class SmemTensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void reduce(SmemTensor&&, SyncFn const& sync_fn, int epi_m,
                               int epi_n, bool, VTensor visit_results) {
      auto coords = coalesce(tC_cD_global(_, _, _, epi_m, epi_n));
      auto values = coalesce(visit_results);
      volatile ElementCompute* raw_smem =
          reinterpret_cast<volatile ElementCompute*>(smem_raw);

      CUTLASS_PRAGMA_UNROLL
      for (int epi_v = 0; epi_v < size(values); ++epi_v) {
        auto frg = values(epi_v);
        CUTLASS_PRAGMA_UNROLL
        for (int lane = 0; lane < FragmentSize; ++lane) {
          int coord_idx = epi_v * FragmentSize + lane;
          auto coord = coords(coord_idx);
          if (elem_less(coord, problem_shape_mn)) {
            int64_t const row = get<0>(coord);
            int64_t const c1_col = get<1>(coord);
            if ((c1_col & 1) == 0) {
              int const local_row =
                  static_cast<int>(row - tile_m_base -
                                   epi_m * kEpilogueTileM);
              int const local_logical_col =
                  static_cast<int>((c1_col - tile_n_base -
                                    epi_n * kEpilogueTileN) >>
                                   1);
              if (local_row >= 0 && local_row < kEpilogueTileM &&
                  local_logical_col >= 0 &&
                  local_logical_col < kLogicalColsPerEpilogueTile) {
                raw_smem[local_row * kLogicalColsPerEpilogueTile +
                         local_logical_col] = frg[lane];
              }
            }
          }
        }
      }

      sync_fn();
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      cutlass::epilogue::fusion::detail::ConsumerStoreArgs<Args...> const&
          args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    auto problem_shape_mn = make_shape(M, N);
    Tensor mD_crd = make_identity_tensor(problem_shape_mn);
    Tensor cD_mn = local_tile(mD_crd, take<0, 2>(args.tile_shape_mnk),
                              make_coord(m, n));
    Tensor tC_cD_global =
        cutlass::epilogue::fusion::sm90_partition_for_epilogue<ReferenceSrc>(
            cD_mn, args.epi_tile, args.tiled_copy, args.thread_idx);
    int64_t const tile_m_base =
        static_cast<int64_t>(m) *
        static_cast<int64_t>(size<0>(args.tile_shape_mnk));
    int64_t const tile_n_base =
        static_cast<int64_t>(n) *
        static_cast<int64_t>(size<1>(args.tile_shape_mnk));

    return ConsumerStoreCallbacks<
        decltype(tC_cD_global), decltype(problem_shape_mn),
        decltype(args.tiled_copy)>(
        cute::move(tC_cD_global), problem_shape_mn, args.thread_idx,
        args.tiled_copy, smem_raw, tile_m_base, tile_n_base);
  }
};

template <int SFVecSize, class EpilogueTile, class CtaTileShapeMNK,
          int FragmentSize, class ElementOutput, class ElementCompute,
          class ElementBlockScaleFactor,
          cutlass::FloatRoundStyle RoundStyle =
              cutlass::FloatRoundStyle::round_to_nearest>
struct C1LoadMnConsumeProbeStore {
  static_assert(size<1>(EpilogueTile{}) % SFVecSize == 0,
                "EpilogueTileN should be divisible by SFVecSize");

  static constexpr int kEpilogueTileM = 64;
  static constexpr int kEpilogueTileN = 32;
  static constexpr int kLogicalColsPerEpilogueTile = kEpilogueTileN / 2;
  static constexpr int kValuesPerEpilogueTile =
      kEpilogueTileM * kLogicalColsPerEpilogueTile;
  static constexpr int kEpilogueTilesM =
      size<0>(CtaTileShapeMNK{}) / kEpilogueTileM;
  static constexpr int kEpilogueTilesN =
      size<1>(CtaTileShapeMNK{}) / kEpilogueTileN;
  static constexpr int kEpilogueTilesPerCta =
      kEpilogueTilesM * kEpilogueTilesN;
  static constexpr int kQueueSlots = 2;

  struct SharedStorage {
    cute::array_aligned<ElementCompute,
                        kQueueSlots * kValuesPerEpilogueTile>
        smem_raw;
    cute::array_aligned<int, kQueueSlots> ready_flags;
    cute::array_aligned<ElementCompute, kQueueSlots> sink;
  };

  using NormalConstStrideMNL = cute::Stride<_0, _0, int64_t>;
  struct Arguments {
    ElementBlockScaleFactor* ptr_scale_factor = nullptr;
    ElementCompute const* norm_constant_ptr = nullptr;
    NormalConstStrideMNL norm_constant_stride = {};
    uint8_t* ptr_payload = nullptr;
    int64_t stride_bytes = 0;
    int32_t debug_timing_mode = 0;
  };
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const&, Arguments const& args, void*) {
    return args;
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape,
                            Arguments const&) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    return (N % SFVecSize == 0);
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const&, Arguments const&, void*, cudaStream_t,
      cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE C1LoadMnConsumeProbeStore() {}

  CUTLASS_HOST_DEVICE
  C1LoadMnConsumeProbeStore(Params const& params,
                            SharedStorage const& shared_storage)
      : params_ptr(&params),
        smem_raw(const_cast<ElementCompute*>(
            shared_storage.smem_raw.data())),
        ready_flags(const_cast<int*>(
            shared_storage.ready_flags.data())),
        sink(const_cast<ElementCompute*>(
            shared_storage.sink.data())) {}

  Params const* params_ptr = nullptr;
  ElementCompute* smem_raw = nullptr;
  int* ready_flags = nullptr;
  ElementCompute* sink = nullptr;

  CUTLASS_DEVICE bool is_producer_load_needed() const { return true; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }

  template <class... Args>
  struct ProducerLoadCallbacks
      : cutlass::epilogue::fusion::EmptyProducerLoadCallbacks {
    CUTLASS_DEVICE
    ProducerLoadCallbacks(ElementCompute* smem_raw_, int* ready_flags_,
                          ElementCompute* sink_, int lane_idx_)
        : smem_raw(smem_raw_),
          ready_flags(ready_flags_),
          sink(sink_),
          lane_idx(lane_idx_) {}

    ElementCompute* smem_raw;
    int* ready_flags;
    ElementCompute* sink;
    int lane_idx;

    CUTLASS_DEVICE void begin() {
      if (lane_idx < kQueueSlots) {
        sink[lane_idx] = ElementCompute(0);
      }
      __syncwarp();
    }

    CUTLASS_DEVICE void consume_slot(int slot) {
      volatile int* ready =
          reinterpret_cast<volatile int*>(ready_flags);
      int wait_spins = 0;
      while (ready[slot] == 0 && wait_spins < (1 << 22)) {
        __nanosleep(16);
        ++wait_spins;
      }
      if (ready[slot] == 0) {
        return;
      }

      volatile ElementCompute* raw =
          reinterpret_cast<volatile ElementCompute*>(smem_raw);
      ElementCompute acc = ElementCompute(0);
      int const slot_offset = slot * kValuesPerEpilogueTile;
      for (int idx = lane_idx; idx < kValuesPerEpilogueTile;
           idx += cutlass::NumThreadsPerWarp) {
        ElementCompute value = raw[slot_offset + idx];
        acc += value < ElementCompute(0) ? -value : value;
      }

      CUTLASS_PRAGMA_UNROLL
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
      }
      if (lane_idx == 0) {
        sink[slot] = acc;
        __threadfence_block();
        ready_flags[slot] = 0;
      }
    }

    CUTLASS_DEVICE void step(uint64_t*, int epi_m, int epi_n, int, bool) {
      int const linear = epi_n * kEpilogueTilesM + epi_m;
      if (linear > 0) {
        consume_slot((linear - 1) % kQueueSlots);
      }
    }

    CUTLASS_DEVICE void end() {
      if constexpr (kEpilogueTilesPerCta > 0) {
        consume_slot((kEpilogueTilesPerCta - 1) % kQueueSlots);
      }
    }
  };

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(
      cutlass::epilogue::fusion::detail::ProducerLoadArgs<Args...> const&
          args) {
    return ProducerLoadCallbacks<Args...>(
        smem_raw, ready_flags, sink, args.thread_idx);
  }

  template <class CoordTensor, class ProblemShapeMN, class TiledCopy_>
  struct ConsumerStoreCallbacks
      : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(CoordTensor&& tC_cD_global_,
                           ProblemShapeMN problem_shape_mn_,
                           int thread_idx_, TiledCopy_ const&,
                           ElementCompute* smem_raw_, int* ready_flags_,
                           int64_t tile_m_base_, int64_t tile_n_base_)
        : tC_cD_global(cute::forward<CoordTensor>(tC_cD_global_)),
          problem_shape_mn(problem_shape_mn_),
          thread_idx(thread_idx_),
          smem_raw(smem_raw_),
          ready_flags(ready_flags_),
          tile_m_base(tile_m_base_),
          tile_n_base(tile_n_base_) {}

    CoordTensor tC_cD_global;
    ProblemShapeMN problem_shape_mn;
    int thread_idx;
    ElementCompute* smem_raw;
    int* ready_flags;
    int64_t tile_m_base;
    int64_t tile_n_base;

    template <class ElementAccumulator, class ElementInput>
    CUTLASS_DEVICE auto visit(
        cutlass::Array<ElementAccumulator, FragmentSize> const&, int, int, int,
        cutlass::Array<ElementInput, FragmentSize> const& frg_input) {
      return frg_input;
    }

    template <class SmemTensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void reduce(SmemTensor&&, SyncFn const& sync_fn, int epi_m,
                               int epi_n, bool, VTensor visit_results) {
      int const linear = epi_n * kEpilogueTilesM + epi_m;
      int const slot = linear % kQueueSlots;
      auto coords = coalesce(tC_cD_global(_, _, _, epi_m, epi_n));
      auto values = coalesce(visit_results);
      volatile ElementCompute* raw_smem =
          reinterpret_cast<volatile ElementCompute*>(smem_raw);
      int const slot_offset = slot * kValuesPerEpilogueTile;

      CUTLASS_PRAGMA_UNROLL
      for (int epi_v = 0; epi_v < size(values); ++epi_v) {
        auto frg = values(epi_v);
        CUTLASS_PRAGMA_UNROLL
        for (int lane = 0; lane < FragmentSize; ++lane) {
          int coord_idx = epi_v * FragmentSize + lane;
          auto coord = coords(coord_idx);
          if (elem_less(coord, problem_shape_mn)) {
            int64_t const row = get<0>(coord);
            int64_t const c1_col = get<1>(coord);
            if ((c1_col & 1) == 0) {
              int const local_row =
                  static_cast<int>(row - tile_m_base -
                                   epi_m * kEpilogueTileM);
              int const local_logical_col =
                  static_cast<int>((c1_col - tile_n_base -
                                    epi_n * kEpilogueTileN) >>
                                   1);
              if (local_row >= 0 && local_row < kEpilogueTileM &&
                  local_logical_col >= 0 &&
                  local_logical_col < kLogicalColsPerEpilogueTile) {
                raw_smem[slot_offset +
                         local_row * kLogicalColsPerEpilogueTile +
                         local_logical_col] = frg[lane];
              }
            }
          }
        }
      }

      sync_fn();
      if (thread_idx == 0) {
        __threadfence_block();
        ready_flags[slot] = linear + 1;
      }
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      cutlass::epilogue::fusion::detail::ConsumerStoreArgs<Args...> const&
          args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    auto problem_shape_mn = make_shape(M, N);
    Tensor mD_crd = make_identity_tensor(problem_shape_mn);
    Tensor cD_mn = local_tile(mD_crd, take<0, 2>(args.tile_shape_mnk),
                              make_coord(m, n));
    Tensor tC_cD_global =
        cutlass::epilogue::fusion::sm90_partition_for_epilogue<ReferenceSrc>(
            cD_mn, args.epi_tile, args.tiled_copy, args.thread_idx);
    int64_t const tile_m_base =
        static_cast<int64_t>(m) *
        static_cast<int64_t>(size<0>(args.tile_shape_mnk));
    int64_t const tile_n_base =
        static_cast<int64_t>(n) *
        static_cast<int64_t>(size<1>(args.tile_shape_mnk));

    return ConsumerStoreCallbacks<
        decltype(tC_cD_global), decltype(problem_shape_mn),
        decltype(args.tiled_copy)>(
        cute::move(tC_cD_global), problem_shape_mn, args.thread_idx,
        args.tiled_copy, smem_raw, ready_flags, tile_m_base, tile_n_base);
  }
};

template <int SFVecSize, class EpilogueTile, class CtaTileShapeMNK,
          int FragmentSize, class ElementOutput, class ElementCompute,
          class ElementBlockScaleFactor,
          cutlass::FloatRoundStyle RoundStyle =
              cutlass::FloatRoundStyle::round_to_nearest,
          bool FullSlotsNoProducerWait = false>
struct C1Warp1ConsumeProbeStore {
  static_assert(size<1>(EpilogueTile{}) % SFVecSize == 0,
                "EpilogueTileN should be divisible by SFVecSize");

  static constexpr int kEpilogueTileM = 64;
  static constexpr int kEpilogueTileN = 32;
  static constexpr int kLogicalColsPerEpilogueTile = kEpilogueTileN / 2;
  static constexpr int kValuesPerEpilogueTile =
      kEpilogueTileM * kLogicalColsPerEpilogueTile;
  static constexpr int kEpilogueTilesM =
      size<0>(CtaTileShapeMNK{}) / kEpilogueTileM;
  static constexpr int kEpilogueTilesN =
      size<1>(CtaTileShapeMNK{}) / kEpilogueTileN;
  static constexpr int kEpilogueTilesPerCta =
      kEpilogueTilesM * kEpilogueTilesN;
  static constexpr int kNoWaitQueueSlots =
      kEpilogueTilesPerCta <= 8 ? kEpilogueTilesPerCta : 8;
  static constexpr int kQueueSlots =
      FullSlotsNoProducerWait
          ? kNoWaitQueueSlots
          : (kEpilogueTilesPerCta <= 8 ? kEpilogueTilesPerCta : 8);

  struct SharedStorage {
    cute::array_aligned<ElementCompute,
                        kQueueSlots * kValuesPerEpilogueTile>
        smem_raw;
    cute::array_aligned<int, kQueueSlots> ready_flags;
    cute::array_aligned<ElementCompute, kQueueSlots> sink;
  };

  using NormalConstStrideMNL = cute::Stride<_0, _0, int64_t>;
  struct Arguments {
    ElementBlockScaleFactor* ptr_scale_factor = nullptr;
    ElementCompute const* norm_constant_ptr = nullptr;
    NormalConstStrideMNL norm_constant_stride = {};
    uint8_t* ptr_payload = nullptr;
    int64_t stride_bytes = 0;
    int32_t debug_timing_mode = 0;
  };
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const&, Arguments const& args, void*) {
    return args;
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape,
                            Arguments const&) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    return (N % SFVecSize == 0);
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const&, Arguments const&, void*, cudaStream_t,
      cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE C1Warp1ConsumeProbeStore() {}

  CUTLASS_HOST_DEVICE
  C1Warp1ConsumeProbeStore(Params const& params,
                           SharedStorage const& shared_storage)
      : params_ptr(&params),
        smem_raw(const_cast<ElementCompute*>(
            shared_storage.smem_raw.data())),
        ready_flags(const_cast<int*>(
            shared_storage.ready_flags.data())),
        sink(const_cast<ElementCompute*>(
            shared_storage.sink.data())) {}

  Params const* params_ptr = nullptr;
  ElementCompute* smem_raw = nullptr;
  int* ready_flags = nullptr;
  ElementCompute* sink = nullptr;

  CUTLASS_DEVICE void vllm_proj4_warp1_init(int lane_idx) {
    if (lane_idx < kQueueSlots) {
      ready_flags[lane_idx] = 0;
      sink[lane_idx] = ElementCompute(0);
    }
    __syncwarp();
    if (lane_idx == 0) {
      __threadfence_block();
    }
  }

  CUTLASS_DEVICE static int ready_token(int64_t problem_n,
                                        int64_t tile_m_base,
                                        int64_t tile_n_base,
                                        int linear) {
    int64_t const cta_m =
        tile_m_base / static_cast<int64_t>(size<0>(CtaTileShapeMNK{}));
    int64_t const cta_n =
        tile_n_base / static_cast<int64_t>(size<1>(CtaTileShapeMNK{}));
    int64_t const num_cta_n =
        (problem_n + static_cast<int64_t>(size<1>(CtaTileShapeMNK{})) - 1) /
        static_cast<int64_t>(size<1>(CtaTileShapeMNK{}));
    int64_t const tile_linear = cta_m * num_cta_n + cta_n;
    return static_cast<int>(tile_linear * kEpilogueTilesPerCta + linear + 1);
  }

  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(
      cutlass::epilogue::fusion::detail::ProducerLoadArgs<Args...> const&) {
    return cutlass::epilogue::fusion::EmptyProducerLoadCallbacks{};
  }

  template <class... Args>
  struct Warp1ConsumerCallbacks
      : cutlass::epilogue::fusion::EmptyProducerLoadCallbacks {
    CUTLASS_DEVICE
    Warp1ConsumerCallbacks(ElementCompute* smem_raw_, int* ready_flags_,
                           ElementCompute* sink_, int lane_idx_,
                           int64_t problem_n_, int64_t tile_m_base_,
                           int64_t tile_n_base_)
        : smem_raw(smem_raw_),
          ready_flags(ready_flags_),
          sink(sink_),
          lane_idx(lane_idx_),
          problem_n(problem_n_),
          tile_m_base(tile_m_base_),
          tile_n_base(tile_n_base_) {}

    ElementCompute* smem_raw;
    int* ready_flags;
    ElementCompute* sink;
    int lane_idx;
    int64_t problem_n;
    int64_t tile_m_base;
    int64_t tile_n_base;

    CUTLASS_DEVICE void begin() {
      if (lane_idx < kQueueSlots) {
        sink[lane_idx] = ElementCompute(0);
      }
      __syncwarp();
    }

    CUTLASS_DEVICE void consume_slot(int slot, int expected_linear) {
      volatile int* ready =
          reinterpret_cast<volatile int*>(ready_flags);
      int const expected_token =
          C1Warp1ConsumeProbeStore::ready_token(
              problem_n, tile_m_base, tile_n_base, expected_linear);
      int wait_spins = 0;
      while (ready[slot] != expected_token && wait_spins < (1 << 22)) {
        __nanosleep(16);
        ++wait_spins;
      }
      if (ready[slot] != expected_token) {
        return;
      }

      volatile ElementCompute* raw =
          reinterpret_cast<volatile ElementCompute*>(smem_raw);
      ElementCompute acc = ElementCompute(0);
      int const slot_offset = slot * kValuesPerEpilogueTile;
      for (int idx = lane_idx; idx < kValuesPerEpilogueTile;
           idx += cutlass::NumThreadsPerWarp) {
        ElementCompute value = raw[slot_offset + idx];
        acc += value < ElementCompute(0) ? -value : value;
      }

      CUTLASS_PRAGMA_UNROLL
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
      }
      if (lane_idx == 0) {
        sink[slot] = acc;
        __threadfence_block();
        ready_flags[slot] = 0;
      }
    }

    CUTLASS_DEVICE void step(uint64_t*, int epi_m, int epi_n, int, bool) {
      int const linear = epi_n * kEpilogueTilesM + epi_m;
      if (linear > 0) {
        int const prev_linear = linear - 1;
        consume_slot(prev_linear % kQueueSlots, prev_linear);
      }
    }

    CUTLASS_DEVICE void end() {
      if constexpr (kEpilogueTilesPerCta > 0) {
        int const last_linear = kEpilogueTilesPerCta - 1;
        consume_slot(last_linear % kQueueSlots, last_linear);
      }
    }
  };

  template <class... Args>
  CUTLASS_DEVICE auto get_warp1_consumer_callbacks(
      cutlass::epilogue::fusion::detail::ProducerLoadArgs<Args...> const&
          args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    int64_t const tile_m_base =
        static_cast<int64_t>(m) *
        static_cast<int64_t>(size<0>(args.tile_shape_mnk));
    int64_t const tile_n_base =
        static_cast<int64_t>(n) *
        static_cast<int64_t>(size<1>(args.tile_shape_mnk));
    return Warp1ConsumerCallbacks<Args...>(
        smem_raw, ready_flags, sink, args.thread_idx,
        static_cast<int64_t>(N), tile_m_base, tile_n_base);
  }

  template <class CoordTensor, class ProblemShapeMN, class TiledCopy_>
  struct ConsumerStoreCallbacks
      : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(CoordTensor&& tC_cD_global_,
                           ProblemShapeMN problem_shape_mn_,
                           int thread_idx_, TiledCopy_ const&,
                           ElementCompute* smem_raw_, int* ready_flags_,
                           int64_t tile_m_base_, int64_t tile_n_base_)
        : tC_cD_global(cute::forward<CoordTensor>(tC_cD_global_)),
          problem_shape_mn(problem_shape_mn_),
          thread_idx(thread_idx_),
          smem_raw(smem_raw_),
          ready_flags(ready_flags_),
          tile_m_base(tile_m_base_),
          tile_n_base(tile_n_base_) {}

    CoordTensor tC_cD_global;
    ProblemShapeMN problem_shape_mn;
    int thread_idx;
    ElementCompute* smem_raw;
    int* ready_flags;
    int64_t tile_m_base;
    int64_t tile_n_base;

    template <class ElementAccumulator, class ElementInput>
    CUTLASS_DEVICE auto visit(
        cutlass::Array<ElementAccumulator, FragmentSize> const&, int, int, int,
        cutlass::Array<ElementInput, FragmentSize> const& frg_input) {
      return frg_input;
    }

    template <class SmemTensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void reduce(SmemTensor&&, SyncFn const& sync_fn, int epi_m,
                               int epi_n, bool, VTensor visit_results) {
      int const linear = epi_n * kEpilogueTilesM + epi_m;
      int const slot = linear % kQueueSlots;
      volatile int* ready =
          reinterpret_cast<volatile int*>(ready_flags);
      if constexpr (!FullSlotsNoProducerWait) {
        int wait_spins = 0;
        while (ready[slot] != 0 && wait_spins < (1 << 22)) {
          __nanosleep(16);
          ++wait_spins;
        }
      }

      auto coords = coalesce(tC_cD_global(_, _, _, epi_m, epi_n));
      auto values = coalesce(visit_results);
      volatile ElementCompute* raw_smem =
          reinterpret_cast<volatile ElementCompute*>(smem_raw);
      int const slot_offset = slot * kValuesPerEpilogueTile;

      CUTLASS_PRAGMA_UNROLL
      for (int epi_v = 0; epi_v < size(values); ++epi_v) {
        auto frg = values(epi_v);
        CUTLASS_PRAGMA_UNROLL
        for (int lane = 0; lane < FragmentSize; ++lane) {
          int coord_idx = epi_v * FragmentSize + lane;
          auto coord = coords(coord_idx);
          if (elem_less(coord, problem_shape_mn)) {
            int64_t const row = get<0>(coord);
            int64_t const c1_col = get<1>(coord);
            int const local_row =
                static_cast<int>(row - tile_m_base -
                                 epi_m * kEpilogueTileM);
            int const local_logical_col =
                static_cast<int>((c1_col - tile_n_base -
                                  epi_n * kEpilogueTileN) >>
                                 1);
            if (local_row >= 0 && local_row < kEpilogueTileM &&
                local_logical_col >= 0 &&
                local_logical_col < kLogicalColsPerEpilogueTile) {
              raw_smem[slot_offset +
                       local_row * kLogicalColsPerEpilogueTile +
                       local_logical_col] = frg[lane];
            }
          }
        }
      }

      sync_fn();
      if (thread_idx == 0) {
        __threadfence_block();
        ready_flags[slot] = C1Warp1ConsumeProbeStore::ready_token(
            static_cast<int64_t>(get<1>(problem_shape_mn)), tile_m_base,
            tile_n_base, linear);
      }
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      cutlass::epilogue::fusion::detail::ConsumerStoreArgs<Args...> const&
          args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    auto problem_shape_mn = make_shape(M, N);
    Tensor mD_crd = make_identity_tensor(problem_shape_mn);
    Tensor cD_mn = local_tile(mD_crd, take<0, 2>(args.tile_shape_mnk),
                              make_coord(m, n));
    Tensor tC_cD_global =
        cutlass::epilogue::fusion::sm90_partition_for_epilogue<ReferenceSrc>(
            cD_mn, args.epi_tile, args.tiled_copy, args.thread_idx);
    int64_t const tile_m_base =
        static_cast<int64_t>(m) *
        static_cast<int64_t>(size<0>(args.tile_shape_mnk));
    int64_t const tile_n_base =
        static_cast<int64_t>(n) *
        static_cast<int64_t>(size<1>(args.tile_shape_mnk));

    return ConsumerStoreCallbacks<
        decltype(tC_cD_global), decltype(problem_shape_mn),
        decltype(args.tiled_copy)>(
        cute::move(tC_cD_global), problem_shape_mn, args.thread_idx,
        args.tiled_copy, smem_raw, ready_flags, tile_m_base, tile_n_base);
  }
};

template <int SFVecSize, class EpilogueTile, class CtaTileShapeMNK,
          int FragmentSize, class ElementOutput, class ElementCompute,
          class ElementBlockScaleFactor,
          cutlass::FloatRoundStyle RoundStyle =
              cutlass::FloatRoundStyle::round_to_nearest,
          bool FullSlotsNoProducerWait = false>
struct C1Warp1ValidOutputStore
    : C1Warp1ConsumeProbeStore<SFVecSize, EpilogueTile, CtaTileShapeMNK,
                               FragmentSize, ElementOutput, ElementCompute,
                               ElementBlockScaleFactor, RoundStyle,
                               FullSlotsNoProducerWait> {
  using Base =
      C1Warp1ConsumeProbeStore<SFVecSize, EpilogueTile, CtaTileShapeMNK,
                               FragmentSize, ElementOutput, ElementCompute,
                               ElementBlockScaleFactor, RoundStyle,
                               FullSlotsNoProducerWait>;
  using SharedStorage = typename Base::SharedStorage;
  using Arguments = typename Base::Arguments;
  using Params = typename Base::Params;
  using NormalConstStrideMNL = typename Base::NormalConstStrideMNL;

  static constexpr int kEpilogueTileM = Base::kEpilogueTileM;
  static constexpr int kEpilogueTileN = Base::kEpilogueTileN;
  static constexpr int kLogicalColsPerEpilogueTile =
      Base::kLogicalColsPerEpilogueTile;
  static constexpr int kValuesPerEpilogueTile =
      Base::kValuesPerEpilogueTile;
  static constexpr int kQueueSlots = Base::kQueueSlots;
  static constexpr int kEpilogueTilesM = Base::kEpilogueTilesM;
  static constexpr int kEpilogueTilesN = Base::kEpilogueTilesN;
  static constexpr int kEpilogueTilesPerCta = Base::kEpilogueTilesPerCta;

  CUTLASS_HOST_DEVICE C1Warp1ValidOutputStore() {}

  CUTLASS_HOST_DEVICE
  C1Warp1ValidOutputStore(Params const& params,
                          SharedStorage const& shared_storage)
      : Base(params, shared_storage) {}

  template <class... Args>
  struct Warp1ConsumerCallbacks
      : cutlass::epilogue::fusion::EmptyProducerLoadCallbacks {
    CUTLASS_DEVICE
    Warp1ConsumerCallbacks(ElementCompute* smem_raw_, int* ready_flags_,
                           ElementCompute* sink_, Params const* params_ptr_,
                           int lane_idx_, int64_t problem_m_,
                           int64_t problem_n_, int64_t tile_m_base_,
                           int64_t tile_n_base_)
        : smem_raw(smem_raw_),
          ready_flags(ready_flags_),
          sink(sink_),
          params_ptr(params_ptr_),
          lane_idx(lane_idx_),
          problem_m(problem_m_),
          problem_n(problem_n_),
          tile_m_base(tile_m_base_),
          tile_n_base(tile_n_base_) {}

    ElementCompute* smem_raw;
    int* ready_flags;
    ElementCompute* sink;
    Params const* params_ptr;
    int lane_idx;
    int64_t problem_m;
    int64_t problem_n;
    int64_t tile_m_base;
    int64_t tile_n_base;

    CUTLASS_DEVICE void begin() {
      if (lane_idx < kQueueSlots) {
        sink[lane_idx] = ElementCompute(0);
      }
      __syncwarp();
    }

    CUTLASS_DEVICE static int64_t scale_offset(int64_t row,
                                               int64_t logical_pack_col,
                                               int64_t num_k_tiles) {
      int64_t const m_tile_idx = row >> 7;
      int64_t const outer_m_idx = row & 31;
      int64_t const inner_m_idx = (row >> 5) & 3;
      int64_t const k_tile_idx = logical_pack_col >> 2;
      int64_t const inner_k_idx = logical_pack_col & 3;
      return ((m_tile_idx * num_k_tiles + k_tile_idx) << 9) |
             (outer_m_idx << 4) | (inner_m_idx << 2) | inner_k_idx;
    }

    CUTLASS_DEVICE void write_row(int slot, int epi_m, int epi_n,
                                  int local_row) {
      int64_t const row =
          tile_m_base + epi_m * kEpilogueTileM + local_row;
      if (row >= problem_m) {
        return;
      }

      int64_t const c1_col_base =
          tile_n_base + epi_n * kEpilogueTileN;
      if (c1_col_base + kEpilogueTileN > problem_n) {
        return;
      }

      int64_t const logical_col_base = c1_col_base >> 1;
      int64_t const logical_pack_col = logical_col_base >> 4;
      int64_t const logical_cols = problem_n >> 1;
      int64_t const num_k_tiles = (logical_cols + 63) >> 6;
      int const slot_offset = slot * kValuesPerEpilogueTile;
      ElementCompute const* raw =
          smem_raw + slot_offset +
          local_row * kLogicalColsPerEpilogueTile;

      cutlass::maximum_absolute_value_reduction<ElementCompute, true> amax_op;
      ElementCompute amax{0};
      CUTLASS_PRAGMA_UNROLL
      for (int col = 0; col < kLogicalColsPerEpilogueTile; ++col) {
        amax = amax_op(amax, raw[col]);
      }

      ElementCompute norm_constant =
          params_ptr->norm_constant_ptr != nullptr
              ? *(params_ptr->norm_constant_ptr)
              : ElementCompute(1);
      ElementCompute fp_max =
          ElementCompute(cutlass::platform::numeric_limits<ElementOutput>::max());
      ElementCompute scale_down_factor =
          cutlass::reciprocal_approximate_ftz<ElementCompute>{}(fp_max);
      ElementCompute pvscale =
          cutlass::multiplies<ElementCompute>{}(
              amax,
              cutlass::multiplies<ElementCompute>{}(norm_constant,
                                                    scale_down_factor));

      ElementBlockScaleFactor qpvscale =
          cutlass::NumericConverter<ElementBlockScaleFactor,
                                    ElementCompute>{}(pvscale);
      params_ptr->ptr_scale_factor[scale_offset(row, logical_pack_col,
                                                num_k_tiles)] = qpvscale;

      ElementCompute qpvscale_up =
          cutlass::NumericConverter<ElementCompute,
                                    ElementBlockScaleFactor>{}(qpvscale);
      ElementCompute acc_scale = qpvscale_up != ElementCompute(0)
                                     ? cutlass::multiplies<ElementCompute>{}(
                                           norm_constant,
                                           cutlass::reciprocal_approximate_ftz<
                                               ElementCompute>{}(qpvscale_up))
                                     : ElementCompute(0);
      acc_scale =
          cutlass::minimum_with_nan_propagation<ElementCompute>{}(
              acc_scale,
              cutlass::platform::numeric_limits<ElementCompute>::max());

      using ConvertFp4 =
          cutlass::NumericConverter<cutlass::float_e2m1_t, ElementCompute,
                                    RoundStyle>;
      ConvertFp4 convert_fp4{};
      uint8_t* row_payload =
          params_ptr->ptr_payload + row * params_ptr->stride_bytes +
          (logical_col_base >> 1);

      CUTLASS_PRAGMA_UNROLL
      for (int byte = 0; byte < kLogicalColsPerEpilogueTile / 2; ++byte) {
        cutlass::float_e2m1_t lo_fp4 =
            convert_fp4(cutlass::multiplies<ElementCompute>{}(
                raw[byte * 2], acc_scale));
        cutlass::float_e2m1_t hi_fp4 =
            convert_fp4(cutlass::multiplies<ElementCompute>{}(
                raw[byte * 2 + 1], acc_scale));
        uint8_t lo = static_cast<uint8_t>(lo_fp4.raw()) & 0x0f;
        uint8_t hi = static_cast<uint8_t>(hi_fp4.raw()) & 0x0f;
        row_payload[byte] = static_cast<uint8_t>(lo | (hi << 4));
      }
    }

    CUTLASS_DEVICE void consume_slot(int slot, int epi_m, int epi_n) {
      volatile int* ready =
          reinterpret_cast<volatile int*>(ready_flags);
      int const expected_linear = epi_n * kEpilogueTilesM + epi_m;
      int const expected_token =
          Base::ready_token(problem_n, tile_m_base, tile_n_base,
                            expected_linear);
      int wait_spins = 0;
      while (ready[slot] != expected_token && wait_spins < (1 << 22)) {
        __nanosleep(16);
        ++wait_spins;
      }
      if (ready[slot] != expected_token) {
        return;
      }

      for (int local_row = lane_idx; local_row < kEpilogueTileM;
           local_row += cutlass::NumThreadsPerWarp) {
        write_row(slot, epi_m, epi_n, local_row);
      }

      __syncwarp();
      if (lane_idx == 0) {
        sink[slot] = ElementCompute(1);
        __threadfence_block();
        ready_flags[slot] = 0;
      }
    }

    CUTLASS_DEVICE void step(uint64_t*, int epi_m, int epi_n, int, bool) {
      int const linear = epi_n * kEpilogueTilesM + epi_m;
      if (linear > 0) {
        int const prev_linear = linear - 1;
        int const prev_epi_m = prev_linear % kEpilogueTilesM;
        int const prev_epi_n = prev_linear / kEpilogueTilesM;
        consume_slot(prev_linear % kQueueSlots, prev_epi_m, prev_epi_n);
      }
    }

    CUTLASS_DEVICE void end() {
      if constexpr (kEpilogueTilesPerCta > 0) {
        int const last_linear = kEpilogueTilesPerCta - 1;
        int const last_epi_m = last_linear % kEpilogueTilesM;
        int const last_epi_n = last_linear / kEpilogueTilesM;
        consume_slot(last_linear % kQueueSlots, last_epi_m, last_epi_n);
      }
    }
  };

  template <class... Args>
  CUTLASS_DEVICE auto get_warp1_consumer_callbacks(
      cutlass::epilogue::fusion::detail::ProducerLoadArgs<Args...> const&
          args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    int64_t const tile_m_base =
        static_cast<int64_t>(m) *
        static_cast<int64_t>(size<0>(args.tile_shape_mnk));
    int64_t const tile_n_base =
        static_cast<int64_t>(n) *
        static_cast<int64_t>(size<1>(args.tile_shape_mnk));
    return Warp1ConsumerCallbacks<Args...>(
        this->smem_raw, this->ready_flags, this->sink, this->params_ptr,
        args.thread_idx, static_cast<int64_t>(M), static_cast<int64_t>(N),
        tile_m_base, tile_n_base);
  }
};

template <int SFVecSize, class EpilogueTile, class CtaTileShapeMNK,
          int FragmentSize, class ElementOutput, class ElementCompute,
          class ElementBlockScaleFactor,
          cutlass::FloatRoundStyle RoundStyle =
              cutlass::FloatRoundStyle::round_to_nearest>
struct C1Warp1LoadMnValidOutputStore
    : C1Warp1ValidOutputStore<SFVecSize, EpilogueTile, CtaTileShapeMNK,
                              FragmentSize, ElementOutput, ElementCompute,
                              ElementBlockScaleFactor, RoundStyle, false> {
  using Base =
      C1Warp1ValidOutputStore<SFVecSize, EpilogueTile, CtaTileShapeMNK,
                              FragmentSize, ElementOutput, ElementCompute,
                              ElementBlockScaleFactor, RoundStyle, false>;
  using BaseProbe =
      C1Warp1ConsumeProbeStore<SFVecSize, EpilogueTile, CtaTileShapeMNK,
                               FragmentSize, ElementOutput, ElementCompute,
                               ElementBlockScaleFactor, RoundStyle, false>;
  using Arguments = typename Base::Arguments;
  using Params = typename Base::Params;
  using NormalConstStrideMNL = typename Base::NormalConstStrideMNL;

  static constexpr int kEpilogueTileM = Base::kEpilogueTileM;
  static constexpr int kEpilogueTileN = Base::kEpilogueTileN;
  static constexpr int kLogicalColsPerEpilogueTile =
      Base::kLogicalColsPerEpilogueTile;
  static constexpr int kValuesPerEpilogueTile =
      Base::kValuesPerEpilogueTile;
  static constexpr int kQueueSlots = Base::kQueueSlots;
  static constexpr int kEpilogueTilesM = Base::kEpilogueTilesM;
  static constexpr int kEpilogueTilesN = Base::kEpilogueTilesN;
  static constexpr int kEpilogueTilesPerCta = Base::kEpilogueTilesPerCta;

  struct SharedStorage : Base::SharedStorage {
    cute::array_aligned<int, kQueueSlots> done_counts;
  };

  CUTLASS_HOST_DEVICE C1Warp1LoadMnValidOutputStore() {}

  CUTLASS_HOST_DEVICE
  C1Warp1LoadMnValidOutputStore(Params const& params,
                                SharedStorage const& shared_storage)
      : Base(params, shared_storage),
        done_counts(const_cast<int*>(shared_storage.done_counts.data())) {}

  int* done_counts = nullptr;

  CUTLASS_DEVICE bool is_producer_load_needed() const { return true; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }

  CUTLASS_DEVICE void vllm_proj4_warp1_init(int lane_idx) {
    Base::vllm_proj4_warp1_init(lane_idx);
    if (lane_idx < kQueueSlots) {
      done_counts[lane_idx] = 0;
    }
    __syncwarp();
  }

  template <class... Args>
  struct DualConsumerCallbacks
      : cutlass::epilogue::fusion::EmptyProducerLoadCallbacks {
    CUTLASS_DEVICE
    DualConsumerCallbacks(ElementCompute* smem_raw_, int* ready_flags_,
                          int* done_counts_, ElementCompute* sink_,
                          Params const* params_ptr_, int lane_idx_,
                          int consumer_id_, int64_t problem_m_,
                          int64_t problem_n_, int64_t tile_m_base_,
                          int64_t tile_n_base_)
        : smem_raw(smem_raw_),
          ready_flags(ready_flags_),
          done_counts(done_counts_),
          sink(sink_),
          params_ptr(params_ptr_),
          lane_idx(lane_idx_ % cutlass::NumThreadsPerWarp),
          consumer_id(consumer_id_),
          problem_m(problem_m_),
          problem_n(problem_n_),
          tile_m_base(tile_m_base_),
          tile_n_base(tile_n_base_) {}

    ElementCompute* smem_raw;
    int* ready_flags;
    int* done_counts;
    ElementCompute* sink;
    Params const* params_ptr;
    int lane_idx;
    int consumer_id;
    int64_t problem_m;
    int64_t problem_n;
    int64_t tile_m_base;
    int64_t tile_n_base;

    CUTLASS_DEVICE void begin() {
      if (lane_idx < kQueueSlots) {
        done_counts[lane_idx] = 0;
      }
      __syncwarp();
    }

    CUTLASS_DEVICE static int64_t scale_offset(int64_t row,
                                               int64_t logical_pack_col,
                                               int64_t num_k_tiles) {
      int64_t const m_tile_idx = row >> 7;
      int64_t const outer_m_idx = row & 31;
      int64_t const inner_m_idx = (row >> 5) & 3;
      int64_t const k_tile_idx = logical_pack_col >> 2;
      int64_t const inner_k_idx = logical_pack_col & 3;
      return ((m_tile_idx * num_k_tiles + k_tile_idx) << 9) |
             (outer_m_idx << 4) | (inner_m_idx << 2) | inner_k_idx;
    }

    CUTLASS_DEVICE void write_row(int slot, int epi_m, int epi_n,
                                  int local_row) {
      int64_t const row =
          tile_m_base + epi_m * kEpilogueTileM + local_row;
      if (row >= problem_m) {
        return;
      }

      int64_t const c1_col_base =
          tile_n_base + epi_n * kEpilogueTileN;
      if (c1_col_base + kEpilogueTileN > problem_n) {
        return;
      }

      int64_t const logical_col_base = c1_col_base >> 1;
      int64_t const logical_pack_col = logical_col_base >> 4;
      int64_t const logical_cols = problem_n >> 1;
      int64_t const num_k_tiles = (logical_cols + 63) >> 6;
      int const slot_offset = slot * kValuesPerEpilogueTile;
      ElementCompute const* raw =
          smem_raw + slot_offset +
          local_row * kLogicalColsPerEpilogueTile;

      cutlass::maximum_absolute_value_reduction<ElementCompute, true> amax_op;
      ElementCompute amax{0};
      CUTLASS_PRAGMA_UNROLL
      for (int col = 0; col < kLogicalColsPerEpilogueTile; ++col) {
        amax = amax_op(amax, raw[col]);
      }

      ElementCompute norm_constant =
          params_ptr->norm_constant_ptr != nullptr
              ? *(params_ptr->norm_constant_ptr)
              : ElementCompute(1);
      ElementCompute fp_max =
          ElementCompute(cutlass::platform::numeric_limits<ElementOutput>::max());
      ElementCompute scale_down_factor =
          cutlass::reciprocal_approximate_ftz<ElementCompute>{}(fp_max);
      ElementCompute pvscale =
          cutlass::multiplies<ElementCompute>{}(
              amax,
              cutlass::multiplies<ElementCompute>{}(norm_constant,
                                                    scale_down_factor));

      ElementBlockScaleFactor qpvscale =
          cutlass::NumericConverter<ElementBlockScaleFactor,
                                    ElementCompute>{}(pvscale);
      params_ptr->ptr_scale_factor[scale_offset(row, logical_pack_col,
                                                num_k_tiles)] = qpvscale;

      ElementCompute qpvscale_up =
          cutlass::NumericConverter<ElementCompute,
                                    ElementBlockScaleFactor>{}(qpvscale);
      ElementCompute acc_scale = qpvscale_up != ElementCompute(0)
                                     ? cutlass::multiplies<ElementCompute>{}(
                                           norm_constant,
                                           cutlass::reciprocal_approximate_ftz<
                                               ElementCompute>{}(qpvscale_up))
                                     : ElementCompute(0);
      acc_scale =
          cutlass::minimum_with_nan_propagation<ElementCompute>{}(
              acc_scale,
              cutlass::platform::numeric_limits<ElementCompute>::max());

      using ConvertFp4 =
          cutlass::NumericConverter<cutlass::float_e2m1_t, ElementCompute,
                                    RoundStyle>;
      ConvertFp4 convert_fp4{};
      uint8_t* row_payload =
          params_ptr->ptr_payload + row * params_ptr->stride_bytes +
          (logical_col_base >> 1);

      CUTLASS_PRAGMA_UNROLL
      for (int byte = 0; byte < kLogicalColsPerEpilogueTile / 2; ++byte) {
        cutlass::float_e2m1_t lo_fp4 =
            convert_fp4(cutlass::multiplies<ElementCompute>{}(
                raw[byte * 2], acc_scale));
        cutlass::float_e2m1_t hi_fp4 =
            convert_fp4(cutlass::multiplies<ElementCompute>{}(
                raw[byte * 2 + 1], acc_scale));
        uint8_t lo = static_cast<uint8_t>(lo_fp4.raw()) & 0x0f;
        uint8_t hi = static_cast<uint8_t>(hi_fp4.raw()) & 0x0f;
        row_payload[byte] = static_cast<uint8_t>(lo | (hi << 4));
      }
    }

    CUTLASS_DEVICE void consume_slot(int slot, int epi_m, int epi_n) {
      volatile int* ready =
          reinterpret_cast<volatile int*>(ready_flags);
      int const expected_linear = epi_n * kEpilogueTilesM + epi_m;
      int const expected_token =
          BaseProbe::ready_token(problem_n, tile_m_base, tile_n_base,
                                 expected_linear);
      int wait_spins = 0;
      while (ready[slot] != expected_token && wait_spins < (1 << 22)) {
        __nanosleep(16);
        ++wait_spins;
      }
      if (ready[slot] != expected_token) {
        return;
      }

      for (int local_row = lane_idx; local_row < kEpilogueTileM;
           local_row += cutlass::NumThreadsPerWarp) {
        if ((local_row & 1) == consumer_id) {
          write_row(slot, epi_m, epi_n, local_row);
        }
      }

      __syncwarp();
      if (lane_idx == 0) {
        int const old = atomicAdd(done_counts + slot, 1);
        if (old == 1) {
          sink[slot] = ElementCompute(1);
          __threadfence_block();
          ready_flags[slot] = 0;
        }
      }
    }

    CUTLASS_DEVICE void step(uint64_t*, int epi_m, int epi_n, int, bool) {
      int const linear = epi_n * kEpilogueTilesM + epi_m;
      if (linear > 0) {
        int const prev_linear = linear - 1;
        int const prev_epi_m = prev_linear % kEpilogueTilesM;
        int const prev_epi_n = prev_linear / kEpilogueTilesM;
        consume_slot(prev_linear % kQueueSlots, prev_epi_m, prev_epi_n);
      }
    }

    CUTLASS_DEVICE void end() {
      if constexpr (kEpilogueTilesPerCta > 0) {
        int const last_linear = kEpilogueTilesPerCta - 1;
        int const last_epi_m = last_linear % kEpilogueTilesM;
        int const last_epi_n = last_linear / kEpilogueTilesM;
        consume_slot(last_linear % kQueueSlots, last_epi_m, last_epi_n);
      }
    }
  };

  template <class... Args>
  CUTLASS_DEVICE auto make_dual_consumer_callbacks(
      cutlass::epilogue::fusion::detail::ProducerLoadArgs<Args...> const&
          args,
      int consumer_id) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    int64_t const tile_m_base =
        static_cast<int64_t>(m) *
        static_cast<int64_t>(size<0>(args.tile_shape_mnk));
    int64_t const tile_n_base =
        static_cast<int64_t>(n) *
        static_cast<int64_t>(size<1>(args.tile_shape_mnk));
    return DualConsumerCallbacks<Args...>(
        this->smem_raw, this->ready_flags, done_counts, this->sink,
        this->params_ptr, args.thread_idx, consumer_id,
        static_cast<int64_t>(M), static_cast<int64_t>(N), tile_m_base,
        tile_n_base);
  }

  template <class... Args>
  CUTLASS_DEVICE auto get_warp1_consumer_callbacks(
      cutlass::epilogue::fusion::detail::ProducerLoadArgs<Args...> const&
          args) {
    return make_dual_consumer_callbacks(args, 0);
  }

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(
      cutlass::epilogue::fusion::detail::ProducerLoadArgs<Args...> const&
          args) {
    return make_dual_consumer_callbacks(args, 1);
  }

  template <class CoordTensor, class ProblemShapeMN, class TiledCopy_>
  struct ConsumerStoreCallbacks
      : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(CoordTensor&& tC_cD_global_,
                           ProblemShapeMN problem_shape_mn_,
                           int thread_idx_, TiledCopy_ const&,
                           ElementCompute* smem_raw_, int* ready_flags_,
                           int* done_counts_, int64_t tile_m_base_,
                           int64_t tile_n_base_)
        : tC_cD_global(cute::forward<CoordTensor>(tC_cD_global_)),
          problem_shape_mn(problem_shape_mn_),
          thread_idx(thread_idx_),
          smem_raw(smem_raw_),
          ready_flags(ready_flags_),
          done_counts(done_counts_),
          tile_m_base(tile_m_base_),
          tile_n_base(tile_n_base_) {}

    CoordTensor tC_cD_global;
    ProblemShapeMN problem_shape_mn;
    int thread_idx;
    ElementCompute* smem_raw;
    int* ready_flags;
    int* done_counts;
    int64_t tile_m_base;
    int64_t tile_n_base;

    template <class ElementAccumulator, class ElementInput>
    CUTLASS_DEVICE auto visit(
        cutlass::Array<ElementAccumulator, FragmentSize> const&, int, int, int,
        cutlass::Array<ElementInput, FragmentSize> const& frg_input) {
      return frg_input;
    }

    template <class SmemTensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void reduce(SmemTensor&&, SyncFn const& sync_fn, int epi_m,
                               int epi_n, bool, VTensor visit_results) {
      int const linear = epi_n * kEpilogueTilesM + epi_m;
      int const slot = linear % kQueueSlots;
      volatile int* ready =
          reinterpret_cast<volatile int*>(ready_flags);
      int wait_spins = 0;
      while (ready[slot] != 0 && wait_spins < (1 << 22)) {
        __nanosleep(16);
        ++wait_spins;
      }

      auto coords = coalesce(tC_cD_global(_, _, _, epi_m, epi_n));
      auto values = coalesce(visit_results);
      volatile ElementCompute* raw_smem =
          reinterpret_cast<volatile ElementCompute*>(smem_raw);
      int const slot_offset = slot * kValuesPerEpilogueTile;

      CUTLASS_PRAGMA_UNROLL
      for (int epi_v = 0; epi_v < size(values); ++epi_v) {
        auto frg = values(epi_v);
        CUTLASS_PRAGMA_UNROLL
        for (int lane = 0; lane < FragmentSize; ++lane) {
          int coord_idx = epi_v * FragmentSize + lane;
          auto coord = coords(coord_idx);
          if (elem_less(coord, problem_shape_mn)) {
            int64_t const row = get<0>(coord);
            int64_t const c1_col = get<1>(coord);
            int const local_row =
                static_cast<int>(row - tile_m_base -
                                 epi_m * kEpilogueTileM);
            int const local_logical_col =
                static_cast<int>((c1_col - tile_n_base -
                                  epi_n * kEpilogueTileN) >>
                                 1);
            if (local_row >= 0 && local_row < kEpilogueTileM &&
                local_logical_col >= 0 &&
                local_logical_col < kLogicalColsPerEpilogueTile) {
              raw_smem[slot_offset +
                       local_row * kLogicalColsPerEpilogueTile +
                       local_logical_col] = frg[lane];
            }
          }
        }
      }

      sync_fn();
      if (thread_idx == 0) {
        done_counts[slot] = 0;
        __threadfence_block();
        ready_flags[slot] = BaseProbe::ready_token(
            static_cast<int64_t>(get<1>(problem_shape_mn)), tile_m_base,
            tile_n_base, linear);
      }
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      cutlass::epilogue::fusion::detail::ConsumerStoreArgs<Args...> const&
          args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    auto problem_shape_mn = make_shape(M, N);
    Tensor mD_crd = make_identity_tensor(problem_shape_mn);
    Tensor cD_mn = local_tile(mD_crd, take<0, 2>(args.tile_shape_mnk),
                              make_coord(m, n));
    Tensor tC_cD_global =
        cutlass::epilogue::fusion::sm90_partition_for_epilogue<ReferenceSrc>(
            cD_mn, args.epi_tile, args.tiled_copy, args.thread_idx);
    int64_t const tile_m_base =
        static_cast<int64_t>(m) *
        static_cast<int64_t>(size<0>(args.tile_shape_mnk));
    int64_t const tile_n_base =
        static_cast<int64_t>(n) *
        static_cast<int64_t>(size<1>(args.tile_shape_mnk));

    return ConsumerStoreCallbacks<
        decltype(tC_cD_global), decltype(problem_shape_mn),
        decltype(args.tiled_copy)>(
        cute::move(tC_cD_global), problem_shape_mn, args.thread_idx,
        args.tiled_copy, this->smem_raw, this->ready_flags, done_counts,
        tile_m_base, tile_n_base);
  }
};

template <class PairCompute, class Warp1Store>
struct C1ActQuantWarp1ConsumeProbeFusion
    : cutlass::epilogue::fusion::Sm90EVT<Warp1Store, PairCompute> {
  using Base =
      cutlass::epilogue::fusion::Sm90EVT<Warp1Store, PairCompute>;
  using ElementAux = cutlass::float_e2m1_t;
  using SharedStorage = typename Base::SharedStorage;
  using Params = typename Base::Params;

  struct Arguments {
    typename PairCompute::Arguments pair{};
    typename Warp1Store::Arguments out{};
  };

  CUTLASS_HOST_DEVICE C1ActQuantWarp1ConsumeProbeFusion() {}

  CUTLASS_HOST_DEVICE
  C1ActQuantWarp1ConsumeProbeFusion(Params const& params,
                                    SharedStorage const& shared_storage)
      : Base(params, shared_storage) {}

  using Base::ops;

  static constexpr typename Base::Arguments make_base_arguments(
      Arguments const& args) {
    return typename Base::Arguments{args.pair, args.out};
  }

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const& problem_shape, Arguments const& args,
      void* workspace) {
    return Base::to_underlying_arguments(problem_shape,
                                         make_base_arguments(args), workspace);
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape,
                            Arguments const& args) {
    return Base::can_implement(problem_shape, make_base_arguments(args));
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape,
                                   Arguments const& args) {
    return Base::get_workspace_size(problem_shape, make_base_arguments(args));
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const& problem_shape, Arguments const& args,
      void* workspace, cudaStream_t stream,
      cutlass::CudaHostAdapter* cuda_adapter = nullptr) {
    return Base::initialize_workspace(problem_shape, make_base_arguments(args),
                                      workspace, stream, cuda_adapter);
  }

  CUTLASS_DEVICE bool vllm_proj4_warp1_consumer_needed() const {
    return true;
  }

  CUTLASS_DEVICE void vllm_proj4_warp1_consume_init(int thread_idx) {
    auto& store = get<1>(this->ops);
    store.vllm_proj4_warp1_init(thread_idx);
  }

  template <class ProducerLoadArgs>
  CUTLASS_DEVICE void vllm_proj4_warp1_consume_callback(
      ProducerLoadArgs const& args) {
    auto& store = get<1>(this->ops);
    auto callbacks = store.get_warp1_consumer_callbacks(args);

    callbacks.begin();

    CUTLASS_PRAGMA_UNROLL
    for (int iter_n = 0; iter_n < Warp1Store::kEpilogueTilesN; ++iter_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int iter_m = 0; iter_m < Warp1Store::kEpilogueTilesM; ++iter_m) {
        callbacks.step(nullptr, iter_m, iter_n, 0, false);
      }
    }

    callbacks.end();
  }
};

template <class PairCompute, class PairedOutputStore>
struct C1ActQuantPairedOutputFusion
    : cutlass::epilogue::fusion::Sm90EVT<PairedOutputStore, PairCompute> {
  using Base =
      cutlass::epilogue::fusion::Sm90EVT<PairedOutputStore, PairCompute>;
  using ElementAux = cutlass::float_e2m1_t;
  using SharedStorage = typename Base::SharedStorage;
  using Params = typename Base::Params;

  struct Arguments {
    typename PairCompute::Arguments pair{};
    typename PairedOutputStore::Arguments out{};
  };

  CUTLASS_HOST_DEVICE C1ActQuantPairedOutputFusion() {}

  CUTLASS_HOST_DEVICE
  C1ActQuantPairedOutputFusion(Params const& params,
                               SharedStorage const& shared_storage)
      : Base(params, shared_storage) {}

  static constexpr typename Base::Arguments make_base_arguments(
      Arguments const& args) {
    return typename Base::Arguments{args.pair, args.out};
  }

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const& problem_shape, Arguments const& args,
      void* workspace) {
    return Base::to_underlying_arguments(problem_shape,
                                         make_base_arguments(args), workspace);
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape,
                            Arguments const& args) {
    return Base::can_implement(problem_shape, make_base_arguments(args));
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape,
                                   Arguments const& args) {
    return Base::get_workspace_size(problem_shape, make_base_arguments(args));
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const& problem_shape, Arguments const& args,
      void* workspace, cudaStream_t stream,
      cutlass::CudaHostAdapter* cuda_adapter = nullptr) {
    return Base::initialize_workspace(problem_shape, make_base_arguments(args),
                                      workspace, stream, cuda_adapter);
  }
};

template <class Config,
          class KernelSchedule =
              cutlass::gemm::collective::KernelScheduleAuto,
          class EpilogueSchedule =
              cutlass::epilogue::collective::EpilogueScheduleAuto>
struct Fp4C1ActQuantGemmSm120 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementD = cutlass::float_e2m1_t;
  using ElementC = void;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 32;
  static constexpr int AlignmentC = 32;

  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;
  using EpilogueTile = Shape<_64, _32>;

  static constexpr int OutputSFVectorSize = 32;
  using ElementSFD = cutlass::float_ue4m3_t;
  using PairCompute =
      C1PairSiluMul<ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using BlockScaleStore = cutlass::epilogue::fusion::Sm120BlockScaleFactorRowStore<
      OutputSFVectorSize, EpilogueTile, PerSmTileShape_MNK, 4,
      cutlass::float_e2m1_t, ElementCompute, ElementSFD,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallbacks = C1ActQuantScaleFusion<PairCompute, BlockScaleStore>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          EpilogueTile, ElementAccumulator, ElementCompute, ElementC,
          LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
          EpilogueSchedule, FusionCallbacks>::CollectiveOp;
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <class ProblemShapeT, class CollectiveMainloop,
          class CollectiveEpilogue, class TileSchedulerTag>
class Task22ShallowPublishQueueGemmKernel
    : public cutlass::gemm::kernel::GemmUniversal<
          ProblemShapeT, CollectiveMainloop, CollectiveEpilogue,
          TileSchedulerTag> {
 public:
  using Base = cutlass::gemm::kernel::GemmUniversal<
      ProblemShapeT, CollectiveMainloop, CollectiveEpilogue, TileSchedulerTag>;

  static constexpr bool kTask22ShallowPublishQueueFork = true;

  using Base::operator();
};

template <class Config,
          class KernelSchedule =
              cutlass::gemm::collective::KernelScheduleAuto,
          class EpilogueSchedule =
              cutlass::epilogue::collective::EpilogueScheduleAuto>
struct Fp4C1ActQuantDirectStoreGemmSm120 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementD = void;
  using ElementC = void;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 32;
  static constexpr int AlignmentC = 32;

  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;
  using EpilogueTile = Shape<_64, _32>;

  static constexpr int OutputSFVectorSize = 32;
  using ElementSFD = cutlass::float_ue4m3_t;
  using PairCompute =
      C1PairSiluMul<ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using BlockScaleStore = cutlass::epilogue::fusion::Sm120BlockScaleFactorRowStore<
      OutputSFVectorSize, EpilogueTile, PerSmTileShape_MNK, 4,
      cutlass::float_e2m1_t, ElementCompute, ElementSFD,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using DirectPayloadStore =
      C1DirectPayloadStore<ElementCompute,
                           cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallbacks = C1ActQuantDirectStoreFusion<
      PairCompute, BlockScaleStore, DirectPayloadStore>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          EpilogueTile, ElementAccumulator, ElementCompute, ElementC,
          LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
          EpilogueSchedule, FusionCallbacks>::CollectiveOp;
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <class Config,
          class KernelSchedule =
              cutlass::gemm::collective::KernelScheduleAuto,
          class EpilogueSchedule =
              cutlass::epilogue::collective::EpilogueScheduleAuto>
struct Fp4C1ActQuantPairedOutputGemmSm120 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementD = void;
  using ElementC = void;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 32;
  static constexpr int AlignmentC = 32;

  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;
  using EpilogueTile = Shape<_64, _32>;

  static constexpr int OutputSFVectorSize = 32;
  using ElementSFD = cutlass::float_ue4m3_t;
  using PairCompute =
      C1PairSiluMul<ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using PairedOutputStore = C1PairedOutputStore<
      OutputSFVectorSize, EpilogueTile, PerSmTileShape_MNK, 4,
      cutlass::float_e2m1_t, ElementCompute, ElementSFD,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallbacks =
      C1ActQuantPairedOutputFusion<PairCompute, PairedOutputStore>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          EpilogueTile, ElementAccumulator, ElementCompute, ElementC,
          LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
          EpilogueSchedule, FusionCallbacks>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <class Config,
          class KernelSchedule =
              cutlass::gemm::collective::KernelScheduleAuto,
          class EpilogueSchedule =
              cutlass::epilogue::collective::EpilogueScheduleAuto>
struct Fp4C1ActQuantRawPublishProbeGemmSm120 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementD = void;
  using ElementC = void;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 32;
  static constexpr int AlignmentC = 32;

  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;
  using EpilogueTile = Shape<_64, _32>;

  static constexpr int OutputSFVectorSize = 32;
  using ElementSFD = cutlass::float_ue4m3_t;
  using PairCompute =
      C1PairSiluMul<ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using RawPublishStore = C1RawPublishProbeStore<
      OutputSFVectorSize, EpilogueTile, PerSmTileShape_MNK, 4,
      cutlass::float_e2m1_t, ElementCompute, ElementSFD,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallbacks =
      C1ActQuantPairedOutputFusion<PairCompute, RawPublishStore>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          EpilogueTile, ElementAccumulator, ElementCompute, ElementC,
          LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
          EpilogueSchedule, FusionCallbacks>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <class Config,
          class KernelSchedule =
              cutlass::gemm::collective::KernelScheduleAuto,
          class EpilogueSchedule =
              cutlass::epilogue::collective::EpilogueScheduleAuto>
struct Fp4C1ActQuantLoadMnConsumeProbeGemmSm120 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementD = void;
  using ElementC = void;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 32;
  static constexpr int AlignmentC = 32;

  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;
  using EpilogueTile = Shape<_64, _32>;

  static constexpr int OutputSFVectorSize = 32;
  using ElementSFD = cutlass::float_ue4m3_t;
  using PairCompute =
      C1PairSiluMul<ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using LoadMnConsumeStore = C1LoadMnConsumeProbeStore<
      OutputSFVectorSize, EpilogueTile, PerSmTileShape_MNK, 4,
      cutlass::float_e2m1_t, ElementCompute, ElementSFD,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallbacks =
      C1ActQuantPairedOutputFusion<PairCompute, LoadMnConsumeStore>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          EpilogueTile, ElementAccumulator, ElementCompute, ElementC,
          LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
          EpilogueSchedule, FusionCallbacks>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <class Config,
          class KernelSchedule =
              cutlass::gemm::collective::KernelScheduleAuto,
          class EpilogueSchedule =
              cutlass::epilogue::collective::EpilogueScheduleAuto>
struct Fp4C1ActQuantWarp1ConsumeProbeGemmSm120 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementD = void;
  using ElementC = void;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 32;
  static constexpr int AlignmentC = 32;

  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;
  using EpilogueTile = Shape<_64, _32>;

  static constexpr int OutputSFVectorSize = 32;
  using ElementSFD = cutlass::float_ue4m3_t;
  using PairCompute =
      C1PairSiluMul<ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using Warp1ConsumeStore = C1Warp1ConsumeProbeStore<
      OutputSFVectorSize, EpilogueTile, PerSmTileShape_MNK, 4,
      cutlass::float_e2m1_t, ElementCompute, ElementSFD,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallbacks =
      C1ActQuantWarp1ConsumeProbeFusion<PairCompute, Warp1ConsumeStore>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          EpilogueTile, ElementAccumulator, ElementCompute, ElementC,
          LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
          EpilogueSchedule, FusionCallbacks>::CollectiveOp;
  static_assert(cute::is_same_v<typename CollectiveEpilogue::FusionCallbacks,
                                FusionCallbacks>);

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::StaticPersistentScheduler>;
  static_assert(!GemmKernel::IsSchedDynamicPersistent);
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <class Config,
          class KernelSchedule =
              cutlass::gemm::collective::KernelScheduleAuto,
          class EpilogueSchedule =
              cutlass::epilogue::collective::EpilogueScheduleAuto,
          bool FullSlotsNoProducerWait = false>
struct Fp4C1ActQuantWarp1ValidProbeGemmSm120 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementD = void;
  using ElementC = void;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 32;
  static constexpr int AlignmentC = 32;

  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;
  using EpilogueTile = Shape<_64, _32>;

  static constexpr int OutputSFVectorSize = 32;
  using ElementSFD = cutlass::float_ue4m3_t;
  using PairCompute =
      C1PairSiluMul<ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using Warp1ValidStore = C1Warp1ValidOutputStore<
      OutputSFVectorSize, EpilogueTile, PerSmTileShape_MNK, 4,
      cutlass::float_e2m1_t, ElementCompute, ElementSFD,
      cutlass::FloatRoundStyle::round_to_nearest,
      FullSlotsNoProducerWait>;
  using FusionCallbacks =
      C1ActQuantWarp1ConsumeProbeFusion<PairCompute, Warp1ValidStore>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          EpilogueTile, ElementAccumulator, ElementCompute, ElementC,
          LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
          EpilogueSchedule, FusionCallbacks>::CollectiveOp;
  static_assert(cute::is_same_v<typename CollectiveEpilogue::FusionCallbacks,
                                FusionCallbacks>);

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::StaticPersistentScheduler>;
  static_assert(!GemmKernel::IsSchedDynamicPersistent);
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <class Config,
          class KernelSchedule =
              cutlass::gemm::collective::KernelScheduleAuto,
          class EpilogueSchedule =
              cutlass::epilogue::collective::EpilogueScheduleAuto>
struct Fp4C1ActQuantWarp1LoadMnValidProbeGemmSm120 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementD = void;
  using ElementC = void;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 32;
  static constexpr int AlignmentC = 32;

  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;
  using EpilogueTile = Shape<_64, _32>;

  static constexpr int OutputSFVectorSize = 32;
  using ElementSFD = cutlass::float_ue4m3_t;
  using PairCompute =
      C1PairSiluMul<ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using Warp1LoadMnValidStore = C1Warp1LoadMnValidOutputStore<
      OutputSFVectorSize, EpilogueTile, PerSmTileShape_MNK, 4,
      cutlass::float_e2m1_t, ElementCompute, ElementSFD,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallbacks =
      C1ActQuantWarp1ConsumeProbeFusion<PairCompute, Warp1LoadMnValidStore>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          EpilogueTile, ElementAccumulator, ElementCompute, ElementC,
          LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
          EpilogueSchedule, FusionCallbacks>::CollectiveOp;
  static_assert(cute::is_same_v<typename CollectiveEpilogue::FusionCallbacks,
                                FusionCallbacks>);

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::StaticPersistentScheduler>;
  static_assert(!GemmKernel::IsSchedDynamicPersistent);
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <class Config,
          class KernelSchedule =
              cutlass::gemm::collective::KernelScheduleAuto,
          class EpilogueSchedule =
              cutlass::epilogue::collective::EpilogueScheduleAuto>
struct Fp4C1ActQuantPairedOutputShallowQueueGemmSm120
    : Fp4C1ActQuantPairedOutputGemmSm120<Config, KernelSchedule,
                                         EpilogueSchedule> {
  using Base = Fp4C1ActQuantPairedOutputGemmSm120<
      Config, KernelSchedule, EpilogueSchedule>;
  using GemmKernel = Task22ShallowPublishQueueGemmKernel<
      Shape<int, int, int, int>, typename Base::CollectiveMainloop,
      typename Base::CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

struct sm120_fp4_c1_config_M256 {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShape_MNK = Shape<_128, _128, _128>;
};

struct sm120_fp4_c1_config_default {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_256, _128, _128>;
  using PerSmTileShape_MNK = Shape<_256, _128, _128>;
};

template <typename Gemm>
typename Gemm::Arguments args_from_options(
    torch::stable::Tensor& act_payload,
    torch::stable::Tensor& interleaved_act_payload,
    torch::stable::Tensor const& act_sf,
    torch::stable::Tensor const& A, torch::stable::Tensor const& B,
    torch::stable::Tensor const& A_sf, torch::stable::Tensor const& B_sf,
    torch::stable::Tensor const& c1_alpha,
    torch::stable::Tensor const& c2_input_global_scale_inv, int M, int N,
    int K, int logical_act_cols, bool use_bfloat16_intermediate) {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementSFD = cutlass::float_ue4m3_t;
  using ElementCompute = float;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(M, N, K, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {static_cast<ElementA const*>(A.data_ptr()), stride_A,
       static_cast<ElementB const*>(B.data_ptr()), stride_B,
       static_cast<ElementSFA const*>(A_sf.data_ptr()), layout_SFA,
       static_cast<ElementSFB const*>(B_sf.data_ptr()), layout_SFB},
      {{},
       nullptr,
       stride_C,
       static_cast<ElementD*>(interleaved_act_payload.data_ptr()),
       stride_D}};

  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.pair.alpha_ptr =
      static_cast<ElementCompute const*>(c1_alpha.data_ptr());
  fusion_args.pair.intermediate_dtype = use_bfloat16_intermediate ? 1 : 0;
  fusion_args.sf.ptr_scale_factor =
      static_cast<ElementSFD*>(act_sf.data_ptr());
  fusion_args.sf.norm_constant_ptr =
      static_cast<ElementCompute const*>(c2_input_global_scale_inv.data_ptr());

  return arguments;
}

template <typename Gemm>
void runGemm(torch::stable::Tensor& act_payload,
             torch::stable::Tensor& interleaved_act_payload,
             torch::stable::Tensor const& act_sf,
             torch::stable::Tensor const& A, torch::stable::Tensor const& B,
             torch::stable::Tensor const& A_sf,
             torch::stable::Tensor const& B_sf,
             torch::stable::Tensor const& c1_alpha,
             torch::stable::Tensor const& c2_input_global_scale_inv, int M,
             int N, int K, int logical_act_cols,
             bool use_bfloat16_intermediate, cudaStream_t stream,
             bool launch_with_pdl = false) {
  Gemm gemm;
  auto arguments = args_from_options<Gemm>(
      act_payload, interleaved_act_payload, act_sf, A, B, A_sf, B_sf,
      c1_alpha, c2_input_global_scale_inv, M, N, K, logical_act_cols,
      use_bfloat16_intermediate);

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace =
      torch::stable::empty(workspace_size, torch::headeronly::ScalarType::Byte,
                           std::nullopt, A.device());

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  CUTLASS_CHECK(
      gemm.run(arguments, workspace.data_ptr(), stream, nullptr,
               launch_with_pdl));
}

template <typename Gemm>
typename Gemm::Arguments direct_args_from_options(
    torch::stable::Tensor& act_payload,
    torch::stable::Tensor const& act_sf,
    torch::stable::Tensor const& A, torch::stable::Tensor const& B,
    torch::stable::Tensor const& A_sf, torch::stable::Tensor const& B_sf,
    torch::stable::Tensor const& c1_alpha,
    torch::stable::Tensor const& c2_input_global_scale_inv, int M, int N,
    int K, int logical_act_cols, bool use_bfloat16_intermediate,
    int32_t store_mode) {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementSFD = cutlass::float_ue4m3_t;
  using ElementCompute = float;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(M, N, K, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {static_cast<ElementA const*>(A.data_ptr()), stride_A,
       static_cast<ElementB const*>(B.data_ptr()), stride_B,
       static_cast<ElementSFA const*>(A_sf.data_ptr()), layout_SFA,
       static_cast<ElementSFB const*>(B_sf.data_ptr()), layout_SFB},
      {{},
       nullptr,
       stride_C,
       static_cast<ElementD const*>(nullptr),
       stride_D}};

  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.pair.alpha_ptr =
      static_cast<ElementCompute const*>(c1_alpha.data_ptr());
  fusion_args.pair.intermediate_dtype = use_bfloat16_intermediate ? 1 : 0;
  fusion_args.sf.ptr_scale_factor =
      static_cast<ElementSFD*>(act_sf.data_ptr());
  fusion_args.sf.norm_constant_ptr =
      static_cast<ElementCompute const*>(c2_input_global_scale_inv.data_ptr());
  fusion_args.direct.ptr_payload =
      static_cast<uint8_t*>(act_payload.data_ptr());
  fusion_args.direct.stride_bytes = act_payload.size(1);
  fusion_args.direct.store_mode = store_mode;

  return arguments;
}

template <typename Gemm>
void runGemmDirect(torch::stable::Tensor& act_payload,
                   torch::stable::Tensor const& act_sf,
                   torch::stable::Tensor const& A,
                   torch::stable::Tensor const& B,
                   torch::stable::Tensor const& A_sf,
                   torch::stable::Tensor const& B_sf,
                   torch::stable::Tensor const& c1_alpha,
                   torch::stable::Tensor const& c2_input_global_scale_inv,
                   int M, int N, int K, int logical_act_cols,
                   bool use_bfloat16_intermediate, int32_t store_mode,
                   cudaStream_t stream, bool launch_with_pdl = false) {
  Gemm gemm;
  auto arguments = direct_args_from_options<Gemm>(
      act_payload, act_sf, A, B, A_sf, B_sf, c1_alpha,
      c2_input_global_scale_inv, M, N, K, logical_act_cols,
      use_bfloat16_intermediate, store_mode);

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace =
      torch::stable::empty(workspace_size, torch::headeronly::ScalarType::Byte,
                           std::nullopt, A.device());

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  CUTLASS_CHECK(
      gemm.run(arguments, workspace.data_ptr(), stream, nullptr,
               launch_with_pdl));
}

template <typename Gemm>
typename Gemm::Arguments paired_output_args_from_options(
    torch::stable::Tensor& act_payload,
    torch::stable::Tensor const& act_sf,
    torch::stable::Tensor const& A, torch::stable::Tensor const& B,
    torch::stable::Tensor const& A_sf, torch::stable::Tensor const& B_sf,
    torch::stable::Tensor const& c1_alpha,
    torch::stable::Tensor const& c2_input_global_scale_inv, int M, int N,
    int K, int logical_act_cols, bool use_bfloat16_intermediate) {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementSFD = cutlass::float_ue4m3_t;
  using ElementCompute = float;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(M, N, K, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {static_cast<ElementA const*>(A.data_ptr()), stride_A,
       static_cast<ElementB const*>(B.data_ptr()), stride_B,
       static_cast<ElementSFA const*>(A_sf.data_ptr()), layout_SFA,
       static_cast<ElementSFB const*>(B_sf.data_ptr()), layout_SFB},
      {{},
       nullptr,
       stride_C,
       static_cast<ElementD const*>(nullptr),
       stride_D}};

  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.pair.alpha_ptr =
      static_cast<ElementCompute const*>(c1_alpha.data_ptr());
  fusion_args.pair.intermediate_dtype = use_bfloat16_intermediate ? 1 : 0;
  fusion_args.out.ptr_scale_factor =
      static_cast<ElementSFD*>(act_sf.data_ptr());
  fusion_args.out.norm_constant_ptr =
      static_cast<ElementCompute const*>(c2_input_global_scale_inv.data_ptr());
  fusion_args.out.ptr_payload = static_cast<uint8_t*>(act_payload.data_ptr());
  fusion_args.out.stride_bytes = act_payload.size(1);
  fusion_args.out.debug_timing_mode = get_c1_paired_output_debug_mode();

  return arguments;
}

template <typename Gemm>
void runGemmPairedOutput(torch::stable::Tensor& act_payload,
                         torch::stable::Tensor const& act_sf,
                         torch::stable::Tensor const& A,
                         torch::stable::Tensor const& B,
                         torch::stable::Tensor const& A_sf,
                         torch::stable::Tensor const& B_sf,
                         torch::stable::Tensor const& c1_alpha,
                         torch::stable::Tensor const& c2_input_global_scale_inv,
                         int M, int N, int K, int logical_act_cols,
                         bool use_bfloat16_intermediate,
                         cudaStream_t stream,
                         bool launch_with_pdl = false) {
  Gemm gemm;
  auto arguments = paired_output_args_from_options<Gemm>(
      act_payload, act_sf, A, B, A_sf, B_sf, c1_alpha,
      c2_input_global_scale_inv, M, N, K, logical_act_cols,
      use_bfloat16_intermediate);

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace =
      torch::stable::empty(workspace_size, torch::headeronly::ScalarType::Byte,
                           std::nullopt, A.device());

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  CUTLASS_CHECK(
      gemm.run(arguments, workspace.data_ptr(), stream, nullptr,
               launch_with_pdl));
}

template <class Config,
          class KernelSchedule =
              cutlass::gemm::collective::KernelScheduleAuto,
          class EpilogueSchedule =
              cutlass::epilogue::collective::EpilogueScheduleAuto>
void runGemmPairedOutputMaybeShallowQueue(
    C1PairedOutputKernelForkMode fork_mode,
    torch::stable::Tensor& act_payload, torch::stable::Tensor const& act_sf,
    torch::stable::Tensor const& A, torch::stable::Tensor const& B,
    torch::stable::Tensor const& A_sf, torch::stable::Tensor const& B_sf,
    torch::stable::Tensor const& c1_alpha,
    torch::stable::Tensor const& c2_input_global_scale_inv, int M, int N,
    int K, int logical_act_cols, bool use_bfloat16_intermediate,
    cudaStream_t stream, bool launch_with_pdl = false) {
  if (fork_mode == C1PairedOutputKernelForkMode::kShallowQueue) {
    using Gemm =
        typename Fp4C1ActQuantPairedOutputShallowQueueGemmSm120<
            Config, KernelSchedule, EpilogueSchedule>::Gemm;
    runGemmPairedOutput<Gemm>(
        act_payload, act_sf, A, B, A_sf, B_sf, c1_alpha,
        c2_input_global_scale_inv, M, N, K, logical_act_cols,
        use_bfloat16_intermediate, stream, launch_with_pdl);
  } else if (fork_mode == C1PairedOutputKernelForkMode::kRawPublishProbe) {
    using Gemm =
        typename Fp4C1ActQuantRawPublishProbeGemmSm120<
            Config, KernelSchedule, EpilogueSchedule>::Gemm;
    runGemmPairedOutput<Gemm>(
        act_payload, act_sf, A, B, A_sf, B_sf, c1_alpha,
        c2_input_global_scale_inv, M, N, K, logical_act_cols,
        use_bfloat16_intermediate, stream, launch_with_pdl);
  } else if (fork_mode ==
             C1PairedOutputKernelForkMode::kLoadMnConsumeProbe) {
    using Gemm =
        typename Fp4C1ActQuantLoadMnConsumeProbeGemmSm120<
            Config, KernelSchedule, EpilogueSchedule>::Gemm;
    runGemmPairedOutput<Gemm>(
        act_payload, act_sf, A, B, A_sf, B_sf, c1_alpha,
        c2_input_global_scale_inv, M, N, K, logical_act_cols,
        use_bfloat16_intermediate, stream, launch_with_pdl);
  } else if (fork_mode ==
             C1PairedOutputKernelForkMode::kWarp1ConsumeProbe) {
    using Gemm =
        typename Fp4C1ActQuantWarp1ConsumeProbeGemmSm120<
            Config, KernelSchedule, EpilogueSchedule>::Gemm;
    runGemmPairedOutput<Gemm>(
        act_payload, act_sf, A, B, A_sf, B_sf, c1_alpha,
        c2_input_global_scale_inv, M, N, K, logical_act_cols,
        use_bfloat16_intermediate, stream, launch_with_pdl);
  } else if (fork_mode ==
             C1PairedOutputKernelForkMode::kWarp1ValidProbe) {
    using Gemm =
        typename Fp4C1ActQuantWarp1ValidProbeGemmSm120<
            Config, KernelSchedule, EpilogueSchedule>::Gemm;
    runGemmPairedOutput<Gemm>(
        act_payload, act_sf, A, B, A_sf, B_sf, c1_alpha,
        c2_input_global_scale_inv, M, N, K, logical_act_cols,
        use_bfloat16_intermediate, stream, launch_with_pdl);
  } else if (fork_mode ==
             C1PairedOutputKernelForkMode::kWarp1ValidNoWaitProbe) {
    using Gemm =
        typename Fp4C1ActQuantWarp1ValidProbeGemmSm120<
            Config, KernelSchedule, EpilogueSchedule, true>::Gemm;
    runGemmPairedOutput<Gemm>(
        act_payload, act_sf, A, B, A_sf, B_sf, c1_alpha,
        c2_input_global_scale_inv, M, N, K, logical_act_cols,
        use_bfloat16_intermediate, stream, launch_with_pdl);
  } else if (fork_mode ==
             C1PairedOutputKernelForkMode::kWarp1LoadMnValidProbe) {
    using Gemm =
        typename Fp4C1ActQuantWarp1LoadMnValidProbeGemmSm120<
            Config, KernelSchedule, EpilogueSchedule>::Gemm;
    runGemmPairedOutput<Gemm>(
        act_payload, act_sf, A, B, A_sf, B_sf, c1_alpha,
        c2_input_global_scale_inv, M, N, K, logical_act_cols,
        use_bfloat16_intermediate, stream, launch_with_pdl);
  } else {
    using Gemm = typename Fp4C1ActQuantPairedOutputGemmSm120<
        Config, KernelSchedule, EpilogueSchedule>::Gemm;
    runGemmPairedOutput<Gemm>(
        act_payload, act_sf, A, B, A_sf, B_sf, c1_alpha,
        c2_input_global_scale_inv, M, N, K, logical_act_cols,
        use_bfloat16_intermediate, stream, launch_with_pdl);
  }
}

}  // namespace vllm::proj4

namespace cutlass::detail {
template <>
struct TagToStrideA<vllm::proj4::RowMajorEvenK> {
  using type = cute::Stride<int64_t, cute::Int<2>, int64_t>;
  using tag = vllm::proj4::RowMajorEvenK;
};
}  // namespace cutlass::detail

namespace cutlass::gemm::collective::detail {
template <>
constexpr cute::UMMA::Major
tag_to_umma_major_A<vllm::proj4::RowMajorEvenK>() {
  return cute::UMMA::Major::K;
}
}  // namespace cutlass::gemm::collective::detail

namespace vllm::proj4 {

template <class Config, typename OutType>
struct Fp4C2InterleavedAGemmSm120 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = RowMajorEvenK;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementD = OutType;
  using ElementC = OutType;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD,
          LayoutDTag, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Gemm>
typename Gemm::Arguments interleaved_a_args_from_options(
    torch::stable::Tensor& D, torch::stable::Tensor const& A,
    torch::stable::Tensor const& B, torch::stable::Tensor const& A_sf,
    torch::stable::Tensor const& B_sf, torch::stable::Tensor const& alpha,
    int M, int N, int K) {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementCompute = float;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  StrideA stride_A =
      make_stride(static_cast<int64_t>(K * 2), cute::Int<2>{}, int64_t(0));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(M, N, K, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {static_cast<ElementA const*>(A.data_ptr()), stride_A,
       static_cast<ElementB const*>(B.data_ptr()), stride_B,
       static_cast<ElementSFA const*>(A_sf.data_ptr()), layout_SFA,
       static_cast<ElementSFB const*>(B_sf.data_ptr()), layout_SFB},
      {{},
       static_cast<ElementD const*>(D.data_ptr()),
       stride_D,
       static_cast<ElementD*>(D.data_ptr()),
       stride_D}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(alpha.data_ptr());
  return arguments;
}

template <typename Gemm>
void runGemmInterleavedA(torch::stable::Tensor& D,
                         torch::stable::Tensor const& A,
                         torch::stable::Tensor const& B,
                         torch::stable::Tensor const& A_sf,
                         torch::stable::Tensor const& B_sf,
                         torch::stable::Tensor const& alpha, int M, int N,
                         int K, cudaStream_t stream) {
  Gemm gemm;
  auto arguments =
      interleaved_a_args_from_options<Gemm>(D, A, B, A_sf, B_sf, alpha, M, N, K);

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace =
      torch::stable::empty(workspace_size, torch::headeronly::ScalarType::Byte,
                           std::nullopt, A.device());

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}

__global__ void compact_interleaved_c1_fp4_payload_kernel(
    uint8_t* __restrict__ dst, uint8_t const* __restrict__ src,
    int64_t rows, int64_t logical_cols, int64_t dst_stride_bytes,
    int64_t src_stride_bytes) {
  int64_t packed_cols = logical_cols / 2;
  int64_t total = rows * packed_cols;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t row = idx / packed_cols;
    int64_t dst_byte_col = idx - row * packed_cols;
    int64_t act_col0 = dst_byte_col * 2;
    int64_t act_col1 = act_col0 + 1;

    uint8_t const* src_row = src + row * src_stride_bytes;
    uint8_t nibble0 = src_row[act_col0] & 0x0f;
    uint8_t nibble1 = src_row[act_col1] & 0x0f;
    dst[row * dst_stride_bytes + dst_byte_col] =
        static_cast<uint8_t>(nibble0 | (nibble1 << 4));
  }
}

}  // namespace vllm::proj4

void compact_interleaved_c1_fp4_payload_sm120a_stream(
    torch::stable::Tensor& act_payload,
    torch::stable::Tensor const& interleaved_payload,
    int64_t logical_act_cols,
    cudaStream_t stream) {
  CHECK_INPUT(act_payload, FLOAT4_E2M1X2, "act_payload");
  CHECK_INPUT(interleaved_payload, FLOAT4_E2M1X2, "interleaved_payload");
  STD_TORCH_CHECK(logical_act_cols > 0 && logical_act_cols % 2 == 0,
                  "logical_act_cols must be positive and even");
  STD_TORCH_CHECK(act_payload.dim() == 2 && interleaved_payload.dim() == 2,
                  "payload tensors must be matrices");
  STD_TORCH_CHECK(act_payload.size(0) == interleaved_payload.size(0),
                  "payload row counts must match");
  STD_TORCH_CHECK(act_payload.size(1) >= logical_act_cols / 2,
                  "act_payload is too narrow");
  STD_TORCH_CHECK(interleaved_payload.size(1) >= logical_act_cols,
                  "interleaved_payload is too narrow");

  const torch::stable::accelerator::DeviceGuard device_guard(
      act_payload.get_device_index());
  int64_t rows = act_payload.size(0);
  int threads = 256;
  int64_t total = rows * (logical_act_cols / 2);
  int blocks = static_cast<int>(
      std::min<int64_t>(65535, (total + threads - 1) / threads));
  vllm::proj4::compact_interleaved_c1_fp4_payload_kernel<<<
      blocks, threads, 0, stream>>>(
      static_cast<uint8_t*>(act_payload.data_ptr()),
      static_cast<uint8_t const*>(interleaved_payload.data_ptr()), rows,
      logical_act_cols, act_payload.size(1), interleaved_payload.size(1));
}

void compact_interleaved_c1_fp4_payload_sm120a(
    torch::stable::Tensor& act_payload,
    torch::stable::Tensor const& interleaved_payload,
    int64_t logical_act_cols) {
  const torch::stable::accelerator::DeviceGuard device_guard(
      act_payload.get_device_index());
  auto stream = get_current_cuda_stream(act_payload.get_device_index());
  compact_interleaved_c1_fp4_payload_sm120a_stream(
      act_payload, interleaved_payload, logical_act_cols, stream);
}

void cutlass_nvfp4_mlp_c1_act_quant_sm120a_impl(
    torch::stable::Tensor& act_payload,
    torch::stable::Tensor& interleaved_act_payload,
    torch::stable::Tensor& act_sf,
    torch::stable::Tensor const& A, torch::stable::Tensor const& B,
    torch::stable::Tensor const& A_sf, torch::stable::Tensor const& B_sf,
    torch::stable::Tensor const& c1_alpha,
    torch::stable::Tensor const& c2_input_global_scale_inv,
    int64_t logical_act_cols, bool use_bfloat16_intermediate,
    cudaStream_t stream, bool run_compact, bool launch_with_pdl) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  CHECK_INPUT(act_payload, FLOAT4_E2M1X2, "act_payload");
  CHECK_INPUT(interleaved_act_payload, FLOAT4_E2M1X2,
              "interleaved_act_payload");
  CHECK_INPUT(act_sf, SF_DTYPE, "act_sf");
  CHECK_INPUT(A, FLOAT4_E2M1X2, "a");
  CHECK_INPUT(B, FLOAT4_E2M1X2, "b");
  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");
  CHECK_INPUT(c1_alpha, torch::headeronly::ScalarType::Float, "c1_alpha");
  CHECK_INPUT(c2_input_global_scale_inv,
              torch::headeronly::ScalarType::Float,
              "c2_input_global_scale_inv");

  STD_TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  STD_TORCH_CHECK(B.dim() == 2, "b must be a matrix");
  STD_TORCH_CHECK(A.size(1) == B.size(1),
                  "a and b shapes cannot be multiplied (", A.size(0), "x",
                  A.size(1), " and ", B.size(0), "x", B.size(1), ")");
  STD_TORCH_CHECK(B.size(0) % 2 == 0,
                  "interleaved C1 weight rows must be even");
  STD_TORCH_CHECK(logical_act_cols > 0,
                  "logical_act_cols must be positive");
  STD_TORCH_CHECK(logical_act_cols % 16 == 0,
                  "logical_act_cols must be divisible by 16");

  auto const m = static_cast<int>(A.size(0));
  auto const n = static_cast<int>(B.size(0));
  auto const k = static_cast<int>(A.size(1) * 2);

  STD_TORCH_CHECK(logical_act_cols * 2 <= n,
                  "logical activation columns exceed C1 output columns");
  STD_TORCH_CHECK(act_payload.size(0) == m,
                  "act_payload row count must match input rows");
  STD_TORCH_CHECK(act_payload.size(1) >= logical_act_cols / 2,
                  "act_payload is too narrow for compact C1 FP4 columns");
  STD_TORCH_CHECK(interleaved_act_payload.size(0) == m,
                  "interleaved_act_payload row count must match input rows");
  STD_TORCH_CHECK(interleaved_act_payload.size(1) >= n / 2,
                  "interleaved_act_payload is too narrow for C1 D store");

  constexpr int alignment = 32;
  STD_TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ",
                  alignment, ", but got k=", k);
  STD_TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ",
                  alignment, ", but got n=", n);
  STD_TORCH_CHECK((n % 32) == 0,
                  "C1 interleaved N must be divisible by 32 so 32 C1 "
                  "columns map to 16 C2 activation columns per scale.");

  auto round_up = [](int x, int y) { return (x + y - 1) / y * y; };
  int rounded_m = round_up(m, 128);
  int rounded_n = round_up(logical_act_cols / 16, 4);
  STD_TORCH_CHECK(act_sf.dim() == 2, "act_sf must be a matrix");
  STD_TORCH_CHECK(
      act_sf.size(0) == rounded_m && act_sf.size(1) == rounded_n,
      "act_sf must be padded and swizzled to shape (", rounded_m, "x",
      rounded_n, "), got (", act_sf.size(0), "x", act_sf.size(1), ")");

  const torch::stable::accelerator::DeviceGuard device_guard(
      A.get_device_index());

  uint32_t const mp2 = std::max(static_cast<uint32_t>(16), next_pow_2(m));
  if (mp2 <= 256) {
    using Gemm =
        vllm::proj4::Fp4C1ActQuantGemmSm120<
            vllm::proj4::sm120_fp4_c1_config_M256>::Gemm;
    vllm::proj4::runGemm<Gemm>(
        act_payload, interleaved_act_payload, act_sf, A, B, A_sf, B_sf,
        c1_alpha, c2_input_global_scale_inv, m, n, k,
        static_cast<int>(logical_act_cols), use_bfloat16_intermediate, stream,
        launch_with_pdl);
  } else {
    using Gemm =
        vllm::proj4::Fp4C1ActQuantGemmSm120<
            vllm::proj4::sm120_fp4_c1_config_default>::Gemm;
    vllm::proj4::runGemm<Gemm>(
        act_payload, interleaved_act_payload, act_sf, A, B, A_sf, B_sf,
        c1_alpha, c2_input_global_scale_inv, m, n, k,
        static_cast<int>(logical_act_cols), use_bfloat16_intermediate, stream,
        launch_with_pdl);
  }
  if (run_compact) {
    compact_interleaved_c1_fp4_payload_sm120a_stream(
        act_payload, interleaved_act_payload, logical_act_cols, stream);
  }
#else
  STD_TORCH_CHECK(false,
                  "Unsupported CUTLASS version. Set VLLM_CUTLASS_SRC_DIR to "
                  "a CUTLASS 3.8+ source directory to enable support.");
#endif
}

void cutlass_nvfp4_mlp_c1_act_quant_sm120a(
    torch::stable::Tensor& act_payload,
    torch::stable::Tensor& interleaved_act_payload,
    torch::stable::Tensor& act_sf,
    torch::stable::Tensor const& A, torch::stable::Tensor const& B,
    torch::stable::Tensor const& A_sf, torch::stable::Tensor const& B_sf,
    torch::stable::Tensor const& c1_alpha,
    torch::stable::Tensor const& c2_input_global_scale_inv,
    int64_t logical_act_cols, bool use_bfloat16_intermediate,
    bool launch_with_pdl) {
  const torch::stable::accelerator::DeviceGuard device_guard(
      A.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(A.get_device_index());
  cutlass_nvfp4_mlp_c1_act_quant_sm120a_impl(
      act_payload, interleaved_act_payload, act_sf, A, B, A_sf, B_sf,
      c1_alpha, c2_input_global_scale_inv, logical_act_cols,
      use_bfloat16_intermediate, stream, true, launch_with_pdl);
}

void cutlass_nvfp4_mlp_c1_act_quant_sm120a_stream(
    torch::stable::Tensor& act_payload,
    torch::stable::Tensor& interleaved_act_payload,
    torch::stable::Tensor& act_sf,
    torch::stable::Tensor const& A, torch::stable::Tensor const& B,
    torch::stable::Tensor const& A_sf, torch::stable::Tensor const& B_sf,
    torch::stable::Tensor const& c1_alpha,
    torch::stable::Tensor const& c2_input_global_scale_inv,
    int64_t logical_act_cols, bool use_bfloat16_intermediate,
    cudaStream_t stream, bool launch_with_pdl) {
  cutlass_nvfp4_mlp_c1_act_quant_sm120a_impl(
      act_payload, interleaved_act_payload, act_sf, A, B, A_sf, B_sf,
      c1_alpha, c2_input_global_scale_inv, logical_act_cols,
      use_bfloat16_intermediate, stream, true, launch_with_pdl);
}

void cutlass_nvfp4_mlp_c1_act_quant_interleaved_sm120a(
    torch::stable::Tensor& act_payload,
    torch::stable::Tensor& interleaved_act_payload,
    torch::stable::Tensor& act_sf,
    torch::stable::Tensor const& A, torch::stable::Tensor const& B,
    torch::stable::Tensor const& A_sf, torch::stable::Tensor const& B_sf,
    torch::stable::Tensor const& c1_alpha,
    torch::stable::Tensor const& c2_input_global_scale_inv,
    int64_t logical_act_cols, bool use_bfloat16_intermediate,
    bool launch_with_pdl) {
  const torch::stable::accelerator::DeviceGuard device_guard(
      A.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(A.get_device_index());
  cutlass_nvfp4_mlp_c1_act_quant_sm120a_impl(
      act_payload, interleaved_act_payload, act_sf, A, B, A_sf, B_sf,
      c1_alpha, c2_input_global_scale_inv, logical_act_cols,
      use_bfloat16_intermediate, stream, false, launch_with_pdl);
}

void cutlass_nvfp4_mlp_c1_act_quant_interleaved_sm120a_stream(
    torch::stable::Tensor& act_payload,
    torch::stable::Tensor& interleaved_act_payload,
    torch::stable::Tensor& act_sf,
    torch::stable::Tensor const& A, torch::stable::Tensor const& B,
    torch::stable::Tensor const& A_sf, torch::stable::Tensor const& B_sf,
    torch::stable::Tensor const& c1_alpha,
    torch::stable::Tensor const& c2_input_global_scale_inv,
    int64_t logical_act_cols, bool use_bfloat16_intermediate,
    cudaStream_t stream, bool launch_with_pdl) {
  cutlass_nvfp4_mlp_c1_act_quant_sm120a_impl(
      act_payload, interleaved_act_payload, act_sf, A, B, A_sf, B_sf,
      c1_alpha, c2_input_global_scale_inv, logical_act_cols,
      use_bfloat16_intermediate, stream, false, launch_with_pdl);
}

void cutlass_nvfp4_mlp_c1_act_quant_direct_store_sm120a(
    torch::stable::Tensor& act_payload,
    torch::stable::Tensor& act_sf,
    torch::stable::Tensor const& A, torch::stable::Tensor const& B,
    torch::stable::Tensor const& A_sf, torch::stable::Tensor const& B_sf,
    torch::stable::Tensor const& c1_alpha,
    torch::stable::Tensor const& c2_input_global_scale_inv,
    int64_t logical_act_cols, bool use_bfloat16_intermediate,
    int32_t store_mode, bool launch_with_pdl) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  CHECK_INPUT(act_payload, FLOAT4_E2M1X2, "act_payload");
  CHECK_INPUT(act_sf, SF_DTYPE, "act_sf");
  CHECK_INPUT(A, FLOAT4_E2M1X2, "a");
  CHECK_INPUT(B, FLOAT4_E2M1X2, "b");
  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");
  CHECK_INPUT(c1_alpha, torch::headeronly::ScalarType::Float, "c1_alpha");
  CHECK_INPUT(c2_input_global_scale_inv,
              torch::headeronly::ScalarType::Float,
              "c2_input_global_scale_inv");
  STD_TORCH_CHECK(store_mode >= 0 && store_mode <= 5 && store_mode != 3,
                  "direct-store mode must be 0, 1, 2, 4, or 5");

  STD_TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  STD_TORCH_CHECK(B.dim() == 2, "b must be a matrix");
  STD_TORCH_CHECK(A.size(1) == B.size(1),
                  "a and b shapes cannot be multiplied (", A.size(0), "x",
                  A.size(1), " and ", B.size(0), "x", B.size(1), ")");
  STD_TORCH_CHECK(B.size(0) % 2 == 0,
                  "interleaved C1 weight rows must be even");
  STD_TORCH_CHECK(logical_act_cols > 0,
                  "logical_act_cols must be positive");
  STD_TORCH_CHECK(logical_act_cols % 16 == 0,
                  "logical_act_cols must be divisible by 16");

  auto const m = static_cast<int>(A.size(0));
  auto const n = static_cast<int>(B.size(0));
  auto const k = static_cast<int>(A.size(1) * 2);

  STD_TORCH_CHECK(logical_act_cols * 2 <= n,
                  "logical activation columns exceed C1 output columns");
  STD_TORCH_CHECK(act_payload.size(0) == m,
                  "act_payload row count must match input rows");
  STD_TORCH_CHECK(act_payload.size(1) >= logical_act_cols / 2,
                  "act_payload is too narrow for compact C1 FP4 columns");

  constexpr int alignment = 32;
  STD_TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ",
                  alignment, ", but got k=", k);
  STD_TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ",
                  alignment, ", but got n=", n);
  STD_TORCH_CHECK((n % 32) == 0,
                  "C1 interleaved N must be divisible by 32 so 32 C1 "
                  "columns map to 16 C2 activation columns per scale.");

  auto round_up = [](int x, int y) { return (x + y - 1) / y * y; };
  int rounded_m = round_up(m, 128);
  int rounded_n = round_up(logical_act_cols / 16, 4);
  STD_TORCH_CHECK(act_sf.dim() == 2, "act_sf must be a matrix");
  STD_TORCH_CHECK(
      act_sf.size(0) == rounded_m && act_sf.size(1) == rounded_n,
      "act_sf must be padded and swizzled to shape (", rounded_m, "x",
      rounded_n, "), got (", act_sf.size(0), "x", act_sf.size(1), ")");

  const torch::stable::accelerator::DeviceGuard device_guard(
      A.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(A.get_device_index());

  uint32_t const mp2 = std::max(static_cast<uint32_t>(16), next_pow_2(m));
  if (mp2 <= 256) {
    using Gemm =
        vllm::proj4::Fp4C1ActQuantDirectStoreGemmSm120<
            vllm::proj4::sm120_fp4_c1_config_M256>::Gemm;
    vllm::proj4::runGemmDirect<Gemm>(
        act_payload, act_sf, A, B, A_sf, B_sf, c1_alpha,
        c2_input_global_scale_inv, m, n, k,
        static_cast<int>(logical_act_cols), use_bfloat16_intermediate,
        store_mode, stream, launch_with_pdl);
  } else {
    using Gemm =
        vllm::proj4::Fp4C1ActQuantDirectStoreGemmSm120<
            vllm::proj4::sm120_fp4_c1_config_default>::Gemm;
    vllm::proj4::runGemmDirect<Gemm>(
        act_payload, act_sf, A, B, A_sf, B_sf, c1_alpha,
        c2_input_global_scale_inv, m, n, k,
        static_cast<int>(logical_act_cols), use_bfloat16_intermediate,
        store_mode, stream, launch_with_pdl);
  }
#else
  STD_TORCH_CHECK(false,
                  "Unsupported CUTLASS version. Set VLLM_CUTLASS_SRC_DIR to "
                  "a CUTLASS 3.8+ source directory to enable support.");
#endif
}

void cutlass_nvfp4_mlp_c1_act_quant_paired_output_sm120a(
    torch::stable::Tensor& act_payload,
    torch::stable::Tensor& act_sf,
    torch::stable::Tensor const& A, torch::stable::Tensor const& B,
    torch::stable::Tensor const& A_sf, torch::stable::Tensor const& B_sf,
    torch::stable::Tensor const& c1_alpha,
    torch::stable::Tensor const& c2_input_global_scale_inv,
    int64_t logical_act_cols, bool use_bfloat16_intermediate,
    bool launch_with_pdl) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  CHECK_INPUT(act_payload, FLOAT4_E2M1X2, "act_payload");
  CHECK_INPUT(act_sf, SF_DTYPE, "act_sf");
  CHECK_INPUT(A, FLOAT4_E2M1X2, "a");
  CHECK_INPUT(B, FLOAT4_E2M1X2, "b");
  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");
  CHECK_INPUT(c1_alpha, torch::headeronly::ScalarType::Float, "c1_alpha");
  CHECK_INPUT(c2_input_global_scale_inv,
              torch::headeronly::ScalarType::Float,
              "c2_input_global_scale_inv");

  STD_TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  STD_TORCH_CHECK(B.dim() == 2, "b must be a matrix");
  STD_TORCH_CHECK(A.size(1) == B.size(1),
                  "a and b shapes cannot be multiplied (", A.size(0), "x",
                  A.size(1), " and ", B.size(0), "x", B.size(1), ")");
  STD_TORCH_CHECK(B.size(0) % 2 == 0,
                  "interleaved C1 weight rows must be even");
  STD_TORCH_CHECK(logical_act_cols > 0,
                  "logical_act_cols must be positive");
  STD_TORCH_CHECK(logical_act_cols % 16 == 0,
                  "logical_act_cols must be divisible by 16");

  auto const m = static_cast<int>(A.size(0));
  auto const n = static_cast<int>(B.size(0));
  auto const k = static_cast<int>(A.size(1) * 2);

  STD_TORCH_CHECK(logical_act_cols * 2 <= n,
                  "logical activation columns exceed C1 output columns");
  STD_TORCH_CHECK(act_payload.size(0) == m,
                  "act_payload row count must match input rows");
  STD_TORCH_CHECK(act_payload.size(1) >= logical_act_cols / 2,
                  "act_payload is too narrow for compact C1 FP4 columns");

  constexpr int alignment = 32;
  STD_TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ",
                  alignment, ", but got k=", k);
  STD_TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ",
                  alignment, ", but got n=", n);
  STD_TORCH_CHECK((n % 32) == 0,
                  "C1 interleaved N must be divisible by 32 so 32 C1 "
                  "columns map to 16 C2 activation columns per scale.");

  auto round_up = [](int x, int y) { return (x + y - 1) / y * y; };
  int rounded_m = round_up(m, 128);
  int rounded_n = round_up(logical_act_cols / 16, 4);
  STD_TORCH_CHECK(act_sf.dim() == 2, "act_sf must be a matrix");
  STD_TORCH_CHECK(
      act_sf.size(0) == rounded_m && act_sf.size(1) == rounded_n,
      "act_sf must be padded and swizzled to shape (", rounded_m, "x",
      rounded_n, "), got (", act_sf.size(0), "x", act_sf.size(1), ")");

  const torch::stable::accelerator::DeviceGuard device_guard(
      A.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(A.get_device_index());

  uint32_t const mp2 = std::max(static_cast<uint32_t>(16), next_pow_2(m));
  auto const schedule_mode = vllm::proj4::get_c1_schedule_mode();
  auto const fork_mode =
      vllm::proj4::get_c1_paired_output_kernel_fork_mode();
  if (mp2 <= 256) {
    if (schedule_mode == vllm::proj4::C1ScheduleMode::kPingpong) {
      vllm::proj4::runGemmPairedOutputMaybeShallowQueue<
          vllm::proj4::sm120_fp4_c1_config_M256,
          cutlass::gemm::KernelTmaWarpSpecializedPingpongNvf4Sm120>(
          fork_mode,
          act_payload, act_sf, A, B, A_sf, B_sf, c1_alpha,
          c2_input_global_scale_inv, m, n, k,
          static_cast<int>(logical_act_cols), use_bfloat16_intermediate,
          stream, launch_with_pdl);
    } else if (schedule_mode == vllm::proj4::C1ScheduleMode::kCooperative) {
      vllm::proj4::runGemmPairedOutputMaybeShallowQueue<
          vllm::proj4::sm120_fp4_c1_config_M256,
          cutlass::gemm::KernelTmaWarpSpecializedNvf4Sm120>(
          fork_mode,
          act_payload, act_sf, A, B, A_sf, B_sf, c1_alpha,
          c2_input_global_scale_inv, m, n, k,
          static_cast<int>(logical_act_cols), use_bfloat16_intermediate,
          stream, launch_with_pdl);
    } else {
      vllm::proj4::runGemmPairedOutputMaybeShallowQueue<
          vllm::proj4::sm120_fp4_c1_config_M256>(
          fork_mode,
          act_payload, act_sf, A, B, A_sf, B_sf, c1_alpha,
          c2_input_global_scale_inv, m, n, k,
          static_cast<int>(logical_act_cols), use_bfloat16_intermediate,
          stream, launch_with_pdl);
    }
  } else {
    if (schedule_mode == vllm::proj4::C1ScheduleMode::kPingpong) {
      vllm::proj4::runGemmPairedOutputMaybeShallowQueue<
          vllm::proj4::sm120_fp4_c1_config_default,
          cutlass::gemm::KernelTmaWarpSpecializedPingpongNvf4Sm120>(
          fork_mode,
          act_payload, act_sf, A, B, A_sf, B_sf, c1_alpha,
          c2_input_global_scale_inv, m, n, k,
          static_cast<int>(logical_act_cols), use_bfloat16_intermediate,
          stream, launch_with_pdl);
    } else if (schedule_mode == vllm::proj4::C1ScheduleMode::kCooperative) {
      vllm::proj4::runGemmPairedOutputMaybeShallowQueue<
          vllm::proj4::sm120_fp4_c1_config_default,
          cutlass::gemm::KernelTmaWarpSpecializedNvf4Sm120>(
          fork_mode,
          act_payload, act_sf, A, B, A_sf, B_sf, c1_alpha,
          c2_input_global_scale_inv, m, n, k,
          static_cast<int>(logical_act_cols), use_bfloat16_intermediate,
          stream, launch_with_pdl);
    } else {
      vllm::proj4::runGemmPairedOutputMaybeShallowQueue<
          vllm::proj4::sm120_fp4_c1_config_default>(
          fork_mode,
          act_payload, act_sf, A, B, A_sf, B_sf, c1_alpha,
          c2_input_global_scale_inv, m, n, k,
          static_cast<int>(logical_act_cols), use_bfloat16_intermediate,
          stream, launch_with_pdl);
    }
  }
#else
  STD_TORCH_CHECK(false,
                  "Unsupported CUTLASS version. Set VLLM_CUTLASS_SRC_DIR to "
                  "a CUTLASS 3.8+ source directory to enable support.");
#endif
}

void cutlass_scaled_fp4_mm_interleaved_a_sm120a(
    torch::stable::Tensor& D, torch::stable::Tensor const& A,
    torch::stable::Tensor const& B, torch::stable::Tensor const& A_sf,
    torch::stable::Tensor const& B_sf, torch::stable::Tensor const& alpha,
    int64_t logical_k) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  CHECK_INPUT(A, FLOAT4_E2M1X2, "a_interleaved");
  CHECK_INPUT(B, FLOAT4_E2M1X2, "b");
  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");
  CHECK_INPUT(alpha, torch::headeronly::ScalarType::Float, "alpha");

  STD_TORCH_CHECK(A.dim() == 2, "a_interleaved must be a matrix");
  STD_TORCH_CHECK(B.dim() == 2, "b must be a matrix");
  STD_TORCH_CHECK(logical_k > 0 && logical_k % 32 == 0,
                  "logical_k must be positive and divisible by 32");
  STD_TORCH_CHECK(A.size(1) * 2 >= logical_k * 2,
                  "a_interleaved is too narrow for logical_k");
  STD_TORCH_CHECK(B.size(1) * 2 == logical_k,
                  "b K does not match logical_k");

  auto const m = static_cast<int>(A.size(0));
  auto const n = static_cast<int>(B.size(0));
  auto const k = static_cast<int>(logical_k);

  constexpr int alignment = 32;
  STD_TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ",
                  alignment, ", but got k=", k);
  STD_TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ",
                  alignment, ", but got n=", n);

  auto round_up = [](int x, int y) { return (x + y - 1) / y * y; };
  int rounded_m = round_up(m, 128);
  int rounded_n = round_up(n, 128);
  int rounded_k = round_up(k / 16, 4);

  STD_TORCH_CHECK(A_sf.dim() == 2, "scale_a must be a matrix");
  STD_TORCH_CHECK(B_sf.dim() == 2, "scale_b must be a matrix");
  STD_TORCH_CHECK(A_sf.size(0) == rounded_m && A_sf.size(1) == rounded_k,
                  "scale_a must be padded and swizzled to a shape (",
                  rounded_m, "x", rounded_k, "), but got a shape (",
                  A_sf.size(0), "x", A_sf.size(1), ")");
  STD_TORCH_CHECK(B_sf.size(0) == rounded_n && B_sf.size(1) == rounded_k,
                  "scale_b must be padded and swizzled to a shape (",
                  rounded_n, "x", rounded_k, "), but got a shape (",
                  B_sf.size(0), "x", B_sf.size(1), ")");

  auto out_dtype = D.scalar_type();
  const torch::stable::accelerator::DeviceGuard device_guard(
      A.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(A.get_device_index());
  uint32_t const mp2 = std::max(static_cast<uint32_t>(16), next_pow_2(m));

  if (out_dtype == torch::headeronly::ScalarType::BFloat16) {
    if (mp2 <= 256) {
      using Gemm =
          vllm::proj4::Fp4C2InterleavedAGemmSm120<
              vllm::proj4::sm120_fp4_c1_config_M256,
              cutlass::bfloat16_t>::Gemm;
      return vllm::proj4::runGemmInterleavedA<Gemm>(
          D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
    }
    using Gemm =
        vllm::proj4::Fp4C2InterleavedAGemmSm120<
            vllm::proj4::sm120_fp4_c1_config_default,
            cutlass::bfloat16_t>::Gemm;
    return vllm::proj4::runGemmInterleavedA<Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  }

  if (out_dtype == torch::headeronly::ScalarType::Half) {
    if (mp2 <= 256) {
      using Gemm =
          vllm::proj4::Fp4C2InterleavedAGemmSm120<
              vllm::proj4::sm120_fp4_c1_config_M256,
              cutlass::half_t>::Gemm;
      return vllm::proj4::runGemmInterleavedA<Gemm>(
          D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
    }
    using Gemm =
        vllm::proj4::Fp4C2InterleavedAGemmSm120<
            vllm::proj4::sm120_fp4_c1_config_default,
            cutlass::half_t>::Gemm;
    return vllm::proj4::runGemmInterleavedA<Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  }

  STD_TORCH_CHECK(false, "Unsupported output data type of interleaved nvfp4 mm (",
                  out_dtype, ")");
#else
  STD_TORCH_CHECK(false,
                  "Unsupported CUTLASS version. Set VLLM_CUTLASS_SRC_DIR to "
                  "a CUTLASS 3.8+ source directory to enable support.");
#endif
}
