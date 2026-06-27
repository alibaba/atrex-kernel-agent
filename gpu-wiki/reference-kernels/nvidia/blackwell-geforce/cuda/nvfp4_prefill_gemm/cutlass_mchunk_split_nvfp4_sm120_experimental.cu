// gpu-wiki archive note:
// Experimental task39 CUTLASS NVFP4 split-chain launcher for SM120 prefill
// probes. Kept as source-map evidence for rejected/diagnostic prefill routes,
// not as a default production kernel.
//
// Standalone CUTLASS NVFP4 split-chain launcher for Task39 probes.

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace {

struct Sm120Fp4ConfigM128 {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShapeMNK = Shape<_128, _128, _128>;
};

struct Sm120Fp4ConfigM256 {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_256, _128, _128>;
  using PerSmTileShapeMNK = Shape<_256, _128, _128>;
};

template <typename Config, typename OutType,
          typename KernelSchedule =
              cutlass::gemm::collective::KernelScheduleAuto>
struct Fp4GemmSm120 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
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
  using PerSmTileShapeMNK = typename Config::PerSmTileShapeMNK;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShapeMNK, ClusterShape,
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
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Gemm>
typename Gemm::Arguments make_arguments(void* D, const void* A, const void* B,
                                        const void* A_sf, const void* B_sf,
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

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  auto layout_SFA =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
  auto layout_SFB =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<ElementA const*>(A), stride_A,
       reinterpret_cast<ElementB const*>(B), stride_B,
       reinterpret_cast<ElementSFA const*>(A_sf), layout_SFA,
       reinterpret_cast<ElementSFB const*>(B_sf), layout_SFB},
      {{}, reinterpret_cast<ElementD const*>(D), stride_D,
       reinterpret_cast<ElementD*>(D), stride_D}};

  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = ElementCompute(1);
  fusion_args.alpha_ptr = nullptr;
  return arguments;
}

template <typename Gemm>
int run_impl(int M, int N, int K, const void* A, const void* B,
             const void* A_sf, const void* B_sf, void* D,
             cudaStream_t stream) {
  auto arguments = make_arguments<Gemm>(D, A, B, A_sf, B_sf, M, N, K);

  if (Gemm::get_workspace_size(arguments) != 0) {
    return -100;
  }

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return -200 - static_cast<int>(status);
  }
  status = gemm.initialize(arguments, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return -300 - static_cast<int>(status);
  }
  status = gemm.run(arguments, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return -400 - static_cast<int>(status);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return -500 - static_cast<int>(err);
  }
  return 0;
}

using GateGemm =
    typename Fp4GemmSm120<Sm120Fp4ConfigM256, cutlass::bfloat16_t>::Gemm;
using DownGemmDefault =
    typename Fp4GemmSm120<Sm120Fp4ConfigM256, cutlass::bfloat16_t>::Gemm;
using DownGemm = typename Fp4GemmSm120<
    Sm120Fp4ConfigM128, cutlass::bfloat16_t,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::Gemm;

}  // namespace

extern "C" int cutlass_nvfp4_mchunk1024_gate2_down_run(
    int M, const void* gate_A, const void* gate_B, const void* gate_A_sf,
    const void* gate_B_sf, const void* down_A, const void* down_B,
    const void* down_A_sf, const void* down_B_sf, void** gate_outs,
    void** down_outs, unsigned long long stream_handle) {
  constexpr int ChunkM = 1024;
  constexpr int GateN = 34816;
  constexpr int GateHalfN = 17408;
  constexpr int GateK = 5120;
  constexpr int DownN = 5120;
  constexpr int DownK = 17408;
  constexpr int GatePackedK = GateK / 2;
  constexpr int DownPackedK = DownK / 2;
  constexpr int GateScaleK = 320;
  constexpr int DownScaleK = 1088;

  if (M <= 0 || M % ChunkM != 0) {
    return -1;
  }
  int const chunks = M / ChunkM;
  if (chunks > 8) {
    return -2;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  auto const* gate_a = reinterpret_cast<uint8_t const*>(gate_A);
  auto const* gate_b = reinterpret_cast<uint8_t const*>(gate_B);
  auto const* gate_a_sf = reinterpret_cast<uint8_t const*>(gate_A_sf);
  auto const* gate_b_sf = reinterpret_cast<uint8_t const*>(gate_B_sf);
  auto const* down_a = reinterpret_cast<uint8_t const*>(down_A);
  auto const* down_b = reinterpret_cast<uint8_t const*>(down_B);
  auto const* down_a_sf = reinterpret_cast<uint8_t const*>(down_A_sf);
  auto const* down_b_sf = reinterpret_cast<uint8_t const*>(down_B_sf);

  for (int chunk = 0; chunk < chunks; ++chunk) {
    int const row = chunk * ChunkM;
    auto const* gate_a_chunk = gate_a + static_cast<size_t>(row) * GatePackedK;
    auto const* gate_a_sf_chunk =
        gate_a_sf + static_cast<size_t>(row) * GateScaleK;
    auto const* down_a_chunk = down_a + static_cast<size_t>(row) * DownPackedK;
    auto const* down_a_sf_chunk =
        down_a_sf + static_cast<size_t>(row) * DownScaleK;

    int status = run_impl<GateGemm>(
        ChunkM, GateHalfN, GateK, gate_a_chunk, gate_b, gate_a_sf_chunk,
        gate_b_sf, gate_outs[chunk * 2], stream);
    if (status != 0) {
      return status;
    }
    status = run_impl<GateGemm>(
        ChunkM, GateHalfN, GateK, gate_a_chunk,
        gate_b + static_cast<size_t>(GateHalfN) * GatePackedK,
        gate_a_sf_chunk, gate_b_sf + static_cast<size_t>(GateHalfN) * GateScaleK,
        gate_outs[chunk * 2 + 1], stream);
    if (status != 0) {
      return status;
    }
    status = run_impl<DownGemm>(
        ChunkM, DownN, DownK, down_a_chunk, down_b, down_a_sf_chunk, down_b_sf,
        down_outs[chunk], stream);
    if (status != 0) {
      return status;
    }
  }
  return 0;
}

extern "C" int cutlass_nvfp4_full_gate_down_run(
    int M, const void* gate_A, const void* gate_B, const void* gate_A_sf,
    const void* gate_B_sf, const void* down_A, const void* down_B,
    const void* down_A_sf, const void* down_B_sf, void* gate_out,
    void* down_out, unsigned long long stream_handle) {
  constexpr int GateN = 34816;
  constexpr int GateK = 5120;
  constexpr int DownN = 5120;
  constexpr int DownK = 17408;

  if (M != 1024 && M != 2048 && M != 4096 && M != 8192) {
    return -1;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  int status = run_impl<GateGemm>(
      M, GateN, GateK, gate_A, gate_B, gate_A_sf, gate_B_sf, gate_out, stream);
  if (status != 0) {
    return status;
  }
  if (M == 1024) {
    status = run_impl<DownGemm>(
        M, DownN, DownK, down_A, down_B, down_A_sf, down_B_sf, down_out,
        stream);
  } else {
    status = run_impl<DownGemmDefault>(
        M, DownN, DownK, down_A, down_B, down_A_sf, down_B_sf, down_out,
        stream);
  }
  if (status != 0) {
    return status;
  }
  return 0;
}
