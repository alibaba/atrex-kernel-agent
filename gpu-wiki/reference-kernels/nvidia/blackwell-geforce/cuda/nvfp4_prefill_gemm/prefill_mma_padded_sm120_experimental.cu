// gpu-wiki archive note:
// Experimental omoExplore task39 SM120 prefill NVFP4 GEMM candidate for
// production-inventory shapes such as Mx34816x5120 and Mx5120x17408. This
// source is useful for studying direct-layout/TMA/padded-M boundaries, but the
// current wiki conclusion keeps it experimental pending production-shape nsys
// and served TTFT gates.
//
// task_39 prefill NVFP4 GEMM candidate: v11c-style mma.sync kernel with
// internal M padding.
//
// C API:
//   prefill_mma_padded_gemm(M, N, K, A, B_rep, A_sf, B_sf_rep, alpha, C,
//                           A_rep_buf, A_sf_rep_buf, stream)
//   prefill_mma_padded_gemm_direct_a_layout(M, N, K, A, B_rep, A_sf, B_sf_rep,
//                                           alpha, C, A_rep_buf,
//                                           A_sf_rep_buf, stream)
//   prefill_mma_padded_quant_bf16_gemm(M, N, K, X_bf16, B_rep,
//                                      input_global_scale, B_sf_rep, alpha, C,
//                                      A_rep_buf, A_sf_rep_buf, stream)
//   prefill_mma_padded_gemm_direct_layout(M, N, K, A, B, A_sf, B_sf, alpha,
//                                         C, A_rep_buf, A_sf_rep_buf, stream)
//   prefill_mma_padded_gemm_direct_layout_fused_a_repack(M, N, K, A, B, A_sf,
//                                         B_sf, alpha, C, unused, unused,
//                                         stream)
//   prefill_mma_padded_repack_weight(N, K, B, B_rep, B_sf, B_sf_rep, stream)
//   prefill_mma_padded_is_supported(M, N, K)
//
// Layout assumptions:
//   A      [M, K/2] uint8 row-major, produced by scaled_fp4_quant.
//   A_sf   CUTLASS/FlashInfer 128x4 swizzled scale layout, already padded to
//          ceil(M/128) rows by vLLM create_fp4_scale_tensor.
//   B      [N, K/2] uint8 row-major. The direct-layout entry point consumes
//          this vLLM/FlashInfer layout directly; the original entry point
//          consumes a pre-repacked tile-major B_rep.
//   B_sf   CUTLASS/FlashInfer 128x4 swizzled scale layout for weights. The
//          direct-layout entry point consumes this layout directly; the original
//          entry point consumes pre-repacked tile-major B_sf_rep.
//   C      [M, N] bf16 output.
//
// Scope is intentionally narrow:
//   This candidate is only for task_39 vLLM prefill-stage NVFP4 GEMM shapes
//   observed in the production inventory:
//     M x 34816 x 5120    (fused gate/up projection)
//     M x 5120  x 17408   (down projection)
//   with M >= 128. Decode, arbitrary NVFP4 GEMM, and unobserved N/K pairs are
//   not in scope for this candidate.

#ifndef USE_PREFILL_TMA_B
#define USE_PREFILL_TMA_B 0
#endif
#ifndef USE_PREFILL_TMA_BREP
#define USE_PREFILL_TMA_BREP 0
#endif
#ifndef USE_PREFILL_TMA_PAIR_M
#define USE_PREFILL_TMA_PAIR_M 0
#endif
#ifndef USE_PREFILL_WS_TMA_BREP
#define USE_PREFILL_WS_TMA_BREP 0
#endif
#ifndef PREFILL_ENABLE_WS_TMA_BREP_ROUTE
#define PREFILL_ENABLE_WS_TMA_BREP_ROUTE 0
#endif
#ifndef PREFILL_ENABLE_M64X2_WIDE_N_ROUTE
#define PREFILL_ENABLE_M64X2_WIDE_N_ROUTE 0
#endif
#ifndef PREFILL_ENABLE_M64X2_MIXED3_ROUTE
#define PREFILL_ENABLE_M64X2_MIXED3_ROUTE 0
#endif
#ifndef PREFILL_ENABLE_M64_N2_AREUSE_ROUTE
#define PREFILL_ENABLE_M64_N2_AREUSE_ROUTE 0
#endif
#ifndef PREFILL_ENABLE_SCALE_BCAST
#define PREFILL_ENABLE_SCALE_BCAST 0
#endif
#ifndef PREFILL_ENABLE_CLUSTER_ROUTE
#define PREFILL_ENABLE_CLUSTER_ROUTE 0
#endif
#ifndef PREFILL_CLUSTER_X
#define PREFILL_CLUSTER_X 2
#endif
#ifndef PREFILL_CLUSTER_SCHED_POLICY
#define PREFILL_CLUSTER_SCHED_POLICY 2
#endif
#ifndef PREFILL_ENABLE_PERSIST_L2
#define PREFILL_ENABLE_PERSIST_L2 0
#endif
#ifndef PREFILL_ENABLE_PERSISTENT_CTA_ROUTE
#define PREFILL_ENABLE_PERSISTENT_CTA_ROUTE 0
#endif
#ifndef PREFILL_PERSISTENT_CTAS_PER_SM
#define PREFILL_PERSISTENT_CTAS_PER_SM 2
#endif
#ifndef PREFILL_PERSISTENT_CTA_COUNT
#define PREFILL_PERSISTENT_CTA_COUNT 0
#endif
#ifndef PREFILL_PERSIST_HIT_RATIO_X100
#define PREFILL_PERSIST_HIT_RATIO_X100 60
#endif
#ifndef PREFILL_PERSIST_MAX_BYTES
#define PREFILL_PERSIST_MAX_BYTES 0
#endif
#ifndef PREFILL_REPACK_THREADS
#define PREFILL_REPACK_THREADS 256
#endif
#ifndef PREFILL_WS_PRODUCER_REGS
#define PREFILL_WS_PRODUCER_REGS 40
#endif
#ifndef PREFILL_WS_CONSUMER_REGS
#define PREFILL_WS_CONSUMER_REGS 112
#endif
#ifndef PREFILL_ENABLE_CUTLASS_WS_ROUTE
#define PREFILL_ENABLE_CUTLASS_WS_ROUTE 0
#endif
#ifndef PREFILL_CUTLASS_INLINE_ALPHA_ONE
#define PREFILL_CUTLASS_INLINE_ALPHA_ONE 0
#endif
#ifndef PREFILL_CUTLASS_DIRECT_RUN
#define PREFILL_CUTLASS_DIRECT_RUN 1
#endif
#ifndef PREFILL_CUTLASS_DOWN1024_MODE
#define PREFILL_CUTLASS_DOWN1024_MODE 2
#endif
#ifndef PREFILL_CUTLASS_SCHEDULER_KIND
#define PREFILL_CUTLASS_SCHEDULER_KIND 0
#endif
#ifndef PREFILL_CUTLASS_NO_SOURCE_C
#define PREFILL_CUTLASS_NO_SOURCE_C 1
#endif
#ifndef PREFILL_CUTLASS_GATEUP_MODE
#define PREFILL_CUTLASS_GATEUP_MODE 0
#endif
#ifndef PREFILL_CUTLASS_M256_STAGE_COUNT
#define PREFILL_CUTLASS_M256_STAGE_COUNT 0
#endif
#ifndef PREFILL_CUTLASS_M256_SCHEDULE_KIND
#define PREFILL_CUTLASS_M256_SCHEDULE_KIND 0
#endif
#ifndef PREFILL_CUTLASS_EPILOGUE_TILE
#define PREFILL_CUTLASS_EPILOGUE_TILE 2
#endif
#ifndef PREFILL_CUTLASS_M256_EPILOGUE_TILE
#define PREFILL_CUTLASS_M256_EPILOGUE_TILE PREFILL_CUTLASS_EPILOGUE_TILE
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <mutex>
#include <stdint.h>

namespace cg = cooperative_groups;

#if USE_PREFILL_TMA_B || USE_PREFILL_WS_TMA_BREP
#include <cuda.h>
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90_desc.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#endif

#if PREFILL_ENABLE_CUTLASS_WS_ROUTE
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#endif

#if PREFILL_ENABLE_CUTLASS_WS_ROUTE
namespace task39_cutlass_ws {
using namespace cute;

struct Sm120Fp4ConfigM128 {
    using ClusterShape = Shape<_1, _1, _1>;
    using MmaTileShape = Shape<_128, _128, _128>;
    using PerSmTileShapeMNK = Shape<_128, _128, _128>;
};

struct Sm120Fp4ConfigM128K256 {
    using ClusterShape = Shape<_1, _1, _1>;
    using MmaTileShape = Shape<_128, _128, _256>;
    using PerSmTileShapeMNK = Shape<_128, _128, _256>;
};

struct Sm120Fp4ConfigM128N256 {
    using ClusterShape = Shape<_1, _1, _1>;
    using MmaTileShape = Shape<_128, _256, _128>;
    using PerSmTileShapeMNK = Shape<_128, _256, _128>;
};

struct Sm120Fp4ConfigM128N192 {
    using ClusterShape = Shape<_1, _1, _1>;
    using MmaTileShape = Shape<_128, Int<192>, _128>;
    using PerSmTileShapeMNK = Shape<_128, Int<192>, _128>;
};

struct Sm120Fp4ConfigM256N64 {
    using ClusterShape = Shape<_1, _1, _1>;
    using MmaTileShape = Shape<_256, _64, _128>;
    using PerSmTileShapeMNK = Shape<_256, _64, _128>;
};

struct Sm120Fp4ConfigM128N64 {
    using ClusterShape = Shape<_1, _1, _1>;
    using MmaTileShape = Shape<_128, _64, _128>;
    using PerSmTileShapeMNK = Shape<_128, _64, _128>;
};

struct Sm120Fp4ConfigM256 {
    using ClusterShape = Shape<_1, _1, _1>;
    using MmaTileShape = Shape<_256, _128, _128>;
    using PerSmTileShapeMNK = Shape<_256, _128, _128>;
};

struct Sm120Fp4ConfigM256N256 {
    using ClusterShape = Shape<_1, _1, _1>;
    using MmaTileShape = Shape<_256, _256, _128>;
    using PerSmTileShapeMNK = Shape<_256, _256, _128>;
};

struct Sm120Fp4ConfigM256N192 {
    using ClusterShape = Shape<_1, _1, _1>;
    using MmaTileShape = Shape<_256, Int<192>, _128>;
    using PerSmTileShapeMNK = Shape<_256, Int<192>, _128>;
};

template <int Stages, int EpiBytes>
struct StageCountFor {
    using Type = cutlass::gemm::collective::StageCount<Stages>;
};

template <int EpiBytes>
struct StageCountFor<0, EpiBytes> {
    using Type = cutlass::gemm::collective::StageCountAutoCarveout<EpiBytes>;
};

template <int Mode>
struct EpilogueTileFor {
    using Type = cutlass::epilogue::collective::EpilogueTileAuto;
};

template <>
struct EpilogueTileFor<1> {
    using Type = Shape<_64, _64>;
};

template <>
struct EpilogueTileFor<2> {
    using Type = Shape<_128, _32>;
};

template <>
struct EpilogueTileFor<3> {
    using Type = Shape<_128, _64>;
};

template <>
struct EpilogueTileFor<4> {
    using Type = Shape<_256, _32>;
};

template <>
struct EpilogueTileFor<5> {
    using Type = Shape<_64, _128>;
};

template <>
struct EpilogueTileFor<6> {
    using Type = Shape<_128, _128>;
};

template <typename Config, typename OutType,
          typename KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto,
          typename TileSchedulerTag = void,
          int MainloopStages = 0,
          int EpilogueTileMode = PREFILL_CUTLASS_EPILOGUE_TILE>
struct Fp4GemmSm120 {
    using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutATag = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 32;

    using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 32;

    using ElementD = OutType;
    using ElementC = cute::conditional_t<PREFILL_CUTLASS_NO_SOURCE_C, void, OutType>;
    using LayoutCTag = cutlass::layout::RowMajor;
    using LayoutDTag = cutlass::layout::RowMajor;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int AlignmentC =
        PREFILL_CUTLASS_NO_SOURCE_C ? 0 : 128 / cutlass::sizeof_bits<OutType>::value;

    using ElementAccumulator = float;
    using ArchTag = cutlass::arch::Sm120;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

    using MmaTileShape = typename Config::MmaTileShape;
    using ClusterShape = typename Config::ClusterShape;
    using PerSmTileShapeMNK = typename Config::PerSmTileShapeMNK;
    using EpilogueTile =
        typename EpilogueTileFor<EpilogueTileMode>::Type;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            ArchTag, OperatorClass, PerSmTileShapeMNK, ClusterShape,
            EpilogueTile, ElementAccumulator,
            ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD,
            LayoutDTag, AlignmentD,
            cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

    using MainloopStageCount =
        typename StageCountFor<MainloopStages, static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>::Type;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
            LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
            ClusterShape, MainloopStageCount,
            KernelSchedule>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
        TileSchedulerTag>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

static int cached_sm_count(int device_id);

template <typename Gemm>
typename Gemm::Arguments make_arguments(
    int M, int N, int K,
    const void* A, const void* B, const void* A_sf, const void* B_sf,
    const void* alpha, void* D)
{
    using ElementA = typename Gemm::ElementA;
    using ElementB = typename Gemm::ElementB;
    using ElementC = typename Gemm::ElementC;
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
        {{}, reinterpret_cast<ElementC const*>(D), stride_D,
         reinterpret_cast<ElementD*>(D), stride_D}};

    int device_id = 0;
    if (cudaGetDevice(&device_id) == cudaSuccess) {
        arguments.hw_info.device_id = device_id;
        arguments.hw_info.sm_count = cached_sm_count(device_id);
    }

    auto& fusion_args = arguments.epilogue.thread;
#if PREFILL_CUTLASS_INLINE_ALPHA_ONE
    (void)alpha;
    fusion_args.alpha = ElementCompute(1);
    fusion_args.alpha_ptr = nullptr;
#else
    fusion_args.alpha_ptr = reinterpret_cast<ElementCompute const*>(alpha);
#endif
    return arguments;
}

static int cached_sm_count(int device_id)
{
    constexpr int kMaxCachedDevices = 16;
    static std::mutex cache_mutex;
    static int cached[kMaxCachedDevices] = {};

    if (device_id < 0 || device_id >= kMaxCachedDevices) {
        int sm_count = 0;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
        return sm_count;
    }

    int sm_count = cached[device_id];
    if (sm_count > 0) {
        return sm_count;
    }

    std::lock_guard<std::mutex> lock(cache_mutex);
    sm_count = cached[device_id];
    if (sm_count <= 0) {
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
        cached[device_id] = sm_count;
    }
    return sm_count;
}

template <typename Gemm>
cutlass::Status ensure_gemm_initialized(
    typename Gemm::Arguments const& arguments, cudaStream_t stream)
{
    static std::once_flag init_once;
    static cutlass::Status init_status = cutlass::Status::kSuccess;
    std::call_once(init_once, [&]() {
        Gemm gemm;
        init_status = gemm.initialize(arguments, nullptr, stream);
    });
    return init_status;
}

template <typename Gemm>
bool run_impl(int M, int N, int K,
              const void* A, const void* B, const void* A_sf, const void* B_sf,
              const void* alpha, void* D, cudaStream_t stream)
{
    auto arguments = make_arguments<Gemm>(M, N, K, A, B, A_sf, B_sf, alpha, D);
    if (Gemm::get_workspace_size(arguments) != 0) {
        return false;
    }

    Gemm gemm;
    cutlass::Status status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return false;
    }
#if PREFILL_CUTLASS_DIRECT_RUN
    status = ensure_gemm_initialized<Gemm>(arguments, stream);
    if (status != cutlass::Status::kSuccess) {
        return false;
    }
    auto params = Gemm::GemmKernel::to_underlying_arguments(arguments, nullptr);
    status = Gemm::run(params, stream);
    if (status != cutlass::Status::kSuccess) {
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
#else
    status = gemm.initialize(arguments, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        return false;
    }
    status = gemm.run(arguments, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
#endif
}

bool run_dispatch(int M, int N, int K,
                  const void* A, const void* B, const void* A_sf, const void* B_sf,
                  const void* alpha, void* D, cudaStream_t stream)
{
    if (!((N == 34816 && K == 5120) || (N == 5120 && K == 17408))) {
        return false;
    }

#if PREFILL_CUTLASS_SCHEDULER_KIND == 1
    using TileSchedulerTag = cutlass::gemm::StaticPersistentScheduler;
#elif PREFILL_CUTLASS_SCHEDULER_KIND == 2
    using TileSchedulerTag = cutlass::gemm::StreamKScheduler;
#else
    using TileSchedulerTag = void;
#endif

    using GemmM128 =
        typename Fp4GemmSm120<Sm120Fp4ConfigM128, cutlass::bfloat16_t,
                              cutlass::gemm::collective::KernelScheduleAuto,
                              TileSchedulerTag>::Gemm;
#if PREFILL_CUTLASS_M256_SCHEDULE_KIND == 1
    using M256KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
#elif PREFILL_CUTLASS_M256_SCHEDULE_KIND == 2
    using M256KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
#else
    using M256KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
#endif
    using GemmM256 =
        typename Fp4GemmSm120<Sm120Fp4ConfigM256, cutlass::bfloat16_t,
                              M256KernelSchedule,
                              TileSchedulerTag,
                              PREFILL_CUTLASS_M256_STAGE_COUNT,
                              PREFILL_CUTLASS_M256_EPILOGUE_TILE>::Gemm;
#if PREFILL_CUTLASS_GATEUP_MODE == 1
    using GemmM128K256 =
        typename Fp4GemmSm120<Sm120Fp4ConfigM128K256, cutlass::bfloat16_t,
                              cutlass::gemm::collective::KernelScheduleAuto,
                              TileSchedulerTag>::Gemm;
#elif PREFILL_CUTLASS_GATEUP_MODE == 2
    using GemmGateUpSpecial =
        typename Fp4GemmSm120<Sm120Fp4ConfigM128N256, cutlass::bfloat16_t,
                              cutlass::gemm::collective::KernelScheduleAuto,
                              TileSchedulerTag>::Gemm;
#elif PREFILL_CUTLASS_GATEUP_MODE == 3
    using GemmGateUpSpecial =
        typename Fp4GemmSm120<Sm120Fp4ConfigM256N64, cutlass::bfloat16_t,
                              M256KernelSchedule,
                              TileSchedulerTag,
                              PREFILL_CUTLASS_M256_STAGE_COUNT,
                              PREFILL_CUTLASS_M256_EPILOGUE_TILE>::Gemm;
#elif PREFILL_CUTLASS_GATEUP_MODE == 4
    using GemmGateUpSpecial =
        typename Fp4GemmSm120<Sm120Fp4ConfigM128N64, cutlass::bfloat16_t,
                              cutlass::gemm::collective::KernelScheduleAuto,
                              TileSchedulerTag>::Gemm;
#elif PREFILL_CUTLASS_GATEUP_MODE == 5
    using GemmGateUpSpecial =
        typename Fp4GemmSm120<Sm120Fp4ConfigM256N256, cutlass::bfloat16_t,
                              M256KernelSchedule,
                              TileSchedulerTag,
                              PREFILL_CUTLASS_M256_STAGE_COUNT,
                              PREFILL_CUTLASS_M256_EPILOGUE_TILE>::Gemm;
#elif PREFILL_CUTLASS_GATEUP_MODE == 6
    using GemmGateUpSpecial =
        typename Fp4GemmSm120<Sm120Fp4ConfigM256N192, cutlass::bfloat16_t,
                              M256KernelSchedule,
                              TileSchedulerTag,
                              PREFILL_CUTLASS_M256_STAGE_COUNT,
                              PREFILL_CUTLASS_M256_EPILOGUE_TILE>::Gemm;
#elif PREFILL_CUTLASS_GATEUP_MODE == 7
    using GemmGateUpSpecial =
        typename Fp4GemmSm120<Sm120Fp4ConfigM128N192, cutlass::bfloat16_t,
                              cutlass::gemm::collective::KernelScheduleAuto,
                              TileSchedulerTag>::Gemm;
#endif

    if (M == 1024 && N == 5120 && K == 17408) {
#if PREFILL_CUTLASS_DOWN1024_MODE == 1
        using GemmDown1024 = GemmM256;
        return run_impl<GemmDown1024>(
            M, N, K, A, B, A_sf, B_sf, alpha, D, stream);
#elif PREFILL_CUTLASS_DOWN1024_MODE == 2
        using GemmDown1024 = GemmM128;
        return run_impl<GemmDown1024>(
            M, N, K, A, B, A_sf, B_sf, alpha, D, stream);
#elif PREFILL_CUTLASS_DOWN1024_MODE == 3
        using GemmDown1024 =
            typename Fp4GemmSm120<
                Sm120Fp4ConfigM128K256, cutlass::bfloat16_t,
                cutlass::gemm::collective::KernelScheduleAuto,
                TileSchedulerTag>::Gemm;
        return run_impl<GemmDown1024>(
            M, N, K, A, B, A_sf, B_sf, alpha, D, stream);
#elif PREFILL_CUTLASS_DOWN1024_MODE == 4
        using GemmDown1024 =
            typename Fp4GemmSm120<
                Sm120Fp4ConfigM128, cutlass::bfloat16_t,
                cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                TileSchedulerTag>::Gemm;
        return run_impl<GemmDown1024>(
            M, N, K, A, B, A_sf, B_sf, alpha, D, stream);
#elif PREFILL_CUTLASS_DOWN1024_MODE == 5
        using GemmDown1024 =
            typename Fp4GemmSm120<
                Sm120Fp4ConfigM256, cutlass::bfloat16_t,
                cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                TileSchedulerTag>::Gemm;
        return run_impl<GemmDown1024>(
            M, N, K, A, B, A_sf, B_sf, alpha, D, stream);
#elif PREFILL_CUTLASS_DOWN1024_MODE == 6
        using GemmDown1024 =
            typename Fp4GemmSm120<
                Sm120Fp4ConfigM128, cutlass::bfloat16_t,
                cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                TileSchedulerTag>::Gemm;
        return run_impl<GemmDown1024>(
            M, N, K, A, B, A_sf, B_sf, alpha, D, stream);
#elif PREFILL_CUTLASS_DOWN1024_MODE == 7
        using GemmDown1024 =
            typename Fp4GemmSm120<
                Sm120Fp4ConfigM128K256, cutlass::bfloat16_t,
                cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                TileSchedulerTag>::Gemm;
        return run_impl<GemmDown1024>(
            M, N, K, A, B, A_sf, B_sf, alpha, D, stream);
#elif PREFILL_CUTLASS_DOWN1024_MODE == 8
        using GemmDown1024 =
            typename Fp4GemmSm120<
                Sm120Fp4ConfigM256, cutlass::bfloat16_t,
                cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                TileSchedulerTag>::Gemm;
        return run_impl<GemmDown1024>(
            M, N, K, A, B, A_sf, B_sf, alpha, D, stream);
#else
        using GemmDown1024 =
            typename Fp4GemmSm120<
                Sm120Fp4ConfigM128K256, cutlass::bfloat16_t,
                cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                TileSchedulerTag>::Gemm;
        return run_impl<GemmDown1024>(
            M, N, K, A, B, A_sf, B_sf, alpha, D, stream);
#endif
    }
    if (M <= 256) {
        return run_impl<GemmM128>(M, N, K, A, B, A_sf, B_sf, alpha, D, stream);
    }
#if PREFILL_CUTLASS_GATEUP_MODE == 1
    if (N == 34816 && K == 5120) {
        return run_impl<GemmM128K256>(
            M, N, K, A, B, A_sf, B_sf, alpha, D, stream);
    }
#elif PREFILL_CUTLASS_GATEUP_MODE >= 2 && PREFILL_CUTLASS_GATEUP_MODE <= 7
    if (N == 34816 && K == 5120) {
        return run_impl<GemmGateUpSpecial>(
            M, N, K, A, B, A_sf, B_sf, alpha, D, stream);
    }
#endif
    return run_impl<GemmM256>(M, N, K, A, B, A_sf, B_sf, alpha, D, stream);
}
}  // namespace task39_cutlass_ws
#endif

// ================================================================
// PTX helpers
// ================================================================

__device__ __forceinline__ void mma_nvfp4_m16n8k64(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3,
    uint32_t sfa, uint32_t sfb)
{
    uint16_t bidA = 0, tidA = 0, bidB = 0, tidB = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(sfa), "h"(bidA), "h"(tidA),
          "r"(sfb), "h"(bidB), "h"(tidB));
}

struct alignas(32) bf16x16_t {
    __nv_bfloat162 v[8];
};

__device__ __forceinline__ float rcp_approx_ftz(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
    return b;
}

__device__ __forceinline__ uint32_t fp32_vec8_to_e2m1_local(float2 (&array)[4]) {
    uint32_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}\n"
        : "=r"(val)
        : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
          "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
    return val;
}

__device__ __forceinline__ void bf16x16_to_e2m1_scaled(
    const bf16x16_t& in_vec, float global_scale, uint32_t& lo, uint32_t& hi,
    uint8_t& sf_out)
{
    auto local_max = __habs2(in_vec.v[0]);
    #pragma unroll
    for (int i = 1; i < 8; i++) {
        local_max = __hmax2(local_max, __habs2(in_vec.v[i]));
    }

    float vec_max = float(__hmax(local_max.x, local_max.y));
    float sf_value = global_scale * (vec_max * rcp_approx_ftz(6.0f));
    __nv_fp8_e4m3 fp8_sf = __nv_fp8_e4m3(sf_value);
    reinterpret_cast<__nv_fp8_e4m3&>(sf_out) = fp8_sf;
    sf_value = float(fp8_sf);

    float output_scale =
        sf_value != 0.0f
            ? rcp_approx_ftz(sf_value * rcp_approx_ftz(global_scale))
            : 0.0f;

    float2 vals0[4];
    float2 vals1[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        vals0[i] = __bfloat1622float2(in_vec.v[i]);
        vals0[i].x *= output_scale;
        vals0[i].y *= output_scale;
        vals1[i] = __bfloat1622float2(in_vec.v[i + 4]);
        vals1[i].x *= output_scale;
        vals1[i].y *= output_scale;
    }
    lo = fp32_vec8_to_e2m1_local(vals0);
    hi = fp32_vec8_to_e2m1_local(vals1);
}

__device__ __forceinline__ uint32_t smem_to_uint(const void* ptr) {
    uint32_t addr;
    asm("{\n\t.reg .u64 u;\n\tcvta.to.shared.u64 u, %1;\n\tcvt.u32.u64 %0, u;\n\t}"
        : "=r"(addr) : "l"(ptr));
    return addr;
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, uint32_t c) {
    uint32_t a = smem_to_uint(mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" :: "r"(c), "r"(a) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t b) {
    uint32_t a = smem_to_uint(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(a), "r"(b) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar) {
    uint32_t a = smem_to_uint(mbar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                 :: "r"(a) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* mbar, uint32_t p) {
    uint32_t a = smem_to_uint(mbar);
    asm volatile(
        "{\n\t.reg .pred P;\n"
        "WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@!P bra WAIT_%=;\n\t}\n"
        :: "r"(a), "r"(p) : "memory");
}

__device__ __forceinline__ void cp_async_bulk(void* smem, const void* gmem,
                                              uint32_t bytes, uint64_t* mbar) {
    uint32_t s = smem_to_uint(smem);
    uint32_t m = smem_to_uint(mbar);
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :: "r"(s), "l"((uint64_t)gmem), "r"(bytes), "r"(m) : "memory");
}

template <uint32_t RegCount>
__device__ __forceinline__ void warp_reg_dealloc() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(RegCount));
#endif
}

template <uint32_t RegCount>
__device__ __forceinline__ void warp_reg_alloc() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(RegCount));
#endif
}

static __host__ __device__ __forceinline__ int round_up_128(int x) {
    return (x + 127) & ~127;
}

static __host__ __device__ __forceinline__ int round_up_64(int x) {
    return (x + 63) & ~63;
}

static __host__ __device__ __forceinline__ bool is_task39_cutlass_win_shape(
    int M, int N, int K) {
    // Standalone screening build for the large prefill buckets. Production
    // routing must still be gated by measured-positive results.
    return (M == 1024 || M == 2048 || M == 4096 || M == 8192 ||
            (M > 256 && M <= 320)) &&
           ((N == 5120 && K == 17408) || (N == 34816 && K == 5120));
}

__device__ __forceinline__ int swizzle_word_qw(int row, int qw) {
    return qw ^ ((row & 4) ? 4 : 0);
}

static __host__ __device__ __forceinline__ int sfa_phys_scale_lane_64(int scale_lane) {
    return scale_lane < 8 ? scale_lane : 8 + ((scale_lane - 4) & 7);
}

static __host__ __device__ __forceinline__ int sfa_logical_scale_lane_64(int phys_lane) {
    return phys_lane < 8 ? phys_lane : 8 + ((phys_lane - 4) & 7);
}

// ================================================================
// Data repack: [rows, K/2] row-major -> tile-major [Kt, tRows, 128, 32].
// Keeps 4B/thread granularity; the higher thread count is faster for this
// standalone per-call repack than a lower-parallelism 16B copy variant.
// Pads rows >= rows_orig with zero bytes.
// ================================================================

__global__ void repack_data_128_padded(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int rows_orig, int rows_padded, int K)
{
    const int Kh = K / 2, Kt = K / 64;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= Kt * rows_padded * 8) return;

    int idx = tid;
    int kt = idx / (rows_padded * 8);
    idx %= (rows_padded * 8);
    int row = idx / 8;
    int qw = idx % 8;

    uint32_t v = 0;
    if (row < rows_orig) {
        int src_off = row * Kh + kt * 32 + qw * 4;
        v = *(const uint32_t*)(src + src_off);
    }

    int tm = row / 128, lm = row % 128, tRows = rows_padded / 128;
    int dst_off = (kt * tRows + tm) * (128 * 32) + lm * 32 + qw * 4;
    *(uint32_t*)(dst + dst_off) = v;
}

__global__ void repack_data_64_padded(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int rows_orig, int rows_padded, int K)
{
    const int Kh = K / 2, Kt = K / 64;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= Kt * rows_padded * 8) return;

    int idx = tid;
    int kt = idx / (rows_padded * 8);
    idx %= (rows_padded * 8);
    int row = idx / 8;
    int qw = idx % 8;

    uint32_t v = 0;
    if (row < rows_orig) {
        int src_off = row * Kh + kt * 32 + qw * 4;
        v = *(const uint32_t*)(src + src_off);
    }

    int tm = row / 64, lm = row % 64, tRows = rows_padded / 64;
    int dst_off = (kt * tRows + tm) * (64 * 32) + lm * 32 + qw * 4;
    *(uint32_t*)(dst + dst_off) = v;
}

__global__ void repack_data_64_bswizzle_padded(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int rows_orig, int rows_padded, int K)
{
    const int Kh = K / 2, Kt = K / 64;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= Kt * rows_padded * 8) return;

    int idx = tid;
    int kt = idx / (rows_padded * 8);
    idx %= (rows_padded * 8);
    int row = idx / 8;
    int qw = idx % 8;

    uint32_t v = 0;
    if (row < rows_orig) {
        int src_off = row * Kh + kt * 32 + qw * 4;
        v = *(const uint32_t*)(src + src_off);
    }

    int tm = row / 64, lm = row % 64, tRows = rows_padded / 64;
    int phys_qw = swizzle_word_qw(lm, qw);
    int dst_off = (kt * tRows + tm) * (64 * 32) + lm * 32 + phys_qw * 4;
    *(uint32_t*)(dst + dst_off) = v;
}

__global__ void repack_data_128_bswizzle_padded(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int rows_orig, int rows_padded, int K)
{
    const int Kh = K / 2, Kt = K / 64;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= Kt * rows_padded * 8) return;

    int idx = tid;
    int kt = idx / (rows_padded * 8);
    idx %= (rows_padded * 8);
    int row = idx / 8;
    int qw = idx % 8;

    uint32_t v = 0;
    if (row < rows_orig) {
        int src_off = row * Kh + kt * 32 + qw * 4;
        v = *(const uint32_t*)(src + src_off);
    }

    int tm = row / 128, lm = row % 128, tRows = rows_padded / 128;
    int phys_qw = swizzle_word_qw(lm, qw);
    int dst_off = (kt * tRows + tm) * (128 * 32) + lm * 32 + phys_qw * 4;
    *(uint32_t*)(dst + dst_off) = v;
}

__global__ void repack_data_64_frag_padded(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int rows_orig, int rows_padded, int K)
{
    const int Kh = K / 2, Kt = K / 64;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= Kt * rows_padded * 8) return;

    int idx = tid;
    int kt = idx / (rows_padded * 8);
    idx %= (rows_padded * 8);
    int row = idx / 8;
    int qw = idx % 8;

    uint32_t v = 0;
    if (row < rows_orig) {
        int src_off = row * Kh + kt * 32 + qw * 4;
        v = *(const uint32_t*)(src + src_off);
    }

    int tm = row / 64, lm = row % 64, tRows = rows_padded / 64;
    int wr = lm / 32;
    int rem32 = lm & 31;
    int a_group = rem32 >> 4;
    int rem16 = rem32 & 15;
    int row_hi = rem16 >> 3;
    int t1 = rem16 & 7;
    int t0 = qw & 3;
    int q_hi = qw >> 2;
    int lane = t1 * 4 + t0;
    int slot = q_hi * 2 + row_hi;
    int dst_off = (kt * tRows + tm) * (64 * 32) +
                  (wr * 256 + a_group * 128 + lane * 4 + slot) * 4;
    *(uint32_t*)(dst + dst_off) = v;
}

__global__ void repack_data_128_frag_padded(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int rows_orig, int rows_padded, int K)
{
    const int Kh = K / 2, Kt = K / 64;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= Kt * rows_padded * 8) return;

    int idx = tid;
    int kt = idx / (rows_padded * 8);
    idx %= (rows_padded * 8);
    int row = idx / 8;
    int qw = idx % 8;

    uint32_t v = 0;
    if (row < rows_orig) {
        int src_off = row * Kh + kt * 32 + qw * 4;
        v = *(const uint32_t*)(src + src_off);
    }

    int tm = row / 128, lm = row % 128, tRows = rows_padded / 128;
    int wc = lm >> 6;
    int rem64 = lm & 63;
    int ni = rem64 >> 3;
    int t1 = rem64 & 7;
    int t0 = qw & 3;
    int q_hi = qw >> 2;
    int lane = t1 * 4 + t0;
    int dst_off = (kt * tRows + tm) * (128 * 32) +
                  (((wc * 8 + ni) * 32 + lane) * 2 + q_hi) * 4;
    *(uint32_t*)(dst + dst_off) = v;
}

__global__ void repack_data_128_frag_bpair_padded(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int rows_orig, int rows_padded, int K)
{
    const int Kh = K / 2, Kt = K / 64;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= Kt * rows_padded * 8) return;

    int idx = tid;
    int kt = idx / (rows_padded * 8);
    idx %= (rows_padded * 8);
    int row = idx / 8;
    int qw = idx % 8;

    uint32_t v = 0;
    if (row < rows_orig) {
        int src_off = row * Kh + kt * 32 + qw * 4;
        v = *(const uint32_t*)(src + src_off);
    }

    int tm = row / 128, lm = row % 128, tRows = rows_padded / 128;
    int wc = lm >> 6;
    int rem64 = lm & 63;
    int ni = rem64 >> 3;
    int t1 = rem64 & 7;
    int t0 = qw & 3;
    int q_hi = qw >> 2;
    int lane = t1 * 4 + t0;
    int nwarp = wc * 2 + (ni >> 2);
    int ni_pair = (ni & 3) >> 1;
    int ni_in_pair = ni & 1;
    int slot = ni_in_pair * 2 + q_hi;
    int dst_off = (kt * tRows + tm) * (128 * 32) +
                  (nwarp * 256 + ni_pair * 128 + lane * 4 + slot) * 4;
    *(uint32_t*)(dst + dst_off) = v;
}

// ================================================================
// Scale repack: CUTLASS swizzled [ceil(rows/128)*128, Ksf] -> tile-major.
// Pads rows >= rows_orig with zero scale bytes.
// ================================================================

__global__ void repack_sf_unswizzle_128_padded(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int rows_orig, int rows_padded, int K)
{
    const int Ksf = K / 16;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= rows_padded * Ksf) return;

    int row = tid / Ksf;
    int k = tid % Ksf;

    int tm = row / 128;
    int r_hi = (row % 128) / 32;
    int r_lo = (row % 128) % 32;
    int k_group = k / 4;
    int k_lo = k % 4;
    int tRows = rows_padded / 128;

    int lm = row % 128;
    int dst_idx = (k_group * tRows + tm) * 512 + lm * 4 + k_lo;

    if (row >= rows_orig) {
        dst[dst_idx] = 0;
        return;
    }

    int src_idx = tm * 128 * Ksf + k_group * 512 + r_lo * 16 + r_hi * 4 + k_lo;
    dst[dst_idx] = src[src_idx];
}

__global__ void repack_sf_unswizzle_64_padded(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int rows_orig, int rows_padded, int K)
{
    const int Ksf = K / 16;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= rows_padded * Ksf) return;

    int row = tid / Ksf;
    int k = tid % Ksf;

    int dst_tm = row / 64;
    int dst_lm = row % 64;
    int k_group = k / 4;
    int k_lo = k % 4;
    int tRows = rows_padded / 64;

    int dst_idx = (k_group * tRows + dst_tm) * 256 + dst_lm * 4 + k_lo;

    if (row >= rows_orig) {
        dst[dst_idx] = 0;
        return;
    }

    int src_tm = row / 128;
    int r_hi = (row % 128) / 32;
    int r_lo = (row % 128) % 32;
    int src_idx = src_tm * 128 * Ksf + k_group * 512 + r_lo * 16 + r_hi * 4 + k_lo;
    dst[dst_idx] = src[src_idx];
}

__global__ void repack_sf_unswizzle_64_frag_padded(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int rows_orig, int rows_padded, int K)
{
    const int Ksg = K / 64;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= rows_padded * Ksg) return;

    int row = tid / Ksg;
    int kg = tid % Ksg;
    int dst_tm = row / 64;
    int lm = row % 64;
    int tRows = rows_padded / 64;
    int mi = lm >> 4;
    int scale_lane = sfa_phys_scale_lane_64(lm & 15);

    uint32_t v = 0;
    if (row < rows_orig) {
        int src_tm = row / 128;
        int r_hi = (row % 128) / 32;
        int r_lo = (row % 128) % 32;
        int src_idx = src_tm * 128 * (K / 16) + kg * 512 + r_lo * 16 + r_hi * 4;
        v = *(const uint32_t*)(src + src_idx);
    }

    int dst_word = (kg * tRows + dst_tm) * 64 + scale_lane * 4 + mi;
    *(uint32_t*)(dst + dst_word * 4) = v;
}

__global__ void repack_data_sf_64_frag_padded(
    const uint8_t* __restrict__ data_src,
    uint8_t* __restrict__ data_dst,
    const uint8_t* __restrict__ sf_src,
    uint8_t* __restrict__ sf_dst,
    int rows_orig, int rows_padded, int K)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int Kt = K / 64;
    const int data_threads = Kt * rows_padded * 8;

    if (tid < data_threads) {
        const int Kh = K / 2;
        int idx = tid;
        int kt = idx / (rows_padded * 8);
        idx %= (rows_padded * 8);
        int row = idx / 8;
        int qw = idx % 8;

        uint32_t v = 0;
        if (row < rows_orig) {
            int src_off = row * Kh + kt * 32 + qw * 4;
            v = *(const uint32_t*)(data_src + src_off);
        }

        int tm = row / 64, lm = row % 64, tRows = rows_padded / 64;
        int wr = lm / 32;
        int rem32 = lm & 31;
        int a_group = rem32 >> 4;
        int rem16 = rem32 & 15;
        int row_hi = rem16 >> 3;
        int t1 = rem16 & 7;
        int t0 = qw & 3;
        int q_hi = qw >> 2;
        int lane = t1 * 4 + t0;
        int slot = q_hi * 2 + row_hi;
        int dst_off = (kt * tRows + tm) * (64 * 32) +
                      (wr * 256 + a_group * 128 + lane * 4 + slot) * 4;
        *(uint32_t*)(data_dst + dst_off) = v;
    }

    const int Ksg = K / 64;
    const int sf_threads = rows_padded * Ksg;
    if (tid < sf_threads) {
        int row = tid / Ksg;
        int kg = tid % Ksg;
        int dst_tm = row / 64;
        int lm = row % 64;
        int tRows = rows_padded / 64;
        int mi = lm >> 4;
        int scale_lane = sfa_phys_scale_lane_64(lm & 15);

        uint32_t v = 0;
        if (row < rows_orig) {
            int src_tm = row / 128;
            int r_hi = (row % 128) / 32;
            int r_lo = (row % 128) % 32;
            int src_idx = src_tm * 128 * (K / 16) + kg * 512 + r_lo * 16 + r_hi * 4;
            v = *(const uint32_t*)(sf_src + src_idx);
        }

        int dst_word = (kg * tRows + dst_tm) * 64 + scale_lane * 4 + mi;
        *(uint32_t*)(sf_dst + dst_word * 4) = v;
    }
}

__global__ void repack_data_sf_64_frag_full(
    const uint8_t* __restrict__ data_src,
    uint8_t* __restrict__ data_dst,
    const uint8_t* __restrict__ sf_src,
    uint8_t* __restrict__ sf_dst,
    int rows, int K)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int Kt = K / 64;
    const int data_threads = Kt * rows * 8;

    if (tid < data_threads) {
        const int Kh = K / 2;
        int idx = tid;
        int kt = idx / (rows * 8);
        idx %= (rows * 8);
        int row = idx / 8;
        int qw = idx % 8;

        uint32_t v = *(const uint32_t*)(data_src + row * Kh + kt * 32 + qw * 4);

        int tm = row / 64, lm = row % 64, tRows = rows / 64;
        int wr = lm / 32;
        int rem32 = lm & 31;
        int a_group = rem32 >> 4;
        int rem16 = rem32 & 15;
        int row_hi = rem16 >> 3;
        int t1 = rem16 & 7;
        int t0 = qw & 3;
        int q_hi = qw >> 2;
        int lane = t1 * 4 + t0;
        int slot = q_hi * 2 + row_hi;
        int dst_off = (kt * tRows + tm) * (64 * 32) +
                      (wr * 256 + a_group * 128 + lane * 4 + slot) * 4;
        *(uint32_t*)(data_dst + dst_off) = v;
    }

    const int Ksg = K / 64;
    const int sf_threads = rows * Ksg;
    if (tid < sf_threads) {
        int row = tid / Ksg;
        int kg = tid % Ksg;
        int dst_tm = row / 64;
        int lm = row % 64;
        int tRows = rows / 64;
        int mi = lm >> 4;
        int scale_lane = sfa_phys_scale_lane_64(lm & 15);

        int src_tm = row >> 7;
        int r = row & 127;
        int r_hi = r >> 5;
        int r_lo = r & 31;
        int src_idx = src_tm * 128 * (K / 16) + kg * 512 + r_lo * 16 + r_hi * 4;
        uint32_t v = *(const uint32_t*)(sf_src + src_idx);

        int dst_word = (kg * tRows + dst_tm) * 64 + scale_lane * 4 + mi;
        *(uint32_t*)(sf_dst + dst_word * 4) = v;
    }
}

template <int ROWS, int K_CONST>
__global__ void repack_data_sf_64_frag_full_static(
    const uint8_t* __restrict__ data_src,
    uint8_t* __restrict__ data_dst,
    const uint8_t* __restrict__ sf_src,
    uint8_t* __restrict__ sf_dst)
{
    constexpr int Kt = K_CONST / 64;
    constexpr int Kh = K_CONST / 2;
    constexpr int Ksf = K_CONST / 16;
    constexpr int TRows = ROWS / 64;
    constexpr int data_threads = Kt * ROWS * 8;
    constexpr int sf_threads = ROWS * Kt;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < data_threads) {
        int idx = tid;
        int kt = idx / (ROWS * 8);
        idx -= kt * (ROWS * 8);
        int row = idx >> 3;
        int qw = idx & 7;

        uint32_t v = *(const uint32_t*)(data_src + row * Kh + kt * 32 + qw * 4);

        int tm = row >> 6;
        int lm = row & 63;
        int wr = lm >> 5;
        int rem32 = lm & 31;
        int a_group = rem32 >> 4;
        int rem16 = rem32 & 15;
        int row_hi = rem16 >> 3;
        int t1 = rem16 & 7;
        int t0 = qw & 3;
        int q_hi = qw >> 2;
        int lane = t1 * 4 + t0;
        int slot = q_hi * 2 + row_hi;
        int dst_off = (kt * TRows + tm) * (64 * 32) +
                      (wr * 256 + a_group * 128 + lane * 4 + slot) * 4;
        *(uint32_t*)(data_dst + dst_off) = v;
    }

    if (tid < sf_threads) {
        int row = tid / Kt;
        int kg = tid - row * Kt;
        int dst_tm = row >> 6;
        int lm = row & 63;
        int mi = lm >> 4;
        int scale_lane = sfa_phys_scale_lane_64(lm & 15);

        int src_tm = row >> 7;
        int r = row & 127;
        int r_hi = r >> 5;
        int r_lo = r & 31;
        int src_idx = src_tm * 128 * Ksf + kg * 512 + r_lo * 16 + r_hi * 4;
        uint32_t v = *(const uint32_t*)(sf_src + src_idx);

        int dst_word = (kg * TRows + dst_tm) * 64 + scale_lane * 4 + mi;
        *(uint32_t*)(sf_dst + dst_word * 4) = v;
    }
}

__global__ void repack_data_sf_64_frag_padded_k17408_m320_static(
    const uint8_t* __restrict__ data_src,
    uint8_t* __restrict__ data_dst,
    const uint8_t* __restrict__ sf_src,
    uint8_t* __restrict__ sf_dst,
    int rows_orig)
{
    constexpr int Kt = 17408 / 64;
    constexpr int Kh = 17408 / 2;
    constexpr int Ksf = 17408 / 16;
    constexpr int RowsPad = 320;
    constexpr int TRows = RowsPad / 64;

    const int kt = blockIdx.y;
    const int row_qw = blockIdx.x * blockDim.x + threadIdx.x;

    if (kt < Kt && row_qw < RowsPad * 8) {
        const int row = row_qw >> 3;
        const int qw = row_qw & 7;

        uint32_t v = 0;
        if (row < rows_orig) {
            const int src_off = row * Kh + kt * 32 + qw * 4;
            v = *(const uint32_t*)(data_src + src_off);
        }

        const int tm = row >> 6;
        const int lm = row & 63;
        const int wr = lm >> 5;
        const int rem32 = lm & 31;
        const int a_group = rem32 >> 4;
        const int rem16 = rem32 & 15;
        const int row_hi = rem16 >> 3;
        const int t1 = rem16 & 7;
        const int t0 = qw & 3;
        const int q_hi = qw >> 2;
        const int frag_lane = t1 * 4 + t0;
        const int slot = q_hi * 2 + row_hi;
        const int dst_off = (kt * TRows + tm) * (64 * 32) +
                            (wr * 256 + a_group * 128 + frag_lane * 4 + slot) * 4;
        *(uint32_t*)(data_dst + dst_off) = v;
    }

    if (kt < Kt && blockIdx.x < 2) {
        const int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < RowsPad) {
            uint32_t v = 0;
            if (row < rows_orig) {
                const int src_tm = row >> 7;
                const int r = row & 127;
                const int r_hi = r >> 5;
                const int r_lo = r & 31;
                const int src_idx = src_tm * 128 * Ksf + kt * 512 + r_lo * 16 + r_hi * 4;
                v = *(const uint32_t*)(sf_src + src_idx);
            }

            const int dst_tm = row >> 6;
            const int lm = row & 63;
            const int mi = lm >> 4;
            const int scale_lane = sfa_phys_scale_lane_64(lm & 15);
            const int dst_word = (kt * TRows + dst_tm) * 64 + scale_lane * 4 + mi;
            *(uint32_t*)(sf_dst + dst_word * 4) = v;
        }
    }
}

__global__ void repack_sf_unswizzle_128_frag_padded(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int rows_orig, int rows_padded, int K)
{
    const int Ksg = K / 64;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= rows_padded * Ksg) return;

    int row = tid / Ksg;
    int kg = tid % Ksg;
    int tm = row / 128;
    int lm = row % 128;
    int tRows = rows_padded / 128;
    int bgroup = lm >> 3;
    int t1 = lm & 7;
    int nwarp = bgroup >> 2;
    int ni = bgroup & 3;

    uint32_t v = 0;
    if (row < rows_orig) {
        int r_hi = (row % 128) / 32;
        int r_lo = (row % 128) % 32;
        int src_idx = tm * 128 * (K / 16) + kg * 512 + r_lo * 16 + r_hi * 4;
        v = *(const uint32_t*)(src + src_idx);
    }

    int dst_word = (kg * tRows + tm) * 128 + ((nwarp * 8 + t1) * 4 + ni);
    *(uint32_t*)(dst + dst_word * 4) = v;
}

// Fused quantization and fragment-major A/A_sf layout generation.
// This replaces:
//   scaled_fp4_quant(..., is_sf_swizzled_layout=True)
//   repack_data_64_frag_padded
//   repack_sf_unswizzle_64_frag_padded
// for the narrow task_39 down-projection route.
__global__ __launch_bounds__(512, 1)
void quant_bf16_to_frag64_padded(
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ input_global_scale,
    uint8_t* __restrict__ A_frag,
    uint8_t* __restrict__ A_sf_frag,
    int M, int M_pad, int K)
{
    const int packed_cols = K / 16;
    const int tid_col = blockIdx.y * blockDim.x + threadIdx.x;
    if (tid_col >= packed_cols) return;

    const int tRows = M_pad / 64;
    const int kt = tid_col >> 2;
    const int col_in_kt = tid_col & 3;
    const int elem = tid_col * 16;
    const float global_scale = input_global_scale[0];

    for (int row = blockIdx.x; row < M_pad; row += gridDim.x) {
        if (row >= M) continue;

        const int tm = row >> 6;
        const int lm = row & 63;
        const int wr = lm >> 5;
        const int rem32 = lm & 31;
        const int a_group = rem32 >> 4;
        const int rem16 = rem32 & 15;
        const int row_hi = rem16 >> 3;
        const int t1 = rem16 & 7;

        const int base_word = (kt * tRows + tm) * 512 +
                              wr * 256 + a_group * 128 +
                              t1 * 16 + row_hi;
        const int q0 = col_in_kt * 2;
        const int q1 = q0 + 1;
        const int t0_0 = q0 & 3;
        const int q_hi0 = q0 >> 2;
        const int t0_1 = q1 & 3;
        const int q_hi1 = q1 >> 2;

        const int mi = lm >> 4;
        const int scale_lane = sfa_phys_scale_lane_64(lm & 15);
        const int sf_word = (kt * tRows + tm) * 64 + scale_lane * 4 + mi;

        bf16x16_t vec = *(const bf16x16_t*)(X + (size_t)row * K + elem);
        uint32_t lo, hi;
        uint8_t sf;
        bf16x16_to_e2m1_scaled(vec, global_scale, lo, hi, sf);

        ((uint32_t*)A_frag)[base_word + t0_0 * 4 + q_hi0 * 2] = lo;
        ((uint32_t*)A_frag)[base_word + t0_1 * 4 + q_hi1 * 2] = hi;
        A_sf_frag[sf_word * 4 + col_in_kt] = sf;
    }
}

template <int M_PAD_CONST>
__global__ __launch_bounds__(512, 1)
void quant_bf16_to_frag64_k17408_padded_static(
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ input_global_scale,
    uint8_t* __restrict__ A_frag,
    uint8_t* __restrict__ A_sf_frag,
    int M)
{
    constexpr int K_CONST = 17408;
    constexpr int packed_cols = K_CONST / 16;
    constexpr int tRows = M_PAD_CONST / 64;
    const int tid_col = blockIdx.y * blockDim.x + threadIdx.x;
    if (tid_col >= packed_cols) return;

    const int kt = tid_col >> 2;
    const int col_in_kt = tid_col & 3;
    const int elem = tid_col << 4;
    const float global_scale = input_global_scale[0];

    for (int row = blockIdx.x; row < M_PAD_CONST; row += gridDim.x) {
        if (row >= M) continue;

        const int tm = row >> 6;
        const int lm = row & 63;
        const int wr = lm >> 5;
        const int rem32 = lm & 31;
        const int a_group = rem32 >> 4;
        const int rem16 = rem32 & 15;
        const int row_hi = rem16 >> 3;
        const int t1 = rem16 & 7;

        const int base_word = (kt * tRows + tm) * 512 +
                              wr * 256 + a_group * 128 +
                              t1 * 16 + row_hi;
        const int q0 = col_in_kt * 2;
        const int q1 = q0 + 1;
        const int t0_0 = q0 & 3;
        const int q_hi0 = q0 >> 2;
        const int t0_1 = q1 & 3;
        const int q_hi1 = q1 >> 2;

        const int mi = lm >> 4;
        const int scale_lane = sfa_phys_scale_lane_64(lm & 15);
        const int sf_word = (kt * tRows + tm) * 64 + scale_lane * 4 + mi;

        bf16x16_t vec =
            *(const bf16x16_t*)(X + (size_t)row * K_CONST + elem);
        uint32_t lo, hi;
        uint8_t sf;
        bf16x16_to_e2m1_scaled(vec, global_scale, lo, hi, sf);

        ((uint32_t*)A_frag)[base_word + t0_0 * 4 + q_hi0 * 2] = lo;
        ((uint32_t*)A_frag)[base_word + t0_1 * 4 + q_hi1 * 2] = hi;
        A_sf_frag[sf_word * 4 + col_in_kt] = sf;
    }
}

// ================================================================
// Compute: 128x128 tile, 4 warps (2x2), each warp 64x64 output
// ================================================================

__device__ __forceinline__ void compute_64x128(
    const uint8_t* sA, const uint8_t* sB,
    const uint8_t* sSFA, const uint8_t* sSFB,
    float acc[2][8][4],
    int wr, int wc, int t0, int t1, int lid)
{
    int ar0 = wr * 32;
    int ar1 = ar0 + 16;
    int ar0_qw0 = swizzle_word_qw(ar0 + t1, t0);
    int ar0_qw1 = swizzle_word_qw(ar0 + t1 + 8, t0);
    int ar0_qw2 = swizzle_word_qw(ar0 + t1, t0 + 4);
    int ar0_qw3 = swizzle_word_qw(ar0 + t1 + 8, t0 + 4);
    int ar1_qw0 = swizzle_word_qw(ar1 + t1, t0);
    int ar1_qw1 = swizzle_word_qw(ar1 + t1 + 8, t0);
    int ar1_qw2 = swizzle_word_qw(ar1 + t1, t0 + 4);
    int ar1_qw3 = swizzle_word_qw(ar1 + t1 + 8, t0 + 4);

    uint32_t a0[4], a1[4];
    a0[0] = *(const uint32_t*)(sA + (ar0 + t1) * 32 + ar0_qw0 * 4);
    a0[1] = *(const uint32_t*)(sA + (ar0 + t1 + 8) * 32 + ar0_qw1 * 4);
    a0[2] = *(const uint32_t*)(sA + (ar0 + t1) * 32 + ar0_qw2 * 4);
    a0[3] = *(const uint32_t*)(sA + (ar0 + t1 + 8) * 32 + ar0_qw3 * 4);
    a1[0] = *(const uint32_t*)(sA + (ar1 + t1) * 32 + ar1_qw0 * 4);
    a1[1] = *(const uint32_t*)(sA + (ar1 + t1 + 8) * 32 + ar1_qw1 * 4);
    a1[2] = *(const uint32_t*)(sA + (ar1 + t1) * 32 + ar1_qw2 * 4);
    a1[3] = *(const uint32_t*)(sA + (ar1 + t1 + 8) * 32 + ar1_qw3 * 4);

    uint32_t sfa0 = *(const uint32_t*)(sSFA + (ar0 + (lid & 1) * 8 + (lid >> 2)) * 4);
    uint32_t sfa1 = *(const uint32_t*)(sSFA + (ar1 + (lid & 1) * 8 + (lid >> 2)) * 4);

    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        int bc = wc * 64 + ni * 8;
        int brow = bc + t1;
        int b_qw0 = swizzle_word_qw(brow, t0);
        int b_qw1 = swizzle_word_qw(brow, t0 + 4);
        uint32_t b[2];
        b[0] = *(const uint32_t*)(sB + brow * 32 + b_qw0 * 4);
        b[1] = *(const uint32_t*)(sB + brow * 32 + b_qw1 * 4);
        uint32_t sfb = *(const uint32_t*)(sSFB + (bc + (lid >> 2)) * 4);
        mma_nvfp4_m16n8k64(acc[0][ni][0], acc[0][ni][1], acc[0][ni][2], acc[0][ni][3],
                           a0[0], a0[1], a0[2], a0[3], b[0], b[1],
                           acc[0][ni][0], acc[0][ni][1], acc[0][ni][2], acc[0][ni][3],
                           sfa0, sfb);
        mma_nvfp4_m16n8k64(acc[1][ni][0], acc[1][ni][1], acc[1][ni][2], acc[1][ni][3],
                           a1[0], a1[1], a1[2], a1[3], b[0], b[1],
                           acc[1][ni][0], acc[1][ni][1], acc[1][ni][2], acc[1][ni][3],
                           sfa1, sfb);
    }
}

__device__ __forceinline__ void compute_64x128_frag(
    const uint8_t* sA, const uint8_t* sB,
    const uint8_t* sSFA, const uint8_t* sSFB,
    float acc[2][8][4],
    int wr, int wc, int t0, int t1, int lid)
{
    int lane = t1 * 4 + t0;
    int ar0 = wr * 32;
    int ar1 = ar0 + 16;

    uint4 av0 = *(const uint4*)(sA + (wr * 256 + lane * 4) * 4);
    uint4 av1 = *(const uint4*)(sA + (wr * 256 + 128 + lane * 4) * 4);
    uint32_t a0[4] = {av0.x, av0.y, av0.z, av0.w};
    uint32_t a1[4] = {av1.x, av1.y, av1.z, av1.w};

    uint32_t sfa0 = *(const uint32_t*)(sSFA + (ar0 + (lid & 1) * 8 + (lid >> 2)) * 4);
    uint32_t sfa1 = *(const uint32_t*)(sSFA + (ar1 + (lid & 1) * 8 + (lid >> 2)) * 4);

    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        int bc = wc * 64 + ni * 8;
        uint2 bv = *(const uint2*)(sB + (((wc * 8 + ni) * 32 + lane) * 2) * 4);
        uint32_t sfb = *(const uint32_t*)(sSFB + (bc + (lid >> 2)) * 4);
        mma_nvfp4_m16n8k64(acc[0][ni][0], acc[0][ni][1], acc[0][ni][2], acc[0][ni][3],
                           a0[0], a0[1], a0[2], a0[3], bv.x, bv.y,
                           acc[0][ni][0], acc[0][ni][1], acc[0][ni][2], acc[0][ni][3],
                           sfa0, sfb);
        mma_nvfp4_m16n8k64(acc[1][ni][0], acc[1][ni][1], acc[1][ni][2], acc[1][ni][3],
                           a1[0], a1[1], a1[2], a1[3], bv.x, bv.y,
                           acc[1][ni][0], acc[1][ni][1], acc[1][ni][2], acc[1][ni][3],
                           sfa1, sfb);
    }
}

__device__ __forceinline__ uint4 load_sfa_m4n4_skew(
    const uint8_t* sSFA, int t0, int t1)
{
    const int logical_lane = (t0 & 1) * 8 + t1;
    const int phys_lane = sfa_phys_scale_lane_64(logical_lane);
#if PREFILL_ENABLE_SCALE_BCAST
    uint4 v = {0, 0, 0, 0};
    if ((t0 & 2) == 0) {
        v = *(const uint4*)(sSFA + (phys_lane * 4) * 4);
    }
    const int src_lane = t1 * 4 + (t0 & 1);
    v.x = __shfl_sync(0xffffffffu, v.x, src_lane);
    v.y = __shfl_sync(0xffffffffu, v.y, src_lane);
    v.z = __shfl_sync(0xffffffffu, v.z, src_lane);
    v.w = __shfl_sync(0xffffffffu, v.w, src_lane);
    return v;
#else
    return *(const uint4*)(sSFA + (phys_lane * 4) * 4);
#endif
}

__device__ __forceinline__ uint4 load_sfb_m4_broadcast(
    const uint8_t* sSFB, int nwarp, int t0, int t1)
{
#if PREFILL_ENABLE_SCALE_BCAST
    uint4 v = {0, 0, 0, 0};
    if (t0 == 0) {
        v = *(const uint4*)(sSFB + ((nwarp * 8 + t1) * 4) * 4);
    }
    const int src_lane = t1 * 4;
    v.x = __shfl_sync(0xffffffffu, v.x, src_lane);
    v.y = __shfl_sync(0xffffffffu, v.y, src_lane);
    v.z = __shfl_sync(0xffffffffu, v.z, src_lane);
    v.w = __shfl_sync(0xffffffffu, v.w, src_lane);
    return v;
#else
    (void)t0;
    return *(const uint4*)(sSFB + ((nwarp * 8 + t1) * 4) * 4);
#endif
}

__device__ __forceinline__ void compute_64x128_frag_m4n4(
    const uint8_t* sA, const uint8_t* sB,
    const uint8_t* sSFA, const uint8_t* sSFB,
    float acc[4][4][4],
    int nwarp, int t0, int t1, int lid)
{
    int lane = t1 * 4 + t0;
    uint4 sfa = load_sfa_m4n4_skew(sSFA, t0, t1);
    uint4 sfbv = load_sfb_m4_broadcast(sSFB, nwarp, t0, t1);

    uint4 av[4];
    av[0] = *(const uint4*)(sA + (0 * 256 + 0 * 128 + lane * 4) * 4);
    av[1] = *(const uint4*)(sA + (0 * 256 + 1 * 128 + lane * 4) * 4);
    av[2] = *(const uint4*)(sA + (1 * 256 + 0 * 128 + lane * 4) * 4);
    av[3] = *(const uint4*)(sA + (1 * 256 + 1 * 128 + lane * 4) * 4);

    uint2 bv[4];
    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        int bgroup = nwarp * 4 + ni;
        int bwc = bgroup >> 3;
        int bni = bgroup & 7;
        bv[ni] = *(const uint2*)(sB + (((bwc * 8 + bni) * 32 + lane) * 2) * 4);
    }

    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        uint32_t sfb = ni == 0 ? sfbv.x : (ni == 1 ? sfbv.y : (ni == 2 ? sfbv.z : sfbv.w));

        mma_nvfp4_m16n8k64(acc[0][ni][0], acc[0][ni][1], acc[0][ni][2], acc[0][ni][3],
                           av[0].x, av[0].y, av[0].z, av[0].w, bv[ni].x, bv[ni].y,
                           acc[0][ni][0], acc[0][ni][1], acc[0][ni][2], acc[0][ni][3],
                           sfa.x, sfb);
        mma_nvfp4_m16n8k64(acc[1][ni][0], acc[1][ni][1], acc[1][ni][2], acc[1][ni][3],
                           av[1].x, av[1].y, av[1].z, av[1].w, bv[ni].x, bv[ni].y,
                           acc[1][ni][0], acc[1][ni][1], acc[1][ni][2], acc[1][ni][3],
                           sfa.y, sfb);
        mma_nvfp4_m16n8k64(acc[2][ni][0], acc[2][ni][1], acc[2][ni][2], acc[2][ni][3],
                           av[2].x, av[2].y, av[2].z, av[2].w, bv[ni].x, bv[ni].y,
                           acc[2][ni][0], acc[2][ni][1], acc[2][ni][2], acc[2][ni][3],
                           sfa.z, sfb);
        mma_nvfp4_m16n8k64(acc[3][ni][0], acc[3][ni][1], acc[3][ni][2], acc[3][ni][3],
                           av[3].x, av[3].y, av[3].z, av[3].w, bv[ni].x, bv[ni].y,
                           acc[3][ni][0], acc[3][ni][1], acc[3][ni][2], acc[3][ni][3],
                           sfa.w, sfb);
    }
}

__device__ __forceinline__ void compute_64x128_frag_m4n8(
    const uint8_t* sA, const uint8_t* sB,
    const uint8_t* sSFA, const uint8_t* sSFB,
    float acc[4][8][4],
    int ngrp, int t0, int t1, int lid)
{
    int lane = t1 * 4 + t0;
    uint4 sfa = load_sfa_m4n4_skew(sSFA, t0, t1);
    uint4 sfb0 = load_sfb_m4_broadcast(sSFB, ngrp * 2 + 0, t0, t1);
    uint4 sfb1 = load_sfb_m4_broadcast(sSFB, ngrp * 2 + 1, t0, t1);
    uint4 bv0_01 = *(const uint4*)(sB + ((ngrp * 2 + 0) * 256 + 0 * 128 + lane * 4) * 4);
    uint4 bv0_23 = *(const uint4*)(sB + ((ngrp * 2 + 0) * 256 + 1 * 128 + lane * 4) * 4);
    uint4 bv1_01 = *(const uint4*)(sB + ((ngrp * 2 + 1) * 256 + 0 * 128 + lane * 4) * 4);
    uint4 bv1_23 = *(const uint4*)(sB + ((ngrp * 2 + 1) * 256 + 1 * 128 + lane * 4) * 4);

    uint4 av[4];
    av[0] = *(const uint4*)(sA + (0 * 256 + 0 * 128 + lane * 4) * 4);
    av[1] = *(const uint4*)(sA + (0 * 256 + 1 * 128 + lane * 4) * 4);
    av[2] = *(const uint4*)(sA + (1 * 256 + 0 * 128 + lane * 4) * 4);
    av[3] = *(const uint4*)(sA + (1 * 256 + 1 * 128 + lane * 4) * 4);

    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        uint4 sfbv = ni < 4 ? sfb0 : sfb1;
        int si = ni & 3;
        uint32_t sfb = si == 0 ? sfbv.x : (si == 1 ? sfbv.y : (si == 2 ? sfbv.z : sfbv.w));
        uint4 bvp = ni < 2 ? bv0_01 : (ni < 4 ? bv0_23 : (ni < 6 ? bv1_01 : bv1_23));
        bool odd_ni = ni & 1;
        uint32_t b0 = odd_ni ? bvp.z : bvp.x;
        uint32_t b1 = odd_ni ? bvp.w : bvp.y;

        mma_nvfp4_m16n8k64(acc[0][ni][0], acc[0][ni][1], acc[0][ni][2], acc[0][ni][3],
                           av[0].x, av[0].y, av[0].z, av[0].w, b0, b1,
                           acc[0][ni][0], acc[0][ni][1], acc[0][ni][2], acc[0][ni][3],
                           sfa.x, sfb);
        mma_nvfp4_m16n8k64(acc[1][ni][0], acc[1][ni][1], acc[1][ni][2], acc[1][ni][3],
                           av[1].x, av[1].y, av[1].z, av[1].w, b0, b1,
                           acc[1][ni][0], acc[1][ni][1], acc[1][ni][2], acc[1][ni][3],
                           sfa.y, sfb);
        mma_nvfp4_m16n8k64(acc[2][ni][0], acc[2][ni][1], acc[2][ni][2], acc[2][ni][3],
                           av[2].x, av[2].y, av[2].z, av[2].w, b0, b1,
                           acc[2][ni][0], acc[2][ni][1], acc[2][ni][2], acc[2][ni][3],
                           sfa.z, sfb);
        mma_nvfp4_m16n8k64(acc[3][ni][0], acc[3][ni][1], acc[3][ni][2], acc[3][ni][3],
                           av[3].x, av[3].y, av[3].z, av[3].w, b0, b1,
                           acc[3][ni][0], acc[3][ni][1], acc[3][ni][2], acc[3][ni][3],
                           sfa.w, sfb);
    }
}

__device__ __forceinline__ void compute_64x128_frag_m4n4_bpair(
    const uint8_t* sA, const uint8_t* sB,
    const uint8_t* sSFA, const uint8_t* sSFB,
    float acc[4][4][4],
    int nwarp, int t0, int t1, int lid)
{
    int lane = t1 * 4 + t0;
    uint4 sfa = load_sfa_m4n4_skew(sSFA, t0, t1);
    uint4 sfbv = load_sfb_m4_broadcast(sSFB, nwarp, t0, t1);

    uint4 av[4];
    av[0] = *(const uint4*)(sA + (0 * 256 + 0 * 128 + lane * 4) * 4);
    av[1] = *(const uint4*)(sA + (0 * 256 + 1 * 128 + lane * 4) * 4);
    av[2] = *(const uint4*)(sA + (1 * 256 + 0 * 128 + lane * 4) * 4);
    av[3] = *(const uint4*)(sA + (1 * 256 + 1 * 128 + lane * 4) * 4);

    uint4 bv01 = *(const uint4*)(sB + (nwarp * 256 + 0 * 128 + lane * 4) * 4);
    uint4 bv23 = *(const uint4*)(sB + (nwarp * 256 + 1 * 128 + lane * 4) * 4);

    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        uint32_t sfb = ni == 0 ? sfbv.x : (ni == 1 ? sfbv.y : (ni == 2 ? sfbv.z : sfbv.w));
        uint4 bvp = ni < 2 ? bv01 : bv23;
        bool odd_ni = ni & 1;
        uint32_t b0 = odd_ni ? bvp.z : bvp.x;
        uint32_t b1 = odd_ni ? bvp.w : bvp.y;

        mma_nvfp4_m16n8k64(acc[0][ni][0], acc[0][ni][1], acc[0][ni][2], acc[0][ni][3],
                           av[0].x, av[0].y, av[0].z, av[0].w, b0, b1,
                           acc[0][ni][0], acc[0][ni][1], acc[0][ni][2], acc[0][ni][3],
                           sfa.x, sfb);
        mma_nvfp4_m16n8k64(acc[1][ni][0], acc[1][ni][1], acc[1][ni][2], acc[1][ni][3],
                           av[1].x, av[1].y, av[1].z, av[1].w, b0, b1,
                           acc[1][ni][0], acc[1][ni][1], acc[1][ni][2], acc[1][ni][3],
                           sfa.y, sfb);
        mma_nvfp4_m16n8k64(acc[2][ni][0], acc[2][ni][1], acc[2][ni][2], acc[2][ni][3],
                           av[2].x, av[2].y, av[2].z, av[2].w, b0, b1,
                           acc[2][ni][0], acc[2][ni][1], acc[2][ni][2], acc[2][ni][3],
                           sfa.z, sfb);
        mma_nvfp4_m16n8k64(acc[3][ni][0], acc[3][ni][1], acc[3][ni][2], acc[3][ni][3],
                           av[3].x, av[3].y, av[3].z, av[3].w, b0, b1,
                           acc[3][ni][0], acc[3][ni][1], acc[3][ni][2], acc[3][ni][3],
                           sfa.w, sfb);
    }
}

__device__ __forceinline__ uint4 load_bpair_pair(
    const uint8_t* sB, int pair_id, int lane)
{
    const int nwarp_group = pair_id >> 1;
    const int ni_pair = pair_id & 1;
    return *(const uint4*)(
        sB + (nwarp_group * 256 + ni_pair * 128 + lane * 4) * 4);
}

template <int ROLE>
__device__ __forceinline__ void compute_64x128_frag_mixed3_bpair(
    const uint8_t* sA, const uint8_t* sB,
    const uint8_t* sSFA, const uint8_t* sSFB,
    float acc[4][6][4],
    int t0, int t1, int lid)
{
    static_assert(ROLE >= 0 && ROLE < 3);
    constexpr int START_NI = ROLE == 0 ? 0 : (ROLE == 1 ? 6 : 12);
    constexpr int COUNT_NI = ROLE == 2 ? 4 : 6;
    constexpr int PAIR_BASE = START_NI / 2;
    constexpr int PAIR_COUNT = (COUNT_NI + 1) / 2;
    constexpr int SFB_GROUP0 = START_NI / 4;
    constexpr int SFB_GROUP1 = (START_NI + COUNT_NI - 1) / 4;

    int lane = t1 * 4 + t0;
    uint4 sfa = load_sfa_m4n4_skew(sSFA, t0, t1);

    uint4 av[4];
    av[0] = *(const uint4*)(sA + (0 * 256 + 0 * 128 + lane * 4) * 4);
    av[1] = *(const uint4*)(sA + (0 * 256 + 1 * 128 + lane * 4) * 4);
    av[2] = *(const uint4*)(sA + (1 * 256 + 0 * 128 + lane * 4) * 4);
    av[3] = *(const uint4*)(sA + (1 * 256 + 1 * 128 + lane * 4) * 4);

    uint4 bv0 = load_bpair_pair(sB, PAIR_BASE + 0, lane);
    uint4 bv1 = load_bpair_pair(sB, PAIR_BASE + 1, lane);
    uint4 bv2 = {0, 0, 0, 0};
    if constexpr (PAIR_COUNT > 2) {
        bv2 = load_bpair_pair(sB, PAIR_BASE + 2, lane);
    }

    uint4 sfb0 = load_sfb_m4_broadcast(sSFB, SFB_GROUP0, t0, t1);
    uint4 sfb1 = {0, 0, 0, 0};
    if constexpr (SFB_GROUP1 != SFB_GROUP0) {
        sfb1 = load_sfb_m4_broadcast(sSFB, SFB_GROUP1, t0, t1);
    }

    #pragma unroll
    for (int ni = 0; ni < COUNT_NI; ni++) {
        const int global_ni = START_NI + ni;
        const int pair_idx = ni >> 1;
        uint4 bvp = pair_idx == 0 ? bv0 : (pair_idx == 1 ? bv1 : bv2);
        const bool odd_ni = ni & 1;
        uint32_t b0 = odd_ni ? bvp.z : bvp.x;
        uint32_t b1 = odd_ni ? bvp.w : bvp.y;

        uint4 sfbv = (global_ni / 4) == SFB_GROUP0 ? sfb0 : sfb1;
        const int si = global_ni & 3;
        uint32_t sfb = si == 0 ? sfbv.x : (si == 1 ? sfbv.y : (si == 2 ? sfbv.z : sfbv.w));

        mma_nvfp4_m16n8k64(acc[0][ni][0], acc[0][ni][1], acc[0][ni][2], acc[0][ni][3],
                           av[0].x, av[0].y, av[0].z, av[0].w, b0, b1,
                           acc[0][ni][0], acc[0][ni][1], acc[0][ni][2], acc[0][ni][3],
                           sfa.x, sfb);
        mma_nvfp4_m16n8k64(acc[1][ni][0], acc[1][ni][1], acc[1][ni][2], acc[1][ni][3],
                           av[1].x, av[1].y, av[1].z, av[1].w, b0, b1,
                           acc[1][ni][0], acc[1][ni][1], acc[1][ni][2], acc[1][ni][3],
                           sfa.y, sfb);
        mma_nvfp4_m16n8k64(acc[2][ni][0], acc[2][ni][1], acc[2][ni][2], acc[2][ni][3],
                           av[2].x, av[2].y, av[2].z, av[2].w, b0, b1,
                           acc[2][ni][0], acc[2][ni][1], acc[2][ni][2], acc[2][ni][3],
                           sfa.z, sfb);
        mma_nvfp4_m16n8k64(acc[3][ni][0], acc[3][ni][1], acc[3][ni][2], acc[3][ni][3],
                           av[3].x, av[3].y, av[3].z, av[3].w, b0, b1,
                           acc[3][ni][0], acc[3][ni][1], acc[3][ni][2], acc[3][ni][3],
                           sfa.w, sfb);
    }
}

__device__ __forceinline__ void compute_64x128_frag_a_brow_m4n4(
    const uint8_t* sA, const uint8_t* sB_row,
    const uint8_t* sSFA, const uint8_t* sSFB_swiz,
    float acc[4][4][4],
    int nwarp, int t0, int t1, int lid)
{
    int lane = t1 * 4 + t0;
    uint4 av[4];
    av[0] = *(const uint4*)(sA + (0 * 256 + 0 * 128 + lane * 4) * 4);
    av[1] = *(const uint4*)(sA + (0 * 256 + 1 * 128 + lane * 4) * 4);
    av[2] = *(const uint4*)(sA + (1 * 256 + 0 * 128 + lane * 4) * 4);
    av[3] = *(const uint4*)(sA + (1 * 256 + 1 * 128 + lane * 4) * 4);

    uint4 sfa = load_sfa_m4n4_skew(sSFA, t0, t1);

    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        const int row = (nwarp * 4 + ni) * 8 + t1;
        uint32_t b0 = *(const uint32_t*)(sB_row + row * 32 + t0 * 4);
        uint32_t b1 = *(const uint32_t*)(sB_row + row * 32 + (t0 + 4) * 4);
        uint32_t sfb = *(const uint32_t*)(
            sSFB_swiz + (row & 31) * 16 + (row >> 5) * 4);

        mma_nvfp4_m16n8k64(acc[0][ni][0], acc[0][ni][1], acc[0][ni][2], acc[0][ni][3],
                           av[0].x, av[0].y, av[0].z, av[0].w, b0, b1,
                           acc[0][ni][0], acc[0][ni][1], acc[0][ni][2], acc[0][ni][3],
                           sfa.x, sfb);
        mma_nvfp4_m16n8k64(acc[1][ni][0], acc[1][ni][1], acc[1][ni][2], acc[1][ni][3],
                           av[1].x, av[1].y, av[1].z, av[1].w, b0, b1,
                           acc[1][ni][0], acc[1][ni][1], acc[1][ni][2], acc[1][ni][3],
                           sfa.y, sfb);
        mma_nvfp4_m16n8k64(acc[2][ni][0], acc[2][ni][1], acc[2][ni][2], acc[2][ni][3],
                           av[2].x, av[2].y, av[2].z, av[2].w, b0, b1,
                           acc[2][ni][0], acc[2][ni][1], acc[2][ni][2], acc[2][ni][3],
                           sfa.z, sfb);
        mma_nvfp4_m16n8k64(acc[3][ni][0], acc[3][ni][1], acc[3][ni][2], acc[3][ni][3],
                           av[3].x, av[3].y, av[3].z, av[3].w, b0, b1,
                           acc[3][ni][0], acc[3][ni][1], acc[3][ni][2], acc[3][ni][3],
                           sfa.w, sfb);
    }
}

__device__ __forceinline__ void compute_128x128(
    const uint8_t* sA, const uint8_t* sB,
    const uint8_t* sSFA, const uint8_t* sSFB,
    float acc[4][8][4],
    int wr, int wc, int t0, int t1, int lid)
{
    #pragma unroll
    for (int mp = 0; mp < 2; mp++) {
        int mi0 = mp * 2;
        int mi1 = mi0 + 1;
        int ar0 = wr * 64 + mi0 * 16;
        int ar1 = wr * 64 + mi1 * 16;
        int ar0_qw0 = swizzle_word_qw(ar0 + t1, t0);
        int ar0_qw1 = swizzle_word_qw(ar0 + t1 + 8, t0);
        int ar0_qw2 = swizzle_word_qw(ar0 + t1, t0 + 4);
        int ar0_qw3 = swizzle_word_qw(ar0 + t1 + 8, t0 + 4);
        int ar1_qw0 = swizzle_word_qw(ar1 + t1, t0);
        int ar1_qw1 = swizzle_word_qw(ar1 + t1 + 8, t0);
        int ar1_qw2 = swizzle_word_qw(ar1 + t1, t0 + 4);
        int ar1_qw3 = swizzle_word_qw(ar1 + t1 + 8, t0 + 4);

        uint32_t a0[4], a1[4];
        a0[0] = *(const uint32_t*)(sA + (ar0 + t1) * 32 + ar0_qw0 * 4);
        a0[1] = *(const uint32_t*)(sA + (ar0 + t1 + 8) * 32 + ar0_qw1 * 4);
        a0[2] = *(const uint32_t*)(sA + (ar0 + t1) * 32 + ar0_qw2 * 4);
        a0[3] = *(const uint32_t*)(sA + (ar0 + t1 + 8) * 32 + ar0_qw3 * 4);
        a1[0] = *(const uint32_t*)(sA + (ar1 + t1) * 32 + ar1_qw0 * 4);
        a1[1] = *(const uint32_t*)(sA + (ar1 + t1 + 8) * 32 + ar1_qw1 * 4);
        a1[2] = *(const uint32_t*)(sA + (ar1 + t1) * 32 + ar1_qw2 * 4);
        a1[3] = *(const uint32_t*)(sA + (ar1 + t1 + 8) * 32 + ar1_qw3 * 4);

        uint32_t sfa0 = *(const uint32_t*)(sSFA + (ar0 + (lid & 1) * 8 + (lid >> 2)) * 4);
        uint32_t sfa1 = *(const uint32_t*)(sSFA + (ar1 + (lid & 1) * 8 + (lid >> 2)) * 4);
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            int bc = wc * 64 + ni * 8;
            int brow = bc + t1;
            int b_qw0 = swizzle_word_qw(brow, t0);
            int b_qw1 = swizzle_word_qw(brow, t0 + 4);
            uint32_t b[2];
            b[0] = *(const uint32_t*)(sB + brow * 32 + b_qw0 * 4);
            b[1] = *(const uint32_t*)(sB + brow * 32 + b_qw1 * 4);
            uint32_t sfb = *(const uint32_t*)(sSFB + (bc + (lid >> 2)) * 4);
            mma_nvfp4_m16n8k64(acc[mi0][ni][0], acc[mi0][ni][1], acc[mi0][ni][2], acc[mi0][ni][3],
                               a0[0], a0[1], a0[2], a0[3], b[0], b[1],
                               acc[mi0][ni][0], acc[mi0][ni][1], acc[mi0][ni][2], acc[mi0][ni][3],
                               sfa0, sfb);
            mma_nvfp4_m16n8k64(acc[mi1][ni][0], acc[mi1][ni][1], acc[mi1][ni][2], acc[mi1][ni][3],
                               a1[0], a1[1], a1[2], a1[3], b[0], b[1],
                               acc[mi1][ni][0], acc[mi1][ni][1], acc[mi1][ni][2], acc[mi1][ni][3],
                               sfa1, sfb);
        }
    }
}

__device__ __forceinline__ void epilogue_64x128_bf16_guarded(
    __nv_bfloat16* C, float acc[2][8][4], float alpha,
    int cm, int cn, int wr, int wc, int t1, int lid, int M_orig, int N)
{
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        int mm = cm + wr * 32 + mi * 16;
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            int nn = cn + wc * 64 + ni * 8;
            int m0 = mm + t1, m1 = m0 + 8, n0 = nn + (lid & 3) * 2;
            if (m0 < M_orig && n0 + 1 < N) {
                __nv_bfloat162 v = __floats2bfloat162_rn(
                    acc[mi][ni][0] * alpha, acc[mi][ni][1] * alpha);
                *(__nv_bfloat162*)(C + (size_t)m0 * N + n0) = v;
            }
            if (m1 < M_orig && n0 + 1 < N) {
                __nv_bfloat162 v = __floats2bfloat162_rn(
                    acc[mi][ni][2] * alpha, acc[mi][ni][3] * alpha);
                *(__nv_bfloat162*)(C + (size_t)m1 * N + n0) = v;
            }
        }
    }
}

__device__ __forceinline__ void epilogue_64x128_m4n4_bf16_guarded(
    __nv_bfloat16* C, float acc[4][4][4], float alpha,
    int cm, int cn, int nwarp, int t1, int lid, int M_orig, int N)
{
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        int mm = cm + mi * 16;
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int nn = cn + nwarp * 32 + ni * 8;
            int m0 = mm + t1, m1 = m0 + 8, n0 = nn + (lid & 3) * 2;
            if (m0 < M_orig && n0 + 1 < N) {
                __nv_bfloat162 v = __floats2bfloat162_rn(
                    acc[mi][ni][0] * alpha, acc[mi][ni][1] * alpha);
                *(__nv_bfloat162*)(C + (size_t)m0 * N + n0) = v;
            }
            if (m1 < M_orig && n0 + 1 < N) {
                __nv_bfloat162 v = __floats2bfloat162_rn(
                    acc[mi][ni][2] * alpha, acc[mi][ni][3] * alpha);
                *(__nv_bfloat162*)(C + (size_t)m1 * N + n0) = v;
            }
        }
    }
}

template <bool AlphaOne>
__device__ __forceinline__ void epilogue_64x128_m4n4_bf16_full(
    __nv_bfloat16* C, float acc[4][4][4], float alpha,
    int cm, int cn, int nwarp, int t1, int lid, int N)
{
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        int mm = cm + mi * 16;
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int nn = cn + nwarp * 32 + ni * 8;
            int m0 = mm + t1, m1 = m0 + 8, n0 = nn + (lid & 3) * 2;
            float v00 = acc[mi][ni][0];
            float v01 = acc[mi][ni][1];
            float v10 = acc[mi][ni][2];
            float v11 = acc[mi][ni][3];
            if constexpr (!AlphaOne) {
                v00 *= alpha;
                v01 *= alpha;
                v10 *= alpha;
                v11 *= alpha;
            }
            *(__nv_bfloat162*)(C + (size_t)m0 * N + n0) =
                __floats2bfloat162_rn(v00, v01);
            *(__nv_bfloat162*)(C + (size_t)m1 * N + n0) =
                __floats2bfloat162_rn(v10, v11);
        }
    }
}

template <bool AlphaOne>
__device__ __forceinline__ void epilogue_64x128_m4n8_bf16_full(
    __nv_bfloat16* C, float acc[4][8][4], float alpha,
    int cm, int cn, int ngrp, int t1, int lid, int N)
{
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        int mm = cm + mi * 16;
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            int nn = cn + ngrp * 64 + ni * 8;
            int m0 = mm + t1, m1 = m0 + 8, n0 = nn + (lid & 3) * 2;
            float v00 = acc[mi][ni][0];
            float v01 = acc[mi][ni][1];
            float v10 = acc[mi][ni][2];
            float v11 = acc[mi][ni][3];
            if constexpr (!AlphaOne) {
                v00 *= alpha;
                v01 *= alpha;
                v10 *= alpha;
                v11 *= alpha;
            }
            *(__nv_bfloat162*)(C + (size_t)m0 * N + n0) =
                __floats2bfloat162_rn(v00, v01);
            *(__nv_bfloat162*)(C + (size_t)m1 * N + n0) =
                __floats2bfloat162_rn(v10, v11);
        }
    }
}

template <int ROLE>
__device__ __forceinline__ void epilogue_64x128_mixed3_bf16_guarded(
    __nv_bfloat16* C, float acc[4][6][4], float alpha,
    int cm, int cn, int t1, int lid, int M_orig, int N)
{
    static_assert(ROLE >= 0 && ROLE < 3);
    constexpr int START_NI = ROLE == 0 ? 0 : (ROLE == 1 ? 6 : 12);
    constexpr int COUNT_NI = ROLE == 2 ? 4 : 6;
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        int mm = cm + mi * 16;
        #pragma unroll
        for (int ni = 0; ni < COUNT_NI; ni++) {
            int nn = cn + (START_NI + ni) * 8;
            int m0 = mm + t1, m1 = m0 + 8, n0 = nn + (lid & 3) * 2;
            if (m0 < M_orig && n0 + 1 < N) {
                __nv_bfloat162 v = __floats2bfloat162_rn(
                    acc[mi][ni][0] * alpha, acc[mi][ni][1] * alpha);
                *(__nv_bfloat162*)(C + (size_t)m0 * N + n0) = v;
            }
            if (m1 < M_orig && n0 + 1 < N) {
                __nv_bfloat162 v = __floats2bfloat162_rn(
                    acc[mi][ni][2] * alpha, acc[mi][ni][3] * alpha);
                *(__nv_bfloat162*)(C + (size_t)m1 * N + n0) = v;
            }
        }
    }
}

template <int ROLE, bool AlphaOne>
__device__ __forceinline__ void epilogue_64x128_mixed3_bf16_full(
    __nv_bfloat16* C, float acc[4][6][4], float alpha,
    int cm, int cn, int t1, int lid, int N)
{
    static_assert(ROLE >= 0 && ROLE < 3);
    constexpr int START_NI = ROLE == 0 ? 0 : (ROLE == 1 ? 6 : 12);
    constexpr int COUNT_NI = ROLE == 2 ? 4 : 6;
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        int mm = cm + mi * 16;
        #pragma unroll
        for (int ni = 0; ni < COUNT_NI; ni++) {
            int nn = cn + (START_NI + ni) * 8;
            int m0 = mm + t1, m1 = m0 + 8, n0 = nn + (lid & 3) * 2;
            float v00 = acc[mi][ni][0];
            float v01 = acc[mi][ni][1];
            float v10 = acc[mi][ni][2];
            float v11 = acc[mi][ni][3];
            if constexpr (!AlphaOne) {
                v00 *= alpha;
                v01 *= alpha;
                v10 *= alpha;
                v11 *= alpha;
            }
            *(__nv_bfloat162*)(C + (size_t)m0 * N + n0) =
                __floats2bfloat162_rn(v00, v01);
            *(__nv_bfloat162*)(C + (size_t)m1 * N + n0) =
                __floats2bfloat162_rn(v10, v11);
        }
    }
}

__device__ __forceinline__ void epilogue_128x128_bf16_guarded(
    __nv_bfloat16* C, float acc[4][8][4], float alpha,
    int cm, int cn, int wr, int wc, int t1, int lid, int M_orig, int N)
{
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        int mm = cm + wr * 64 + mi * 16;
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            int nn = cn + wc * 64 + ni * 8;
            int m0 = mm + t1, m1 = m0 + 8, n0 = nn + (lid & 3) * 2;
            if (m0 < M_orig && n0 + 1 < N) {
                __nv_bfloat162 v = __floats2bfloat162_rn(
                    acc[mi][ni][0] * alpha, acc[mi][ni][1] * alpha);
                *(__nv_bfloat162*)(C + (size_t)m0 * N + n0) = v;
            }
            if (m1 < M_orig && n0 + 1 < N) {
                __nv_bfloat162 v = __floats2bfloat162_rn(
                    acc[mi][ni][2] * alpha, acc[mi][ni][3] * alpha);
                *(__nv_bfloat162*)(C + (size_t)m1 * N + n0) = v;
            }
        }
    }
}

// ================================================================
// GEMM kernel: 128x128 tile, 2-stage cp.async.bulk.
// ================================================================

template <int STAGES>
__global__ __launch_bounds__(128, 2)
void kern_prefill_mma_padded_bf16_m64(
    const uint8_t* __restrict__ At, const uint8_t* __restrict__ Bt,
    const uint8_t* __restrict__ Ast, const uint8_t* __restrict__ Bst,
    __nv_bfloat16* __restrict__ C, const float* __restrict__ alpha_ptr,
    int M_orig, int M_tiles, int N, int K)
{
    const int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31;
    const int nwarp = wid, t0 = lid & 3, t1 = lid >> 2;
    const int cm = blockIdx.x * 64, cn = blockIdx.y * 128;
    const int Kt = K / 64, tM = M_tiles, tN = N / 128;
    const float alpha = *alpha_ptr;

    __shared__ __align__(128) uint8_t sA[STAGES][2048];
    __shared__ __align__(128) uint8_t sB[STAGES][4096];
    __shared__ __align__(128) uint8_t sSFA[STAGES][256];
    __shared__ __align__(128) uint8_t sSFB[STAGES][512];
    __shared__ __align__(64)  uint64_t mbar[STAGES];

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    if (tid == 0) for (int s = 0; s < STAGES; s++) mbarrier_init(&mbar[s], 1);
    __syncthreads();

    auto load = [&](int kt, int s) {
        if (tid == 0) {
            mbarrier_arrive_expect_tx(&mbar[s], 2048 + 4096 + 256 + 512);
            size_t am = (size_t)kt * tM + blockIdx.x;
            size_t bn = (size_t)kt * tN + blockIdx.y;
            cp_async_bulk(sA[s], At + am * 2048, 2048, &mbar[s]);
            cp_async_bulk(sB[s], Bt + bn * 4096, 4096, &mbar[s]);
            cp_async_bulk(sSFA[s], Ast + am * 256, 256, &mbar[s]);
            cp_async_bulk(sSFB[s], Bst + bn * 512, 512, &mbar[s]);
        }
    };

    for (int s = 0; s < STAGES - 1 && s < Kt; s++) load(s, s);
    for (int kt = 0; kt < Kt; kt++) {
        int s = kt % STAGES, ph = (kt / STAGES) & 1;
        mbarrier_wait_parity(&mbar[s], ph);
        int kn = kt + STAGES - 1;
        if (kn < Kt) load(kn, kn % STAGES);
        compute_64x128_frag_m4n4(sA[s], sB[s], sSFA[s], sSFB[s], acc, nwarp, t0, t1, lid);
        __syncthreads();
    }

    if (M_orig == M_tiles * 64) {
        if (alpha == 1.0f) {
            epilogue_64x128_m4n4_bf16_full<true>(
                C, acc, alpha, cm, cn, nwarp, t1, lid, N);
        } else {
            epilogue_64x128_m4n4_bf16_full<false>(
                C, acc, alpha, cm, cn, nwarp, t1, lid, N);
        }
    } else {
        epilogue_64x128_m4n4_bf16_guarded(
            C, acc, alpha, cm, cn, nwarp, t1, lid, M_orig, N);
    }
}

template <int STAGES>
__global__ __launch_bounds__(256, 2)
void kern_prefill_mma_padded_bf16_m64x2(
    const uint8_t* __restrict__ At, const uint8_t* __restrict__ Bt,
    const uint8_t* __restrict__ Ast, const uint8_t* __restrict__ Bst,
    __nv_bfloat16* __restrict__ C, const float* __restrict__ alpha_ptr,
    int M_orig, int M_tiles, int N, int K)
{
    const int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31;
    const int pair = wid >> 2;
    const int nwarp = wid & 3;
    const int t0 = lid & 3, t1 = lid >> 2;
    const int m_tile = blockIdx.x * 2 + pair;
    const bool valid_m_tile = m_tile < M_tiles;
    const int cm = m_tile * 64, cn = blockIdx.y * 128;
    const int Kt = K / 64, tM = M_tiles, tN = N / 128;
    const float alpha = *alpha_ptr;

    __shared__ __align__(128) uint8_t sA[2][STAGES][2048];
    __shared__ __align__(128) uint8_t sB[STAGES][4096];
    __shared__ __align__(128) uint8_t sSFA[2][STAGES][256];
    __shared__ __align__(128) uint8_t sSFB[STAGES][512];
    __shared__ __align__(64)  uint64_t mbar[STAGES];

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; s++) mbarrier_init(&mbar[s], 1);
    }
    __syncthreads();

    auto load = [&](int kt, int s) {
        if (tid == 0) {
            const int m0 = blockIdx.x * 2;
            const int m1 = m0 + 1;
            const bool valid0 = m0 < M_tiles;
            const bool valid1 = m1 < M_tiles;
            uint32_t bytes = 4096 + 512;
            if (valid0) bytes += 2048 + 256;
            if (valid1) bytes += 2048 + 256;
            mbarrier_arrive_expect_tx(&mbar[s], bytes);

            if (valid0) {
                size_t am0 = (size_t)kt * tM + m0;
                cp_async_bulk(sA[0][s], At + am0 * 2048, 2048, &mbar[s]);
                cp_async_bulk(sSFA[0][s], Ast + am0 * 256, 256, &mbar[s]);
            }
            if (valid1) {
                size_t am1 = (size_t)kt * tM + m1;
                cp_async_bulk(sA[1][s], At + am1 * 2048, 2048, &mbar[s]);
                cp_async_bulk(sSFA[1][s], Ast + am1 * 256, 256, &mbar[s]);
            }

            size_t bn = (size_t)kt * tN + blockIdx.y;
            cp_async_bulk(sB[s], Bt + bn * 4096, 4096, &mbar[s]);
            cp_async_bulk(sSFB[s], Bst + bn * 512, 512, &mbar[s]);
        }
    };

    for (int s = 0; s < STAGES - 1 && s < Kt; s++) load(s, s);
    #pragma unroll 1
    for (int kt = 0; kt < Kt; kt++) {
        int s = kt % STAGES, ph = (kt / STAGES) & 1;
        mbarrier_wait_parity(&mbar[s], ph);
        int kn = kt + STAGES - 1;
        if (kn < Kt) load(kn, kn % STAGES);
        if (valid_m_tile) {
            compute_64x128_frag_m4n4(
                sA[pair][s], sB[s], sSFA[pair][s], sSFB[s],
                acc, nwarp, t0, t1, lid);
        }
        __syncthreads();
    }

    if (valid_m_tile) {
        if (M_orig == M_tiles * 64) {
            if (alpha == 1.0f) {
                epilogue_64x128_m4n4_bf16_full<true>(
                    C, acc, alpha, cm, cn, nwarp, t1, lid, N);
            } else {
                epilogue_64x128_m4n4_bf16_full<false>(
                    C, acc, alpha, cm, cn, nwarp, t1, lid, N);
            }
        } else {
            epilogue_64x128_m4n4_bf16_guarded(
                C, acc, alpha, cm, cn, nwarp, t1, lid, M_orig, N);
        }
    }
}

template <int STAGES>
__global__ __launch_bounds__(128, 2)
void kern_prefill_mma_padded_bf16_m64x2_wide_n(
    const uint8_t* __restrict__ At, const uint8_t* __restrict__ Bt,
    const uint8_t* __restrict__ Ast, const uint8_t* __restrict__ Bst,
    __nv_bfloat16* __restrict__ C, const float* __restrict__ alpha_ptr,
    int M_orig, int M_tiles, int N, int K)
{
    const int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31;
    const int pair = wid >> 1;
    const int ngrp = wid & 1;
    const int t0 = lid & 3, t1 = lid >> 2;
    const int m_tile = blockIdx.x * 2 + pair;
    const bool valid_m_tile = m_tile < M_tiles;
    const int cm = m_tile * 64, cn = blockIdx.y * 128;
    const int Kt = K / 64, tM = M_tiles, tN = N / 128;
    const float alpha = *alpha_ptr;

    __shared__ __align__(128) uint8_t sA[2][STAGES][2048];
    __shared__ __align__(128) uint8_t sB[STAGES][4096];
    __shared__ __align__(128) uint8_t sSFA[2][STAGES][256];
    __shared__ __align__(128) uint8_t sSFB[STAGES][512];
    __shared__ __align__(64)  uint64_t mbar[STAGES];

    float acc[4][8][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; s++) mbarrier_init(&mbar[s], 1);
    }
    __syncthreads();

    auto load = [&](int kt, int s) {
        if (tid == 0) {
            const int m0 = blockIdx.x * 2;
            const int m1 = m0 + 1;
            const bool valid0 = m0 < M_tiles;
            const bool valid1 = m1 < M_tiles;
            uint32_t bytes = 4096 + 512;
            if (valid0) bytes += 2048 + 256;
            if (valid1) bytes += 2048 + 256;
            mbarrier_arrive_expect_tx(&mbar[s], bytes);

            if (valid0) {
                size_t am0 = (size_t)kt * tM + m0;
                cp_async_bulk(sA[0][s], At + am0 * 2048, 2048, &mbar[s]);
                cp_async_bulk(sSFA[0][s], Ast + am0 * 256, 256, &mbar[s]);
            }
            if (valid1) {
                size_t am1 = (size_t)kt * tM + m1;
                cp_async_bulk(sA[1][s], At + am1 * 2048, 2048, &mbar[s]);
                cp_async_bulk(sSFA[1][s], Ast + am1 * 256, 256, &mbar[s]);
            }

            size_t bn = (size_t)kt * tN + blockIdx.y;
            cp_async_bulk(sB[s], Bt + bn * 4096, 4096, &mbar[s]);
            cp_async_bulk(sSFB[s], Bst + bn * 512, 512, &mbar[s]);
        }
    };

    for (int s = 0; s < STAGES - 1 && s < Kt; s++) load(s, s);
    #pragma unroll 1
    for (int kt = 0; kt < Kt; kt++) {
        int s = kt % STAGES, ph = (kt / STAGES) & 1;
        mbarrier_wait_parity(&mbar[s], ph);
        int kn = kt + STAGES - 1;
        if (kn < Kt) load(kn, kn % STAGES);
        if (valid_m_tile) {
            compute_64x128_frag_m4n8(
                sA[pair][s], sB[s], sSFA[pair][s], sSFB[s],
                acc, ngrp, t0, t1, lid);
        }
        __syncthreads();
    }

    if (valid_m_tile) {
        if (M_orig == M_tiles * 64) {
            if (alpha == 1.0f) {
                epilogue_64x128_m4n8_bf16_full<true>(
                    C, acc, alpha, cm, cn, ngrp, t1, lid, N);
            } else {
                epilogue_64x128_m4n8_bf16_full<false>(
                    C, acc, alpha, cm, cn, ngrp, t1, lid, N);
            }
        } else {
            epilogue_128x128_bf16_guarded(
                C, acc, alpha, cm, cn, 0, ngrp, t1, lid, M_orig, N);
        }
    }
}

template <int STAGES>
__global__ __launch_bounds__(256, 2)
void kern_prefill_mma_padded_bf16_m64x2_bpair(
    const uint8_t* __restrict__ At, const uint8_t* __restrict__ Bt,
    const uint8_t* __restrict__ Ast, const uint8_t* __restrict__ Bst,
    __nv_bfloat16* __restrict__ C, const float* __restrict__ alpha_ptr,
    int M_orig, int M_tiles, int N, int K)
{
    const int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31;
    const int pair = wid >> 2;
    const int nwarp = wid & 3;
    const int t0 = lid & 3, t1 = lid >> 2;
    const int m_tile = blockIdx.x * 2 + pair;
    const bool valid_m_tile = m_tile < M_tiles;
    const int cm = m_tile * 64, cn = blockIdx.y * 128;
    const int Kt = K / 64, tM = M_tiles, tN = N / 128;
    const float alpha = *alpha_ptr;

    __shared__ __align__(128) uint8_t sA[2][STAGES][2048];
    __shared__ __align__(128) uint8_t sB[STAGES][4096];
    __shared__ __align__(128) uint8_t sSFA[2][STAGES][256];
    __shared__ __align__(128) uint8_t sSFB[STAGES][512];
    __shared__ __align__(64)  uint64_t mbar[STAGES];

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; s++) mbarrier_init(&mbar[s], 1);
    }
    __syncthreads();

    auto load = [&](int kt, int s) {
        if (tid == 0) {
            const int m0 = blockIdx.x * 2;
            const int m1 = m0 + 1;
            const bool valid0 = m0 < M_tiles;
            const bool valid1 = m1 < M_tiles;
            uint32_t bytes = 4096 + 512;
            if (valid0) bytes += 2048 + 256;
            if (valid1) bytes += 2048 + 256;
            mbarrier_arrive_expect_tx(&mbar[s], bytes);

            if (valid0) {
                size_t am0 = (size_t)kt * tM + m0;
                cp_async_bulk(sA[0][s], At + am0 * 2048, 2048, &mbar[s]);
                cp_async_bulk(sSFA[0][s], Ast + am0 * 256, 256, &mbar[s]);
            }
            if (valid1) {
                size_t am1 = (size_t)kt * tM + m1;
                cp_async_bulk(sA[1][s], At + am1 * 2048, 2048, &mbar[s]);
                cp_async_bulk(sSFA[1][s], Ast + am1 * 256, 256, &mbar[s]);
            }

            size_t bn = (size_t)kt * tN + blockIdx.y;
            cp_async_bulk(sB[s], Bt + bn * 4096, 4096, &mbar[s]);
            cp_async_bulk(sSFB[s], Bst + bn * 512, 512, &mbar[s]);
        }
    };

    for (int s = 0; s < STAGES - 1 && s < Kt; s++) load(s, s);
    #pragma unroll 1
    for (int kt = 0; kt < Kt; kt++) {
        int s = kt % STAGES, ph = (kt / STAGES) & 1;
        mbarrier_wait_parity(&mbar[s], ph);
        int kn = kt + STAGES - 1;
        if (kn < Kt) load(kn, kn % STAGES);
        if (valid_m_tile) {
            compute_64x128_frag_m4n4_bpair(
                sA[pair][s], sB[s], sSFA[pair][s], sSFB[s],
                acc, nwarp, t0, t1, lid);
        }
        __syncthreads();
    }

    if (valid_m_tile) {
        if (M_orig == M_tiles * 64) {
            if (alpha == 1.0f) {
                epilogue_64x128_m4n4_bf16_full<true>(
                    C, acc, alpha, cm, cn, nwarp, t1, lid, N);
            } else {
                epilogue_64x128_m4n4_bf16_full<false>(
                    C, acc, alpha, cm, cn, nwarp, t1, lid, N);
            }
        } else {
            epilogue_64x128_m4n4_bf16_guarded(
                C, acc, alpha, cm, cn, nwarp, t1, lid, M_orig, N);
        }
    }
}

template <int STAGES>
__global__ __launch_bounds__(256, 2)
void kern_prefill_mma_padded_bf16_m64_n2_areuse_bpair(
    const uint8_t* __restrict__ At, const uint8_t* __restrict__ Bt,
    const uint8_t* __restrict__ Ast, const uint8_t* __restrict__ Bst,
    __nv_bfloat16* __restrict__ C, const float* __restrict__ alpha_ptr,
    int M_orig, int M_tiles, int N, int K)
{
    const int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31;
    const int n_sub = wid >> 2;
    const int nwarp = wid & 3;
    const int t0 = lid & 3, t1 = lid >> 2;
    const int m_tile = blockIdx.x;
    const int n_tile = blockIdx.y * 2 + n_sub;
    const int n_tile0 = blockIdx.y * 2;
    const int n_tile1 = n_tile0 + 1;
    const int cm = m_tile * 64;
    const int cn = n_tile * 128;
    const int Kt = K / 64, tM = M_tiles, tN = N / 128;
    const bool valid_n_tile = n_tile < tN;
    const bool valid_n0 = n_tile0 < tN;
    const bool valid_n1 = n_tile1 < tN;
    const float alpha = *alpha_ptr;

    __shared__ __align__(128) uint8_t sA[STAGES][2048];
    __shared__ __align__(128) uint8_t sB[2][STAGES][4096];
    __shared__ __align__(128) uint8_t sSFA[STAGES][256];
    __shared__ __align__(128) uint8_t sSFB[2][STAGES][512];
    __shared__ __align__(64)  uint64_t mbar[STAGES];

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; s++) mbarrier_init(&mbar[s], 1);
    }
    __syncthreads();

    auto load = [&](int kt, int s) {
        if (tid == 0) {
            uint32_t bytes = 2048 + 256;
            if (valid_n0) bytes += 4096 + 512;
            if (valid_n1) bytes += 4096 + 512;
            mbarrier_arrive_expect_tx(&mbar[s], bytes);

            size_t am = (size_t)kt * tM + m_tile;
            cp_async_bulk(sA[s], At + am * 2048, 2048, &mbar[s]);
            cp_async_bulk(sSFA[s], Ast + am * 256, 256, &mbar[s]);

            if (valid_n0) {
                size_t bn0 = (size_t)kt * tN + n_tile0;
                cp_async_bulk(sB[0][s], Bt + bn0 * 4096, 4096, &mbar[s]);
                cp_async_bulk(sSFB[0][s], Bst + bn0 * 512, 512, &mbar[s]);
            }
            if (valid_n1) {
                size_t bn1 = (size_t)kt * tN + n_tile1;
                cp_async_bulk(sB[1][s], Bt + bn1 * 4096, 4096, &mbar[s]);
                cp_async_bulk(sSFB[1][s], Bst + bn1 * 512, 512, &mbar[s]);
            }
        }
    };

    for (int s = 0; s < STAGES - 1 && s < Kt; s++) load(s, s);
    #pragma unroll 1
    for (int kt = 0; kt < Kt; kt++) {
        int s = kt % STAGES, ph = (kt / STAGES) & 1;
        mbarrier_wait_parity(&mbar[s], ph);
        int kn = kt + STAGES - 1;
        if (kn < Kt) load(kn, kn % STAGES);
        if (valid_n_tile) {
            compute_64x128_frag_m4n4_bpair(
                sA[s], sB[n_sub][s], sSFA[s], sSFB[n_sub][s],
                acc, nwarp, t0, t1, lid);
        }
        __syncthreads();
    }

    if (valid_n_tile) {
        if (M_orig == M_tiles * 64) {
            if (alpha == 1.0f) {
                epilogue_64x128_m4n4_bf16_full<true>(
                    C, acc, alpha, cm, cn, nwarp, t1, lid, N);
            } else {
                epilogue_64x128_m4n4_bf16_full<false>(
                    C, acc, alpha, cm, cn, nwarp, t1, lid, N);
            }
        } else {
            epilogue_64x128_m4n4_bf16_guarded(
                C, acc, alpha, cm, cn, nwarp, t1, lid, M_orig, N);
        }
    }
}

template <int STAGES>
__global__ __launch_bounds__(256, 2)
void kern_prefill_mma_padded_bf16_m64x2_bpair_persistent(
    const uint8_t* __restrict__ At, const uint8_t* __restrict__ Bt,
    const uint8_t* __restrict__ Ast, const uint8_t* __restrict__ Bst,
    __nv_bfloat16* __restrict__ C, const float* __restrict__ alpha_ptr,
    int M_orig, int M_tiles, int N, int K)
{
    const int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31;
    const int pair = wid >> 2;
    const int nwarp = wid & 3;
    const int t0 = lid & 3, t1 = lid >> 2;
    const int Kt = K / 64, tM = M_tiles, tN = N / 128;
    const int m_pairs = (M_tiles + 1) / 2;
    const int total_tiles = m_pairs * tN;
    const bool m_pairs_pow2 = (m_pairs & (m_pairs - 1)) == 0;
    const int m_pair_mask = m_pairs - 1;
    const int m_pair_shift = 31 - __clz((unsigned)m_pairs);
    const float alpha = *alpha_ptr;

    __shared__ __align__(128) uint8_t sA[2][STAGES][2048];
    __shared__ __align__(128) uint8_t sB[STAGES][4096];
    __shared__ __align__(128) uint8_t sSFA[2][STAGES][256];
    __shared__ __align__(128) uint8_t sSFB[STAGES][512];
    __shared__ __align__(64)  uint64_t mbar[STAGES];

    for (int tile = (int)blockIdx.x; tile < total_tiles; tile += (int)gridDim.x) {
        const int n_tile = m_pairs_pow2 ? (tile >> m_pair_shift) : (tile / m_pairs);
        const int m_pair = m_pairs_pow2 ? (tile & m_pair_mask) : (tile - n_tile * m_pairs);
        const int m0 = m_pair * 2;
        const int m1 = m0 + 1;
        const bool valid0 = m0 < M_tiles;
        const bool valid1 = m1 < M_tiles;
        const int m_tile = m0 + pair;
        const bool valid_m_tile = m_tile < M_tiles;
        const int cm = m_tile * 64;
        const int cn = n_tile * 128;

        float acc[4][4][4];
        #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int j = 0; j < 4; j++)
                acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

        if (tid == 0) {
            #pragma unroll
            for (int s = 0; s < STAGES; s++) mbarrier_init(&mbar[s], 1);
        }
        __syncthreads();

        auto load = [&](int kt, int s) {
            if (tid == 0) {
                uint32_t bytes = 4096 + 512;
                if (valid0) bytes += 2048 + 256;
                if (valid1) bytes += 2048 + 256;
                mbarrier_arrive_expect_tx(&mbar[s], bytes);

                if (valid0) {
                    size_t am0 = (size_t)kt * tM + m0;
                    cp_async_bulk(sA[0][s], At + am0 * 2048, 2048, &mbar[s]);
                    cp_async_bulk(sSFA[0][s], Ast + am0 * 256, 256, &mbar[s]);
                }
                if (valid1) {
                    size_t am1 = (size_t)kt * tM + m1;
                    cp_async_bulk(sA[1][s], At + am1 * 2048, 2048, &mbar[s]);
                    cp_async_bulk(sSFA[1][s], Ast + am1 * 256, 256, &mbar[s]);
                }

                size_t bn = (size_t)kt * tN + n_tile;
                cp_async_bulk(sB[s], Bt + bn * 4096, 4096, &mbar[s]);
                cp_async_bulk(sSFB[s], Bst + bn * 512, 512, &mbar[s]);
            }
        };

        for (int s = 0; s < STAGES - 1 && s < Kt; s++) load(s, s);
        #pragma unroll 1
        for (int kt = 0; kt < Kt; kt++) {
            int s = kt % STAGES, ph = (kt / STAGES) & 1;
            mbarrier_wait_parity(&mbar[s], ph);
            int kn = kt + STAGES - 1;
            if (kn < Kt) load(kn, kn % STAGES);
            if (valid_m_tile) {
                compute_64x128_frag_m4n4_bpair(
                    sA[pair][s], sB[s], sSFA[pair][s], sSFB[s],
                    acc, nwarp, t0, t1, lid);
            }
            __syncthreads();
        }

        if (valid_m_tile) {
            if (M_orig == M_tiles * 64) {
                if (alpha == 1.0f) {
                    epilogue_64x128_m4n4_bf16_full<true>(
                        C, acc, alpha, cm, cn, nwarp, t1, lid, N);
                } else {
                    epilogue_64x128_m4n4_bf16_full<false>(
                        C, acc, alpha, cm, cn, nwarp, t1, lid, N);
                }
            } else {
                epilogue_64x128_m4n4_bf16_guarded(
                    C, acc, alpha, cm, cn, nwarp, t1, lid, M_orig, N);
            }
        }
        __syncthreads();
    }
}

template <int STAGES>
__global__ __launch_bounds__(192, 2)
void kern_prefill_mma_padded_bf16_m64x2_mixed3_bpair(
    const uint8_t* __restrict__ At, const uint8_t* __restrict__ Bt,
    const uint8_t* __restrict__ Ast, const uint8_t* __restrict__ Bst,
    __nv_bfloat16* __restrict__ C, const float* __restrict__ alpha_ptr,
    int M_orig, int M_tiles, int N, int K)
{
    const int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31;
    const int pair = wid / 3;
    const int role = wid - pair * 3;
    const int t0 = lid & 3, t1 = lid >> 2;
    const int m_tile = blockIdx.x * 2 + pair;
    const bool valid_m_tile = m_tile < M_tiles;
    const int cm = m_tile * 64, cn = blockIdx.y * 128;
    const int Kt = K / 64, tM = M_tiles, tN = N / 128;
    const float alpha = *alpha_ptr;

    __shared__ __align__(128) uint8_t sA[2][STAGES][2048];
    __shared__ __align__(128) uint8_t sB[STAGES][4096];
    __shared__ __align__(128) uint8_t sSFA[2][STAGES][256];
    __shared__ __align__(128) uint8_t sSFB[STAGES][512];
    __shared__ __align__(64)  uint64_t mbar[STAGES];

    float acc[4][6][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 6; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; s++) mbarrier_init(&mbar[s], 1);
    }
    __syncthreads();

    auto load = [&](int kt, int s) {
        if (tid == 0) {
            const int m0 = blockIdx.x * 2;
            const int m1 = m0 + 1;
            const bool valid0 = m0 < M_tiles;
            const bool valid1 = m1 < M_tiles;
            uint32_t bytes = 4096 + 512;
            if (valid0) bytes += 2048 + 256;
            if (valid1) bytes += 2048 + 256;
            mbarrier_arrive_expect_tx(&mbar[s], bytes);

            if (valid0) {
                size_t am0 = (size_t)kt * tM + m0;
                cp_async_bulk(sA[0][s], At + am0 * 2048, 2048, &mbar[s]);
                cp_async_bulk(sSFA[0][s], Ast + am0 * 256, 256, &mbar[s]);
            }
            if (valid1) {
                size_t am1 = (size_t)kt * tM + m1;
                cp_async_bulk(sA[1][s], At + am1 * 2048, 2048, &mbar[s]);
                cp_async_bulk(sSFA[1][s], Ast + am1 * 256, 256, &mbar[s]);
            }

            size_t bn = (size_t)kt * tN + blockIdx.y;
            cp_async_bulk(sB[s], Bt + bn * 4096, 4096, &mbar[s]);
            cp_async_bulk(sSFB[s], Bst + bn * 512, 512, &mbar[s]);
        }
    };

    for (int s = 0; s < STAGES - 1 && s < Kt; s++) load(s, s);
    #pragma unroll 1
    for (int kt = 0; kt < Kt; kt++) {
        int s = kt % STAGES, ph = (kt / STAGES) & 1;
        mbarrier_wait_parity(&mbar[s], ph);
        int kn = kt + STAGES - 1;
        if (kn < Kt) load(kn, kn % STAGES);
        if (valid_m_tile) {
            if (role == 0) {
                compute_64x128_frag_mixed3_bpair<0>(
                    sA[pair][s], sB[s], sSFA[pair][s], sSFB[s],
                    acc, t0, t1, lid);
            } else if (role == 1) {
                compute_64x128_frag_mixed3_bpair<1>(
                    sA[pair][s], sB[s], sSFA[pair][s], sSFB[s],
                    acc, t0, t1, lid);
            } else {
                compute_64x128_frag_mixed3_bpair<2>(
                    sA[pair][s], sB[s], sSFA[pair][s], sSFB[s],
                    acc, t0, t1, lid);
            }
        }
        __syncthreads();
    }

    if (valid_m_tile) {
        if (M_orig == M_tiles * 64) {
            if (alpha == 1.0f) {
                if (role == 0) {
                    epilogue_64x128_mixed3_bf16_full<0, true>(
                        C, acc, alpha, cm, cn, t1, lid, N);
                } else if (role == 1) {
                    epilogue_64x128_mixed3_bf16_full<1, true>(
                        C, acc, alpha, cm, cn, t1, lid, N);
                } else {
                    epilogue_64x128_mixed3_bf16_full<2, true>(
                        C, acc, alpha, cm, cn, t1, lid, N);
                }
            } else {
                if (role == 0) {
                    epilogue_64x128_mixed3_bf16_full<0, false>(
                        C, acc, alpha, cm, cn, t1, lid, N);
                } else if (role == 1) {
                    epilogue_64x128_mixed3_bf16_full<1, false>(
                        C, acc, alpha, cm, cn, t1, lid, N);
                } else {
                    epilogue_64x128_mixed3_bf16_full<2, false>(
                        C, acc, alpha, cm, cn, t1, lid, N);
                }
            }
        } else {
            if (role == 0) {
                epilogue_64x128_mixed3_bf16_guarded<0>(
                    C, acc, alpha, cm, cn, t1, lid, M_orig, N);
            } else if (role == 1) {
                epilogue_64x128_mixed3_bf16_guarded<1>(
                    C, acc, alpha, cm, cn, t1, lid, M_orig, N);
            } else {
                epilogue_64x128_mixed3_bf16_guarded<2>(
                    C, acc, alpha, cm, cn, t1, lid, M_orig, N);
            }
        }
    }
}

__device__ __forceinline__ uint32_t load_a_frag_word_from_cutlass_layout(
    const uint8_t* __restrict__ A, int M, int K, int m_tile, int kt, int word);

__device__ __forceinline__ uint32_t load_asf_frag_word_from_cutlass_layout(
    const uint8_t* __restrict__ Asf, int M, int K, int m_tile, int kg, int word);

#if USE_PREFILL_TMA_B
template <int STAGES>
__global__ __launch_bounds__(128, 4)
void kern_prefill_mma_padded_bf16_m64_tma_b(
    const uint8_t* __restrict__ At, const uint8_t* __restrict__ Bt,
    const uint8_t* __restrict__ Ast, const uint8_t* __restrict__ Bst,
    __nv_bfloat16* __restrict__ C, const float* __restrict__ alpha_ptr,
    int M_orig, int M_tiles, int N, int K,
    CUTE_GRID_CONSTANT CUtensorMap const b_tma_desc)
{
    (void)Bt;
    const int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31;
    const int nwarp = wid, t0 = lid & 3, t1 = lid >> 2;
    const int cm = blockIdx.x * 64, cn = blockIdx.y * 128;
    const int Kt = K / 64, tM = M_tiles, tN = N / 128;
    const float alpha = *alpha_ptr;

    __shared__ __align__(128) uint8_t sA[STAGES][2048];
    __shared__ __align__(128) uint8_t sB[STAGES][4096];
    __shared__ __align__(128) uint8_t sSFA[STAGES][256];
    __shared__ __align__(128) uint8_t sSFB[STAGES][512];
    __shared__ __align__(64)  uint64_t mbar[STAGES];
    __shared__ __align__(64)  uint64_t tma_b_mbar[STAGES];

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; s++) {
            mbarrier_init(&mbar[s], 1);
            cute::initialize_barrier(tma_b_mbar[s], 1);
        }
    }
    __syncthreads();
    if (tid == 0) {
        cute::prefetch_tma_descriptor(&b_tma_desc);
    }
    __syncthreads();

    auto load = [&](int kt, int s) {
        if (tid == 0) {
            mbarrier_arrive_expect_tx(&mbar[s], 2048 + 256 + 512);
            size_t am = (size_t)kt * tM + blockIdx.x;
            cp_async_bulk(sA[s], At + am * 2048, 2048, &mbar[s]);
            cp_async_bulk(sSFA[s], Ast + am * 256, 256, &mbar[s]);
            cp_async_bulk(sSFB[s], Bst + ((size_t)kt * tN + blockIdx.y) * 512,
                          512, &mbar[s]);

            cute::set_barrier_transaction_bytes(tma_b_mbar[s], 4096);
            cute::SM90_TMA_LOAD_2D::copy(
                &b_tma_desc,
                &tma_b_mbar[s],
                static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                sB[s],
                0,
                static_cast<int>(((size_t)kt * tN + blockIdx.y) * 128));
        }
    };

    for (int s = 0; s < STAGES - 1 && s < Kt; s++) load(s, s);
    for (int kt = 0; kt < Kt; kt++) {
        int s = kt % STAGES, ph = (kt / STAGES) & 1;
        mbarrier_wait_parity(&mbar[s], ph);
        mbarrier_wait_parity(&tma_b_mbar[s], ph);
        int kn = kt + STAGES - 1;
        if (kn < Kt) load(kn, kn % STAGES);
        compute_64x128_frag_m4n4(sA[s], sB[s], sSFA[s], sSFB[s], acc, nwarp, t0, t1, lid);
        __syncthreads();
    }

    epilogue_64x128_m4n4_bf16_guarded(C, acc, alpha, cm, cn, nwarp, t1, lid, M_orig, N);
}

template <int STAGES>
__global__ __launch_bounds__(128, 4)
void kern_prefill_mma_padded_bf16_m64_tma_direct_b(
    const uint8_t* __restrict__ At, const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ Ast, const uint8_t* __restrict__ Bsf,
    __nv_bfloat16* __restrict__ C, const float* __restrict__ alpha_ptr,
    int M_orig, int M_tiles, int N, int K,
    CUTE_GRID_CONSTANT CUtensorMap const b_tma_desc)
{
    (void)B;
    const int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31;
    const int nwarp = wid, t0 = lid & 3, t1 = lid >> 2;
    const int cm = blockIdx.x * 64, cn = blockIdx.y * 128;
    const int Kt = K / 64, tM = M_tiles, Ksf = K / 16;
    const float alpha = *alpha_ptr;

    __shared__ __align__(128) uint8_t sA[STAGES][2048];
    __shared__ __align__(128) uint8_t sB_row[STAGES][4096];
    __shared__ __align__(128) uint8_t sSFA[STAGES][256];
    __shared__ __align__(128) uint8_t sSFB_swiz[STAGES][512];
    __shared__ __align__(64)  uint64_t mbar[STAGES];
    __shared__ __align__(64)  uint64_t tma_b_mbar[STAGES];

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; s++) {
            mbarrier_init(&mbar[s], 1);
            cute::initialize_barrier(tma_b_mbar[s], 1);
        }
    }
    __syncthreads();
    if (tid == 0) {
        cute::prefetch_tma_descriptor(&b_tma_desc);
    }
    __syncthreads();

    auto load = [&](int kt, int s) {
        if (tid == 0) {
            mbarrier_arrive_expect_tx(&mbar[s], 2048 + 256 + 512);
            size_t am = (size_t)kt * tM + blockIdx.x;
            cp_async_bulk(sA[s], At + am * 2048, 2048, &mbar[s]);
            cp_async_bulk(sSFA[s], Ast + am * 256, 256, &mbar[s]);
            cp_async_bulk(sSFB_swiz[s],
                          Bsf + (size_t)blockIdx.y * 128 * Ksf + (size_t)kt * 512,
                          512, &mbar[s]);

            cute::set_barrier_transaction_bytes(tma_b_mbar[s], 4096);
            cute::SM90_TMA_LOAD_2D::copy(
                &b_tma_desc,
                &tma_b_mbar[s],
                static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                sB_row[s],
                kt * 32,
                blockIdx.y * 128);
        }
    };

    for (int s = 0; s < STAGES - 1 && s < Kt; s++) load(s, s);
    for (int kt = 0; kt < Kt; kt++) {
        int s = kt % STAGES, ph = (kt / STAGES) & 1;
        mbarrier_wait_parity(&mbar[s], ph);
        mbarrier_wait_parity(&tma_b_mbar[s], ph);
        int kn = kt + STAGES - 1;
        if (kn < Kt) load(kn, kn % STAGES);
        compute_64x128_frag_a_brow_m4n4(
            sA[s], sB_row[s], sSFA[s], sSFB_swiz[s], acc, nwarp, t0, t1, lid);
        __syncthreads();
    }

    epilogue_64x128_m4n4_bf16_guarded(C, acc, alpha, cm, cn, nwarp, t1, lid, M_orig, N);
}

template <int STAGES>
__global__ __launch_bounds__(128, 4)
void kern_prefill_mma_padded_bf16_m64_tma_direct_b_k17408_n5120_mtiles5(
    const uint8_t* __restrict__ At, const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ Ast, const uint8_t* __restrict__ Bsf,
    __nv_bfloat16* __restrict__ C, const float* __restrict__ alpha_ptr,
    int M_orig,
    CUTE_GRID_CONSTANT CUtensorMap const b_tma_desc)
{
    (void)B;
    constexpr int N_CONST = 5120;
    constexpr int Kt = 17408 / 64;
    constexpr int Ksf = 17408 / 16;
    constexpr int M_TILES = 5;
    const int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31;
    const int nwarp = wid, t0 = lid & 3, t1 = lid >> 2;
    const int cm = blockIdx.x * 64, cn = blockIdx.y * 128;
    const float alpha = *alpha_ptr;

    __shared__ __align__(128) uint8_t sA[STAGES][2048];
    __shared__ __align__(128) uint8_t sB_row[STAGES][4096];
    __shared__ __align__(128) uint8_t sSFA[STAGES][256];
    __shared__ __align__(128) uint8_t sSFB_swiz[STAGES][512];
    __shared__ __align__(64)  uint64_t mbar[STAGES];
    __shared__ __align__(64)  uint64_t tma_b_mbar[STAGES];

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; s++) {
            mbarrier_init(&mbar[s], 1);
            cute::initialize_barrier(tma_b_mbar[s], 1);
        }
    }
    __syncthreads();
    if (tid == 0) {
        cute::prefetch_tma_descriptor(&b_tma_desc);
    }
    __syncthreads();

    auto load = [&](int kt, int s) {
        if (tid == 0) {
            mbarrier_arrive_expect_tx(&mbar[s], 2048 + 256 + 512);
            size_t am = (size_t)kt * M_TILES + blockIdx.x;
            cp_async_bulk(sA[s], At + am * 2048, 2048, &mbar[s]);
            cp_async_bulk(sSFA[s], Ast + am * 256, 256, &mbar[s]);
            cp_async_bulk(sSFB_swiz[s],
                          Bsf + (size_t)blockIdx.y * 128 * Ksf + (size_t)kt * 512,
                          512, &mbar[s]);

            cute::set_barrier_transaction_bytes(tma_b_mbar[s], 4096);
            cute::SM90_TMA_LOAD_2D::copy(
                &b_tma_desc,
                &tma_b_mbar[s],
                static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                sB_row[s],
                kt * 32,
                blockIdx.y * 128);
        }
    };

    for (int s = 0; s < STAGES - 1; s++) load(s, s);
    #pragma unroll 1
    for (int kt = 0; kt < Kt; kt++) {
        int s = kt % STAGES, ph = (kt / STAGES) & 1;
        mbarrier_wait_parity(&mbar[s], ph);
        mbarrier_wait_parity(&tma_b_mbar[s], ph);
        int kn = kt + STAGES - 1;
        if (kn < Kt) load(kn, kn % STAGES);
        compute_64x128_frag_a_brow_m4n4(
            sA[s], sB_row[s], sSFA[s], sSFB_swiz[s], acc, nwarp, t0, t1, lid);
        __syncthreads();
    }

    epilogue_64x128_m4n4_bf16_guarded(
        C, acc, alpha, cm, cn, nwarp, t1, lid, M_orig, N_CONST);
}

template <int STAGES>
__global__ __launch_bounds__(128, 4)
void kern_prefill_mma_padded_bf16_m64_direct_a_tma_direct_b_k17408_n5120_mtiles5(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ Asf, const uint8_t* __restrict__ Bsf,
    __nv_bfloat16* __restrict__ C, const float* __restrict__ alpha_ptr,
    int M_orig,
    CUTE_GRID_CONSTANT CUtensorMap const b_tma_desc)
{
    (void)B;
    constexpr int N_CONST = 5120;
    constexpr int K_CONST = 17408;
    constexpr int Kt = K_CONST / 64;
    constexpr int Ksf = K_CONST / 16;
    const int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31;
    const int nwarp = wid, t0 = lid & 3, t1 = lid >> 2;
    const int cm = blockIdx.x * 64, cn = blockIdx.y * 128;
    const float alpha = *alpha_ptr;

    __shared__ __align__(128) uint8_t sA[2048];
    __shared__ __align__(128) uint8_t sB_row[STAGES][4096];
    __shared__ __align__(128) uint8_t sSFA[256];
    __shared__ __align__(128) uint8_t sSFB_swiz[STAGES][512];
    __shared__ __align__(64)  uint64_t sfb_mbar[STAGES];
    __shared__ __align__(64)  uint64_t tma_b_mbar[STAGES];

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; s++) {
            mbarrier_init(&sfb_mbar[s], 1);
            cute::initialize_barrier(tma_b_mbar[s], 1);
        }
    }
    __syncthreads();
    if (tid == 0) {
        cute::prefetch_tma_descriptor(&b_tma_desc);
    }
    __syncthreads();

    auto load_b = [&](int kt, int s) {
        if (tid == 0) {
            mbarrier_arrive_expect_tx(&sfb_mbar[s], 512);
            cp_async_bulk(sSFB_swiz[s],
                          Bsf + (size_t)blockIdx.y * 128 * Ksf + (size_t)kt * 512,
                          512, &sfb_mbar[s]);

            cute::set_barrier_transaction_bytes(tma_b_mbar[s], 4096);
            cute::SM90_TMA_LOAD_2D::copy(
                &b_tma_desc,
                &tma_b_mbar[s],
                static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                sB_row[s],
                kt * 32,
                blockIdx.y * 128);
        }
    };

    for (int s = 0; s < STAGES - 1; s++) load_b(s, s);
    #pragma unroll 1
    for (int kt = 0; kt < Kt; kt++) {
        int s = kt % STAGES, ph = (kt / STAGES) & 1;
        mbarrier_wait_parity(&sfb_mbar[s], ph);
        mbarrier_wait_parity(&tma_b_mbar[s], ph);
        int kn = kt + STAGES - 1;
        if (kn < Kt) load_b(kn, kn % STAGES);

        #pragma unroll
        for (int w = tid; w < 512; w += 128) {
            ((uint32_t*)sA)[w] = load_a_frag_word_from_cutlass_layout(
                A, M_orig, K_CONST, blockIdx.x, kt, w);
        }
        for (int w = tid; w < 64; w += 128) {
            ((uint32_t*)sSFA)[w] = load_asf_frag_word_from_cutlass_layout(
                Asf, M_orig, K_CONST, blockIdx.x, kt, w);
        }
        __syncthreads();
        compute_64x128_frag_a_brow_m4n4(
            sA, sB_row[s], sSFA, sSFB_swiz[s], acc, nwarp, t0, t1, lid);
        __syncthreads();
    }

    epilogue_64x128_m4n4_bf16_guarded(
        C, acc, alpha, cm, cn, nwarp, t1, lid, M_orig, N_CONST);
}

template <int STAGES>
__global__ __launch_bounds__(128, 2)
void kern_prefill_mma_padded_bf16_m64_coop_arepack_tma_direct_b_k17408_n5120_mtiles5(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ Asf, const uint8_t* __restrict__ Bsf,
    __nv_bfloat16* __restrict__ C, const float* __restrict__ alpha_ptr,
    int M_orig,
    uint8_t* __restrict__ A_frag,
    uint8_t* __restrict__ A_sf_frag,
    CUTE_GRID_CONSTANT CUtensorMap const b_tma_desc)
{
    (void)B;
    constexpr int N_CONST = 5120;
    constexpr int K_CONST = 17408;
    constexpr int Kh = K_CONST / 2;
    constexpr int Kt = K_CONST / 64;
    constexpr int Ksf = K_CONST / 16;
    constexpr int RowsPad = 320;
    constexpr int M_TILES = 5;
    constexpr int TRows = RowsPad / 64;

    const int tid = threadIdx.x;
    const int block_linear = blockIdx.y * gridDim.x + blockIdx.x;
    const int grid_threads = gridDim.x * gridDim.y * blockDim.x;
    const int linear_tid = block_linear * blockDim.x + tid;

    constexpr int data_words = Kt * RowsPad * 8;
    for (int idx = linear_tid; idx < data_words; idx += grid_threads) {
        int rem = idx;
        const int kt = rem / (RowsPad * 8);
        rem -= kt * RowsPad * 8;
        const int row = rem >> 3;
        const int qw = rem & 7;

        uint32_t v = 0;
        if (row < M_orig) {
            const int src_off = row * Kh + kt * 32 + qw * 4;
            v = *(const uint32_t*)(A + src_off);
        }

        const int tm = row >> 6;
        const int lm = row & 63;
        const int wr = lm >> 5;
        const int rem32 = lm & 31;
        const int a_group = rem32 >> 4;
        const int rem16 = rem32 & 15;
        const int row_hi = rem16 >> 3;
        const int t1 = rem16 & 7;
        const int t0 = qw & 3;
        const int q_hi = qw >> 2;
        const int frag_lane = t1 * 4 + t0;
        const int slot = q_hi * 2 + row_hi;
        const int dst_off = (kt * TRows + tm) * (64 * 32) +
                            (wr * 256 + a_group * 128 + frag_lane * 4 + slot) * 4;
        *(uint32_t*)(A_frag + dst_off) = v;
    }

    constexpr int sf_words = Kt * RowsPad;
    for (int idx = linear_tid; idx < sf_words; idx += grid_threads) {
        const int kt = idx / RowsPad;
        const int row = idx - kt * RowsPad;

        uint32_t v = 0;
        if (row < M_orig) {
            const int src_tm = row >> 7;
            const int r = row & 127;
            const int r_hi = r >> 5;
            const int r_lo = r & 31;
            const int src_idx = src_tm * 128 * Ksf + kt * 512 + r_lo * 16 + r_hi * 4;
            v = *(const uint32_t*)(Asf + src_idx);
        }

        const int dst_tm = row >> 6;
        const int lm = row & 63;
        const int mi = lm >> 4;
        const int scale_lane = sfa_phys_scale_lane_64(lm & 15);
        const int dst_word = (kt * TRows + dst_tm) * 64 + scale_lane * 4 + mi;
        *(uint32_t*)(A_sf_frag + dst_word * 4) = v;
    }

    cg::this_grid().sync();

    const int wid = tid >> 5, lid = tid & 31;
    const int nwarp = wid, t0 = lid & 3, t1 = lid >> 2;
    const int cm = blockIdx.x * 64, cn = blockIdx.y * 128;
    const float alpha = *alpha_ptr;

    __shared__ __align__(128) uint8_t sA[STAGES][2048];
    __shared__ __align__(128) uint8_t sB_row[STAGES][4096];
    __shared__ __align__(128) uint8_t sSFA[STAGES][256];
    __shared__ __align__(128) uint8_t sSFB_swiz[STAGES][512];
    __shared__ __align__(64)  uint64_t mbar[STAGES];
    __shared__ __align__(64)  uint64_t tma_b_mbar[STAGES];

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; s++) {
            mbarrier_init(&mbar[s], 1);
            cute::initialize_barrier(tma_b_mbar[s], 1);
        }
    }
    __syncthreads();
    if (tid == 0) {
        cute::prefetch_tma_descriptor(&b_tma_desc);
    }
    __syncthreads();

    auto load = [&](int kt, int s) {
        if (tid == 0) {
            mbarrier_arrive_expect_tx(&mbar[s], 2048 + 256 + 512);
            size_t am = (size_t)kt * M_TILES + blockIdx.x;
            cp_async_bulk(sA[s], A_frag + am * 2048, 2048, &mbar[s]);
            cp_async_bulk(sSFA[s], A_sf_frag + am * 256, 256, &mbar[s]);
            cp_async_bulk(sSFB_swiz[s],
                          Bsf + (size_t)blockIdx.y * 128 * Ksf + (size_t)kt * 512,
                          512, &mbar[s]);

            cute::set_barrier_transaction_bytes(tma_b_mbar[s], 4096);
            cute::SM90_TMA_LOAD_2D::copy(
                &b_tma_desc,
                &tma_b_mbar[s],
                static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                sB_row[s],
                kt * 32,
                blockIdx.y * 128);
        }
    };

    for (int s = 0; s < STAGES - 1; s++) load(s, s);
    #pragma unroll 1
    for (int kt = 0; kt < Kt; kt++) {
        int s = kt % STAGES, ph = (kt / STAGES) & 1;
        mbarrier_wait_parity(&mbar[s], ph);
        mbarrier_wait_parity(&tma_b_mbar[s], ph);
        int kn = kt + STAGES - 1;
        if (kn < Kt) load(kn, kn % STAGES);
        compute_64x128_frag_a_brow_m4n4(
            sA[s], sB_row[s], sSFA[s], sSFB_swiz[s], acc, nwarp, t0, t1, lid);
        __syncthreads();
    }

    epilogue_64x128_m4n4_bf16_guarded(
        C, acc, alpha, cm, cn, nwarp, t1, lid, M_orig, N_CONST);
}

template <int STAGES>
__global__ __launch_bounds__(256, 2)
void kern_prefill_mma_padded_bf16_m64x2_tma_direct_b_k17408_n5120_mtiles5(
    const uint8_t* __restrict__ At, const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ Ast, const uint8_t* __restrict__ Bsf,
    __nv_bfloat16* __restrict__ C, const float* __restrict__ alpha_ptr,
    int M_orig,
    CUTE_GRID_CONSTANT CUtensorMap const b_tma_desc)
{
    (void)B;
    constexpr int N_CONST = 5120;
    constexpr int Kt = 17408 / 64;
    constexpr int Ksf = 17408 / 16;
    constexpr int M_TILES = 5;

    const int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31;
    const int pair = wid >> 2;
    const int nwarp = wid & 3;
    const int t0 = lid & 3, t1 = lid >> 2;
    const int m_tile = blockIdx.x * 2 + pair;
    const bool valid_m_tile = m_tile < M_TILES;
    const int cm = m_tile * 64, cn = blockIdx.y * 128;
    const float alpha = *alpha_ptr;

    __shared__ __align__(128) uint8_t sA[2][STAGES][2048];
    __shared__ __align__(128) uint8_t sB_row[STAGES][4096];
    __shared__ __align__(128) uint8_t sSFA[2][STAGES][256];
    __shared__ __align__(128) uint8_t sSFB_swiz[STAGES][512];
    __shared__ __align__(64)  uint64_t mbar[STAGES];
    __shared__ __align__(64)  uint64_t tma_b_mbar[STAGES];

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; s++) {
            mbarrier_init(&mbar[s], 1);
            cute::initialize_barrier(tma_b_mbar[s], 1);
        }
    }
    __syncthreads();
    if (tid == 0) {
        cute::prefetch_tma_descriptor(&b_tma_desc);
    }
    __syncthreads();

    auto load = [&](int kt, int s) {
        if (tid == 0) {
            const int m0 = blockIdx.x * 2;
            const int m1 = m0 + 1;
            const bool valid0 = m0 < M_TILES;
            const bool valid1 = m1 < M_TILES;
            uint32_t bytes = 512;
            if (valid0) bytes += 2048 + 256;
            if (valid1) bytes += 2048 + 256;
            mbarrier_arrive_expect_tx(&mbar[s], bytes);
            if (valid0) {
                size_t am0 = (size_t)kt * M_TILES + m0;
                cp_async_bulk(sA[0][s], At + am0 * 2048, 2048, &mbar[s]);
                cp_async_bulk(sSFA[0][s], Ast + am0 * 256, 256, &mbar[s]);
            }
            if (valid1) {
                size_t am1 = (size_t)kt * M_TILES + m1;
                cp_async_bulk(sA[1][s], At + am1 * 2048, 2048, &mbar[s]);
                cp_async_bulk(sSFA[1][s], Ast + am1 * 256, 256, &mbar[s]);
            }
            cp_async_bulk(sSFB_swiz[s],
                          Bsf + (size_t)blockIdx.y * 128 * Ksf + (size_t)kt * 512,
                          512, &mbar[s]);

            cute::set_barrier_transaction_bytes(tma_b_mbar[s], 4096);
            cute::SM90_TMA_LOAD_2D::copy(
                &b_tma_desc,
                &tma_b_mbar[s],
                static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                sB_row[s],
                kt * 32,
                blockIdx.y * 128);
        }
    };

    for (int s = 0; s < STAGES - 1; s++) load(s, s);
    #pragma unroll 1
    for (int kt = 0; kt < Kt; kt++) {
        int s = kt % STAGES, ph = (kt / STAGES) & 1;
        mbarrier_wait_parity(&mbar[s], ph);
        mbarrier_wait_parity(&tma_b_mbar[s], ph);
        int kn = kt + STAGES - 1;
        if (kn < Kt) load(kn, kn % STAGES);
        if (valid_m_tile) {
            compute_64x128_frag_a_brow_m4n4(
                sA[pair][s], sB_row[s], sSFA[pair][s], sSFB_swiz[s],
                acc, nwarp, t0, t1, lid);
        }
        __syncthreads();
    }

    if (valid_m_tile) {
        epilogue_64x128_m4n4_bf16_guarded(
            C, acc, alpha, cm, cn, nwarp, t1, lid, M_orig, N_CONST);
    }
}

#if USE_PREFILL_TMA_BREP && USE_PREFILL_WS_TMA_BREP
template <int STAGES>
__global__ __launch_bounds__(160, 1)
void kern_prefill_mma_padded_bf16_m64_ws_tma_brep(
    const uint8_t* __restrict__ At, const uint8_t* __restrict__ Bt,
    const uint8_t* __restrict__ Ast, const uint8_t* __restrict__ Bst,
    __nv_bfloat16* __restrict__ C, const float* __restrict__ alpha_ptr,
    int M_orig, int M_tiles, int N, int K,
    CUTE_GRID_CONSTANT CUtensorMap const b_tma_desc)
{
    (void)Bt;
    const int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31;
    const int Kt = K / 64, tM = M_tiles, tN = N / 128;

    __shared__ __align__(128) uint8_t sA[STAGES][2048];
    __shared__ __align__(128) uint8_t sB[STAGES][4096];
    __shared__ __align__(128) uint8_t sSFA[STAGES][256];
    __shared__ __align__(128) uint8_t sSFB[STAGES][512];
    __shared__ __align__(64)  uint64_t full_bulk_mbar[STAGES];
    __shared__ __align__(64)  uint64_t full_b_mbar[STAGES];
    __shared__ __align__(64)  uint64_t empty_mbar[STAGES];

    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; s++) {
            mbarrier_init(&full_bulk_mbar[s], 1);
            cute::initialize_barrier(full_b_mbar[s], 1);
            mbarrier_init(&empty_mbar[s], 4);
        }
    }
    __syncthreads();
    if (tid == 0) cute::prefetch_tma_descriptor(&b_tma_desc);
    __syncthreads();

    if (wid == 0) {
        warp_reg_dealloc<PREFILL_WS_PRODUCER_REGS>();
        if (tid == 0) {
            #pragma unroll 1
            for (int kt = 0; kt < Kt; kt++) {
                const int s = kt % STAGES;
                const int ph = (kt / STAGES) & 1;
                if (kt >= STAGES) {
                    mbarrier_wait_parity(&empty_mbar[s], ph ^ 1);
                }

                const size_t am = (size_t)kt * tM + blockIdx.x;
                const size_t bn = (size_t)kt * tN + blockIdx.y;
                mbarrier_arrive_expect_tx(&full_bulk_mbar[s], 2048 + 256 + 512);
                cp_async_bulk(sA[s], At + am * 2048, 2048, &full_bulk_mbar[s]);
                cp_async_bulk(sSFA[s], Ast + am * 256, 256, &full_bulk_mbar[s]);
                cp_async_bulk(sSFB[s], Bst + bn * 512, 512, &full_bulk_mbar[s]);

                cute::set_barrier_transaction_bytes(full_b_mbar[s], 4096);
                cute::SM90_TMA_LOAD_2D::copy(
                    &b_tma_desc,
                    &full_b_mbar[s],
                    static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                    sB[s],
                    0,
                    static_cast<int>(bn * 128));
            }
        }
        return;
    }

    warp_reg_alloc<PREFILL_WS_CONSUMER_REGS>();
    const int math_wid = wid - 1;
    const int nwarp = math_wid;
    const int t0 = lid & 3, t1 = lid >> 2;
    const int cm = blockIdx.x * 64, cn = blockIdx.y * 128;
    const float alpha = *alpha_ptr;

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    #pragma unroll 1
    for (int kt = 0; kt < Kt; kt++) {
        const int s = kt % STAGES;
        const int ph = (kt / STAGES) & 1;
        mbarrier_wait_parity(&full_bulk_mbar[s], ph);
        mbarrier_wait_parity(&full_b_mbar[s], ph);
        if (N == 34816 && K == 5120) {
            compute_64x128_frag_m4n4_bpair(
                sA[s], sB[s], sSFA[s], sSFB[s], acc, nwarp, t0, t1, lid);
        } else {
            compute_64x128_frag_m4n4(
                sA[s], sB[s], sSFA[s], sSFB[s], acc, nwarp, t0, t1, lid);
        }
        if (lid == 0) mbarrier_arrive(&empty_mbar[s]);
    }

    if (M_orig == M_tiles * 64) {
        if (alpha == 1.0f) {
            epilogue_64x128_m4n4_bf16_full<true>(
                C, acc, alpha, cm, cn, nwarp, t1, lid, N);
        } else {
            epilogue_64x128_m4n4_bf16_full<false>(
                C, acc, alpha, cm, cn, nwarp, t1, lid, N);
        }
    } else {
        epilogue_64x128_m4n4_bf16_guarded(
            C, acc, alpha, cm, cn, nwarp, t1, lid, M_orig, N);
    }
}

template <int STAGES>
__global__ __launch_bounds__(288, 1)
void kern_prefill_mma_padded_bf16_m64x2_ws_tma_brep(
    const uint8_t* __restrict__ At, const uint8_t* __restrict__ Bt,
    const uint8_t* __restrict__ Ast, const uint8_t* __restrict__ Bst,
    __nv_bfloat16* __restrict__ C, const float* __restrict__ alpha_ptr,
    int M_orig, int M_tiles, int N, int K,
    CUTE_GRID_CONSTANT CUtensorMap const b_tma_desc)
{
    (void)Bt;
    const int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31;
    const int Kt = K / 64, tM = M_tiles, tN = N / 128;

    __shared__ __align__(128) uint8_t sA[2][STAGES][2048];
    __shared__ __align__(128) uint8_t sB[STAGES][4096];
    __shared__ __align__(128) uint8_t sSFA[2][STAGES][256];
    __shared__ __align__(128) uint8_t sSFB[STAGES][512];
    __shared__ __align__(64)  uint64_t full_bulk_mbar[STAGES];
    __shared__ __align__(64)  uint64_t full_b_mbar[STAGES];
    __shared__ __align__(64)  uint64_t empty_mbar[STAGES];

    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; s++) {
            mbarrier_init(&full_bulk_mbar[s], 1);
            cute::initialize_barrier(full_b_mbar[s], 1);
            mbarrier_init(&empty_mbar[s], 8);
        }
    }
    __syncthreads();
    if (tid == 0) cute::prefetch_tma_descriptor(&b_tma_desc);
    __syncthreads();

    if (wid == 0) {
        warp_reg_dealloc<PREFILL_WS_PRODUCER_REGS>();
        if (tid == 0) {
            const int m0 = blockIdx.x * 2;
            const int m1 = m0 + 1;
            const bool valid0 = m0 < M_tiles;
            const bool valid1 = m1 < M_tiles;
            #pragma unroll 1
            for (int kt = 0; kt < Kt; kt++) {
                const int s = kt % STAGES;
                const int ph = (kt / STAGES) & 1;
                if (kt >= STAGES) {
                    mbarrier_wait_parity(&empty_mbar[s], ph ^ 1);
                }

                uint32_t bytes = 4096 + 512;
                if (valid0) bytes += 2048 + 256;
                if (valid1) bytes += 2048 + 256;
                mbarrier_arrive_expect_tx(&full_bulk_mbar[s], bytes - 4096);

                if (valid0) {
                    const size_t am0 = (size_t)kt * tM + m0;
                    cp_async_bulk(sA[0][s], At + am0 * 2048, 2048, &full_bulk_mbar[s]);
                    cp_async_bulk(sSFA[0][s], Ast + am0 * 256, 256, &full_bulk_mbar[s]);
                }
                if (valid1) {
                    const size_t am1 = (size_t)kt * tM + m1;
                    cp_async_bulk(sA[1][s], At + am1 * 2048, 2048, &full_bulk_mbar[s]);
                    cp_async_bulk(sSFA[1][s], Ast + am1 * 256, 256, &full_bulk_mbar[s]);
                }

                const size_t bn = (size_t)kt * tN + blockIdx.y;
                cp_async_bulk(sSFB[s], Bst + bn * 512, 512, &full_bulk_mbar[s]);
                cute::set_barrier_transaction_bytes(full_b_mbar[s], 4096);
                cute::SM90_TMA_LOAD_2D::copy(
                    &b_tma_desc,
                    &full_b_mbar[s],
                    static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                    sB[s],
                    0,
                    static_cast<int>(bn * 128));
            }
        }
        return;
    }

    warp_reg_alloc<PREFILL_WS_CONSUMER_REGS>();
    const int math_wid = wid - 1;
    const int pair = math_wid >> 2;
    const int nwarp = math_wid & 3;
    const int t0 = lid & 3, t1 = lid >> 2;
    const int m_tile = blockIdx.x * 2 + pair;
    const bool valid_m_tile = m_tile < M_tiles;
    const int cm = m_tile * 64, cn = blockIdx.y * 128;
    const float alpha = *alpha_ptr;

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    #pragma unroll 1
    for (int kt = 0; kt < Kt; kt++) {
        const int s = kt % STAGES;
        const int ph = (kt / STAGES) & 1;
        mbarrier_wait_parity(&full_bulk_mbar[s], ph);
        mbarrier_wait_parity(&full_b_mbar[s], ph);
        if (valid_m_tile) {
            if (N == 34816 && K == 5120) {
                compute_64x128_frag_m4n4_bpair(
                    sA[pair][s], sB[s], sSFA[pair][s], sSFB[s],
                    acc, nwarp, t0, t1, lid);
            } else {
                compute_64x128_frag_m4n4(
                    sA[pair][s], sB[s], sSFA[pair][s], sSFB[s],
                    acc, nwarp, t0, t1, lid);
            }
        }
        if (lid == 0) mbarrier_arrive(&empty_mbar[s]);
    }

    if (valid_m_tile) {
        if (M_orig == M_tiles * 64) {
            if (alpha == 1.0f) {
                epilogue_64x128_m4n4_bf16_full<true>(
                    C, acc, alpha, cm, cn, nwarp, t1, lid, N);
            } else {
                epilogue_64x128_m4n4_bf16_full<false>(
                    C, acc, alpha, cm, cn, nwarp, t1, lid, N);
            }
        } else {
            epilogue_64x128_m4n4_bf16_guarded(
                C, acc, alpha, cm, cn, nwarp, t1, lid, M_orig, N);
        }
    }
}

#endif
#endif

__global__ __launch_bounds__(128, 3)
void kern_prefill_mma_padded_bf16(
    const uint8_t* __restrict__ At, const uint8_t* __restrict__ Bt,
    const uint8_t* __restrict__ Ast, const uint8_t* __restrict__ Bst,
    __nv_bfloat16* __restrict__ C, const float* __restrict__ alpha_ptr,
    int M_orig, int M_tiles, int N, int K)
{
    const int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31;
    const int wr = wid >> 1, wc = wid & 1, t0 = lid & 3, t1 = lid >> 2;
    const int cm = blockIdx.x * 128, cn = blockIdx.y * 128;
    const int Kt = K / 64, tM = M_tiles, tN = N / 128;
    const float alpha = *alpha_ptr;

    constexpr int S = 2;
    __shared__ __align__(128) uint8_t sA[S][4096];
    __shared__ __align__(128) uint8_t sB[S][4096];
    __shared__ __align__(128) uint8_t sSFA[S][512];
    __shared__ __align__(128) uint8_t sSFB[S][512];
    __shared__ __align__(64)  uint64_t mbar[S];

    float acc[4][8][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    if (tid == 0) for (int s = 0; s < S; s++) mbarrier_init(&mbar[s], 1);
    __syncthreads();

    auto load = [&](int kt, int s) {
        if (tid == 0) {
            mbarrier_arrive_expect_tx(&mbar[s], 4096 + 4096 + 512 + 512);
            size_t am = (size_t)kt * tM + blockIdx.x;
            size_t bn = (size_t)kt * tN + blockIdx.y;
            cp_async_bulk(sA[s], At + am * 4096, 4096, &mbar[s]);
            cp_async_bulk(sB[s], Bt + bn * 4096, 4096, &mbar[s]);
            cp_async_bulk(sSFA[s], Ast + am * 512, 512, &mbar[s]);
            cp_async_bulk(sSFB[s], Bst + bn * 512, 512, &mbar[s]);
        }
    };

    for (int s = 0; s < S - 1 && s < Kt; s++) load(s, s);
    for (int kt = 0; kt < Kt; kt++) {
        int s = kt % S, ph = (kt / S) & 1;
        mbarrier_wait_parity(&mbar[s], ph);
        int kn = kt + S - 1;
        if (kn < Kt) load(kn, kn % S);
        compute_128x128(sA[s], sB[s], sSFA[s], sSFB[s], acc, wr, wc, t0, t1, lid);
        __syncthreads();
    }

    epilogue_128x128_bf16_guarded(C, acc, alpha, cm, cn, wr, wc, t1, lid, M_orig, N);
}

__device__ __forceinline__ uint32_t load_b_frag_word_from_cutlass_layout(
    const uint8_t* __restrict__ B, int N, int K, int n_tile, int kt, int word)
{
    const int Kh = K / 2;
    int q_hi = word & 1;
    int tmp = word >> 1;
    int lane = tmp & 31;
    int bgroup = tmp >> 5;
    int wc = bgroup >> 3;
    int ni = bgroup & 7;
    int t1 = lane >> 2;
    int t0 = lane & 3;
    int lm = wc * 64 + ni * 8 + t1;
    int row = n_tile * 128 + lm;
    if (row >= N) return 0;
    int qw = q_hi * 4 + t0;
    return *(const uint32_t*)(B + (size_t)row * Kh + kt * 32 + qw * 4);
}

__device__ __forceinline__ uint32_t load_bsf_frag_word_from_cutlass_layout(
    const uint8_t* __restrict__ Bsf, int N, int K, int n_tile, int kg, int word)
{
    const int Ksf = K / 16;
    int nwarp = word >> 5;
    int rem = word & 31;
    int t1 = rem >> 2;
    int ni = rem & 3;
    int lm = (nwarp * 4 + ni) * 8 + t1;
    int row = n_tile * 128 + lm;
    if (row >= N) return 0;
    int r_hi = lm >> 5;
    int r_lo = lm & 31;
    int src_idx = n_tile * 128 * Ksf + kg * 512 + r_lo * 16 + r_hi * 4;
    return *(const uint32_t*)(Bsf + src_idx);
}

__device__ __forceinline__ uint32_t load_a_frag_word_from_cutlass_layout(
    const uint8_t* __restrict__ A, int M, int K, int m_tile, int kt, int word)
{
    const int Kh = K / 2;
    int slot = word & 3;
    int tmp = word >> 2;
    int lane = tmp & 31;
    int a_group = (tmp >> 5) & 1;
    int wr = tmp >> 6;
    int t1 = lane >> 2;
    int t0 = lane & 3;
    int q_hi = slot >> 1;
    int row_hi = slot & 1;
    int lm = wr * 32 + a_group * 16 + row_hi * 8 + t1;
    int row = m_tile * 64 + lm;
    if (row >= M) return 0;
    int qw = q_hi * 4 + t0;
    return *(const uint32_t*)(A + (size_t)row * Kh + kt * 32 + qw * 4);
}

__device__ __forceinline__ uint32_t load_asf_frag_word_from_cutlass_layout(
    const uint8_t* __restrict__ Asf, int M, int K, int m_tile, int kg, int word)
{
    const int Ksf = K / 16;
    int mi = word & 3;
    int scale_lane = sfa_logical_scale_lane_64(word >> 2);
    int lm = mi * 16 + scale_lane;
    int row = m_tile * 64 + lm;
    if (row >= M) return 0;
    int src_tm = row / 128;
    int row128 = row & 127;
    int r_hi = row128 / 32;
    int r_lo = row128 & 31;
    int src_idx = src_tm * 128 * Ksf + kg * 512 + r_lo * 16 + r_hi * 4;
    return *(const uint32_t*)(Asf + src_idx);
}

template <int STAGES>
__global__ __launch_bounds__(128, 4)
void kern_prefill_mma_padded_bf16_m64_direct_a(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bt,
    const uint8_t* __restrict__ Asf, const uint8_t* __restrict__ Bst,
    __nv_bfloat16* __restrict__ C, const float* __restrict__ alpha_ptr,
    int M_orig, int M_tiles, int N, int K)
{
    const int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31;
    const int nwarp = wid, t0 = lid & 3, t1 = lid >> 2;
    const int cm = blockIdx.x * 64, cn = blockIdx.y * 128;
    const int Kt = K / 64, tN = N / 128;
    const float alpha = *alpha_ptr;

    __shared__ __align__(128) uint8_t sA[2048];
    __shared__ __align__(128) uint8_t sSFA[256];
    __shared__ __align__(128) uint8_t sB[STAGES][4096];
    __shared__ __align__(128) uint8_t sSFB[STAGES][512];
    __shared__ __align__(64)  uint64_t mbar[STAGES];

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    if (tid == 0) for (int s = 0; s < STAGES; s++) mbarrier_init(&mbar[s], 1);
    __syncthreads();

    auto load_b = [&](int kt, int s) {
        if (tid == 0) {
            mbarrier_arrive_expect_tx(&mbar[s], 4096 + 512);
            size_t bn = (size_t)kt * tN + blockIdx.y;
            cp_async_bulk(sB[s], Bt + bn * 4096, 4096, &mbar[s]);
            cp_async_bulk(sSFB[s], Bst + bn * 512, 512, &mbar[s]);
        }
    };

    for (int s = 0; s < STAGES - 1 && s < Kt; s++) load_b(s, s);
    for (int kt = 0; kt < Kt; kt++) {
        int s = kt % STAGES, ph = (kt / STAGES) & 1;
        mbarrier_wait_parity(&mbar[s], ph);
        int kn = kt + STAGES - 1;
        if (kn < Kt) load_b(kn, kn % STAGES);

        #pragma unroll
        for (int w = tid; w < 512; w += 128) {
            ((uint32_t*)sA)[w] = load_a_frag_word_from_cutlass_layout(
                A, M_orig, K, blockIdx.x, kt, w);
        }
        for (int w = tid; w < 64; w += 128) {
            ((uint32_t*)sSFA)[w] = load_asf_frag_word_from_cutlass_layout(
                Asf, M_orig, K, blockIdx.x, kt, w);
        }
        __syncthreads();
        if (N == 34816 && K == 5120) {
            compute_64x128_frag_m4n4_bpair(
                sA, sB[s], sSFA, sSFB[s], acc, nwarp, t0, t1, lid);
        } else {
            compute_64x128_frag_m4n4(
                sA, sB[s], sSFA, sSFB[s], acc, nwarp, t0, t1, lid);
        }
        __syncthreads();
    }

    epilogue_64x128_m4n4_bf16_guarded(C, acc, alpha, cm, cn, nwarp, t1, lid, M_orig, N);
}

__global__ __launch_bounds__(128, 4)
void kern_prefill_mma_padded_bf16_m64_direct_b(
    const uint8_t* __restrict__ At, const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ Ast, const uint8_t* __restrict__ Bsf,
    __nv_bfloat16* __restrict__ C, const float* __restrict__ alpha_ptr,
    int M_orig, int M_tiles, int N, int K)
{
    const int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31;
    const int nwarp = wid, t0 = lid & 3, t1 = lid >> 2;
    const int cm = blockIdx.x * 64, cn = blockIdx.y * 128;
    const int Kt = K / 64, tM = M_tiles;
    const float alpha = *alpha_ptr;

    __shared__ __align__(128) uint8_t sA[2048];
    __shared__ __align__(128) uint8_t sB[4096];
    __shared__ __align__(128) uint8_t sSFA[256];
    __shared__ __align__(128) uint8_t sSFB[512];

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    for (int kt = 0; kt < Kt; kt++) {
        size_t am = (size_t)kt * tM + blockIdx.x;

        for (int w = tid; w < 512; w += blockDim.x) {
            ((uint32_t*)sA)[w] = ((const uint32_t*)(At + am * 2048))[w];
        }
        for (int w = tid; w < 64; w += blockDim.x) {
            ((uint32_t*)sSFA)[w] = ((const uint32_t*)(Ast + am * 256))[w];
        }
        for (int w = tid; w < 1024; w += blockDim.x) {
            ((uint32_t*)sB)[w] =
                load_b_frag_word_from_cutlass_layout(B, N, K, blockIdx.y, kt, w);
        }
        for (int w = tid; w < 128; w += blockDim.x) {
            ((uint32_t*)sSFB)[w] =
                load_bsf_frag_word_from_cutlass_layout(Bsf, N, K, blockIdx.y, kt, w);
        }

        __syncthreads();
        compute_64x128_frag_m4n4(sA, sB, sSFA, sSFB, acc, nwarp, t0, t1, lid);
        __syncthreads();
    }

    epilogue_64x128_m4n4_bf16_guarded(C, acc, alpha, cm, cn, nwarp, t1, lid, M_orig, N);
}

template <int N_CONST, int K_CONST>
__global__ __launch_bounds__(128, 3)
void kern_prefill_mma_padded_bf16_static(
    const uint8_t* __restrict__ At, const uint8_t* __restrict__ Bt,
    const uint8_t* __restrict__ Ast, const uint8_t* __restrict__ Bst,
    __nv_bfloat16* __restrict__ C, const float* __restrict__ alpha_ptr,
    int M_orig, int M_tiles)
{
    const int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31;
    const int wr = wid >> 1, wc = wid & 1, t0 = lid & 3, t1 = lid >> 2;
    const int cm = blockIdx.x * 128, cn = blockIdx.y * 128;
    constexpr int Kt = K_CONST / 64;
    constexpr int tN = N_CONST / 128;
    const int tM = M_tiles;
    const float alpha = *alpha_ptr;

    constexpr int S = 2;
    __shared__ __align__(128) uint8_t sA[S][4096];
    __shared__ __align__(128) uint8_t sB[S][4096];
    __shared__ __align__(128) uint8_t sSFA[S][512];
    __shared__ __align__(128) uint8_t sSFB[S][512];
    __shared__ __align__(64)  uint64_t mbar[S];

    float acc[4][8][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    if (tid == 0) for (int s = 0; s < S; s++) mbarrier_init(&mbar[s], 1);
    __syncthreads();

    auto load = [&](int kt, int s) {
        if (tid == 0) {
            mbarrier_arrive_expect_tx(&mbar[s], 4096 + 4096 + 512 + 512);
            size_t am = (size_t)kt * tM + blockIdx.x;
            size_t bn = (size_t)kt * tN + blockIdx.y;
            cp_async_bulk(sA[s], At + am * 4096, 4096, &mbar[s]);
            cp_async_bulk(sB[s], Bt + bn * 4096, 4096, &mbar[s]);
            cp_async_bulk(sSFA[s], Ast + am * 512, 512, &mbar[s]);
            cp_async_bulk(sSFB[s], Bst + bn * 512, 512, &mbar[s]);
        }
    };

    for (int s = 0; s < S - 1 && s < Kt; s++) load(s, s);
    #pragma unroll 1
    for (int kt = 0; kt < Kt; kt++) {
        int s = kt % S, ph = (kt / S) & 1;
        mbarrier_wait_parity(&mbar[s], ph);
        int kn = kt + S - 1;
        if (kn < Kt) load(kn, kn % S);
        compute_128x128(sA[s], sB[s], sSFA[s], sSFB[s], acc, wr, wc, t0, t1, lid);
        __syncthreads();
    }

    epilogue_128x128_bf16_guarded(
        C, acc, alpha, cm, cn, wr, wc, t1, lid, M_orig, N_CONST);
}

#if USE_PREFILL_TMA_B
#if USE_PREFILL_TMA_BREP
static bool make_prefill_brep_tma_desc(
    CUtensorMap* desc, const unsigned char* B_rep, int N, int K)
{
    static bool driver_initialized = false;
    if (!driver_initialized) {
        CUresult init_status = cuInit(0);
        if (init_status != CUDA_SUCCESS) {
            printf("[task39_prefill_tma_b] cuInit failed: %d\n", (int)init_status);
            return false;
        }
        driver_initialized = true;
    }

    const int N_pad = round_up_128(N);
    const int Kt = K / 64;
    const int tN = N_pad / 128;
    const cuuint64_t global_dims[2] = {
        32u,
        static_cast<cuuint64_t>(Kt * tN * 128),
    };
    const cuuint64_t global_strides[1] = {32u};
    const cuuint32_t box_dims[2] = {32u, 128u};
    const cuuint32_t element_strides[2] = {1u, 1u};

    CUresult status = cuTensorMapEncodeTiled(
        desc,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,
        const_cast<unsigned char*>(B_rep),
        global_dims,
        global_strides,
        box_dims,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        printf("[task39_prefill_tma_b] cuTensorMapEncodeTiled failed: %d N=%d K=%d B_rep=%p\n",
               (int)status, N, K, B_rep);
        return false;
    }
    return true;
}
#endif

static bool make_prefill_brow_tma_desc(
    CUtensorMap* desc, const unsigned char* B, int N, int K)
{
    static bool driver_initialized = false;
    if (!driver_initialized) {
        CUresult init_status = cuInit(0);
        if (init_status != CUDA_SUCCESS) {
            printf("[task39_prefill_tma_direct_b] cuInit failed: %d\n", (int)init_status);
            return false;
        }
        driver_initialized = true;
    }

    const cuuint64_t global_dims[2] = {
        static_cast<cuuint64_t>(K / 2),
        static_cast<cuuint64_t>(N),
    };
    const cuuint64_t global_strides[1] = {
        static_cast<cuuint64_t>(K / 2),
    };
    const cuuint32_t box_dims[2] = {32u, 128u};
    const cuuint32_t element_strides[2] = {1u, 1u};

    CUresult status = cuTensorMapEncodeTiled(
        desc,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,
        const_cast<unsigned char*>(B),
        global_dims,
        global_strides,
        box_dims,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        printf("[task39_prefill_tma_direct_b] cuTensorMapEncodeTiled failed: %d N=%d K=%d B=%p\n",
               (int)status, N, K, B);
        return false;
    }
    return true;
}

static bool can_launch_prefill_coop_arepack(int total_blocks, int block_threads) {
    static int initialized = 0;
    static int coop_supported = 0;
    static int coop_capacity_blocks = 0;

    if (!initialized) {
        int dev = -1;
        int sm_count = 0;
        int blocks_per_sm = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&coop_supported, cudaDevAttrCooperativeLaunch, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks_per_sm,
                kern_prefill_mma_padded_bf16_m64_coop_arepack_tma_direct_b_k17408_n5120_mtiles5<3>,
                block_threads,
                0) != cudaSuccess) {
            coop_supported = 0;
            coop_capacity_blocks = 0;
        } else {
            coop_capacity_blocks = blocks_per_sm * sm_count;
        }
        initialized = 1;
    }

    return coop_supported && coop_capacity_blocks >= total_blocks;
}
#endif

extern "C" int prefill_mma_padded_is_supported(int M, int N, int K) {
    return is_task39_cutlass_win_shape(M, N, K);
}

extern "C" int prefill_mma_padded_round_m(int M) {
    return round_up_64(M);
}

#if PREFILL_ENABLE_PERSISTENT_CTA_ROUTE
static __host__ int prefill_persistent_cta_count() {
#if PREFILL_PERSISTENT_CTA_COUNT > 0
    return PREFILL_PERSISTENT_CTA_COUNT;
#else
    static int cached_device = -2;
    static int cached_sms = 0;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return PREFILL_PERSISTENT_CTAS_PER_SM;
    if (dev != cached_device) {
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cached_sms = sms > 0 ? sms : 1;
        cached_device = dev;
    }
    constexpr int ctas_per_sm =
        (PREFILL_PERSISTENT_CTAS_PER_SM > 0) ? PREFILL_PERSISTENT_CTAS_PER_SM : 1;
    return cached_sms * ctas_per_sm;
#endif
}
#endif

#if PREFILL_ENABLE_CLUSTER_ROUTE || PREFILL_ENABLE_PERSIST_L2
#if PREFILL_ENABLE_PERSIST_L2
static __host__ size_t prefill_min_size(size_t a, size_t b) {
    return a < b ? a : b;
}

static __host__ size_t prefill_bpair_persist_window_bytes(size_t requested) {
    static int cached_device = -2;
    static size_t cached_max_window = 0;
    static size_t cached_max_persist = 0;

    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    if (dev != cached_device) {
        int max_window = 0;
        int max_persist = 0;
        cudaDeviceGetAttribute(&max_window, cudaDevAttrMaxAccessPolicyWindowSize, dev);
        cudaDeviceGetAttribute(&max_persist, cudaDevAttrMaxPersistingL2CacheSize, dev);
        cached_max_window = max_window > 0 ? (size_t)max_window : 0;
        cached_max_persist = max_persist > 0 ? (size_t)max_persist : 0;
        if (cached_max_persist > 0) {
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, cached_max_persist);
        }
        cached_device = dev;
    }

    size_t window = requested;
    if (cached_max_window > 0) window = prefill_min_size(window, cached_max_window);
#if PREFILL_PERSIST_MAX_BYTES > 0
    window = prefill_min_size(window, (size_t)PREFILL_PERSIST_MAX_BYTES);
#endif
    return window;
}
#endif

template <int STAGES>
static __host__ cudaError_t launch_prefill_m64x2_bpair_ex(
    dim3 grid, dim3 block, cudaStream_t s,
    const uint8_t* A_rep, const uint8_t* B_rep,
    const uint8_t* A_sf_rep, const uint8_t* B_sf_rep,
    __nv_bfloat16* C, const float* alpha,
    int M, int M_tiles, int N, int K)
{
    cudaLaunchAttribute attrs[3];
    unsigned int attr_count = 0;

#if PREFILL_ENABLE_CLUSTER_ROUTE
    constexpr unsigned int cluster_x = (PREFILL_CLUSTER_X > 0) ? PREFILL_CLUSTER_X : 1;
    if (cluster_x > 1 && (grid.x % cluster_x) == 0) {
        attrs[attr_count].id = cudaLaunchAttributeClusterDimension;
        attrs[attr_count].val.clusterDim.x = cluster_x;
        attrs[attr_count].val.clusterDim.y = 1;
        attrs[attr_count].val.clusterDim.z = 1;
        attr_count++;

        attrs[attr_count].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
        attrs[attr_count].val.clusterSchedulingPolicyPreference =
            (cudaClusterSchedulingPolicy)PREFILL_CLUSTER_SCHED_POLICY;
        attr_count++;
    }
#endif

#if PREFILL_ENABLE_PERSIST_L2
    const int Kt = K / 64;
    const int tN = N / 128;
    const size_t b_rep_bytes = (size_t)Kt * (size_t)tN * 4096u;
    const size_t persist_bytes = prefill_bpair_persist_window_bytes(b_rep_bytes);
    if (persist_bytes > 0) {
        float hit_ratio = ((float)PREFILL_PERSIST_HIT_RATIO_X100) * 0.01f;
        if (hit_ratio <= 0.0f) hit_ratio = 0.01f;
        if (hit_ratio > 1.0f) hit_ratio = 1.0f;
        attrs[attr_count].id = cudaLaunchAttributeAccessPolicyWindow;
        attrs[attr_count].val.accessPolicyWindow.base_ptr = (void*)B_rep;
        attrs[attr_count].val.accessPolicyWindow.num_bytes = persist_bytes;
        attrs[attr_count].val.accessPolicyWindow.hitRatio = hit_ratio;
        attrs[attr_count].val.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attrs[attr_count].val.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        attr_count++;
    }
#endif

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = s;
    cfg.attrs = attr_count ? attrs : nullptr;
    cfg.numAttrs = attr_count;

    return cudaLaunchKernelEx(
        &cfg,
        kern_prefill_mma_padded_bf16_m64x2_bpair<STAGES>,
        A_rep, B_rep, A_sf_rep, B_sf_rep, C, alpha, M, M_tiles, N, K);
}
#endif

// ================================================================
// C API: GEMM entry point
// ================================================================

extern "C" void prefill_mma_padded_gemm(
    int M, int N, int K,
    const void* A,           // [M, K/2] uint8 row-major
    const void* B_rep,       // [tiled] pre-repacked B data
    const void* A_sf,        // swizzled scale layout, padded to ceil(M/128)
    const void* B_sf_rep,    // [tiled] pre-repacked B scale factors
    const void* alpha,       // float32 device pointer
    void* C,                 // [M, N] bf16 output
    void* A_rep_buf,         // workspace [ceil(M/128)*128 * K/2]
    void* A_sf_rep_buf,      // workspace [ceil(M/128)*128 * K/16]
    unsigned long long stream)
{
    if (!prefill_mma_padded_is_supported(M, N, K)) return;

    cudaStream_t s = (cudaStream_t)stream;
    int M_pad = round_up_64(M);
    int Kt = K / 64;
    int th = PREFILL_REPACK_THREADS;

    if (M == M_pad) {
        int data_threads = Kt * M_pad * 8;
        dim3 repack_grid((data_threads + th - 1) / th);
        if (K == 5120 && M == 1024) {
            repack_data_sf_64_frag_full_static<1024, 5120>
                <<<repack_grid, th, 0, s>>>(
                    (const uint8_t*)A, (uint8_t*)A_rep_buf,
                    (const uint8_t*)A_sf, (uint8_t*)A_sf_rep_buf);
        } else if (K == 5120 && M == 2048) {
            repack_data_sf_64_frag_full_static<2048, 5120>
                <<<repack_grid, th, 0, s>>>(
                    (const uint8_t*)A, (uint8_t*)A_rep_buf,
                    (const uint8_t*)A_sf, (uint8_t*)A_sf_rep_buf);
        } else if (K == 5120 && M == 4096) {
            repack_data_sf_64_frag_full_static<4096, 5120>
                <<<repack_grid, th, 0, s>>>(
                    (const uint8_t*)A, (uint8_t*)A_rep_buf,
                    (const uint8_t*)A_sf, (uint8_t*)A_sf_rep_buf);
        } else if (K == 5120 && M == 8192) {
            repack_data_sf_64_frag_full_static<8192, 5120>
                <<<repack_grid, th, 0, s>>>(
                    (const uint8_t*)A, (uint8_t*)A_rep_buf,
                    (const uint8_t*)A_sf, (uint8_t*)A_sf_rep_buf);
        } else if (K == 17408 && M == 1024) {
            repack_data_sf_64_frag_full_static<1024, 17408>
                <<<repack_grid, th, 0, s>>>(
                    (const uint8_t*)A, (uint8_t*)A_rep_buf,
                    (const uint8_t*)A_sf, (uint8_t*)A_sf_rep_buf);
        } else if (K == 17408 && M == 2048) {
            repack_data_sf_64_frag_full_static<2048, 17408>
                <<<repack_grid, th, 0, s>>>(
                    (const uint8_t*)A, (uint8_t*)A_rep_buf,
                    (const uint8_t*)A_sf, (uint8_t*)A_sf_rep_buf);
        } else if (K == 17408 && M == 4096) {
            repack_data_sf_64_frag_full_static<4096, 17408>
                <<<repack_grid, th, 0, s>>>(
                    (const uint8_t*)A, (uint8_t*)A_rep_buf,
                    (const uint8_t*)A_sf, (uint8_t*)A_sf_rep_buf);
        } else if (K == 17408 && M == 8192) {
            repack_data_sf_64_frag_full_static<8192, 17408>
                <<<repack_grid, th, 0, s>>>(
                    (const uint8_t*)A, (uint8_t*)A_rep_buf,
                    (const uint8_t*)A_sf, (uint8_t*)A_sf_rep_buf);
        } else {
            repack_data_sf_64_frag_full<<<repack_grid, th, 0, s>>>(
                (const uint8_t*)A, (uint8_t*)A_rep_buf,
                (const uint8_t*)A_sf, (uint8_t*)A_sf_rep_buf,
                M_pad, K);
        }
    } else if (K == 17408 && M_pad == 320) {
        constexpr int repack_th = 512;
        dim3 repack_grid((M_pad * 8 + repack_th - 1) / repack_th, Kt);
        repack_data_sf_64_frag_padded_k17408_m320_static<<<repack_grid, repack_th, 0, s>>>(
            (const uint8_t*)A, (uint8_t*)A_rep_buf,
            (const uint8_t*)A_sf, (uint8_t*)A_sf_rep_buf,
            M);
    } else {
        int data_threads = Kt * M_pad * 8;
        repack_data_sf_64_frag_padded<<<(data_threads + th - 1) / th, th, 0, s>>>(
            (const uint8_t*)A, (uint8_t*)A_rep_buf,
            (const uint8_t*)A_sf, (uint8_t*)A_sf_rep_buf,
            M, M_pad, K);
    }

    int M_tiles = M_pad / 64;
#if USE_PREFILL_TMA_B && USE_PREFILL_TMA_BREP && USE_PREFILL_WS_TMA_BREP && PREFILL_ENABLE_WS_TMA_BREP_ROUTE
    CUtensorMap b_tma_desc;
    if (make_prefill_brep_tma_desc(&b_tma_desc, (const unsigned char*)B_rep, N, K)) {
        if (M_tiles >= 2) {
            dim3 grid((M_tiles + 1) / 2, N / 128);
            dim3 block(288);
            kern_prefill_mma_padded_bf16_m64x2_ws_tma_brep<3><<<grid, block, 0, s>>>(
                (const uint8_t*)A_rep_buf, (const uint8_t*)B_rep,
                (const uint8_t*)A_sf_rep_buf, (const uint8_t*)B_sf_rep,
                (__nv_bfloat16*)C, (const float*)alpha,
                M, M_tiles, N, K, b_tma_desc);
        } else {
            dim3 grid(M_tiles, N / 128);
            dim3 block(160);
            kern_prefill_mma_padded_bf16_m64_ws_tma_brep<3><<<grid, block, 0, s>>>(
                (const uint8_t*)A_rep_buf, (const uint8_t*)B_rep,
                (const uint8_t*)A_sf_rep_buf, (const uint8_t*)B_sf_rep,
                (__nv_bfloat16*)C, (const float*)alpha,
                M, M_tiles, N, K, b_tma_desc);
        }
        return;
    }
#endif
#if PREFILL_ENABLE_M64X2_WIDE_N_ROUTE
    if (M_tiles >= 2 && N == 34816 && K == 5120) {
        dim3 grid((M_tiles + 1) / 2, N / 128);
        dim3 block(128);
        kern_prefill_mma_padded_bf16_m64x2_wide_n<3><<<grid, block, 0, s>>>(
            (const uint8_t*)A_rep_buf, (const uint8_t*)B_rep,
            (const uint8_t*)A_sf_rep_buf, (const uint8_t*)B_sf_rep,
            (__nv_bfloat16*)C, (const float*)alpha,
            M, M_tiles, N, K);
        return;
    }
#endif
#if PREFILL_ENABLE_M64X2_MIXED3_ROUTE
    if (M_tiles >= 2 && N == 34816 && K == 5120) {
        dim3 grid((M_tiles + 1) / 2, N / 128);
        dim3 block(192);
        kern_prefill_mma_padded_bf16_m64x2_mixed3_bpair<3><<<grid, block, 0, s>>>(
            (const uint8_t*)A_rep_buf, (const uint8_t*)B_rep,
            (const uint8_t*)A_sf_rep_buf, (const uint8_t*)B_sf_rep,
            (__nv_bfloat16*)C, (const float*)alpha,
            M, M_tiles, N, K);
        return;
    }
#endif
#if PREFILL_ENABLE_M64_N2_AREUSE_ROUTE
    if (N == 34816 && K == 5120) {
        dim3 grid(M_tiles, (N / 128 + 1) / 2);
        dim3 block(256);
        kern_prefill_mma_padded_bf16_m64_n2_areuse_bpair<3><<<grid, block, 0, s>>>(
            (const uint8_t*)A_rep_buf, (const uint8_t*)B_rep,
            (const uint8_t*)A_sf_rep_buf, (const uint8_t*)B_sf_rep,
            (__nv_bfloat16*)C, (const float*)alpha,
            M, M_tiles, N, K);
        return;
    }
#endif
    if (N == 34816 && K == 5120) {
        dim3 grid((M_tiles + 1) / 2, N / 128);
        dim3 block(256);
#if PREFILL_ENABLE_PERSISTENT_CTA_ROUTE
        const int total_tiles = ((M_tiles + 1) / 2) * (N / 128);
        int persistent_ctas = prefill_persistent_cta_count();
        if (persistent_ctas < 1) persistent_ctas = 1;
        if (persistent_ctas > total_tiles) persistent_ctas = total_tiles;
        kern_prefill_mma_padded_bf16_m64x2_bpair_persistent<3>
            <<<dim3(persistent_ctas), block, 0, s>>>(
                (const uint8_t*)A_rep_buf, (const uint8_t*)B_rep,
                (const uint8_t*)A_sf_rep_buf, (const uint8_t*)B_sf_rep,
                (__nv_bfloat16*)C, (const float*)alpha,
                M, M_tiles, N, K);
#elif PREFILL_ENABLE_CLUSTER_ROUTE || PREFILL_ENABLE_PERSIST_L2
        cudaError_t launch_status = launch_prefill_m64x2_bpair_ex<3>(
            grid, block, s,
            (const uint8_t*)A_rep_buf, (const uint8_t*)B_rep,
            (const uint8_t*)A_sf_rep_buf, (const uint8_t*)B_sf_rep,
            (__nv_bfloat16*)C, (const float*)alpha,
            M, M_tiles, N, K);
        (void)launch_status;
#else
        kern_prefill_mma_padded_bf16_m64x2_bpair<3><<<grid, block, 0, s>>>(
            (const uint8_t*)A_rep_buf, (const uint8_t*)B_rep,
            (const uint8_t*)A_sf_rep_buf, (const uint8_t*)B_sf_rep,
            (__nv_bfloat16*)C, (const float*)alpha,
            M, M_tiles, N, K);
#endif
        return;
    }
    if (N == 5120 && K == 17408) {
        dim3 grid((M_tiles + 1) / 2, N / 128);
        dim3 block(256);
        kern_prefill_mma_padded_bf16_m64x2<3><<<grid, block, 0, s>>>(
            (const uint8_t*)A_rep_buf, (const uint8_t*)B_rep,
            (const uint8_t*)A_sf_rep_buf, (const uint8_t*)B_sf_rep,
            (__nv_bfloat16*)C, (const float*)alpha,
            M, M_tiles, N, K);
    } else if (M <= 2048 || (N == 34816 && K == 5120)) {
        dim3 grid((M_tiles + 1) / 2, N / 128);
        dim3 block(256);
        kern_prefill_mma_padded_bf16_m64x2<3><<<grid, block, 0, s>>>(
            (const uint8_t*)A_rep_buf, (const uint8_t*)B_rep,
            (const uint8_t*)A_sf_rep_buf, (const uint8_t*)B_sf_rep,
            (__nv_bfloat16*)C, (const float*)alpha,
            M, M_tiles, N, K);
    } else {
        dim3 grid(M_tiles, N / 128);
        dim3 block(128);
        kern_prefill_mma_padded_bf16_m64<3><<<grid, block, 0, s>>>(
            (const uint8_t*)A_rep_buf, (const uint8_t*)B_rep,
            (const uint8_t*)A_sf_rep_buf, (const uint8_t*)B_sf_rep,
            (__nv_bfloat16*)C, (const float*)alpha,
            M, M_tiles, N, K);
    }
}

extern "C" void prefill_mma_padded_gemm_direct_a_layout(
    int M, int N, int K,
    const void* A,           // [M, K/2] uint8 row-major
    const void* B_rep,       // [tiled] pre-repacked B data
    const void* A_sf,        // swizzled scale layout, padded to ceil(M/128)
    const void* B_sf_rep,    // [tiled] pre-repacked B scale factors
    const void* alpha,       // float32 device pointer
    void* C,                 // [M, N] bf16 output
    void* A_rep_buf,         // unused; kept to preserve C ABI shape
    void* A_sf_rep_buf,      // unused; kept to preserve C ABI shape
    unsigned long long stream)
{
    if (!prefill_mma_padded_is_supported(M, N, K)) return;
    if (A_rep_buf == nullptr || A_sf_rep_buf == nullptr) return;

    cudaStream_t s = (cudaStream_t)stream;
    int M_pad = round_up_64(M);
    int M_tiles = M_pad / 64;
    dim3 grid(M_tiles, N / 128);
    dim3 block(128);
    kern_prefill_mma_padded_bf16_m64_direct_a<2><<<grid, block, 0, s>>>(
        (const uint8_t*)A, (const uint8_t*)B_rep,
        (const uint8_t*)A_sf, (const uint8_t*)B_sf_rep,
        (__nv_bfloat16*)C, (const float*)alpha,
        M, M_tiles, N, K);
}

extern "C" void prefill_mma_padded_quant_bf16_gemm(
    int M, int N, int K,
    const void* X_bf16,      // [M, K] bf16 activation
    const void* B_rep,       // [tiled] pre-repacked B data
    const void* input_scale, // float32 device pointer for scaled_fp4_quant
    const void* B_sf_rep,    // [tiled] pre-repacked B scale factors
    const void* alpha,       // float32 device pointer for GEMM epilogue
    void* C,                 // [M, N] bf16 output
    void* A_rep_buf,         // workspace [ceil(M/64)*64 * K/2]
    void* A_sf_rep_buf,      // workspace [ceil(M/64)*64 * K/16]
    unsigned long long stream)
{
    if (!prefill_mma_padded_is_supported(M, N, K)) return;

    cudaStream_t s = (cudaStream_t)stream;
    int M_pad = round_up_64(M);
    int M_tiles = M_pad / 64;
    int packed_cols = K / 16;

    dim3 q_block(512);
    dim3 q_grid(M_pad < 128 ? M_pad : 128,
                (packed_cols + q_block.x - 1) / q_block.x);
    if (K == 17408 && M_pad == 320) {
        quant_bf16_to_frag64_k17408_padded_static<320><<<q_grid, q_block, 0, s>>>(
            (const __nv_bfloat16*)X_bf16,
            (const float*)input_scale,
            (uint8_t*)A_rep_buf,
            (uint8_t*)A_sf_rep_buf,
            M);
    } else {
        quant_bf16_to_frag64_padded<<<q_grid, q_block, 0, s>>>(
            (const __nv_bfloat16*)X_bf16,
            (const float*)input_scale,
            (uint8_t*)A_rep_buf,
            (uint8_t*)A_sf_rep_buf,
            M,
            M_pad,
            K);
    }

    dim3 grid(M_tiles, N / 128);
    dim3 block(128);
    if (N == 34816 && K == 5120) {
        dim3 grid2((M_tiles + 1) / 2, N / 128);
        dim3 block2(256);
        kern_prefill_mma_padded_bf16_m64x2_bpair<3><<<grid2, block2, 0, s>>>(
            (const uint8_t*)A_rep_buf, (const uint8_t*)B_rep,
            (const uint8_t*)A_sf_rep_buf, (const uint8_t*)B_sf_rep,
            (__nv_bfloat16*)C, (const float*)alpha,
            M, M_tiles, N, K);
        return;
    }
#if USE_PREFILL_TMA_B && USE_PREFILL_TMA_BREP
    CUtensorMap b_tma_desc;
    if (make_prefill_brep_tma_desc(&b_tma_desc, (const unsigned char*)B_rep, N, K)) {
        kern_prefill_mma_padded_bf16_m64_tma_b<2><<<grid, block, 0, s>>>(
            (const uint8_t*)A_rep_buf, (const uint8_t*)B_rep,
            (const uint8_t*)A_sf_rep_buf, (const uint8_t*)B_sf_rep,
            (__nv_bfloat16*)C, (const float*)alpha,
            M, M_tiles, N, K, b_tma_desc);
        return;
    }
#endif
    kern_prefill_mma_padded_bf16_m64<2><<<grid, block, 0, s>>>(
        (const uint8_t*)A_rep_buf, (const uint8_t*)B_rep,
        (const uint8_t*)A_sf_rep_buf, (const uint8_t*)B_sf_rep,
        (__nv_bfloat16*)C, (const float*)alpha,
        M, M_tiles, N, K);
}

extern "C" void prefill_mma_padded_gemm_direct_layout(
    int M, int N, int K,
    const void* A,           // [M, K/2] uint8 row-major
    const void* B,           // [N, K/2] uint8 vLLM/FlashInfer CUTLASS layout
    const void* A_sf,        // swizzled scale layout, padded to ceil(M/128)
    const void* B_sf,        // swizzled scale layout, padded to ceil(N/128)
    const void* alpha,       // float32 device pointer
    void* C,                 // [M, N] bf16 output
    void* A_rep_buf,         // workspace [ceil(M/64)*64 * K/2]
    void* A_sf_rep_buf,      // workspace [ceil(M/64)*64 * K/16]
    unsigned long long stream)
{
    if (!prefill_mma_padded_is_supported(M, N, K)) return;

    cudaStream_t s = (cudaStream_t)stream;
#if PREFILL_ENABLE_CUTLASS_WS_ROUTE
    if (task39_cutlass_ws::run_dispatch(
            M, N, K, A, B, A_sf, B_sf, alpha, C, s)) {
        return;
    }
#endif
    int M_pad = round_up_64(M);
    int Kt = K / 64;
    int th = PREFILL_REPACK_THREADS;

    if (K == 17408 && M_pad == 320) {
        constexpr int repack_th = 512;
        dim3 repack_grid((M_pad * 8 + repack_th - 1) / repack_th, Kt);
        repack_data_sf_64_frag_padded_k17408_m320_static<<<repack_grid, repack_th, 0, s>>>(
            (const uint8_t*)A, (uint8_t*)A_rep_buf,
            (const uint8_t*)A_sf, (uint8_t*)A_sf_rep_buf,
            M);
    } else {
        int data_threads = Kt * M_pad * 8;
        repack_data_sf_64_frag_padded<<<(data_threads + th - 1) / th, th, 0, s>>>(
            (const uint8_t*)A, (uint8_t*)A_rep_buf,
            (const uint8_t*)A_sf, (uint8_t*)A_sf_rep_buf,
            M, M_pad, K);
    }

    int M_tiles = M_pad / 64;
    dim3 grid(M_tiles, N / 128);
    dim3 block(128);
#if USE_PREFILL_TMA_B
    CUtensorMap b_tma_desc;
    if (make_prefill_brow_tma_desc(&b_tma_desc, (const unsigned char*)B, N, K)) {
        if (N == 5120 && K == 17408 && M_tiles == 5) {
#if USE_PREFILL_TMA_PAIR_M
            dim3 pair_grid((M_tiles + 1) / 2, N / 128);
            dim3 pair_block(256);
            kern_prefill_mma_padded_bf16_m64x2_tma_direct_b_k17408_n5120_mtiles5<2>
                <<<pair_grid, pair_block, 0, s>>>(
                    (const uint8_t*)A_rep_buf, (const uint8_t*)B,
                    (const uint8_t*)A_sf_rep_buf, (const uint8_t*)B_sf,
                    (__nv_bfloat16*)C, (const float*)alpha,
                    M, b_tma_desc);
#else
            kern_prefill_mma_padded_bf16_m64_tma_direct_b_k17408_n5120_mtiles5<3>
                <<<grid, block, 0, s>>>(
                    (const uint8_t*)A_rep_buf, (const uint8_t*)B,
                    (const uint8_t*)A_sf_rep_buf, (const uint8_t*)B_sf,
                    (__nv_bfloat16*)C, (const float*)alpha,
                    M, b_tma_desc);
#endif
        } else {
            kern_prefill_mma_padded_bf16_m64_tma_direct_b<2><<<grid, block, 0, s>>>(
                (const uint8_t*)A_rep_buf, (const uint8_t*)B,
                (const uint8_t*)A_sf_rep_buf, (const uint8_t*)B_sf,
                (__nv_bfloat16*)C, (const float*)alpha,
                M, M_tiles, N, K, b_tma_desc);
        }
        return;
    }
#endif
    kern_prefill_mma_padded_bf16_m64_direct_b<<<grid, block, 0, s>>>(
        (const uint8_t*)A_rep_buf, (const uint8_t*)B,
        (const uint8_t*)A_sf_rep_buf, (const uint8_t*)B_sf,
        (__nv_bfloat16*)C, (const float*)alpha,
        M, M_tiles, N, K);
}

extern "C" void prefill_mma_padded_gemm_direct_layout_fused_a_repack(
    int M, int N, int K,
    const void* A,           // [M, K/2] uint8 row-major
    const void* B,           // [N, K/2] uint8 vLLM/FlashInfer CUTLASS layout
    const void* A_sf,        // swizzled scale layout, padded to ceil(M/128)
    const void* B_sf,        // swizzled scale layout, padded to ceil(N/128)
    const void* alpha,       // float32 device pointer
    void* C,                 // [M, N] bf16 output
    void* A_rep_buf,         // unused; kept to preserve C ABI shape
    void* A_sf_rep_buf,      // unused; kept to preserve C ABI shape
    unsigned long long stream)
{
    (void)A_rep_buf;
    (void)A_sf_rep_buf;
    if (!prefill_mma_padded_is_supported(M, N, K)) return;

    cudaStream_t s = (cudaStream_t)stream;
#if PREFILL_ENABLE_CUTLASS_WS_ROUTE
    if (task39_cutlass_ws::run_dispatch(
            M, N, K, A, B, A_sf, B_sf, alpha, C, s)) {
        return;
    }
#endif
    int M_pad = round_up_64(M);
    int M_tiles = M_pad / 64;
    dim3 grid(M_tiles, N / 128);
    dim3 block(128);
#if USE_PREFILL_TMA_B
    CUtensorMap b_tma_desc;
    if (N == 5120 && K == 17408 && M_tiles == 5 &&
        make_prefill_brow_tma_desc(&b_tma_desc, (const unsigned char*)B, N, K)) {
        const int total_blocks = grid.x * grid.y * grid.z;
        if (can_launch_prefill_coop_arepack(total_blocks, block.x)) {
            const uint8_t* A_p = (const uint8_t*)A;
            const uint8_t* B_p = (const uint8_t*)B;
            const uint8_t* A_sf_p = (const uint8_t*)A_sf;
            const uint8_t* B_sf_p = (const uint8_t*)B_sf;
            __nv_bfloat16* C_p = (__nv_bfloat16*)C;
            const float* alpha_p = (const float*)alpha;
            uint8_t* A_rep_p = (uint8_t*)A_rep_buf;
            uint8_t* A_sf_rep_p = (uint8_t*)A_sf_rep_buf;
            void* args[] = {
                &A_p,
                &B_p,
                &A_sf_p,
                &B_sf_p,
                &C_p,
                &alpha_p,
                &M,
                &A_rep_p,
                &A_sf_rep_p,
                &b_tma_desc,
            };
            cudaLaunchCooperativeKernel(
                (void*)kern_prefill_mma_padded_bf16_m64_coop_arepack_tma_direct_b_k17408_n5120_mtiles5<3>,
                grid,
                block,
                args,
                0,
                s);
            return;
        }
    }
#endif
    return;
}

// ================================================================
// C API: one-time weight repack
// ================================================================

extern "C" void prefill_mma_padded_repack_weight(
    int N, int K,
    const void* B,           // [N, K/2] uint8 row-major
    void* B_rep,             // output: tile-major data
    const void* B_sf,        // swizzled scale layout, padded to ceil(N/128)
    void* B_sf_rep,          // output: tile-major scale factors
    unsigned long long stream)
{
    if (!((N == 5120 && K == 17408) || (N == 34816 && K == 5120))) return;

    cudaStream_t s = (cudaStream_t)stream;
    int N_pad = round_up_128(N);
    int Kt = K / 64;
    int Ksg = K / 64;
    int th = 256;

    int data_threads = Kt * N_pad * 8;
    if (N == 34816 && K == 5120) {
        repack_data_128_frag_bpair_padded<<<(data_threads + th - 1) / th, th, 0, s>>>(
            (const uint8_t*)B, (uint8_t*)B_rep, N, N_pad, K);
    } else {
        repack_data_128_frag_padded<<<(data_threads + th - 1) / th, th, 0, s>>>(
            (const uint8_t*)B, (uint8_t*)B_rep, N, N_pad, K);
    }

    int sf_threads = N_pad * Ksg;
    repack_sf_unswizzle_128_frag_padded<<<(sf_threads + th - 1) / th, th, 0, s>>>(
        (const uint8_t*)B_sf, (uint8_t*)B_sf_rep, N, N_pad, K);
}
