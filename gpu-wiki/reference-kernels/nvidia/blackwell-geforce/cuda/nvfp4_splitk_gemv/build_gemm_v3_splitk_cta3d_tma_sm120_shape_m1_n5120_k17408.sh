#!/usr/bin/env bash
set -euo pipefail

# gpu-wiki archive note:
# Build helper for the SM120 CTA-3D TMA task38 snapshot:
# gemm_v3_splitk_cta3d_tma_sm120_shape_m1_n5120_k17408.cu.
# It intentionally bakes the scoped fast path macros used by the archived
# kernel. Do not treat this as a generic Split-K build script.

cd "$(dirname "$0")"

if [ -z "${CUTLASS_DIR:-}" ]; then
  echo "CUTLASS_DIR must point to a CUTLASS checkout." >&2
  echo "Example: export CUTLASS_DIR=/path/to/cutlass" >&2
  exit 2
fi

MTILE_N_GROUPS="${MTILE_N_GROUPS:-3}"

nvcc -gencode arch=compute_120a,code=sm_120a \
     -shared -Xcompiler -fPIC -O3 -std=c++17 \
     -I"${CUTLASS_DIR}/include" \
     -I"${CUTLASS_DIR}/tools/util/include" \
     -DB_SWIZZLE_MODE=0 -DZERO_A3_ONLY=1 -DUSE_TMA_B=1 -DTMA_B_CTA_3D=1 \
     -DMTILE_N_GROUPS="${MTILE_N_GROUPS}" \
     -cudart shared -lcuda \
     -o gemm_v3_splitk_cta3d_tma_sm120_shape_m1_n5120_k17408.so \
     gemm_v3_splitk_cta3d_tma_sm120_shape_m1_n5120_k17408.cu

echo "Built gemm_v3_splitk_cta3d_tma_sm120_shape_m1_n5120_k17408.so ($(date))"
