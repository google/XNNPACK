// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "bench/bgemm.h"
#include "bench/packq-benchmark.h"
#include "bench/utils.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/packq.h"
#include <benchmark/benchmark.h>

static void x8_packq(benchmark::State& state, const char* net,
                     xnn_x8_packq_f32qp8_ukernel_fn ukernel,
                     uint64_t arch_flags, size_t mr, size_t kr) {
  benchmark::utils::CheckArchFlags(state, arch_flags);
  constexpr size_t sr = 1;
  x8_packq(state, ukernel, mr, kr, sr);
}

#define XNN_UKERNEL(arch_flags, ukernel, unroll)                             \
  BENCHMARK_CAPTURE_BGEMM(x8_packq, ukernel##_mr1_kr1_, ukernel, arch_flags, \
                          /*mr=*/1, /*kr=*/1);                               \
  BENCHMARK_CAPTURE_BGEMM(x8_packq, ukernel##_mr1_kr2_, ukernel, arch_flags, \
                          /*mr=*/1, /*kr=*/2);                               \
  BENCHMARK_CAPTURE_BGEMM(x8_packq, ukernel##_mr1_kr4_, ukernel, arch_flags, \
                          /*mr=*/1, /*kr=*/4);                               \
  BENCHMARK_CAPTURE_BGEMM(x8_packq, ukernel##_mr2_kr1_, ukernel, arch_flags, \
                          /*mr=*/2, /*kr=*/1);                               \
  BENCHMARK_CAPTURE_BGEMM(x8_packq, ukernel##_mr2_kr2_, ukernel, arch_flags, \
                          /*mr=*/2, /*kr=*/2);                               \
  BENCHMARK_CAPTURE_BGEMM(x8_packq, ukernel##_mr2_kr4_, ukernel, arch_flags, \
                          /*mr=*/2, /*kr=*/4);                               \
  BENCHMARK_CAPTURE_BGEMM(x8_packq, ukernel##_mr4_kr1_, ukernel, arch_flags, \
                          /*mr=*/4, /*kr=*/1);                               \
  BENCHMARK_CAPTURE_BGEMM(x8_packq, ukernel##_mr4_kr2_, ukernel, arch_flags, \
                          /*mr=*/4, /*kr=*/2);                               \
  BENCHMARK_CAPTURE_BGEMM(x8_packq, ukernel##_mr4_kr4_, ukernel, arch_flags, \
                          /*mr=*/4, /*kr=*/4);

#include "src/x8-packq/x8-packq.inc"

#undef XNN_UKERNEL

#if XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME2_ACLE
static void x8_packq_sme2(benchmark::State& state,
                          xnn_x8_packq_f32qp8_ukernel_fn ukernel,
                          size_t (*get_mr)()) {
  if (!benchmark::utils::CheckArchFlags(state, xnn_arch_arm_sme2)) {
    return;
  }
  x8_packq(state, ukernel, get_mr(), /*kr=*/4, /*sr=*/1, xnn_arch_arm_sme2);
}

BENCHMARK_CAPTURE_BGEMM(x8_packq_sme2, xnn_x8_packq_f32qp8_ukernel__sme2_1x32c4_,
                        xnn_x8_packq_f32qp8_ukernel__sme2,
                        xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x32c4__sme2_get_mr);
BENCHMARK_CAPTURE_BGEMM(x8_packq_sme2, xnn_x8_packq_f32qp8_ukernel__sme2_32x32c4_,
                        xnn_x8_packq_f32qp8_ukernel__sme2,
                        xnn_qp8_f32_qc4w_gemm_minmax_ukernel_32x32c4__sme2_get_mr);
#endif  // XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME2_ACLE

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
