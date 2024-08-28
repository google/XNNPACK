// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <benchmark/benchmark.h>
#include "bench/bgemm.h"
#include "bench/packq-benchmark.h"
#include "bench/utils.h"
#include "xnnpack/common.h"
#include "xnnpack/packq.h"


static void x8_packq(benchmark::State& state, const char* net,
                      xnn_x8_packq_f32qp8_ukernel_fn ukernel,
                      uint64_t arch_flags, size_t mr, size_t kr) {
  benchmark::utils::CheckArchFlags(state, arch_flags);
  constexpr size_t sr = 1;
  x8_packq(state, ukernel, mr, kr, sr);
}

#define XNN_UKERNEL(arch_flags, ukernel, unroll)                                                \
BENCHMARK_CAPTURE_BGEMM(x8_packq, ukernel##_mr1_kr1_, ukernel, arch_flags, /*mr=*/1, /*kr=*/1); \
BENCHMARK_CAPTURE_BGEMM(x8_packq, ukernel##_mr1_kr2_, ukernel, arch_flags, /*mr=*/1, /*kr=*/2); \
BENCHMARK_CAPTURE_BGEMM(x8_packq, ukernel##_mr1_kr4_, ukernel, arch_flags, /*mr=*/1, /*kr=*/4); \
BENCHMARK_CAPTURE_BGEMM(x8_packq, ukernel##_mr2_kr1_, ukernel, arch_flags, /*mr=*/2, /*kr=*/1); \
BENCHMARK_CAPTURE_BGEMM(x8_packq, ukernel##_mr2_kr2_, ukernel, arch_flags, /*mr=*/2, /*kr=*/2); \
BENCHMARK_CAPTURE_BGEMM(x8_packq, ukernel##_mr2_kr4_, ukernel, arch_flags, /*mr=*/2, /*kr=*/4); \
BENCHMARK_CAPTURE_BGEMM(x8_packq, ukernel##_mr4_kr1_, ukernel, arch_flags, /*mr=*/4, /*kr=*/1); \
BENCHMARK_CAPTURE_BGEMM(x8_packq, ukernel##_mr4_kr2_, ukernel, arch_flags, /*mr=*/4, /*kr=*/2); \
BENCHMARK_CAPTURE_BGEMM(x8_packq, ukernel##_mr4_kr4_, ukernel, arch_flags, /*mr=*/4, /*kr=*/4);

#include "src/x8-packq/x8-packq.h"

#undef XNN_UKERNEL

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif

