// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x8-packw.yaml
//   Generator: tools/generate-packw-test.py


#include <benchmark/benchmark.h>
#include "bench/bgemm.h"
#include "bench/packw-benchmark.h"
#include "bench/utils.h"
#include "xnnpack/common.h"
#include "xnnpack/hardware-config.h"
#include "xnnpack/packw.h"


static void x8_packw_gemm_goi_ukernel_x2__scalar_int_u2(benchmark::State& state, const char* net) {
  x8_packw(state,
    xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u2,
    /*nr=*/2, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x8_packw_gemm_goi_ukernel_x2__scalar_int_u2)

static void x8_packw_gemm_goi_ukernel_x4__scalar_int_u2(benchmark::State& state, const char* net) {
  x8_packw(state,
    xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u2,
    /*nr=*/4, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x8_packw_gemm_goi_ukernel_x4__scalar_int_u2)

static void x8_packw_gemm_goi_ukernel_x8__scalar_int_u2(benchmark::State& state, const char* net) {
  x8_packw(state,
    xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u2,
    /*nr=*/8, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x8_packw_gemm_goi_ukernel_x8__scalar_int_u2)

static void x8_packw_gemm_goi_ukernel_x16__scalar_int_u2(benchmark::State& state, const char* net) {
  x8_packw(state,
    xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u2,
    /*nr=*/16, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x8_packw_gemm_goi_ukernel_x16__scalar_int_u2)

static void x8_packw_gemm_goi_ukernel_x32__scalar_int_u2(benchmark::State& state, const char* net) {
  x8_packw(state,
    xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u2,
    /*nr=*/32, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x8_packw_gemm_goi_ukernel_x32__scalar_int_u2)

static void x8_packw_gemm_goi_ukernel_x2__scalar_int_u4(benchmark::State& state, const char* net) {
  x8_packw(state,
    xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u4,
    /*nr=*/2, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x8_packw_gemm_goi_ukernel_x2__scalar_int_u4)

static void x8_packw_gemm_goi_ukernel_x4__scalar_int_u4(benchmark::State& state, const char* net) {
  x8_packw(state,
    xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u4,
    /*nr=*/4, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x8_packw_gemm_goi_ukernel_x4__scalar_int_u4)

static void x8_packw_gemm_goi_ukernel_x8__scalar_int_u4(benchmark::State& state, const char* net) {
  x8_packw(state,
    xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u4,
    /*nr=*/8, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x8_packw_gemm_goi_ukernel_x8__scalar_int_u4)

static void x8_packw_gemm_goi_ukernel_x16__scalar_int_u4(benchmark::State& state, const char* net) {
  x8_packw(state,
    xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u4,
    /*nr=*/16, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x8_packw_gemm_goi_ukernel_x16__scalar_int_u4)

static void x8_packw_gemm_goi_ukernel_x32__scalar_int_u4(benchmark::State& state, const char* net) {
  x8_packw(state,
    xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u4,
    /*nr=*/32, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x8_packw_gemm_goi_ukernel_x32__scalar_int_u4)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
