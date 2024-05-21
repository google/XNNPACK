// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qd8-f16-qb4w-gemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py

#include <benchmark/benchmark.h>
#include "bench/gemm-benchmark.h"
#include "bench/utils.h"

#include <xnnpack/isa-checks.h>
#include <xnnpack/gemm.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>


static void qd8_f16_qb4w_gemm_minmax_ukernel_1x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x2__scalar,
    xnn_init_f16_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_1x2__scalar)

static void qd8_f16_qb4w_gemm_minmax_ukernel_1x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x4__scalar,
    xnn_init_f16_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_1x4__scalar)

static void qd8_f16_qb4w_gemm_minmax_ukernel_1x8__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8__scalar,
    xnn_init_f16_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_1x8__scalar)

static void qd8_f16_qb4w_gemm_minmax_ukernel_2x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x2__scalar,
    xnn_init_f16_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_2x2__scalar)

static void qd8_f16_qb4w_gemm_minmax_ukernel_2x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x4__scalar,
    xnn_init_f16_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_2x4__scalar)

static void qd8_f16_qb4w_gemm_minmax_ukernel_2x8__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x8__scalar,
    xnn_init_f16_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_2x8__scalar)

static void qd8_f16_qb4w_gemm_minmax_ukernel_4x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x4__scalar,
    xnn_init_f16_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_4x4__scalar)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
