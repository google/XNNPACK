// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qd8-bf16-qb4w-gemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py

#include <cstdint>
#include <functional>

#include <benchmark/benchmark.h>
#include "bench/gemm-benchmark.h"
#include "bench/utils.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/packw.h"

namespace {



static void qd8_bf16_qb4w_gemm_minmax_ukernel_1x2__scalar(benchmark::State& state) {
  GEMMBenchmark(state,
    xnn_qd8_bf16_qb4w_gemm_minmax_ukernel_1x2__scalar,
    xnn_init_bf16_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_bf16_qb4w_gemm_minmax_ukernel_1x2__scalar)

static void qd8_bf16_qb4w_gemm_minmax_ukernel_1x4__scalar(benchmark::State& state) {
  GEMMBenchmark(state,
    xnn_qd8_bf16_qb4w_gemm_minmax_ukernel_1x4__scalar,
    xnn_init_bf16_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_bf16_qb4w_gemm_minmax_ukernel_1x4__scalar)

static void qd8_bf16_qb4w_gemm_minmax_ukernel_1x8__scalar(benchmark::State& state) {
  GEMMBenchmark(state,
    xnn_qd8_bf16_qb4w_gemm_minmax_ukernel_1x8__scalar,
    xnn_init_bf16_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_bf16_qb4w_gemm_minmax_ukernel_1x8__scalar)

static void qd8_bf16_qb4w_gemm_minmax_ukernel_2x2__scalar(benchmark::State& state) {
  GEMMBenchmark(state,
    xnn_qd8_bf16_qb4w_gemm_minmax_ukernel_2x2__scalar,
    xnn_init_bf16_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_bf16_qb4w_gemm_minmax_ukernel_2x2__scalar)

static void qd8_bf16_qb4w_gemm_minmax_ukernel_2x4__scalar(benchmark::State& state) {
  GEMMBenchmark(state,
    xnn_qd8_bf16_qb4w_gemm_minmax_ukernel_2x4__scalar,
    xnn_init_bf16_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_bf16_qb4w_gemm_minmax_ukernel_2x4__scalar)

static void qd8_bf16_qb4w_gemm_minmax_ukernel_2x8__scalar(benchmark::State& state) {
  GEMMBenchmark(state,
    xnn_qd8_bf16_qb4w_gemm_minmax_ukernel_2x8__scalar,
    xnn_init_bf16_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_bf16_qb4w_gemm_minmax_ukernel_2x8__scalar)

static void qd8_bf16_qb4w_gemm_minmax_ukernel_4x4__scalar(benchmark::State& state) {
  GEMMBenchmark(state,
    xnn_qd8_bf16_qb4w_gemm_minmax_ukernel_4x4__scalar,
    xnn_init_bf16_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_bf16_qb4w_gemm_minmax_ukernel_4x4__scalar)

}  // namespace

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
