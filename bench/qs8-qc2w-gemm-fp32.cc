// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-qc2w-gemm-minmax-fp32.yaml
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



static void qs8_qc2w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_qc2w_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf)

static void qs8_qc2w_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_qc2w_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic)

static void qs8_qc2w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_qc2w_gemm_goi_w,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic)

static void qs8_qc2w_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_qc2w_gemm_goi_w,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic)

static void qs8_qc2w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_qc2w_gemm_goi_w,
    /*mr=*/3, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic)

static void qs8_qc2w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_qc2w_gemm_goi_w,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic)

}  // namespace

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
