// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qd8-f16-qc2w-gemm-minmax.yaml
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



#if XNN_ENABLE_AVX2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f16_qc2w_gemm_minmax_ukernel_1x8c8__avx2_madd(benchmark::State& state) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc2w_gemm_minmax_ukernel_1x8c8__avx2_madd,
      xnn_init_f16_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc2w_gemm_minmax_ukernel_1x8c8__avx2_madd)

  static void qd8_f16_qc2w_gemm_minmax_ukernel_2x8c8__avx2_madd(benchmark::State& state) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc2w_gemm_minmax_ukernel_2x8c8__avx2_madd,
      xnn_init_f16_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc2w_gemm_minmax_ukernel_2x8c8__avx2_madd)

  static void qd8_f16_qc2w_gemm_minmax_ukernel_3x8c8__avx2_madd(benchmark::State& state) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc2w_gemm_minmax_ukernel_3x8c8__avx2_madd,
      xnn_init_f16_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc2w_gemm_minmax_ukernel_3x8c8__avx2_madd)

  static void qd8_f16_qc2w_gemm_minmax_ukernel_4x8c8__avx2_madd(benchmark::State& state) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc2w_gemm_minmax_ukernel_4x8c8__avx2_madd,
      xnn_init_f16_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc2w_gemm_minmax_ukernel_4x8c8__avx2_madd)

  static void qd8_f16_qc2w_gemm_minmax_ukernel_5x8c8__avx2_madd(benchmark::State& state) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc2w_gemm_minmax_ukernel_5x8c8__avx2_madd,
      xnn_init_f16_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc2w_gemm_minmax_ukernel_5x8c8__avx2_madd)

  static void qd8_f16_qc2w_gemm_minmax_ukernel_6x8c8__avx2_madd(benchmark::State& state) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc2w_gemm_minmax_ukernel_6x8c8__avx2_madd,
      xnn_init_f16_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc2w_gemm_minmax_ukernel_6x8c8__avx2_madd)

  static void qd8_f16_qc2w_gemm_minmax_ukernel_7x8c8__avx2_madd(benchmark::State& state) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc2w_gemm_minmax_ukernel_7x8c8__avx2_madd,
      xnn_init_f16_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc2w_gemm_minmax_ukernel_7x8c8__avx2_madd)

  static void qd8_f16_qc2w_gemm_minmax_ukernel_8x8c8__avx2_madd(benchmark::State& state) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc2w_gemm_minmax_ukernel_8x8c8__avx2_madd,
      xnn_init_f16_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc2w_gemm_minmax_ukernel_8x8c8__avx2_madd)
#endif  // XNN_ENABLE_AVX2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


}  // namespace

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
