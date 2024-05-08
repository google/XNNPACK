// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qd8-f32-qb4w-gemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py

#include <benchmark/benchmark.h>
#include "bench/gemm-benchmark.h"
#include "bench/utils.h"
#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pack.h"


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x8c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x8c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x16c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x8c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x8c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x16c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x16c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x8c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x8c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x16c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x16c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x8c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x8c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x16c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x16c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_5x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x8c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_5x8c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_5x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x16c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_5x16c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_6x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x8c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_6x8c4__neondot)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_6x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x16c4__neondot,
      xnn_init_f32_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_6x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__avx_ld128,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__avx_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__avx_ld128,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__avx_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__avx_ld128,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__avx_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__avx_ld128,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__avx_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__avx_ld64,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__avx_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__avx_ld64,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__avx_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__avx_ld64,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__avx_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__avx_ld64,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__avx_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse41_ld128,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse41_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse41_ld128,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse41_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse41_ld128,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse41_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse41_ld128,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse41_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse41_ld64,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse41_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse41_ld64,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse41_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse41_ld64,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse41_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse41_ld64,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse41_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse2_ld128,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse2_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse2_ld128,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse2_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse2_ld128,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse2_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse2_ld128,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse2_ld128)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse2_ld64,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse2_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse2_ld64,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse2_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse2_ld64,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse2_ld64)

  static void qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse2_ld64,
      xnn_init_f32_qb4w_minmax_sse_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse2_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


static void qd8_f32_qb4w_gemm_minmax_ukernel_1x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x2__scalar,
    xnn_init_f32_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x2__scalar)

static void qd8_f32_qb4w_gemm_minmax_ukernel_1x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4__scalar,
    xnn_init_f32_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x4__scalar)

static void qd8_f32_qb4w_gemm_minmax_ukernel_1x8__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x8__scalar,
    xnn_init_f32_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_1x8__scalar)

static void qd8_f32_qb4w_gemm_minmax_ukernel_2x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x2__scalar,
    xnn_init_f32_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x2__scalar)

static void qd8_f32_qb4w_gemm_minmax_ukernel_2x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4__scalar,
    xnn_init_f32_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x4__scalar)

static void qd8_f32_qb4w_gemm_minmax_ukernel_2x8__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x8__scalar,
    xnn_init_f32_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_2x8__scalar)

static void qd8_f32_qb4w_gemm_minmax_ukernel_4x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4__scalar,
    xnn_init_f32_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM_BL(qd8_f32_qb4w_gemm_minmax_ukernel_4x4__scalar)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
