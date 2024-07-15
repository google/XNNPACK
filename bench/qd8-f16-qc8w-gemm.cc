// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qd8-f16-qc8w-gemm-minmax.yaml
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
#include "xnnpack/packw.h"


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_1x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x32c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_1x32c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_2x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x16c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_2x16c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_2x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x32c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_2x32c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_3x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x16c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_3x16c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_3x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x32c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_3x32c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_4x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x32c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_4x32c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_5x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x16c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_5x16c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_5x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x32c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_5x32c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_6x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x8c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_6x8c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_6x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x16c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_6x16c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_6x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x32c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_6x32c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_7x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x16c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_7x16c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_7x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x32c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_7x32c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_8x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x16c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_8x16c8__neoni8mm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_8x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x32c8__neoni8mm,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_8x32c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_AVX512AMX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f16_qc8w_gemm_minmax_ukernel_1x64c4__avx512amx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x64c4__avx512amx,
      xnn_init_f16_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVX512AMX);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_1x64c4__avx512amx)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_7x64c4__avx512amx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x64c4__avx512amx,
      xnn_init_f16_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVX512AMX);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_7x64c4__avx512amx)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx,
      xnn_init_f16_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/16, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVX512AMX);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx_prfm,
      xnn_init_f16_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/16, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVX512AMX);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx_prfm)
#endif  // XNN_ENABLE_AVX512AMX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx256vnni,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX256VNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx256vnni)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avx256vnni,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX256VNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avx256vnni)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avx256vnni,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX256VNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avx256vnni)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avx256vnni,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX256VNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avx256vnni)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_9x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_9x8c8__avx256vnni,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/9, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX256VNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_9x8c8__avx256vnni)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_10x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_10x8c8__avx256vnni,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/10, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX256VNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_10x8c8__avx256vnni)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_12x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_12x8c8__avx256vnni,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/12, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX256VNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_12x8c8__avx256vnni)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_14x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_14x8c8__avx256vnni,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/14, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX256VNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_14x8c8__avx256vnni)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx256vnni_prfm,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX256VNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx256vnni_prfm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avx256vnni_prfm,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX256VNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avx256vnni_prfm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avx256vnni_prfm,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX256VNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avx256vnni_prfm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avx256vnni_prfm,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX256VNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avx256vnni_prfm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_9x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_9x8c8__avx256vnni_prfm,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/9, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX256VNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_9x8c8__avx256vnni_prfm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_10x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_10x8c8__avx256vnni_prfm,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/10, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX256VNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_10x8c8__avx256vnni_prfm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_12x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_12x8c8__avx256vnni_prfm,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/12, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX256VNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_12x8c8__avx256vnni_prfm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_14x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_14x8c8__avx256vnni_prfm,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/14, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX256VNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_14x8c8__avx256vnni_prfm)
#endif  // XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx256skx,
      xnn_init_f16_minmax_avx_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX256SKX);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx256skx)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avx256skx,
      xnn_init_f16_minmax_avx_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX256SKX);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avx256skx)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avx256skx,
      xnn_init_f16_minmax_avx_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX256SKX);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avx256skx)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avx256skx,
      xnn_init_f16_minmax_avx_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX256SKX);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avx256skx)
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__avxvnni,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__avxvnni)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__avxvnni,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__avxvnni)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__avxvnni,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__avxvnni)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_6x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x8c8__avxvnni,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_6x8c8__avxvnni)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avxvnni,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avxvnni)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avxvnni,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avxvnni)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__avxvnni_prfm,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__avxvnni_prfm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__avxvnni_prfm,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__avxvnni_prfm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__avxvnni_prfm,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__avxvnni_prfm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_6x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x8c8__avxvnni_prfm,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_6x8c8__avxvnni_prfm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avxvnni_prfm,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avxvnni_prfm)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avxvnni_prfm,
      xnn_init_f16_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avxvnni_prfm)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx2,
      xnn_init_f16_minmax_avx_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx2)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__avx2,
      xnn_init_f16_minmax_avx_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__avx2)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__avx2,
      xnn_init_f16_minmax_avx_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__avx2)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__avx2,
      xnn_init_f16_minmax_avx_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__avx2)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f16_qc8w_gemm_minmax_ukernel_1x8c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c4__neondotfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_1x8c4__neondotfp16arith)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_1x16c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c4__neondotfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_1x16c4__neondotfp16arith)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_2x8c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c4__neondotfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_2x8c4__neondotfp16arith)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_2x16c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x16c4__neondotfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_2x16c4__neondotfp16arith)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_3x8c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x8c4__neondotfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_3x8c4__neondotfp16arith)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_3x16c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x16c4__neondotfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_3x16c4__neondotfp16arith)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_4x8c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c4__neondotfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_4x8c4__neondotfp16arith)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_cortex_a55,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_cortex_a55)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_ld128,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_ld128)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__neondotfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__neondotfp16arith)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_5x8c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c4__neondotfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_5x8c4__neondotfp16arith)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_5x16c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x16c4__neondotfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_5x16c4__neondotfp16arith)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_6x8c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x8c4__neondotfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_6x8c4__neondotfp16arith)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_6x16c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x16c4__neondotfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_6x16c4__neondotfp16arith)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f16_qc8w_gemm_minmax_ukernel_1x8c2s4__neonfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c2s4__neonfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_1x8c2s4__neonfp16arith)

  static void qd8_f16_qc8w_gemm_minmax_ukernel_2x8c2s4__neonfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c2s4__neonfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_2x8c2s4__neonfp16arith)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void qd8_f16_qc8w_gemm_minmax_ukernel_4x8c4__asm_aarch32_neondotfp16arith_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c4__asm_aarch32_neondotfp16arith_cortex_a55,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f16_qc8w_gemm_minmax_ukernel_4x8c4__asm_aarch32_neondotfp16arith_cortex_a55)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
