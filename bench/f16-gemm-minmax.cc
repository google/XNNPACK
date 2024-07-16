// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-gemm-minmax.yaml
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


#if XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void f16_gemm_minmax_ukernel_1x32__avx512fp16_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_1x32__avx512fp16_broadcast,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4,
      /*mr=*/1, /*nr=*/32, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512FP16);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_1x32__avx512fp16_broadcast)

  static void f16_gemm_minmax_ukernel_4x32__avx512fp16_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_4x32__avx512fp16_broadcast,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4,
      /*mr=*/4, /*nr=*/32, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512FP16);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_4x32__avx512fp16_broadcast)

  static void f16_gemm_minmax_ukernel_5x32__avx512fp16_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_5x32__avx512fp16_broadcast,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4,
      /*mr=*/5, /*nr=*/32, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512FP16);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_5x32__avx512fp16_broadcast)

  static void f16_gemm_minmax_ukernel_6x32__avx512fp16_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_6x32__avx512fp16_broadcast,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4,
      /*mr=*/6, /*nr=*/32, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512FP16);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_6x32__avx512fp16_broadcast)

  static void f16_gemm_minmax_ukernel_7x32__avx512fp16_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_7x32__avx512fp16_broadcast,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4,
      /*mr=*/7, /*nr=*/32, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512FP16);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_7x32__avx512fp16_broadcast)

  static void f16_gemm_minmax_ukernel_8x32__avx512fp16_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_8x32__avx512fp16_broadcast,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4,
      /*mr=*/8, /*nr=*/32, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512FP16);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_8x32__avx512fp16_broadcast)

  static void f16_gemm_minmax_ukernel_1x64__avx512fp16_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_1x64__avx512fp16_broadcast,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4,
      /*mr=*/1, /*nr=*/64, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512FP16);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_1x64__avx512fp16_broadcast)

  static void f16_gemm_minmax_ukernel_4x64__avx512fp16_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_4x64__avx512fp16_broadcast,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4,
      /*mr=*/4, /*nr=*/64, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512FP16);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_4x64__avx512fp16_broadcast)

  static void f16_gemm_minmax_ukernel_5x64__avx512fp16_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_5x64__avx512fp16_broadcast,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4,
      /*mr=*/5, /*nr=*/64, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512FP16);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_5x64__avx512fp16_broadcast)

  static void f16_gemm_minmax_ukernel_6x64__avx512fp16_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_6x64__avx512fp16_broadcast,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4,
      /*mr=*/6, /*nr=*/64, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512FP16);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_6x64__avx512fp16_broadcast)

  static void f16_gemm_minmax_ukernel_7x64__avx512fp16_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_7x64__avx512fp16_broadcast,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4,
      /*mr=*/7, /*nr=*/64, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512FP16);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_7x64__avx512fp16_broadcast)

  static void f16_gemm_minmax_ukernel_8x64__avx512fp16_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_8x64__avx512fp16_broadcast,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4,
      /*mr=*/8, /*nr=*/64, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512FP16);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_8x64__avx512fp16_broadcast)
#endif  // XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f16_gemm_minmax_ukernel_1x8__avx2_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_1x8__avx2_broadcast,
      xnn_init_f16_minmax_avx_params,
      xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_1x8__avx2_broadcast)

  static void f16_gemm_minmax_ukernel_4x8__avx2_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_4x8__avx2_broadcast,
      xnn_init_f16_minmax_avx_params,
      xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_4x8__avx2_broadcast)

  static void f16_gemm_minmax_ukernel_5x8__avx2_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_5x8__avx2_broadcast,
      xnn_init_f16_minmax_avx_params,
      xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_5x8__avx2_broadcast)

  static void f16_gemm_minmax_ukernel_6x8__avx2_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_6x8__avx2_broadcast,
      xnn_init_f16_minmax_avx_params,
      xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_6x8__avx2_broadcast)

  static void f16_gemm_minmax_ukernel_7x8__avx2_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_7x8__avx2_broadcast,
      xnn_init_f16_minmax_avx_params,
      xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16,
      /*mr=*/7, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_7x8__avx2_broadcast)

  static void f16_gemm_minmax_ukernel_1x16__avx2_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_1x16__avx2_broadcast,
      xnn_init_f16_minmax_avx_params,
      xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_1x16__avx2_broadcast)

  static void f16_gemm_minmax_ukernel_3x16__avx2_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_3x16__avx2_broadcast,
      xnn_init_f16_minmax_avx_params,
      xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_3x16__avx2_broadcast)

  static void f16_gemm_minmax_ukernel_4x16__avx2_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_4x16__avx2_broadcast,
      xnn_init_f16_minmax_avx_params,
      xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_4x16__avx2_broadcast)

  static void f16_gemm_minmax_ukernel_5x16__avx2_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_5x16__avx2_broadcast,
      xnn_init_f16_minmax_avx_params,
      xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm,
      /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_5x16__avx2_broadcast)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void f16_gemm_minmax_ukernel_1x8__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_1x8__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_1x8__asm_aarch64_neonfp16arith_ld64)

  static void f16_gemm_minmax_ukernel_4x8__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_4x8__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_4x8__asm_aarch64_neonfp16arith_ld64)

  static void f16_gemm_minmax_ukernel_6x8__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_6x8__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_6x8__asm_aarch64_neonfp16arith_ld64)

  static void f16_gemm_minmax_ukernel_8x8__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_8x8__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_8x8__asm_aarch64_neonfp16arith_ld64)

  static void f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld32,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld32)

  static void f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64)

  static void f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld32,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld32)

  static void f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64)

  static void f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55)

  static void f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0)

  static void f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75)

  static void f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld32,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld32)

  static void f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld64)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64)

  static void f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64)

  static void f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64)

  static void f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64)

  static void f16_gemm_minmax_ukernel_1x16__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_1x16__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_1x16__neonfp16arith_ld64)

  static void f16_gemm_minmax_ukernel_4x16__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_4x16__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_4x16__neonfp16arith_ld64)

  static void f16_gemm_minmax_ukernel_6x16__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_6x16__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_6x16__neonfp16arith_ld64)

  static void f16_gemm_minmax_ukernel_8x16__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f16_gemm_minmax_ukernel_8x16__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8,
      /*mr=*/8, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_minmax_ukernel_8x16__neonfp16arith_ld64)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
