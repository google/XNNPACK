// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-qc4w-gemm-minmax-fp32.yaml
//   Generator: tools/generate-gemm-test.py

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


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_qc4w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld32_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld32_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld32_2)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld32_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld32_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld32_2)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld32_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld32_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld32_2)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld32_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld32_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld32_2)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld32_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld32_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld32_2)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld64_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld64_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld64_2)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld64_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld64_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld64_2)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld64_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld64_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld64_2)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld64_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld64_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld64_2)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld64_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld64_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld64_2)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld128_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld128_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld128_2)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld128_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld128_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld128_2)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld128_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld128_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld128_2)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128_2)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld128_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld128_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld128_2)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_AVX512VNNI && XNN_ARCH_X86_64 && XNN_ENABLE_ASSEMBLY
  static void qs8_qc4w_gemm_minmax_fp32_ukernel_1x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_1x16c8__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_1x16c8__asm_amd64_avx512vnni)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_2x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_2x16c8__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_2x16c8__asm_amd64_avx512vnni)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_3x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_3x16c8__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_3x16c8__asm_amd64_avx512vnni)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_4x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_4x16c8__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_4x16c8__asm_amd64_avx512vnni)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_5x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_5x16c8__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_5x16c8__asm_amd64_avx512vnni)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_6x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_6x16c8__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_6x16c8__asm_amd64_avx512vnni)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_7x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_7x16c8__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_7x16c8__asm_amd64_avx512vnni)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_8x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_8x16c8__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_8x16c8__asm_amd64_avx512vnni)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_9x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_9x16c8__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_9x16c8__asm_amd64_avx512vnni)
#endif  // XNN_ENABLE_AVX512VNNI && XNN_ARCH_X86_64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qs8_qc4w_gemm_minmax_fp32_ukernel_1x4c8__ssse3_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_1x4c8__ssse3_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc4uw_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_1x4c8__ssse3_madd)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_2x4c8__ssse3_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_2x4c8__ssse3_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc4uw_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_2x4c8__ssse3_madd)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_3x4c8__ssse3_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_3x4c8__ssse3_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc4uw_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_3x4c8__ssse3_madd)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_4x4c8__ssse3_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_4x4c8__ssse3_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc4uw_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_4x4c8__ssse3_madd)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_5x4c8__ssse3_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_5x4c8__ssse3_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc4uw_gemm_goi_w,
      /*mr=*/5, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_5x4c8__ssse3_madd)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_6x4c8__ssse3_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_6x4c8__ssse3_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc4uw_gemm_goi_w,
      /*mr=*/6, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_6x4c8__ssse3_madd)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_1x4c8__ssse3_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_1x4c8__ssse3_madd_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc4uw_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_1x4c8__ssse3_madd_prfm)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_2x4c8__ssse3_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_2x4c8__ssse3_madd_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc4uw_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_2x4c8__ssse3_madd_prfm)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_3x4c8__ssse3_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_3x4c8__ssse3_madd_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc4uw_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_3x4c8__ssse3_madd_prfm)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_4x4c8__ssse3_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_4x4c8__ssse3_madd_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc4uw_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_4x4c8__ssse3_madd_prfm)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_5x4c8__ssse3_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_5x4c8__ssse3_madd_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc4uw_gemm_goi_w,
      /*mr=*/5, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_5x4c8__ssse3_madd_prfm)

  static void qs8_qc4w_gemm_minmax_fp32_ukernel_6x4c8__ssse3_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_6x4c8__ssse3_madd_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc4uw_gemm_goi_w,
      /*mr=*/6, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_6x4c8__ssse3_madd_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


static void qs8_qc4w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_scalar,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc4w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
