// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-gemm-minmax-rndnu.yaml
//   Generator: tools/generate-gemm-test.py

#include <benchmark/benchmark.h>
#include "bench/gemm-benchmark.h"
#include "bench/utils.h"

#include <xnnpack/isa-checks.h>
#include <xnnpack/gemm.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c8__neoni8mm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c8__neoni8mm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c8__neoni8mm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c8__neoni8mm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neoni8mm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c8__neoni8mm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_gemm_minmax_rndnu_ukernel_6x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_6x8c8__neoni8mm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_6x8c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_gemm_minmax_rndnu_ukernel_6x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_6x16c8__neoni8mm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_6x16c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_gemm_minmax_rndnu_ukernel_8x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_8x8c8__neoni8mm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_8x8c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_gemm_minmax_rndnu_ukernel_8x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_8x16c8__neoni8mm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_8x16c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c4__asm_aarch32_neondot_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__asm_aarch32_neondot_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c4__asm_aarch32_neondot_ld64)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld64)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld128,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld128)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neondot,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c8__aarch64_neondot_ld128)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c8__neondot_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c8__neondot_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c8__neondot_ld64)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neondot,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c8__aarch64_neondot_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c8__aarch64_neondot_ld128,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c8__aarch64_neondot_ld128)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c8__neondot_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c8__neondot_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c8__neondot_ld64)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neondot,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neondot,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_gemm_minmax_rndnu_ukernel_6x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_6x8c4__neondot,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_6x8c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_gemm_minmax_rndnu_ukernel_6x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_6x16c4__neondot,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_6x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_gemm_minmax_rndnu_ukernel_8x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_8x8c4__neondot,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_8x8c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_gemm_minmax_rndnu_ukernel_8x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_8x16c4__neondot,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/8, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_8x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7)
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm)
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7)
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm)
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53)
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm)
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64)
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm)
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mlal)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c8__asm_aarch64_neon_mull)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c16__asm_aarch64_neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c16__asm_aarch64_neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/16, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c16__asm_aarch64_neon_mlal)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64_prfm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch64_neon_mlal_lane_ld64_prfm)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64_prfm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64_prfm)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mlal_ld4r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c4s2__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4s2__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/2,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c4s2__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c4s2__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4s2__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/2,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c4s2__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mull_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mull_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mull_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld4r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld4r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mull_ld4r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mull_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mull_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mull_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mull_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/2,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/2,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mlal_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mlal_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mlal_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mlal_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mlal_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mlal_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mlal_ld4r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mlal_ld4r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mlal_ld4r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld4r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld4r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld4r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c2s4__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2s4__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c2s4__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c2s4__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2s4__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c2s4__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mlal_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mlal_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mlal_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mlal_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mlal_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mlal_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/2,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/2,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld4r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld4r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mlal_ld4r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld4r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld4r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld4r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c2s4__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2s4__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c2s4__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c2s4__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c4__neon_mlal_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c4__neon_mull_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c4__neon_mull_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c4__neon_mull_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c4__neon_mull_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/2,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c4s2__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c4s2__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c4s2__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/2,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c4s2__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld4r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld4r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld4r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mull_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mull_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mull_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld4r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld4r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mull_ld4r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8c2s4__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2s4__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8c2s4__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8c2s4__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mull_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mull_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mull_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mull_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/2,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8c4s2__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8c4s2__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4s2__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/2,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8c4s2__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld4r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld4r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld4r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld4r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld4r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld4r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16c2s4__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2s4__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16c2s4__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16c2s4__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/2,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16c4s2__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16c4s2__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4s2__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/2,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16c4s2__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld4r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld4r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld4r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mull_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mull_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mull_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld4r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld4r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mull_ld4r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/2,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c4s2__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c4s2__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4s2__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/2,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c4s2__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld4r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld4r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mlal_ld4r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld4r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld4r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld4r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c2s4__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2s4__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c2s4__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c2s4__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2s4__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c2s4__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld2r)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/2,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/2,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane_prfm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8__neon_mull_addw_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8__neon_mull_addw_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8__neon_mull_addw_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c8__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c8__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c8__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c8__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c8__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c8__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x8c16__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c16__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/16, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x8c16__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane_prfm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16__neon_mull_addw_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16__neon_mull_addw_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16__neon_mull_addw_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c8__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c8__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c8__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c8__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c8__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c8__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_1x16c16__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c16__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/16, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x16c16__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8__neon_mlal_lane,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8__neon_mlal_lane_prfm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8__neon_mull_addw_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8__neon_mull_addw_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8__neon_mull_addw_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c8__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c8__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c8__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c8__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c8__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c8__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x8c16__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c16__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/16, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x8c16__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mull_addw_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mull_addw_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mull_addw_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c8__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c8__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c8__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c8__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c8__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c8__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/16, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8c8__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c8__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8c8__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8c8__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c8__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8c8__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x8c16__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c16__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/16, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x8c16__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mull_addw_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16c8__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c8__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16c8__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16c8__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c8__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16c8__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_3x16c16__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c16__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/16, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x16c16__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane_prfm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x8c16__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c16__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/16, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x8c16__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane_prfm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mull_addw_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mull_addw_dup,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mull_addw_dup)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c8__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c8__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c8__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c8__neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c8__neon_mull,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c8__neon_mull)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_4x16c16__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c16__neon_mlal,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/16, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x16c16__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane_prfm,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


static void qs8_gemm_minmax_rndnu_ukernel_1x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_gemm_minmax_rndnu_ukernel_1x2__scalar,
    xnn_init_qs8_conv_minmax_rndnu_scalar_params,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x2__scalar)

static void qs8_gemm_minmax_rndnu_ukernel_1x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_gemm_minmax_rndnu_ukernel_1x4__scalar,
    xnn_init_qs8_conv_minmax_rndnu_scalar_params,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_1x4__scalar)

static void qs8_gemm_minmax_rndnu_ukernel_2x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar,
    xnn_init_qs8_conv_minmax_rndnu_scalar_params,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x2__scalar)

static void qs8_gemm_minmax_rndnu_ukernel_2x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_gemm_minmax_rndnu_ukernel_2x4__scalar,
    xnn_init_qs8_conv_minmax_rndnu_scalar_params,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_2x4__scalar)

static void qs8_gemm_minmax_rndnu_ukernel_3x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar,
    xnn_init_qs8_conv_minmax_rndnu_scalar_params,
    /*mr=*/3, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x2__scalar)

static void qs8_gemm_minmax_rndnu_ukernel_3x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar,
    xnn_init_qs8_conv_minmax_rndnu_scalar_params,
    /*mr=*/3, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_3x4__scalar)

static void qs8_gemm_minmax_rndnu_ukernel_4x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar,
    xnn_init_qs8_conv_minmax_rndnu_scalar_params,
    /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x2__scalar)

static void qs8_gemm_minmax_rndnu_ukernel_4x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar,
    xnn_init_qs8_conv_minmax_rndnu_scalar_params,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qs8_gemm_minmax_rndnu_ukernel_4x4__scalar)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
