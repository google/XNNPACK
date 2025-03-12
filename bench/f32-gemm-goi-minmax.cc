// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-gemm-goi-minmax.yaml
//   Generator: tools/generate-gemm-test.py

#include <benchmark/benchmark.h>
#include "bench/gemm-benchmark.h"
#include "bench/utils.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/packw.h"


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128)

  static void f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm)

  static void f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
