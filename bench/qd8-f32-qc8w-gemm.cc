// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qd8-f32-qc8w-gemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py

#include <benchmark/benchmark.h>
#include "bench/gemm-benchmark.h"
#include "bench/utils.h"

#include <xnnpack/isa-checks.h>
#include <xnnpack/gemm.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x2__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x2__wasm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x2__wasm)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x4__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4__wasm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x4__wasm)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x8__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8__wasm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x8__wasm)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x2__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x2__wasm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x2__wasm)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x4__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4__wasm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x4__wasm)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x8__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8__wasm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x8__wasm)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x4__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4__wasm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x4__wasm)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x32c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x32c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x32c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x32c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x32c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x32c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x32c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x32c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_5x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x32c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_5x32c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x32c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x32c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_7x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x32c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_7x32c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_8x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x32c8__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_8x32c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__aarch64_neondot_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__aarch64_neondot_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__aarch64_neondot_ld128)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__neondot_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__neondot_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__neondot_ld64)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__aarch64_neondot_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__aarch64_neondot_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__aarch64_neondot_ld128)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__neondot_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__neondot_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__neondot_ld64)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x8c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x8c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld64)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld128)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_5x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_5x8c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_5x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_5x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x8c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8__neon_mlal_lane,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x8__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x8__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8__neon_mlal_lane_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x8__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x8c2s4__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c2s4__neon_mlal,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x8c2s4__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16__neon_mlal_lane,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x16__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8__neon_mlal_lane,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x8__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x8__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8__neon_mlal_lane_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x8__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x8c2s4__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c2s4__neon_mlal,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x8c2s4__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16__neon_mlal_lane,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x16__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8__neon_mlal_lane,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x8__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x8__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8__neon_mlal_lane_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x8__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16__neon_mlal_lane,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x16__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8__neon_mlal_lane,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x8__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x8__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8__neon_mlal_lane_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x8__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16__neon_mlal_lane,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x16__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8__neon_mlal_lane,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x8__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x8__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8__neon_mlal_lane_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x8__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16__neon_mlal_lane,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x16__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__avx512vnni,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c4__avx512vnni,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x16c4__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c4__avx512vnni,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x16c4__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__avx512vnni,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_5x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c4__avx512vnni,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_5x16c4__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c4__avx512vnni,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x16c4__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_7x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c4__avx512vnni,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_7x16c4__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_8x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c4__avx512vnni,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_8x16c4__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__avx512vnni_prfm,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c4__avx512vnni_prfm,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x16c4__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c4__avx512vnni_prfm,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x16c4__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__avx512vnni_prfm,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_5x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c4__avx512vnni_prfm,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_5x16c4__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c4__avx512vnni_prfm,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x16c4__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_7x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c4__avx512vnni_prfm,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_7x16c4__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_8x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c4__avx512vnni_prfm,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_8x16c4__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512vnni,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__avx512vnni,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__avx512vnni,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__avx512vnni,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__avx512vnni,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__avx512vnni,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512vnni,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512vnni,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__avx512vnni_prfm,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__avx512vnni_prfm,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__avx512vnni_prfm,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__avx512vnni_prfm,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__avx512vnni_prfm,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512vnni_prfm,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512vnni_prfm,
      xnn_init_f32_minmax_avx512vnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx512vnni,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avx512vnni,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avx512vnni,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avx512vnni,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avx512vnni,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avx512vnni,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avx512vnni,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avx512vnni,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avx512vnni)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx512vnni_prfm,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avx512vnni_prfm,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avx512vnni_prfm,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avx512vnni_prfm,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avx512vnni_prfm,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avx512vnni_prfm,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avx512vnni_prfm,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avx512vnni_prfm,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512VNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avx512vnni_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avxvnni,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avxvnni)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avxvnni,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avxvnni)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avxvnni,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avxvnni)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avxvnni,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avxvnni)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avxvnni,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avxvnni)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avxvnni,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avxvnni)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avxvnni_prfm,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avxvnni_prfm)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avxvnni_prfm,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avxvnni_prfm)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avxvnni_prfm,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avxvnni_prfm)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avxvnni_prfm,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avxvnni_prfm)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avxvnni_prfm,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avxvnni_prfm)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avxvnni_prfm,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avxvnni_prfm)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__avxvnni_u2_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__avxvnni_u2_acc2,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__avxvnni_u2_acc2)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x8c4__avxvnni_u2_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c4__avxvnni_u2_acc2,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x8c4__avxvnni_u2_acc2)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x8c4__avxvnni_u2_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c4__avxvnni_u2_acc2,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x8c4__avxvnni_u2_acc2)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__avxvnni_u2_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__avxvnni_u2_acc2,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__avxvnni_u2_acc2)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_5x8c4__avxvnni_u2_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c4__avxvnni_u2_acc2,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_5x8c4__avxvnni_u2_acc2)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x8c4__avxvnni_u2_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c4__avxvnni_u2_acc2,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x8c4__avxvnni_u2_acc2)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_7x8c4__avxvnni_u2_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c4__avxvnni_u2_acc2,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_7x8c4__avxvnni_u2_acc2)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_8x8c4__avxvnni_u2_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c4__avxvnni_u2_acc2,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_8x8c4__avxvnni_u2_acc2)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__avxvnni_u4_acc4(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__avxvnni_u4_acc4,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__avxvnni_u4_acc4)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x8c4__avxvnni_u4_acc4(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c4__avxvnni_u4_acc4,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x8c4__avxvnni_u4_acc4)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x8c4__avxvnni_u4_acc4(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c4__avxvnni_u4_acc4,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x8c4__avxvnni_u4_acc4)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__avxvnni_u4_acc4(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__avxvnni_u4_acc4,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__avxvnni_u4_acc4)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_5x8c4__avxvnni_u4_acc4(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c4__avxvnni_u4_acc4,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_5x8c4__avxvnni_u4_acc4)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x8c4__avxvnni_u4_acc4(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c4__avxvnni_u4_acc4,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x8c4__avxvnni_u4_acc4)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_7x8c4__avxvnni_u4_acc4(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c4__avxvnni_u4_acc4,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_7x8c4__avxvnni_u4_acc4)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc8w_gemm_minmax_ukernel_8x8c4__avxvnni_u4_acc4(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c4__avxvnni_u4_acc4,
      xnn_init_f32_minmax_avxvnni_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckAVXVNNI);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_8x8c4__avxvnni_u4_acc4)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx512skx,
      xnn_init_f32_minmax_avx_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avx512skx,
      xnn_init_f32_minmax_avx_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avx512skx,
      xnn_init_f32_minmax_avx_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avx512skx,
      xnn_init_f32_minmax_avx_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avx512skx,
      xnn_init_f32_minmax_avx_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avx512skx,
      xnn_init_f32_minmax_avx_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avx512skx,
      xnn_init_f32_minmax_avx_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avx512skx,
      xnn_init_f32_minmax_avx_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512skx,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__avx512skx,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__avx512skx,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__avx512skx,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__avx512skx,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__avx512skx,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512skx,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512skx,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512skx_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512skx_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__avx512skx_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__avx512skx_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__avx512skx_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__avx512skx_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__avx512skx_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__avx512skx_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__avx512skx_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__avx512skx_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__avx512skx_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__avx512skx_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512skx_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512skx_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512skx_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512skx_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__xop_ld64,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__xop_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__xop_ld128,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__xop_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__xop_ld64,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__xop_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__xop_ld128,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__xop_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__xop_ld64,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__xop_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__xop_ld128,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__xop_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__xop_ld64,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__xop_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__xop_ld128,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__xop_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx2,
      xnn_init_f32_minmax_avx_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx2)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avx2,
      xnn_init_f32_minmax_avx_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avx2)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avx2,
      xnn_init_f32_minmax_avx_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avx2)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avx2,
      xnn_init_f32_minmax_avx_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avx2)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__avx_ld64,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__avx_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__avx_ld128,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__avx_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__avx_ld64,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__avx_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__avx_ld128,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__avx_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__avx_ld64,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__avx_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__avx_ld128,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__avx_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__avx_ld64,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__avx_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__avx_ld128,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__avx_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse41_ld64,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse41_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse41_ld128,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse41_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse41_ld64,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse41_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse41_ld128,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse41_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse41_ld64,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse41_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse41_ld128,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse41_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse41_ld64,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse41_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse41_ld128,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse41_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse2_ld64,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse2_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse2_ld128,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse2_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse2_ld64,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse2_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse2_ld128,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse2_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse2_ld64,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse2_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse2_ld128,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse2_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse2_ld64,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse2_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse2_ld128,
      xnn_init_f32_minmax_sse_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse2_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_1x4c16__wasmsdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c16__wasmsdot,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/16, /*sr=*/1,
      benchmark::utils::CheckWAsmSDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x4c16__wasmsdot)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_2x4c16__wasmsdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c16__wasmsdot,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/16, /*sr=*/1,
      benchmark::utils::CheckWAsmSDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x4c16__wasmsdot)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_3x4c16__wasmsdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c16__wasmsdot,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/16, /*sr=*/1,
      benchmark::utils::CheckWAsmSDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_3x4c16__wasmsdot)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc8w_gemm_minmax_ukernel_4x4c16__wasmsdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c16__wasmsdot,
      xnn_init_f32_minmax_wasmsimd_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/16, /*sr=*/1,
      benchmark::utils::CheckWAsmSDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x4c16__wasmsdot)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


static void qd8_f32_qc8w_gemm_minmax_ukernel_1x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x2__scalar,
    xnn_init_f32_minmax_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x2__scalar)

static void qd8_f32_qc8w_gemm_minmax_ukernel_1x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x4__scalar)

static void qd8_f32_qc8w_gemm_minmax_ukernel_1x8__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8__scalar,
    xnn_init_f32_minmax_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_1x8__scalar)

static void qd8_f32_qc8w_gemm_minmax_ukernel_2x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x2__scalar,
    xnn_init_f32_minmax_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x2__scalar)

static void qd8_f32_qc8w_gemm_minmax_ukernel_2x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x4__scalar)

static void qd8_f32_qc8w_gemm_minmax_ukernel_2x8__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8__scalar,
    xnn_init_f32_minmax_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_2x8__scalar)

static void qd8_f32_qc8w_gemm_minmax_ukernel_4x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qd8_f32_qc8w_gemm_minmax_ukernel_4x4__scalar)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
