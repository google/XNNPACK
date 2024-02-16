// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qu8-gemm-minmax-fp32.yaml
//   Generator: tools/generate-gemm-test.py

#include <benchmark/benchmark.h>
#include "bench/gemm-benchmark.h"
#include "bench/utils.h"

#include <xnnpack/isa-checks.h>
#include <xnnpack/gemm.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic,
      xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic,
      xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic,
      xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_2x4__wasm_fmagic(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4__wasm_fmagic,
      xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4__wasm_fmagic)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_3x2__wasm_fmagic(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x2__wasm_fmagic,
      xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x2__wasm_fmagic)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_3x4__wasm_fmagic(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4__wasm_fmagic,
      xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4__wasm_fmagic)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_4x2__wasm_fmagic(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x2__wasm_fmagic,
      xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x2__wasm_fmagic)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_4x4__wasm_fmagic(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4__wasm_fmagic,
      xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4__wasm_fmagic)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qu8_gemm_minmax_fp32_ukernel_1x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x8c8__neoni8mm,
      xnn_init_qu8_conv_minmax_fp32_neonv8_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x8c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qu8_gemm_minmax_fp32_ukernel_1x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x16c8__neoni8mm,
      xnn_init_qu8_conv_minmax_fp32_neonv8_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x16c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qu8_gemm_minmax_fp32_ukernel_2x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x8c8__neoni8mm,
      xnn_init_qu8_conv_minmax_fp32_neonv8_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x8c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qu8_gemm_minmax_fp32_ukernel_2x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x16c8__neoni8mm,
      xnn_init_qu8_conv_minmax_fp32_neonv8_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x16c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qu8_gemm_minmax_fp32_ukernel_3x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x8c8__neoni8mm,
      xnn_init_qu8_conv_minmax_fp32_neonv8_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x8c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qu8_gemm_minmax_fp32_ukernel_3x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x16c8__neoni8mm,
      xnn_init_qu8_conv_minmax_fp32_neonv8_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x16c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qu8_gemm_minmax_fp32_ukernel_4x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x8c8__neoni8mm,
      xnn_init_qu8_conv_minmax_fp32_neonv8_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x8c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qu8_gemm_minmax_fp32_ukernel_4x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x16c8__neoni8mm,
      xnn_init_qu8_conv_minmax_fp32_neonv8_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x16c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qu8_gemm_minmax_fp32_ukernel_6x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_6x8c8__neoni8mm,
      xnn_init_qu8_conv_minmax_fp32_neonv8_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_6x8c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qu8_gemm_minmax_fp32_ukernel_6x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_6x16c8__neoni8mm,
      xnn_init_qu8_conv_minmax_fp32_neonv8_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_6x16c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qu8_gemm_minmax_fp32_ukernel_8x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_8x8c8__neoni8mm,
      xnn_init_qu8_conv_minmax_fp32_neonv8_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_8x8c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qu8_gemm_minmax_fp32_ukernel_8x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_8x16c8__neoni8mm,
      xnn_init_qu8_conv_minmax_fp32_neonv8_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_8x16c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qu8_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55,
      xnn_init_qu8_conv_minmax_fp32_neonv8_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qu8_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128,
      xnn_init_qu8_conv_minmax_fp32_neonv8_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qu8_gemm_minmax_fp32_ukernel_1x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x16c4__neondot,
      xnn_init_qu8_conv_minmax_fp32_neonv8_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qu8_gemm_minmax_fp32_ukernel_2x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x16c4__neondot,
      xnn_init_qu8_conv_minmax_fp32_neonv8_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qu8_gemm_minmax_fp32_ukernel_4x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x16c4__neondot,
      xnn_init_qu8_conv_minmax_fp32_neonv8_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qu8_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane,
      xnn_init_qu8_conv_minmax_fp32_neonv8_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONV8);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qu8_gemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane,
      xnn_init_qu8_conv_minmax_fp32_neonv8_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONV8);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qu8_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_fp32_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qu8_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_fp32_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qu8_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_fp32_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qu8_gemm_minmax_fp32_ukernel_4x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x16__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_fp32_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x16__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM
  static void qu8_gemm_minmax_fp32_ukernel_1x1c4__armsimd32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x1c4__armsimd32,
      xnn_init_qu8_conv_minmax_fp32_armsimd32_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/1, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckARMV6);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x1c4__armsimd32)
#endif  // XNN_ARCH_ARM


#if XNN_ARCH_ARM
  static void qu8_gemm_minmax_fp32_ukernel_1x2c4__armsimd32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x2c4__armsimd32,
      xnn_init_qu8_conv_minmax_fp32_armsimd32_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/2, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckARMV6);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x2c4__armsimd32)
#endif  // XNN_ARCH_ARM


#if XNN_ARCH_ARM
  static void qu8_gemm_minmax_fp32_ukernel_2x1c4__armsimd32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x1c4__armsimd32,
      xnn_init_qu8_conv_minmax_fp32_armsimd32_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/1, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckARMV6);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x1c4__armsimd32)
#endif  // XNN_ARCH_ARM


#if XNN_ARCH_ARM
  static void qu8_gemm_minmax_fp32_ukernel_2x2c4__armsimd32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x2c4__armsimd32,
      xnn_init_qu8_conv_minmax_fp32_armsimd32_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/2, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckARMV6);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x2c4__armsimd32)
#endif  // XNN_ARCH_ARM


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x8c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x8c8__avx512skx,
      xnn_init_qu8_conv_minmax_fp32_avx2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x8c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x8c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x8c8__avx512skx,
      xnn_init_qu8_conv_minmax_fp32_avx2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x8c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x8c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x8c8__avx512skx,
      xnn_init_qu8_conv_minmax_fp32_avx2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x8c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_4x8c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x8c8__avx512skx,
      xnn_init_qu8_conv_minmax_fp32_avx2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x8c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_init_qu8_conv_minmax_fp32_avx512_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx,
      xnn_init_qu8_conv_minmax_fp32_avx512_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx,
      xnn_init_qu8_conv_minmax_fp32_avx512_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx,
      xnn_init_qu8_conv_minmax_fp32_avx512_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_5x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_5x16c8__avx512skx,
      xnn_init_qu8_conv_minmax_fp32_avx512_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_5x16c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_6x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_6x16c8__avx512skx,
      xnn_init_qu8_conv_minmax_fp32_avx512_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_6x16c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_7x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_7x16c8__avx512skx,
      xnn_init_qu8_conv_minmax_fp32_avx512_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_7x16c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_8x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_8x16c8__avx512skx,
      xnn_init_qu8_conv_minmax_fp32_avx512_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_8x16c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm,
      xnn_init_qu8_conv_minmax_fp32_avx512_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx_prfm,
      xnn_init_qu8_conv_minmax_fp32_avx512_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx_prfm,
      xnn_init_qu8_conv_minmax_fp32_avx512_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx_prfm,
      xnn_init_qu8_conv_minmax_fp32_avx512_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_5x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_5x16c8__avx512skx_prfm,
      xnn_init_qu8_conv_minmax_fp32_avx512_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_5x16c8__avx512skx_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_6x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_6x16c8__avx512skx_prfm,
      xnn_init_qu8_conv_minmax_fp32_avx512_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_6x16c8__avx512skx_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm,
      xnn_init_qu8_conv_minmax_fp32_avx512_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_8x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_8x16c8__avx512skx_prfm,
      xnn_init_qu8_conv_minmax_fp32_avx512_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX512SKX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_8x16c8__avx512skx_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c2__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__xop_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2__xop_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c2__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__xop_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2__xop_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_4x4c2__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__xop_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2__xop_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c2__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__xop_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2__xop_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c2__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__xop_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2__xop_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_4x4c2__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__xop_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2__xop_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c2s4__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__xop_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2s4__xop_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c2s4__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__xop_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2s4__xop_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_4x4c2s4__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__xop_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2s4__xop_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c2s4__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__xop_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2s4__xop_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c2s4__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__xop_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2s4__xop_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_4x4c2s4__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__xop_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2s4__xop_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x8c8__avx2,
      xnn_init_qu8_conv_minmax_fp32_avx2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x8c8__avx2)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x8c8__avx2,
      xnn_init_qu8_conv_minmax_fp32_avx2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x8c8__avx2)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x8c8__avx2,
      xnn_init_qu8_conv_minmax_fp32_avx2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x8c8__avx2)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_4x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x8c8__avx2,
      xnn_init_qu8_conv_minmax_fp32_avx2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x8c8__avx2)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c2__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2__avx_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_4x4c2__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2__avx_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c2__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2__avx_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c2__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2__avx_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_4x4c2__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2__avx_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c8__avx_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


static void qu8_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic)

static void qu8_gemm_minmax_fp32_ukernel_1x2__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x2__scalar_imagic)

static void qu8_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf)

static void qu8_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic)

static void qu8_gemm_minmax_fp32_ukernel_1x4__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4__scalar_imagic)

static void qu8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf)

static void qu8_gemm_minmax_fp32_ukernel_2x2__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_2x2__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x2__scalar_fmagic)

static void qu8_gemm_minmax_fp32_ukernel_2x2__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_2x2__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x2__scalar_imagic)

static void qu8_gemm_minmax_fp32_ukernel_2x2__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_2x2__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x2__scalar_lrintf)

static void qu8_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic)

static void qu8_gemm_minmax_fp32_ukernel_2x4__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4__scalar_imagic)

static void qu8_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf)

static void qu8_gemm_minmax_fp32_ukernel_3x2__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_3x2__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/3, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x2__scalar_fmagic)

static void qu8_gemm_minmax_fp32_ukernel_3x2__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_3x2__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/3, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x2__scalar_imagic)

static void qu8_gemm_minmax_fp32_ukernel_3x2__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_3x2__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/3, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x2__scalar_lrintf)

static void qu8_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/3, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic)

static void qu8_gemm_minmax_fp32_ukernel_3x4__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/3, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4__scalar_imagic)

static void qu8_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/3, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf)

static void qu8_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic)

static void qu8_gemm_minmax_fp32_ukernel_4x2__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_4x2__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x2__scalar_imagic)

static void qu8_gemm_minmax_fp32_ukernel_4x2__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_4x2__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x2__scalar_lrintf)

static void qu8_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic)

static void qu8_gemm_minmax_fp32_ukernel_4x4__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4__scalar_imagic)

static void qu8_gemm_minmax_fp32_ukernel_4x4__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4__scalar_lrintf)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
