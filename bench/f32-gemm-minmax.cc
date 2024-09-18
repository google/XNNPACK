// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-gemm-minmax.yaml
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


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  static void f32_gemm_minmax_ukernel_1x32__hvx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x32__hvx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/32, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckHVX);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x32__hvx_broadcast)

  static void f32_gemm_minmax_ukernel_1x64__hvx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x64__hvx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/64, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckHVX);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x64__hvx_broadcast)

  static void f32_gemm_minmax_ukernel_1x128__hvx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x128__hvx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/128, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckHVX);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x128__hvx_broadcast)

  static void f32_gemm_minmax_ukernel_2x128__hvx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_2x128__hvx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/2, /*nr=*/128, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckHVX);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_2x128__hvx_broadcast)

  static void f32_gemm_minmax_ukernel_4x64__hvx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x64__hvx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/64, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckHVX);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x64__hvx_broadcast)

  static void f32_gemm_minmax_ukernel_7x64__hvx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_7x64__hvx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/7, /*nr=*/64, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckHVX);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_7x64__hvx_broadcast)

  static void f32_gemm_minmax_ukernel_8x32__hvx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_8x32__hvx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/8, /*nr=*/32, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckHVX);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_8x32__hvx_broadcast)

  static void f32_gemm_minmax_ukernel_16x32__hvx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_16x32__hvx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/16, /*nr=*/32, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckHVX);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_16x32__hvx_broadcast)
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  static void f32_gemm_minmax_ukernel_1x4v__rvv(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x4v__rvv,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/4 * xnn_init_hardware_config()->vlenb / sizeof(float), /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckRVV);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x4v__rvv)

  static void f32_gemm_minmax_ukernel_7x4v__rvv(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_7x4v__rvv,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/7, /*nr=*/4 * xnn_init_hardware_config()->vlenb / sizeof(float), /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckRVV);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_7x4v__rvv)
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat)

  static void f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat)

  static void f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat)

  static void f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat)

  static void f32_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd)

  static void f32_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma)

  static void f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat)

  static void f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_splat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_splat)

  static void f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_loadsplat)

  static void f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_splat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_splat)

  static void f32_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd)

  static void f32_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd_fma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd_fma,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd_fma)

  static void f32_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/2, /*kr=*/4, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd)

  static void f32_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/2, /*kr=*/4, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma)

  static void f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat)

  static void f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_splat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_splat)

  static void f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_loadsplat)

  static void f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_splat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_splat)

  static void f32_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd)

  static void f32_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd_fma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd_fma,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd_fma)

  static void f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat)

  static void f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_splat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_splat)

  static void f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_loadsplat)

  static void f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_splat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_splat)

  static void f32_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd)

  static void f32_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd_fma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd_fma,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd_fma)

  static void f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat)

  static void f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat)

  static void f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_loadsplat)

  static void f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_splat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_splat)

  static void f32_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd)

  static void f32_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd_fma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd_fma,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd_fma)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat)

  static void f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat)

  static void f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat)

  static void f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat)

  static void f32_gemm_minmax_ukernel_1x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8s4__wasmsimd_arm)

  static void f32_gemm_minmax_ukernel_1x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8s4__wasmsimd_x86)

  static void f32_gemm_minmax_ukernel_3x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_3x8__wasmsimd_arm_loadsplat)

  static void f32_gemm_minmax_ukernel_3x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_3x8__wasmsimd_arm_splat)

  static void f32_gemm_minmax_ukernel_3x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_3x8__wasmsimd_x86_loadsplat)

  static void f32_gemm_minmax_ukernel_3x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_3x8__wasmsimd_x86_splat)

  static void f32_gemm_minmax_ukernel_3x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_3x8s4__wasmsimd_arm)

  static void f32_gemm_minmax_ukernel_3x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_3x8s4__wasmsimd_x86)

  static void f32_gemm_minmax_ukernel_4x2c4__wasmsimd_arm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x2c4__wasmsimd_arm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/2, /*kr=*/4, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x2c4__wasmsimd_arm)

  static void f32_gemm_minmax_ukernel_4x2c4__wasmsimd_x86(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x2c4__wasmsimd_x86,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/2, /*kr=*/4, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x2c4__wasmsimd_x86)

  static void f32_gemm_minmax_ukernel_4x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__wasmsimd_arm_loadsplat)

  static void f32_gemm_minmax_ukernel_4x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__wasmsimd_arm_splat)

  static void f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat)

  static void f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_splat)

  static void f32_gemm_minmax_ukernel_4x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8s4__wasmsimd_arm)

  static void f32_gemm_minmax_ukernel_4x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8s4__wasmsimd_x86)

  static void f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_loadsplat)

  static void f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_splat)

  static void f32_gemm_minmax_ukernel_5x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x8__wasmsimd_x86_loadsplat)

  static void f32_gemm_minmax_ukernel_5x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x8__wasmsimd_x86_splat)

  static void f32_gemm_minmax_ukernel_5x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x8s4__wasmsimd_arm)

  static void f32_gemm_minmax_ukernel_5x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x8s4__wasmsimd_x86)

  static void f32_gemm_minmax_ukernel_6x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__wasmsimd_arm_loadsplat)

  static void f32_gemm_minmax_ukernel_6x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__wasmsimd_arm_splat)

  static void f32_gemm_minmax_ukernel_6x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__wasmsimd_x86_loadsplat)

  static void f32_gemm_minmax_ukernel_6x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__wasmsimd_x86_splat)

  static void f32_gemm_minmax_ukernel_6x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8s4__wasmsimd_arm)

  static void f32_gemm_minmax_ukernel_6x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8s4__wasmsimd_x86)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_gemm_minmax_ukernel_1x4__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x4__wasm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x4__wasm)

  static void f32_gemm_minmax_ukernel_2x4__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_2x4__wasm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_2x4__wasm)

  static void f32_gemm_minmax_ukernel_4x2__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x2__wasm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x2__wasm)

  static void f32_gemm_minmax_ukernel_4x4__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x4__wasm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x4__wasm)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53)

  static void f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm)

  static void f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7)

  static void f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53)

  static void f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53_prfm)

  static void f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55)

  static void f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75)

  static void f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75_prfm)

  static void f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_ld64)
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void f32_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2)

  static void f32_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_gemm_minmax_ukernel_1x16__neon_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x16__neon_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x16__neon_lane_ld128)

  static void f32_gemm_minmax_ukernel_2x16__neon_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_2x16__neon_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_2x16__neon_lane_ld128)

  static void f32_gemm_minmax_ukernel_3x16__neon_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x16__neon_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_3x16__neon_lane_ld128)

  static void f32_gemm_minmax_ukernel_4x16__neon_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x16__neon_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x16__neon_lane_ld128)

  static void f32_gemm_minmax_ukernel_5x16__neon_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x16__neon_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x16__neon_lane_ld128)

  static void f32_gemm_minmax_ukernel_6x16__neon_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x16__neon_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x16__neon_lane_ld128)

  static void f32_gemm_minmax_ukernel_1x8__neon_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__neon_lane_ld128)

  static void f32_gemm_minmax_ukernel_4x8__neon_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__neon_lane_ld128)

  static void f32_gemm_minmax_ukernel_6x8__neon_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__neon_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__neon_lane_ld128)

  static void f32_gemm_minmax_ukernel_1x8__neon_dup_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__neon_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__neon_dup_ld64)

  static void f32_gemm_minmax_ukernel_1x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__neon_lane_ld64)

  static void f32_gemm_minmax_ukernel_1x8s4__neon(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8s4__neon,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8s4__neon)

  static void f32_gemm_minmax_ukernel_4x2__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x2__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x2__neon_lane_ld64)

  static void f32_gemm_minmax_ukernel_4x8__neon_dup_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__neon_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__neon_dup_ld64)

  static void f32_gemm_minmax_ukernel_4x8__neon_dup_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__neon_dup_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__neon_dup_ld128)

  static void f32_gemm_minmax_ukernel_4x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__neon_lane_ld64)

  static void f32_gemm_minmax_ukernel_4x8s4__neon(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8s4__neon,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8s4__neon)

  static void f32_gemm_minmax_ukernel_5x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x8__neon_lane_ld64)

  static void f32_gemm_minmax_ukernel_6x2__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x2__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x2__neon_lane_ld64)

  static void f32_gemm_minmax_ukernel_6x8__neon_dup_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__neon_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__neon_dup_ld64)

  static void f32_gemm_minmax_ukernel_6x8__neon_dup_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__neon_dup_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__neon_dup_ld128)

  static void f32_gemm_minmax_ukernel_6x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__neon_lane_ld64)

  static void f32_gemm_minmax_ukernel_6x8s4__neon(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8s4__neon,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8s4__neon)

  static void f32_gemm_minmax_ukernel_8x8s4__neon(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_8x8s4__neon,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_8x8s4__neon)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_gemm_minmax_ukernel_1x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x16__avx512f_broadcast)

  static void f32_gemm_minmax_ukernel_4x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x16__avx512f_broadcast)

  static void f32_gemm_minmax_ukernel_5x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x16__avx512f_broadcast)

  static void f32_gemm_minmax_ukernel_6x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x16__avx512f_broadcast)

  static void f32_gemm_minmax_ukernel_7x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_7x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_7x16__avx512f_broadcast)

  static void f32_gemm_minmax_ukernel_8x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_8x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_8x16__avx512f_broadcast)

  static void f32_gemm_minmax_ukernel_1x8__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__avx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__avx_broadcast)

  static void f32_gemm_minmax_ukernel_1x16__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x16__avx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x16__avx_broadcast)

  static void f32_gemm_minmax_ukernel_3x16__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x16__avx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_3x16__avx_broadcast)

  static void f32_gemm_minmax_ukernel_4x8__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__avx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__avx_broadcast)

  static void f32_gemm_minmax_ukernel_4x16__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x16__avx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x16__avx_broadcast)

  static void f32_gemm_minmax_ukernel_5x8__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__avx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x8__avx_broadcast)

  static void f32_gemm_minmax_ukernel_5x16__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x16__avx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x16__avx_broadcast)

  static void f32_gemm_minmax_ukernel_6x8__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__avx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__avx_broadcast)

  static void f32_gemm_minmax_ukernel_6x16__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x16__avx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x16__avx_broadcast)

  static void f32_gemm_minmax_ukernel_7x8__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_7x8__avx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_7x8__avx_broadcast)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void f32_gemm_minmax_ukernel_4x4__asm_aarch32_vfp_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x4__asm_aarch32_vfp_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x4__asm_aarch32_vfp_ld64)
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53)

  static void f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm)

  static void f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75)

  static void f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm)

  static void f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64)

  static void f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2)

  static void f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm)

  static void f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4)

  static void f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm)

  static void f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm)

  static void f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128)

  static void f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2)

  static void f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm)

  static void f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4)

  static void f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm)

  static void f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm)

  static void f32_gemm_minmax_ukernel_1x12__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x12__asm_aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/12, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x12__asm_aarch64_neonfma_cortex_a53)

  static void f32_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/1, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld64)

  static void f32_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/1, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld128)

  static void f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75)

  static void f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75_prfm)

  static void f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64)

  static void f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld128)

  static void f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53)

  static void f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm)

  static void f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55)

  static void f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75)

  static void f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm)

  static void f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64)

  static void f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128)

  static void f32_gemm_minmax_ukernel_4x12__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x12__asm_aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/12, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x12__asm_aarch64_neonfma_cortex_a53)

  static void f32_gemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75)

  static void f32_gemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75_prfm)

  static void f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53)

  static void f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53_prfm)

  static void f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55)

  static void f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73)

  static void f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75)

  static void f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75_prfm)

  static void f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64)

  static void f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64
  static void f32_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128)

  static void f32_gemm_minmax_ukernel_2x16__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_2x16__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_2x16__aarch64_neonfma_lane_ld128)

  static void f32_gemm_minmax_ukernel_3x16__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x16__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_3x16__aarch64_neonfma_lane_ld128)

  static void f32_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128)

  static void f32_gemm_minmax_ukernel_5x16__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x16__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x16__aarch64_neonfma_lane_ld128)

  static void f32_gemm_minmax_ukernel_6x16__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x16__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x16__aarch64_neonfma_lane_ld128)

  static void f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128)

  static void f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128)

  static void f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128)

  static void f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64)
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_gemm_minmax_ukernel_1x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__neonfma_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__neonfma_dup_ld64)

  static void f32_gemm_minmax_ukernel_1x8s4__neonfma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8s4__neonfma,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8s4__neonfma)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  static void f32_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64)

  static void f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64)
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_gemm_minmax_ukernel_4x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__neonfma_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__neonfma_dup_ld64)

  static void f32_gemm_minmax_ukernel_4x8__neonfma_dup_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__neonfma_dup_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__neonfma_dup_ld128)

  static void f32_gemm_minmax_ukernel_4x8s4__neonfma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8s4__neonfma,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8s4__neonfma)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  static void f32_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64)

  static void f32_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64)

  static void f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64)
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_gemm_minmax_ukernel_6x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__neonfma_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__neonfma_dup_ld64)

  static void f32_gemm_minmax_ukernel_6x8__neonfma_dup_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__neonfma_dup_ld128,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__neonfma_dup_ld128)

  static void f32_gemm_minmax_ukernel_6x8s4__neonfma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8s4__neonfma,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8s4__neonfma)

  static void f32_gemm_minmax_ukernel_8x8s4__neonfma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_8x8s4__neonfma,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_8x8s4__neonfma)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_gemm_minmax_ukernel_1x8__sse_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__sse_dup,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__sse_dup)

  static void f32_gemm_minmax_ukernel_1x8__sse_load1(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__sse_load1,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__sse_load1)

  static void f32_gemm_minmax_ukernel_1x8s4__sse(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8s4__sse,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8s4__sse)

  static void f32_gemm_minmax_ukernel_3x8__sse_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8__sse_dup,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_3x8__sse_dup)

  static void f32_gemm_minmax_ukernel_3x8__sse_load1(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8__sse_load1,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_3x8__sse_load1)

  static void f32_gemm_minmax_ukernel_3x8s4__sse(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8s4__sse,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_3x8s4__sse)

  static void f32_gemm_minmax_ukernel_4x2c4__sse(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x2c4__sse,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/2, /*kr=*/4, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x2c4__sse)

  static void f32_gemm_minmax_ukernel_4x8__sse_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__sse_dup,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__sse_dup)

  static void f32_gemm_minmax_ukernel_4x8__sse_load1(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__sse_load1,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__sse_load1)

  static void f32_gemm_minmax_ukernel_4x8s4__sse(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8s4__sse,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8s4__sse)

  static void f32_gemm_minmax_ukernel_5x8__sse_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__sse_dup,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x8__sse_dup)

  static void f32_gemm_minmax_ukernel_5x8__sse_load1(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__sse_load1,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x8__sse_load1)

  static void f32_gemm_minmax_ukernel_5x8s4__sse(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8s4__sse,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x8s4__sse)

  static void f32_gemm_minmax_ukernel_6x2c4__sse(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x2c4__sse,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/2, /*kr=*/4, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x2c4__sse)

  static void f32_gemm_minmax_ukernel_6x8__sse_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__sse_dup,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__sse_dup)

  static void f32_gemm_minmax_ukernel_6x8__sse_load1(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__sse_load1,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__sse_load1)

  static void f32_gemm_minmax_ukernel_6x8s4__sse(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8s4__sse,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8s4__sse)

  static void f32_gemm_minmax_ukernel_1x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x8__fma3_broadcast)

  static void f32_gemm_minmax_ukernel_1x16__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x16__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x16__fma3_broadcast)

  static void f32_gemm_minmax_ukernel_1x16s4__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x16s4__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckFMA3);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x16s4__fma3_broadcast)

  static void f32_gemm_minmax_ukernel_3x16__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x16__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_3x16__fma3_broadcast)

  static void f32_gemm_minmax_ukernel_3x16s4__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x16s4__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckFMA3);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_3x16s4__fma3_broadcast)

  static void f32_gemm_minmax_ukernel_4x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x8__fma3_broadcast)

  static void f32_gemm_minmax_ukernel_4x16__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x16__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x16__fma3_broadcast)

  static void f32_gemm_minmax_ukernel_4x16s4__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x16s4__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckFMA3);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x16s4__fma3_broadcast)

  static void f32_gemm_minmax_ukernel_5x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x8__fma3_broadcast)

  static void f32_gemm_minmax_ukernel_5x16__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x16__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x16__fma3_broadcast)

  static void f32_gemm_minmax_ukernel_5x16__fma3_broadcast_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x16__fma3_broadcast_prfm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x16__fma3_broadcast_prfm)

  static void f32_gemm_minmax_ukernel_5x16s4__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x16s4__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckFMA3);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_5x16s4__fma3_broadcast)

  static void f32_gemm_minmax_ukernel_6x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x8__fma3_broadcast)

  static void f32_gemm_minmax_ukernel_6x16__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x16__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x16__fma3_broadcast)

  static void f32_gemm_minmax_ukernel_6x16s4__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x16s4__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckFMA3);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_6x16s4__fma3_broadcast)

  static void f32_gemm_minmax_ukernel_7x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_7x8__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_7x8__fma3_broadcast)

  static void f32_gemm_minmax_ukernel_8x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_8x8__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_f32_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }

  BENCHMARK_GEMM(f32_gemm_minmax_ukernel_8x8__fma3_broadcast)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


static void f32_gemm_minmax_ukernel_1x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_f32_gemm_minmax_ukernel_1x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    xnn_pack_f32_gemm_goi_w,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(f32_gemm_minmax_ukernel_1x4__scalar)

static void f32_gemm_minmax_ukernel_2x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_f32_gemm_minmax_ukernel_2x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    xnn_pack_f32_gemm_goi_w,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(f32_gemm_minmax_ukernel_2x4__scalar)

static void f32_gemm_minmax_ukernel_4x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_f32_gemm_minmax_ukernel_4x2__scalar,
    xnn_init_f32_minmax_scalar_params,
    xnn_pack_f32_gemm_goi_w,
    /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x2__scalar)

static void f32_gemm_minmax_ukernel_4x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_f32_gemm_minmax_ukernel_4x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    xnn_pack_f32_gemm_goi_w,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(f32_gemm_minmax_ukernel_4x4__scalar)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
