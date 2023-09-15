// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qd8-f32-qc4w-gemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py

#include <benchmark/benchmark.h>
#include "bench/gemm-benchmark.h"
#include "bench/utils.h"

#include <xnnpack/isa-checks.h>
#include <xnnpack/gemm.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__neondot,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c4__neondot,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x8c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c4__neondot,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x8c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c4__neondot,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x8c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__neondot,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__neondot,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__neondot,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__neondot,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16c4__neondot,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__aarch64_neondot_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__aarch64_neondot_ld128,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__aarch64_neondot_ld128)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__neondot_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__neondot_ld64,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__neondot_ld64)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__aarch64_neondot_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__aarch64_neondot_ld128,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__aarch64_neondot_ld128)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__neondot_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__neondot_ld64,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__neondot_ld64)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16__neon_mlal_lane,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16__neon_mlal_lane,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x16__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16__neon_mlal_lane,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x16__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16__neon_mlal_lane,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x16__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16__neon_mlal_lane,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x16__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


static void qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    /*mr=*/1, /*nr=*/1, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar)

static void qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar)

static void qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar)

static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar)

static void qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar)

static void qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar)

static void qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar)

static void qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar)

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
      /*isa_check=*/nullptr);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
