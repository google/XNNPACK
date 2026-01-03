// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qd8-f32-qc2w-gemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py

#include <cstdint>
#include <functional>

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



#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc2w_gemm_minmax_ukernel_1x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc2w_gemm_minmax_ukernel_1x8c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_1x8c4__neondot)

  static void qd8_f32_qc2w_gemm_minmax_ukernel_2x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc2w_gemm_minmax_ukernel_2x8c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_2x8c4__neondot)

  static void qd8_f32_qc2w_gemm_minmax_ukernel_3x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc2w_gemm_minmax_ukernel_3x8c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_3x8c4__neondot)

  static void qd8_f32_qc2w_gemm_minmax_ukernel_4x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc2w_gemm_minmax_ukernel_4x8c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_4x8c4__neondot)

  static void qd8_f32_qc2w_gemm_minmax_ukernel_5x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc2w_gemm_minmax_ukernel_5x8c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_5x8c4__neondot)

  static void qd8_f32_qc2w_gemm_minmax_ukernel_6x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc2w_gemm_minmax_ukernel_6x8c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_6x8c4__neondot)

  static void qd8_f32_qc2w_gemm_minmax_ukernel_7x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc2w_gemm_minmax_ukernel_7x8c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_7x8c4__neondot)

  static void qd8_f32_qc2w_gemm_minmax_ukernel_8x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc2w_gemm_minmax_ukernel_8x8c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_8x8c4__neondot)

  static void qd8_f32_qc2w_gemm_minmax_ukernel_1x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc2w_gemm_minmax_ukernel_1x16c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_1x16c4__neondot)

  static void qd8_f32_qc2w_gemm_minmax_ukernel_2x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc2w_gemm_minmax_ukernel_2x16c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_2x16c4__neondot)

  static void qd8_f32_qc2w_gemm_minmax_ukernel_3x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc2w_gemm_minmax_ukernel_3x16c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_3x16c4__neondot)

  static void qd8_f32_qc2w_gemm_minmax_ukernel_4x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc2w_gemm_minmax_ukernel_4x16c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_4x16c4__neondot)

  static void qd8_f32_qc2w_gemm_minmax_ukernel_5x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc2w_gemm_minmax_ukernel_5x16c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_5x16c4__neondot)

  static void qd8_f32_qc2w_gemm_minmax_ukernel_6x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc2w_gemm_minmax_ukernel_6x16c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_6x16c4__neondot)

  static void qd8_f32_qc2w_gemm_minmax_ukernel_7x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc2w_gemm_minmax_ukernel_7x16c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_7x16c4__neondot)

  static void qd8_f32_qc2w_gemm_minmax_ukernel_8x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc2w_gemm_minmax_ukernel_8x16c4__neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_qd8_qc2w_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_8x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


static void qd8_f32_qc2w_gemm_minmax_ukernel_1x1__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc2w_gemm_minmax_ukernel_1x1__scalar,
    xnn_init_f32_minmax_scalar_params,
    xnn_pack_qd8_qc2w_gemm_goi_w,
    /*mr=*/1, /*nr=*/1, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_1x1__scalar)

static void qd8_f32_qc2w_gemm_minmax_ukernel_1x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc2w_gemm_minmax_ukernel_1x2__scalar,
    xnn_init_f32_minmax_scalar_params,
    xnn_pack_qd8_qc2w_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_1x2__scalar)

static void qd8_f32_qc2w_gemm_minmax_ukernel_1x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc2w_gemm_minmax_ukernel_1x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    xnn_pack_qd8_qc2w_gemm_goi_w,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_1x4__scalar)

static void qd8_f32_qc2w_gemm_minmax_ukernel_1x8__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc2w_gemm_minmax_ukernel_1x8__scalar,
    xnn_init_f32_minmax_scalar_params,
    xnn_pack_qd8_qc2w_gemm_goi_w,
    /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_1x8__scalar)

static void qd8_f32_qc2w_gemm_minmax_ukernel_2x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc2w_gemm_minmax_ukernel_2x2__scalar,
    xnn_init_f32_minmax_scalar_params,
    xnn_pack_qd8_qc2w_gemm_goi_w,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_2x2__scalar)

static void qd8_f32_qc2w_gemm_minmax_ukernel_2x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc2w_gemm_minmax_ukernel_2x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    xnn_pack_qd8_qc2w_gemm_goi_w,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_2x4__scalar)

static void qd8_f32_qc2w_gemm_minmax_ukernel_2x8__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc2w_gemm_minmax_ukernel_2x8__scalar,
    xnn_init_f32_minmax_scalar_params,
    xnn_pack_qd8_qc2w_gemm_goi_w,
    /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_2x8__scalar)

static void qd8_f32_qc2w_gemm_minmax_ukernel_4x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc2w_gemm_minmax_ukernel_4x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    xnn_pack_qd8_qc2w_gemm_goi_w,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_f32_qc2w_gemm_minmax_ukernel_4x4__scalar)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
