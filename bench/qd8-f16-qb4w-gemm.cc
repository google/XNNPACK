// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qd8-f16-qb4w-gemm-minmax.yaml
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
  static void qd8_f16_qb4w_gemm_minmax_ukernel_1x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_1x8c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_1x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x16c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_1x16c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_1x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x32c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_1x32c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_2x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x8c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_2x8c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_2x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x16c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_2x16c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_2x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x32c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_2x32c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_3x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x8c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_3x8c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_3x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x16c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_3x16c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_3x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x32c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_3x32c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_4x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x8c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_4x8c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_4x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x16c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_4x16c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_4x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x32c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_4x32c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_5x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_5x8c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_5x8c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_5x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_5x16c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_5x16c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_5x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_5x32c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_5x32c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_6x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x8c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_6x8c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_6x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x16c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_6x16c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_6x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x32c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_6x32c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_7x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_7x8c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_7x8c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_7x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_7x16c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_7x16c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_7x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_7x32c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_7x32c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_8x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_8x8c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_8x8c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_8x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_8x16c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_8x16c8__neoni8mm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_8x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_8x32c8__neoni8mm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_8x32c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f16_qb4w_gemm_minmax_ukernel_1x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8c8__avx2,
      xnn_init_f16_qb4w_minmax_avx_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_1x8c8__avx2)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_2x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x8c8__avx2,
      xnn_init_f16_qb4w_minmax_avx_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_2x8c8__avx2)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_3x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x8c8__avx2,
      xnn_init_f16_qb4w_minmax_avx_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_3x8c8__avx2)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_4x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x8c8__avx2,
      xnn_init_f16_qb4w_minmax_avx_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_4x8c8__avx2)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


static void qd8_f16_qb4w_gemm_minmax_ukernel_1x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x2__scalar,
    xnn_init_f16_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_1x2__scalar)

static void qd8_f16_qb4w_gemm_minmax_ukernel_1x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x4__scalar,
    xnn_init_f16_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_1x4__scalar)

static void qd8_f16_qb4w_gemm_minmax_ukernel_1x8__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8__scalar,
    xnn_init_f16_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_1x8__scalar)

static void qd8_f16_qb4w_gemm_minmax_ukernel_2x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x2__scalar,
    xnn_init_f16_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_2x2__scalar)

static void qd8_f16_qb4w_gemm_minmax_ukernel_2x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x4__scalar,
    xnn_init_f16_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_2x4__scalar)

static void qd8_f16_qb4w_gemm_minmax_ukernel_2x8__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x8__scalar,
    xnn_init_f16_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_2x8__scalar)

static void qd8_f16_qb4w_gemm_minmax_ukernel_4x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x4__scalar,
    xnn_init_f16_qb4w_minmax_scalar_params,
    xnn_pack_qs8_qb4w_gemm_goi_w,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_4x4__scalar)

#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f16_qb4w_gemm_minmax_ukernel_1x8c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8c4__neondotfp16arith,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_1x8c4__neondotfp16arith)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_1x16c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x16c4__neondotfp16arith,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_1x16c4__neondotfp16arith)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_2x8c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x8c4__neondotfp16arith,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_2x8c4__neondotfp16arith)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_2x16c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x16c4__neondotfp16arith,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_2x16c4__neondotfp16arith)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_3x8c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x8c4__neondotfp16arith,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_3x8c4__neondotfp16arith)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_4x8c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x8c4__neondotfp16arith,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_4x8c4__neondotfp16arith)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_4x16c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x16c4__neondotfp16arith,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_4x16c4__neondotfp16arith)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_5x8c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_5x8c4__neondotfp16arith,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_5x8c4__neondotfp16arith)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f16_qb4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_2x16__neonfp16arith_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x16__neonfp16arith_mlal_lane,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_2x16__neonfp16arith_mlal_lane)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_3x16__neonfp16arith_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x16__neonfp16arith_mlal_lane,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_3x16__neonfp16arith_mlal_lane)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_4x16__neonfp16arith_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x16__neonfp16arith_mlal_lane,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_4x16__neonfp16arith_mlal_lane)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane_prfm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane_prfm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_2x16__neonfp16arith_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x16__neonfp16arith_mlal_lane_prfm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_2x16__neonfp16arith_mlal_lane_prfm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_3x16__neonfp16arith_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x16__neonfp16arith_mlal_lane_prfm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_3x16__neonfp16arith_mlal_lane_prfm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_4x16__neonfp16arith_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x16__neonfp16arith_mlal_lane_prfm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_4x16__neonfp16arith_mlal_lane_prfm)

  static void qd8_f16_qb4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane_prfm,
      xnn_init_f16_qb4w_minmax_scalar_params,
      xnn_pack_qs8_qb4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM_BL(qd8_f16_qb4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane_prfm)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
