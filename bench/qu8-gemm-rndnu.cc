// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qu8-gemm-minmax-rndnu.yaml
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


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7)

  static void qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm)

  static void qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7)

  static void qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm)

  static void qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53)

  static void qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm)

  static void qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64)

  static void qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm)
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53)

  static void qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm)

  static void qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75)

  static void qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75_prfm,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75_prfm)

  static void qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64)

  static void qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64_prfm,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64_prfm)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane)

  static void qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane)

  static void qu8_gemm_minmax_rndnu_ukernel_2x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_2x8__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_2x8__neon_mlal_lane)

  static void qu8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane)

  static void qu8_gemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane)

  static void qu8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane)

  static void qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane)

  static void qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane)

  static void qu8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane)

  static void qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


static void qu8_gemm_minmax_rndnu_ukernel_1x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_rndnu_ukernel_1x2__scalar,
    xnn_init_qu8_conv_minmax_rndnu_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_1x2__scalar)

static void qu8_gemm_minmax_rndnu_ukernel_1x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_rndnu_ukernel_1x4__scalar,
    xnn_init_qu8_conv_minmax_rndnu_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_1x4__scalar)

static void qu8_gemm_minmax_rndnu_ukernel_2x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_rndnu_ukernel_2x2__scalar,
    xnn_init_qu8_conv_minmax_rndnu_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_2x2__scalar)

static void qu8_gemm_minmax_rndnu_ukernel_2x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_rndnu_ukernel_2x4__scalar,
    xnn_init_qu8_conv_minmax_rndnu_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_2x4__scalar)

static void qu8_gemm_minmax_rndnu_ukernel_3x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_rndnu_ukernel_3x2__scalar,
    xnn_init_qu8_conv_minmax_rndnu_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/3, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_3x2__scalar)

static void qu8_gemm_minmax_rndnu_ukernel_3x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_rndnu_ukernel_3x4__scalar,
    xnn_init_qu8_conv_minmax_rndnu_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/3, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_3x4__scalar)

static void qu8_gemm_minmax_rndnu_ukernel_4x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_rndnu_ukernel_4x2__scalar,
    xnn_init_qu8_conv_minmax_rndnu_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_4x2__scalar)

static void qu8_gemm_minmax_rndnu_ukernel_4x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_rndnu_ukernel_4x4__scalar,
    xnn_init_qu8_conv_minmax_rndnu_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*isa_check=*/nullptr);
}

BENCHMARK_GEMM(qu8_gemm_minmax_rndnu_ukernel_4x4__scalar)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
