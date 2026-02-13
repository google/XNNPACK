// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-qc2w-gemm-minmax-fp32.yaml
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

namespace {



#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_qc2w_gemm_minmax_fp32_ukernel_1x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_1x8c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc2w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_1x8c4__neondot)

  static void qs8_qc2w_gemm_minmax_fp32_ukernel_4x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_4x8c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc2w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_4x8c4__neondot)

  static void qs8_qc2w_gemm_minmax_fp32_ukernel_6x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_6x8c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc2w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_6x8c4__neondot)

  static void qs8_qc2w_gemm_minmax_fp32_ukernel_8x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_8x8c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc2w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_8x8c4__neondot)

  static void qs8_qc2w_gemm_minmax_fp32_ukernel_1x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_1x16c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc2w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_1x16c4__neondot)

  static void qs8_qc2w_gemm_minmax_fp32_ukernel_4x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_4x16c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc2w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_4x16c4__neondot)

  static void qs8_qc2w_gemm_minmax_fp32_ukernel_6x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_6x16c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc2w_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_6x16c4__neondot)

  static void qs8_qc2w_gemm_minmax_fp32_ukernel_8x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_8x16c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_qc2w_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_8x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qs8_qc2w_gemm_minmax_fp32_ukernel_1x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_1x8c8__avx256skx_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc2w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_1x8c8__avx256skx_madd)

  static void qs8_qc2w_gemm_minmax_fp32_ukernel_2x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_2x8c8__avx256skx_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc2w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_2x8c8__avx256skx_madd)

  static void qs8_qc2w_gemm_minmax_fp32_ukernel_3x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_3x8c8__avx256skx_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc2w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_3x8c8__avx256skx_madd)

  static void qs8_qc2w_gemm_minmax_fp32_ukernel_4x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_4x8c8__avx256skx_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc2w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_4x8c8__avx256skx_madd)

  static void qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avx256skx_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc2w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avx256skx_madd)

  static void qs8_qc2w_gemm_minmax_fp32_ukernel_6x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_6x8c8__avx256skx_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc2w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_6x8c8__avx256skx_madd)

  static void qs8_qc2w_gemm_minmax_fp32_ukernel_7x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_7x8c8__avx256skx_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc2w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_7x8c8__avx256skx_madd)

  static void qs8_qc2w_gemm_minmax_fp32_ukernel_8x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_8x8c8__avx256skx_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc2w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_8x8c8__avx256skx_madd)
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qs8_qc2w_gemm_minmax_fp32_ukernel_1x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_1x8c8__avx2_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc2w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_1x8c8__avx2_madd)

  static void qs8_qc2w_gemm_minmax_fp32_ukernel_2x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_2x8c8__avx2_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc2w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_2x8c8__avx2_madd)

  static void qs8_qc2w_gemm_minmax_fp32_ukernel_3x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_3x8c8__avx2_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc2w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_3x8c8__avx2_madd)

  static void qs8_qc2w_gemm_minmax_fp32_ukernel_4x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_4x8c8__avx2_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc2w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_4x8c8__avx2_madd)

  static void qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avx2_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc2w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_5x8c8__avx2_madd)

  static void qs8_qc2w_gemm_minmax_fp32_ukernel_6x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_6x8c8__avx2_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc2w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_6x8c8__avx2_madd)

  static void qs8_qc2w_gemm_minmax_fp32_ukernel_7x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_7x8c8__avx2_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc2w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_7x8c8__avx2_madd)

  static void qs8_qc2w_gemm_minmax_fp32_ukernel_8x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_8x8c8__avx2_madd,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_qc2w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_8x8c8__avx2_madd)
#endif  // XNN_ENABLE_AVX2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


static void qs8_qc2w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_qc2w_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf)

static void qs8_qc2w_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_qc2w_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic)

static void qs8_qc2w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_qc2w_gemm_goi_w,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic)

static void qs8_qc2w_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_qc2w_gemm_goi_w,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic)

static void qs8_qc2w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_qc2w_gemm_goi_w,
    /*mr=*/3, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic)

static void qs8_qc2w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc2w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_qc2w_gemm_goi_w,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc2w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic)

}  // namespace

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
