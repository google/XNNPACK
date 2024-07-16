// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x16-packw.yaml
//   Generator: tools/generate-packw-test.py


#include <benchmark/benchmark.h>
#include "bench/bgemm.h"
#include "bench/packw-benchmark.h"
#include "bench/utils.h"
#include "xnnpack/common.h"
#include "xnnpack/hardware-config.h"
#include "xnnpack/packw.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12_prfm(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16_prfm(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12_prfm(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16_prfm(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x16_packw_gemm_goi_ukernel_x8__avx2_u16(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }
  BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x8__avx2_u16)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x16_packw_gemm_goi_ukernel_x8__avx2_u16_prfm(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }
  BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x8__avx2_u16_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x16_packw_gemm_goi_ukernel_x16__avx2_u16(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }
  BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x16__avx2_u16)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }
  BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


static void x16_packw_gemm_goi_ukernel_x8__scalar_int_u4(benchmark::State& state, const char* net) {
  x16_packw(state,
    xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_u4,
    /*nr=*/8, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x8__scalar_int_u4)

static void x16_packw_gemm_goi_ukernel_x16__scalar_int_u4(benchmark::State& state, const char* net) {
  x16_packw(state,
    xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_u4,
    /*nr=*/16, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x16__scalar_int_u4)

static void x16_packw_gemm_goi_ukernel_x32__scalar_int_u4(benchmark::State& state, const char* net) {
  x16_packw(state,
    xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4,
    /*nr=*/32, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x32__scalar_int_u4)

static void x16_packw_gemm_goi_ukernel_x64__scalar_int_u4(benchmark::State& state, const char* net) {
  x16_packw(state,
    xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4,
    /*nr=*/64, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x16_packw_gemm_goi_ukernel_x64__scalar_int_u4)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
