// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x32-packw.yaml
//   Generator: tools/generate-packw-test.py


#include <benchmark/benchmark.h>
#include "bench/bgemm.h"
#include "bench/packw-benchmark.h"
#include "bench/utils.h"
#include "xnnpack/common.h"
#include "xnnpack/hardware-config.h"
#include "xnnpack/packw.h"


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  static void x32_packw_gemm_goi_ukernel_x1v__rvv_u2(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x1v__rvv_u2,
      /*nr=*/1 * xnn_init_hardware_config()->vlenb / sizeof(uint32_t), /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckRVV);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x1v__rvv_u2)
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  static void x32_packw_gemm_goi_ukernel_x1v__rvv_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x1v__rvv_u4,
      /*nr=*/1 * xnn_init_hardware_config()->vlenb / sizeof(uint32_t), /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckRVV);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x1v__rvv_u4)
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  static void x32_packw_gemm_goi_ukernel_x1v__rvv_u8(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x1v__rvv_u8,
      /*nr=*/1 * xnn_init_hardware_config()->vlenb / sizeof(uint32_t), /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckRVV);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x1v__rvv_u8)
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  static void x32_packw_gemm_goi_ukernel_x2v__rvv_u2(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x2v__rvv_u2,
      /*nr=*/2 * xnn_init_hardware_config()->vlenb / sizeof(uint32_t), /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckRVV);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x2v__rvv_u2)
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  static void x32_packw_gemm_goi_ukernel_x2v__rvv_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x2v__rvv_u4,
      /*nr=*/2 * xnn_init_hardware_config()->vlenb / sizeof(uint32_t), /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckRVV);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x2v__rvv_u4)
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  static void x32_packw_gemm_goi_ukernel_x2v__rvv_u8(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x2v__rvv_u8,
      /*nr=*/2 * xnn_init_hardware_config()->vlenb / sizeof(uint32_t), /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckRVV);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x2v__rvv_u8)
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  static void x32_packw_gemm_goi_ukernel_x4v__rvv_u2(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x4v__rvv_u2,
      /*nr=*/4 * xnn_init_hardware_config()->vlenb / sizeof(uint32_t), /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckRVV);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x4v__rvv_u2)
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  static void x32_packw_gemm_goi_ukernel_x4v__rvv_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x4v__rvv_u4,
      /*nr=*/4 * xnn_init_hardware_config()->vlenb / sizeof(uint32_t), /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckRVV);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x4v__rvv_u4)
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  static void x32_packw_gemm_goi_ukernel_x4v__rvv_u8(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x4v__rvv_u8,
      /*nr=*/4 * xnn_init_hardware_config()->vlenb / sizeof(uint32_t), /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckRVV);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x4v__rvv_u8)
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  static void x32_packw_gemm_goi_ukernel_x8v__rvv_u2(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8v__rvv_u2,
      /*nr=*/8 * xnn_init_hardware_config()->vlenb / sizeof(uint32_t), /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckRVV);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8v__rvv_u2)
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  static void x32_packw_gemm_goi_ukernel_x8v__rvv_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8v__rvv_u4,
      /*nr=*/8 * xnn_init_hardware_config()->vlenb / sizeof(uint32_t), /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckRVV);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8v__rvv_u4)
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  static void x32_packw_gemm_goi_ukernel_x8v__rvv_u8(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8v__rvv_u8,
      /*nr=*/8 * xnn_init_hardware_config()->vlenb / sizeof(uint32_t), /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckRVV);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8v__rvv_u8)
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_u4,
      /*nr=*/2, /*kr=*/4, /*sr=*/1);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_u4)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void x32_packw_gemm_goi_ukernel_x8__wasmsimd_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_u4,
      /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8__wasmsimd_u4)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_u4,
      /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_u4)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2,
      /*nr=*/2, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2_prfm,
      /*nr=*/2, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u4,
      /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u4)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u4_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u4_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u8(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u8,
      /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u8)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u8_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u8_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u8_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u4,
      /*nr=*/12, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u4)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u4_prfm,
      /*nr=*/12, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u4_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u8(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u8,
      /*nr=*/12, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u8)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u8_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u8_prfm,
      /*nr=*/12, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u8_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x16__avx512f_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_u4,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x16__avx512f_u4)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x16__avx512f_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_u4_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x16__avx512f_u4_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x8__avx_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__avx_u4,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8__avx_u4)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x8__avx_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__avx_u4_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8__avx_u4_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x8s4__avx_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_u4,
      /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckAVX);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8s4__avx_u4)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x8s4__avx_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_u4_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckAVX);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8s4__avx_u4_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x16__avx_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__avx_u4,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x16__avx_u4)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x16__avx_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__avx_u4_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x16__avx_u4_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x16s4__avx_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_u4,
      /*nr=*/16, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckAVX);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x16s4__avx_u4)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x16s4__avx_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_u4_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckAVX);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x16s4__avx_u4_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x2c4__sse2_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_u4,
      /*nr=*/2, /*kr=*/4, /*sr=*/1);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x2c4__sse2_u4)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x2c4__sse2_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_u4_prfm,
      /*nr=*/2, /*kr=*/4, /*sr=*/1);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x2c4__sse2_u4_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x8__sse2_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u4,
      /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8__sse2_u4)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x8__sse2_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u4_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8__sse2_u4_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x8__sse2_u8(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u8,
      /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8__sse2_u8)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x8__sse2_u8_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u8_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8__sse2_u8_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x8s4__sse2_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_u4,
      /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8s4__sse2_u4)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x8s4__sse2_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_u4_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8s4__sse2_u4_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x8s4__sse2_u8(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_u8,
      /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8s4__sse2_u8)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x8s4__sse2_u8_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_u8_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8s4__sse2_u8_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x16__sse2_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__sse2_u4,
      /*nr=*/16, /*kr=*/1, /*sr=*/1);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x16__sse2_u4)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x16__sse2_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__sse2_u4_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/1);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x16__sse2_u4_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x16__sse2_u8(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__sse2_u8,
      /*nr=*/16, /*kr=*/1, /*sr=*/1);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x16__sse2_u8)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x16__sse2_u8_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__sse2_u8_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/1);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x16__sse2_u8_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x16s4__sse2_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_u4,
      /*nr=*/16, /*kr=*/1, /*sr=*/4);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x16s4__sse2_u4)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x16s4__sse2_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_u4_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/4);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x16s4__sse2_u4_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x16s4__sse2_u8(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_u8,
      /*nr=*/16, /*kr=*/1, /*sr=*/4);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x16s4__sse2_u8)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_gemm_goi_ukernel_x16s4__sse2_u8_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_u8_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/4);
  }
  BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x16s4__sse2_u8_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


static void x32_packw_gemm_goi_ukernel_x2__scalar_float_u4(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_u4,
    /*nr=*/2, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x2__scalar_float_u4)

static void x32_packw_gemm_goi_ukernel_x2__scalar_int_u4(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int_u4,
    /*nr=*/2, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x2__scalar_int_u4)

static void x32_packw_gemm_goi_ukernel_x3__scalar_float_u4(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x3__scalar_float_u4,
    /*nr=*/3, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x3__scalar_float_u4)

static void x32_packw_gemm_goi_ukernel_x3__scalar_int_u4(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x3__scalar_int_u4,
    /*nr=*/3, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x3__scalar_int_u4)

static void x32_packw_gemm_goi_ukernel_x4__scalar_float_u4(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_u4,
    /*nr=*/4, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x4__scalar_float_u4)

static void x32_packw_gemm_goi_ukernel_x4__scalar_int_u4(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int_u4,
    /*nr=*/4, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x4__scalar_int_u4)

static void x32_packw_gemm_goi_ukernel_x8__scalar_float_u4(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float_u4,
    /*nr=*/8, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8__scalar_float_u4)

static void x32_packw_gemm_goi_ukernel_x8__scalar_int_u4(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int_u4,
    /*nr=*/8, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x8__scalar_int_u4)

static void x32_packw_gemm_goi_ukernel_x16__scalar_float_u4(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x16__scalar_float_u4,
    /*nr=*/16, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x16__scalar_float_u4)

static void x32_packw_gemm_goi_ukernel_x16__scalar_int_u4(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x16__scalar_int_u4,
    /*nr=*/16, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x32_packw_gemm_goi_ukernel_x16__scalar_int_u4)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
