// clang-format off
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-rsum.yaml
//   Generator: tools/generate-reduce-discontiguous-benchmark.py

#include "bench/rsum-benchmark.h"
#include "bench/utils.h"
#include <benchmark/benchmark.h>

#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/reduce.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"


BENCHMARK_CAPTURE(qs8_rsum, scalar_u1,
                  xnn_qs8_rsum_ukernel__scalar_u1,
                  /*init_params=*/nullptr)
  ->Apply(BenchmarkRSUM)
  ->UseRealTime();

BENCHMARK_CAPTURE(qs8_rsum, scalar_u2,
                  xnn_qs8_rsum_ukernel__scalar_u2,
                  /*init_params=*/nullptr)
  ->Apply(BenchmarkRSUM)
  ->UseRealTime();

BENCHMARK_CAPTURE(qs8_rsum, scalar_u4,
                  xnn_qs8_rsum_ukernel__scalar_u4,
                  /*init_params=*/nullptr)
  ->Apply(BenchmarkRSUM)
  ->UseRealTime();

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qs8_rsum, neon_u16,
                    xnn_qs8_rsum_ukernel__neon_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qs8_rsum, neon_u32_acc2,
                    xnn_qs8_rsum_ukernel__neon_u32_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qs8_rsum, neon_u64_acc2,
                    xnn_qs8_rsum_ukernel__neon_u64_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qs8_rsum, neon_u64_acc4,
                    xnn_qs8_rsum_ukernel__neon_u64_acc4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(qs8_rsum, neondot_u16,
                    xnn_qs8_rsum_ukernel__neondot_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONDOT)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(qs8_rsum, neondot_u32_acc2,
                    xnn_qs8_rsum_ukernel__neondot_u32_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONDOT)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(qs8_rsum, neondot_u64_acc2,
                    xnn_qs8_rsum_ukernel__neondot_u64_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONDOT)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(qs8_rsum, neondot_u64_acc4,
                    xnn_qs8_rsum_ukernel__neondot_u64_acc4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONDOT)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, ssse3_u16,
                    xnn_qs8_rsum_ukernel__ssse3_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckSSSE3)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, ssse3_u32_acc2,
                    xnn_qs8_rsum_ukernel__ssse3_u32_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckSSSE3)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, ssse3_u64_acc2,
                    xnn_qs8_rsum_ukernel__ssse3_u64_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckSSSE3)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, ssse3_u64_acc4,
                    xnn_qs8_rsum_ukernel__ssse3_u64_acc4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckSSSE3)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx2_u32,
                    xnn_qs8_rsum_ukernel__avx2_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx2_u64_acc2,
                    xnn_qs8_rsum_ukernel__avx2_u64_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx2_u128_acc2,
                    xnn_qs8_rsum_ukernel__avx2_u128_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx2_u128_acc4,
                    xnn_qs8_rsum_ukernel__avx2_u128_acc4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rsum, avxvnni_u32,
                    xnn_qs8_rsum_ukernel__avxvnni_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVXVNNI)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rsum, avxvnni_u64_acc2,
                    xnn_qs8_rsum_ukernel__avxvnni_u64_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVXVNNI)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rsum, avxvnni_u128_acc2,
                    xnn_qs8_rsum_ukernel__avxvnni_u128_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVXVNNI)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rsum, avxvnni_u128_acc4,
                    xnn_qs8_rsum_ukernel__avxvnni_u128_acc4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVXVNNI)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rsum, avx256vnni_u32,
                    xnn_qs8_rsum_ukernel__avx256vnni_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX256VNNI)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rsum, avx256vnni_u64_acc2,
                    xnn_qs8_rsum_ukernel__avx256vnni_u64_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX256VNNI)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rsum, avx256vnni_u128_acc2,
                    xnn_qs8_rsum_ukernel__avx256vnni_u128_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX256VNNI)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rsum, avx256vnni_u128_acc4,
                    xnn_qs8_rsum_ukernel__avx256vnni_u128_acc4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX256VNNI)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rsum, avx256skx_u32,
                    xnn_qs8_rsum_ukernel__avx256skx_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX256SKX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rsum, avx256skx_u64_acc2,
                    xnn_qs8_rsum_ukernel__avx256skx_u64_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX256SKX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rsum, avx256skx_u128_acc2,
                    xnn_qs8_rsum_ukernel__avx256skx_u128_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX256SKX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rsum, avx256skx_u128_acc4,
                    xnn_qs8_rsum_ukernel__avx256skx_u128_acc4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX256SKX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rsum, avx512skx_u64,
                    xnn_qs8_rsum_ukernel__avx512skx_u64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rsum, avx512skx_u128_acc2,
                    xnn_qs8_rsum_ukernel__avx512skx_u128_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rsum, avx512skx_u256_acc2,
                    xnn_qs8_rsum_ukernel__avx512skx_u256_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rsum, avx512skx_u256_acc4,
                    xnn_qs8_rsum_ukernel__avx512skx_u256_acc4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rsum, avx512vnni_u64,
                    xnn_qs8_rsum_ukernel__avx512vnni_u64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512VNNI)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rsum, avx512vnni_u128_acc2,
                    xnn_qs8_rsum_ukernel__avx512vnni_u128_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512VNNI)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rsum, avx512vnni_u256_acc2,
                    xnn_qs8_rsum_ukernel__avx512vnni_u256_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512VNNI)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rsum, avx512vnni_u256_acc4,
                    xnn_qs8_rsum_ukernel__avx512vnni_u256_acc4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512VNNI)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(qs8_rsum, wasmrelaxedsimd_u16,
                    xnn_qs8_rsum_ukernel__wasmrelaxedsimd_u16,
                    /*init_params=*/nullptr)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(qs8_rsum, wasmrelaxedsimd_u32_acc2,
                    xnn_qs8_rsum_ukernel__wasmrelaxedsimd_u32_acc2,
                    /*init_params=*/nullptr)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(qs8_rsum, wasmrelaxedsimd_u64_acc2,
                    xnn_qs8_rsum_ukernel__wasmrelaxedsimd_u64_acc2,
                    /*init_params=*/nullptr)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(qs8_rsum, wasmrelaxedsimd_u64_acc4,
                    xnn_qs8_rsum_ukernel__wasmrelaxedsimd_u64_acc4,
                    /*init_params=*/nullptr)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(qs8_rsum, wasmsimd_u8,
                    xnn_qs8_rsum_ukernel__wasmsimd_u8,
                    /*init_params=*/nullptr)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(qs8_rsum, wasmsimd_u16_acc2,
                    xnn_qs8_rsum_ukernel__wasmsimd_u16_acc2,
                    /*init_params=*/nullptr)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(qs8_rsum, wasmsimd_u32_acc2,
                    xnn_qs8_rsum_ukernel__wasmsimd_u32_acc2,
                    /*init_params=*/nullptr)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(qs8_rsum, wasmsimd_u32_acc4,
                    xnn_qs8_rsum_ukernel__wasmsimd_u32_acc4,
                    /*init_params=*/nullptr)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  BENCHMARK_CAPTURE(qs8_rsum, rvv_u1v,
                    xnn_qs8_rsum_ukernel__rvv_u1v,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckRVV)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  BENCHMARK_CAPTURE(qs8_rsum, rvv_u2v,
                    xnn_qs8_rsum_ukernel__rvv_u2v,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckRVV)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
