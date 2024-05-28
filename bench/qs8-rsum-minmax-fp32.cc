// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-rsum-minmax-fp32.yaml
//   Generator: tools/generate-rdsum-benchmark.py

#include "bench/rsum-benchmark.h"
#include "bench/utils.h"
#include <benchmark/benchmark.h>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/reduce.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>


BENCHMARK_CAPTURE(qs8_rsum, scalar_imagic_u1,
                  xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u1,
                  xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params)
  ->Apply(BenchmarkRSUM)
  ->UseRealTime();

BENCHMARK_CAPTURE(qs8_rsum, scalar_imagic_u2,
                  xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u2,
                  xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params)
  ->Apply(BenchmarkRSUM)
  ->UseRealTime();

BENCHMARK_CAPTURE(qs8_rsum, scalar_imagic_u4,
                  xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u4,
                  xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params)
  ->Apply(BenchmarkRSUM)
  ->UseRealTime();

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qs8_rsum, neon_u16,
                    xnn_qs8_rsum_minmax_fp32_ukernel__neon_u16,
                    xnn_init_qs8_avgpool_minmax_fp32_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qs8_rsum, neon_u32,
                    xnn_qs8_rsum_minmax_fp32_ukernel__neon_u32,
                    xnn_init_qs8_avgpool_minmax_fp32_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qs8_rsum, neon_u32_acc2,
                    xnn_qs8_rsum_minmax_fp32_ukernel__neon_u32_acc2,
                    xnn_init_qs8_avgpool_minmax_fp32_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qs8_rsum, neon_u64,
                    xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64,
                    xnn_init_qs8_avgpool_minmax_fp32_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qs8_rsum, neon_u64_acc2,
                    xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64_acc2,
                    xnn_init_qs8_avgpool_minmax_fp32_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qs8_rsum, neon_u64_acc4,
                    xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64_acc4,
                    xnn_init_qs8_avgpool_minmax_fp32_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(qs8_rsum, neondot_u16,
                    xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u16,
                    xnn_init_qs8_avgpool_minmax_fp32_neon_params,
                    benchmark::utils::CheckNEONDOT)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(qs8_rsum, neondot_u32,
                    xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u32,
                    xnn_init_qs8_avgpool_minmax_fp32_neon_params,
                    benchmark::utils::CheckNEONDOT)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(qs8_rsum, neondot_u32_acc2,
                    xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u32_acc2,
                    xnn_init_qs8_avgpool_minmax_fp32_neon_params,
                    benchmark::utils::CheckNEONDOT)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(qs8_rsum, neondot_u64,
                    xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u64,
                    xnn_init_qs8_avgpool_minmax_fp32_neon_params,
                    benchmark::utils::CheckNEONDOT)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(qs8_rsum, neondot_u64_acc2,
                    xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u64_acc2,
                    xnn_init_qs8_avgpool_minmax_fp32_neon_params,
                    benchmark::utils::CheckNEONDOT)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(qs8_rsum, neondot_u64_acc4,
                    xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u64_acc4,
                    xnn_init_qs8_avgpool_minmax_fp32_neon_params,
                    benchmark::utils::CheckNEONDOT)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, ssse3_u16,
                    xnn_qs8_rsum_minmax_fp32_ukernel__ssse3_u16,
                    xnn_init_qs8_avgpool_minmax_fp32_ssse3_params,
                    benchmark::utils::CheckSSSE3)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, ssse3_u32_acc2,
                    xnn_qs8_rsum_minmax_fp32_ukernel__ssse3_u32_acc2,
                    xnn_init_qs8_avgpool_minmax_fp32_ssse3_params,
                    benchmark::utils::CheckSSSE3)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, ssse3_u64_acc4,
                    xnn_qs8_rsum_minmax_fp32_ukernel__ssse3_u64_acc4,
                    xnn_init_qs8_avgpool_minmax_fp32_ssse3_params,
                    benchmark::utils::CheckSSSE3)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, sse41_u16,
                    xnn_qs8_rsum_minmax_fp32_ukernel__sse41_u16,
                    xnn_init_qs8_avgpool_minmax_fp32_sse4_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, sse41_u32,
                    xnn_qs8_rsum_minmax_fp32_ukernel__sse41_u32,
                    xnn_init_qs8_avgpool_minmax_fp32_sse4_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, sse41_u32_acc2,
                    xnn_qs8_rsum_minmax_fp32_ukernel__sse41_u32_acc2,
                    xnn_init_qs8_avgpool_minmax_fp32_sse4_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, sse41_u64,
                    xnn_qs8_rsum_minmax_fp32_ukernel__sse41_u64,
                    xnn_init_qs8_avgpool_minmax_fp32_sse4_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, sse41_u64_acc2,
                    xnn_qs8_rsum_minmax_fp32_ukernel__sse41_u64_acc2,
                    xnn_init_qs8_avgpool_minmax_fp32_sse4_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, sse41_u64_acc4,
                    xnn_qs8_rsum_minmax_fp32_ukernel__sse41_u64_acc4,
                    xnn_init_qs8_avgpool_minmax_fp32_sse4_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx2_u32,
                    xnn_qs8_rsum_minmax_fp32_ukernel__avx2_u32,
                    xnn_init_qs8_avgpool_minmax_fp32_avx2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx2_u64,
                    xnn_qs8_rsum_minmax_fp32_ukernel__avx2_u64,
                    xnn_init_qs8_avgpool_minmax_fp32_avx2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx2_u64_acc2,
                    xnn_qs8_rsum_minmax_fp32_ukernel__avx2_u64_acc2,
                    xnn_init_qs8_avgpool_minmax_fp32_avx2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx2_u128,
                    xnn_qs8_rsum_minmax_fp32_ukernel__avx2_u128,
                    xnn_init_qs8_avgpool_minmax_fp32_avx2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx2_u128_acc2,
                    xnn_qs8_rsum_minmax_fp32_ukernel__avx2_u128_acc2,
                    xnn_init_qs8_avgpool_minmax_fp32_avx2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx2_u128_acc4,
                    xnn_qs8_rsum_minmax_fp32_ukernel__avx2_u128_acc4,
                    xnn_init_qs8_avgpool_minmax_fp32_avx2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx512skx_u64,
                    xnn_qs8_rsum_minmax_fp32_ukernel__avx512skx_u64,
                    xnn_init_qs8_avgpool_minmax_fp32_avx2_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx512skx_u128,
                    xnn_qs8_rsum_minmax_fp32_ukernel__avx512skx_u128,
                    xnn_init_qs8_avgpool_minmax_fp32_avx2_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx512skx_u256,
                    xnn_qs8_rsum_minmax_fp32_ukernel__avx512skx_u256,
                    xnn_init_qs8_avgpool_minmax_fp32_avx2_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx512skx_u128_acc2,
                    xnn_qs8_rsum_minmax_fp32_ukernel__avx512skx_u128_acc2,
                    xnn_init_qs8_avgpool_minmax_fp32_avx2_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx512skx_u256_acc2,
                    xnn_qs8_rsum_minmax_fp32_ukernel__avx512skx_u256_acc2,
                    xnn_init_qs8_avgpool_minmax_fp32_avx2_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx512skx_u256_acc4,
                    xnn_qs8_rsum_minmax_fp32_ukernel__avx512skx_u256_acc4,
                    xnn_init_qs8_avgpool_minmax_fp32_avx2_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx512vnni_u64,
                    xnn_qs8_rsum_minmax_fp32_ukernel__avx512vnni_u64,
                    xnn_init_qs8_avgpool_minmax_fp32_avx2_params,
                    benchmark::utils::CheckAVX512VNNI)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx512vnni_u128,
                    xnn_qs8_rsum_minmax_fp32_ukernel__avx512vnni_u128,
                    xnn_init_qs8_avgpool_minmax_fp32_avx2_params,
                    benchmark::utils::CheckAVX512VNNI)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx512vnni_u256,
                    xnn_qs8_rsum_minmax_fp32_ukernel__avx512vnni_u256,
                    xnn_init_qs8_avgpool_minmax_fp32_avx2_params,
                    benchmark::utils::CheckAVX512VNNI)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx512vnni_u128_acc2,
                    xnn_qs8_rsum_minmax_fp32_ukernel__avx512vnni_u128_acc2,
                    xnn_init_qs8_avgpool_minmax_fp32_avx2_params,
                    benchmark::utils::CheckAVX512VNNI)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx512vnni_u256_acc2,
                    xnn_qs8_rsum_minmax_fp32_ukernel__avx512vnni_u256_acc2,
                    xnn_init_qs8_avgpool_minmax_fp32_avx2_params,
                    benchmark::utils::CheckAVX512VNNI)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rsum, avx512vnni_u256_acc4,
                    xnn_qs8_rsum_minmax_fp32_ukernel__avx512vnni_u256_acc4,
                    xnn_init_qs8_avgpool_minmax_fp32_avx2_params,
                    benchmark::utils::CheckAVX512VNNI)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
