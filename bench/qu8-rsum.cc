// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qu8-rsum.yaml
//   Generator: tools/generate-rdsum-benchmark.py

#include "bench/rsum-benchmark.h"
#include "bench/utils.h"
#include <benchmark/benchmark.h>

#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/reduce.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"


BENCHMARK_CAPTURE(qu8_rsum, scalar_u1,
                  xnn_qu8_rsum_ukernel__scalar_u1,
                  /*init_params=*/nullptr)
  ->Apply(BenchmarkRSUM)
  ->UseRealTime();

BENCHMARK_CAPTURE(qu8_rsum, scalar_u2,
                  xnn_qu8_rsum_ukernel__scalar_u2,
                  /*init_params=*/nullptr)
  ->Apply(BenchmarkRSUM)
  ->UseRealTime();

BENCHMARK_CAPTURE(qu8_rsum, scalar_u4,
                  xnn_qu8_rsum_ukernel__scalar_u4,
                  /*init_params=*/nullptr)
  ->Apply(BenchmarkRSUM)
  ->UseRealTime();

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qu8_rsum, neon_u16,
                    xnn_qu8_rsum_ukernel__neon_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qu8_rsum, neon_u32,
                    xnn_qu8_rsum_ukernel__neon_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qu8_rsum, neon_u32_acc2,
                    xnn_qu8_rsum_ukernel__neon_u32_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qu8_rsum, neon_u64,
                    xnn_qu8_rsum_ukernel__neon_u64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qu8_rsum, neon_u64_acc2,
                    xnn_qu8_rsum_ukernel__neon_u64_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qu8_rsum, neon_u64_acc4,
                    xnn_qu8_rsum_ukernel__neon_u64_acc4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qu8_rsum, sse2_u16,
                    xnn_qu8_rsum_ukernel__sse2_u16,
                    /*init_params=*/nullptr)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qu8_rsum, sse2_u32,
                    xnn_qu8_rsum_ukernel__sse2_u32,
                    /*init_params=*/nullptr)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qu8_rsum, sse2_u32_acc2,
                    xnn_qu8_rsum_ukernel__sse2_u32_acc2,
                    /*init_params=*/nullptr)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qu8_rsum, sse2_u64,
                    xnn_qu8_rsum_ukernel__sse2_u64,
                    /*init_params=*/nullptr)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qu8_rsum, sse2_u64_acc2,
                    xnn_qu8_rsum_ukernel__sse2_u64_acc2,
                    /*init_params=*/nullptr)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qu8_rsum, sse2_u64_acc4,
                    xnn_qu8_rsum_ukernel__sse2_u64_acc4,
                    /*init_params=*/nullptr)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qu8_rsum, avx2_u32,
                    xnn_qu8_rsum_ukernel__avx2_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qu8_rsum, avx2_u64,
                    xnn_qu8_rsum_ukernel__avx2_u64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qu8_rsum, avx2_u64_acc2,
                    xnn_qu8_rsum_ukernel__avx2_u64_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qu8_rsum, avx2_u128,
                    xnn_qu8_rsum_ukernel__avx2_u128,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qu8_rsum, avx2_u128_acc2,
                    xnn_qu8_rsum_ukernel__avx2_u128_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qu8_rsum, avx2_u128_acc4,
                    xnn_qu8_rsum_ukernel__avx2_u128_acc4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
