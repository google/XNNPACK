// clang-format off
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-rdsum-minmax-fp32.yaml
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


BENCHMARK_CAPTURE(qs8_rdsum, scalar_c4,
                  xnn_qs8_rdsum_ukernel_7p7x__scalar_c4,
                  /*init_params=*/nullptr)
  ->Apply(BenchmarkRDSUM)
  ->UseRealTime();

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qs8_rdsum, neon_c16,
                    xnn_qs8_rdsum_ukernel_7p7x__neon_c16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qs8_rdsum, neon_c32,
                    xnn_qs8_rdsum_ukernel_7p7x__neon_c32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qs8_rdsum, neon_c64,
                    xnn_qs8_rdsum_ukernel_7p7x__neon_c64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rdsum, sse41_c16,
                    xnn_qs8_rdsum_ukernel_7p7x__sse41_c16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckSSE41)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rdsum, sse41_c32,
                    xnn_qs8_rdsum_ukernel_7p7x__sse41_c32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckSSE41)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rdsum, sse41_c64,
                    xnn_qs8_rdsum_ukernel_7p7x__sse41_c64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckSSE41)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rdsum, avx2_c32,
                    xnn_qs8_rdsum_ukernel_7p7x__avx2_c32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_rdsum, avx2_c64,
                    xnn_qs8_rdsum_ukernel_7p7x__avx2_c64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rdsum, avx512skx_c64,
                    xnn_qs8_rdsum_ukernel_7p7x__avx512skx_c64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(qs8_rdsum, avx512skx_c128,
                    xnn_qs8_rdsum_ukernel_7p7x__avx512skx_c128,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(qs8_rdsum, wasmsimd_c16,
                    xnn_qs8_rdsum_ukernel_7p7x__wasmsimd_c16,
                    /*init_params=*/nullptr)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(qs8_rdsum, wasmsimd_c32,
                    xnn_qs8_rdsum_ukernel_7p7x__wasmsimd_c32,
                    /*init_params=*/nullptr)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(qs8_rdsum, wasmsimd_c64,
                    xnn_qs8_rdsum_ukernel_7p7x__wasmsimd_c64,
                    /*init_params=*/nullptr)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  BENCHMARK_CAPTURE(qs8_rdsum, rvv_u1v,
                    xnn_qs8_rdsum_ukernel_7p7x__rvv_u1v,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckRVV)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  BENCHMARK_CAPTURE(qs8_rdsum, rvv_u2v,
                    xnn_qs8_rdsum_ukernel_7p7x__rvv_u2v,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckRVV)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
