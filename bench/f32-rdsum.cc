// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-rdsum.yaml
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


BENCHMARK_CAPTURE(f32_rdsum, scalar_c4,
                  xnn_f32_rdsum_ukernel_7p7x__scalar_c4,
                  xnn_init_f32_scaleminmax_scalar_params)
  ->Apply(BenchmarkRDSUM)
  ->UseRealTime();

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_rdsum, neon_c16,
                    xnn_f32_rdsum_ukernel_7p7x__neon_c16,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_rdsum, neon_c32,
                    xnn_f32_rdsum_ukernel_7p7x__neon_c32,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_rdsum, neon_c64,
                    xnn_f32_rdsum_ukernel_7p7x__neon_c64,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rdsum, sse_c16,
                    xnn_f32_rdsum_ukernel_7p7x__sse_c16,
                    xnn_init_f32_scaleminmax_scalar_params)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rdsum, sse_c32,
                    xnn_f32_rdsum_ukernel_7p7x__sse_c32,
                    xnn_init_f32_scaleminmax_scalar_params)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rdsum, sse_c64,
                    xnn_f32_rdsum_ukernel_7p7x__sse_c64,
                    xnn_init_f32_scaleminmax_scalar_params)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rdsum, avx_c16,
                    xnn_f32_rdsum_ukernel_7p7x__avx_c16,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rdsum, avx_c32,
                    xnn_f32_rdsum_ukernel_7p7x__avx_c32,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rdsum, avx_c64,
                    xnn_f32_rdsum_ukernel_7p7x__avx_c64,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rdsum, avx512f_c16,
                    xnn_f32_rdsum_ukernel_7p7x__avx512f_c16,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rdsum, avx512f_c32,
                    xnn_f32_rdsum_ukernel_7p7x__avx512f_c32,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rdsum, avx512f_c64,
                    xnn_f32_rdsum_ukernel_7p7x__avx512f_c64,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_rdsum, wasmsimd_c16,
                    xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16,
                    xnn_init_f32_scaleminmax_scalar_params)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_rdsum, wasmsimd_c32,
                    xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32,
                    xnn_init_f32_scaleminmax_scalar_params)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_rdsum, wasmsimd_c64,
                    xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64,
                    xnn_init_f32_scaleminmax_scalar_params)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
