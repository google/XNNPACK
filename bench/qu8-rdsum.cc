// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qu8-rdsum.yaml
//   Generator: tools/generate-rdsum-benchmark.py

#include "rsum-benchmark.h"
#include "utils.h"
#include <benchmark/benchmark.h>

#include "xnnpack.h"
#include "xnnpack/buffer.h"
#include "xnnpack/common.h"
#include "xnnpack/reduce.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"


BENCHMARK_CAPTURE(qu8_rdsum, scalar_c4,
                  xnn_qu8_rdsum_ukernel_7p7x__scalar_c4,
                  /*init_params=*/nullptr)
  ->Apply(BenchmarkRDSUM)
  ->UseRealTime();

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qu8_rdsum, neon_u16,
                    xnn_qu8_rdsum_ukernel_7p7x__neon_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qu8_rdsum, neon_u32,
                    xnn_qu8_rdsum_ukernel_7p7x__neon_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qu8_rdsum, neon_u64,
                    xnn_qu8_rdsum_ukernel_7p7x__neon_u64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qu8_rdsum, ssse3_c16,
                    xnn_qu8_rdsum_ukernel_7p7x__ssse3_c16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckSSSE3)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qu8_rdsum, ssse3_c32,
                    xnn_qu8_rdsum_ukernel_7p7x__ssse3_c32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckSSSE3)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qu8_rdsum, ssse3_c64,
                    xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckSSSE3)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(qu8_rdsum, wasmsimd_c16,
                    xnn_qu8_rdsum_ukernel_7p7x__wasmsimd_c16,
                    /*init_params=*/nullptr)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(qu8_rdsum, wasmsimd_c32,
                    xnn_qu8_rdsum_ukernel_7p7x__wasmsimd_c32,
                    /*init_params=*/nullptr)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(qu8_rdsum, wasmsimd_c64,
                    xnn_qu8_rdsum_ukernel_7p7x__wasmsimd_c64,
                    /*init_params=*/nullptr)
    ->Apply(BenchmarkRDSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
