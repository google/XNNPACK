// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "bench/rsum-benchmark.h"
#include "bench/utils.h"
#include <benchmark/benchmark.h>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/gavgpool.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE( f32_rsum_discontig, neon_c4,
                    xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE( f32_rsum_discontig, sse_c4,
                    xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4,
                    xnn_init_f32_scaleminmax_sse_params)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE( f32_rsum_discontig, wasmsimd_arm_c4,
                    xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4,
                    xnn_init_f32_scaleminmax_scalar_params)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE( f32_rsum_discontig, wasmsimd_x86_c4,
                    xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4,
                    xnn_init_f32_scaleminmax_scalar_params)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE( f32_rsum_discontig, wasm_c1,
                    xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1,
                    xnn_init_f32_scaleminmax_scalar_params)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


BENCHMARK_CAPTURE( f32_rsum_discontig, scalar_c1,
                  xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1,
                  xnn_init_f32_scaleminmax_scalar_params)
  ->Apply(BenchmarkBatch)
  ->UseRealTime();

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE( f32_rsum_discontig, neon_c16,
                    xnn_f32_rdsum_minmax_ukernel_7p7x__neon_c16,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE( f32_rsum_discontig, neon_c32,
                    xnn_f32_rdsum_minmax_ukernel_7p7x__neon_c32,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE( f32_rsum_discontig, neon_c64,
                    xnn_f32_rdsum_minmax_ukernel_7p7x__neon_c64,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
