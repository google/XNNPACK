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

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/reduce.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>


BENCHMARK_CAPTURE(f32_rsum_discontig, scalar_c4,
                  xnn_f32_rdsum_ukernel_7p7x__scalar_c4,
                  xnn_init_f32_scale_scalar_params)
  ->Apply(BenchmarkBatch)
  ->UseRealTime();

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_rsum_discontig, neon_c16,
                    xnn_f32_rdsum_ukernel_7p7x__neon_c16,
                    xnn_init_f32_scale_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_rsum_discontig, neon_c32,
                    xnn_f32_rdsum_ukernel_7p7x__neon_c32,
                    xnn_init_f32_scale_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_rsum_discontig, neon_c64,
                    xnn_f32_rdsum_ukernel_7p7x__neon_c64,
                    xnn_init_f32_scale_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
