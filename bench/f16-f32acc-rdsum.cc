// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-f32acc-rdsum.yaml
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


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_f32acc_rsum_discontig, neonfp16arith_c16,
                    xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16,
                    xnn_init_f16_f32acc_scale_scalar_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_f32acc_rsum_discontig, neonfp16arith_c32,
                    xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32,
                    xnn_init_f16_f32acc_scale_scalar_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_f32acc_rsum_discontig, neonfp16arith_c64,
                    xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64,
                    xnn_init_f16_f32acc_scale_scalar_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
