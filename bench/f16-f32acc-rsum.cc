// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-f32acc-rsum.yaml
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


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_f32acc_rsum, neonfp16arith_u4,
                    xnn_f16_f32acc_rsum_ukernel__neonfp16arith_u4,
                    xnn_init_f16_f32acc_scale_scalar_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_f32acc_rsum, neonfp16arith_u8,
                    xnn_f16_f32acc_rsum_ukernel__neonfp16arith_u8,
                    xnn_init_f16_f32acc_scale_scalar_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_f32acc_rsum, neonfp16arith_u16_acc2,
                    xnn_f16_f32acc_rsum_ukernel__neonfp16arith_u16_acc2,
                    xnn_init_f16_f32acc_scale_scalar_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_f32acc_rsum, neonfp16arith_u24_acc3,
                    xnn_f16_f32acc_rsum_ukernel__neonfp16arith_u24_acc3,
                    xnn_init_f16_f32acc_scale_scalar_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_f32acc_rsum, neonfp16arith_u32_acc2,
                    xnn_f16_f32acc_rsum_ukernel__neonfp16arith_u32_acc2,
                    xnn_init_f16_f32acc_scale_scalar_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_f32acc_rsum, neonfp16arith_u32_acc4,
                    xnn_f16_f32acc_rsum_ukernel__neonfp16arith_u32_acc4,
                    xnn_init_f16_f32acc_scale_scalar_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32acc_rsum, f16c_u8,
                    xnn_f16_f32acc_rsum_ukernel__f16c_u8,
                    xnn_init_f16_f32acc_scale_scalar_params,
                    benchmark::utils::CheckF16C)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32acc_rsum, f16c_u16_acc2,
                    xnn_f16_f32acc_rsum_ukernel__f16c_u16_acc2,
                    xnn_init_f16_f32acc_scale_scalar_params,
                    benchmark::utils::CheckF16C)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32acc_rsum, f16c_u24_acc3,
                    xnn_f16_f32acc_rsum_ukernel__f16c_u24_acc3,
                    xnn_init_f16_f32acc_scale_scalar_params,
                    benchmark::utils::CheckF16C)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32acc_rsum, f16c_u32_acc2,
                    xnn_f16_f32acc_rsum_ukernel__f16c_u32_acc2,
                    xnn_init_f16_f32acc_scale_scalar_params,
                    benchmark::utils::CheckF16C)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32acc_rsum, f16c_u32_acc4,
                    xnn_f16_f32acc_rsum_ukernel__f16c_u32_acc4,
                    xnn_init_f16_f32acc_scale_scalar_params,
                    benchmark::utils::CheckF16C)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32acc_rsum, avx512skx_u16,
                    xnn_f16_f32acc_rsum_ukernel__avx512skx_u16,
                    xnn_init_f16_f32acc_scale_scalar_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32acc_rsum, avx512skx_u32_acc2,
                    xnn_f16_f32acc_rsum_ukernel__avx512skx_u32_acc2,
                    xnn_init_f16_f32acc_scale_scalar_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32acc_rsum, avx512skx_u48_acc3,
                    xnn_f16_f32acc_rsum_ukernel__avx512skx_u48_acc3,
                    xnn_init_f16_f32acc_scale_scalar_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32acc_rsum, avx512skx_u64_acc2,
                    xnn_f16_f32acc_rsum_ukernel__avx512skx_u64_acc2,
                    xnn_init_f16_f32acc_scale_scalar_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32acc_rsum, avx512skx_u64_acc4,
                    xnn_f16_f32acc_rsum_ukernel__avx512skx_u64_acc4,
                    xnn_init_f16_f32acc_scale_scalar_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32acc_rsum, avx512skx_u128_acc4,
                    xnn_f16_f32acc_rsum_ukernel__avx512skx_u128_acc4,
                    xnn_init_f16_f32acc_scale_scalar_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
