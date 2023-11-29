// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-f16-vcvt.yaml
//   Generator: tools/generate-vcvt-test.py


#include <benchmark/benchmark.h>
#include "bench/utils.h"
#include "bench/vcvt-benchmark.h"

#include <xnnpack.h>
#include <xnnpack/common.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/vcvt.h>


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(qs8_f16_vcvt, neonfp16arith_u8,
                    xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u8,
                    xnn_init_qs8_f16_cvt_neonfp16arith_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(qs8_f16_vcvt, neonfp16arith_u16,
                    xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u16,
                    xnn_init_qs8_f16_cvt_neonfp16arith_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(qs8_f16_vcvt, neonfp16arith_u24,
                    xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u24,
                    xnn_init_qs8_f16_cvt_neonfp16arith_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(qs8_f16_vcvt, neonfp16arith_u32,
                    xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u32,
                    xnn_init_qs8_f16_cvt_neonfp16arith_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
