// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-qs8-vcvt.yaml
//   Generator: tools/generate-vcvt-test.py


#include <benchmark/benchmark.h>
#include "bench/utils.h"
#include "bench/vcvt-benchmark.h"
#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/vcvt.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_qs8_vcvt, neonfp16arith_u8,
                    xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u8,
                    xnn_init_f16_qs8_cvt_neonfp16arith_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, int8_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_qs8_vcvt, neonfp16arith_u16,
                    xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u16,
                    xnn_init_f16_qs8_cvt_neonfp16arith_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, int8_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_qs8_vcvt, neonfp16arith_u24,
                    xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u24,
                    xnn_init_f16_qs8_cvt_neonfp16arith_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, int8_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_qs8_vcvt, neonfp16arith_u32,
                    xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u32,
                    xnn_init_f16_qs8_cvt_neonfp16arith_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, int8_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_qs8_vcvt, neonfp16arith_u64,
                    xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u64,
                    xnn_init_f16_qs8_cvt_neonfp16arith_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, int8_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


BENCHMARK_CAPTURE(f16_qs8_vcvt, scalar_fmagic_u1,
                  xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u1,
                  xnn_init_f16_qs8_cvt_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, int8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f16_qs8_vcvt, scalar_fmagic_u2,
                  xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u2,
                  xnn_init_f16_qs8_cvt_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, int8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f16_qs8_vcvt, scalar_fmagic_u3,
                  xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u3,
                  xnn_init_f16_qs8_cvt_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, int8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f16_qs8_vcvt, scalar_fmagic_u4,
                  xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u4,
                  xnn_init_f16_qs8_cvt_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, int8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f16_qs8_vcvt, scalar_imagic_u1,
                  xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u1,
                  xnn_init_f16_qs8_cvt_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, int8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f16_qs8_vcvt, scalar_imagic_u2,
                  xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u2,
                  xnn_init_f16_qs8_cvt_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, int8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f16_qs8_vcvt, scalar_imagic_u3,
                  xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u3,
                  xnn_init_f16_qs8_cvt_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, int8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f16_qs8_vcvt, scalar_imagic_u4,
                  xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u4,
                  xnn_init_f16_qs8_cvt_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, int8_t>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
