// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/s32-f32-vcvt.yaml
//   Generator: tools/generate-vcvt-test.py


#include <benchmark/benchmark.h>
#include "bench/utils.h"
#include "bench/vcvt-benchmark.h"
#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/vcvt.h"


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(s32_f32_vcvt, wasmsimd_u4,
                    xnn_s32_f32_vcvt_ukernel__wasmsimd_u4,
                    xnn_init_s32_f32_cvt_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(s32_f32_vcvt, wasmsimd_u8,
                    xnn_s32_f32_vcvt_ukernel__wasmsimd_u8,
                    xnn_init_s32_f32_cvt_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(s32_f32_vcvt, wasmsimd_u12,
                    xnn_s32_f32_vcvt_ukernel__wasmsimd_u12,
                    xnn_init_s32_f32_cvt_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(s32_f32_vcvt, wasmsimd_u16,
                    xnn_s32_f32_vcvt_ukernel__wasmsimd_u16,
                    xnn_init_s32_f32_cvt_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(s32_f32_vcvt, neon_u4,
                    xnn_s32_f32_vcvt_ukernel__neon_u4,
                    xnn_init_s32_f32_cvt_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(s32_f32_vcvt, neon_u8,
                    xnn_s32_f32_vcvt_ukernel__neon_u8,
                    xnn_init_s32_f32_cvt_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(s32_f32_vcvt, neon_u12,
                    xnn_s32_f32_vcvt_ukernel__neon_u12,
                    xnn_init_s32_f32_cvt_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(s32_f32_vcvt, neon_u16,
                    xnn_s32_f32_vcvt_ukernel__neon_u16,
                    xnn_init_s32_f32_cvt_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(s32_f32_vcvt, avx512f_u16,
                    xnn_s32_f32_vcvt_ukernel__avx512f_u16,
                    xnn_init_s32_f32_cvt_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(s32_f32_vcvt, avx512f_u32,
                    xnn_s32_f32_vcvt_ukernel__avx512f_u32,
                    xnn_init_s32_f32_cvt_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(s32_f32_vcvt, avx512f_u48,
                    xnn_s32_f32_vcvt_ukernel__avx512f_u48,
                    xnn_init_s32_f32_cvt_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(s32_f32_vcvt, avx512f_u64,
                    xnn_s32_f32_vcvt_ukernel__avx512f_u64,
                    xnn_init_s32_f32_cvt_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(s32_f32_vcvt, avx2_u8,
                    xnn_s32_f32_vcvt_ukernel__avx2_u8,
                    xnn_init_s32_f32_cvt_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(s32_f32_vcvt, avx2_u16,
                    xnn_s32_f32_vcvt_ukernel__avx2_u16,
                    xnn_init_s32_f32_cvt_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(s32_f32_vcvt, avx2_u24,
                    xnn_s32_f32_vcvt_ukernel__avx2_u24,
                    xnn_init_s32_f32_cvt_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(s32_f32_vcvt, avx2_u32,
                    xnn_s32_f32_vcvt_ukernel__avx2_u32,
                    xnn_init_s32_f32_cvt_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


BENCHMARK_CAPTURE(s32_f32_vcvt, scalar_u1,
                  xnn_s32_f32_vcvt_ukernel__scalar_u1,
                  xnn_init_s32_f32_cvt_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, float>)
  ->UseRealTime();

BENCHMARK_CAPTURE(s32_f32_vcvt, scalar_u2,
                  xnn_s32_f32_vcvt_ukernel__scalar_u2,
                  xnn_init_s32_f32_cvt_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, float>)
  ->UseRealTime();

BENCHMARK_CAPTURE(s32_f32_vcvt, scalar_u3,
                  xnn_s32_f32_vcvt_ukernel__scalar_u3,
                  xnn_init_s32_f32_cvt_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, float>)
  ->UseRealTime();

BENCHMARK_CAPTURE(s32_f32_vcvt, scalar_u4,
                  xnn_s32_f32_vcvt_ukernel__scalar_u4,
                  xnn_init_s32_f32_cvt_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, float>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
