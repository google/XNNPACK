// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-f16-vcvt.yaml
//   Generator: tools/generate-vcvt-test.py


#include <benchmark/benchmark.h>
#include "bench/utils.h"
#include "bench/vcvt-benchmark.h"
#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/vcvt.h"


#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_f16_vcvt, wasmrelaxedsimd_u8,
                    xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_f16_vcvt, wasmrelaxedsimd_u16,
                    xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_f16_vcvt, wasmrelaxedsimd_u24,
                    xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u24)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_f16_vcvt, wasmrelaxedsimd_u32,
                    xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_f16_vcvt, wasmsimd_u8,
                    xnn_f32_f16_vcvt_ukernel__wasmsimd_u8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_f16_vcvt, wasmsimd_u16,
                    xnn_f32_f16_vcvt_ukernel__wasmsimd_u16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_f16_vcvt, wasmsimd_u24,
                    xnn_f32_f16_vcvt_ukernel__wasmsimd_u24)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_f16_vcvt, wasmsimd_u32,
                    xnn_f32_f16_vcvt_ukernel__wasmsimd_u32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_f16_vcvt, neon_u8,
                    xnn_f32_f16_vcvt_ukernel__neon_u8,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_f16_vcvt, neon_u16,
                    xnn_f32_f16_vcvt_ukernel__neon_u16,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_f16_vcvt, neon_u24,
                    xnn_f32_f16_vcvt_ukernel__neon_u24,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_f16_vcvt, neon_u32,
                    xnn_f32_f16_vcvt_ukernel__neon_u32,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_f16_vcvt, avx512skx_u16,
                    xnn_f32_f16_vcvt_ukernel__avx512skx_u16,
                    nullptr /* init params */,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_f16_vcvt, avx512skx_u32,
                    xnn_f32_f16_vcvt_ukernel__avx512skx_u32,
                    nullptr /* init params */,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_f16_vcvt, avx_u8,
                    xnn_f32_f16_vcvt_ukernel__avx_u8,
                    nullptr /* init params */,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_f16_vcvt, avx_u16,
                    xnn_f32_f16_vcvt_ukernel__avx_u16,
                    nullptr /* init params */,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_f16_vcvt, avx_u24,
                    xnn_f32_f16_vcvt_ukernel__avx_u24,
                    nullptr /* init params */,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_f16_vcvt, avx_u32,
                    xnn_f32_f16_vcvt_ukernel__avx_u32,
                    nullptr /* init params */,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_f16_vcvt, sse41_u8,
                    xnn_f32_f16_vcvt_ukernel__sse41_u8,
                    nullptr /* init params */,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_f16_vcvt, sse41_u16,
                    xnn_f32_f16_vcvt_ukernel__sse41_u16,
                    nullptr /* init params */,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_f16_vcvt, sse41_u24,
                    xnn_f32_f16_vcvt_ukernel__sse41_u24,
                    nullptr /* init params */,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_f16_vcvt, sse41_u32,
                    xnn_f32_f16_vcvt_ukernel__sse41_u32,
                    nullptr /* init params */,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_f16_vcvt, sse2_u8,
                    xnn_f32_f16_vcvt_ukernel__sse2_u8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_f16_vcvt, sse2_u16,
                    xnn_f32_f16_vcvt_ukernel__sse2_u16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_f16_vcvt, sse2_u24,
                    xnn_f32_f16_vcvt_ukernel__sse2_u24)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_f16_vcvt, sse2_u32,
                    xnn_f32_f16_vcvt_ukernel__sse2_u32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_f16_vcvt, neonfp16_u8,
                    xnn_f32_f16_vcvt_ukernel__neonfp16_u8,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFP16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_f16_vcvt, neonfp16_u16,
                    xnn_f32_f16_vcvt_ukernel__neonfp16_u16,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFP16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_f16_vcvt, f16c_u8,
                    xnn_f32_f16_vcvt_ukernel__f16c_u8,
                    nullptr /* init params */,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_f16_vcvt, f16c_u16,
                    xnn_f32_f16_vcvt_ukernel__f16c_u16,
                    nullptr /* init params */,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


BENCHMARK_CAPTURE(f32_f16_vcvt, scalar_bitcast_u1,
                  xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u1)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_f16_vcvt, scalar_bitcast_u2,
                  xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u2)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_f16_vcvt, scalar_bitcast_u3,
                  xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u3)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_f16_vcvt, scalar_bitcast_u4,
                  xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u4)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_f16_vcvt, scalar_fabsf_u1,
                  xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u1)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_f16_vcvt, scalar_fabsf_u2,
                  xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u2)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_f16_vcvt, scalar_fabsf_u3,
                  xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u3)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_f16_vcvt, scalar_fabsf_u4,
                  xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u4)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
