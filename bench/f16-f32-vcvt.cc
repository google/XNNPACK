// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-f32-vcvt.yaml
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
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmrelaxedsimd_int16_u8,
                    xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_u8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmrelaxedsimd_int16_u16,
                    xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_u16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmrelaxedsimd_int16_u24,
                    xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_u24)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmrelaxedsimd_int16_u32,
                    xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_u32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmrelaxedsimd_int32_u8,
                    xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_u8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmrelaxedsimd_int32_u16,
                    xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_u16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmrelaxedsimd_int32_u24,
                    xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_u24)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmrelaxedsimd_int32_u32,
                    xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_u32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmsimd_int16_u8,
                    xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_u8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmsimd_int16_u16,
                    xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_u16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmsimd_int16_u24,
                    xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_u24)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmsimd_int16_u32,
                    xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_u32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmsimd_int32_u8,
                    xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_u8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmsimd_int32_u16,
                    xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_u16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmsimd_int32_u24,
                    xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_u24)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmsimd_int32_u32,
                    xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_u32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f16_f32_vcvt, neon_int16_u8,
                    xnn_f16_f32_vcvt_ukernel__neon_int16_u8,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f16_f32_vcvt, neon_int16_u16,
                    xnn_f16_f32_vcvt_ukernel__neon_int16_u16,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f16_f32_vcvt, neon_int16_u24,
                    xnn_f16_f32_vcvt_ukernel__neon_int16_u24,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f16_f32_vcvt, neon_int16_u32,
                    xnn_f16_f32_vcvt_ukernel__neon_int16_u32,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f16_f32_vcvt, neon_int32_u8,
                    xnn_f16_f32_vcvt_ukernel__neon_int32_u8,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f16_f32_vcvt, neon_int32_u16,
                    xnn_f16_f32_vcvt_ukernel__neon_int32_u16,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f16_f32_vcvt, neon_int32_u24,
                    xnn_f16_f32_vcvt_ukernel__neon_int32_u24,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f16_f32_vcvt, neon_int32_u32,
                    xnn_f16_f32_vcvt_ukernel__neon_int32_u32,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, avx512skx_u16,
                    xnn_f16_f32_vcvt_ukernel__avx512skx_u16,
                    nullptr /* init params */,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, avx512skx_u32,
                    xnn_f16_f32_vcvt_ukernel__avx512skx_u32,
                    nullptr /* init params */,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, avx_int16_u8,
                    xnn_f16_f32_vcvt_ukernel__avx_int16_u8,
                    nullptr /* init params */,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, avx_int16_u16,
                    xnn_f16_f32_vcvt_ukernel__avx_int16_u16,
                    nullptr /* init params */,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, avx_int16_u24,
                    xnn_f16_f32_vcvt_ukernel__avx_int16_u24,
                    nullptr /* init params */,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, avx_int16_u32,
                    xnn_f16_f32_vcvt_ukernel__avx_int16_u32,
                    nullptr /* init params */,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, avx_int32_u8,
                    xnn_f16_f32_vcvt_ukernel__avx_int32_u8,
                    nullptr /* init params */,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, avx_int32_u16,
                    xnn_f16_f32_vcvt_ukernel__avx_int32_u16,
                    nullptr /* init params */,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, avx_int32_u24,
                    xnn_f16_f32_vcvt_ukernel__avx_int32_u24,
                    nullptr /* init params */,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, avx_int32_u32,
                    xnn_f16_f32_vcvt_ukernel__avx_int32_u32,
                    nullptr /* init params */,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse41_int16_u8,
                    xnn_f16_f32_vcvt_ukernel__sse41_int16_u8,
                    nullptr /* init params */,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse41_int16_u16,
                    xnn_f16_f32_vcvt_ukernel__sse41_int16_u16,
                    nullptr /* init params */,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse41_int16_u24,
                    xnn_f16_f32_vcvt_ukernel__sse41_int16_u24,
                    nullptr /* init params */,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse41_int16_u32,
                    xnn_f16_f32_vcvt_ukernel__sse41_int16_u32,
                    nullptr /* init params */,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse41_int32_u8,
                    xnn_f16_f32_vcvt_ukernel__sse41_int32_u8,
                    nullptr /* init params */,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse41_int32_u16,
                    xnn_f16_f32_vcvt_ukernel__sse41_int32_u16,
                    nullptr /* init params */,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse41_int32_u24,
                    xnn_f16_f32_vcvt_ukernel__sse41_int32_u24,
                    nullptr /* init params */,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse41_int32_u32,
                    xnn_f16_f32_vcvt_ukernel__sse41_int32_u32,
                    nullptr /* init params */,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse2_int16_u8,
                    xnn_f16_f32_vcvt_ukernel__sse2_int16_u8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse2_int16_u16,
                    xnn_f16_f32_vcvt_ukernel__sse2_int16_u16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse2_int16_u24,
                    xnn_f16_f32_vcvt_ukernel__sse2_int16_u24)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse2_int16_u32,
                    xnn_f16_f32_vcvt_ukernel__sse2_int16_u32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse2_int32_u8,
                    xnn_f16_f32_vcvt_ukernel__sse2_int32_u8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse2_int32_u16,
                    xnn_f16_f32_vcvt_ukernel__sse2_int32_u16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse2_int32_u24,
                    xnn_f16_f32_vcvt_ukernel__sse2_int32_u24)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse2_int32_u32,
                    xnn_f16_f32_vcvt_ukernel__sse2_int32_u32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f16_f32_vcvt, neonfp16_u8,
                    xnn_f16_f32_vcvt_ukernel__neonfp16_u8,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFP16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f16_f32_vcvt, neonfp16_u16,
                    xnn_f16_f32_vcvt_ukernel__neonfp16_u16,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFP16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, f16c_u8,
                    xnn_f16_f32_vcvt_ukernel__f16c_u8,
                    nullptr /* init params */,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, f16c_u16,
                    xnn_f16_f32_vcvt_ukernel__f16c_u16,
                    nullptr /* init params */,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


BENCHMARK_CAPTURE(f16_f32_vcvt, scalar_u1,
                  xnn_f16_f32_vcvt_ukernel__scalar_u1)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f16_f32_vcvt, scalar_u2,
                  xnn_f16_f32_vcvt_ukernel__scalar_u2)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f16_f32_vcvt, scalar_u3,
                  xnn_f16_f32_vcvt_ukernel__scalar_u3)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f16_f32_vcvt, scalar_u4,
                  xnn_f16_f32_vcvt_ukernel__scalar_u4)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
