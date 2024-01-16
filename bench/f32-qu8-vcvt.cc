// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-qu8-vcvt.yaml
//   Generator: tools/generate-vcvt-test.py


#include <benchmark/benchmark.h>
#include "bench/utils.h"
#include "bench/vcvt-benchmark.h"

#include <xnnpack.h>
#include <xnnpack/common.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/vcvt.h>


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_cvt_u8,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u8,
                    xnn_init_f32_qu8_cvt_wasmsimd_cvt_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_cvt_u16,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u16,
                    xnn_init_f32_qu8_cvt_wasmsimd_cvt_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_cvt_u24,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u24,
                    xnn_init_f32_qu8_cvt_wasmsimd_cvt_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_cvt_u32,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u32,
                    xnn_init_f32_qu8_cvt_wasmsimd_cvt_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_magic_u8,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u8,
                    xnn_init_f32_qu8_cvt_wasmsimd_magic_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_magic_u16,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u16,
                    xnn_init_f32_qu8_cvt_wasmsimd_magic_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_magic_u24,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u24,
                    xnn_init_f32_qu8_cvt_wasmsimd_magic_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_magic_u32,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u32,
                    xnn_init_f32_qu8_cvt_wasmsimd_magic_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasm_fmagic_u1,
                    xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u1,
                    xnn_init_f32_qu8_cvt_scalar_fmagic_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasm_fmagic_u2,
                    xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u2,
                    xnn_init_f32_qu8_cvt_scalar_fmagic_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasm_fmagic_u3,
                    xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u3,
                    xnn_init_f32_qu8_cvt_scalar_fmagic_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasm_fmagic_u4,
                    xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u4,
                    xnn_init_f32_qu8_cvt_scalar_fmagic_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neonv8_u8,
                    xnn_f32_qu8_vcvt_ukernel__neonv8_u8,
                    xnn_init_f32_qu8_cvt_neonv8_params,
                    benchmark::utils::CheckNEONV8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neonv8_u16,
                    xnn_f32_qu8_vcvt_ukernel__neonv8_u16,
                    xnn_init_f32_qu8_cvt_neonv8_params,
                    benchmark::utils::CheckNEONV8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neonv8_u24,
                    xnn_f32_qu8_vcvt_ukernel__neonv8_u24,
                    xnn_init_f32_qu8_cvt_neonv8_params,
                    benchmark::utils::CheckNEONV8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neonv8_u32,
                    xnn_f32_qu8_vcvt_ukernel__neonv8_u32,
                    xnn_init_f32_qu8_cvt_neonv8_params,
                    benchmark::utils::CheckNEONV8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neon_u8,
                    xnn_f32_qu8_vcvt_ukernel__neon_u8,
                    xnn_init_f32_qu8_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neon_u16,
                    xnn_f32_qu8_vcvt_ukernel__neon_u16,
                    xnn_init_f32_qu8_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neon_u24,
                    xnn_f32_qu8_vcvt_ukernel__neon_u24,
                    xnn_init_f32_qu8_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neon_u32,
                    xnn_f32_qu8_vcvt_ukernel__neon_u32,
                    xnn_init_f32_qu8_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx512skx_u32,
                    xnn_f32_qu8_vcvt_ukernel__avx512skx_u32,
                    xnn_init_f32_qu8_cvt_avx512_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx512skx_u64,
                    xnn_f32_qu8_vcvt_ukernel__avx512skx_u64,
                    xnn_init_f32_qu8_cvt_avx512_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx512skx_u96,
                    xnn_f32_qu8_vcvt_ukernel__avx512skx_u96,
                    xnn_init_f32_qu8_cvt_avx512_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx512skx_u128,
                    xnn_f32_qu8_vcvt_ukernel__avx512skx_u128,
                    xnn_init_f32_qu8_cvt_avx512_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx2_u16,
                    xnn_f32_qu8_vcvt_ukernel__avx2_u16,
                    xnn_init_f32_qu8_cvt_avx2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx2_u32,
                    xnn_f32_qu8_vcvt_ukernel__avx2_u32,
                    xnn_init_f32_qu8_cvt_avx2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx2_u48,
                    xnn_f32_qu8_vcvt_ukernel__avx2_u48,
                    xnn_init_f32_qu8_cvt_avx2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx2_u64,
                    xnn_f32_qu8_vcvt_ukernel__avx2_u64,
                    xnn_init_f32_qu8_cvt_avx2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx_u8,
                    xnn_f32_qu8_vcvt_ukernel__avx_u8,
                    xnn_init_f32_qu8_cvt_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx_u16,
                    xnn_f32_qu8_vcvt_ukernel__avx_u16,
                    xnn_init_f32_qu8_cvt_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx_u24,
                    xnn_f32_qu8_vcvt_ukernel__avx_u24,
                    xnn_init_f32_qu8_cvt_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx_u32,
                    xnn_f32_qu8_vcvt_ukernel__avx_u32,
                    xnn_init_f32_qu8_cvt_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, sse2_u8,
                    xnn_f32_qu8_vcvt_ukernel__sse2_u8,
                    xnn_init_f32_qu8_cvt_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, sse2_u16,
                    xnn_f32_qu8_vcvt_ukernel__sse2_u16,
                    xnn_init_f32_qu8_cvt_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, sse2_u24,
                    xnn_f32_qu8_vcvt_ukernel__sse2_u24,
                    xnn_init_f32_qu8_cvt_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, sse2_u32,
                    xnn_f32_qu8_vcvt_ukernel__sse2_u32,
                    xnn_init_f32_qu8_cvt_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_fmagic_u1,
                  xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u1,
                  xnn_init_f32_qu8_cvt_scalar_fmagic_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_fmagic_u2,
                  xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u2,
                  xnn_init_f32_qu8_cvt_scalar_fmagic_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_fmagic_u3,
                  xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u3,
                  xnn_init_f32_qu8_cvt_scalar_fmagic_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_fmagic_u4,
                  xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u4,
                  xnn_init_f32_qu8_cvt_scalar_fmagic_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_imagic_u1,
                  xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u1,
                  xnn_init_f32_qu8_cvt_scalar_imagic_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_imagic_u2,
                  xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u2,
                  xnn_init_f32_qu8_cvt_scalar_imagic_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_imagic_u3,
                  xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u3,
                  xnn_init_f32_qu8_cvt_scalar_imagic_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_imagic_u4,
                  xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u4,
                  xnn_init_f32_qu8_cvt_scalar_imagic_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_lrintf_u1,
                  xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u1,
                  xnn_init_f32_qu8_cvt_scalar_lrintf_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_lrintf_u2,
                  xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u2,
                  xnn_init_f32_qu8_cvt_scalar_lrintf_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_lrintf_u3,
                  xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u3,
                  xnn_init_f32_qu8_cvt_scalar_lrintf_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_lrintf_u4,
                  xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u4,
                  xnn_init_f32_qu8_cvt_scalar_lrintf_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
