// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/microparams.h>
#include <xnnpack/vunary.h>

#include "bench/f32-vunary-benchmark.h"
#include "bench/utils.h"
#include <benchmark/benchmark.h>

void f32_vsqrt(benchmark::State& state, xnn_f32_vsqrt_ukernel_fn vsqrt,
               xnn_init_f32_sqrt_params_fn init_params = nullptr,
               benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  f32_vunary_benchmark<xnn_f32_sqrt_params>(state, vsqrt, init_params,
                                            isa_check,
                                            /*range_min=*/0.0f,
                                            /*range_max=*/10.0f);
}

#if XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_vsqrt, aarch64_neon_sqrt_u4,
                    xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u4)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, aarch64_neon_sqrt_u8,
                    xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, aarch64_neon_sqrt_u16,
                    xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr1rsqrts1fma1adj_u4,
                    xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u4,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr1rsqrts1fma1adj_u8,
                    xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u8,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr1rsqrts1fma1adj_u16,
                    xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u16,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr2fma1adj_u4,
                    xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u4,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr2fma1adj_u8,
                    xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u8,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr2fma1adj_u16,
                    xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u16,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  BENCHMARK_CAPTURE(f32_vsqrt, rvv_sqrt_u1v,
                    xnn_f32_vsqrt_ukernel__rvv_sqrt_u1v,
                    nullptr /* init params */,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, rvv_sqrt_u2v,
                    xnn_f32_vsqrt_ukernel__rvv_sqrt_u2v,
                    nullptr /* init params */,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, rvv_sqrt_u4v,
                    xnn_f32_vsqrt_ukernel__rvv_sqrt_u4v,
                    nullptr /* init params */,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, rvv_sqrt_u8v,
                    xnn_f32_vsqrt_ukernel__rvv_sqrt_u8v,
                    nullptr /* init params */,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_vsqrt, avx512f_nr1fma1adj_u16,
                    xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u16,
                    xnn_init_f32_sqrt_avx512_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, avx512f_nr1fma1adj_u32,
                    xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u32,
                    xnn_init_f32_sqrt_avx512_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, avx512f_nr1fma1adj_u64,
                    xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u64,
                    xnn_init_f32_sqrt_avx512_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vsqrt, fma3_nr1fma1adj_u8,
                    xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u8,
                    xnn_init_f32_sqrt_fma_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, fma3_nr1fma1adj_u16,
                    xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u16,
                    xnn_init_f32_sqrt_fma_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, fma3_nr1fma1adj_u32,
                    xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u32,
                    xnn_init_f32_sqrt_fma_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vsqrt, avx_sqrt_u8,
                    xnn_f32_vsqrt_ukernel__avx_sqrt_u8,
                    xnn_init_f32_sqrt_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, avx_sqrt_u16,
                    xnn_f32_vsqrt_ukernel__avx_sqrt_u16,
                    xnn_init_f32_sqrt_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, avx_sqrt_u32,
                    xnn_f32_vsqrt_ukernel__avx_sqrt_u32,
                    xnn_init_f32_sqrt_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vsqrt, sse_sqrt_u4,
                    xnn_f32_vsqrt_ukernel__sse_sqrt_u4)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, sse_sqrt_u8,
                    xnn_f32_vsqrt_ukernel__sse_sqrt_u8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, sse_sqrt_u16,
                    xnn_f32_vsqrt_ukernel__sse_sqrt_u16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_vsqrt, wasmsimd_sqrt_u4,
                    xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u4)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, wasmsimd_sqrt_u8,
                    xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, wasmsimd_sqrt_u16,
                    xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(f32_vsqrt, scalar_sqrt_u1,
                  xnn_f32_vsqrt_ukernel__scalar_sqrt_u1)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vsqrt, scalar_sqrt_u2,
                  xnn_f32_vsqrt_ukernel__scalar_sqrt_u2)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vsqrt, scalar_sqrt_u4,
                  xnn_f32_vsqrt_ukernel__scalar_sqrt_u4)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
