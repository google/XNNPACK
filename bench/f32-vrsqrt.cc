// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vrsqrt.yaml
//   Generator: tools/generate-vunary-benchmark.py

#include <stddef.h>
#include <stdint.h>

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

void f32_vrsqrt(benchmark::State& state, xnn_f32_vrsqrt_ukernel_fn ukernel,
              xnn_init_f32_rsqrt_params_fn init_params = nullptr,
              benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  f32_vunary_benchmark<xnn_f32_rsqrt_params>(
      state, ukernel,
      init_params,
      isa_check,
      /*range_min=*/1e-05,
      /*range_max=*/10.0);
}

BENCHMARK_CAPTURE(f32_vrsqrt, scalar_rsqrt_u1,
                  xnn_f32_vrsqrt_ukernel__scalar_rsqrt_u1,
                  /*init_params=*/nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vrsqrt, scalar_rsqrt_u2,
                  xnn_f32_vrsqrt_ukernel__scalar_rsqrt_u2,
                  /*init_params=*/nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vrsqrt, scalar_rsqrt_u4,
                  xnn_f32_vrsqrt_ukernel__scalar_rsqrt_u4,
                  /*init_params=*/nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_vrsqrt, sse_rsqrt_u4,
                    xnn_f32_vrsqrt_ukernel__sse_rsqrt_u4,
                    xnn_init_f32_rsqrt_sse_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrsqrt, sse_rsqrt_u8,
                    xnn_f32_vrsqrt_ukernel__sse_rsqrt_u8,
                    xnn_init_f32_rsqrt_sse_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrsqrt, sse_rsqrt_u16,
                    xnn_f32_vrsqrt_ukernel__sse_rsqrt_u16,
                    xnn_init_f32_rsqrt_sse_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrsqrt, avx_rsqrt_u8,
                    xnn_f32_vrsqrt_ukernel__avx_rsqrt_u8,
                    xnn_init_f32_rsqrt_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrsqrt, avx_rsqrt_u16,
                    xnn_f32_vrsqrt_ukernel__avx_rsqrt_u16,
                    xnn_init_f32_rsqrt_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrsqrt, avx_rsqrt_u32,
                    xnn_f32_vrsqrt_ukernel__avx_rsqrt_u32,
                    xnn_init_f32_rsqrt_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrsqrt, fma3_rsqrt_u8,
                    xnn_f32_vrsqrt_ukernel__fma3_rsqrt_u8,
                    xnn_init_f32_rsqrt_fma3_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrsqrt, fma3_rsqrt_u16,
                    xnn_f32_vrsqrt_ukernel__fma3_rsqrt_u16,
                    xnn_init_f32_rsqrt_fma3_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrsqrt, fma3_rsqrt_u32,
                    xnn_f32_vrsqrt_ukernel__fma3_rsqrt_u32,
                    xnn_init_f32_rsqrt_fma3_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrsqrt, avx512f_rsqrt_u16,
                    xnn_f32_vrsqrt_ukernel__avx512f_rsqrt_u16,
                    xnn_init_f32_rsqrt_avx512_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrsqrt, avx512f_rsqrt_u32,
                    xnn_f32_vrsqrt_ukernel__avx512f_rsqrt_u32,
                    xnn_init_f32_rsqrt_avx512_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrsqrt, avx512f_rsqrt_u64,
                    xnn_f32_vrsqrt_ukernel__avx512f_rsqrt_u64,
                    xnn_init_f32_rsqrt_avx512_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
