// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vrndne.yaml
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

void f32_vrndne(benchmark::State& state, xnn_f32_vround_ukernel_fn ukernel,
              xnn_init_f32_rnd_params_fn init_params = nullptr,
              benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  f32_vunary_benchmark<xnn_f32_rnd_params>(
      state, ukernel,
      init_params,
      isa_check,
      /*range_min=*/-10.0,
      /*range_max=*/10.0);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_vrndne, neon_u4,
                    xnn_f32_vrndne_ukernel__neon_u4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrndne, neon_u8,
                    xnn_f32_vrndne_ukernel__neon_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrndne, neonv8_u4,
                    xnn_f32_vrndne_ukernel__neonv8_u4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONV8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrndne, neonv8_u8,
                    xnn_f32_vrndne_ukernel__neonv8_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONV8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_vrndne, sse2_u4,
                    xnn_f32_vrndne_ukernel__sse2_u4,
                    xnn_init_f32_rnd_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrndne, sse2_u8,
                    xnn_f32_vrndne_ukernel__sse2_u8,
                    xnn_init_f32_rnd_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrndne, sse41_u4,
                    xnn_f32_vrndne_ukernel__sse41_u4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrndne, sse41_u8,
                    xnn_f32_vrndne_ukernel__sse41_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrndne, avx_u8,
                    xnn_f32_vrndne_ukernel__avx_u8,
                    xnn_init_f32_rnd_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrndne, avx_u16,
                    xnn_f32_vrndne_ukernel__avx_u16,
                    xnn_init_f32_rnd_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrndne, avx512f_u16,
                    xnn_f32_vrndne_ukernel__avx512f_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrndne, avx512f_u32,
                    xnn_f32_vrndne_ukernel__avx512f_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_vrndne, wasmsimd_u4,
                    xnn_f32_vrndne_ukernel__wasmsimd_u4,
                    /*init_params=*/nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrndne, wasmsimd_u8,
                    xnn_f32_vrndne_ukernel__wasmsimd_u8,
                    /*init_params=*/nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(f32_vrndne, scalar_libm_u1,
                  xnn_f32_vrndne_ukernel__scalar_libm_u1,
                  /*init_params=*/nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vrndne, scalar_libm_u2,
                  xnn_f32_vrndne_ukernel__scalar_libm_u2,
                  /*init_params=*/nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vrndne, scalar_libm_u4,
                  xnn_f32_vrndne_ukernel__scalar_libm_u4,
                  /*init_params=*/nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
