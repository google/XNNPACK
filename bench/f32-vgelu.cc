// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vgelu.yaml
//   Generator: tools/generate-vunary-benchmark.py

#include <stddef.h>
#include <stdint.h>

#include <benchmark/benchmark.h>
#include "bench/f32-vunary-benchmark.h"
#include "bench/utils.h"
#include "xnnpack/common.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"

void f32_vgelu(benchmark::State& state, xnn_f32_vgelu_ukernel_fn ukernel,
              xnn_init_f32_default_params_fn init_params = nullptr,
              benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  f32_vunary_benchmark<xnn_f32_default_params>(
      state, ukernel,
      init_params,
      isa_check,
      /*range_min=*/-10.0,
      /*range_max=*/10.0);
}

BENCHMARK_CAPTURE(f32_vgelu, scalar_u1,
                  xnn_f32_vgelu_ukernel__scalar_u1,
                  /*init_params=*/nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vgelu, scalar_u2,
                  xnn_f32_vgelu_ukernel__scalar_u2,
                  /*init_params=*/nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vgelu, scalar_u4,
                  xnn_f32_vgelu_ukernel__scalar_u4,
                  /*init_params=*/nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vgelu, scalar_rational_12_10_div_u1,
                  xnn_f32_vgelu_ukernel__scalar_rational_12_10_div_u1,
                  /*init_params=*/nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vgelu, scalar_rational_12_10_div_u2,
                  xnn_f32_vgelu_ukernel__scalar_rational_12_10_div_u2,
                  /*init_params=*/nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vgelu, scalar_rational_12_10_div_u4,
                  xnn_f32_vgelu_ukernel__scalar_rational_12_10_div_u4,
                  /*init_params=*/nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vgelu, scalar_rational_12_10_div_u8,
                  xnn_f32_vgelu_ukernel__scalar_rational_12_10_div_u8,
                  /*init_params=*/nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_vgelu, sse2_rational_12_10_div_u4,
                    xnn_f32_vgelu_ukernel__sse2_rational_12_10_div_u4,
                    /*init_params=*/nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, sse2_rational_12_10_div_u8,
                    xnn_f32_vgelu_ukernel__sse2_rational_12_10_div_u8,
                    /*init_params=*/nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, sse2_rational_12_10_div_u12,
                    xnn_f32_vgelu_ukernel__sse2_rational_12_10_div_u12,
                    /*init_params=*/nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, sse2_rational_12_10_div_u16,
                    xnn_f32_vgelu_ukernel__sse2_rational_12_10_div_u16,
                    /*init_params=*/nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, avx_rational_12_10_div_u8,
                    xnn_f32_vgelu_ukernel__avx_rational_12_10_div_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, avx_rational_12_10_div_u16,
                    xnn_f32_vgelu_ukernel__avx_rational_12_10_div_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, avx_rational_12_10_div_u24,
                    xnn_f32_vgelu_ukernel__avx_rational_12_10_div_u24,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, avx_rational_12_10_div_u32,
                    xnn_f32_vgelu_ukernel__avx_rational_12_10_div_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, fma3_rational_12_10_div_u8,
                    xnn_f32_vgelu_ukernel__fma3_rational_12_10_div_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, fma3_rational_12_10_div_u16,
                    xnn_f32_vgelu_ukernel__fma3_rational_12_10_div_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, fma3_rational_12_10_div_u24,
                    xnn_f32_vgelu_ukernel__fma3_rational_12_10_div_u24,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, fma3_rational_12_10_div_u32,
                    xnn_f32_vgelu_ukernel__fma3_rational_12_10_div_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, avx512f_rational_12_10_div_u16,
                    xnn_f32_vgelu_ukernel__avx512f_rational_12_10_div_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, avx512f_rational_12_10_div_u32,
                    xnn_f32_vgelu_ukernel__avx512f_rational_12_10_div_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, avx512f_rational_12_10_div_u48,
                    xnn_f32_vgelu_ukernel__avx512f_rational_12_10_div_u48,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, avx512f_rational_12_10_div_u64,
                    xnn_f32_vgelu_ukernel__avx512f_rational_12_10_div_u64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, avx512f_rational_12_10_nr_u16,
                    xnn_f32_vgelu_ukernel__avx512f_rational_12_10_nr_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, avx512f_rational_12_10_nr_u32,
                    xnn_f32_vgelu_ukernel__avx512f_rational_12_10_nr_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, avx512f_rational_12_10_nr_u48,
                    xnn_f32_vgelu_ukernel__avx512f_rational_12_10_nr_u48,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, avx512f_rational_12_10_nr_u64,
                    xnn_f32_vgelu_ukernel__avx512f_rational_12_10_nr_u64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_vgelu, neon_rational_12_10_div_u4,
                    xnn_f32_vgelu_ukernel__neon_rational_12_10_div_u4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, neon_rational_12_10_div_u8,
                    xnn_f32_vgelu_ukernel__neon_rational_12_10_div_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, neon_rational_12_10_div_u12,
                    xnn_f32_vgelu_ukernel__neon_rational_12_10_div_u12,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, neon_rational_12_10_div_u16,
                    xnn_f32_vgelu_ukernel__neon_rational_12_10_div_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  BENCHMARK_CAPTURE(f32_vgelu, hvx_rational_12_10_div_u32,
                    xnn_f32_vgelu_ukernel__hvx_rational_12_10_div_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckHVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, hvx_rational_12_10_div_u64,
                    xnn_f32_vgelu_ukernel__hvx_rational_12_10_div_u64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckHVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, hvx_rational_12_10_div_u128,
                    xnn_f32_vgelu_ukernel__hvx_rational_12_10_div_u128,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckHVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_vgelu, wasmsimd_rational_12_10_div_u4,
                    xnn_f32_vgelu_ukernel__wasmsimd_rational_12_10_div_u4,
                    /*init_params=*/nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, wasmsimd_rational_12_10_div_u8,
                    xnn_f32_vgelu_ukernel__wasmsimd_rational_12_10_div_u8,
                    /*init_params=*/nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, wasmsimd_rational_12_10_div_u12,
                    xnn_f32_vgelu_ukernel__wasmsimd_rational_12_10_div_u12,
                    /*init_params=*/nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vgelu, wasmsimd_rational_12_10_div_u16,
                    xnn_f32_vgelu_ukernel__wasmsimd_rational_12_10_div_u16,
                    /*init_params=*/nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
