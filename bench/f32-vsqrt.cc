// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vsqrt.yaml
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

void f32_vsqrt(benchmark::State& state, xnn_f32_vsqrt_ukernel_fn ukernel,
              xnn_init_f32_sqrt_params_fn init_params = nullptr,
              benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  f32_vunary_benchmark<xnn_f32_sqrt_params>(
      state, ukernel,
      init_params,
      isa_check,
      /*range_min=*/0.0,
      /*range_max=*/10.0);
}

#if XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_vsqrt, aarch64_neon_sqrt_u4,
                    xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, aarch64_neon_sqrt_u8,
                    xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, aarch64_neon_sqrt_u16,
                    xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM64

#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  BENCHMARK_CAPTURE(f32_vsqrt, rvv_sqrt_u1v,
                    xnn_f32_vsqrt_ukernel__rvv_sqrt_u1v,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, rvv_sqrt_u2v,
                    xnn_f32_vsqrt_ukernel__rvv_sqrt_u2v,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, rvv_sqrt_u4v,
                    xnn_f32_vsqrt_ukernel__rvv_sqrt_u4v,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, rvv_sqrt_u8v,
                    xnn_f32_vsqrt_ukernel__rvv_sqrt_u8v,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_vsqrt, sse_sqrt_u4,
                    xnn_f32_vsqrt_ukernel__sse_sqrt_u4,
                    /*init_params=*/nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, sse_sqrt_u8,
                    xnn_f32_vsqrt_ukernel__sse_sqrt_u8,
                    /*init_params=*/nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, sse_sqrt_u16,
                    xnn_f32_vsqrt_ukernel__sse_sqrt_u16,
                    /*init_params=*/nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, sse_rsqrt_u4,
                    xnn_f32_vsqrt_ukernel__sse_rsqrt_u4,
                    /*init_params=*/nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, sse_rsqrt_u8,
                    xnn_f32_vsqrt_ukernel__sse_rsqrt_u8,
                    /*init_params=*/nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, sse_rsqrt_u12,
                    xnn_f32_vsqrt_ukernel__sse_rsqrt_u12,
                    /*init_params=*/nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, avx_sqrt_u8,
                    xnn_f32_vsqrt_ukernel__avx_sqrt_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, avx_sqrt_u16,
                    xnn_f32_vsqrt_ukernel__avx_sqrt_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, avx_sqrt_u32,
                    xnn_f32_vsqrt_ukernel__avx_sqrt_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, avx_rsqrt_u8,
                    xnn_f32_vsqrt_ukernel__avx_rsqrt_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, avx_rsqrt_u16,
                    xnn_f32_vsqrt_ukernel__avx_rsqrt_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, avx_rsqrt_u32,
                    xnn_f32_vsqrt_ukernel__avx_rsqrt_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, fma3_rsqrt_u8,
                    xnn_f32_vsqrt_ukernel__fma3_rsqrt_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, fma3_rsqrt_u16,
                    xnn_f32_vsqrt_ukernel__fma3_rsqrt_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, fma3_rsqrt_u32,
                    xnn_f32_vsqrt_ukernel__fma3_rsqrt_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, avx512f_rsqrt_u16,
                    xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, avx512f_rsqrt_u32,
                    xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, avx512f_rsqrt_u48,
                    xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u48,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_vsqrt, wasmsimd_sqrt_u4,
                    xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u4,
                    /*init_params=*/nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, wasmsimd_sqrt_u8,
                    xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u8,
                    /*init_params=*/nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, wasmsimd_sqrt_u16,
                    xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u16,
                    /*init_params=*/nullptr)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(f32_vsqrt, scalar_sqrt_u1,
                  xnn_f32_vsqrt_ukernel__scalar_sqrt_u1,
                  /*init_params=*/nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vsqrt, scalar_sqrt_u2,
                  xnn_f32_vsqrt_ukernel__scalar_sqrt_u2,
                  /*init_params=*/nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vsqrt, scalar_sqrt_u4,
                  xnn_f32_vsqrt_ukernel__scalar_sqrt_u4,
                  /*init_params=*/nullptr)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
