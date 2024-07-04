// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vclamp.yaml
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

void f32_vclamp(benchmark::State& state, xnn_f32_vclamp_ukernel_fn ukernel,
              xnn_init_f32_minmax_params_fn init_params = nullptr,
              benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  f32_vunary_benchmark<xnn_f32_minmax_params>(
      state, ukernel,
      [init_params](xnn_f32_minmax_params* params) -> size_t {
        init_params(params, -INFINITY, INFINITY);
        return sizeof(*params);
      },
      isa_check,
      /*range_min=*/0.0,
      /*range_max=*/10.0);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_vclamp, neon_u4,
                    xnn_f32_vclamp_ukernel__neon_u4,
                    xnn_init_f32_minmax_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vclamp, neon_u8,
                    xnn_f32_vclamp_ukernel__neon_u8,
                    xnn_init_f32_minmax_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vclamp, neon_u16,
                    xnn_f32_vclamp_ukernel__neon_u16,
                    xnn_init_f32_minmax_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  BENCHMARK_CAPTURE(f32_vclamp, rvv_u1v,
                    xnn_f32_vclamp_ukernel__rvv_u1v,
                    xnn_init_f32_minmax_scalar_params,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vclamp, rvv_u2v,
                    xnn_f32_vclamp_ukernel__rvv_u2v,
                    xnn_init_f32_minmax_scalar_params,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vclamp, rvv_u4v,
                    xnn_f32_vclamp_ukernel__rvv_u4v,
                    xnn_init_f32_minmax_scalar_params,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vclamp, rvv_u8v,
                    xnn_f32_vclamp_ukernel__rvv_u8v,
                    xnn_init_f32_minmax_scalar_params,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_vclamp, sse_u4,
                    xnn_f32_vclamp_ukernel__sse_u4,
                    xnn_init_f32_minmax_sse_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vclamp, sse_u8,
                    xnn_f32_vclamp_ukernel__sse_u8,
                    xnn_init_f32_minmax_sse_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vclamp, avx_u8,
                    xnn_f32_vclamp_ukernel__avx_u8,
                    xnn_init_f32_minmax_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vclamp, avx_u16,
                    xnn_f32_vclamp_ukernel__avx_u16,
                    xnn_init_f32_minmax_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vclamp, avx512f_u16,
                    xnn_f32_vclamp_ukernel__avx512f_u16,
                    xnn_init_f32_minmax_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vclamp, avx512f_u32,
                    xnn_f32_vclamp_ukernel__avx512f_u32,
                    xnn_init_f32_minmax_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_vclamp, wasmsimd_arm_u4,
                    xnn_f32_vclamp_ukernel__wasmsimd_arm_u4,
                    xnn_init_f32_minmax_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vclamp, wasmsimd_arm_u8,
                    xnn_f32_vclamp_ukernel__wasmsimd_arm_u8,
                    xnn_init_f32_minmax_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vclamp, wasmsimd_x86_u4,
                    xnn_f32_vclamp_ukernel__wasmsimd_x86_u4,
                    xnn_init_f32_minmax_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vclamp, wasmsimd_x86_u8,
                    xnn_f32_vclamp_ukernel__wasmsimd_x86_u8,
                    xnn_init_f32_minmax_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_vclamp, wasm_u1,
                    xnn_f32_vclamp_ukernel__wasm_u1,
                    xnn_init_f32_minmax_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vclamp, wasm_u2,
                    xnn_f32_vclamp_ukernel__wasm_u2,
                    xnn_init_f32_minmax_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vclamp, wasm_u4,
                    xnn_f32_vclamp_ukernel__wasm_u4,
                    xnn_init_f32_minmax_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(f32_vclamp, scalar_u1,
                  xnn_f32_vclamp_ukernel__scalar_u1,
                  xnn_init_f32_minmax_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vclamp, scalar_u2,
                  xnn_f32_vclamp_ukernel__scalar_u2,
                  xnn_init_f32_minmax_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vclamp, scalar_u4,
                  xnn_f32_vclamp_ukernel__scalar_u4,
                  xnn_init_f32_minmax_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
