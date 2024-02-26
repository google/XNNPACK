// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vtanh.yaml
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

void f32_vtanh(benchmark::State& state, xnn_f32_vtanh_ukernel_fn ukernel,
              xnn_init_f32_tanh_params_fn init_params = nullptr,
              benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  f32_vunary_benchmark<xnn_f32_tanh_params>(
      state, ukernel,
      init_params,
      isa_check,
      /*range_min=*/-10.0,
      /*range_max=*/10.0);
}

BENCHMARK_CAPTURE(f32_vtanh, scalar_expm1minus_rr1_lut8_p4h3ts_div_u1,
                  xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3ts_div_u1,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vtanh, scalar_expm1minus_rr1_lut8_p4h3ts_div_u2,
                  xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3ts_div_u2,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vtanh, scalar_expm1minus_rr1_lut8_p4h3ts_div_u4,
                  xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3ts_div_u4,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vtanh, scalar_expm1minus_rr1_p6h5ts_div_u1,
                  xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5ts_div_u1,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vtanh, scalar_expm1minus_rr1_p6h5ts_div_u2,
                  xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5ts_div_u2,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vtanh, scalar_expm1minus_rr1_p6h5ts_div_u4,
                  xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5ts_div_u4,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vtanh, fma_expm1minus_rr1_lut8_p4h3ts_div_u1,
                  xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3ts_div_u1,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vtanh, fma_expm1minus_rr1_lut8_p4h3ts_div_u2,
                  xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3ts_div_u2,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vtanh, fma_expm1minus_rr1_lut8_p4h3ts_div_u4,
                  xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3ts_div_u4,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vtanh, fma_expm1minus_rr1_p6h5ts_div_u1,
                  xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5ts_div_u1,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vtanh, fma_expm1minus_rr1_p6h5ts_div_u2,
                  xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5ts_div_u2,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vtanh, fma_expm1minus_rr1_p6h5ts_div_u4,
                  xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5ts_div_u4,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_vtanh, wasm_expm1minus_rr1_lut8_p4h3ts_div_u1,
                    xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_lut8_p4h3ts_div_u1,
                    xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasm_expm1minus_rr1_lut8_p4h3ts_div_u2,
                    xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_lut8_p4h3ts_div_u2,
                    xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasm_expm1minus_rr1_lut8_p4h3ts_div_u4,
                    xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_lut8_p4h3ts_div_u4,
                    xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasm_expm1minus_rr1_p6h5ts_div_u1,
                    xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_p6h5ts_div_u1,
                    xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasm_expm1minus_rr1_p6h5ts_div_u2,
                    xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_p6h5ts_div_u2,
                    xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasm_expm1minus_rr1_p6h5ts_div_u4,
                    xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_p6h5ts_div_u4,
                    xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_vtanh, sse2_expm1minus_rr1_lut8_p4h3ts_div_u4,
                    xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u4,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse2_expm1minus_rr1_lut8_p4h3ts_div_u8,
                    xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u8,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse2_expm1minus_rr1_lut8_p4h3ts_div_u12,
                    xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u12,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse2_expm1minus_rr1_lut8_p4h3ts_div_u16,
                    xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u16,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse2_expm1minus_rr1_p6h5ts_div_u4,
                    xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u4,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse2_expm1minus_rr1_p6h5ts_div_u8,
                    xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u8,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse2_expm1minus_rr1_p6h5ts_div_u12,
                    xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u12,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse2_expm1minus_rr1_p6h5ts_div_u16,
                    xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u16,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse2_expm1minus_rr1_p6h5ts_nr1_u4,
                    xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u4,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse2_expm1minus_rr1_p6h5ts_nr1_u8,
                    xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u8,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse2_expm1minus_rr1_p6h5ts_nr1_u12,
                    xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u12,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse2_expm1minus_rr1_p6h5ts_nr1_u16,
                    xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u16,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse2_expm1minus_rr1_p6h5ts_nr2_u4,
                    xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u4,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse2_expm1minus_rr1_p6h5ts_nr2_u8,
                    xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u8,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse2_expm1minus_rr1_p6h5ts_nr2_u12,
                    xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u12,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse2_expm1minus_rr1_p6h5ts_nr2_u16,
                    xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u16,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_lut8_p4h3ts_div_u4,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u4,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_lut8_p4h3ts_div_u8,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u8,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_lut8_p4h3ts_div_u12,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u12,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_lut8_p4h3ts_div_u16,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u16,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_lut8_p4h3ts_div_u20,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u20,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_lut8_p4h3ts_div_u24,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u24,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_p6h5ts_div_u4,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u4,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_p6h5ts_div_u8,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u8,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_p6h5ts_div_u12,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u12,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_p6h5ts_div_u16,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u16,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_p6h5ts_div_u20,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u20,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_p6h5ts_div_u24,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u24,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_p6h5ts_nr1_u4,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u4,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_p6h5ts_nr1_u8,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u8,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_p6h5ts_nr1_u12,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u12,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_p6h5ts_nr1_u16,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u16,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_p6h5ts_nr1_u20,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u20,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_p6h5ts_nr1_u24,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u24,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_p6h5ts_nr2_u4,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u4,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_p6h5ts_nr2_u8,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u8,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_p6h5ts_nr2_u12,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u12,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_p6h5ts_nr2_u16,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u16,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_p6h5ts_nr2_u20,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u20,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, sse41_expm1minus_rr1_p6h5ts_nr2_u24,
                    xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u24,
                    xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u8,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u8,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u16,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u16,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u24,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u24,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u32,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u32,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u40,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u40,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u48,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u48,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u56,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u56,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u64,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u64,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u72,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u72,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u80,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u80,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_lut8_p4h3ts_div_u8,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u8,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_lut8_p4h3ts_div_u16,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u16,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_lut8_p4h3ts_div_u24,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u24,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_lut8_p4h3ts_div_u32,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u32,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_div_u8,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u8,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_div_u16,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u16,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_div_u24,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u24,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_div_u32,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u32,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_div_u40,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u40,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_div_u48,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u48,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_div_u56,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u56,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_div_u64,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u64,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_div_u72,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u72,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_div_u80,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u80,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_nr1_u8,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u8,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_nr1_u16,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u16,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_nr1_u24,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u24,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_nr1_u32,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u32,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_nr1_u40,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u40,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_nr1_u48,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u48,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_nr1_u56,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u56,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_nr1_u64,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u64,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_nr1_u72,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u72,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_nr1_u80,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u80,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_nr2_u8,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u8,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_nr2_u16,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u16,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_nr2_u24,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u24,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_nr2_u32,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u32,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_nr2_u40,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u40,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_nr2_u48,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u48,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_nr2_u56,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u56,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_nr2_u64,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u64,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_nr2_u72,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u72,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx_expm1minus_rr1_p6h5ts_nr2_u80,
                    xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u80,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u8,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u8,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u16,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u16,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u24,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u24,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u32,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u32,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u40,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u40,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u48,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u48,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u56,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u56,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u64,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u64,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u72,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u72,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u80,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u80,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u8,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u8,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u16,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u16,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u24,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u24,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u32,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u32,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u40,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u40,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u48,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u48,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u56,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u56,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u64,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u64,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u72,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u72,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u80,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u80,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut8_p4h3ts_div_u8,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u8,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut8_p4h3ts_div_u16,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u16,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut8_p4h3ts_div_u24,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u24,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut8_p4h3ts_div_u32,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u32,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u8,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u8,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u16,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u16,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u24,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u24,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u32,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u32,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_div_u8,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u8,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_div_u16,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u16,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_div_u24,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u24,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_div_u32,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u32,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_div_u40,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u40,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_div_u48,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u48,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_div_u56,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u56,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_div_u64,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u64,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_div_u72,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u72,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_div_u80,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u80,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_nr1_u8,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u8,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_nr1_u16,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u16,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_nr1_u24,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u24,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_nr1_u32,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u32,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_nr1_u40,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u40,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_nr1_u48,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u48,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_nr1_u56,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u56,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_nr1_u64,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u64,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_nr1_u72,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u72,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_nr1_u80,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u80,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_nr1adj_u8,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u8,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_nr1adj_u16,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u16,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_nr1adj_u24,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u24,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_nr1adj_u32,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u32,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_nr1adj_u40,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u40,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_nr1adj_u48,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u48,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_nr1adj_u56,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u56,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_nr1adj_u64,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u64,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_nr1adj_u72,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u72,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, fma3_expm1minus_rr1_p6h5ts_nr1adj_u80,
                    xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u80,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u8,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u8,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u16,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u16,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u24,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u24,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u32,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u32,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u40,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u40,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u48,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u48,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u56,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u56,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u64,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u64,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u72,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u72,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u80,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u80,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u8,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u8,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u16,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u16,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u24,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u24,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u32,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u32,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u40,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u40,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u48,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u48,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u56,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u56,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u64,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u64,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u72,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u72,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u80,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u80,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u8,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u8,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u16,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u16,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u24,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u24,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u32,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u32,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u40,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u40,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u48,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u48,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u56,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u56,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u64,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u64,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u72,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u72,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u80,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u80,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u8,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u8,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u16,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u16,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u24,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u24,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u32,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u32,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u40,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u40,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u48,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u48,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u56,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u56,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u64,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u64,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u72,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u72,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u80,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u80,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u8,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u8,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u16,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u16,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u24,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u24,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u32,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u32,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u40,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u40,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u48,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u48,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u56,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u56,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u64,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u64,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u72,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u72,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u80,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u80,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u8,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u8,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u16,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u16,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u24,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u24,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u32,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u32,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u40,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u40,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u48,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u48,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u56,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u56,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u64,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u64,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u72,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u72,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u80,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u80,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_div_u8,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u8,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_div_u16,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u16,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_div_u24,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u24,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_div_u32,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u32,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_div_u40,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u40,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_div_u48,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u48,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_div_u56,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u56,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_div_u64,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u64,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_div_u72,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u72,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_div_u80,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u80,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_nr1_u8,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u8,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_nr1_u16,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u16,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_nr1_u24,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u24,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_nr1_u32,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u32,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_nr1_u40,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u40,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_nr1_u48,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u48,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_nr1_u56,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u56,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_nr1_u64,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u64,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_nr1_u72,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u72,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_nr1_u80,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u80,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_nr1adj_u8,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u8,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_nr1adj_u16,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u16,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_nr1adj_u24,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u24,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_nr1adj_u32,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u32,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_nr1adj_u40,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u40,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_nr1adj_u48,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u48,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_nr1adj_u56,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u56,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_nr1adj_u64,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u64,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_nr1adj_u72,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u72,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx2_expm1minus_rr1_p6h5ts_nr1adj_u80,
                    xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u80,
                    xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u16,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u16,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u32,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u32,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u48,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u48,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u64,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u64,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u80,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u80,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u96,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u96,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u112,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u112,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u128,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u128,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u144,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u144,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u160,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u160,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u16,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u16,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u32,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u32,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u48,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u48,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u64,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u64,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u80,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u80,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u96,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u96,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u112,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u112,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u128,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u128,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u144,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u144,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u160,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u160,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u16,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u16,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u32,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u32,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u48,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u48,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u64,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u64,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u80,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u80,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u96,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u96,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u112,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u112,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u128,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u128,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u144,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u144,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u160,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u160,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u16,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u16,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u32,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u32,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u48,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u48,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u64,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u64,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u80,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u80,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u96,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u96,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u112,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u112,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u128,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u128,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u144,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u144,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u160,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u160,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u16,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u16,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u32,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u32,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u48,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u48,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u64,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u64,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u80,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u80,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u96,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u96,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u112,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u112,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u128,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u128,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u144,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u144,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u160,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u160,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u16,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u16,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u32,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u32,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u48,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u48,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u64,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u64,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u80,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u80,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u96,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u96,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u112,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u112,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u128,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u128,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u144,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u144,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u160,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u160,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_p6h5ts_div_u16,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u16,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_p6h5ts_div_u32,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u32,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_p6h5ts_div_u48,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u48,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_p6h5ts_div_u64,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u64,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_p6h5ts_div_u80,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u80,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_p6h5ts_div_u96,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u96,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_p6h5ts_div_u112,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u112,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_p6h5ts_div_u128,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u128,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_p6h5ts_div_u144,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u144,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_p6h5ts_div_u160,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u160,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_p6h5ts_nr1_u16,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u16,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_p6h5ts_nr1_u32,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u32,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_p6h5ts_nr1_u48,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u48,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_p6h5ts_nr1_u64,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u64,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_p6h5ts_nr1_u80,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u80,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_p6h5ts_nr1_u96,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u96,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_p6h5ts_nr1_u112,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u112,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_p6h5ts_nr1_u128,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u128,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_p6h5ts_nr1_u144,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u144,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, avx512skx_expm1minus_rr1_p6h5ts_nr1_u160,
                    xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u160,
                    xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u4,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u4,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u8,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u8,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u12,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u12,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u16,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u16,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u4,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u4,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u8,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u8,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u12,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u12,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u16,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u16,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u4,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u4,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u8,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u8,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u12,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u12,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u16,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u16,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u4,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u4,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u8,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u8,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u12,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u12,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u16,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u16,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u4,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u4,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u8,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u8,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u12,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u12,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u16,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u16,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u4,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u4,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u8,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u8,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u12,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u12,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u16,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u16,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u4,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u4,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u8,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u8,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u12,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u12,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u16,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u16,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u4,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u4,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u8,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u8,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u12,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u12,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u16,
                    xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u16,
                    xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_vtanh, aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u4,
                    xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u4,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u8,
                    xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u8,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u12,
                    xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u12,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u16,
                    xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u16,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u4,
                    xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u4,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u8,
                    xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u8,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u12,
                    xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u12,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u16,
                    xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u16,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_vtanh, neon_expm1minus_rr1_p6h5ts_nr2recps_u4,
                    xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u4,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neon_expm1minus_rr1_p6h5ts_nr2recps_u8,
                    xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u8,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neon_expm1minus_rr1_p6h5ts_nr2recps_u12,
                    xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u12,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neon_expm1minus_rr1_p6h5ts_nr2recps_u16,
                    xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u16,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u4,
                    xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u4,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u8,
                    xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u8,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u12,
                    xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u12,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u16,
                    xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u16,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u4,
                    xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u4,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u8,
                    xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u8,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u12,
                    xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u12,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u16,
                    xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u16,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u4,
                    xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u4,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u8,
                    xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u8,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u12,
                    xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u12,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u16,
                    xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u16,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neonfma_expm1minus_rr1_p6h5ts_nr2fma_u4,
                    xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u4,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neonfma_expm1minus_rr1_p6h5ts_nr2fma_u8,
                    xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u8,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neonfma_expm1minus_rr1_p6h5ts_nr2fma_u12,
                    xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u12,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neonfma_expm1minus_rr1_p6h5ts_nr2fma_u16,
                    xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u16,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neonfma_expm1minus_rr1_p6h5ts_nr2recps_u4,
                    xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u4,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neonfma_expm1minus_rr1_p6h5ts_nr2recps_u8,
                    xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u8,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neonfma_expm1minus_rr1_p6h5ts_nr2recps_u12,
                    xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u12,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vtanh, neonfma_expm1minus_rr1_p6h5ts_nr2recps_u16,
                    xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u16,
                    xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
