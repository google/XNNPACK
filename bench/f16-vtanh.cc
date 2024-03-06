// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vtanh.yaml
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

#include "bench/f16-vunary-benchmark.h"
#include "bench/utils.h"
#include <benchmark/benchmark.h>

void f16_vtanh(benchmark::State& state, xnn_f16_vtanh_ukernel_fn ukernel,
              xnn_init_f16_tanh_params_fn init_params = nullptr,
              benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  f16_vunary_benchmark<xnn_f16_tanh_params>(
      state, ukernel,
      init_params,
      isa_check,
      /*range_min=*/-5.0,
      /*range_max=*/5.0);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_vtanh, f16c_expm1minus_rr1_p3h2ts_div_u8,
                    xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u8,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_expm1minus_rr1_p3h2ts_div_u16,
                    xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u16,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_expm1minus_rr1_p3h2ts_div_u24,
                    xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u24,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_expm1minus_rr1_p3h2ts_div_u32,
                    xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u32,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_expm1minus_rr1_p3h2ts_div_u40,
                    xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u40,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_expm1minus_rr1_p3h2ts_div_u48,
                    xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u48,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_expm1minus_rr1_p3h2ts_div_u56,
                    xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u56,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_expm1minus_rr1_p3h2ts_div_u64,
                    xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u64,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_expm1minus_rr1_p3h2ts_div_u72,
                    xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u72,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_expm1minus_rr1_p3h2ts_div_u80,
                    xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u80,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_expm1minus_rr1_p3h2ts_rcp_u8,
                    xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u8,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_expm1minus_rr1_p3h2ts_rcp_u16,
                    xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u16,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_expm1minus_rr1_p3h2ts_rcp_u24,
                    xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u24,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_expm1minus_rr1_p3h2ts_rcp_u32,
                    xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u32,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_expm1minus_rr1_p3h2ts_rcp_u40,
                    xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u40,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_expm1minus_rr1_p3h2ts_rcp_u48,
                    xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u48,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_expm1minus_rr1_p3h2ts_rcp_u56,
                    xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u56,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_expm1minus_rr1_p3h2ts_rcp_u64,
                    xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u64,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_expm1minus_rr1_p3h2ts_rcp_u72,
                    xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u72,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_expm1minus_rr1_p3h2ts_rcp_u80,
                    xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u80,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_polynomial_p19h9t2_u8,
                    xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u8,
                    xnn_init_f16_tanh_avx_polynomial_p19h9t2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_polynomial_p19h9t2_u16,
                    xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u16,
                    xnn_init_f16_tanh_avx_polynomial_p19h9t2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_polynomial_p19h9t2_u24,
                    xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u24,
                    xnn_init_f16_tanh_avx_polynomial_p19h9t2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_polynomial_p19h9t2_u32,
                    xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u32,
                    xnn_init_f16_tanh_avx_polynomial_p19h9t2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_polynomial_p19h9t2_u40,
                    xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u40,
                    xnn_init_f16_tanh_avx_polynomial_p19h9t2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_polynomial_p19h9t2_u48,
                    xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u48,
                    xnn_init_f16_tanh_avx_polynomial_p19h9t2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_polynomial_p19h9t2_u56,
                    xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u56,
                    xnn_init_f16_tanh_avx_polynomial_p19h9t2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_polynomial_p19h9t2_u64,
                    xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u64,
                    xnn_init_f16_tanh_avx_polynomial_p19h9t2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_polynomial_p19h9t2_u72,
                    xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u72,
                    xnn_init_f16_tanh_avx_polynomial_p19h9t2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, f16c_polynomial_p19h9t2_u80,
                    xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u80,
                    xnn_init_f16_tanh_avx_polynomial_p19h9t2_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_expm1minus_rr1_p3h2ts_div_u8,
                    xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u8,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_expm1minus_rr1_p3h2ts_div_u16,
                    xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u16,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_expm1minus_rr1_p3h2ts_div_u24,
                    xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u24,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_expm1minus_rr1_p3h2ts_div_u32,
                    xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u32,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_expm1minus_rr1_p3h2ts_div_u40,
                    xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u40,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_expm1minus_rr1_p3h2ts_div_u48,
                    xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u48,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_expm1minus_rr1_p3h2ts_div_u56,
                    xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u56,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_expm1minus_rr1_p3h2ts_div_u64,
                    xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u64,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_expm1minus_rr1_p3h2ts_div_u72,
                    xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u72,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_expm1minus_rr1_p3h2ts_div_u80,
                    xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u80,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_expm1minus_rr1_p3h2ts_rcp_u8,
                    xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u8,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_expm1minus_rr1_p3h2ts_rcp_u16,
                    xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u16,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_expm1minus_rr1_p3h2ts_rcp_u24,
                    xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u24,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_expm1minus_rr1_p3h2ts_rcp_u32,
                    xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u32,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_expm1minus_rr1_p3h2ts_rcp_u40,
                    xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u40,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_expm1minus_rr1_p3h2ts_rcp_u48,
                    xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u48,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_expm1minus_rr1_p3h2ts_rcp_u56,
                    xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u56,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_expm1minus_rr1_p3h2ts_rcp_u64,
                    xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u64,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_expm1minus_rr1_p3h2ts_rcp_u72,
                    xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u72,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_expm1minus_rr1_p3h2ts_rcp_u80,
                    xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u80,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_polynomial_p19h9t2_u8,
                    xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u8,
                    xnn_init_f16_tanh_avx_polynomial_p19h9t2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_polynomial_p19h9t2_u16,
                    xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u16,
                    xnn_init_f16_tanh_avx_polynomial_p19h9t2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_polynomial_p19h9t2_u24,
                    xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u24,
                    xnn_init_f16_tanh_avx_polynomial_p19h9t2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_polynomial_p19h9t2_u32,
                    xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u32,
                    xnn_init_f16_tanh_avx_polynomial_p19h9t2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_polynomial_p19h9t2_u40,
                    xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u40,
                    xnn_init_f16_tanh_avx_polynomial_p19h9t2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_polynomial_p19h9t2_u48,
                    xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u48,
                    xnn_init_f16_tanh_avx_polynomial_p19h9t2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_polynomial_p19h9t2_u56,
                    xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u56,
                    xnn_init_f16_tanh_avx_polynomial_p19h9t2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_polynomial_p19h9t2_u64,
                    xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u64,
                    xnn_init_f16_tanh_avx_polynomial_p19h9t2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_polynomial_p19h9t2_u72,
                    xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u72,
                    xnn_init_f16_tanh_avx_polynomial_p19h9t2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, fma3_polynomial_p19h9t2_u80,
                    xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u80,
                    xnn_init_f16_tanh_avx_polynomial_p19h9t2_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, avx2_expm1minus_rr1_p3h2ts_div_u8,
                    xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u8,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, avx2_expm1minus_rr1_p3h2ts_div_u16,
                    xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u16,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, avx2_expm1minus_rr1_p3h2ts_div_u24,
                    xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u24,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, avx2_expm1minus_rr1_p3h2ts_div_u32,
                    xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u32,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, avx2_expm1minus_rr1_p3h2ts_div_u40,
                    xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u40,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, avx2_expm1minus_rr1_p3h2ts_div_u48,
                    xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u48,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, avx2_expm1minus_rr1_p3h2ts_div_u56,
                    xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u56,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, avx2_expm1minus_rr1_p3h2ts_div_u64,
                    xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u64,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, avx2_expm1minus_rr1_p3h2ts_div_u72,
                    xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u72,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, avx2_expm1minus_rr1_p3h2ts_div_u80,
                    xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u80,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, avx2_expm1minus_rr1_p3h2ts_rcp_u8,
                    xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u8,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, avx2_expm1minus_rr1_p3h2ts_rcp_u16,
                    xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u16,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, avx2_expm1minus_rr1_p3h2ts_rcp_u24,
                    xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u24,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, avx2_expm1minus_rr1_p3h2ts_rcp_u32,
                    xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u32,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, avx2_expm1minus_rr1_p3h2ts_rcp_u40,
                    xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u40,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, avx2_expm1minus_rr1_p3h2ts_rcp_u48,
                    xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u48,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, avx2_expm1minus_rr1_p3h2ts_rcp_u56,
                    xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u56,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, avx2_expm1minus_rr1_p3h2ts_rcp_u64,
                    xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u64,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, avx2_expm1minus_rr1_p3h2ts_rcp_u72,
                    xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u72,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, avx2_expm1minus_rr1_p3h2ts_rcp_u80,
                    xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u80,
                    xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f16_vtanh, aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u8,
                    xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u16,
                    xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u24,
                    xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u24,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u32,
                    xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u40,
                    xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u40,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u48,
                    xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u48,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u56,
                    xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u56,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u64,
                    xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u72,
                    xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u72,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u80,
                    xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u80,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u8,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u16,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u24,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u24,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u32,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u40,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u40,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u48,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u48,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u56,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u56,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u64,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u72,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u72,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u80,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u80,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u8,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u16,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u24,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u24,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u32,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u40,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u40,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u48,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u48,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u56,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u56,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u64,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u72,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u72,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u80,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u80,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u8,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u16,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u24,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u24,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u32,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u40,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u40,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u48,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u48,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u56,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u56,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u64,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u72,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u72,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vtanh, neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u80,
                    xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u80,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
