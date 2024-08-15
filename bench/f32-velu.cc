// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-velu.yaml
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

void f32_velu(benchmark::State& state, xnn_f32_velu_ukernel_fn ukernel,
              xnn_init_f32_elu_params_fn init_params = nullptr,
              benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  f32_vunary_benchmark<xnn_f32_elu_params>(
      state, ukernel,
      [init_params](xnn_f32_elu_params* params) -> size_t {
        init_params(params, /*prescale=*/1.0f, /*alpha=*/1.0f, /*beta=*/1.0f);
        return sizeof(*params);
      },
      isa_check,
      /*range_min=*/-20.0,
      /*range_max=*/10.0);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_velu, neon_rr2_lut16_p3_u4,
                    xnn_f32_velu_ukernel__neon_rr2_lut16_p3_u4,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neon_rr2_lut16_p3_u8,
                    xnn_f32_velu_ukernel__neon_rr2_lut16_p3_u8,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neon_rr2_lut16_p3_u12,
                    xnn_f32_velu_ukernel__neon_rr2_lut16_p3_u12,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neon_rr2_lut16_p3_u16,
                    xnn_f32_velu_ukernel__neon_rr2_lut16_p3_u16,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neon_rr2_lut16_p3_u20,
                    xnn_f32_velu_ukernel__neon_rr2_lut16_p3_u20,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neon_rr2_lut16_p3_u24,
                    xnn_f32_velu_ukernel__neon_rr2_lut16_p3_u24,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neon_rr2_p6_u4,
                    xnn_f32_velu_ukernel__neon_rr2_p6_u4,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neon_rr2_p6_u8,
                    xnn_f32_velu_ukernel__neon_rr2_p6_u8,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neon_rr2_p6_u12,
                    xnn_f32_velu_ukernel__neon_rr2_p6_u12,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neon_rr2_p6_u16,
                    xnn_f32_velu_ukernel__neon_rr2_p6_u16,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neon_rr2_p6_u20,
                    xnn_f32_velu_ukernel__neon_rr2_p6_u20,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neon_rr2_p6_u24,
                    xnn_f32_velu_ukernel__neon_rr2_p6_u24,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_rr1_lut16_p3_u4,
                    xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_u4,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_rr1_lut16_p3_u8,
                    xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_u8,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_rr1_lut16_p3_u12,
                    xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_u12,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_rr1_lut16_p3_u16,
                    xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_u16,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_rr1_lut16_p3_u20,
                    xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_u20,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_rr1_lut16_p3_u24,
                    xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_u24,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_rr1_p6_u4,
                    xnn_f32_velu_ukernel__neonfma_rr1_p6_u4,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_rr1_p6_u8,
                    xnn_f32_velu_ukernel__neonfma_rr1_p6_u8,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_rr1_p6_u12,
                    xnn_f32_velu_ukernel__neonfma_rr1_p6_u12,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_rr1_p6_u16,
                    xnn_f32_velu_ukernel__neonfma_rr1_p6_u16,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_rr1_p6_u20,
                    xnn_f32_velu_ukernel__neonfma_rr1_p6_u20,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_rr1_p6_u24,
                    xnn_f32_velu_ukernel__neonfma_rr1_p6_u24,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_velu, sse2_rr2_lut16_p3_u4,
                    xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_u4,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse2_rr2_lut16_p3_u8,
                    xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_u8,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse2_rr2_lut16_p3_u12,
                    xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_u12,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse2_rr2_lut16_p3_u16,
                    xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_u16,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse2_rr2_lut16_p3_u20,
                    xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_u20,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse2_rr2_lut16_p3_u24,
                    xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_u24,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse2_rr2_p6_u4,
                    xnn_f32_velu_ukernel__sse2_rr2_p6_u4,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse2_rr2_p6_u8,
                    xnn_f32_velu_ukernel__sse2_rr2_p6_u8,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse2_rr2_p6_u12,
                    xnn_f32_velu_ukernel__sse2_rr2_p6_u12,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse2_rr2_p6_u16,
                    xnn_f32_velu_ukernel__sse2_rr2_p6_u16,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse2_rr2_p6_u20,
                    xnn_f32_velu_ukernel__sse2_rr2_p6_u20,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse2_rr2_p6_u24,
                    xnn_f32_velu_ukernel__sse2_rr2_p6_u24,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_rr2_lut16_p3_u4,
                    xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_u4,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_rr2_lut16_p3_u8,
                    xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_u8,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_rr2_lut16_p3_u12,
                    xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_u12,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_rr2_lut16_p3_u16,
                    xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_u16,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_rr2_lut16_p3_u20,
                    xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_u20,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_rr2_lut16_p3_u24,
                    xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_u24,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_rr2_p6_u4,
                    xnn_f32_velu_ukernel__sse41_rr2_p6_u4,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_rr2_p6_u8,
                    xnn_f32_velu_ukernel__sse41_rr2_p6_u8,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_rr2_p6_u12,
                    xnn_f32_velu_ukernel__sse41_rr2_p6_u12,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_rr2_p6_u16,
                    xnn_f32_velu_ukernel__sse41_rr2_p6_u16,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_rr2_p6_u20,
                    xnn_f32_velu_ukernel__sse41_rr2_p6_u20,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_rr2_p6_u24,
                    xnn_f32_velu_ukernel__sse41_rr2_p6_u24,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_rr2_lut4_p4_perm_u8,
                    xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_u8,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_rr2_lut4_p4_perm_u16,
                    xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_u16,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_rr2_lut4_p4_perm_u24,
                    xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_u24,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_rr2_lut4_p4_perm_u32,
                    xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_u32,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_rr2_lut4_p4_perm_u40,
                    xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_u40,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_rr2_lut4_p4_perm_u48,
                    xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_u48,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_rr2_lut16_p3_u8,
                    xnn_f32_velu_ukernel__avx_rr2_lut16_p3_u8,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_rr2_lut16_p3_u16,
                    xnn_f32_velu_ukernel__avx_rr2_lut16_p3_u16,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_rr2_lut16_p3_u24,
                    xnn_f32_velu_ukernel__avx_rr2_lut16_p3_u24,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_rr2_lut16_p3_u32,
                    xnn_f32_velu_ukernel__avx_rr2_lut16_p3_u32,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_rr2_lut16_p3_u40,
                    xnn_f32_velu_ukernel__avx_rr2_lut16_p3_u40,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_rr2_lut16_p3_u48,
                    xnn_f32_velu_ukernel__avx_rr2_lut16_p3_u48,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_rr2_p6_u8,
                    xnn_f32_velu_ukernel__avx_rr2_p6_u8,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_rr2_p6_u16,
                    xnn_f32_velu_ukernel__avx_rr2_p6_u16,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_rr2_p6_u24,
                    xnn_f32_velu_ukernel__avx_rr2_p6_u24,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_rr2_p6_u32,
                    xnn_f32_velu_ukernel__avx_rr2_p6_u32,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_rr2_p6_u40,
                    xnn_f32_velu_ukernel__avx_rr2_p6_u40,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_rr2_p6_u48,
                    xnn_f32_velu_ukernel__avx_rr2_p6_u48,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut4_p4_perm_u8,
                    xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u8,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut4_p4_perm_u16,
                    xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u16,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut4_p4_perm_u24,
                    xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u24,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut4_p4_perm_u32,
                    xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u32,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut4_p4_perm_u40,
                    xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u40,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut4_p4_perm_u48,
                    xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u48,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut4_p4_perm_u56,
                    xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u56,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut4_p4_perm_u64,
                    xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u64,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut4_p4_perm_u72,
                    xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u72,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut4_p4_perm_u80,
                    xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u80,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut8_p4_perm_u8,
                    xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u8,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut8_p4_perm_u16,
                    xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u16,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut8_p4_perm_u24,
                    xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u24,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut8_p4_perm_u32,
                    xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u32,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut8_p4_perm_u40,
                    xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u40,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut8_p4_perm_u48,
                    xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u48,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut8_p4_perm_u56,
                    xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u56,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut8_p4_perm_u64,
                    xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u64,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut8_p4_perm_u72,
                    xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u72,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut8_p4_perm_u80,
                    xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u80,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut16_p3_gather_u8,
                    xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u8,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut16_p3_gather_u16,
                    xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u16,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut16_p3_gather_u24,
                    xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u24,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut16_p3_gather_u32,
                    xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u32,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut16_p3_gather_u40,
                    xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u40,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut16_p3_gather_u48,
                    xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u48,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut16_p3_gather_u56,
                    xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u56,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut16_p3_gather_u64,
                    xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u64,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut16_p3_gather_u72,
                    xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u72,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_lut16_p3_gather_u80,
                    xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u80,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_p6_u8,
                    xnn_f32_velu_ukernel__avx2_rr1_p6_u8,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_p6_u16,
                    xnn_f32_velu_ukernel__avx2_rr1_p6_u16,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_p6_u24,
                    xnn_f32_velu_ukernel__avx2_rr1_p6_u24,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_p6_u32,
                    xnn_f32_velu_ukernel__avx2_rr1_p6_u32,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_p6_u40,
                    xnn_f32_velu_ukernel__avx2_rr1_p6_u40,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_p6_u48,
                    xnn_f32_velu_ukernel__avx2_rr1_p6_u48,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_p6_u56,
                    xnn_f32_velu_ukernel__avx2_rr1_p6_u56,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_p6_u64,
                    xnn_f32_velu_ukernel__avx2_rr1_p6_u64,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_p6_u72,
                    xnn_f32_velu_ukernel__avx2_rr1_p6_u72,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_rr1_p6_u80,
                    xnn_f32_velu_ukernel__avx2_rr1_p6_u80,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_rr1_lut16_p3_perm_u16,
                    xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_u16,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_rr1_lut16_p3_perm_u32,
                    xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_u32,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_rr1_lut16_p3_perm_u48,
                    xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_u48,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_rr1_lut16_p3_perm_u64,
                    xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_u64,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_rr1_lut16_p3_perm_u80,
                    xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_u80,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_rr1_lut16_p3_perm_u96,
                    xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_u96,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_rr1_lut16_p3_perm_u112,
                    xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_u112,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_rr1_lut16_p3_perm_u128,
                    xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_u128,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_rr1_p6_u16,
                    xnn_f32_velu_ukernel__avx512f_rr1_p6_u16,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_rr1_p6_u32,
                    xnn_f32_velu_ukernel__avx512f_rr1_p6_u32,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_rr1_p6_u48,
                    xnn_f32_velu_ukernel__avx512f_rr1_p6_u48,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_rr1_p6_u64,
                    xnn_f32_velu_ukernel__avx512f_rr1_p6_u64,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_rr1_p6_u80,
                    xnn_f32_velu_ukernel__avx512f_rr1_p6_u80,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_rr1_p6_u96,
                    xnn_f32_velu_ukernel__avx512f_rr1_p6_u96,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_rr1_p6_u112,
                    xnn_f32_velu_ukernel__avx512f_rr1_p6_u112,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_rr1_p6_u128,
                    xnn_f32_velu_ukernel__avx512f_rr1_p6_u128,
                    xnn_init_f32_elu_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_rr2_lut16_p3_u4,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_u4,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_rr2_lut16_p3_u8,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_u8,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_rr2_lut16_p3_u12,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_u12,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_rr2_lut16_p3_u16,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_u16,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_rr2_lut16_p3_u20,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_u20,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_rr2_lut16_p3_u24,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_u24,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_rr2_p6_u4,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_u4,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_rr2_p6_u8,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_u8,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_rr2_p6_u12,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_u12,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_rr2_p6_u16,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_u16,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_rr2_p6_u20,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_u20,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_rr2_p6_u24,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_u24,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_rr2_lut16_p3_u4,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_u4,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_rr2_lut16_p3_u8,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_u8,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_rr2_lut16_p3_u12,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_u12,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_rr2_lut16_p3_u16,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_u16,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_rr2_lut16_p3_u20,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_u20,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_rr2_lut16_p3_u24,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_u24,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_rr2_p6_u4,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_u4,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_rr2_p6_u8,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_u8,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_rr2_p6_u12,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_u12,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_rr2_p6_u16,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_u16,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_rr2_p6_u20,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_u20,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_rr2_p6_u24,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_u24,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_rr2_lut16_p3_u4,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_u4,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_rr2_lut16_p3_u8,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_u8,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_rr2_lut16_p3_u12,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_u12,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_rr2_lut16_p3_u16,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_u16,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_rr2_lut16_p3_u20,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_u20,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_rr2_lut16_p3_u24,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_u24,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_rr2_p6_u4,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_u4,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_rr2_p6_u8,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_u8,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_rr2_p6_u12,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_u12,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_rr2_p6_u16,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_u16,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_rr2_p6_u20,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_u20,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_rr2_p6_u24,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_u24,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_rr2_lut16_p3_u4,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_u4,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_rr2_lut16_p3_u8,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_u8,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_rr2_lut16_p3_u12,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_u12,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_rr2_lut16_p3_u16,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_u16,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_rr2_lut16_p3_u20,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_u20,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_rr2_lut16_p3_u24,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_u24,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_rr2_p6_u4,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_u4,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_rr2_p6_u8,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_u8,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_rr2_p6_u12,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_u12,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_rr2_p6_u16,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_u16,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_rr2_p6_u20,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_u20,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_rr2_p6_u24,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_u24,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_velu, wasm_rr2_lut16_p3_u1,
                    xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_u1,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasm_rr2_lut16_p3_u2,
                    xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_u2,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasm_rr2_lut16_p3_u3,
                    xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_u3,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasm_rr2_lut16_p3_u4,
                    xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_u4,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasm_rr2_lut16_p3_u5,
                    xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_u5,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasm_rr2_lut16_p3_u6,
                    xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_u6,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasm_rr2_p6_u1,
                    xnn_f32_velu_ukernel__wasm_rr2_p6_u1,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasm_rr2_p6_u2,
                    xnn_f32_velu_ukernel__wasm_rr2_p6_u2,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasm_rr2_p6_u3,
                    xnn_f32_velu_ukernel__wasm_rr2_p6_u3,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasm_rr2_p6_u4,
                    xnn_f32_velu_ukernel__wasm_rr2_p6_u4,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasm_rr2_p6_u5,
                    xnn_f32_velu_ukernel__wasm_rr2_p6_u5,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasm_rr2_p6_u6,
                    xnn_f32_velu_ukernel__wasm_rr2_p6_u6,
                    xnn_init_f32_elu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(f32_velu, scalar_rr2_lut16_p3_u1,
                  xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u1,
                  xnn_init_f32_elu_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_velu, scalar_rr2_lut16_p3_u2,
                  xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u2,
                  xnn_init_f32_elu_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_velu, scalar_rr2_lut16_p3_u3,
                  xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u3,
                  xnn_init_f32_elu_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_velu, scalar_rr2_lut16_p3_u4,
                  xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u4,
                  xnn_init_f32_elu_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_velu, scalar_rr2_lut16_p3_u5,
                  xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u5,
                  xnn_init_f32_elu_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_velu, scalar_rr2_lut16_p3_u6,
                  xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u6,
                  xnn_init_f32_elu_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_velu, scalar_rr2_p6_u1,
                  xnn_f32_velu_ukernel__scalar_rr2_p6_u1,
                  xnn_init_f32_elu_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_velu, scalar_rr2_p6_u2,
                  xnn_f32_velu_ukernel__scalar_rr2_p6_u2,
                  xnn_init_f32_elu_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_velu, scalar_rr2_p6_u3,
                  xnn_f32_velu_ukernel__scalar_rr2_p6_u3,
                  xnn_init_f32_elu_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_velu, scalar_rr2_p6_u4,
                  xnn_f32_velu_ukernel__scalar_rr2_p6_u4,
                  xnn_init_f32_elu_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_velu, scalar_rr2_p6_u5,
                  xnn_f32_velu_ukernel__scalar_rr2_p6_u5,
                  xnn_init_f32_elu_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_velu, scalar_rr2_p6_u6,
                  xnn_f32_velu_ukernel__scalar_rr2_p6_u6,
                  xnn_init_f32_elu_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
