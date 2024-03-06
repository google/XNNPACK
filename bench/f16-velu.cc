// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-velu.yaml
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

void f16_velu(benchmark::State& state, xnn_f16_velu_ukernel_fn ukernel,
              xnn_init_f16_elu_params_fn init_params = nullptr,
              benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  f16_vunary_benchmark<xnn_f16_elu_params>(
      state, ukernel,
      [init_params](xnn_f16_elu_params* params) -> size_t {
        init_params(params,
                    /*prescale=*/UINT16_C(0x3C00),  // prescale = 1.0h
                    /*alpha=*/UINT16_C(0x3C00),     // alpha = 1.0h
                    /*beta=*/UINT16_C(0x3C00));     // beta = 1.0h
        return sizeof(*params);
      },
      isa_check,
      /*range_min=*/-9.0,
      /*range_max=*/9.0);
}

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_velu, neonfp16arith_rr1_p3_u8,
                    xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u8,
                    xnn_init_f16_elu_fp16arith_rr1_p3_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_velu, neonfp16arith_rr1_p3_u16,
                    xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16,
                    xnn_init_f16_elu_fp16arith_rr1_p3_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_velu, avx2_rr1_p3_u8,
                    xnn_f16_velu_ukernel__avx2_rr1_p3_u8,
                    xnn_init_f16_elu_avx2_rr1_p3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_velu, avx2_rr1_p3_u16,
                    xnn_f16_velu_ukernel__avx2_rr1_p3_u16,
                    xnn_init_f16_elu_avx2_rr1_p3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
