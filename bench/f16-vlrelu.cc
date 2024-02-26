// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vlrelu.yaml
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

void f16_vlrelu(benchmark::State& state, xnn_f16_vlrelu_ukernel_fn ukernel,
              xnn_init_f16_lrelu_params_fn init_params = nullptr,
              benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  f16_vunary_benchmark<xnn_f16_lrelu_params>(
      state, ukernel,
      [init_params](xnn_f16_lrelu_params* params) -> size_t {
        init_params(params, UINT16_C(0x1F00));  // 0.01h
        return sizeof(*params);
      },
      isa_check,
      /*range_min=*/-5.0,
      /*range_max=*/5.0);
}

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_vlrelu, neonfp16arith_u8,
                    xnn_f16_vlrelu_ukernel__neonfp16arith_u8,
                    xnn_init_f16_lrelu_fp16arith_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vlrelu, neonfp16arith_u16,
                    xnn_f16_vlrelu_ukernel__neonfp16arith_u16,
                    xnn_init_f16_lrelu_fp16arith_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_vlrelu, f16c_u8,
                    xnn_f16_vlrelu_ukernel__f16c_u8,
                    xnn_init_f16_lrelu_avx_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vlrelu, f16c_u16,
                    xnn_f16_vlrelu_ukernel__f16c_u16,
                    xnn_init_f16_lrelu_avx_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
