// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vrsqrt.yaml
//   Generator: tools/generate-vunary-benchmark.py

#include <stddef.h>
#include <stdint.h>

#include <benchmark/benchmark.h>
#include "bench/f16-vunary-benchmark.h"
#include "bench/utils.h"
#include "xnnpack/common.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"

void f16_vrsqrt(benchmark::State& state, xnn_f16_vrsqrt_ukernel_fn ukernel,
              xnn_init_f16_rsqrt_params_fn init_params = nullptr,
              benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  f16_vunary_benchmark<xnn_f16_rsqrt_params>(
      state, ukernel,
      init_params,
      isa_check,
      /*range_min=*/1e-05,
      /*range_max=*/10.0);
}

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_vrsqrt, neonfp16arith_rsqrt_u8,
                    xnn_f16_vrsqrt_ukernel__neonfp16arith_rsqrt_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vrsqrt, neonfp16arith_rsqrt_u16,
                    xnn_f16_vrsqrt_ukernel__neonfp16arith_rsqrt_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vrsqrt, neonfp16arith_rsqrt_u32,
                    xnn_f16_vrsqrt_ukernel__neonfp16arith_rsqrt_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_vrsqrt, f16c_rsqrt_u8,
                    xnn_f16_vrsqrt_ukernel__f16c_rsqrt_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vrsqrt, f16c_rsqrt_u16,
                    xnn_f16_vrsqrt_ukernel__f16c_rsqrt_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vrsqrt, f16c_rsqrt_u32,
                    xnn_f16_vrsqrt_ukernel__f16c_rsqrt_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
