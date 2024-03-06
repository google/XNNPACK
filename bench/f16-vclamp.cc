// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vclamp.yaml
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

void f16_vclamp(benchmark::State& state, xnn_f16_vclamp_ukernel_fn ukernel,
              xnn_init_f16_minmax_params_fn init_params = nullptr,
              benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  f16_vunary_benchmark<xnn_f16_minmax_params>(
      state, ukernel,
      [init_params](xnn_f16_minmax_params* params) -> size_t {
        init_params(params,
            UINT16_C(0xAC00),  // -1.0h
            UINT16_C(0x3C00));  // 1.0h
        return sizeof(*params);
      },
      isa_check,
      /*range_min=*/-10.0,
      /*range_max=*/10.0);
}

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_vclamp, neonfp16arith_u8,
                    xnn_f16_vclamp_ukernel__neonfp16arith_u8,
                    xnn_init_f16_minmax_fp16arith_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vclamp, neonfp16arith_u16,
                    xnn_f16_vclamp_ukernel__neonfp16arith_u16,
                    xnn_init_f16_minmax_fp16arith_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_RISCV_FP16_VECTOR && XNN_ARCH_RISCV
  BENCHMARK_CAPTURE(f16_vclamp, rvvfp16arith_u1v,
                    xnn_f16_vclamp_ukernel__rvvfp16arith_u1v,
                    xnn_init_f16_minmax_fp16arith_params,
                    benchmark::utils::CheckRVVFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vclamp, rvvfp16arith_u2v,
                    xnn_f16_vclamp_ukernel__rvvfp16arith_u2v,
                    xnn_init_f16_minmax_fp16arith_params,
                    benchmark::utils::CheckRVVFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vclamp, rvvfp16arith_u4v,
                    xnn_f16_vclamp_ukernel__rvvfp16arith_u4v,
                    xnn_init_f16_minmax_fp16arith_params,
                    benchmark::utils::CheckRVVFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vclamp, rvvfp16arith_u8v,
                    xnn_f16_vclamp_ukernel__rvvfp16arith_u8v,
                    xnn_init_f16_minmax_fp16arith_params,
                    benchmark::utils::CheckRVVFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_RISCV_FP16_VECTOR && XNN_ARCH_RISCV

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_vclamp, f16c_u8,
                    xnn_f16_vclamp_ukernel__f16c_u8,
                    xnn_init_f16_minmax_avx_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vclamp, f16c_u16,
                    xnn_f16_vclamp_ukernel__f16c_u16,
                    xnn_init_f16_minmax_avx_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
