// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vsqrt.yaml
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

void f16_vsqrt(benchmark::State& state, xnn_f16_vsqrt_ukernel_fn ukernel,
              xnn_init_f16_sqrt_params_fn init_params = nullptr,
              benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  f16_vunary_benchmark<xnn_f16_sqrt_params>(
      state, ukernel,
      init_params,
      isa_check,
      /*range_min=*/0.0,
      /*range_max=*/1.0);
}

#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f16_vsqrt, aarch64_neonfp16arith_sqrt_u8,
                    xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsqrt, aarch64_neonfp16arith_sqrt_u16,
                    xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsqrt, aarch64_neonfp16arith_sqrt_u32,
                    xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_vsqrt, neonfp16arith_nr1fma1adj_u8,
                    xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsqrt, neonfp16arith_nr1fma1adj_u16,
                    xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsqrt, neonfp16arith_nr1fma1adj_u32,
                    xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_vsqrt, fp16arith_sqrt_u1,
                    xnn_f16_vsqrt_ukernel__fp16arith_sqrt_u1,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsqrt, fp16arith_sqrt_u2,
                    xnn_f16_vsqrt_ukernel__fp16arith_sqrt_u2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsqrt, fp16arith_sqrt_u4,
                    xnn_f16_vsqrt_ukernel__fp16arith_sqrt_u4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  BENCHMARK_CAPTURE(f16_vsqrt, avx512fp16_sqrt_u32,
                    xnn_f16_vsqrt_ukernel__avx512fp16_sqrt_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512FP16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsqrt, avx512fp16_sqrt_u64,
                    xnn_f16_vsqrt_ukernel__avx512fp16_sqrt_u64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512FP16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsqrt, avx512fp16_sqrt_u128,
                    xnn_f16_vsqrt_ukernel__avx512fp16_sqrt_u128,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512FP16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_vsqrt, avx512skx_sqrt_u16,
                    xnn_f16_vsqrt_ukernel__avx512skx_sqrt_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsqrt, avx512skx_sqrt_u32,
                    xnn_f16_vsqrt_ukernel__avx512skx_sqrt_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsqrt, avx512skx_sqrt_u64,
                    xnn_f16_vsqrt_ukernel__avx512skx_sqrt_u64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsqrt, f16c_rsqrt_u8,
                    xnn_f16_vsqrt_ukernel__f16c_rsqrt_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsqrt, f16c_rsqrt_u16,
                    xnn_f16_vsqrt_ukernel__f16c_rsqrt_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsqrt, f16c_rsqrt_u32,
                    xnn_f16_vsqrt_ukernel__f16c_rsqrt_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsqrt, f16c_sqrt_u8,
                    xnn_f16_vsqrt_ukernel__f16c_sqrt_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsqrt, f16c_sqrt_u16,
                    xnn_f16_vsqrt_ukernel__f16c_sqrt_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsqrt, f16c_sqrt_u32,
                    xnn_f16_vsqrt_ukernel__f16c_sqrt_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
