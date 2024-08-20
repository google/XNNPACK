// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vsigmoid.yaml
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

void f16_vsigmoid(benchmark::State& state, xnn_f16_vsigmoid_ukernel_fn ukernel,
              xnn_init_f16_sigmoid_params_fn init_params = nullptr,
              benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  f16_vunary_benchmark<xnn_f16_sigmoid_params>(
      state, ukernel,
      init_params,
      isa_check,
      /*range_min=*/-10.0,
      /*range_max=*/10.0);
}

#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f16_vsigmoid, aarch64_neonfp16arith_rr2_p2_div_u8,
                    xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, aarch64_neonfp16arith_rr2_p2_div_u16,
                    xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, aarch64_neonfp16arith_rr2_p2_div_u24,
                    xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u24,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, aarch64_neonfp16arith_rr2_p2_div_u32,
                    xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, aarch64_neonfp16arith_rr2_p2_div_u40,
                    xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u40,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, aarch64_neonfp16arith_rr2_p2_div_u48,
                    xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u48,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, aarch64_neonfp16arith_rr2_p2_div_u56,
                    xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u56,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, aarch64_neonfp16arith_rr2_p2_div_u64,
                    xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_vsigmoid, neonfp16arith_rr2_p2_nr1fma_u8,
                    xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, neonfp16arith_rr2_p2_nr1fma_u16,
                    xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, neonfp16arith_rr2_p2_nr1fma_u24,
                    xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u24,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, neonfp16arith_rr2_p2_nr1fma_u32,
                    xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, neonfp16arith_rr2_p2_nr1fma_u40,
                    xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u40,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, neonfp16arith_rr2_p2_nr1fma_u48,
                    xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u48,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, neonfp16arith_rr2_p2_nr1fma_u56,
                    xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u56,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, neonfp16arith_rr2_p2_nr1fma_u64,
                    xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, neonfp16arith_rr2_p2_nr1recps_u8,
                    xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, neonfp16arith_rr2_p2_nr1recps_u16,
                    xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, neonfp16arith_rr2_p2_nr1recps_u24,
                    xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u24,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, neonfp16arith_rr2_p2_nr1recps_u32,
                    xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, neonfp16arith_rr2_p2_nr1recps_u40,
                    xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u40,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, neonfp16arith_rr2_p2_nr1recps_u48,
                    xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u48,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, neonfp16arith_rr2_p2_nr1recps_u56,
                    xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u56,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, neonfp16arith_rr2_p2_nr1recps_u64,
                    xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_vsigmoid, avx2_rr1_p2_div_u8,
                    xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, avx2_rr1_p2_div_u16,
                    xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, avx2_rr1_p2_div_u24,
                    xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u24,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, avx2_rr1_p2_div_u32,
                    xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, avx2_rr1_p2_div_u40,
                    xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u40,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, avx2_rr1_p2_div_u48,
                    xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u48,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, avx2_rr1_p2_div_u56,
                    xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u56,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, avx2_rr1_p2_div_u64,
                    xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, avx2_rr1_p2_rcp_u8,
                    xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, avx2_rr1_p2_rcp_u16,
                    xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, avx2_rr1_p2_rcp_u24,
                    xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u24,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, avx2_rr1_p2_rcp_u32,
                    xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, avx2_rr1_p2_rcp_u40,
                    xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u40,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, avx2_rr1_p2_rcp_u48,
                    xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u48,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, avx2_rr1_p2_rcp_u56,
                    xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u56,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vsigmoid, avx2_rr1_p2_rcp_u64,
                    xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u64,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
