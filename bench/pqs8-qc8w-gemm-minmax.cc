// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/pqs8-qc8w-gemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py

#include <benchmark/benchmark.h>
#include "bench/gemm-benchmark.h"
#include "bench/utils.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/packw.h"


#if XNN_ENABLE_ARM_SME2 && XNN_ARCH_ARM64
  #if XNN_ENABLE_KLEIDIAI
  static void pqs8_qc8w_gemm_minmax_ukernel_1x32c4__neonsme2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_pqs8_qc8w_gemm_minmax_ukernel_1x32c4__neonsme2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_kai_qs8_qc8w_weights_and_biases_sme2,
      xnn_packed_stride_kai_qs8_qc8w_weights_and_biases_sme2,
      /*mr=*/1, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*mr_packed=*/1,
      benchmark::utils::CheckNEONSME2);
  }

  BENCHMARK_GEMM(pqs8_qc8w_gemm_minmax_ukernel_1x32c4__neonsme2)

  static void pqs8_qc8w_gemm_minmax_ukernel_32x32c4__neonsme2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_pqs8_qc8w_gemm_minmax_ukernel_32x32c4__neonsme2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_kai_qs8_qc8w_weights_and_biases_sme2,
      xnn_packed_stride_kai_qs8_qc8w_weights_and_biases_sme2,
      /*mr=*/32, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*mr_packed=*/32,
      benchmark::utils::CheckNEONSME2);
  }

  BENCHMARK_GEMM(pqs8_qc8w_gemm_minmax_ukernel_32x32c4__neonsme2)
  #endif  // XNN_ENABLE_KLEIDIAI
#endif  // XNN_ENABLE_ARM_SME2 && XNN_ARCH_ARM64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
