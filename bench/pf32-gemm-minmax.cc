// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/pf32-gemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py

#include <benchmark/benchmark.h>
#include "gemm-benchmark.h"
#include "utils.h"
#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pack.h"
#include "xnnpack/packw.h"


#if XNN_ENABLE_ARM_SME2 && XNN_ARCH_ARM64
  #if XNN_ENABLE_KLEIDIAI
  static void pf32_gemm_minmax_ukernel_1x32__neonsme2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_pf32_gemm_minmax_ukernel_1x32__neonsme2,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_kai_f32_weights_and_biases,
      xnn_packed_stride_kai_f32_weights_and_biases,
      /*mr=*/1, /*nr=*/32, /*kr=*/1, /*sr=*/1,
      /*mr_packed=*/32,
      benchmark::utils::CheckNEONSME2);
  }

  BENCHMARK_GEMM(pf32_gemm_minmax_ukernel_1x32__neonsme2)

  static void pf32_gemm_minmax_ukernel_32x32__neonsme2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_kai_f32_weights_and_biases,
      xnn_packed_stride_kai_f32_weights_and_biases,
      /*mr=*/32, /*nr=*/32, /*kr=*/1, /*sr=*/1,
      /*mr_packed=*/32,
      benchmark::utils::CheckNEONSME2);
  }

  BENCHMARK_GEMM(pf32_gemm_minmax_ukernel_32x32__neonsme2)
  #endif  // XNN_ENABLE_KLEIDIAI
#endif  // XNN_ENABLE_ARM_SME2 && XNN_ARCH_ARM64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
