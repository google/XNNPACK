// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qp8-f32-qc8w-gemm-minmax.yaml
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


#if XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  #if XNN_ENABLE_KLEIDIAI
  static void qp8_f32_qc8w_gemm_minmax_ukernel_16x4c8__neoni8mm_mstep4(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x4c8__neoni8mm_mstep4,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_kai_qs8_weights_and_biases,
      xnn_packed_stride_kai_qs8_weights_and_biases,
      /*mr=*/16, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*mr_packed=*/4,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qp8_f32_qc8w_gemm_minmax_ukernel_16x4c8__neoni8mm_mstep4)
  #endif  // XNN_ENABLE_KLEIDIAI
#endif  // XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64
  #if XNN_ENABLE_KLEIDIAI
  static void qp8_f32_qc8w_gemm_minmax_ukernel_1x4c4__aarch64_neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x4c4__aarch64_neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_kai_qs8_weights_and_biases,
      xnn_packed_stride_kai_qs8_weights_and_biases,
      /*mr=*/1, /*nr=*/4, /*kr=*/4, /*sr=*/1,
      /*mr_packed=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qp8_f32_qc8w_gemm_minmax_ukernel_1x4c4__aarch64_neondot)

  static void qp8_f32_qc8w_gemm_minmax_ukernel_1x4c8__aarch64_neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x4c8__aarch64_neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_kai_qs8_weights_and_biases,
      xnn_packed_stride_kai_qs8_weights_and_biases,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*mr_packed=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qp8_f32_qc8w_gemm_minmax_ukernel_1x4c8__aarch64_neondot)

  static void qp8_f32_qc8w_gemm_minmax_ukernel_16x4c4__aarch64_neondot_mstep4(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x4c4__aarch64_neondot_mstep4,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_kai_qs8_weights_and_biases,
      xnn_packed_stride_kai_qs8_weights_and_biases,
      /*mr=*/16, /*nr=*/4, /*kr=*/4, /*sr=*/1,
      /*mr_packed=*/4,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qp8_f32_qc8w_gemm_minmax_ukernel_16x4c4__aarch64_neondot_mstep4)
  #endif  // XNN_ENABLE_KLEIDIAI
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
