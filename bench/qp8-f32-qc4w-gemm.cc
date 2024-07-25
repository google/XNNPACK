// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qp8-f32-qc4w-gemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py

#include <benchmark/benchmark.h>
#include "bench/gemm-benchmark.h"
#include "bench/utils.h"
#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pack.h"
#include "xnnpack/packw.h"


#if XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  #if XNN_ENABLE_KLEIDIAI
  static void qp8_f32_qc4w_gemm_minmax_ukernel_4x4c16s2__aarch64_neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qp8_f32_qc4w_gemm_minmax_ukernel_4x4c16s2__aarch64_neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_kai_qs4_weights_and_biases,
      xnn_packed_stride_kai_qs4_weights_and_biases,
      /*mr=*/4, /*nr=*/4, /*kr=*/16, /*sr=*/2,
      /*mr_packed=*/4,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qp8_f32_qc4w_gemm_minmax_ukernel_4x4c16s2__aarch64_neoni8mm)

  static void qp8_f32_qc4w_gemm_minmax_ukernel_4x8c16s2__aarch64_neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qp8_f32_qc4w_gemm_minmax_ukernel_4x8c16s2__aarch64_neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_kai_qs4_weights_and_biases,
      xnn_packed_stride_kai_qs4_weights_and_biases,
      /*mr=*/4, /*nr=*/8, /*kr=*/16, /*sr=*/2,
      /*mr_packed=*/4,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qp8_f32_qc4w_gemm_minmax_ukernel_4x8c16s2__aarch64_neoni8mm)

  static void qp8_f32_qc4w_gemm_minmax_ukernel_8x4c16s2__aarch64_neoni8mm_mstep2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qp8_f32_qc4w_gemm_minmax_ukernel_8x4c16s2__aarch64_neoni8mm_mstep2,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_kai_qs4_weights_and_biases,
      xnn_packed_stride_kai_qs4_weights_and_biases,
      /*mr=*/8, /*nr=*/4, /*kr=*/16, /*sr=*/2,
      /*mr_packed=*/4,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qp8_f32_qc4w_gemm_minmax_ukernel_8x4c16s2__aarch64_neoni8mm_mstep2)

  static void qp8_f32_qc4w_gemm_minmax_ukernel_8x8c16s2__aarch64_neoni8mm_mstep2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qp8_f32_qc4w_gemm_minmax_ukernel_8x8c16s2__aarch64_neoni8mm_mstep2,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_kai_qs4_weights_and_biases,
      xnn_packed_stride_kai_qs4_weights_and_biases,
      /*mr=*/8, /*nr=*/8, /*kr=*/16, /*sr=*/2,
      /*mr_packed=*/4,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_GEMM(qp8_f32_qc4w_gemm_minmax_ukernel_8x8c16s2__aarch64_neoni8mm_mstep2)
  #endif  // XNN_ENABLE_KLEIDIAI
#endif  // XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64
  #if XNN_ENABLE_KLEIDIAI
  static void qp8_f32_qc4w_gemm_minmax_ukernel_1x4c16s2__aarch64_neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x4c16s2__aarch64_neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_kai_qs4_weights_and_biases,
      xnn_packed_stride_kai_qs4_weights_and_biases,
      /*mr=*/1, /*nr=*/4, /*kr=*/16, /*sr=*/2,
      /*mr_packed=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qp8_f32_qc4w_gemm_minmax_ukernel_1x4c16s2__aarch64_neondot)

  static void qp8_f32_qc4w_gemm_minmax_ukernel_1x8c16s2__aarch64_neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x8c16s2__aarch64_neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_kai_qs4_weights_and_biases,
      xnn_packed_stride_kai_qs4_weights_and_biases,
      /*mr=*/1, /*nr=*/8, /*kr=*/16, /*sr=*/2,
      /*mr_packed=*/1,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qp8_f32_qc4w_gemm_minmax_ukernel_1x8c16s2__aarch64_neondot)
  #endif  // XNN_ENABLE_KLEIDIAI
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
