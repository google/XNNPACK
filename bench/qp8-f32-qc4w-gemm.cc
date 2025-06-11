// clang-format off
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
#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/packw.h"


#if XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  #if XNN_ENABLE_KLEIDIAI
  static void qp8_f32_qc4w_gemm_minmax_ukernel_4x4c16s2__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qp8_f32_qc4w_gemm_minmax_ukernel_4x4c16s2__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_kai_qs4_weights_and_biases,
      xnn_packed_stride_kai_qs4_weights_and_biases,
      /*mr=*/4, /*nr=*/4, /*kr=*/16, /*sr=*/2,
      /*mr_packed=*/4,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qp8_f32_qc4w_gemm_minmax_ukernel_4x4c16s2__neoni8mm)

  static void qp8_f32_qc4w_gemm_minmax_ukernel_4x8c16s2__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qp8_f32_qc4w_gemm_minmax_ukernel_4x8c16s2__neoni8mm,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_kai_qs4_weights_and_biases,
      xnn_packed_stride_kai_qs4_weights_and_biases,
      /*mr=*/4, /*nr=*/8, /*kr=*/16, /*sr=*/2,
      /*mr_packed=*/4,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qp8_f32_qc4w_gemm_minmax_ukernel_4x8c16s2__neoni8mm)

  static void qp8_f32_qc4w_gemm_minmax_ukernel_8x4c16s2__neoni8mm_mstep2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qp8_f32_qc4w_gemm_minmax_ukernel_8x4c16s2__neoni8mm_mstep2,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_kai_qs4_weights_and_biases,
      xnn_packed_stride_kai_qs4_weights_and_biases,
      /*mr=*/8, /*nr=*/4, /*kr=*/16, /*sr=*/2,
      /*mr_packed=*/4,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qp8_f32_qc4w_gemm_minmax_ukernel_8x4c16s2__neoni8mm_mstep2)

  static void qp8_f32_qc4w_gemm_minmax_ukernel_8x8c16s2__neoni8mm_mstep2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qp8_f32_qc4w_gemm_minmax_ukernel_8x8c16s2__neoni8mm_mstep2,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_kai_qs4_weights_and_biases,
      xnn_packed_stride_kai_qs4_weights_and_biases,
      /*mr=*/8, /*nr=*/8, /*kr=*/16, /*sr=*/2,
      /*mr_packed=*/4,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qp8_f32_qc4w_gemm_minmax_ukernel_8x8c16s2__neoni8mm_mstep2)
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
      /*arch_flags=*/xnn_arch_arm_neon_dot);
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
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qp8_f32_qc4w_gemm_minmax_ukernel_1x8c16s2__aarch64_neondot)

  static void qp8_f32_qc4w_gemm_minmax_ukernel_1x4c8s2__aarch64_neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x4c8s2__aarch64_neondot,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_kai_qs4_weights_and_biases,
      xnn_packed_stride_kai_qs4_weights_and_biases,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/2,
      /*mr_packed=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qp8_f32_qc4w_gemm_minmax_ukernel_1x4c8s2__aarch64_neondot)

  static void qp8_f32_qc4w_gemm_minmax_ukernel_16x4c8s2__aarch64_neondot_mstep4(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x4c8s2__aarch64_neondot_mstep4,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_kai_qs4_weights_and_biases,
      xnn_packed_stride_kai_qs4_weights_and_biases,
      /*mr=*/16, /*nr=*/4, /*kr=*/8, /*sr=*/2,
      /*mr_packed=*/4,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qp8_f32_qc4w_gemm_minmax_ukernel_16x4c8s2__aarch64_neondot_mstep4)
  #endif  // XNN_ENABLE_KLEIDIAI
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_SME2 && XNN_ARCH_ARM64
  #if XNN_ENABLE_KLEIDIAI
  static void qp8_f32_qc4w_gemm_minmax_ukernel_1x128c4__neonsme2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x128c4__neonsme2,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_kai_qs4_weights_and_biases_sme,
      xnn_packed_stride_kai_qs4_weights_and_biases_sme,
      /*mr=*/1, /*nr=*/128, /*kr=*/4, /*sr=*/1,
      /*mr_packed=*/1,
      /*arch_flags=*/xnn_arch_arm_sme2);
  }

  BENCHMARK_GEMM(qp8_f32_qc4w_gemm_minmax_ukernel_1x128c4__neonsme2)

  static void qp8_f32_qc4w_gemm_minmax_ukernel_32x128c4__neonsme2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qp8_f32_qc4w_gemm_minmax_ukernel_32x128c4__neonsme2,
      xnn_init_f32_minmax_scalar_params,
      xnn_pack_kai_qs4_weights_and_biases_sme,
      xnn_packed_stride_kai_qs4_weights_and_biases_sme,
      /*mr=*/32, /*nr=*/128, /*kr=*/4, /*sr=*/1,
      /*mr_packed=*/32,
      /*arch_flags=*/xnn_arch_arm_sme2);
  }

  BENCHMARK_GEMM(qp8_f32_qc4w_gemm_minmax_ukernel_32x128c4__neonsme2)
  #endif  // XNN_ENABLE_KLEIDIAI
#endif  // XNN_ENABLE_ARM_SME2 && XNN_ARCH_ARM64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
