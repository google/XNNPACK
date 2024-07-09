// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x8-packq.yaml
//   Generator: tools/generate-packq-test.py


#include <benchmark/benchmark.h>
#include "bench/bgemm.h"
#include "bench/packq-benchmark.h"
#include "xnnpack/common.h"
#include "xnnpack/packq.h"


#if XNN_ARCH_ARM64
  #if XNN_ENABLE_KLEIDIAI
  static void x8_packq_f32qp8_ukernel__aarch64_neon_u2_mr_1_kr_1(
      benchmark::State& state, const char* net) {
    x8_packq(state,
      xnn_x8_packq_f32qp8_ukernel__aarch64_neon_u2,
      /*mr=*/1, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x8_packq_f32qp8_ukernel__aarch64_neon_u2_mr_1_kr_1)
  static void x8_packq_f32qp8_ukernel__aarch64_neon_u2_mr_1_kr_2(
      benchmark::State& state, const char* net) {
    x8_packq(state,
      xnn_x8_packq_f32qp8_ukernel__aarch64_neon_u2,
      /*mr=*/1, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x8_packq_f32qp8_ukernel__aarch64_neon_u2_mr_1_kr_2)
  static void x8_packq_f32qp8_ukernel__aarch64_neon_u2_mr_1_kr_4(
      benchmark::State& state, const char* net) {
    x8_packq(state,
      xnn_x8_packq_f32qp8_ukernel__aarch64_neon_u2,
      /*mr=*/1, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x8_packq_f32qp8_ukernel__aarch64_neon_u2_mr_1_kr_4)
  static void x8_packq_f32qp8_ukernel__aarch64_neon_u2_mr_2_kr_1(
      benchmark::State& state, const char* net) {
    x8_packq(state,
      xnn_x8_packq_f32qp8_ukernel__aarch64_neon_u2,
      /*mr=*/2, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x8_packq_f32qp8_ukernel__aarch64_neon_u2_mr_2_kr_1)
  static void x8_packq_f32qp8_ukernel__aarch64_neon_u2_mr_2_kr_2(
      benchmark::State& state, const char* net) {
    x8_packq(state,
      xnn_x8_packq_f32qp8_ukernel__aarch64_neon_u2,
      /*mr=*/2, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x8_packq_f32qp8_ukernel__aarch64_neon_u2_mr_2_kr_2)
  static void x8_packq_f32qp8_ukernel__aarch64_neon_u2_mr_2_kr_4(
      benchmark::State& state, const char* net) {
    x8_packq(state,
      xnn_x8_packq_f32qp8_ukernel__aarch64_neon_u2,
      /*mr=*/2, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x8_packq_f32qp8_ukernel__aarch64_neon_u2_mr_2_kr_4)
  static void x8_packq_f32qp8_ukernel__aarch64_neon_u2_mr_4_kr_1(
      benchmark::State& state, const char* net) {
    x8_packq(state,
      xnn_x8_packq_f32qp8_ukernel__aarch64_neon_u2,
      /*mr=*/4, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x8_packq_f32qp8_ukernel__aarch64_neon_u2_mr_4_kr_1)
  static void x8_packq_f32qp8_ukernel__aarch64_neon_u2_mr_4_kr_2(
      benchmark::State& state, const char* net) {
    x8_packq(state,
      xnn_x8_packq_f32qp8_ukernel__aarch64_neon_u2,
      /*mr=*/4, /*kr=*/2, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x8_packq_f32qp8_ukernel__aarch64_neon_u2_mr_4_kr_2)
  static void x8_packq_f32qp8_ukernel__aarch64_neon_u2_mr_4_kr_4(
      benchmark::State& state, const char* net) {
    x8_packq(state,
      xnn_x8_packq_f32qp8_ukernel__aarch64_neon_u2,
      /*mr=*/4, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x8_packq_f32qp8_ukernel__aarch64_neon_u2_mr_4_kr_4)
  #endif  // XNN_ENABLE_KLEIDIAI
#endif  // XNN_ARCH_ARM64


static void x8_packq_f32qp8_ukernel__scalar_u1_mr_1_kr_1(
    benchmark::State& state, const char* net) {
  x8_packq(state,
    xnn_x8_packq_f32qp8_ukernel__scalar_u1,
    /*mr=*/1, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x8_packq_f32qp8_ukernel__scalar_u1_mr_1_kr_1)
static void x8_packq_f32qp8_ukernel__scalar_u1_mr_1_kr_2(
    benchmark::State& state, const char* net) {
  x8_packq(state,
    xnn_x8_packq_f32qp8_ukernel__scalar_u1,
    /*mr=*/1, /*kr=*/2, /*sr=*/1);
}
BENCHMARK_BGEMM(x8_packq_f32qp8_ukernel__scalar_u1_mr_1_kr_2)
static void x8_packq_f32qp8_ukernel__scalar_u1_mr_1_kr_4(
    benchmark::State& state, const char* net) {
  x8_packq(state,
    xnn_x8_packq_f32qp8_ukernel__scalar_u1,
    /*mr=*/1, /*kr=*/4, /*sr=*/1);
}
BENCHMARK_BGEMM(x8_packq_f32qp8_ukernel__scalar_u1_mr_1_kr_4)
static void x8_packq_f32qp8_ukernel__scalar_u1_mr_2_kr_1(
    benchmark::State& state, const char* net) {
  x8_packq(state,
    xnn_x8_packq_f32qp8_ukernel__scalar_u1,
    /*mr=*/2, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x8_packq_f32qp8_ukernel__scalar_u1_mr_2_kr_1)
static void x8_packq_f32qp8_ukernel__scalar_u1_mr_2_kr_2(
    benchmark::State& state, const char* net) {
  x8_packq(state,
    xnn_x8_packq_f32qp8_ukernel__scalar_u1,
    /*mr=*/2, /*kr=*/2, /*sr=*/1);
}
BENCHMARK_BGEMM(x8_packq_f32qp8_ukernel__scalar_u1_mr_2_kr_2)
static void x8_packq_f32qp8_ukernel__scalar_u1_mr_2_kr_4(
    benchmark::State& state, const char* net) {
  x8_packq(state,
    xnn_x8_packq_f32qp8_ukernel__scalar_u1,
    /*mr=*/2, /*kr=*/4, /*sr=*/1);
}
BENCHMARK_BGEMM(x8_packq_f32qp8_ukernel__scalar_u1_mr_2_kr_4)
static void x8_packq_f32qp8_ukernel__scalar_u1_mr_4_kr_1(
    benchmark::State& state, const char* net) {
  x8_packq(state,
    xnn_x8_packq_f32qp8_ukernel__scalar_u1,
    /*mr=*/4, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x8_packq_f32qp8_ukernel__scalar_u1_mr_4_kr_1)
static void x8_packq_f32qp8_ukernel__scalar_u1_mr_4_kr_2(
    benchmark::State& state, const char* net) {
  x8_packq(state,
    xnn_x8_packq_f32qp8_ukernel__scalar_u1,
    /*mr=*/4, /*kr=*/2, /*sr=*/1);
}
BENCHMARK_BGEMM(x8_packq_f32qp8_ukernel__scalar_u1_mr_4_kr_2)
static void x8_packq_f32qp8_ukernel__scalar_u1_mr_4_kr_4(
    benchmark::State& state, const char* net) {
  x8_packq(state,
    xnn_x8_packq_f32qp8_ukernel__scalar_u1,
    /*mr=*/4, /*kr=*/4, /*sr=*/1);
}
BENCHMARK_BGEMM(x8_packq_f32qp8_ukernel__scalar_u1_mr_4_kr_4)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
