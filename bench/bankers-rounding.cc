// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "unary_operator.h"
#include "utils.h"
#include "xnnpack.h"
#include "xnnpack/math.h"
#include <benchmark/benchmark.h>
#ifdef BENCHMARK_TENSORFLOW_LITE
#include "tensorflow/lite/schema/schema_generated.h"
#endif  // BENCHMARK_TENSORFLOW_LITE

static void xnnpack_bankers_rounding_f16(benchmark::State& state) {
  benchmark_unary_operator<xnn_float16, xnn_float16>(
      xnn_create_bankers_rounding_nc_f16, xnn_reshape_bankers_rounding_nc_f16,
      xnn_setup_bankers_rounding_nc_f16, state);
}

static void xnnpack_bankers_rounding_f32(benchmark::State& state) {
  benchmark_unary_operator<float, float>(
      xnn_create_bankers_rounding_nc_f32, xnn_reshape_bankers_rounding_nc_f32,
      xnn_setup_bankers_rounding_nc_f32, state);
}

BENCHMARK(xnnpack_bankers_rounding_f16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
    ->UseRealTime();
BENCHMARK(xnnpack_bankers_rounding_f32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

#ifdef BENCHMARK_TENSORFLOW_LITE

static void tflite_bankers_rounding_f32(benchmark::State& state) {
  benchmark_tflite_unary_operator<float, float>(state,
                                                tflite::BuiltinOperator_ROUND);
}

BENCHMARK(tflite_bankers_rounding_f32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // BENCHMARK_TENSORFLOW_LITE

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
