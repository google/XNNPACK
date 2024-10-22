// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack.h"

#include "unary_operator.h"
#include "utils.h"
#include <benchmark/benchmark.h>
#ifdef BENCHMARK_TENSORFLOW_LITE
#include "tensorflow/lite/schema/schema_generated.h"
#endif  // BENCHMARK_TENSORFLOW_LITE

static void xnnpack_reciprocal_square_root_f32(benchmark::State& state) {
  benchmark_unary_operator<float, float>(
      xnn_create_reciprocal_square_root_nc_f32,
      xnn_reshape_reciprocal_square_root_nc_f32,
      xnn_setup_reciprocal_square_root_nc_f32, state);
}

BENCHMARK(xnnpack_reciprocal_square_root_f32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

#ifdef BENCHMARK_TENSORFLOW_LITE

static void tflite_reciprocal_square_root_f32(benchmark::State& state) {
  benchmark_tflite_unary_operator<float, float>(state,
                                                tflite::BuiltinOperator_RSQRT);
}

BENCHMARK(tflite_reciprocal_square_root_f32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // BENCHMARK_TENSORFLOW_LITE

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
