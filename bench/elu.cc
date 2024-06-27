// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack.h"

#include <cstdint>
#include <limits>

#include "unary_operator.h"
#include "bench/utils.h"
#include <benchmark/benchmark.h>
#ifdef BENCHMARK_TENSORFLOW_LITE
#include "flatbuffers/include/flatbuffers/flatbuffer_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#endif  // BENCHMARK_TENSORFLOW_LITE


static void xnnpack_elu_f16(benchmark::State& state) {
  benchmark_unary_operator<float16, float16>(
      [](uint32_t flags, xnn_operator_t* op) {
        return xnn_create_elu_nc_f16(
            /*alpha=*/1.0f, flags, op);
      },
      xnn_reshape_elu_nc_f16, xnn_setup_elu_nc_f16, state);
}

static void xnnpack_elu_f32(benchmark::State& state) {
  benchmark_unary_operator<float, float>(
      [](uint32_t flags, xnn_operator_t* op) {
        return xnn_create_elu_nc_f32(
            /*alpha=*/1.0f, flags, op);
      },
      xnn_reshape_elu_nc_f32, xnn_setup_elu_nc_f32, state);
}

static void xnnpack_elu_qs8(benchmark::State& state) {
  benchmark_unary_operator<int8_t, int8_t>(
      [](uint32_t flags, xnn_operator_t* op) {
        return xnn_create_elu_nc_qs8(
            1.0f /* alpha */, 0 /* input zero point */, 1.0f /* input scale */,
            0 /* output zero point */, 1.0f /* output scale */,
            std::numeric_limits<int8_t>::min(),
            std::numeric_limits<int8_t>::max(), flags, op);
      },
      xnn_reshape_elu_nc_qs8, xnn_setup_elu_nc_qs8, state);
}

BENCHMARK(xnnpack_elu_f16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
BENCHMARK(xnnpack_elu_f32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
BENCHMARK(xnnpack_elu_qs8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();

BENCHMARK(xnnpack_elu_f16)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
  ->UseRealTime();
BENCHMARK(xnnpack_elu_f32)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK(xnnpack_elu_qs8)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
  ->UseRealTime();

#ifdef BENCHMARK_TENSORFLOW_LITE
static void tflite_elu_f32(benchmark::State& state) {
  benchmark_tflite_unary_operator<float, float>(state,
                                                tflite::BuiltinOperator_ELU);
}

static void tflite_elu_qs8(benchmark::State& state) {
  benchmark_tflite_unary_operator<int8_t, int8_t>(
      state,
      [](flatbuffers::FlatBufferBuilder& builder) {
        return tflite::CreateQuantizationParameters(
            builder, 0 /*min*/, 0 /*max*/,
            builder.CreateVector<float>({1.0f /* scale */}),
            builder.CreateVector<int64_t>({1 /* zero point */}));
      },
      [](flatbuffers::FlatBufferBuilder& builder) {
        return tflite::CreateQuantizationParameters(
            builder, 0 /*min*/, 0 /*max*/,
            builder.CreateVector<float>({1.0f /* scale */}),
            builder.CreateVector<int64_t>({1 /* zero point */}));
      },
      tflite::BuiltinOperator_ELU);
}

  BENCHMARK(tflite_elu_f32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK(tflite_elu_qs8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
#endif  // BENCHMARK_TENSORFLOW_LITE

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
