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


static void xnnpack_leaky_relu_f16(benchmark::State& state) {
  benchmark_unary_operator<float16, float16>(
      [](uint32_t flags, xnn_operator_t* op) {
        return xnn_create_leaky_relu_nc_f16(

            0.01f /* negative slope */, flags, op);
      },
      xnn_reshape_leaky_relu_nc_f16, xnn_setup_leaky_relu_nc_f16, state);
}

static void xnnpack_leaky_relu_f32(benchmark::State& state) {
  benchmark_unary_operator<float, float>(
      [](uint32_t flags, xnn_operator_t* op) {
        return xnn_create_leaky_relu_nc_f32(

            0.01f /* negative slope */, flags, op);
      },
      xnn_reshape_leaky_relu_nc_f32, xnn_setup_leaky_relu_nc_f32, state);
}

static void xnnpack_leaky_relu_qs8(benchmark::State& state) {
  benchmark_unary_operator<int8_t, int8_t>(
      [](uint32_t flags, xnn_operator_t* op) {
        return xnn_create_leaky_relu_nc_qs8(
            0.1f /* negative slope */, 5 /* input zero point */,
            0.75f /* input scale */, -5 /* output zero point */,
            0.5f /* output scale */, flags, op);
      },
      xnn_reshape_leaky_relu_nc_qs8, xnn_setup_leaky_relu_nc_qs8, state);
}

static void xnnpack_leaky_relu_qu8(benchmark::State& state) {
  benchmark_unary_operator<uint8_t, uint8_t>(
      [](uint32_t flags, xnn_operator_t* op) {
        return xnn_create_leaky_relu_nc_qu8(
            0.1f /* negative slope */, 5 /* input zero point */,
            0.75f /* input scale */, -5 /* output zero point */,
            0.5f /* output scale */, flags, op);
      },
      xnn_reshape_leaky_relu_nc_qu8, xnn_setup_leaky_relu_nc_qu8, state);
}

BENCHMARK(xnnpack_leaky_relu_f16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
BENCHMARK(xnnpack_leaky_relu_f32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
BENCHMARK(xnnpack_leaky_relu_qs8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
BENCHMARK(xnnpack_leaky_relu_qu8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();

BENCHMARK(xnnpack_leaky_relu_f16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
BENCHMARK(xnnpack_leaky_relu_f32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
BENCHMARK(xnnpack_leaky_relu_qs8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
BENCHMARK(xnnpack_leaky_relu_qu8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();

#ifdef BENCHMARK_TENSORFLOW_LITE
static void tflite_leaky_relu_f32(benchmark::State& state) {
  benchmark_tflite_unary_operator<float, float>(
      state, tflite::BuiltinOperator_LEAKY_RELU);
}

static void tflite_leaky_relu_qs8(benchmark::State& state) {
  benchmark_tflite_unary_operator<int8_t, int8_t>(
      state,
      [](flatbuffers::FlatBufferBuilder& builder) {
        return tflite::CreateQuantizationParameters(
            builder, 0 /*min*/, 0 /*max*/,
            builder.CreateVector<float>({0.75f /* scale */}),
            builder.CreateVector<int64_t>({5 /* zero point */}));
      },
      [](flatbuffers::FlatBufferBuilder& builder) {
        return tflite::CreateQuantizationParameters(
            builder, 0 /*min*/, 0 /*max*/,
            builder.CreateVector<float>({0.5f /* scale */}),
            builder.CreateVector<int64_t>({-5 /* zero point */}));
      },
      tflite::BuiltinOperator_LEAKY_RELU);
}

static void tflite_leaky_relu_qu8(benchmark::State& state) {
  benchmark_tflite_unary_operator<int8_t, int8_t>(
      state,
      [](flatbuffers::FlatBufferBuilder& builder) {
        return tflite::CreateQuantizationParameters(
            builder, 0 /*min*/, 0 /*max*/,
            builder.CreateVector<float>({0.75f /* scale */}),
            builder.CreateVector<int64_t>({133 /* zero point */}));
      },
      [](flatbuffers::FlatBufferBuilder& builder) {
        return tflite::CreateQuantizationParameters(
            builder, 0 /*min*/, 0 /*max*/,
            builder.CreateVector<float>({0.5f /* scale */}),
            builder.CreateVector<int64_t>({123 /* zero point */}));
      },
      tflite::BuiltinOperator_LEAKY_RELU);
}

  BENCHMARK(tflite_leaky_relu_f32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK(tflite_leaky_relu_qs8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK(tflite_leaky_relu_qu8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
#endif  // BENCHMARK_TENSORFLOW_LITE

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
