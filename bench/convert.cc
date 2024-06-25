// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack.h"

#include <limits>

#include "unary_operator.h"
#include "bench/utils.h"
#include <benchmark/benchmark.h>
#ifdef BENCHMARK_TENSORFLOW_LITE
#include "flatbuffers/include/flatbuffers/flatbuffers.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#endif  // BENCHMARK_TENSORFLOW_LITE


void xnnpack_convert_f16_f32(benchmark::State& state) {
  benchmark_unary_operator<float16, float>(xnn_create_convert_nc_f16_f32,
                                           xnn_reshape_convert_nc_f16_f32,
                                           xnn_setup_convert_nc_f16_f32, state);
}

void xnnpack_convert_f32_f16(benchmark::State& state) {
  benchmark_unary_operator<float, float16>(xnn_create_convert_nc_f32_f16,
                                           xnn_reshape_convert_nc_f32_f16,
                                           xnn_setup_convert_nc_f32_f16, state);
}

void xnnpack_convert_f32_qs8(benchmark::State& state) {
  benchmark_unary_operator<float, int8_t>(
      [](uint32_t flags, xnn_operator_t* op) {
        return xnn_create_convert_nc_f32_qs8(
            1.0f / 128.0f /* scale */, 1 /* zero point */,
            std::numeric_limits<int8_t>::min(),
            std::numeric_limits<int8_t>::max(), flags, op);
      },
      xnn_reshape_convert_nc_f32_qs8, xnn_setup_convert_nc_f32_qs8, state);
}

void xnnpack_convert_f32_qu8(benchmark::State& state) {
  benchmark_unary_operator<float, uint8_t>(
      [](uint32_t flags, xnn_operator_t* op) {
        return xnn_create_convert_nc_f32_qu8(
            1.0f / 128.0f /* scale */, 127 /* zero point */,
            std::numeric_limits<uint8_t>::min(),
            std::numeric_limits<uint8_t>::max(), flags, op);
      },
      xnn_reshape_convert_nc_f32_qu8, xnn_setup_convert_nc_f32_qu8, state);
}

void xnnpack_convert_qs8(benchmark::State& state) {
  benchmark_unary_operator<int8_t, int8_t>(
      [](uint32_t flags, xnn_operator_t* op) {
        return xnn_create_convert_nc_qs8(
            0.75f /* input scale */, -1 /* input zero point */,
            0.5f /* output scale */, 1 /* output zero point */, flags, op);
      },
      xnn_reshape_convert_nc_qs8, xnn_setup_convert_nc_qs8, state);
}

void xnnpack_convert_qs8_f32(benchmark::State& state) {
  benchmark_unary_operator<int8_t, float>(
      [](uint32_t flags, xnn_operator_t* op) {
        return xnn_create_convert_nc_qs8_f32(1.0f / 255.0f /* scale */,
                                             -128 /* zero point */, flags, op);
      },
      xnn_reshape_convert_nc_qs8_f32, xnn_setup_convert_nc_qs8_f32, state);
}

void xnnpack_convert_qu8(benchmark::State& state) {
  benchmark_unary_operator<uint8_t, uint8_t>(
      [](uint32_t flags, xnn_operator_t* op) {
        return xnn_create_convert_nc_qu8(0.75f /* scale */,
                                         125 /* zero point */, 0.5f /* scale */,
                                         130 /* zero point */, flags, op);
      },
      xnn_reshape_convert_nc_qu8, xnn_setup_convert_nc_qu8, state);
}

void xnnpack_convert_qu8_f32(benchmark::State& state) {
  benchmark_unary_operator<uint8_t, float>(
      [](uint32_t flags, xnn_operator_t* op) {
        return xnn_create_convert_nc_qu8_f32(1.0f / 128.0f /* scale */,
                                             128 /* zero point */, flags, op);
      },
      xnn_reshape_convert_nc_qu8_f32, xnn_setup_convert_nc_qu8_f32, state);
}

#ifdef BENCHMARK_TENSORFLOW_LITE
void tflite_convert_f16_f32(benchmark::State& state) {
  benchmark_tflite_unary_operator<float16, float>(
      state, tflite::BuiltinOperator_DEQUANTIZE);
}

void tflite_convert_f32_qs8(benchmark::State& state) {
  benchmark_tflite_unary_operator<float, int8_t>(
      state, no_quantization,
      [](flatbuffers::FlatBufferBuilder& builder) {
        return tflite::CreateQuantizationParameters(
            builder, 0 /*min*/, 0 /*max*/,
            builder.CreateVector<float>({1.0f / 128.0f /* scale */}),
            builder.CreateVector<int64_t>({1 /* zero point */}));
      },
      tflite::BuiltinOperator_QUANTIZE);
}

void tflite_convert_f32_qu8(benchmark::State& state) {
  benchmark_tflite_unary_operator<float, uint8_t>(
      state, no_quantization,
      [](flatbuffers::FlatBufferBuilder& builder) {
        return tflite::CreateQuantizationParameters(
            builder, 0 /*min*/, 0 /*max*/,
            builder.CreateVector<float>({1.0f / 128.0f /* scale */}),
            builder.CreateVector<int64_t>({127 /* zero point */}));
      },
      tflite::BuiltinOperator_QUANTIZE);
}

void tflite_convert_qs8(benchmark::State& state) {
  benchmark_tflite_unary_operator<int8_t, int8_t>(
      state,
      [](flatbuffers::FlatBufferBuilder& builder) {
        return tflite::CreateQuantizationParameters(
            builder, 0 /*min*/, 0 /*max*/,
            builder.CreateVector<float>({0.75f /* scale */}),
            builder.CreateVector<int64_t>({-1 /* zero point */}));
      },
      [](flatbuffers::FlatBufferBuilder& builder) {
        return tflite::CreateQuantizationParameters(
            builder, 0 /*min*/, 0 /*max*/,
            builder.CreateVector<float>({0.5f /* scale */}),
            builder.CreateVector<int64_t>({1 /* zero point */}));
      },
      tflite::BuiltinOperator_QUANTIZE);
}

void tflite_convert_qs8_f32(benchmark::State& state) {
  benchmark_tflite_unary_operator<int8_t, float>(
      state,
      [](flatbuffers::FlatBufferBuilder& builder) {
        return tflite::CreateQuantizationParameters(
            builder, 0 /*min*/, 0 /*max*/,
            builder.CreateVector<float>({1.0f / 255.0f /* scale */}),
            builder.CreateVector<int64_t>({-128 /* zero point */}));
      },
      no_quantization, tflite::BuiltinOperator_DEQUANTIZE);
}

void tflite_convert_qu8(benchmark::State& state) {
  benchmark_tflite_unary_operator<uint8_t, uint8_t>(
      state,
      [](flatbuffers::FlatBufferBuilder& builder) {
        return tflite::CreateQuantizationParameters(
            builder, 0 /*min*/, 0 /*max*/,
            builder.CreateVector<float>({0.75f /* scale */}),
            builder.CreateVector<int64_t>({125 /* zero point */}));
      },
      [](flatbuffers::FlatBufferBuilder& builder) {
        return tflite::CreateQuantizationParameters(
            builder, 0 /*min*/, 0 /*max*/,
            builder.CreateVector<float>({0.5f /* scale */}),
            builder.CreateVector<int64_t>({130 /* zero point */}));
      },
      tflite::BuiltinOperator_QUANTIZE);
}

void tflite_convert_qu8_f32(benchmark::State& state) {
  benchmark_tflite_unary_operator<uint8_t, float>(
      state,
      [](flatbuffers::FlatBufferBuilder& builder) {
        return tflite::CreateQuantizationParameters(
            builder, 0 /*min*/, 0 /*max*/,
            builder.CreateVector<float>({1.0f / 128.0f /* scale */}),
            builder.CreateVector<int64_t>({128 /* zero point */}));
      },
      no_quantization, tflite::BuiltinOperator_DEQUANTIZE);
}
#endif  // BENCHMARK_TENSORFLOW_LITE

BENCHMARK(xnnpack_convert_f16_f32)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
  ->UseRealTime();
BENCHMARK(xnnpack_convert_f32_f16)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
  ->UseRealTime();
BENCHMARK(xnnpack_convert_f32_qs8)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, int8_t>)
  ->UseRealTime();
BENCHMARK(xnnpack_convert_f32_qu8)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();
BENCHMARK(xnnpack_convert_qs8)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
  ->UseRealTime();
BENCHMARK(xnnpack_convert_qs8_f32)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
  ->UseRealTime();
BENCHMARK(xnnpack_convert_qu8)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
  ->UseRealTime();
BENCHMARK(xnnpack_convert_qu8_f32)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, float>)
  ->UseRealTime();

#ifdef BENCHMARK_TENSORFLOW_LITE
  BENCHMARK(tflite_convert_f16_f32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK(tflite_convert_f32_qs8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, int8_t>)
    ->UseRealTime();
  BENCHMARK(tflite_convert_f32_qu8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK(tflite_convert_qs8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK(tflite_convert_qs8_f32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK(tflite_convert_qu8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK(tflite_convert_qu8_f32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, float>)
    ->UseRealTime();
#endif  // BENCHMARK_TENSORFLOW_LITE

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
