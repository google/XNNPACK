#ifndef THIRD_PARTY_XNNPACK_BENCH_BENCHMARK_OPERATOR_H_
#define THIRD_PARTY_XNNPACK_BENCH_BENCHMARK_OPERATOR_H_

// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <fp16/fp16.h>
#include "bench/utils.h"
#include <benchmark/benchmark.h>
#ifdef BENCHMARK_TENSORFLOW_LITE
#include "flatbuffers/include/flatbuffers/flatbuffers.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#endif  // BENCHMARK_TENSORFLOW_LITE

struct float16 {
  uint16_t value;

  float16() = default;
  float16(float value) : value(fp16_ieee_from_fp32_value(value)) {}  // NOLINT
};

template <typename In, typename Out, typename Create, typename Reshape,
          typename Setup>
static void benchmark_unary_operator(Create create, Reshape reshape,
                                     Setup setup, benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_real_distribution<float> f32dist(0.0f, 10.0f);

  std::vector<In> input(batch_size + XNN_EXTRA_BYTES / sizeof(In));
  std::vector<Out> output(batch_size);
  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::fill(output.begin(), output.end(), 0);

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t op = nullptr;
  status = create(0 /* flags */, &op);
  if (status != xnn_status_success || op == nullptr) {
    state.SkipWithError("failed to create Abs operator");
    return;
  }

  status = reshape(op, batch_size,
                   /*channels=*/1, /*input_stride=*/1, /*output_stride=*/1,
                   /*threadpool=*/nullptr);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape Abs operator");
    return;
  }

  status = setup(op, input.data(), output.data());
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup Abs operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(op, /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run Abs operator");
      return;
    }
  }

  status = xnn_delete_operator(op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete Abs operator");
    return;
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["elements"] = benchmark::Counter(
      uint64_t(state.iterations()) * batch_size, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * batch_size * sizeof(uint16_t);
  state.counters["bytes"] =
      benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration,
                         benchmark::Counter::kIsRate);
}

#ifdef BENCHMARK_TENSORFLOW_LITE

template <typename T>
struct TypeToTfliteType {
  using type = T;
};
template <>
struct TypeToTfliteType<float16> {
  using type = TfLiteFloat16;
};

template <typename In, typename Out, class BuildInQuantization,
          class BuildOutQuantization>
static void benchmark_tflite_unary_operator(
    benchmark::State& state, BuildInQuantization in_quantization,
    BuildOutQuantization out_quantization, tflite::BuiltinOperator op_code) {
  const size_t batch_size = state.range(0);

  flatbuffers::FlatBufferBuilder builder;
  const flatbuffers::Offset<tflite::OperatorCode> operator_code =
      CreateOperatorCode(builder, op_code);

  const std::array<flatbuffers::Offset<tflite::Buffer>, 1> buffers{{
      tflite::CreateBuffer(builder, builder.CreateVector({})),
  }};

  const std::array<int32_t, 1> shape{{static_cast<int32_t>(batch_size)}};

  const std::array<flatbuffers::Offset<tflite::Tensor>, 2> tensors{{
      tflite::CreateTensor(
          builder, builder.CreateVector<int32_t>(shape.data(), shape.size()),
          tflite::TensorTypeFor<typename TypeToTfliteType<In>::type>::value,
          /*buffer=*/0, /*name=*/0, in_quantization(builder)),
      tflite::CreateTensor(
          builder, builder.CreateVector<int32_t>(shape.data(), shape.size()),
          tflite::TensorTypeFor<typename TypeToTfliteType<Out>::type>::value,
          /*buffer=*/0, /*name=*/0, out_quantization(builder)),
  }};

  const std::array<int32_t, 1> op_inputs{{0}};
  const std::array<int32_t, 1> op_outputs{{1}};
  flatbuffers::Offset<tflite::Operator> op = tflite::CreateOperator(
      builder, 0 /* opcode_index */,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()));

  const std::array<int32_t, 1> graph_inputs{{0}};
  const std::array<int32_t, 1> graph_outputs{{1}};
  const flatbuffers::Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(graph_inputs.data(), graph_inputs.size()),
      builder.CreateVector<int32_t>(graph_outputs.data(), graph_outputs.size()),
      builder.CreateVector(&op, 1));

  const flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1), builder.CreateString("Abs model"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  const tflite::Model* model = tflite::GetModel(builder.GetBufferPointer());
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  tflite::InterpreterBuilder interpreterBuilder(model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (interpreterBuilder(&interpreter) != kTfLiteOk || interpreter == nullptr) {
    state.SkipWithError("failed to create TFLite interpreter");
    return;
  }
  interpreter->SetNumThreads(1);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    state.SkipWithError("failed to allocate tensors");
    return;
  }

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32dist = std::uniform_real_distribution<float>(0.0f, 10.0f);
  In* input_ptr = reinterpret_cast<In*>(
      interpreter->typed_tensor<typename TypeToTfliteType<In>::type>(0));
  std::generate(input_ptr, input_ptr + batch_size,
                [&]() { return f32dist(rng); });

  for (auto _ : state) {
    if (interpreter->Invoke() != kTfLiteOk) {
      state.SkipWithError("failed to invoke TFLite interpreter");
      return;
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["elements"] = benchmark::Counter(
      uint64_t(state.iterations()) * batch_size, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * batch_size * sizeof(float);
  state.counters["bytes"] =
      benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration,
                         benchmark::Counter::kIsRate);

  interpreter.reset();
}

static flatbuffers::Offset<tflite::QuantizationParameters> no_quantization(
    flatbuffers::FlatBufferBuilder& builder) {
  return 0;
}

template <typename In, typename Out>
static void benchmark_tflite_unary_operator(benchmark::State& state,
                                            tflite::BuiltinOperator op_code) {
  return benchmark_tflite_unary_operator<In, Out>(state, no_quantization,
                                                  no_quantization, op_code);
}
#endif  // BENCHMARK_TENSORFLOW_LITE

#endif  // THIRD_PARTY_XNNPACK_BENCH_BENCHMARK_OPERATOR_H_
