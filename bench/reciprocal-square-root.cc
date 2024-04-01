// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include "bench/utils.h"
#include <benchmark/benchmark.h>

#ifdef BENCHMARK_TENSORFLOW_LITE
#include "flatbuffers/include/flatbuffers/buffer.h"
#include "flatbuffers/include/flatbuffers/flatbuffer_builder.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#endif  // BENCHMARK_TENSORFLOW_LITE

static void xnnpack_reciprocal_square_root_f32(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.1f, 5.0f),
                          std::ref(rng));

  std::vector<float> input(batch_size + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> output(batch_size);
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::fill(output.begin(), output.end(), std::nanf(""));

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t rsqrt_op = nullptr;
  status = xnn_create_reciprocal_square_root_nc_f32(0 /* flags */, &rsqrt_op);
  if (status != xnn_status_success || rsqrt_op == nullptr) {
    state.SkipWithError("failed to create Reciprocal Square Root operator");
    return;
  }

  status = xnn_reshape_reciprocal_square_root_nc_f32(rsqrt_op, batch_size,
                                                     /*channels=*/1,
                                                     /*input_stride=*/1,
                                                     /*output_stride=*/1,
                                                     /*threadpool=*/nullptr);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape Reciprocal Square Root operator");
    return;
  }

  status = xnn_setup_reciprocal_square_root_nc_f32(rsqrt_op, input.data(),
                                                   output.data());
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup Reciprocal Square Root operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(rsqrt_op, /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run Reciprocal Square Root operator");
      return;
    }
  }

  status = xnn_delete_operator(rsqrt_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete Reciprocal Square Root operator");
    return;
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["elements"] =
      benchmark::Counter(static_cast<uint64_t>(state.iterations()) * batch_size,
                         benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * batch_size * sizeof(float);
  state.counters["bytes"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * bytes_per_iteration,
      benchmark::Counter::kIsRate);
}

#ifdef BENCHMARK_TENSORFLOW_LITE
static void tflite_reciprocal_square_root_f32(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.1f, 5.0f),
                          std::ref(rng));

  flatbuffers::FlatBufferBuilder builder;
  const flatbuffers::Offset<tflite::OperatorCode> operator_code =
      CreateOperatorCode(builder, tflite::BuiltinOperator_SQRT);

  const std::array<flatbuffers::Offset<tflite::Buffer>, 1> buffers{{
      tflite::CreateBuffer(builder, builder.CreateVector({})),
  }};

  const std::array<int32_t, 1> shape{{
      static_cast<int32_t>(batch_size),
  }};

  const std::array<flatbuffers::Offset<tflite::Tensor>, 2> tensors{{
      tflite::CreateTensor(
          builder, builder.CreateVector<int32_t>(shape.data(), shape.size()),
          tflite::TensorType_FLOAT32),
      tflite::CreateTensor(
          builder, builder.CreateVector<int32_t>(shape.data(), shape.size()),
          tflite::TensorType_FLOAT32),
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
      builder.CreateVector(&subgraph, 1),
      builder.CreateString("Reciprocal Square Root model"),
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

  std::generate(interpreter->typed_tensor<float>(0),
                interpreter->typed_tensor<float>(0) + batch_size,
                std::ref(f32rng));

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

  state.counters["elements"] =
      benchmark::Counter(static_cast<uint64_t>(state.iterations()) * batch_size,
                         benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * batch_size * sizeof(float);
  state.counters["bytes"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * bytes_per_iteration,
      benchmark::Counter::kIsRate);

  interpreter.reset();
}
#endif  // BENCHMARK_TENSORFLOW_LITE

BENCHMARK(xnnpack_reciprocal_square_root_f32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

#ifdef BENCHMARK_TENSORFLOW_LITE
BENCHMARK(tflite_reciprocal_square_root_f32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // BENCHMARK_TENSORFLOW_LITE

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
