// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <xnnpack.h>

#include <benchmark/benchmark.h>
#include "bench/utils.h"
#ifdef BENCHMARK_TENSORFLOW_LITE
#include "flatbuffers/include/flatbuffers/flatbuffers.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#endif  // BENCHMARK_TENSORFLOW_LITE


static void xnnpack_sigmoid_q8(benchmark::State& state) {
  const size_t batch_size = state.range(0);
  const size_t channels = state.range(1);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

  std::vector<uint8_t> input(batch_size * channels);
  std::vector<uint8_t> output(batch_size * channels);
  std::generate(input.begin(), input.end(), std::ref(u8rng));
  std::fill(output.begin(), output.end(), 0xA5);

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t sigmoid_op = nullptr;
  status = xnn_create_sigmoid_nc_q8(
    channels, channels /* input stride */, channels /* output stride */,
    127 /* input zero point */, 1.0f /* input scale */,
    0 /* output zero point */, 1.0f / 256.0f /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */, &sigmoid_op);
  if (status != xnn_status_success || sigmoid_op == nullptr) {
    state.SkipWithError("failed to create Sigmoid operator");
    return;
  }

  status = xnn_setup_sigmoid_nc_q8(
    sigmoid_op,
    batch_size,
    input.data(), output.data(),
    nullptr /* thread pool */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup Sigmoid operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(sigmoid_op, nullptr /* thread pool */);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run Sigmoid operator");
      return;
    }
  }

  status = xnn_delete_operator(sigmoid_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete Sigmoid operator");
    return;
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();

  const size_t elements_per_iteration = batch_size * channels;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * elements_per_iteration * sizeof(uint8_t);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

static void xnnpack_sigmoid_f32(benchmark::State& state) {
  const size_t batch_size = state.range(0);
  const size_t channels = state.range(1);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f), rng);

  std::vector<float> input(batch_size * channels);
  std::vector<float> output(batch_size * channels);
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::fill(output.begin(), output.end(), std::nanf(""));

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t sigmoid_op = nullptr;
  status = xnn_create_sigmoid_nc_f32(
    channels, channels /* input stride */, channels /* output stride */,
    0 /* flags */, &sigmoid_op);
  if (status != xnn_status_success || sigmoid_op == nullptr) {
    state.SkipWithError("failed to create Sigmoid operator");
    return;
  }

  status = xnn_setup_sigmoid_nc_f32(
    sigmoid_op,
    batch_size,
    input.data(), output.data(),
    nullptr /* thread pool */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup Sigmoid operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(sigmoid_op, nullptr /* thread pool */);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run Sigmoid operator");
      return;
    }
  }

  status = xnn_delete_operator(sigmoid_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete Sigmoid operator");
    return;
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();

  const size_t elements_per_iteration = batch_size * channels;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * elements_per_iteration * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#ifdef BENCHMARK_TENSORFLOW_LITE
static void tflite_sigmoid_f32(benchmark::State& state) {
  const size_t batch_size = state.range(0);
  const size_t channels = state.range(1);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f), rng);

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::OperatorCode> operator_code =
      CreateOperatorCode(builder, tflite::BuiltinOperator_LOGISTIC);

  flatbuffers::Offset<tflite::Buffer> buffers[1] = {
    tflite::CreateBuffer(builder, builder.CreateVector({})),
  };

  const int32_t input_shape[4] = {
    static_cast<int32_t>(batch_size),
    static_cast<int32_t>(1 /* height */),
    static_cast<int32_t>(1 /* width */),
    static_cast<int32_t>(channels)
  };
  const int32_t output_shape[4] = {
    static_cast<int32_t>(batch_size),
    static_cast<int32_t>(1 /* height */),
    static_cast<int32_t>(1 /* width */),
    static_cast<int32_t>(channels)
  };

  flatbuffers::Offset<tflite::Tensor> tensors[2] = {
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(input_shape, 4),
                         tflite::TensorType_FLOAT32),
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(output_shape, 4),
                         tflite::TensorType_FLOAT32),
  };

  const int32_t op_inputs[1] = { 0 };
  const int32_t op_outputs[1] = { 1 };
  flatbuffers::Offset<tflite::Operator> op = tflite::CreateOperator(
      builder,
      0 /* opcode_index */,
      builder.CreateVector<int32_t>(op_inputs, 1),
      builder.CreateVector<int32_t>(op_outputs, 1));

  const int32_t graph_inputs[1] = { 0 };
  const int32_t graph_outputs[1] = { 1 };
  flatbuffers::Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      builder,
      builder.CreateVector(tensors, 2),
      builder.CreateVector<int32_t>(graph_inputs, 1),
      builder.CreateVector<int32_t>(graph_outputs, 1),
      builder.CreateVector(&op, 1));

  flatbuffers::Offset<flatbuffers::String> description = builder.CreateString("Sigmoid model");

  flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(builder,
      TFLITE_SCHEMA_VERSION,
      builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1),
      description,
      builder.CreateVector(buffers, 1));

  builder.Finish(model_buffer);

  const tflite::Model* model = tflite::GetModel(builder.GetBufferPointer());
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder interpreterBuilder(model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (interpreterBuilder(&interpreter) != kTfLiteOk) {
    state.SkipWithError("failed to create TFLite interpreter");
    return;
  }
  if (interpreter == nullptr) {
    state.SkipWithError("TFLite interpreter is null");
    return;
  }
  interpreter->SetNumThreads(1);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    state.SkipWithError("failed to allocate tensors");
    return;
  }

  std::generate(
    interpreter->typed_tensor<float>(0),
    interpreter->typed_tensor<float>(0) + batch_size * channels,
    std::ref(f32rng));

  for (auto _ : state) {
    if (interpreter->Invoke() != kTfLiteOk) {
      state.SkipWithError("failed to invoke TFLite interpreter");
      return;
    }
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();

  const size_t elements_per_iteration = batch_size * channels;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * elements_per_iteration * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);

  interpreter.reset();
}
#endif  // BENCHMARK_TENSORFLOW_LITE

static void CharacteristicArguments(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"N", "C"});

  int32_t c = 16;
  for (int32_t n = 224; n >= 7; n /= 2) {
    b->Args({n * n, c});
    c *= 2;
  }
}

BENCHMARK(xnnpack_sigmoid_q8)->Apply(CharacteristicArguments)->UseRealTime();
BENCHMARK(xnnpack_sigmoid_f32)->Apply(CharacteristicArguments)->UseRealTime();

#ifdef BENCHMARK_TENSORFLOW_LITE
  BENCHMARK(tflite_sigmoid_f32)->Apply(CharacteristicArguments)->UseRealTime();
#endif  // BENCHMARK_TENSORFLOW_LITE

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
