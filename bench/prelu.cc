// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include <xnnpack.h>

#include "bench/utils.h"
#include <benchmark/benchmark.h>
#ifdef BENCHMARK_TENSORFLOW_LITE
#include "flatbuffers/include/flatbuffers/flatbuffers.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#endif  // BENCHMARK_TENSORFLOW_LITE


void xnnpack_prelu_f32(benchmark::State& state, const char* net) {
  const size_t batch_size = state.range(0);
  const size_t height = state.range(1);
  const size_t width = state.range(2);
  const size_t channels = state.range(3);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32irng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::ref(rng));
  auto f32wrng = std::bind(std::uniform_real_distribution<float>(0.25f, 0.75f), std::ref(rng));

  std::vector<float> input(batch_size * height * width * channels + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(input.begin(), input.end(), std::ref(f32irng));
  std::vector<float> slope(channels);
  std::generate(slope.begin(), slope.end(), std::ref(f32wrng));
  std::vector<float> output(batch_size * height * width * channels);

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t prelu_op = nullptr;
  status = xnn_create_prelu_nc_f32(
    channels, channels /* input stride */, channels /* output stride */,
    slope.data(),
    0 /* flags */, nullptr, nullptr, &prelu_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to create FP32 PReLU operator");
    return;
  }

  status = xnn_reshape_prelu_nc_f32(
    prelu_op,
    batch_size * height * width,
    /*threadpool=*/nullptr);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape FP32 PReLU operator");
    return;
  }

  status = xnn_setup_prelu_nc_f32(
    prelu_op,
    input.data(), output.data());
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup FP32 PReLU operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(prelu_op, /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run FP32 PReLU operator");
      return;
    }
  }

  status = xnn_delete_operator(prelu_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete FP32 PReLU operator");
    return;
  }
  prelu_op = nullptr;

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = batch_size * height * width * channels;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = (2 * elements_per_iteration + channels) * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#ifdef BENCHMARK_TENSORFLOW_LITE
void tflite_prelu_f32(benchmark::State& state, const char* net) {
  const size_t batch_size = state.range(0);
  const size_t height = state.range(1);
  const size_t width = state.range(2);
  const size_t channels = state.range(3);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32irng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::ref(rng));
  auto f32wrng = std::bind(std::uniform_real_distribution<float>(0.25f, 0.75f), std::ref(rng));

  std::vector<float> slope(channels);
  std::generate(slope.begin(), slope.end(), std::ref(f32wrng));

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::OperatorCode> operator_code =
      CreateOperatorCode(builder, tflite::BuiltinOperator_PRELU);

  flatbuffers::Offset<tflite::Buffer> buffers[2] = {
    tflite::CreateBuffer(builder, builder.CreateVector({})),
    tflite::CreateBuffer(builder, builder.CreateVector(
      reinterpret_cast<const uint8_t*>(slope.data()),
      sizeof(float) * slope.size())),
  };

  const int32_t input_shape[4] = {
    static_cast<int32_t>(batch_size),
    static_cast<int32_t>(height),
    static_cast<int32_t>(width),
    static_cast<int32_t>(channels)
  };
  const int32_t output_shape[4] = {
    static_cast<int32_t>(batch_size),
    static_cast<int32_t>(height),
    static_cast<int32_t>(width),
    static_cast<int32_t>(channels)
  };
  const int32_t slope_shape[1] = {
    static_cast<int32_t>(channels)
  };

  flatbuffers::Offset<tflite::Tensor> tensors[3] = {
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(input_shape, 4),
                         tflite::TensorType_FLOAT32),
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(slope_shape, 1),
                         tflite::TensorType_FLOAT32,
                         1 /* buffer id */),
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(output_shape, 4),
                         tflite::TensorType_FLOAT32),
  };

  const int32_t op_inputs[2] = { 0, 1 };
  const int32_t op_outputs[1] = { 2 };
  flatbuffers::Offset<tflite::Operator> op = tflite::CreateOperator(
      builder,
      0 /* opcode_index */,
      builder.CreateVector<int32_t>(op_inputs, 2),
      builder.CreateVector<int32_t>(op_outputs, 1));

  const int32_t graph_inputs[1] = { 0 };
  const int32_t graph_outputs[1] = { 2 };
  flatbuffers::Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      builder,
      builder.CreateVector(tensors, 3),
      builder.CreateVector<int32_t>(graph_inputs, 1),
      builder.CreateVector<int32_t>(graph_outputs, 1),
      builder.CreateVector(&op, 1));

  flatbuffers::Offset<flatbuffers::String> description = builder.CreateString("PReLU model");

  flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(builder,
      TFLITE_SCHEMA_VERSION,
      builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1),
      description,
      builder.CreateVector(buffers, 2));

  builder.Finish(model_buffer);

  const tflite::Model* model = tflite::GetModel(builder.GetBufferPointer());
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
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
    interpreter->typed_tensor<float>(0) + batch_size * height * width * channels,
    std::ref(f32irng));

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

  const size_t elements_per_iteration = batch_size * height * width * channels;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = (2 * elements_per_iteration + channels) * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);

  interpreter.reset();
}
#endif  // BENCHMARK_TENSORFLOW_LITE

// Characteristic arguments for ImageNet classification models
static void ImageNet(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"N", "H", "W", "C"});

  int32_t c = 16;
  for (int32_t hw = 224 / 2; hw >= 7; hw /= 2) {
    b->Args({1, hw, hw, c});
    b->Args({1, hw, hw, c * 2});
    c *= 2;
  }
}

BENCHMARK_CAPTURE(xnnpack_prelu_f32, imagenet, "ImageNet 224x224")->Apply(ImageNet)->UseRealTime();

#ifdef BENCHMARK_TENSORFLOW_LITE
  BENCHMARK_CAPTURE(tflite_prelu_f32, imagenet, "ImageNet 224x224")->Apply(ImageNet)->UseRealTime();
#endif  // BENCHMARK_TENSORFLOW_LITE

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
