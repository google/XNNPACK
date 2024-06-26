// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"

#include <benchmark/benchmark.h>
#ifdef BENCHMARK_TENSORFLOW_LITE
#include "flatbuffers/include/flatbuffers/flatbuffers.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#endif  // BENCHMARK_TENSORFLOW_LITE
#include "bench/utils.h"

static void xnnpack_average_pooling_qu8(benchmark::State& state, const char* net) {
  const size_t batch_size = state.range(0);
  const size_t input_height = state.range(1);
  const size_t input_width = state.range(2);
  const size_t pooling_size = state.range(3);
  const size_t padding_size = state.range(4);
  const size_t stride = state.range(5);
  const size_t channels = state.range(6);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), std::ref(rng));

  const size_t output_height = (2 * padding_size + input_height - pooling_size) / stride + 1;
  const size_t output_width = (2 * padding_size + input_width - pooling_size) / stride + 1;

  std::vector<uint8_t> input(batch_size * input_height * input_width * channels + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::generate(input.begin(), input.end(), std::ref(u8rng));
  std::vector<uint8_t> output(batch_size * output_height * output_width * channels);
  std::fill(output.begin(), output.end(), 0xA5);

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t pooling_op = nullptr;
  status = xnn_create_average_pooling2d_nhwc_qu8(
    padding_size, padding_size, padding_size, padding_size,
    pooling_size, pooling_size,
    stride, stride,
    127 /* input zero point */, 0.75f /* input scale */,
    127 /* output zero point */, 1.25f /* output scale */,
    0, 255,
    0 /* flags */, &pooling_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to create Average Pooling operator");
    return;
  }

  size_t workspace_size = 0;
  size_t workspace_alignment = 0;
  status = xnn_reshape_average_pooling2d_nhwc_qu8(
    pooling_op,
    batch_size, input_height, input_width,
    channels, /*input_pixel_stride=*/channels, /*output_pixel_stride=*/channels,
    &workspace_size, &workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/nullptr);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape Average Pooling operator");
    return;
  }

  std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);

  status = xnn_setup_average_pooling2d_nhwc_qu8(
    pooling_op,
    workspace.data(),
    input.data(), output.data());
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup Average Pooling operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(pooling_op, /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run Average Pooling operator");
      return;
    }
  }

  status = xnn_delete_operator(pooling_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete Average Pooling operator");
    return;
  }
  pooling_op = nullptr;

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["bytes"] = benchmark::Counter(
    uint64_t(state.iterations()) *
      batch_size * (input_height * input_width + output_height * output_width) * channels * sizeof(uint8_t),
    benchmark::Counter::kIsRate);
}

static void xnnpack_average_pooling_f32(benchmark::State& state, const char* net) {
  const size_t batch_size = state.range(0);
  const size_t input_height = state.range(1);
  const size_t input_width = state.range(2);
  const size_t pooling_size = state.range(3);
  const size_t padding_size = state.range(4);
  const size_t stride = state.range(5);
  const size_t channels = state.range(6);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  const size_t output_height = (2 * padding_size + input_height - pooling_size) / stride + 1;
  const size_t output_width = (2 * padding_size + input_width - pooling_size) / stride + 1;

  std::vector<float> input(batch_size * input_height * input_width * channels + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::vector<float> output(batch_size * output_height * output_width * channels);
  std::fill(output.begin(), output.end(), std::nanf(""));

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t pooling_op = nullptr;
  status = xnn_create_average_pooling2d_nhwc_f32(
    padding_size, padding_size, padding_size, padding_size,
    pooling_size, pooling_size,
    stride, stride,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */, &pooling_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to create Average Pooling operator");
    return;
  }

  size_t workspace_size = 0;
  size_t workspace_alignment = 0;
  status = xnn_reshape_average_pooling2d_nhwc_f32(
    pooling_op,
    batch_size, input_height, input_width,
    channels, /*input_pixel_stride=*/channels, /*output_pixel_stride=*/channels,
    &workspace_size, &workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/nullptr);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape Average Pooling operator");
    return;
  }

  std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);

  status = xnn_setup_average_pooling2d_nhwc_f32(
    pooling_op,
    workspace.data(),
    input.data(), output.data());
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup Average Pooling operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(pooling_op, /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run Average Pooling operator");
      return;
    }
  }

  status = xnn_delete_operator(pooling_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete Average Pooling operator");
    return;
  }
  pooling_op = nullptr;

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["bytes"] = benchmark::Counter(
    uint64_t(state.iterations()) *
      batch_size * (input_height * input_width + output_height * output_width) * channels * sizeof(float),
    benchmark::Counter::kIsRate);
}

#ifdef BENCHMARK_TENSORFLOW_LITE
void tflite_average_pooling_f32(benchmark::State& state, const char* net) {
  const size_t batch_size = state.range(0);
  const size_t input_height = state.range(1);
  const size_t input_width = state.range(2);
  const size_t pooling_size = state.range(3);
  const size_t padding_size = state.range(4);
  const size_t stride = state.range(5);
  const size_t channels = state.range(6);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  tflite::Padding padding = tflite::Padding_VALID;
  if (2 * padding_size == (pooling_size - 1)) {
    padding = tflite::Padding_SAME;
  } else if (padding_size == 0) {
    padding = tflite::Padding_VALID;
  } else {
    state.SkipWithError("unsupported padding");
    return;
  }

  const size_t output_height = (2 * padding_size + input_height - pooling_size) / stride + 1;
  const size_t output_width = (2 * padding_size + input_width - pooling_size) / stride + 1;

  std::vector<float> input(batch_size * input_height * input_width * channels + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::vector<float> output(batch_size * output_height * output_width * channels);
  std::fill(output.begin(), output.end(), std::nanf(""));

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::OperatorCode> operator_code =
      CreateOperatorCode(builder, tflite::BuiltinOperator_AVERAGE_POOL_2D);

  flatbuffers::Offset<tflite::Pool2DOptions> pool2d_options = CreatePool2DOptions(
      builder, padding,
      stride /* stride_w */, stride /* stride_h */,
      pooling_size /* filter_width */, pooling_size /* filter_height */,
      tflite::ActivationFunctionType_NONE);

  flatbuffers::Offset<tflite::Buffer> buffers[1] = {
    tflite::CreateBuffer(builder, builder.CreateVector({})),
  };

  const int32_t input_shape[4] = {
    static_cast<int32_t>(batch_size),
    static_cast<int32_t>(input_height),
    static_cast<int32_t>(input_width),
    static_cast<int32_t>(channels)
  };
  const int32_t output_shape[4] = {
    static_cast<int32_t>(batch_size),
    static_cast<int32_t>(output_height),
    static_cast<int32_t>(output_width),
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
  flatbuffers::Offset<tflite::Operator> op = CreateOperator(
      builder,
      0 /* opcode_index */,
      builder.CreateVector<int32_t>(op_inputs, 1),
      builder.CreateVector<int32_t>(op_outputs, 1),
      tflite::BuiltinOptions_Pool2DOptions,
      pool2d_options.Union());

  const int32_t graph_inputs[1] = { 0 };
  const int32_t graph_outputs[1] = { 1 };
  flatbuffers::Offset<tflite::SubGraph> subgraph = CreateSubGraph(
      builder,
      builder.CreateVector(tensors, 2),
      builder.CreateVector<int32_t>(graph_inputs, 1),
      builder.CreateVector<int32_t>(graph_outputs, 1),
      builder.CreateVector(&op, 1));

  flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(builder,
      TFLITE_SCHEMA_VERSION,
      builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1),
      builder.CreateString("AVERAGE_POOL_2D model"),
      builder.CreateVector(buffers, 1));

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
    interpreter->typed_tensor<float>(0) + batch_size * input_height * input_width * channels,
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

  state.counters["bytes"] = benchmark::Counter(
    uint64_t(state.iterations()) *
      batch_size * (input_height * input_width + output_height * output_width) * channels * sizeof(float),
    benchmark::Counter::kIsRate);
}
#endif  // BENCHMARK_TENSORFLOW_LITE

// Final global average pooling in ImageNet classification models.
static void ImageNet(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

  /*       N   H   W   K  P  S   C */
  b->Args({1, 13, 13, 13, 0, 1, 1000});
  b->Args({1,  7,  7,  7, 0, 1, 1000});
}

// ShuffleNet v1 with 1 group.
static void ShuffleNetV1G1(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

  /*       N   H   W  K  P  S   C */
  b->Args({1, 56, 56, 3, 1, 2,  24});
  b->Args({1, 28, 28, 3, 1, 2, 144});
  b->Args({1, 14, 14, 3, 1, 2, 288});
  b->Args({1,  7,  7, 3, 1, 2, 576});
}

// ShuffleNet v1 with 2 groups.
static void ShuffleNetV1G2(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

  /*       N   H   W  K  P  S   C */
  b->Args({1, 56, 56, 3, 1, 2,  24});
  b->Args({1, 28, 28, 3, 1, 2, 200});
  b->Args({1, 14, 14, 3, 1, 2, 400});
  b->Args({1,  7,  7, 3, 1, 2, 800});
}

// ShuffleNet v1 with 3 groups.
static void ShuffleNetV1G3(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

  /*       N   H   W  K  P  S   C */
  b->Args({1, 56, 56, 3, 1, 2,  24});
  b->Args({1, 28, 28, 3, 1, 2, 240});
  b->Args({1, 14, 14, 3, 1, 2, 480});
  b->Args({1,  7,  7, 3, 1, 2, 960});
}

// ShuffleNet v1 with 4 groups.
static void ShuffleNetV1G4(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

  /*       N   H   W  K  P  S    C */
  b->Args({1, 56, 56, 3, 1, 2,   24});
  b->Args({1, 28, 28, 3, 1, 2,  272});
  b->Args({1, 14, 14, 3, 1, 2,  576});
  b->Args({1,  7,  7, 3, 1, 2, 1088});
}

// ShuffleNet v1 with 8 groups.
static void ShuffleNetV1G8(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

  /*       N   H   W  K  P  S    C */
  b->Args({1, 56, 56, 3, 1, 2,   24});
  b->Args({1, 28, 28, 3, 1, 2,  384});
  b->Args({1, 14, 14, 3, 1, 2,  768});
  b->Args({1,  7,  7, 3, 1, 2, 1536});
}

BENCHMARK_CAPTURE(xnnpack_average_pooling_f32, imagenet, "ImageNet")->Apply(ImageNet)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_average_pooling_f32, shufflenet_v1_g1, "ShuffleNet v1 (1 group)")->Apply(ShuffleNetV1G1)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_average_pooling_f32, shufflenet_v1_g2, "ShuffleNet v1 (2 groups)")->Apply(ShuffleNetV1G2)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_average_pooling_f32, shufflenet_v1_g3, "ShuffleNet v1 (3 groups)")->Apply(ShuffleNetV1G3)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_average_pooling_f32, shufflenet_v1_g4, "ShuffleNet v1 (4 groups)")->Apply(ShuffleNetV1G4)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_average_pooling_f32, shufflenet_v1_g8, "ShuffleNet v1 (8 groups)")->Apply(ShuffleNetV1G8)->UseRealTime();

#ifdef BENCHMARK_TENSORFLOW_LITE
BENCHMARK_CAPTURE(tflite_average_pooling_f32, imagenet, "ImageNet")->Apply(ImageNet)->UseRealTime();
BENCHMARK_CAPTURE(tflite_average_pooling_f32, shufflenet_v1_g1, "ShuffleNet v1 (1 group)")->Apply(ShuffleNetV1G1)->UseRealTime();
BENCHMARK_CAPTURE(tflite_average_pooling_f32, shufflenet_v1_g2, "ShuffleNet v1 (2 groups)")->Apply(ShuffleNetV1G2)->UseRealTime();
BENCHMARK_CAPTURE(tflite_average_pooling_f32, shufflenet_v1_g3, "ShuffleNet v1 (3 groups)")->Apply(ShuffleNetV1G3)->UseRealTime();
BENCHMARK_CAPTURE(tflite_average_pooling_f32, shufflenet_v1_g4, "ShuffleNet v1 (4 groups)")->Apply(ShuffleNetV1G4)->UseRealTime();
BENCHMARK_CAPTURE(tflite_average_pooling_f32, shufflenet_v1_g8, "ShuffleNet v1 (8 groups)")->Apply(ShuffleNetV1G8)->UseRealTime();
#endif  // BENCHMARK_TENSORFLOW_LITE

BENCHMARK_CAPTURE(xnnpack_average_pooling_qu8, imagenet, "ImageNet")->Apply(ImageNet)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_average_pooling_qu8, shufflenet_v1_g1, "ShuffleNet v1 (1 group)")->Apply(ShuffleNetV1G1)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_average_pooling_qu8, shufflenet_v1_g2, "ShuffleNet v1 (2 groups)")->Apply(ShuffleNetV1G2)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_average_pooling_qu8, shufflenet_v1_g3, "ShuffleNet v1 (3 groups)")->Apply(ShuffleNetV1G3)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_average_pooling_qu8, shufflenet_v1_g4, "ShuffleNet v1 (4 groups)")->Apply(ShuffleNetV1G4)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_average_pooling_qu8, shufflenet_v1_g8, "ShuffleNet v1 (8 groups)")->Apply(ShuffleNetV1G8)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
