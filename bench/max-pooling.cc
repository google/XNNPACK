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
#include <random>
#include <vector>

#include <xnnpack.h>

#include <benchmark/benchmark.h>
#include "bench/utils.h"


void max_pooling_u8(benchmark::State& state, const char* net) {
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

  std::vector<uint8_t> input(batch_size * input_height * input_width * channels);
  std::generate(input.begin(), input.end(), std::ref(u8rng));
  std::vector<uint8_t> output(batch_size * output_height * output_width * channels);
  std::fill(output.begin(), output.end(), 0xA5);

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t pooling_op = nullptr;
  status = xnn_create_max_pooling2d_nhwc_u8(
    padding_size, padding_size, padding_size, padding_size,
    pooling_size, pooling_size,
    stride, stride,
    1 /* dilation height */, 1 /* dilation width */,
    0, 255,
    0 /* flags */, &pooling_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to create Max Pooling operator");
    return;
  }

  status = xnn_reshape_max_pooling2d_nhwc_u8(
    pooling_op,
    batch_size, input_height, input_width,
    channels, /*input_pixel_stride=*/channels, /*output_pixel_stride=*/channels,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/nullptr);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape Max Pooling operator");
    return;
  }

  status = xnn_setup_max_pooling2d_nhwc_u8(
    pooling_op,
    input.data(), output.data());
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup Max Pooling operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(pooling_op, /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run Max Pooling operator");
      return;
    }
  }

  status = xnn_delete_operator(pooling_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete Max Pooling operator");
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

void max_pooling_f32(benchmark::State& state, const char* net) {
  const size_t batch_size = state.range(0);
  const size_t input_height = state.range(1);
  const size_t input_width = state.range(2);
  const size_t pooling_size = state.range(3);
  const size_t padding_size = state.range(4);
  const size_t stride = state.range(5);
  const size_t channels = state.range(6);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), std::ref(rng));

  const size_t output_height = (2 * padding_size + input_height - pooling_size) / stride + 1;
  const size_t output_width = (2 * padding_size + input_width - pooling_size) / stride + 1;

  std::vector<float> input(batch_size * input_height * input_width * channels);
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::vector<float> output(batch_size * output_height * output_width * channels);
  std::fill(output.begin(), output.end(), nanf(""));

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t pooling_op = nullptr;
  status = xnn_create_max_pooling2d_nhwc_f32(
    padding_size, padding_size, padding_size, padding_size,
    pooling_size, pooling_size,
    stride, stride,
    1 /* dilation height */, 1 /* dilation width */,
    -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity(),
    0 /* flags */, &pooling_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to create Max Pooling operator");
    return;
  }

  status = xnn_reshape_max_pooling2d_nhwc_f32(
    pooling_op,
    batch_size, input_height, input_width,
    channels, /*input_pixel_stride=*/channels, /*output_pixel_stride=*/channels,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/nullptr);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape Max Pooling operator");
    return;
  }

  status = xnn_setup_max_pooling2d_nhwc_f32(
    pooling_op,
    input.data(), output.data());
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup Max Pooling operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(pooling_op, /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run Max Pooling operator");
      return;
    }
  }

  status = xnn_delete_operator(pooling_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete Max Pooling operator");
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

// ShuffleNet v1/v2.
static void ShuffleNet(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

  /*       N   H   W    K  P  S   C */
  b->Args({1, 112, 112, 3, 1, 2, 24});
}

// SqueezeNet 1.0
static void SqueezeNetV10(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

  /*********** MaxPool 1 ************/
  /*       N   H    W   K  P  S   C */
  b->Args({1, 111, 111, 3, 0, 2,  96});
  /*********** MaxPool 4 ************/
  /*       N   H    W   K  P  S   C */
  b->Args({1,  27,  27, 3, 0, 2, 256});
  /*********** MaxPool 8 ************/
  /*       N   H    W   K  P  S   C */
  b->Args({1,  13,  13, 3, 0, 2, 512});
}

// SqueezeNet 1.1
static void SqueezeNetV11(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

  /*********** MaxPool 1 ***********/
  /*       N   H    W   K  P  S   C */
  b->Args({1, 111, 111, 3, 0, 2,  64});
  /*********** MaxPool 3 ************/
  /*       N   H    W   K  P  S   C */
  b->Args({1,  55,  55, 3, 0, 2, 128});
  /*********** MaxPool 5 ************/
  /*       N   H    W   K  P  S   C */
  b->Args({1,  13,  13, 3, 0, 2, 256});
}

static void VGG(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

  /*       N   H    W   K  P  S   C */
  b->Args({1, 224, 224, 2, 1, 2,  64});
  b->Args({1, 112, 112, 2, 1, 2, 128});
  b->Args({1,  56,  56, 2, 1, 2, 256});
  b->Args({1,  28,  28, 2, 1, 2, 512});
  b->Args({1,  14,  14, 2, 1, 2, 512});
}

BENCHMARK_CAPTURE(max_pooling_f32, shufflenet, "ShuffleNet v1/v2")->Apply(ShuffleNet)->UseRealTime();
BENCHMARK_CAPTURE(max_pooling_f32, squeezenet_v10, "SqueezeNet v1.0")->Apply(SqueezeNetV10)->UseRealTime();
BENCHMARK_CAPTURE(max_pooling_f32, squeezenet_v11, "SqueezeNet v1.1")->Apply(SqueezeNetV11)->UseRealTime();
BENCHMARK_CAPTURE(max_pooling_f32, vgg, "VGG")->Apply(VGG);

BENCHMARK_CAPTURE(max_pooling_u8, shufflenet, "ShuffleNet v1/v2")->Apply(ShuffleNet)->UseRealTime();
BENCHMARK_CAPTURE(max_pooling_u8, squeezenet_v10, "SqueezeNet v1.0")->Apply(SqueezeNetV10)->UseRealTime();
BENCHMARK_CAPTURE(max_pooling_u8, squeezenet_v11, "SqueezeNet v1.1")->Apply(SqueezeNetV11)->UseRealTime();
BENCHMARK_CAPTURE(max_pooling_u8, vgg, "VGG")->Apply(VGG);

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
