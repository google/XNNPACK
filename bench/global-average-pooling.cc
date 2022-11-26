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
#include <fp16.h>
#include "bench/utils.h"

#ifndef XNN_NO_QU8_OPERATORS
static void global_average_pooling_qu8(benchmark::State& state) {
  const size_t batch_size = state.range(0);
  const size_t input_height = state.range(1);
  const size_t input_width = state.range(2);
  const size_t channels = state.range(3);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), std::ref(rng));

  std::vector<uint8_t> input(batch_size * input_height * input_width * channels);
  std::generate(input.begin(), input.end(), std::ref(u8rng));
  std::vector<uint8_t> output(batch_size * channels);

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
  }

  xnn_operator_t global_pooling_op = nullptr;
  status = xnn_create_global_average_pooling_nwc_qu8(
    channels, channels /* input stride */, channels /* output stride */,
    127 /* input zero point */, 0.75f /* input scale */,
    127 /* output zero point */, 1.25f /* output scale */,
    0, 255,
    0 /* flags */, &global_pooling_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to create Global Average Pooling operator");
  }

  status = xnn_setup_global_average_pooling_nwc_qu8(
    global_pooling_op,
    batch_size, input_height * input_width,
    input.data(), output.data(),
    nullptr /* thread pool */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup Global Average Pooling operator");
  }

  for (auto _ : state) {
    xnn_run_operator(global_pooling_op, nullptr /* thread pool */);
  }

  status = xnn_delete_operator(global_pooling_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete Global Average Pooling operator");
  }
  global_pooling_op = nullptr;

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["bytes"] = benchmark::Counter(
    uint64_t(state.iterations()) *
      batch_size * (input_height * input_width + 1) * channels * sizeof(uint8_t),
    benchmark::Counter::kIsRate);
}
#endif  // XNN_NO_QU8_OPERATORS

#ifndef XNN_NO_QS8_OPERATORS
static void global_average_pooling_qs8(benchmark::State& state) {
  const size_t batch_size = state.range(0);
  const size_t input_height = state.range(1);
  const size_t input_width = state.range(2);
  const size_t channels = state.range(3);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i8rng = std::bind(
    std::uniform_int_distribution<uint32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()), std::ref(rng));

  std::vector<int8_t> input(batch_size * input_height * input_width * channels);
  std::generate(input.begin(), input.end(), std::ref(i8rng));
  std::vector<int8_t> output(batch_size * channels);

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
  }

  xnn_operator_t global_pooling_op = nullptr;
  status = xnn_create_global_average_pooling_nwc_qs8(
    channels, channels /* input stride */, channels /* output stride */,
    -1 /* input zero point */, 0.75f /* input scale */,
    -1 /* output zero point */, 1.25f /* output scale */,
    -128, 127,
    0 /* flags */, &global_pooling_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to create Global Average Pooling operator");
  }

  status = xnn_setup_global_average_pooling_nwc_qs8(
    global_pooling_op,
    batch_size, input_height * input_width,
    input.data(), output.data(),
    nullptr /* thread pool */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup Global Average Pooling operator");
  }

  for (auto _ : state) {
    xnn_run_operator(global_pooling_op, nullptr /* thread pool */);
  }

  status = xnn_delete_operator(global_pooling_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete Global Average Pooling operator");
  }
  global_pooling_op = nullptr;

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["bytes"] = benchmark::Counter(
    uint64_t(state.iterations()) *
      batch_size * (input_height * input_width + 1) * channels * sizeof(int8_t),
    benchmark::Counter::kIsRate);
}
#endif  // XNN_NO_QS8_OPERATORS

#ifndef XNN_NO_F16_OPERATORS
static void global_average_pooling_f16(benchmark::State& state) {
  const size_t batch_size = state.range(0);
  const size_t input_height = state.range(1);
  const size_t input_width = state.range(2);
  const size_t channels = state.range(3);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.1f, 1.0f), std::ref(rng));
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

  std::vector<uint16_t> input(batch_size * input_height * input_width * channels);
  std::generate(input.begin(), input.end(), std::ref(f16rng));
  std::vector<uint16_t> output(batch_size * channels);

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
  }

  xnn_operator_t global_pooling_op = nullptr;
  status = xnn_create_global_average_pooling_nwc_f16(
    channels, channels /* input stride */, channels /* output stride */,
    -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity(),
    0 /* flags */, &global_pooling_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to create Global Average Pooling operator");
  }

  status = xnn_setup_global_average_pooling_nwc_f16(
    global_pooling_op,
    batch_size, input_height * input_width,
    input.data(), output.data(),
    nullptr /* thread pool */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup Global Average Pooling operator");
  }

  for (auto _ : state) {
    xnn_run_operator(global_pooling_op, nullptr /* thread pool */);
  }

  status = xnn_delete_operator(global_pooling_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete Global Average Pooling operator");
  }
  global_pooling_op = nullptr;

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["bytes"] = benchmark::Counter(
    uint64_t(state.iterations()) *
      batch_size * (input_height * input_width + 1) * channels * sizeof(uint16_t),
    benchmark::Counter::kIsRate);
}
#endif  // XNN_NO_F16_OPERATORS

static void global_average_pooling_f32(benchmark::State& state) {
  const size_t batch_size = state.range(0);
  const size_t input_height = state.range(1);
  const size_t input_width = state.range(2);
  const size_t channels = state.range(3);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<float> input(batch_size * input_height * input_width * channels);
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::vector<float> output(batch_size * channels);

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
  }

  xnn_operator_t global_pooling_op = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    channels, channels /* input stride */, channels /* output stride */,
    -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity(),
    0 /* flags */, &global_pooling_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to create Global Average Pooling operator");
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    global_pooling_op,
    batch_size, input_height * input_width,
    input.data(), output.data(),
    nullptr /* thread pool */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup Global Average Pooling operator");
  }

  for (auto _ : state) {
    xnn_run_operator(global_pooling_op, nullptr /* thread pool */);
  }

  status = xnn_delete_operator(global_pooling_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete Global Average Pooling operator");
  }
  global_pooling_op = nullptr;

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["bytes"] = benchmark::Counter(
    uint64_t(state.iterations()) *
      batch_size * (input_height * input_width + 1) * channels * sizeof(float),
    benchmark::Counter::kIsRate);
}

static void ImageNetArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "C"});

  /*       N  IH  IW    C */
  b->Args({1,  7,  7, 1000});
  b->Args({1, 13, 13, 1000});
}

#ifndef XNN_NO_QU8_OPERATORS
BENCHMARK(global_average_pooling_qu8)->Apply(ImageNetArguments)->UseRealTime();
#endif  // XNN_NO_QU8_OPERATORS
#ifndef XNN_NO_QS8_OPERATORS
BENCHMARK(global_average_pooling_qs8)->Apply(ImageNetArguments)->UseRealTime();
#endif  // XNN_NO_QS8_OPERATORS
#ifndef XNN_NO_F16_OPERATORS
BENCHMARK(global_average_pooling_f16)->Apply(ImageNetArguments)->UseRealTime();
#endif  // XNN_NO_F16_OPERATORS
BENCHMARK(global_average_pooling_f32)->Apply(ImageNetArguments)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
