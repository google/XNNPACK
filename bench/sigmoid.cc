// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <xnnpack.h>

#include <benchmark/benchmark.h>
#include "bench/utils.h"


static void xnn_sigmoid_q8(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  const size_t channels = static_cast<size_t>(state.range(1));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

  std::vector<uint8_t> input(batch_size * channels);
  std::vector<uint8_t> output(batch_size * channels);
  std::generate(input.begin(), input.end(), std::ref(u8rng));
  std::fill(output.begin(), output.end(), 0xA5);

  xnn_status status = xnn_initialize();
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

static void xnn_sigmoid_f32(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  const size_t channels = static_cast<size_t>(state.range(1));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-25.0f, 25.0f), rng);

  std::vector<float> input(batch_size * channels);
  std::vector<float> output(batch_size * channels);
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::fill(output.begin(), output.end(), std::nanf(""));

  xnn_status status = xnn_initialize();
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

static void CharacteristicArguments(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"N", "C"});

  int32_t c = 16;
  for (int32_t n = 224; n >= 7; n /= 2) {
    b->Args({n * n, c});
    c *= 2;
  }
}

BENCHMARK(xnn_sigmoid_q8)->Apply(CharacteristicArguments)->UseRealTime();
BENCHMARK(xnn_sigmoid_f32)->Apply(CharacteristicArguments)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
