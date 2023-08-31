// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <xnnpack.h>

#include <benchmark/benchmark.h>
#include "bench/utils.h"

void xnnpack_fully_connected_f32(benchmark::State& state, const char* net) {
  const size_t batch_size = state.range(0);
  const size_t input_channels = state.range(1);
  const size_t output_channels = state.range(2);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));

  std::vector<float> input(batch_size * input_channels + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::vector<float> kernel(input_channels * output_channels);
  std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));
  std::vector<float> bias(output_channels);
  std::generate(bias.begin(), bias.end(), std::ref(f32rng));
  const size_t output_elements = batch_size * output_channels;

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (kernel.size() + bias.size() + output_elements));
  std::vector<float> output(output_elements * num_buffers);

  std::vector<xnn_operator_t> ops(num_buffers);
  for (xnn_operator_t& op : ops) {
    status = xnn_create_fully_connected_nc_f32(
      input_channels, output_channels,
      input_channels, output_channels,
      kernel.data(), bias.data(),
      -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity(),
      /*flags=*/0, nullptr, nullptr, &op);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to create FP32 Fully Connected operator");
      return;
    }
  }

  for (size_t i = 0; i < ops.size(); i++) {
    status = xnn_reshape_fully_connected_nc_f32(
      ops[i],
      batch_size,
      /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to setup FP32 Fully Connected operator");
      return;
    }
  }

  for (size_t i = 0; i < ops.size(); i++) {
    status = xnn_setup_fully_connected_nc_f32(
      ops[i],
      input.data(), output.data() + i * output_elements);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to setup FP32 Fully Connected operator");
      return;
    }
  }

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    status = xnn_run_operator(ops[buffer_index], /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run FP32 Fully Connected operator");
      return;
    }
  }

  for (xnn_operator_t& op : ops) {
    status = xnn_delete_operator(op);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to delete FP32 Fully Connected operator");
      return;
    }
    op = nullptr;
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 *
      batch_size * input_channels * output_channels,
    benchmark::Counter::kIsRate);
}

void xnnpack_dynamic_fully_connected_f32(benchmark::State& state, const char* net) {
  const size_t batch_size = state.range(0);
  const size_t input_channels = state.range(1);
  const size_t output_channels = state.range(2);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));

  std::vector<float> input(batch_size * input_channels + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::vector<float> kernel(input_channels * output_channels);
  std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));
  std::vector<float> bias(output_channels);
  std::generate(bias.begin(), bias.end(), std::ref(f32rng));
  const size_t output_elements = batch_size * output_channels;

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (kernel.size() + bias.size() + output_elements));
  std::vector<float> output(output_elements * num_buffers);

  std::vector<xnn_operator_t> ops(num_buffers);
  for (xnn_operator_t& op : ops) {
    status = xnn_create_dynamic_fully_connected_nc_f32(
      -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity(),
      /*flags=*/0, &op);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to create FP32 Dynamic Fully Connected operator");
      return;
    }
  }

  std::vector<std::unique_ptr<std::vector<char>>> workspaces;

  for (size_t i = 0; i < ops.size(); i++) {
    size_t workspace_size = 0;
    size_t workspace_alignment = 0;
    status = xnn_reshape_dynamic_fully_connected_nc_f32(
      ops[i],
      batch_size,
      input_channels, output_channels,
      input_channels, output_channels,
      &workspace_size, &workspace_alignment,
      /*threadpool=*/nullptr);

    if (status != xnn_status_success) {
      state.SkipWithError("failed to reshape FP32 Dynamic Fully Connected operator");
      return;
    }

    auto workspace = std::make_unique<std::vector<char>>(workspace_size);
    char* workspace_ptr = workspace->data();

    workspaces.push_back(std::move(workspace));

    status = xnn_setup_dynamic_fully_connected_nc_f32(
      ops[i],
      workspace_ptr,
      input.data(),
      kernel.data(),
      bias.data(),
      output.data() + i * output_elements);

    if (status != xnn_status_success) {
      state.SkipWithError("failed to setup FP32 Dynamic Fully Connected operator");
      return;
    }
  }

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    status = xnn_run_operator(ops[buffer_index], /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run FP32 Dynamic Fully Connected operator");
      return;
    }
  }

  for (xnn_operator_t& op : ops) {
    status = xnn_delete_operator(op);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to delete FP32 Dynamic Fully Connected operator");
      return;
    }
    op = nullptr;
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 *
      batch_size * input_channels * output_channels,
    benchmark::Counter::kIsRate);
}

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
