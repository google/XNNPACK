// Copyright 2020 Google LLC
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


static void xnnpack_truncation_f32(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f), std::ref(rng));

  std::vector<float> input(batch_size + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> output(batch_size);
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::fill(output.begin(), output.end(), std::nanf(""));

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t truncation_op = nullptr;
  status = xnn_create_truncation_nc_f32(
    0 /* flags */, &truncation_op);
  if (status != xnn_status_success || truncation_op == nullptr) {
    state.SkipWithError("failed to create Truncation operator");
    return;
  }

  status = xnn_reshape_truncation_nc_f32(truncation_op, batch_size,
    /*channels=*/1, /*input_stride=*/1, /*output_stride=*/1, /*threadpool=*/nullptr);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape Truncation operator");
    return;
  }

  status = xnn_setup_truncation_nc_f32(truncation_op, input.data(), output.data());
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup Truncation operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(truncation_op, /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run Truncation operator");
      return;
    }
  }

  status = xnn_delete_operator(truncation_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete Truncation operator");
    return;
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * batch_size, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * batch_size * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

BENCHMARK(xnnpack_truncation_f32)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
