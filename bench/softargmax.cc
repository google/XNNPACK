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


static void softargmax_q8(benchmark::State& state) {
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

  xnn_operator_t softargmax_op = nullptr;
  status = xnn_create_softargmax_nc_q8(
    channels, channels /* input stride */, channels /* output stride */,
    1.0f /* input scale */,
    0 /* output zero point */, 1.0f / 256.0f /* output scale */,
    0 /* flags */, &softargmax_op);
  if (status != xnn_status_success || softargmax_op == nullptr) {
    state.SkipWithError("failed to create SoftArgMax operator");
    return;
  }

  status = xnn_setup_softargmax_nc_q8(
    softargmax_op,
    batch_size,
    input.data(), output.data(),
    nullptr /* thread pool */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup SoftArgMax operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(softargmax_op, nullptr /* thread pool */);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run SoftArgMax operator");
      return;
    }
  }

  status = xnn_delete_operator(softargmax_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete SoftArgMax operator");
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

static void CharacteristicArguments(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"N", "C"});

  // CIFAR-10
  b->Args({1, 10});
  // CIFAR-100 */
  b->Args({1, 100});
  // ImageNet-1K
  b->Args({1, 1000});
  // ImageNet-1K+1
  b->Args({1, 1001});
  // ImageNet-22K
  b->Args({1, 21841});
}

BENCHMARK(softargmax_q8)->Apply(CharacteristicArguments)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
