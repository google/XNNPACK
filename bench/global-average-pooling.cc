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


static void global_average_pooling_q8(benchmark::State& state) {
  const size_t batch_size = state.range(0);
  const size_t input_height = state.range(1);
  const size_t input_width = state.range(2);
  const size_t channels = state.range(3);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

  std::vector<uint8_t> input(batch_size * input_height * input_width * channels);
  std::generate(input.begin(), input.end(), std::ref(u8rng));
  std::vector<uint8_t> output(batch_size * channels);

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
  }

  xnn_operator_t global_pooling_op = nullptr;
  status = xnn_create_global_average_pooling_nwc_q8(
    channels, channels /* input stride */, channels /* output stride */,
    127 /* input zero point */, 0.75f /* input scale */,
    127 /* output zero point */, 1.25f /* output scale */,
    0, 255,
    0 /* flags */, &global_pooling_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to create Global Average Pooling operator");
  }

  status = xnn_setup_global_average_pooling_nwc_q8(
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

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["bytes"] = benchmark::Counter(
    uint64_t(state.iterations()) *
      batch_size * (input_height * input_width + 1) * channels * sizeof(uint8_t),
    benchmark::Counter::kIsRate);
}

static void ImageNetArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "C"});

  /*       N  IH  IW    C */
  b->Args({1,  7,  7, 1000});
  b->Args({1, 13, 13, 1000});
}

BENCHMARK(global_average_pooling_q8)->Apply(ImageNetArguments)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
