// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
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


static void add_nc_q8(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  const size_t channels = static_cast<size_t>(state.range(1));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

  std::vector<uint8_t> a(batch_size * channels);
  std::vector<uint8_t> b(batch_size * channels);
  std::vector<uint8_t> y(batch_size * channels);
  std::generate(a.begin(), a.end(), std::ref(u8rng));
  std::generate(b.begin(), b.end(), std::ref(u8rng));

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t add_op = nullptr;
  status = xnn_create_add_nc_q8(
    channels, channels /* a_stride */, channels /* b_stride */, channels /* sum_stride */,
    127 /* a:zero point */, 1.0f /* a:scale */,
    127 /* b:zero point */, 1.0f /* b:scale */,
    127 /* y:zero point */, 1.0f /* y:scale */,
    1 /* y:min */, 254 /* y:max */,
    0 /* flags */, &add_op);
  if (status != xnn_status_success || add_op == nullptr) {
    state.SkipWithError("failed to create Q8 Add operator");
    return;
  }

  status = xnn_setup_add_nc_q8(
    add_op,
    batch_size,
    a.data(), b.data(), y.data(),
    nullptr /* thread pool */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup Q8 Add operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(add_op, nullptr /* thread pool */);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run Q8 Add operator");
      return;
    }
  }

  status = xnn_delete_operator(add_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete Q8 Add operator");
    return;
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();

  const size_t elements_per_iteration = batch_size * channels;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 3 * elements_per_iteration * sizeof(uint8_t);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

static void add_nc_q8_inplace(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  const size_t channels = static_cast<size_t>(state.range(1));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

  std::vector<uint8_t> a(batch_size * channels);
  std::vector<uint8_t> y(batch_size * channels);
  std::generate(a.begin(), a.end(), std::ref(u8rng));

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t add_op = nullptr;
  status = xnn_create_add_nc_q8(
    channels, channels /* a_stride */, channels /* b_stride */, channels /* sum_stride */,
    127 /* a:zero point */, 1.0f /* a:scale */,
    127 /* b:zero point */, 1.0f /* b:scale */,
    127 /* y:zero point */, 1.0f /* y:scale */,
    1 /* y:min */, 254 /* y:max */,
    0 /* flags */, &add_op);
  if (status != xnn_status_success || add_op == nullptr) {
    state.SkipWithError("failed to create Q8 Add operator");
    return;
  }

  status = xnn_setup_add_nc_q8(
    add_op,
    batch_size,
    a.data(), y.data(), y.data(),
    nullptr /* thread pool */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup Q8 Add operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(add_op, nullptr /* thread pool */);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run Q8 Add operator");
      return;
    }
  }

  status = xnn_delete_operator(add_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete Q8 Add operator");
    return;
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();

  const size_t elements_per_iteration = batch_size * channels;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 3 * elements_per_iteration * sizeof(uint8_t);
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

BENCHMARK(add_nc_q8)->Apply(CharacteristicArguments)->UseRealTime();
BENCHMARK(add_nc_q8_inplace)->Apply(CharacteristicArguments)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
