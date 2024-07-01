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

#include "xnnpack.h"

#include <benchmark/benchmark.h>
#include "bench/utils.h"


static void channel_shuffle_x8(benchmark::State& state, const char* net) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  const size_t groups = static_cast<size_t>(state.range(1));
  const size_t group_channels = static_cast<size_t>(state.range(2));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), std::ref(rng));

  std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) + batch_size * groups * group_channels);
  std::vector<uint8_t> output(batch_size * groups * group_channels);
  std::generate(input.begin(), input.end(), std::ref(u8rng));

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t channel_shuffle_op = nullptr;
  status = xnn_create_channel_shuffle_nc_x8(
    groups, group_channels,
    groups * group_channels /* input stride */,
    groups * group_channels /* output stride */,
    0 /* flags */, &channel_shuffle_op);
  if (status != xnn_status_success || channel_shuffle_op == nullptr) {
    state.SkipWithError("failed to create X8 Channel Shuffle operator");
    return;
  }

  status = xnn_reshape_channel_shuffle_nc_x8(
    channel_shuffle_op,
    batch_size,
    /*threadpool=*/nullptr);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape X8 Channel Shuffle operator");
    return;
  }

  status = xnn_setup_channel_shuffle_nc_x8(
    channel_shuffle_op,
    input.data(), output.data());
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup X8 Channel Shuffle operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(channel_shuffle_op, /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run X8 Channel Shuffle operator");
      return;
    }
  }

  status = xnn_delete_operator(channel_shuffle_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete X8 Channel Shuffle operator");
    return;
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = batch_size * groups * group_channels;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * elements_per_iteration * sizeof(uint8_t);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

static void channel_shuffle_x32(benchmark::State& state, const char* net) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  const size_t groups = static_cast<size_t>(state.range(1));
  const size_t group_channels = static_cast<size_t>(state.range(2));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) + batch_size * groups * group_channels);
  std::vector<float> output(batch_size * groups * group_channels);
  std::generate(input.begin(), input.end(), std::ref(f32rng));

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t channel_shuffle_op = nullptr;
  status = xnn_create_channel_shuffle_nc_x32(
    groups, group_channels,
    groups * group_channels /* input stride */,
    groups * group_channels /* output stride */,
    0 /* flags */, &channel_shuffle_op);
  if (status != xnn_status_success || channel_shuffle_op == nullptr) {
    state.SkipWithError("failed to create X32 Channel Shuffle operator");
    return;
  }

  status = xnn_reshape_channel_shuffle_nc_x32(
    channel_shuffle_op,
    batch_size,
    /*threadpool=*/nullptr);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape X32 Channel Shuffle operator");
    return;
  }

  status = xnn_setup_channel_shuffle_nc_x32(
    channel_shuffle_op,
    input.data(), output.data());
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup X32 Channel Shuffle operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(channel_shuffle_op, /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run X32 Channel Shuffle operator");
      return;
    }
  }

  status = xnn_delete_operator(channel_shuffle_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete X32 Channel Shuffle operator");
    return;
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = batch_size * groups * group_channels;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * elements_per_iteration * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

static void ShuffleNetV1G2Arguments(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"N", "G", "GC"});

  /******** Stage 2 ********/
  /*        H    W  G   CG */
  b->Args({56 * 56, 2,  25});
  b->Args({28 * 28, 2,  25});

  /******** Stage 3 ********/
  /*        H    W  G   CG */
  b->Args({28 * 28, 2,  50});
  b->Args({14 * 14, 2,  50});

  /******** Stage 4 ********/
  /*        H    W  G   CG */
  b->Args({14 * 14, 2, 100});
  b->Args({ 7 *  7, 2, 100});
}

static void ShuffleNetV1G3Arguments(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"N", "G", "GC"});

  /******** Stage 2 *******/
  /*        H    W  G  CG */
  b->Args({56 * 56, 3, 20});
  b->Args({28 * 28, 3, 20});

  /******** Stage 3 *******/
  /*        H    W  G  CG */
  b->Args({28 * 28, 3, 40});
  b->Args({14 * 14, 3, 40});

  /******** Stage 4 *******/
  /*        H    W  G  CG */
  b->Args({14 * 14, 3, 80});
  b->Args({ 7 *  7, 3, 80});
}

static void ShuffleNetV1G4Arguments(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"N", "G", "GC"});

  /******** Stage 2 *******/
  /*        H    W  G  CG */
  b->Args({56 * 56, 4, 17});
  b->Args({28 * 28, 4, 17});

  /******** Stage 3 *******/
  /*        H    W  G  CG */
  b->Args({28 * 28, 4, 34});
  b->Args({14 * 14, 4, 34});

  /******** Stage 4 *******/
  /*        H    W  G  CG */
  b->Args({14 * 14, 4, 68});
  b->Args({ 7 *  7, 4, 68});
}

static void ShuffleNetV1G8Arguments(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"N", "G", "GC"});

  /******** Stage 2 *******/
  /*        H    W  G  CG */
  b->Args({56 * 56, 8, 12});
  b->Args({28 * 28, 8, 12});

  /******** Stage 3 *******/
  /*        H    W  G  CG */
  b->Args({28 * 28, 8, 24});
  b->Args({14 * 14, 8, 24});

  /******** Stage 4 *******/
  /*        H    W  G  CG */
  b->Args({14 * 14, 8, 48});
  b->Args({ 7 *  7, 8, 48});
}

static void ShuffleNetV2x0_5Arguments(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"N", "G", "GC"});

  /******** Stage 2 *******/
  /*        H    W  G  CG */
  b->Args({28 * 28, 2, 24});

  /******** Stage 3 *******/
  /*        H    W  G  CG */
  b->Args({14 * 14, 2, 48});

  /******** Stage 4 *******/
  /*        H    W  G  CG */
  b->Args({ 7 *  7, 2, 96});
}

static void ShuffleNetV2x1_0Arguments(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"N", "G", "GC"});

  /******** Stage 2 ********/
  /*        H    W  G   CG */
  b->Args({28 * 28, 2,  58});

  /******** Stage 3 ********/
  /*        H    W  G   CG */
  b->Args({14 * 14, 2, 116});

  /******** Stage 4 ********/
  /*        H    W  G   CG */
  b->Args({ 7 *  7, 2, 232});
}

static void ShuffleNetV2x1_5Arguments(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"N", "G", "GC"});

  /******** Stage 2 ********/
  /*        H    W  G   CG */
  b->Args({28 * 28, 2,  88});

  /******** Stage 3 ********/
  /*        H    W  G   CG */
  b->Args({14 * 14, 2, 176});

  /******** Stage 4 ********/
  /*        H    W  G   CG */
  b->Args({ 7 *  7, 2, 352});
}

static void ShuffleNetV2x2_0Arguments(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"N", "G", "GC"});

  /******** Stage 2 ********/
  /*        H    W  G   CG */
  b->Args({28 * 28, 2, 122});

  /******** Stage 3 ********/
  /*        H    W  G   CG */
  b->Args({14 * 14, 2, 244});

  /******** Stage 4 ********/
  /*        H    W  G   CG */
  b->Args({ 7 *  7, 2, 488});
}

BENCHMARK_CAPTURE(channel_shuffle_x8, shufflenet_v1_g2, "ShuffleNet v1 (2 groups)")->Apply(ShuffleNetV1G2Arguments)->UseRealTime();
BENCHMARK_CAPTURE(channel_shuffle_x8, shufflenet_v1_g3, "ShuffleNet v1 (3 groups)")->Apply(ShuffleNetV1G3Arguments)->UseRealTime();
BENCHMARK_CAPTURE(channel_shuffle_x8, shufflenet_v1_g4, "ShuffleNet v1 (4 groups)")->Apply(ShuffleNetV1G4Arguments)->UseRealTime();
BENCHMARK_CAPTURE(channel_shuffle_x8, shufflenet_v1_g8, "ShuffleNet v1 (8 groups)")->Apply(ShuffleNetV1G8Arguments)->UseRealTime();
BENCHMARK_CAPTURE(channel_shuffle_x8, shufflenet_v2_x05, "ShuffleNet v2 x0.5")->Apply(ShuffleNetV2x0_5Arguments)->UseRealTime();
BENCHMARK_CAPTURE(channel_shuffle_x8, shufflenet_v2_x10, "ShuffleNet v2 x1.0")->Apply(ShuffleNetV2x1_0Arguments)->UseRealTime();
BENCHMARK_CAPTURE(channel_shuffle_x8, shufflenet_v2_x15, "ShuffleNet v2 x1.5")->Apply(ShuffleNetV2x1_5Arguments)->UseRealTime();
BENCHMARK_CAPTURE(channel_shuffle_x8, shufflenet_v2_x20, "ShuffleNet v2 x2.0")->Apply(ShuffleNetV2x2_0Arguments)->UseRealTime();

BENCHMARK_CAPTURE(channel_shuffle_x32, shufflenet_v1_g2, "ShuffleNet v1 (2 groups)")->Apply(ShuffleNetV1G2Arguments)->UseRealTime();
BENCHMARK_CAPTURE(channel_shuffle_x32, shufflenet_v1_g3, "ShuffleNet v1 (3 groups)")->Apply(ShuffleNetV1G3Arguments)->UseRealTime();
BENCHMARK_CAPTURE(channel_shuffle_x32, shufflenet_v1_g4, "ShuffleNet v1 (4 groups)")->Apply(ShuffleNetV1G4Arguments)->UseRealTime();
BENCHMARK_CAPTURE(channel_shuffle_x32, shufflenet_v1_g8, "ShuffleNet v1 (8 groups)")->Apply(ShuffleNetV1G8Arguments)->UseRealTime();
BENCHMARK_CAPTURE(channel_shuffle_x32, shufflenet_v2_x05, "ShuffleNet v2 x0.5")->Apply(ShuffleNetV2x0_5Arguments)->UseRealTime();
BENCHMARK_CAPTURE(channel_shuffle_x32, shufflenet_v2_x10, "ShuffleNet v2 x1.0")->Apply(ShuffleNetV2x1_0Arguments)->UseRealTime();
BENCHMARK_CAPTURE(channel_shuffle_x32, shufflenet_v2_x15, "ShuffleNet v2 x1.5")->Apply(ShuffleNetV2x1_5Arguments)->UseRealTime();
BENCHMARK_CAPTURE(channel_shuffle_x32, shufflenet_v2_x20, "ShuffleNet v2 x2.0")->Apply(ShuffleNetV2x2_0Arguments)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
