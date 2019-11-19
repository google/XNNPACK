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
#include <random>
#include <vector>

#include <xnnpack.h>

#include <benchmark/benchmark.h>
#include "bench/utils.h"


static void average_pooling_q8(benchmark::State& state, const char* net) {
  const size_t batch_size = state.range(0);
  const size_t input_height = state.range(1);
  const size_t input_width = state.range(2);
  const size_t pooling_size = state.range(3);
  const size_t padding_size = state.range(4);
  const size_t stride = state.range(5);
  const size_t channels = state.range(6);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

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
  status = xnn_create_average_pooling2d_nhwc_q8(
    padding_size, padding_size, padding_size, padding_size,
    pooling_size, pooling_size,
    stride, stride,
    channels, channels /* input pixel stride */, channels /* output pixel stride */,
    127 /* input zero point */, 0.75f /* input scale */,
    127 /* output zero point */, 1.25f /* output scale */,
    0, 255,
    0 /* flags */, &pooling_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to create Average Pooling operator");
    return;
  }

  status = xnn_setup_average_pooling2d_nhwc_q8(
    pooling_op,
    batch_size, input_height, input_width,
    input.data(), output.data(),
    nullptr /* thread pool */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup Average Pooling operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(pooling_op, nullptr /* thread pool */);
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

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();

  state.counters["bytes"] = benchmark::Counter(
    uint64_t(state.iterations()) *
      batch_size * (input_height * input_width + output_height * output_width) * channels * sizeof(uint8_t),
    benchmark::Counter::kIsRate);
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

BENCHMARK_CAPTURE(average_pooling_q8, shufflenet_v1_g1, "ShuffleNet v1 (1 group)")->Apply(ShuffleNetV1G1)->UseRealTime();
BENCHMARK_CAPTURE(average_pooling_q8, shufflenet_v1_g2, "ShuffleNet v1 (2 groups)")->Apply(ShuffleNetV1G2)->UseRealTime();
BENCHMARK_CAPTURE(average_pooling_q8, shufflenet_v1_g3, "ShuffleNet v1 (3 groups)")->Apply(ShuffleNetV1G3)->UseRealTime();
BENCHMARK_CAPTURE(average_pooling_q8, shufflenet_v1_g4, "ShuffleNet v1 (4 groups)")->Apply(ShuffleNetV1G4)->UseRealTime();
BENCHMARK_CAPTURE(average_pooling_q8, shufflenet_v1_g8, "ShuffleNet v1 (8 groups)")->Apply(ShuffleNetV1G8)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
