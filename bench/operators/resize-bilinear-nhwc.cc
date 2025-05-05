// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "bench/utils.h"
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include <benchmark/benchmark.h>

template <typename T>
static void xnnpack_resize_bilinear(benchmark::State& state) {
  const size_t in_width = static_cast<size_t>(state.range(0));
  const size_t in_height = static_cast<size_t>(state.range(1));
  const size_t out_width = static_cast<size_t>(state.range(2));
  const size_t out_height = static_cast<size_t>(state.range(3));
  const size_t channels = static_cast<size_t>(state.range(4));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());

  xnnpack::Buffer<T> input(in_height * in_width * channels,
                           xnnpack::XnnExtraBytes);
  xnnpack::Buffer<T> output(out_height * out_width * channels);

  xnnpack::fill_uniform_random_bits(input.data(), input.size(), rng);

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t op = nullptr;
  status = xnn_create_resize_bilinear2d_nhwc(xnn_datatype_of<T>(), out_height,
                                             out_width, 0, &op);
  if (status != xnn_status_success || op == nullptr) {
    state.SkipWithError("failed to create ResizeBilinear operator");
    return;
  }

  size_t workspace_size = 0;
  size_t workspace_alignment = 1;
  status = xnn_reshape_resize_bilinear2d_nhwc(
      op, 1, in_height, in_width, channels, channels, channels, &workspace_size,
      &workspace_alignment, /*threadpool=*/nullptr);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape ResizeBilinear operator");
    return;
  }

  xnnpack::Buffer<T, XNN_ALLOCATION_ALIGNMENT> workspace(workspace_size);

  status = xnn_setup_resize_bilinear2d_nhwc(op, workspace.data(), input.data(),
                                            output.data());
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup ResizeBilinear operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(op, /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run ResizeBilinear operator");
      return;
    }
  }

  status = xnn_delete_operator(op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete ResizeBilinear operator");
    return;
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void CharacteristicArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"InW", "InH", "OutW", "OutH", "C"});

  for (int c : {1, 3, 4}) {
    b->Args({256, 256, 512, 512, c});
    b->Args({512, 512, 256, 256, c});
  }

  b->Args({12, 8, 24, 16, 256});
  b->Args({5, 33, 10, 33, 192});
  b->Args({20, 33, 40, 33, 96});
  b->Args({6, 5, 12, 10, 96});
  b->Args({24, 20, 48, 40, 48});
  b->Args({80, 66, 160, 66, 24});
}

static void xnnpack_resize_bilinear_f32(benchmark::State& state) {
  xnnpack_resize_bilinear<float>(state);
}
static void xnnpack_resize_bilinear_f16(benchmark::State& state) {
  xnnpack_resize_bilinear<xnn_float16>(state);
}
static void xnnpack_resize_bilinear_qu8(benchmark::State& state) {
  xnnpack_resize_bilinear<xnnpack::quantized<uint8_t>>(state);
}
static void xnnpack_resize_bilinear_qs8(benchmark::State& state) {
  xnnpack_resize_bilinear<xnnpack::quantized<int8_t>>(state);
}

BENCHMARK(xnnpack_resize_bilinear_f32)
    ->Apply(CharacteristicArguments)
    ->UseRealTime();
BENCHMARK(xnnpack_resize_bilinear_f16)
    ->Apply(CharacteristicArguments)
    ->UseRealTime();
BENCHMARK(xnnpack_resize_bilinear_qu8)
    ->Apply(CharacteristicArguments)
    ->UseRealTime();
BENCHMARK(xnnpack_resize_bilinear_qs8)
    ->Apply(CharacteristicArguments)
    ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
