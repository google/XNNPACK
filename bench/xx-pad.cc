// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include "bench/utils.h"
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/pad.h"
#include "test/replicable_random_device.h"
#include <benchmark/benchmark.h>

static void xx_pad(benchmark::State& state, xnn_pad_ukernel_fn pad,
                   uint64_t arch_flags = 0) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t rows = state.range(0);
  const size_t channels = state.range(1);
  const size_t pre_padding = state.range(2);
  const size_t post_padding = state.range(3);

  const size_t input_stride = channels;
  const size_t output_stride = pre_padding + channels + post_padding;

  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> input(
      rows * input_stride, xnnpack::XnnExtraBytes);
  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> output(
      rows * output_stride, xnnpack::XnnExtraBytes);

  xnnpack::ReplicableRandomDevice rng;
  xnnpack::fill_uniform_random_bits(input.data(), input.size(), rng);
  const uint32_t fill_pattern = 0xDEADBEEF;

  for (auto _ : state) {
    pad(rows, channels, pre_padding, post_padding, input.data(), input_stride,
        output.data(), output_stride, fill_pattern);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = rows * output_stride;
  state.counters["elements"] =
      benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration,
                         benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = rows * (input_stride + output_stride);
  state.counters["bytes"] =
      benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration,
                         benchmark::Counter::kIsRate);
}

static void BenchmarkPad(benchmark::Benchmark* b) {
  b->ArgNames({"rows", "channels", "pre_padding", "post_padding"});
  b->Args({1, 16, 16, 16});
  b->Args({1, 64, 64, 64});
  b->Args({1, 256, 256, 256});
  b->Args({1, 1024, 1024, 1024});
  b->Args({1, 8192, 1024, 1024});
  b->Args({100, 256, 64, 64});
  b->Args({1024, 16, 16, 16});
}

#define XNN_PAD_UKERNEL(arch_flags, ukernel, tile_size) \
  BENCHMARK_CAPTURE(xx_pad, ukernel, ukernel, arch_flags) \
      ->Apply(BenchmarkPad)                               \
      ->UseRealTime();
#include "src/xx-pad/xx-pad.inc"
#undef XNN_PAD_UKERNEL

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
