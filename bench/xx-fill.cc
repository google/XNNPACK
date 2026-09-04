// Copyright 2026 Léandre Le Duc
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <benchmark/benchmark.h>

#include <cmath>
#include <vector>

#include "bench/utils.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/fill.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/microfnptr.h"

static void xx_fill(benchmark::State& state, xnn_fill_ukernel_fn fill,
                    uint64_t arch_flags = 0) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t rows = state.range(0);
  const size_t channels = state.range(1);

  const size_t output_stride = channels;

  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> output(
      rows * output_stride, xnnpack::XnnExtraBytes);

  const uint32_t fill_pattern = 0xDEADBEEF;

  for (auto _ : state) {
    fill(rows, channels, output.data(), output_stride, fill_pattern);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = rows * output_stride;
  state.counters["elements"] =
      benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration,
                         benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = rows * output_stride;
  state.counters["bytes"] =
      benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration,
                         benchmark::Counter::kIsRate);
}

static void BenchmarkFill(benchmark::Benchmark* b) {
  b->ArgNames({"rows", "channels"});
  b->Args({1, 16});
  b->Args({1, 64});
  b->Args({1, 256});
  b->Args({1, 1024});
  b->Args({1, 8192});
  b->Args({100, 256});
  b->Args({1024, 16});
}

#define XNN_FILL_UKERNEL(arch_flags, ukernel)              \
  BENCHMARK_CAPTURE(xx_fill, ukernel, ukernel, arch_flags) \
      ->Apply(BenchmarkFill)                               \
      ->UseRealTime();
#include "src/xx-fill/xx-fill.inc"
#undef XNN_FILL_UKERNEL

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
