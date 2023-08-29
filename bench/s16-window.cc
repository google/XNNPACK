// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include "bench/utils.h"
#include <benchmark/benchmark.h>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/window.h>


void s16_window(
    benchmark::State& state,
    xnn_s16_window_ukernel_fn window,
    benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }
  const size_t rows = state.range(0);
  const size_t channels = state.range(1);

  std::vector<int16_t, AlignedAllocator<int16_t, 64>> input(
      (rows * channels) + XNN_EXTRA_BYTES / sizeof(int16_t));
  std::vector<int16_t, AlignedAllocator<int16_t, 64>> weights(
      channels + XNN_EXTRA_BYTES / sizeof(int16_t));
  std::vector<int16_t, AlignedAllocator<int16_t, 64>> output(
      (rows * channels) + XNN_EXTRA_BYTES / sizeof(int16_t));
  std::iota(input.begin(), input.end(), 0);
  std::fill(weights.begin(), weights.end(), 0);
  std::iota(output.begin(), output.end(), 0);

  for (auto _ : state) {
    window(rows, channels * sizeof(int16_t), input.data(), weights.data(), output.data(), 12 /* shift */);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void BenchmarkKernelSize(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"rows", "channels"});
  b->Args({1, 400});
  b->Args({10, 400});
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(s16_window, s16_neon_u8,
                    xnn_s16_window_ukernel__neon_u8,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkKernelSize)
    ->UseRealTime();
  BENCHMARK_CAPTURE(s16_window, s16_neon_u16,
                    xnn_s16_window_ukernel__neon_u16,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkKernelSize)
    ->UseRealTime();
  BENCHMARK_CAPTURE(s16_window, s16_neon_u24,
                    xnn_s16_window_ukernel__neon_u24,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkKernelSize)
    ->UseRealTime();
  BENCHMARK_CAPTURE(s16_window, s16_neon_u32,
                    xnn_s16_window_ukernel__neon_u32,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkKernelSize)
    ->UseRealTime();

  BENCHMARK_CAPTURE(s16_window, s16_shift12_neon_u8,
                    xnn_s16_window_shift12_ukernel__neon_u8,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkKernelSize)
    ->UseRealTime();
  BENCHMARK_CAPTURE(s16_window, s16_shift12_neon_u16,
                    xnn_s16_window_shift12_ukernel__neon_u16,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkKernelSize)
    ->UseRealTime();
  BENCHMARK_CAPTURE(s16_window, s16_shift12_neon_u24,
                    xnn_s16_window_shift12_ukernel__neon_u24,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkKernelSize)
    ->UseRealTime();
  BENCHMARK_CAPTURE(s16_window, s16_shift12_neon_u32,
                    xnn_s16_window_shift12_ukernel__neon_u32,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkKernelSize)
    ->UseRealTime();

  BENCHMARK_CAPTURE(s16_window, s16_shift15_neon_u8,
                    xnn_s16_window_shift15_ukernel__neon_u8,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkKernelSize)
    ->UseRealTime();
  BENCHMARK_CAPTURE(s16_window, s16_shift15_neon_u16,
                    xnn_s16_window_shift15_ukernel__neon_u16,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkKernelSize)
    ->UseRealTime();
  BENCHMARK_CAPTURE(s16_window, s16_shift15_neon_u24,
                    xnn_s16_window_shift15_ukernel__neon_u24,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkKernelSize)
    ->UseRealTime();
  BENCHMARK_CAPTURE(s16_window, s16_shift15_neon_u32,
                    xnn_s16_window_shift15_ukernel__neon_u32,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkKernelSize)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

BENCHMARK_CAPTURE(s16_window, s16_scalar_u1,
                  xnn_s16_window_ukernel__scalar_u1)
  ->Apply(BenchmarkKernelSize)
  ->UseRealTime();
BENCHMARK_CAPTURE(s16_window, s16_scalar_u2,
                  xnn_s16_window_ukernel__scalar_u2)
  ->Apply(BenchmarkKernelSize)
  ->UseRealTime();
BENCHMARK_CAPTURE(s16_window, s16_scalar_u3,
                  xnn_s16_window_ukernel__scalar_u3)
  ->Apply(BenchmarkKernelSize)
  ->UseRealTime();
BENCHMARK_CAPTURE(s16_window, s16_scalar_u4,
                  xnn_s16_window_ukernel__scalar_u4)
  ->Apply(BenchmarkKernelSize)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
