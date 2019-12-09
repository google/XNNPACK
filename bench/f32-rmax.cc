// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "bench/utils.h"
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/common.h>
#include <xnnpack/params.h>
#include <xnnpack/rmax.h>


static void f32_rmax(
  benchmark::State& state,
  xnn_f32_rmax_ukernel_function f32_rmax)
{
  const size_t n = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f), rng);

  std::vector<float, AlignedAllocator<float, 64>> x(n);
  std::generate(x.begin(), x.end(), std::ref(f32rng));

  float y;
  for (auto _ : state) {
    f32_rmax(n * sizeof(float), x.data(), &y);
  }

    state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();

    state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * n, benchmark::Counter::kIsRate);

  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * n * sizeof(float), benchmark::Counter::kIsRate);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rmax, sse, xnn_f32_rmax_ukernel__sse)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_rmax, avx, xnn_f32_rmax_ukernel__avx)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_rmax, avx512f, xnn_f32_rmax_ukernel__avx512f)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_rmax, neon, xnn_f32_rmax_ukernel__neon)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
BENCHMARK_CAPTURE(f32_rmax, scalar, xnn_f32_rmax_ukernel__scalar)
  ->RangeMultiplier(10)
  ->Range(1000, 100000000)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
