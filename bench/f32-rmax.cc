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
  xnn_f32_rmax_ukernel_function f32_rmax,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }

  const size_t elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f), std::ref(rng));

  std::vector<float, AlignedAllocator<float, 64>> x(elements);
  std::generate(x.begin(), x.end(), std::ref(f32rng));

  float y;
  for (auto _ : state) {
    f32_rmax(elements * sizeof(float), x.data(), &y);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = elements * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rmax, sse, xnn_f32_rmax_ukernel__sse)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_rmax, avx, xnn_f32_rmax_ukernel__avx, benchmark::utils::CheckAVX)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_rmax, avx512f, xnn_f32_rmax_ukernel__avx512f, benchmark::utils::CheckAVX512F)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_rmax, neon, xnn_f32_rmax_ukernel__neon, benchmark::utils::CheckNEON)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_rmax, wasmsimd_arm, xnn_f32_rmax_ukernel__wasmsimd_arm)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_rmax, wasmsimd_x86, xnn_f32_rmax_ukernel__wasmsimd_x86)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(f32_rmax, scalar, xnn_f32_rmax_ukernel__scalar)
  ->RangeMultiplier(10)
  ->Range(1000, 100000000)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
