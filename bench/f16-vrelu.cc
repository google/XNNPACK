// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include <fp16/fp16.h>
#include "bench/utils.h"
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/common.h>
#include <xnnpack/params.h>
#include <xnnpack/vunary.h>


static void f16_vrelu(
  benchmark::State& state,
  xnn_f16_vrelu_ukernel_function f16_vrelu,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }

  const size_t elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f), std::ref(rng));
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> x(elements);
  std::generate(x.begin(), x.end(), std::ref(f16rng));
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> y(elements);
  std::generate(x.begin(), x.end(), std::ref(f16rng));

  for (auto _ : state) {
    f16_vrelu(elements * sizeof(uint16_t), x.data(), y.data(), NULL);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * elements * sizeof(uint16_t);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f16_vrelu, neonfp16arith_x8, xnn_f16_vrelu_ukernel__neonfp16arith_x8, benchmark::utils::CheckNEONFP16ARITH)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_vrelu, neonfp16arith_x16, xnn_f16_vrelu_ukernel__neonfp16arith_x16, benchmark::utils::CheckNEONFP16ARITH)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
