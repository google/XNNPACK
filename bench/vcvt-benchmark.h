// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "bench/utils.h"

#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"


namespace {

template <typename In, typename Out, typename UKernelFn, typename Params>
void cvt_benchmark(
  benchmark::State& state,
  uint64_t arch_flags,
  UKernelFn cvt,
  const Params* params)
{
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 10.0f), std::ref(rng));

  std::vector<In, AlignedAllocator<In, 64>> x(num_elements + XNN_EXTRA_BYTES / sizeof(In));
  std::vector<Out, AlignedAllocator<Out, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), f32rng);

  for (auto _ : state) {
    cvt(num_elements * sizeof(In), x.data(), y.data(), params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = num_elements * (sizeof(In) + sizeof(Out));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

template <typename In, typename Out, typename UKernelFn>
void cvt_benchmark(
    benchmark::State& state,
    uint64_t arch_flags,
    UKernelFn cvt,
    std::nullptr_t) {
  cvt_benchmark<In, Out, UKernelFn, void>(state, arch_flags, cvt, nullptr);
}

}  // namespace
