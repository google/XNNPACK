// Copyright 2022 Google LLC
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

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/vunary.h>


static void u64_u32_vsqrtshift(
  benchmark::State& state,
  xnn_u64_u32_vsqrtshift_ukernel_fn vsqrtshift,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }
  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto u64rng = std::bind(std::uniform_int_distribution<uint64_t>(), std::ref(rng));

  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> x(num_elements + XNN_EXTRA_BYTES / sizeof(uint64_t));
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), std::ref(u64rng));
  std::fill(y.begin(), y.end(), UINT32_C(0xDEADBEEF));

  for (auto _ : state) {
    vsqrtshift(num_elements * sizeof(uint64_t), x.data(), y.data(), 1 /* shift */);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = num_elements * (sizeof(uint64_t) + sizeof(uint32_t));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

BENCHMARK_CAPTURE(u64_u32_vsqrtshift, scalar_cvtu32_sqrt_cvtu32f64_u1,
                  xnn_u64_u32_vsqrtshift_ukernel__scalar_cvtu32_sqrt_cvtu32f64_u1)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint64_t, uint32_t>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
