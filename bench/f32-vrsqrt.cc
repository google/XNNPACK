// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/microparams.h>
#include <xnnpack/vunary.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "bench/utils.h"
#include <benchmark/benchmark.h>

static void f32_vrsqrt(benchmark::State& state,
                       xnn_f32_vrsqrt_ukernel_fn vrsqrt,
                       xnn_init_f32_rsqrt_params_fn init_params = nullptr,
                       benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);
  std::vector<float, AlignedAllocator<float, 64>> input(num_elements);
  std::vector<float, AlignedAllocator<float, 64>> output(num_elements);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(
                              std::numeric_limits<float>::epsilon(), 10.0f),
                          std::ref(rng));
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::fill(output.begin(), output.end(), std::nanf(""));

  union xnn_f32_rsqrt_params params;
  if (init_params != nullptr) {
    init_params(&params);
  }
  for (auto _ : state) {
    vrsqrt(num_elements * sizeof(float), input.data(), output.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * elements_per_iteration,
      benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * num_elements * sizeof(float);
  state.counters["bytes"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * bytes_per_iteration,
      benchmark::Counter::kIsRate);
}

BENCHMARK_CAPTURE(f32_vrsqrt, scalar_recip_sqrt_u1,
                  xnn_f32_vrsqrt_ukernel__scalar_recip_sqrt_u1)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f32_vrsqrt, scalar_recip_sqrt_u2,
                  xnn_f32_vrsqrt_ukernel__scalar_recip_sqrt_u2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f32_vrsqrt, scalar_recip_sqrt_u4,
                  xnn_f32_vrsqrt_ukernel__scalar_recip_sqrt_u4)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f32_vrsqrt, scalar_recip_sqrt_u8,
                  xnn_f32_vrsqrt_ukernel__scalar_recip_sqrt_u8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f32_vrsqrt, scalar_recip_sqrt_u16,
                  xnn_f32_vrsqrt_ukernel__scalar_recip_sqrt_u16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
