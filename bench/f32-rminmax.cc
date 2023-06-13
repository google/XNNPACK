// Copyright 2023 Google LLC
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
#include <xnnpack/reduce.h>


static void f32_rminmax(
  benchmark::State& state,
  xnn_f32_reduce_ukernel_fn rminmax,
  xnn_init_f32_default_params_fn init_params = nullptr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::ref(rng));

  std::vector<float, AlignedAllocator<float, 64>> input(elements);
  std::generate(input.begin(), input.end(), std::ref(f32rng));

  xnn_f32_default_params params;
  if (init_params != nullptr) {
    init_params(&params);
  }

  float output[2] = {std::nanf(""), std::nanf("")};
  for (auto _ : state) {
    rminmax(elements * sizeof(float), input.data(), output, &params);
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

BENCHMARK_CAPTURE(f32_rminmax, scalar_x1,
                  xnn_f32_rminmax_ukernel__scalar_x1)
  ->Apply(benchmark::utils::ReductionParameters<float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_rminmax, scalar_x2_acc2,
                  xnn_f32_rminmax_ukernel__scalar_x2_acc2)
  ->Apply(benchmark::utils::ReductionParameters<float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_rminmax, scalar_x3_acc3,
                  xnn_f32_rminmax_ukernel__scalar_x3_acc3)
  ->Apply(benchmark::utils::ReductionParameters<float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_rminmax, scalar_x4_acc2,
                  xnn_f32_rminmax_ukernel__scalar_x4_acc2)
  ->Apply(benchmark::utils::ReductionParameters<float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_rminmax, scalar_x4_acc4,
                  xnn_f32_rminmax_ukernel__scalar_x4_acc4)
  ->Apply(benchmark::utils::ReductionParameters<float>)
  ->UseRealTime();

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rminmax, sse_x4,
                    xnn_f32_rminmax_ukernel__sse_x4)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rminmax, sse_x8_acc2,
                    xnn_f32_rminmax_ukernel__sse_x8_acc2)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rminmax, sse_x12_acc3,
                    xnn_f32_rminmax_ukernel__sse_x12_acc3)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rminmax, sse_x16_acc2,
                    xnn_f32_rminmax_ukernel__sse_x16_acc2)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rminmax, sse_x16_acc4,
                    xnn_f32_rminmax_ukernel__sse_x16_acc4)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_rminmax, neon_x4,
                    xnn_f32_rminmax_ukernel__neon_x4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rminmax, neon_x8_acc2,
                    xnn_f32_rminmax_ukernel__neon_x8_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rminmax, neon_x12_acc3,
                    xnn_f32_rminmax_ukernel__neon_x12_acc3,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rminmax, neon_x16_acc2,
                    xnn_f32_rminmax_ukernel__neon_x16_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rminmax, neon_x16_acc4,
                    xnn_f32_rminmax_ukernel__neon_x16_acc4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif

