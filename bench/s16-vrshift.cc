// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/params.h>
#include <xnnpack/vrshift.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include "bench/utils.h"
#include <benchmark/benchmark.h>

void vrshift(
    benchmark::State& state,
    xnn_s16_vrshift_ukernel_function vrshift,
    benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }
  const size_t channels = state.range(0);

  std::vector<int16_t, AlignedAllocator<int16_t, 64>> input(
      channels + XNN_EXTRA_BYTES / sizeof(int16_t));
  std::vector<int16_t, AlignedAllocator<int16_t, 64>> output(channels);
  std::iota(input.begin(), input.end(), 0);
  std::iota(output.begin(), output.end(), 1);

  for (auto _ : state) {
    vrshift(channels, input.data(), uint32_t(4), output.data());
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void BenchmarkKernelSize(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"channels"});
  b->Args({32});
  b->Args({64});
  b->Args({117});
  b->Args({400});
  b->Args({1000});
  b->Args({10000});
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
BENCHMARK_CAPTURE(vrshift, s16_neon_x8, xnn_s16_vrshift_ukernel__neon_x8)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(vrshift, s16_neon_x16, xnn_s16_vrshift_ukernel__neon_x16)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(vrshift, s16_neon_x24, xnn_s16_vrshift_ukernel__neon_x24)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(vrshift, s16_neon_x32, xnn_s16_vrshift_ukernel__neon_x32)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

BENCHMARK_CAPTURE(vrshift, s16_scalar_x1, xnn_s16_vrshift_ukernel__scalar_x1)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(vrshift, s16_scalar_x2, xnn_s16_vrshift_ukernel__scalar_x2)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(vrshift, s16_scalar_x3, xnn_s16_vrshift_ukernel__scalar_x3)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(vrshift, s16_scalar_x4, xnn_s16_vrshift_ukernel__scalar_x4)
    ->Apply(BenchmarkKernelSize)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
