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
#include <xnnpack/rmaxabs.h>


void s16_rmaxabs(
    benchmark::State& state,
    xnn_s16_rmaxabs_ukernel_fn rmaxabs,
    benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }
  const size_t channels = state.range(0);

  std::vector<int16_t, AlignedAllocator<int16_t, 64>> input(
      (channels) + XNN_EXTRA_BYTES / sizeof(int16_t));
  std::iota(input.begin(), input.end(), 0);

  uint16_t output = UINT16_C(0);
  for (auto _ : state) {
    rmaxabs(channels, input.data(), &output);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void BenchmarkBatch(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"batch"});
  b->Args({32});
  b->Args({64});
  b->Args({216});
  b->Args({400});
  b->Args({1000});
  b->Args({10000});
  b->Args({100000});
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(s16_rmaxabs, s16_neon_x8,
                    xnn_s16_rmaxabs_ukernel__neon_x8,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
  BENCHMARK_CAPTURE(s16_rmaxabs, s16_neon_x16,
                    xnn_s16_rmaxabs_ukernel__neon_x16,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
  BENCHMARK_CAPTURE(s16_rmaxabs, s16_neon_x24,
                    xnn_s16_rmaxabs_ukernel__neon_x24,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
  BENCHMARK_CAPTURE(s16_rmaxabs, s16_neon_x32,
                    xnn_s16_rmaxabs_ukernel__neon_x32,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

BENCHMARK_CAPTURE(s16_rmaxabs, s16_scalar_x1,
                  xnn_s16_rmaxabs_ukernel__scalar_x1)
  ->Apply(BenchmarkBatch)
  ->UseRealTime();
BENCHMARK_CAPTURE(s16_rmaxabs, s16_scalar_x2,
                  xnn_s16_rmaxabs_ukernel__scalar_x2)
  ->Apply(BenchmarkBatch)
  ->UseRealTime();
BENCHMARK_CAPTURE(s16_rmaxabs, s16_scalar_x3,
                  xnn_s16_rmaxabs_ukernel__scalar_x3)
  ->Apply(BenchmarkBatch)
  ->UseRealTime();
BENCHMARK_CAPTURE(s16_rmaxabs, s16_scalar_x4,
                  xnn_s16_rmaxabs_ukernel__scalar_x4)
  ->Apply(BenchmarkBatch)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
