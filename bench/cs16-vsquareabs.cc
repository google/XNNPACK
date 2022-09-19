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
#include <xnnpack/microparams-init.h>
#include <xnnpack/vsquareabs.h>


void cs16_vsquareabs(
    benchmark::State& state,
    xnn_cs16_vsquareabs_ukernel_function vsquareabs,
    benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }
  const size_t batch = state.range(0);

  std::vector<int16_t, AlignedAllocator<int16_t, 64>> input(
      batch * 2 + XNN_EXTRA_BYTES / sizeof(int16_t));
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> output(batch);
  std::iota(input.begin(), input.end(), 0);
  std::iota(output.begin(), output.end(), 0);

  for (auto _ : state) {
    vsquareabs(batch * sizeof(int16_t) * 2, input.data(), output.data());
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
  b->Args({117});
  b->Args({400});
  b->Args({1000});
  b->Args({10000});
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_neon_x4,
                    xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x4,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
  BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_neon_x8,
                    xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x8,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
  BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_neon_x12,
                    xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x12,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
  BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_neon_x16,
                    xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x16,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_scalar_x1,
                  xnn_cs16_vsquareabs_ukernel__scalar_x1)
  ->Apply(BenchmarkBatch)
  ->UseRealTime();
BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_scalar_x2,
                  xnn_cs16_vsquareabs_ukernel__scalar_x2)
  ->Apply(BenchmarkBatch)
  ->UseRealTime();
BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_scalar_x3,
                  xnn_cs16_vsquareabs_ukernel__scalar_x3)
  ->Apply(BenchmarkBatch)
  ->UseRealTime();
BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_scalar_x4,
                  xnn_cs16_vsquareabs_ukernel__scalar_x4)
  ->Apply(BenchmarkBatch)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
