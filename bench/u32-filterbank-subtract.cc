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
#include <xnnpack/filterbank.h>
#include <xnnpack/microfnptr.h>


void filterbank_subtract(
    benchmark::State& state,
    xnn_u32_filterbank_subtract_ukernel_fn filterbank_subtract,
    benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }
  const size_t batch = state.range(0);

  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> input(batch + XNN_EXTRA_BYTES / sizeof(uint32_t));
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> noise_estimate(batch + XNN_EXTRA_BYTES / sizeof(uint32_t));
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> output(batch);
  std::iota(input.begin(), input.end(), 0);
  std::iota(noise_estimate.begin(), noise_estimate.end(), 1);
  std::iota(output.begin(), output.end(), 0);

  for (auto _ : state) {
    filterbank_subtract(batch, input.data(),
        655, 655, 15729, 15729, 819, 0, 14,
        noise_estimate.data(), output.data());
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void BenchmarkKernelSize(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"batch"});
  b->Args({48});
  b->Args({480});
  b->Args({1000});
  b->Args({10000});
  b->Args({48000});
}

BENCHMARK_CAPTURE(filterbank_subtract, u32_scalar_x1, xnn_u32_filterbank_subtract_ukernel__scalar_x2)->Apply(BenchmarkKernelSize)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
