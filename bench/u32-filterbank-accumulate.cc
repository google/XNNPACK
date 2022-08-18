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


void filterbank_accumulate(
    benchmark::State& state,
    xnn_u32_filterbank_accumulate_ukernel_function filterbank_accumulate,
    benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }
  const size_t rows = state.range(0);
  const size_t batch = state.range(1);

  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> input(batch + XNN_EXTRA_BYTES / sizeof(uint32_t));
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> input_offset(rows);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> weight_offset(rows);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> weight_widths(rows);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> weights(batch + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> unweights(batch + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> output(rows);
  std::iota(input.begin(), input.end(), 0);
  std::fill(input_offset.begin(), input_offset.end(), 0);
  std::fill(weight_offset.begin(), weight_offset.end(), 0);
  std::fill(weight_widths.begin(), weight_widths.end(), rows);
  std::iota(weights.begin(), weights.end(), 0);
  std::iota(unweights.begin(), unweights.end(), 0);
  std::iota(output.begin(), output.end(), 0);

  for (auto _ : state) {
    filterbank_accumulate(rows, batch, input.data(), input_offset.data(), weight_offset.data(), weight_widths.data(), weights.data(), unweights.data(), output.data());
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void BenchmarkKernelSize(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"rows", "batch"});
  b->Args({1, 237});
  b->Args({10, 237});
  b->Args({100, 237});
  b->Args({1000, 237});
  b->Args({1000, 1000});
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
BENCHMARK_CAPTURE(filterbank_accumulate, u32_neon_x1,  xnn_u32_filterbank_accumulate_ukernel__neon_x1,  benchmark::utils::CheckNEON)->Apply(BenchmarkKernelSize)->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

BENCHMARK_CAPTURE(filterbank_accumulate, u32_scalar_x1, xnn_u32_filterbank_accumulate_ukernel__scalar_x1)->Apply(BenchmarkKernelSize)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
