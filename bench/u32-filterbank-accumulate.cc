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
    xnn_u32_filterbank_accumulate_ukernel_fn filterbank_accumulate,
    benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }
  const size_t rows = state.range(0);
  const size_t batch = state.range(1);
  const size_t input_size = (rows + 1) * batch;

  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> input(input_size);
  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> weight_widths(rows + 1);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> weights(input_size * 2);
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> output(rows);
  std::iota(input.begin(), input.end(), 0);
  std::fill(weight_widths.begin(), weight_widths.end(), static_cast<uint8_t>(batch));
  std::iota(weights.begin(), weights.end(), 0);
  std::iota(output.begin(), output.end(), 0);

  for (auto _ : state) {
    filterbank_accumulate(rows, input.data(), weight_widths.data(), weights.data(), output.data());
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
  b->Args({5, 1});
  b->Args({10, 2});
  b->Args({7, 3});
  b->Args({5, 4});
  b->Args({5, 5});
  b->Args({3, 6});
  b->Args({4, 7});
  b->Args({2, 8});
  b->Args({2, 9});
  b->Args({2, 10});
  b->Args({3, 11});
  b->Args({1, 13});
}

#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
BENCHMARK_CAPTURE(filterbank_accumulate, u32_aarch32_arm_x1,   xnn_u32_filterbank_accumulate_ukernel__asm_aarch32_arm_x1)->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(filterbank_accumulate, u32_aarch32_neon_x1,  xnn_u32_filterbank_accumulate_ukernel__asm_aarch32_neon_x1, benchmark::utils::CheckNEON)->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(filterbank_accumulate, u32_aarch32_neon_x2,  xnn_u32_filterbank_accumulate_ukernel__asm_aarch32_neon_x2, benchmark::utils::CheckNEON)->Apply(BenchmarkKernelSize)->UseRealTime();
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
BENCHMARK_CAPTURE(filterbank_accumulate, u32_neon_x1,  xnn_u32_filterbank_accumulate_ukernel__neon_x1,  benchmark::utils::CheckNEON)->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(filterbank_accumulate, u32_neon_x2,  xnn_u32_filterbank_accumulate_ukernel__neon_x2,  benchmark::utils::CheckNEON)->Apply(BenchmarkKernelSize)->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

BENCHMARK_CAPTURE(filterbank_accumulate, u32_scalar_x1, xnn_u32_filterbank_accumulate_ukernel__scalar_x1)->Apply(BenchmarkKernelSize)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
