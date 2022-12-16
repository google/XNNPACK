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
#include <xnnpack/vlog.h>


void u32_vlog(
    benchmark::State& state,
    xnn_u32_vlog_ukernel_fn vlog,
    benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }
  const size_t num_elements = state.range(0);

  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> input(
      num_elements + XNN_EXTRA_BYTES / sizeof(uint32_t));
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> output(num_elements);
  std::iota(input.begin(), input.end(), 0);
  std::iota(output.begin(), output.end(), 0);

  for (auto _ : state) {
    vlog(num_elements, input.data(), 4, 16, output.data());
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = num_elements * (sizeof(uint32_t) + sizeof(uint16_t));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

BENCHMARK_CAPTURE(u32_vlog, scalar_x1,
                  xnn_u32_vlog_ukernel__scalar_x1)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint32_t, uint16_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(u32_vlog, scalar_x2,
                  xnn_u32_vlog_ukernel__scalar_x2)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint32_t, uint16_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(u32_vlog, scalar_x3,
                  xnn_u32_vlog_ukernel__scalar_x3)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint32_t, uint16_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(u32_vlog, scalar_x4,
                  xnn_u32_vlog_ukernel__scalar_x4)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint32_t, uint16_t>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
