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


void vlog(
    benchmark::State& state,
    xnn_u32_vlog_ukernel_function vlog,
    benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }
  const size_t batch = state.range(0);

  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> input(
      batch + XNN_EXTRA_BYTES / sizeof(uint32_t));
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> output(batch);
  std::iota(input.begin(), input.end(), 0);
  std::iota(output.begin(), output.end(), 0);

  for (auto _ : state) {
    vlog(batch, input.data(), 4, 16, output.data());
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void BenchmarkKernelSize(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"batch"});
  b->Args({32});
  b->Args({64});
  b->Args({117});
  b->Args({216});
  b->Args({400});
  b->Args({1000});
  b->Args({10000});
}


BENCHMARK_CAPTURE(vlog, u32_scalar_x1, xnn_u32_vlog_ukernel__scalar_x1)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(vlog, u32_scalar_x2, xnn_u32_vlog_ukernel__scalar_x2)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(vlog, u32_scalar_x3, xnn_u32_vlog_ukernel__scalar_x3)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(vlog, u32_scalar_x4, xnn_u32_vlog_ukernel__scalar_x4)
    ->Apply(BenchmarkKernelSize)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
