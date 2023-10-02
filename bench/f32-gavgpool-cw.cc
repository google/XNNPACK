// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include "bench/utils.h"
#include <benchmark/benchmark.h>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/gavgpool.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>


void f32_gavgpool_cw(
    benchmark::State& state,
    xnn_f32_gavgpool_cw_ukernel_fn gavgpool_cw,
    xnn_init_f32_gavgpool_params_fn init_params,
    benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }
  const size_t channels = state.range(0);
  const size_t elements = state.range(1);

  std::vector<float, AlignedAllocator<float, 64>> input(elements * channels + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> output(channels);
  std::iota(input.begin(), input.end(), 0.0f);

  // Prepare parameters.
  union xnn_f32_gavgpool_params params;
  init_params(&params,
    1.0f /* scale */, -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity(), elements);

  for (auto _ : state) {
    gavgpool_cw(elements, channels, input.data(), output.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void BenchmarkBatch(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"channels", "elements"});
  b->Args({1, 1024});
  b->Args({2, 1024});
  b->Args({4, 1024});
  b->Args({6, 1024});
  b->Args({8, 1024});
  b->Args({16, 1024});
  b->Args({1024, 1024});
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_gavgpool_cw, f32_neon_u4,
                    xnn_f32_gavgpool_cw_ukernel__neon_u4,
                    xnn_init_f32_gavgpool_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_gavgpool_cw, f32_sse_u4,
                    xnn_f32_gavgpool_cw_ukernel__sse_u4,
                    xnn_init_f32_gavgpool_sse_params)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD
  BENCHMARK_CAPTURE(f32_gavgpool_cw, f32_wasmsimd_arm_u4,
                    xnn_f32_gavgpool_cw_ukernel__wasmsimd_arm_u4,
                    xnn_init_f32_gavgpool_scalar_params)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_gavgpool_cw, f32_wasmsimd_x86_u4,
                    xnn_f32_gavgpool_cw_ukernel__wasmsimd_x86_u4,
                    xnn_init_f32_gavgpool_scalar_params)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD

BENCHMARK_CAPTURE(f32_gavgpool_cw, f32_scalar_u1,
                  xnn_f32_gavgpool_cw_ukernel__scalar_u1,
                  xnn_init_f32_gavgpool_scalar_params)
  ->Apply(BenchmarkBatch)
  ->UseRealTime();


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
