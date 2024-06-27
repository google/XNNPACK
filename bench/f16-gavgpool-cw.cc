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

#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/gavgpool.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"


void f16_gavgpool_cw(
    benchmark::State& state,
    xnn_f16_gavgpool_cw_ukernel_fn gavgpool_cw,
    xnn_init_f16_gavgpool_neon_params_fn init_params,
    benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }
  const size_t channels = state.range(0);
  const size_t elements = state.range(1);

  std::vector<int16_t, AlignedAllocator<int16_t, 64>> input(elements * channels + XNN_EXTRA_BYTES / sizeof(int16_t));
  std::vector<int16_t> output(channels);
  std::iota(input.begin(), input.end(), 0);

  // Prepare parameters.
  union xnn_f16_gavgpool_params params;
  init_params(&params,
    UINT16_C(0x3C00) /* scale */, UINT16_C(0xFC00) /* -inf */, UINT16_C(0x7C00) /* inf */, elements);

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

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(f16_gavgpool_cw, f16_neon_u8,
                    xnn_f16_gavgpool_cw_ukernel__neonfp16arith_u8,
                    xnn_init_f16_gavgpool_neonfp16arith_params,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(BenchmarkBatch)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
