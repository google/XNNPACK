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
#include <xnnpack/fft.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>


void cs16_fftr(
    benchmark::State& state,
    xnn_cs16_fftr_ukernel_fn fftr,
    benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if ((isa_check != nullptr) && !isa_check(state)) {
    return;
  }
  const size_t samples = state.range(0);

  assert(samples % 2 == 0);
  const size_t sample_size = samples * 2 + 2;

  std::vector<int16_t, AlignedAllocator<int16_t, 64>> data(sample_size + XNN_EXTRA_BYTES / sizeof(int16_t));
  std::vector<int16_t, AlignedAllocator<int16_t, 64>> twiddle(samples);

  std::iota(data.begin(), data.end(), 0);
  std::iota(twiddle.begin(), twiddle.end(), 2);

  for (auto _ : state) {
    fftr(samples, data.data(), twiddle.data());
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void BenchmarkKernelSize(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"samples"});
  b->Args({256});
  b->Args({1024});
}
#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
BENCHMARK_CAPTURE(cs16_fftr, cs16_aarch32_neon_x1, xnn_cs16_fftr_ukernel__asm_aarch32_neon_x1)->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(cs16_fftr, cs16_aarch32_neon_x4, xnn_cs16_fftr_ukernel__asm_aarch32_neon_x4)->Apply(BenchmarkKernelSize)->UseRealTime();
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
BENCHMARK_CAPTURE(cs16_fftr, cs16_neon_x4, xnn_cs16_fftr_ukernel__neon_x4)->Apply(BenchmarkKernelSize)->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

BENCHMARK_CAPTURE(cs16_fftr, cs16_scalar_x1, xnn_cs16_fftr_ukernel__scalar_x1)->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(cs16_fftr, cs16_scalar_x2, xnn_cs16_fftr_ukernel__scalar_x2)->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(cs16_fftr, cs16_scalar_x4, xnn_cs16_fftr_ukernel__scalar_x4)->Apply(BenchmarkKernelSize)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
