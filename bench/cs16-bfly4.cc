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


void cs16_bfly4(
    benchmark::State& state,
    xnn_cs16_bfly4_ukernel_fn bfly4,
    benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if ((isa_check != nullptr) && !isa_check(state)) {
    return;
  }
  const size_t fft_size = state.range(0);
  const size_t batch = state.range(1);
  const size_t samples = state.range(2);
  const size_t stride = state.range(3);

  assert(fft_size == samples * stride * 4);  // 4 for bfly4.

  std::vector<int16_t, AlignedAllocator<int16_t, 64>> output(fft_size * 2);
  std::vector<int16_t, AlignedAllocator<int16_t, 64>> twiddle(fft_size * 3 / 4 * 2);

  std::iota(output.begin(), output.end(), 0);
  std::iota(twiddle.begin(), twiddle.end(), 0);

  for (auto _ : state) {
    bfly4(batch, samples * sizeof(int16_t) * 2, output.data(), twiddle.data(), stride * sizeof(int16_t) * 2);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void BenchmarkKernelSize(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"fft_size", "batch", "samples", "stride"});
  b->Args({256, 1, 1, 64});
  b->Args({256, 4, 1, 64});
  b->Args({256, 1, 4, 16});
  b->Args({256, 4, 4, 16});
  b->Args({256, 1, 16, 4});
  b->Args({256, 4, 16, 4});
  b->Args({256, 1, 64, 1});
}

static void BenchmarkSamples1KernelSize(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"fft_size", "batch", "samples", "stride"});
  b->Args({256, 1, 1, 64});
  b->Args({256, 4, 1, 64});
  b->Args({256, 16, 1, 64});
  b->Args({256, 64, 1, 64});
}
static void BenchmarkSamples4KernelSize(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"fft_size", "batch", "samples", "stride"});
  b->Args({256, 1, 4, 16});
  b->Args({256, 4, 4, 16});
  b->Args({256, 16, 4, 16});
}

#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
BENCHMARK_CAPTURE(cs16_bfly4, samples1__asm_aarch32_neon_x1, xnn_cs16_bfly4_samples1_ukernel__asm_aarch32_neon_x1)
  ->Apply(BenchmarkSamples1KernelSize)->UseRealTime();
BENCHMARK_CAPTURE(cs16_bfly4, samples1__asm_aarch32_neon_x2, xnn_cs16_bfly4_samples1_ukernel__asm_aarch32_neon_x2)
  ->Apply(BenchmarkSamples1KernelSize)->UseRealTime();
BENCHMARK_CAPTURE(cs16_bfly4, samples1__asm_aarch32_neon_x4, xnn_cs16_bfly4_samples1_ukernel__asm_aarch32_neon_x4)
  ->Apply(BenchmarkSamples1KernelSize)->UseRealTime();
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
BENCHMARK_CAPTURE(cs16_bfly4, samples1__neon, xnn_cs16_bfly4_samples1_ukernel__neon)
  ->Apply(BenchmarkSamples1KernelSize)->UseRealTime();
BENCHMARK_CAPTURE(cs16_bfly4, samples4__neon, xnn_cs16_bfly4_samples4_ukernel__neon)
  ->Apply(BenchmarkSamples4KernelSize)->UseRealTime();
BENCHMARK_CAPTURE(cs16_bfly4, neon_x1, xnn_cs16_bfly4_ukernel__neon_x1)
  ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(cs16_bfly4, neon_x4, xnn_cs16_bfly4_ukernel__neon_x4)
  ->Apply(BenchmarkKernelSize)->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

BENCHMARK_CAPTURE(cs16_bfly4, samples1__scalar, xnn_cs16_bfly4_samples1_ukernel__scalar)
  ->Apply(BenchmarkSamples1KernelSize)->UseRealTime();
BENCHMARK_CAPTURE(cs16_bfly4, samples4__scalar, xnn_cs16_bfly4_samples4_ukernel__scalar)
  ->Apply(BenchmarkSamples4KernelSize)->UseRealTime();
BENCHMARK_CAPTURE(cs16_bfly4, scalar_x1, xnn_cs16_bfly4_ukernel__scalar_x1)
  ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(cs16_bfly4, scalar_x2, xnn_cs16_bfly4_ukernel__scalar_x2)
  ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(cs16_bfly4, scalar_x4, xnn_cs16_bfly4_ukernel__scalar_x4)
  ->Apply(BenchmarkKernelSize)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
