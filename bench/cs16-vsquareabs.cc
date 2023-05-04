// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <complex>
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
    xnn_cs16_vsquareabs_ukernel_fn vsquareabs,
    benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if ((isa_check != nullptr) && !isa_check(state)) {
    return;
  }
  const size_t num_elements = state.range(0);

  std::vector<int16_t, AlignedAllocator<int16_t, 64>> input(
      num_elements * 2 + XNN_EXTRA_BYTES / sizeof(int16_t));
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> output(num_elements);
  std::iota(input.begin(), input.end(), 0);
  std::iota(output.begin(), output.end(), 0);

  for (auto _ : state) {
    vsquareabs(num_elements * sizeof(int16_t) * 2, input.data(), output.data());
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = num_elements * (sizeof(std::complex<int16_t>) + sizeof(uint32_t));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_neon_x4,
                    xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x4,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<std::complex<int16_t>, uint32_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_neon_x8,
                    xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x8,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<std::complex<int16_t>, uint32_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_neon_x12,
                    xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x12,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<std::complex<int16_t>, uint32_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_neon_x16,
                    xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x16,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<std::complex<int16_t>, uint32_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_HEXAGON
  BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_hexagon_x2,
                    xnn_cs16_vsquareabs_ukernel__hexagon_x2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<std::complex<int16_t>, uint32_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_hexagon_x4,
                    xnn_cs16_vsquareabs_ukernel__hexagon_x4)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<std::complex<int16_t>, uint32_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_hexagon_x6,
                    xnn_cs16_vsquareabs_ukernel__hexagon_x6)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<std::complex<int16_t>, uint32_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_hexagon_x8,
                    xnn_cs16_vsquareabs_ukernel__hexagon_x8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<std::complex<int16_t>, uint32_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_hexagon_x10,
                    xnn_cs16_vsquareabs_ukernel__hexagon_x10)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<std::complex<int16_t>, uint32_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_hexagon_x12,
                    xnn_cs16_vsquareabs_ukernel__hexagon_x12)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<std::complex<int16_t>, uint32_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_HEXAGON

BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_scalar_x1,
                  xnn_cs16_vsquareabs_ukernel__scalar_x1)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<std::complex<int16_t>, uint32_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_scalar_x2,
                  xnn_cs16_vsquareabs_ukernel__scalar_x2)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<std::complex<int16_t>, uint32_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_scalar_x3,
                  xnn_cs16_vsquareabs_ukernel__scalar_x3)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<std::complex<int16_t>, uint32_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(cs16_vsquareabs, cs16_scalar_x4,
                  xnn_cs16_vsquareabs_ukernel__scalar_x4)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<std::complex<int16_t>, uint32_t>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
