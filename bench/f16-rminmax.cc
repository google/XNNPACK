// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include <fp16/fp16.h>
#include "bench/utils.h"

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/reduce.h>


static void f16_rminmax(
  benchmark::State& state,
  xnn_f16_reduce_ukernel_fn rminmax,
  xnn_init_f16_default_params_fn init_params = nullptr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::ref(rng));
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> input(elements);
  std::generate(input.begin(), input.end(), std::ref(f16rng));

  xnn_f16_default_params params;
  if (init_params != nullptr) {
    init_params(&params);
  }

  uint16_t output[2] = {UINT16_C(0x7E00) /* NaN */, UINT16_C(0x7E00) /* NaN */};
  for (auto _ : state) {
    rminmax(elements * sizeof(uint16_t), input.data(), output, &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = elements * sizeof(uint16_t);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f16_rminmax, neonfp16arith_u8,
                    xnn_f16_rminmax_ukernel__neonfp16arith_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::ReductionParameters<uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_rminmax, neonfp16arith_u16_acc2,
                    xnn_f16_rminmax_ukernel__neonfp16arith_u16_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::ReductionParameters<uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_rminmax, neonfp16arith_u24_acc3,
                    xnn_f16_rminmax_ukernel__neonfp16arith_u24_acc3,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::ReductionParameters<uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_rminmax, neonfp16arith_u32_acc2,
                    xnn_f16_rminmax_ukernel__neonfp16arith_u32_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::ReductionParameters<uint16_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_rminmax, neonfp16arith_u32_acc4,
                    xnn_f16_rminmax_ukernel__neonfp16arith_u32_acc4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Apply(benchmark::utils::ReductionParameters<uint16_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

BENCHMARK_CAPTURE(f16_rminmax, scalar_u1,
                  xnn_f16_rminmax_ukernel__scalar_u1)
  ->Apply(benchmark::utils::ReductionParameters<uint16_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f16_rminmax, scalar_u2_acc2,
                  xnn_f16_rminmax_ukernel__scalar_u2_acc2)
  ->Apply(benchmark::utils::ReductionParameters<uint16_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f16_rminmax, scalar_u3_acc3,
                  xnn_f16_rminmax_ukernel__scalar_u3_acc3)
  ->Apply(benchmark::utils::ReductionParameters<uint16_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f16_rminmax, scalar_u4_acc2,
                  xnn_f16_rminmax_ukernel__scalar_u4_acc2)
  ->Apply(benchmark::utils::ReductionParameters<uint16_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f16_rminmax, scalar_u4_acc4,
                  xnn_f16_rminmax_ukernel__scalar_u4_acc4)
  ->Apply(benchmark::utils::ReductionParameters<uint16_t>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
