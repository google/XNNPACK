// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "bench/utils.h"

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/reduce.h>


static void f32_rsum(
  benchmark::State& state,
  xnn_f32_rsum_ukernel_fn rsum,
  xnn_init_f32_scale_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::ref(rng));

  std::vector<float, AlignedAllocator<float, 64>> input(elements);
  std::generate(input.begin(), input.end(), std::ref(f32rng));

  xnn_f32_scale_params params;
  init_params(&params, /*scale=*/0.1f);

  float output = std::nanf("");
  for (auto _ : state) {
    rsum(elements * sizeof(float), input.data(), &output, &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = elements * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rsum, avx512f_u16,
                    xnn_f32_rsum_ukernel__avx512f_u16,
                    xnn_init_f32_scale_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rsum, avx512f_u32_acc2,
                    xnn_f32_rsum_ukernel__avx512f_u32_acc2,
                    xnn_init_f32_scale_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rsum, avx512f_u48_acc3,
                    xnn_f32_rsum_ukernel__avx512f_u48_acc3,
                    xnn_init_f32_scale_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rsum, avx512f_u64_acc2,
                    xnn_f32_rsum_ukernel__avx512f_u64_acc2,
                    xnn_init_f32_scale_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rsum, avx512f_u64_acc4,
                    xnn_f32_rsum_ukernel__avx512f_u64_acc4,
                    xnn_init_f32_scale_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_rsum, avx_u8,
                    xnn_f32_rsum_ukernel__avx_u8,
                    xnn_init_f32_scale_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rsum, avx_u16_acc2,
                    xnn_f32_rsum_ukernel__avx_u16_acc2,
                    xnn_init_f32_scale_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rsum, avx_u24_acc3,
                    xnn_f32_rsum_ukernel__avx_u24_acc3,
                    xnn_init_f32_scale_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rsum, avx_u32_acc2,
                    xnn_f32_rsum_ukernel__avx_u32_acc2,
                    xnn_init_f32_scale_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rsum, avx_u32_acc4,
                    xnn_f32_rsum_ukernel__avx_u32_acc4,
                    xnn_init_f32_scale_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_rsum, sse_u4,
                    xnn_f32_rsum_ukernel__sse_u4,
                    xnn_init_f32_scale_scalar_params)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rsum, sse_u8_acc2,
                    xnn_f32_rsum_ukernel__sse_u8_acc2,
                    xnn_init_f32_scale_scalar_params)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rsum, sse_u12_acc3,
                    xnn_f32_rsum_ukernel__sse_u12_acc3,
                    xnn_init_f32_scale_scalar_params)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rsum, sse_u16_acc2,
                    xnn_f32_rsum_ukernel__sse_u16_acc2,
                    xnn_init_f32_scale_scalar_params)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rsum, sse_u16_acc4,
                    xnn_f32_rsum_ukernel__sse_u16_acc4,
                    xnn_init_f32_scale_scalar_params)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_rsum, neon_u4,
                    xnn_f32_rsum_ukernel__neon_u4,
                    xnn_init_f32_scale_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rsum, neon_u8_acc2,
                    xnn_f32_rsum_ukernel__neon_u8_acc2,
                    xnn_init_f32_scale_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rsum, neon_u12_acc3,
                    xnn_f32_rsum_ukernel__neon_u12_acc3,
                    xnn_init_f32_scale_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rsum, neon_u16_acc2,
                    xnn_f32_rsum_ukernel__neon_u16_acc2,
                    xnn_init_f32_scale_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rsum, neon_u16_acc4,
                    xnn_f32_rsum_ukernel__neon_u16_acc4,
                    xnn_init_f32_scale_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_rsum, wasmsimd_u4,
                    xnn_f32_rsum_ukernel__wasmsimd_u4,
                    xnn_init_f32_scale_scalar_params)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rsum, wasmsimd_u8_acc2,
                    xnn_f32_rsum_ukernel__wasmsimd_u8_acc2,
                    xnn_init_f32_scale_scalar_params)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rsum, wasmsimd_u12_acc3,
                    xnn_f32_rsum_ukernel__wasmsimd_u12_acc3,
                    xnn_init_f32_scale_scalar_params)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rsum, wasmsimd_u16_acc2,
                    xnn_f32_rsum_ukernel__wasmsimd_u16_acc2,
                    xnn_init_f32_scale_scalar_params)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rsum, wasmsimd_u16_acc4,
                    xnn_f32_rsum_ukernel__wasmsimd_u16_acc4,
                    xnn_init_f32_scale_scalar_params)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(f32_rsum, scalar_u1,
                  xnn_f32_rsum_ukernel__scalar_u1,
                  xnn_init_f32_scale_scalar_params)
  ->Apply(benchmark::utils::ReductionParameters<float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_rsum, scalar_u2_acc2,
                  xnn_f32_rsum_ukernel__scalar_u2_acc2,
                  xnn_init_f32_scale_scalar_params)
  ->Apply(benchmark::utils::ReductionParameters<float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_rsum, scalar_u3_acc3,
                  xnn_f32_rsum_ukernel__scalar_u3_acc3,
                  xnn_init_f32_scale_scalar_params)
  ->Apply(benchmark::utils::ReductionParameters<float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_rsum, scalar_u4_acc2,
                  xnn_f32_rsum_ukernel__scalar_u4_acc2,
                  xnn_init_f32_scale_scalar_params)
  ->Apply(benchmark::utils::ReductionParameters<float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_rsum, scalar_u4_acc4,
                  xnn_f32_rsum_ukernel__scalar_u4_acc4,
                  xnn_init_f32_scale_scalar_params)
  ->Apply(benchmark::utils::ReductionParameters<float>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
