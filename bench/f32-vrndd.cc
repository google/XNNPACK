// Copyright 2023 Google LLC
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
#include <xnnpack/vunary.h>


static void f32_vrndd(
  benchmark::State& state,
  xnn_f32_vround_ukernel_fn vrndd,
  xnn_init_f32_rnd_params_fn init_params = nullptr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t elements = state.range(0);
  std::vector<float, AlignedAllocator<float, 64>> input(elements + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float, AlignedAllocator<float, 64>> output(elements);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f), std::ref(rng));
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::fill(output.begin(), output.end(), std::nanf(""));

  union xnn_f32_rnd_params params;
  if (init_params) {
    init_params(&params);
  }
  for (auto _ : state) {
    vrndd(elements * sizeof(float), input.data(), output.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * elements * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_vrndd, neonv8_u4,
                    xnn_f32_vrndd_ukernel__neonv8_u4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONV8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrndd, neonv8_u8,
                    xnn_f32_vrndd_ukernel__neonv8_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEONV8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrndd, neon_u4,
                    xnn_f32_vrndd_ukernel__neon_u4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrndd, neon_u8,
                    xnn_f32_vrndd_ukernel__neon_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_vrndd, avx512f_u16,
                    xnn_f32_vrndd_ukernel__avx512f_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrndd, avx512f_u32,
                    xnn_f32_vrndd_ukernel__avx512f_u32,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrndd, avx_u8,
                    xnn_f32_vrndd_ukernel__avx_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrndd, avx_u16,
                    xnn_f32_vrndd_ukernel__avx_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrndd, sse41_u4,
                    xnn_f32_vrndd_ukernel__sse41_u4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrndd, sse41_u8,
                    xnn_f32_vrndd_ukernel__sse41_u8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrndd, sse2_u4,
                    xnn_f32_vrndd_ukernel__sse2_u4,
                    xnn_init_f32_rnd_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrndd, sse2_u8,
                    xnn_f32_vrndd_ukernel__sse2_u8,
                    xnn_init_f32_rnd_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_vrndd, wasmsimd_u4,
                    xnn_f32_vrndd_ukernel__wasmsimd_u4)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrndd, wasmsimd_u8,
                    xnn_f32_vrndd_ukernel__wasmsimd_u8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(f32_vrndd, scalar_libm_u1,
                  xnn_f32_vrndd_ukernel__scalar_libm_u1)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vrndd, scalar_libm_u2,
                  xnn_f32_vrndd_ukernel__scalar_libm_u2)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vrndd, scalar_libm_u4,
                  xnn_f32_vrndd_ukernel__scalar_libm_u4)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
