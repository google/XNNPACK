// Copyright 2021 Google LLC
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

#include <fp16/fp16.h>
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/common.h>
#include <xnnpack/params.h>
#include <xnnpack/params-init.h>
#include <xnnpack/vcvt.h>


static void f32_qu8_vcvt(
  benchmark::State& state,
  xnn_f32_qu8_vcvt_ukernel_function cvt,
  xnn_init_f32_qu8_cvt_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f), std::ref(rng));

  std::vector<float, AlignedAllocator<float, 64>> x(num_elements + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), std::ref(f32rng));
  std::fill(y.begin(), y.end(), UINT8_C(0xA5));

  xnn_f32_qu8_cvt_params params;
  init_params(&params,
    25.0f /* scale */,
    127 /* output zero point */,
    std::numeric_limits<uint8_t>::min() + 1 /* output min */,
    std::numeric_limits<uint8_t>::max() - 1 /* output max */);
  for (auto _ : state) {
    cvt(num_elements * sizeof(uint8_t), x.data(), y.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = num_elements * (sizeof(int8_t) + sizeof(float));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neonv8_x8,
                    xnn_f32_qu8_vcvt_ukernel__neonv8_x8,
                    xnn_init_f32_qu8_cvt_neonv8_params,
                    benchmark::utils::CheckNEONV8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neonv8_x16,
                    xnn_f32_qu8_vcvt_ukernel__neonv8_x16,
                    xnn_init_f32_qu8_cvt_neonv8_params,
                    benchmark::utils::CheckNEONV8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neonv8_x24,
                    xnn_f32_qu8_vcvt_ukernel__neonv8_x24,
                    xnn_init_f32_qu8_cvt_neonv8_params,
                    benchmark::utils::CheckNEONV8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neonv8_x32,
                    xnn_f32_qu8_vcvt_ukernel__neonv8_x32,
                    xnn_init_f32_qu8_cvt_neonv8_params,
                    benchmark::utils::CheckNEONV8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_qu8_vcvt, neon_x8,
                    xnn_f32_qu8_vcvt_ukernel__neon_x8,
                    xnn_init_f32_qu8_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neon_x16,
                    xnn_f32_qu8_vcvt_ukernel__neon_x16,
                    xnn_init_f32_qu8_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neon_x24,
                    xnn_f32_qu8_vcvt_ukernel__neon_x24,
                    xnn_init_f32_qu8_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neon_x32,
                    xnn_f32_qu8_vcvt_ukernel__neon_x32,
                    xnn_init_f32_qu8_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx_x8,
                    xnn_f32_qu8_vcvt_ukernel__avx_x8,
                    xnn_init_f32_qu8_cvt_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx_x16,
                    xnn_f32_qu8_vcvt_ukernel__avx_x16,
                    xnn_init_f32_qu8_cvt_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx_x24,
                    xnn_f32_qu8_vcvt_ukernel__avx_x24,
                    xnn_init_f32_qu8_cvt_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx_x32,
                    xnn_f32_qu8_vcvt_ukernel__avx_x32,
                    xnn_init_f32_qu8_cvt_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_qu8_vcvt, sse2_x8,
                    xnn_f32_qu8_vcvt_ukernel__sse2_x8,
                    xnn_init_f32_qu8_cvt_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, sse2_x16,
                    xnn_f32_qu8_vcvt_ukernel__sse2_x16,
                    xnn_init_f32_qu8_cvt_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, sse2_x24,
                    xnn_f32_qu8_vcvt_ukernel__sse2_x24,
                    xnn_init_f32_qu8_cvt_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, sse2_x32,
                    xnn_f32_qu8_vcvt_ukernel__sse2_x32,
                    xnn_init_f32_qu8_cvt_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_cvt_x8,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_x8,
                    xnn_init_f32_qu8_cvt_wasmsimd_cvt_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_cvt_x16,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_x16,
                    xnn_init_f32_qu8_cvt_wasmsimd_cvt_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_cvt_x24,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_x24,
                    xnn_init_f32_qu8_cvt_wasmsimd_cvt_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_cvt_x32,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_x32,
                    xnn_init_f32_qu8_cvt_wasmsimd_cvt_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_magic_x8,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_x8,
                    xnn_init_f32_qu8_cvt_wasmsimd_magic_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_magic_x16,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_x16,
                    xnn_init_f32_qu8_cvt_wasmsimd_magic_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_magic_x24,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_x24,
                    xnn_init_f32_qu8_cvt_wasmsimd_magic_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_magic_x32,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_x32,
                    xnn_init_f32_qu8_cvt_wasmsimd_magic_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasm_magic_fminmax_x1,
                    xnn_f32_qu8_vcvt_ukernel__wasm_magic_fminmax_x1,
                    xnn_init_f32_qu8_cvt_scalar_magic_fminmax_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasm_magic_fminmax_x2,
                    xnn_f32_qu8_vcvt_ukernel__wasm_magic_fminmax_x2,
                    xnn_init_f32_qu8_cvt_scalar_magic_fminmax_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasm_magic_fminmax_x3,
                    xnn_f32_qu8_vcvt_ukernel__wasm_magic_fminmax_x3,
                    xnn_init_f32_qu8_cvt_scalar_magic_fminmax_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasm_magic_fminmax_x4,
                    xnn_f32_qu8_vcvt_ukernel__wasm_magic_fminmax_x4,
                    xnn_init_f32_qu8_cvt_scalar_magic_fminmax_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD

BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_magic_fminmax_x1,
                  xnn_f32_qu8_vcvt_ukernel__scalar_magic_fminmax_x1,
                  xnn_init_f32_qu8_cvt_scalar_magic_fminmax_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_magic_fminmax_x2,
                  xnn_f32_qu8_vcvt_ukernel__scalar_magic_fminmax_x2,
                  xnn_init_f32_qu8_cvt_scalar_magic_fminmax_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_magic_fminmax_x3,
                  xnn_f32_qu8_vcvt_ukernel__scalar_magic_fminmax_x3,
                  xnn_init_f32_qu8_cvt_scalar_magic_fminmax_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_magic_fminmax_x4,
                  xnn_f32_qu8_vcvt_ukernel__scalar_magic_fminmax_x4,
                  xnn_init_f32_qu8_cvt_scalar_magic_fminmax_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_magic_iminmax_x1,
                  xnn_f32_qu8_vcvt_ukernel__scalar_magic_iminmax_x1,
                  xnn_init_f32_qu8_cvt_scalar_magic_iminmax_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_magic_iminmax_x2,
                  xnn_f32_qu8_vcvt_ukernel__scalar_magic_iminmax_x2,
                  xnn_init_f32_qu8_cvt_scalar_magic_iminmax_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_magic_iminmax_x3,
                  xnn_f32_qu8_vcvt_ukernel__scalar_magic_iminmax_x3,
                  xnn_init_f32_qu8_cvt_scalar_magic_iminmax_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_magic_iminmax_x4,
                  xnn_f32_qu8_vcvt_ukernel__scalar_magic_iminmax_x4,
                  xnn_init_f32_qu8_cvt_scalar_magic_iminmax_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
