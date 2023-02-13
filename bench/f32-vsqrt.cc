// Copyright 2020 Google LLC
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


static void f32_vsqrt(
  benchmark::State& state,
  xnn_f32_vsqrt_ukernel_fn vsqrt,
  xnn_init_f32_sqrt_params_fn init_params = nullptr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);
  std::vector<float, AlignedAllocator<float, 64>> input(num_elements);
  std::vector<float, AlignedAllocator<float, 64>> output(num_elements);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 10.0f), std::ref(rng));
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::fill(output.begin(), output.end(), std::nanf(""));

  union xnn_f32_sqrt_params params;
  if (init_params != nullptr) {
    init_params(&params);
  }
  for (auto _ : state) {
    vsqrt(num_elements * sizeof(float), input.data(), output.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * num_elements * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_vsqrt, aarch64_neon_sqrt_x4,
                    xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_x4)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, aarch64_neon_sqrt_x8,
                    xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_x8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM64 || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr1rsqrts1fma1adj_x4,
                    xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x4,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr1rsqrts1fma1adj_x8,
                    xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x8,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr1rsqrts1fma1adj_x12,
                    xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x12,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr1rsqrts1fma1adj_x16,
                    xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x16,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr1rsqrts1fma1adj_x20,
                    xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x20,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr1rsqrts1fma1adj_x24,
                    xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x24,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr1rsqrts1fma1adj_x28,
                    xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x28,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr1rsqrts1fma1adj_x32,
                    xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x32,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr1rsqrts1fma1adj_x36,
                    xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x36,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr1rsqrts1fma1adj_x40,
                    xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x40,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr2fma1adj_x4,
                    xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_x4,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr2fma1adj_x8,
                    xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_x8,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr2fma1adj_x12,
                    xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_x12,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr2fma1adj_x16,
                    xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_x16,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr2fma1adj_x20,
                    xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_x20,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr2fma1adj_x24,
                    xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_x24,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr2fma1adj_x28,
                    xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_x28,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr2fma1adj_x32,
                    xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_x32,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr2fma1adj_x36,
                    xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_x36,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, neonfma_nr2fma1adj_x40,
                    xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_x40,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM64 || XNN_ARCH_ARM64

#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  BENCHMARK_CAPTURE(f32_vsqrt, rvv_sqrt_x1v,
                    xnn_f32_vsqrt_ukernel__rvv_sqrt_x1v,
                    nullptr /* init params */,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, rvv_sqrt_x2v,
                    xnn_f32_vsqrt_ukernel__rvv_sqrt_x2v,
                    nullptr /* init params */,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, rvv_sqrt_x4v,
                    xnn_f32_vsqrt_ukernel__rvv_sqrt_x4v,
                    nullptr /* init params */,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, rvv_sqrt_x8v,
                    xnn_f32_vsqrt_ukernel__rvv_sqrt_x8v,
                    nullptr /* init params */,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_vsqrt, avx512f_nr1fma1adj_x16,
                    xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_x16,
                    xnn_init_f32_sqrt_avx512_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, avx512f_nr1fma1adj_x32,
                    xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_x32,
                    xnn_init_f32_sqrt_avx512_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, avx512f_nr1fma1adj_x48,
                    xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_x48,
                    xnn_init_f32_sqrt_avx512_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, avx512f_nr1fma1adj_x64,
                    xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_x64,
                    xnn_init_f32_sqrt_avx512_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, avx512f_nr1fma1adj_x80,
                    xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_x80,
                    xnn_init_f32_sqrt_avx512_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, avx512f_nr1fma1adj_x96,
                    xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_x96,
                    xnn_init_f32_sqrt_avx512_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, avx512f_nr1fma1adj_x112,
                    xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_x112,
                    xnn_init_f32_sqrt_avx512_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, avx512f_nr1fma1adj_x128,
                    xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_x128,
                    xnn_init_f32_sqrt_avx512_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vsqrt, fma3_nr1fma1adj_x8,
                    xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_x8,
                    xnn_init_f32_sqrt_fma_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, fma3_nr1fma1adj_x16,
                    xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_x16,
                    xnn_init_f32_sqrt_fma_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, fma3_nr1fma1adj_x24,
                    xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_x24,
                    xnn_init_f32_sqrt_fma_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, fma3_nr1fma1adj_x32,
                    xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_x32,
                    xnn_init_f32_sqrt_fma_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, fma3_nr1fma1adj_x40,
                    xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_x40,
                    xnn_init_f32_sqrt_fma_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, fma3_nr1fma1adj_x48,
                    xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_x48,
                    xnn_init_f32_sqrt_fma_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, fma3_nr1fma1adj_x56,
                    xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_x56,
                    xnn_init_f32_sqrt_fma_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, fma3_nr1fma1adj_x64,
                    xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_x64,
                    xnn_init_f32_sqrt_fma_params,
                    benchmark::utils::CheckFMA3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vsqrt, avx_sqrt_x8,
                    xnn_f32_vsqrt_ukernel__avx_sqrt_x8,
                    xnn_init_f32_sqrt_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, avx_sqrt_x16,
                    xnn_f32_vsqrt_ukernel__avx_sqrt_x16,
                    xnn_init_f32_sqrt_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vsqrt, sse_sqrt_x4,
                    xnn_f32_vsqrt_ukernel__sse_sqrt_x4)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, sse_sqrt_x8,
                    xnn_f32_vsqrt_ukernel__sse_sqrt_x8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_vsqrt, wasmsimd_sqrt_x4,
                    xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_x4)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vsqrt, wasmsimd_sqrt_x8,
                    xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_x8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(f32_vsqrt, scalar_sqrt_x1,
                  xnn_f32_vsqrt_ukernel__scalar_sqrt_x1)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vsqrt, scalar_sqrt_x2,
                  xnn_f32_vsqrt_ukernel__scalar_sqrt_x2)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vsqrt, scalar_sqrt_x4,
                  xnn_f32_vsqrt_ukernel__scalar_sqrt_x4)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
