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
#include <xnnpack/vcvt.h>


static void f32_f16_vcvt(
  benchmark::State& state,
  xnn_f32_f16_vcvt_ukernel_function cvt,
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
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), std::ref(f32rng));
  std::fill(y.begin(), y.end(), UINT16_C(0x7E00));

  for (auto _ : state) {
    cvt(num_elements * sizeof(uint16_t), x.data(), y.data(), nullptr /* params */);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = num_elements * (sizeof(uint16_t) + sizeof(float));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_f16_vcvt, neonfp16_x8,
                    xnn_f32_f16_vcvt_ukernel__neonfp16_x8,
                    benchmark::utils::CheckNEONFP16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, neonfp16_x16,
                    xnn_f32_f16_vcvt_ukernel__neonfp16_x16,
                    benchmark::utils::CheckNEONFP16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_f16_vcvt, avx512skx_x16,
                    xnn_f32_f16_vcvt_ukernel__avx512skx_x16,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, avx512skx_x32,
                    xnn_f32_f16_vcvt_ukernel__avx512skx_x32,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_f16_vcvt, f16c_x8,
                    xnn_f32_f16_vcvt_ukernel__f16c_x8,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, f16c_x16,
                    xnn_f32_f16_vcvt_ukernel__f16c_x16,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime(); 

  BENCHMARK_CAPTURE(f32_f16_vcvt, avx_x8,
                    xnn_f32_f16_vcvt_ukernel__avx_x8,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, avx_x16,
                    xnn_f32_f16_vcvt_ukernel__avx_x16,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime(); 
  BENCHMARK_CAPTURE(f32_f16_vcvt, avx_x24,
                    xnn_f32_f16_vcvt_ukernel__avx_x24,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime(); 
  BENCHMARK_CAPTURE(f32_f16_vcvt, avx_x32,
                    xnn_f32_f16_vcvt_ukernel__avx_x32,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime(); 

  BENCHMARK_CAPTURE(f32_f16_vcvt, sse41_x8,
                    xnn_f32_f16_vcvt_ukernel__sse41_x8,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, sse41_x16,
                    xnn_f32_f16_vcvt_ukernel__sse41_x16,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime(); 
  BENCHMARK_CAPTURE(f32_f16_vcvt, sse41_x24,
                    xnn_f32_f16_vcvt_ukernel__sse41_x24,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime(); 
  BENCHMARK_CAPTURE(f32_f16_vcvt, sse41_x32,
                    xnn_f32_f16_vcvt_ukernel__sse41_x32,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime(); 

  BENCHMARK_CAPTURE(f32_f16_vcvt, sse2_x8,
                    xnn_f32_f16_vcvt_ukernel__sse2_x8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, sse2_x16,
                    xnn_f32_f16_vcvt_ukernel__sse2_x16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime(); 
  BENCHMARK_CAPTURE(f32_f16_vcvt, sse2_x24,
                    xnn_f32_f16_vcvt_ukernel__sse2_x24)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime(); 
  BENCHMARK_CAPTURE(f32_f16_vcvt, sse2_x32,
                    xnn_f32_f16_vcvt_ukernel__sse2_x32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime(); 
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD
  BENCHMARK_CAPTURE(f32_f16_vcvt, wasmsimd_x8,
                    xnn_f32_f16_vcvt_ukernel__wasmsimd_x8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, wasmsimd_x16,
                    xnn_f32_f16_vcvt_ukernel__wasmsimd_x16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime(); 
  BENCHMARK_CAPTURE(f32_f16_vcvt, wasmsimd_x24,
                    xnn_f32_f16_vcvt_ukernel__wasmsimd_x24)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime(); 
  BENCHMARK_CAPTURE(f32_f16_vcvt, wasmsimd_x32,
                    xnn_f32_f16_vcvt_ukernel__wasmsimd_x32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime(); 
#endif  // XNN_ARCH_WASMSIMD

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
