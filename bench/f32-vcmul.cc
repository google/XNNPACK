// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <complex>
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
#include <xnnpack/vbinary.h>


static void f32_vcmul(
  benchmark::State& state,
  xnn_f32_vbinary_ukernel_fn vcmul,
  xnn_init_f32_default_params_fn init_params = nullptr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::ref(rng));

  std::vector<float, AlignedAllocator<float, 64>> a(num_elements * 2 + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float, AlignedAllocator<float, 64>> b(num_elements * 2 + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float, AlignedAllocator<float, 64>> product(num_elements * 2);
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::generate(b.begin(), b.end(), std::ref(f32rng));

  union xnn_f32_default_params params;
  if (init_params != nullptr) {
    init_params(&params);
  }
  for (auto _ : state) {
    vcmul(num_elements * sizeof(float), a.data(), b.data(), product.data(), init_params == nullptr ? nullptr : &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t num_elements_per_iteration = num_elements;
  state.counters["num_elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * num_elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 3 * num_elements * sizeof(std::complex<float>);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_vcmul, neon_x4,
                    xnn_f32_vcmul_ukernel__neon_x4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<std::complex<float>, std::complex<float>>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vcmul, neon_x8,
                    xnn_f32_vcmul_ukernel__neon_x8,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<std::complex<float>, std::complex<float>>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vcmul, neon_x12,
                    xnn_f32_vcmul_ukernel__neon_x12,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<std::complex<float>, std::complex<float>>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vcmul, neon_x16,
                    xnn_f32_vcmul_ukernel__neon_x16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<std::complex<float>, std::complex<float>>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_vcmul, sse_x4,
                    xnn_f32_vcmul_ukernel__sse_x4)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<std::complex<float>, std::complex<float>>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vcmul, sse_x8,
                    xnn_f32_vcmul_ukernel__sse_x8)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<std::complex<float>, std::complex<float>>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vcmul, sse_x12,
                    xnn_f32_vcmul_ukernel__sse_x12)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<std::complex<float>, std::complex<float>>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vcmul, sse_x16,
                    xnn_f32_vcmul_ukernel__sse_x16)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<std::complex<float>, std::complex<float>>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_vcmul, wasmsimd_x4,
                    xnn_f32_vcmul_ukernel__wasmsimd_x4)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<std::complex<float>, std::complex<float>>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vcmul, wasmsimd_x8,
                    xnn_f32_vcmul_ukernel__wasmsimd_x8)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<std::complex<float>, std::complex<float>>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vcmul, wasmsimd_x12,
                    xnn_f32_vcmul_ukernel__wasmsimd_x12)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<std::complex<float>, std::complex<float>>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vcmul, wasmsimd_x16,
                    xnn_f32_vcmul_ukernel__wasmsimd_x16)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<std::complex<float>, std::complex<float>>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(f32_vcmul, scalar_x1,
                  xnn_f32_vcmul_ukernel__scalar_x1)
  ->Apply(benchmark::utils::BinaryElementwiseParameters<std::complex<float>, std::complex<float>>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vcmul, scalar_x2,
                  xnn_f32_vcmul_ukernel__scalar_x2)
  ->Apply(benchmark::utils::BinaryElementwiseParameters<std::complex<float>, std::complex<float>>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vcmul, scalar_x4,
                  xnn_f32_vcmul_ukernel__scalar_x4)
  ->Apply(benchmark::utils::BinaryElementwiseParameters<std::complex<float>, std::complex<float>>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vcmul, scalar_x8,
                  xnn_f32_vcmul_ukernel__scalar_x8)
  ->Apply(benchmark::utils::BinaryElementwiseParameters<std::complex<float>, std::complex<float>>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
