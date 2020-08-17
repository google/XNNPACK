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
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/common.h>
#include <xnnpack/params.h>
#include <xnnpack/vunary.h>


static void f32_relu(
  benchmark::State& state,
  xnn_f32_relu_ukernel_function f32_relu)
{
  const size_t n = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f), std::ref(rng));

  std::vector<float, AlignedAllocator<float, 64>> x(n);
  std::generate(x.begin(), x.end(), std::ref(f32rng));
  std::vector<float, AlignedAllocator<float, 64>> y(n);
  std::generate(x.begin(), x.end(), std::ref(f32rng));

  for (auto _ : state) {
    f32_relu(n * sizeof(float), x.data(), y.data(), NULL);
  }

    state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();

    state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * n, benchmark::Counter::kIsRate);

  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * n * sizeof(float), benchmark::Counter::kIsRate);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_relu, sse_x4, xnn_f32_relu_ukernel__sse_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_relu, sse_x8, xnn_f32_relu_ukernel__sse_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_relu, avx_x8, xnn_f32_relu_ukernel__avx_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_relu, avx_x16, xnn_f32_relu_ukernel__avx_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_relu, avx512f_x16, xnn_f32_relu_ukernel__avx512f_x16)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_relu, avx512f_x32, xnn_f32_relu_ukernel__avx512f_x32)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_relu, neon_x4, xnn_f32_relu_ukernel__neon_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_relu, neon_x8, xnn_f32_relu_ukernel__neon_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  BENCHMARK_CAPTURE(f32_relu, wasm_x1, xnn_f32_relu_ukernel__wasm_x1)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_relu, wasm_x2, xnn_f32_relu_ukernel__wasm_x2)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_relu, wasm_x4, xnn_f32_relu_ukernel__wasm_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_relu, wasm32_shr_x1, xnn_f32_relu_ukernel__wasm32_shr_x1)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_relu, wasm32_shr_x2, xnn_f32_relu_ukernel__wasm32_shr_x2)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_relu, wasm32_shr_x4, xnn_f32_relu_ukernel__wasm32_shr_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD

#if XNN_ARCH_WASMSIMD
  BENCHMARK_CAPTURE(f32_relu, wasmsimd_x4, xnn_f32_relu_ukernel__wasmsimd_x4)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_relu, wasmsimd_x8, xnn_f32_relu_ukernel__wasmsimd_x8)
    ->RangeMultiplier(10)
    ->Range(1000, 100000000)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD

BENCHMARK_CAPTURE(f32_relu, scalar_x1, xnn_f32_relu_ukernel__scalar_x1)
  ->RangeMultiplier(10)
  ->Range(1000, 100000000)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_relu, scalar_x2, xnn_f32_relu_ukernel__scalar_x2)
  ->RangeMultiplier(10)
  ->Range(1000, 100000000)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_relu, scalar_x4, xnn_f32_relu_ukernel__scalar_x4)
  ->RangeMultiplier(10)
  ->Range(1000, 100000000)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
