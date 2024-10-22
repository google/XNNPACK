// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "utils.h"
#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/requantization-stubs.h"
#include "xnnpack/buffer.h"
#include <benchmark/benchmark.h>

static void qu8_requantization(
  benchmark::State& state,
  xnn_qu8_requantization_fn requantize,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(), std::ref(rng));

  xnnpack::Buffer<int32_t, XNN_ALLOCATION_ALIGNMENT> input(num_elements);
  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> output(num_elements);
  std::generate(input.begin(), input.end(), std::ref(i32rng));

  for (auto _ : state) {
    requantize(
      num_elements,
      input.data(),
      0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */,
      output.data());
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = num_elements * (sizeof(int32_t) + sizeof(uint8_t));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qu8_requantization, fp32__neon,
                    xnn_qu8_requantize_fp32__neon,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, uint8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qu8_requantization, gemmlowp__neon,
                    xnn_qu8_requantize_gemmlowp__neon,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qu8_requantization, fp32__sse2,
                    xnn_qu8_requantize_fp32__sse2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, uint8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qu8_requantization, gemmlowp__sse2,
                    xnn_qu8_requantize_gemmlowp__sse2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qu8_requantization, gemmlowp__ssse3,
                    xnn_qu8_requantize_gemmlowp__ssse3,
                    benchmark::utils::CheckSSSE3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qu8_requantization, gemmlowp__sse41,
                    xnn_qu8_requantize_gemmlowp__sse41,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(qu8_requantization, fp32__wasmsimd,
                    xnn_qu8_requantize_fp32__wasmsimd)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, uint8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qu8_requantization, gemmlowp__wasmsimd,
                    xnn_qu8_requantize_gemmlowp__wasmsimd)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(qu8_requantization, fp32__scalar_lrintf,
                  xnn_qu8_requantize_fp32__scalar_lrintf)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, uint8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(qu8_requantization, fp32__scalar_fmagic,
                  xnn_qu8_requantize_fp32__scalar_fmagic)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, uint8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(qu8_requantization, gemmlowp__scalar,
                  xnn_qu8_requantize_gemmlowp__scalar)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int32_t, uint8_t>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
