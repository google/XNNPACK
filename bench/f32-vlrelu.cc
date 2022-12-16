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

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/vunary.h>


static void f32_vlrelu(
  benchmark::State& state,
  xnn_f32_vlrelu_ukernel_fn vlrelu,
  xnn_init_f32_lrelu_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t elements = state.range(0);
  std::vector<float, AlignedAllocator<float, 64>> input(elements);
  std::vector<float, AlignedAllocator<float, 64>> output(elements);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-5.0f, 5.0f), std::ref(rng));
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::fill(output.begin(), output.end(), std::nanf(""));

  union xnn_f32_lrelu_params params;
  init_params(&params, 0.01f);
  for (auto _ : state) {
    vlrelu(elements * sizeof(float), input.data(), output.data(), &params);
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

#if XNN_ARCH_ARM64 || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_vlrelu, neon_x4,
                    xnn_f32_vlrelu_ukernel__neon_x4,
                    xnn_init_f32_lrelu_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vlrelu, neon_x8,
                    xnn_f32_vlrelu_ukernel__neon_x8,
                    xnn_init_f32_lrelu_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM64 || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_vlrelu, sse_x4,
                    xnn_f32_vlrelu_ukernel__sse_x4,
                    xnn_init_f32_lrelu_sse_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vlrelu, sse_x8,
                    xnn_f32_vlrelu_ukernel__sse_x8,
                    xnn_init_f32_lrelu_sse_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vlrelu, sse2_x4,
                    xnn_f32_vlrelu_ukernel__sse2_x4,
                    xnn_init_f32_lrelu_sse_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vlrelu, sse2_x8,
                    xnn_f32_vlrelu_ukernel__sse2_x8,
                    xnn_init_f32_lrelu_sse_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vlrelu, sse41_x4,
                    xnn_f32_vlrelu_ukernel__sse41_x4,
                    xnn_init_f32_lrelu_sse_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vlrelu, sse41_x8,
                    xnn_f32_vlrelu_ukernel__sse41_x8,
                    xnn_init_f32_lrelu_sse_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vlrelu, avx_x8,
                    xnn_f32_vlrelu_ukernel__avx_x8,
                    xnn_init_f32_lrelu_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vlrelu, avx_x16,
                    xnn_f32_vlrelu_ukernel__avx_x16,
                    xnn_init_f32_lrelu_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vlrelu, avx512f_x16,
                    xnn_f32_vlrelu_ukernel__avx512f_x16,
                    xnn_init_f32_lrelu_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vlrelu, avx512f_x32,
                    xnn_f32_vlrelu_ukernel__avx512f_x32,
                    xnn_init_f32_lrelu_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_vlrelu, wasmrelaxedsimd_laneselect_x4,
                    xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_laneselect_x4,
                    xnn_init_f32_lrelu_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vlrelu, wasmrelaxedsimd_laneselect_x8,
                    xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_laneselect_x8,
                    xnn_init_f32_lrelu_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vlrelu, wasmrelaxedsimd_iminmax_x4,
                    xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_iminmax_x4,
                    xnn_init_f32_lrelu_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vlrelu, wasmrelaxedsimd_iminmax_x8,
                    xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_iminmax_x8,
                    xnn_init_f32_lrelu_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_vlrelu, wasmsimd_laneselect_x4,
                    xnn_f32_vlrelu_ukernel__wasmsimd_laneselect_x4,
                    xnn_init_f32_lrelu_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vlrelu, wasmsimd_laneselect_x8,
                    xnn_f32_vlrelu_ukernel__wasmsimd_laneselect_x8,
                    xnn_init_f32_lrelu_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vlrelu, wasmsimd_iminmax_x4,
                    xnn_f32_vlrelu_ukernel__wasmsimd_iminmax_x4,
                    xnn_init_f32_lrelu_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vlrelu, wasmsimd_iminmax_x8,
                    xnn_f32_vlrelu_ukernel__wasmsimd_iminmax_x8,
                    xnn_init_f32_lrelu_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_vlrelu, wasm_x1,
                    xnn_f32_vlrelu_ukernel__wasm_x1,
                    xnn_init_f32_lrelu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vlrelu, wasm_x2,
                    xnn_f32_vlrelu_ukernel__wasm_x2,
                    xnn_init_f32_lrelu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vlrelu, wasm_x4,
                    xnn_f32_vlrelu_ukernel__wasm_x4,
                    xnn_init_f32_lrelu_scalar_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(f32_vlrelu, scalar_x1,
                  xnn_f32_vlrelu_ukernel__scalar_x1,
                  xnn_init_f32_lrelu_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vlrelu, scalar_x2,
                  xnn_f32_vlrelu_ukernel__scalar_x2,
                  xnn_init_f32_lrelu_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vlrelu, scalar_x4,
                  xnn_f32_vlrelu_ukernel__scalar_x4,
                  xnn_init_f32_lrelu_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
