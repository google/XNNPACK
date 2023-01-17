// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <cpuinfo.h>
#include <pthreadpool.h>

#include <benchmark/benchmark.h>
#include <fp16/fp16.h>

#include "bench/utils.h"
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


struct ComputeErrorContext {
  const float* input;
  const float* output;
  float* error;
};

static void ComputeError(
  struct ComputeErrorContext* context,
  size_t start,
  size_t range)
{
  const float* input = context->input;
  const float* output = context->output;
  float* error = context->error;
  for (size_t i = start; i < start + range; i++) {
    const double input_val = input[i];
    const double output_ref = std::tanh(input_val);
    const double abs_error = std::abs(output_ref - double(output[i]));
    const float output_abs = std::abs(output_ref);
    const float output_ulp = uint32_as_float(float_as_uint32(output_abs) + 1) - output_abs;
    error[i] = float(abs_error / output_ulp);
  }
}

static void TanhError(benchmark::State& state,
  xnn_f32_unary_math_fn tanh,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (!cpuinfo_initialize()) {
    state.SkipWithError("failed cpuinfo init");
    return;
  }
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  // The smallest x for which tanh(x) is not -1.0f (-0x1.205966p+3f).
  const uint32_t min_input = 0xC1102CB3;
  // The largest x for which tanh(x) is not 1.0f (0x1.205966p+3f).
  const uint32_t max_input = 0x41102CB3;
  // Number of elements in one block of inputs/outputs.
  // Combining multiple elements in a block reduce function call overhead.
  const size_t block_size = 16384;
  // Number of elements in one parallelization tile. Worker threads process this many elements in each task.
  const size_t tile_size = 64;

  uint32_t num_threads = cpuinfo_get_cores_count();
  #if XNN_ARCH_ARM || XNN_ARCH_ARM64
    // Use all cores except for the least performant cluster
    if (cpuinfo_get_clusters_count() > 1) {
      num_threads -= cpuinfo_get_cluster(cpuinfo_get_clusters_count() - 1)->core_count;
    }
  #endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool(
    pthreadpool_create(num_threads), pthreadpool_destroy);

  std::vector<float, AlignedAllocator<float, 64>> x(block_size);
  std::vector<float, AlignedAllocator<float, 64>> y(block_size);
  std::vector<float> ulp_error(block_size);
  float max_ulp_error = 0.0f;

  ComputeErrorContext context;
  context.input = x.data();
  context.output = y.data();
  context.error = ulp_error.data();
  for (auto _ : state) {
    for (uint32_t n = min_input; int32_t(n) < 0; n -= block_size) {
      for (uint32_t i = 0; i < block_size; i++) {
        x[i] = uint32_as_float(std::max<uint32_t>(n - i, 0x80000000));
      }
      std::fill(y.begin(), y.end(), std::nanf(""));

      tanh(block_size * sizeof(float), x.data(), y.data());

      pthreadpool_parallelize_1d_tile_1d(
          threadpool.get(),
          reinterpret_cast<pthreadpool_task_1d_tile_1d_t>(ComputeError),
          static_cast<void*>(&context),
          block_size, tile_size, 0 /* flags */);

      max_ulp_error = std::accumulate(ulp_error.cbegin(), ulp_error.cend(), max_ulp_error,
        static_cast<const float& (*)(const float&, const float&)>(std::max<float>));
    }
    for (uint32_t n = 0; n < max_input; n += block_size) {
      for (uint32_t i = 0; i < block_size; i++) {
        x[i] = uint32_as_float(std::min<uint32_t>(n + i, max_input));
      }
      std::fill(y.begin(), y.end(), std::nanf(""));

      tanh(block_size * sizeof(float), x.data(), y.data());

      pthreadpool_parallelize_1d_tile_1d(
          threadpool.get(),
          reinterpret_cast<pthreadpool_task_1d_tile_1d_t>(ComputeError),
          static_cast<void*>(&context),
          block_size, tile_size, 0 /* flags */);

      max_ulp_error = std::accumulate(ulp_error.cbegin(), ulp_error.cend(), max_ulp_error,
        static_cast<const float& (*)(const float&, const float&)>(std::max<float>));
    }
  }

  state.counters["ULPERROR"] = benchmark::Counter(max_ulp_error);
}

#if XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(TanhError, aarch64_neonfma_expm1_rr1_p6_div,
                    xnn_math_f32_tanh__aarch64_neonfma_expm1_rr1_p6_div)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(TanhError, neonfma_expm1_rr1_p6_nr1recps1fma,
                    xnn_math_f32_tanh__neonfma_expm1_rr1_p6_nr1recps1fma,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfma_expm1_rr1_p6_nr2fma,
                    xnn_math_f32_tanh__neonfma_expm1_rr1_p6_nr2fma,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfma_expm1_rr1_p6_nr2recps,
                    xnn_math_f32_tanh__neonfma_expm1_rr1_p6_nr2recps,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(TanhError, neon_expm1_rr1_p6_nr2recps,
                    xnn_math_f32_tanh__neon_expm1_rr1_p6_nr2recps,
                    benchmark::utils::CheckNEON)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neon_expm1_rr2_p6_nr2recps,
                    xnn_math_f32_tanh__neon_expm1_rr2_p6_nr2recps,
                    benchmark::utils::CheckNEON)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(TanhError, avx512f_expm1_rr1_p6_div,
                    xnn_math_f32_tanh__avx512f_expm1_rr1_p6_div,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx512f_expm1_rr1_lut4_p4_perm_div,
                    xnn_math_f32_tanh__avx512f_expm1_rr1_lut4_p4_perm_div,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(TanhError, avx2_expm1_rr1_lut4_p4_perm_div,
                    xnn_math_f32_tanh__avx2_expm1_rr1_lut4_p4_perm_div,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx2_expm1_rr1_lut8_p4_perm_div,
                    xnn_math_f32_tanh__avx2_expm1_rr1_lut8_p4_perm_div,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx2_expm1_rr1_p6_div,
                    xnn_math_f32_tanh__avx2_expm1_rr1_p6_div,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(TanhError, fma3_expm1_rr1_lut4_p4_perm_div,
                    xnn_math_f32_tanh__fma3_expm1_rr1_lut4_p4_perm_div,
                    benchmark::utils::CheckFMA3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, fma3_expm1_rr1_p6_div,
                    xnn_math_f32_tanh__fma3_expm1_rr1_p6_div,
                    benchmark::utils::CheckFMA3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(TanhError, avx_expm1_rr1_lut4_p4_perm_div,
                    xnn_math_f32_tanh__avx_expm1_rr1_lut4_p4_perm_div,
                    benchmark::utils::CheckAVX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx_expm1_rr1_p6_div,
                    xnn_math_f32_tanh__avx_expm1_rr1_p6_div,
                    benchmark::utils::CheckAVX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(TanhError, sse2_expm1_rr1_p6_div,
                    xnn_math_f32_tanh__sse2_expm1_rr1_p6_div)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(TanhError, wasmsimd_expm1_rr1_p6_div_abs_min,
                    xnn_math_f32_tanh__wasmsimd_expm1_rr1_p6_div_abs_min)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, wasmsimd_expm1_rr1_p6_div_abs_pmin,
                    xnn_math_f32_tanh__wasmsimd_expm1_rr1_p6_div_abs_pmin)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, wasmsimd_expm1_rr1_p6_div_nabs_max,
                    xnn_math_f32_tanh__wasmsimd_expm1_rr1_p6_div_nabs_max)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, wasmsimd_expm1_rr1_p6_div_nabs_pmax,
                    xnn_math_f32_tanh__wasmsimd_expm1_rr1_p6_div_nabs_pmax)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(TanhError, fma_expm1_rr1_lut4_p4_div,
                  xnn_math_f32_tanh__fma_expm1_rr1_lut4_p4_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1_rr1_lut8_p3_div,
                  xnn_math_f32_tanh__fma_expm1_rr1_lut8_p3_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1_rr1_lut8_p4_div,
                  xnn_math_f32_tanh__fma_expm1_rr1_lut8_p4_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1_rr1_lut16_p3_div,
                  xnn_math_f32_tanh__fma_expm1_rr1_lut16_p3_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1_rr1_lut16_p4_div,
                  xnn_math_f32_tanh__fma_expm1_rr1_lut16_p4_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1_rr1_lut32_p3_div,
                  xnn_math_f32_tanh__fma_expm1_rr1_lut32_p3_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1_rr1_lut64_p3_div,
                  xnn_math_f32_tanh__fma_expm1_rr1_lut64_p3_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1_rr1_p6_div,
                  xnn_math_f32_tanh__fma_expm1_rr1_p6_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);

BENCHMARK_CAPTURE(TanhError, scalar_expm1_rr1_lut4_p4_div,
                  xnn_math_f32_tanh__scalar_expm1_rr1_lut4_p4_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1_rr1_lut8_p3_div,
                  xnn_math_f32_tanh__scalar_expm1_rr1_lut8_p3_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1_rr2_lut8_p3_div,
                  xnn_math_f32_tanh__scalar_expm1_rr2_lut8_p3_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1_rr1_lut8_p4_div,
                  xnn_math_f32_tanh__scalar_expm1_rr1_lut8_p4_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1_rr2_lut8_p4_div,
                  xnn_math_f32_tanh__scalar_expm1_rr2_lut8_p4_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1_rr1_lut16_p3_div,
                  xnn_math_f32_tanh__scalar_expm1_rr1_lut16_p3_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1_rr1_lut16_p4_div,
                  xnn_math_f32_tanh__scalar_expm1_rr1_lut16_p4_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1_rr1_lut32_p3_div,
                  xnn_math_f32_tanh__scalar_expm1_rr1_lut32_p3_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1_rr1_lut64_p3_div,
                  xnn_math_f32_tanh__scalar_expm1_rr1_lut64_p3_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1_rr1_p6_div,
                  xnn_math_f32_tanh__scalar_expm1_rr1_p6_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1_rr2_p6_div,
                  xnn_math_f32_tanh__scalar_expm1_rr2_p6_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
