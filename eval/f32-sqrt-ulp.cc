// Copyright 2019 Google LLC
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

#if XNN_ENABLE_CPUINFO
  #include <cpuinfo.h>
#endif  // XNN_ENABLE_CPUINFO
#include <pthreadpool.h>

#include <benchmark/benchmark.h>

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
    const double output_ref = std::sqrt(double(input[i]));
    const double abs_error = std::abs(output_ref - double(output[i]));
    const float output_abs = std::abs(output_ref);
    const float output_ulp = uint32_as_float(float_as_uint32(output_abs) + 1) - output_abs;
    error[i] = float(abs_error / output_ulp);
  }
}

static void SqrtError(benchmark::State& state,
  xnn_f32_unary_math_fn sqrt,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const uint32_t min_input = 0x3F800000;
  const uint32_t max_input = 0x41800000;
  // Number of elements in one block of inputs/outputs.
  // Combining multiple elements in a block reduce function call overhead.
  const size_t block_size = 1048576;
  // Number of elements in one parallelization tile. Worker threads process this many elements in each task.
  const size_t tile_size = 64;

  // Default: as many as logical processors in the system
  size_t num_threads = 0;
  #if XNN_ENABLE_CPUINFO
    if (cpuinfo_initialize()) {
      num_threads = cpuinfo_get_processors_count();
      #if XNN_ARCH_ARM || XNN_ARCH_ARM64
        // Use all cores except for the least performant cluster
        if (cpuinfo_get_clusters_count() > 1) {
          num_threads -= cpuinfo_get_cluster(cpuinfo_get_clusters_count() - 1)->core_count;
        }
      #endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
    }
  #endif  // XNN_ENABLE_CPUINFO

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
    for (uint32_t n = min_input; n < max_input; n += block_size) {
      for (uint32_t i = 0; i < block_size; i++) {
        x[i] = uint32_as_float(std::min<uint32_t>(n + i, max_input));
      }
      std::fill(y.begin(), y.end(), std::nanf(""));

      sqrt(block_size * sizeof(float), x.data(), y.data());

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

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(SqrtError, neonfma_nr1fma,
                    xnn_math_f32_sqrt__neonfma_nr1fma,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, neonfma_nr2fma,
                    xnn_math_f32_sqrt__neonfma_nr2fma,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, neonfma_nr3fma,
                    xnn_math_f32_sqrt__neonfma_nr3fma,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, neonfma_nr2fma1adj,
                    xnn_math_f32_sqrt__neonfma_nr2fma1adj,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, neonfma_nr1rsqrts1fma1adj,
                    xnn_math_f32_sqrt__neonfma_nr1rsqrts1fma1adj,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(SqrtError, neon_nr1rsqrts,
                    xnn_math_f32_sqrt__neon_nr1rsqrts,
                    benchmark::utils::CheckNEON)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, neon_nr2rsqrts,
                    xnn_math_f32_sqrt__neon_nr2rsqrts,
                    benchmark::utils::CheckNEON)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, neon_nr3rsqrts,
                    xnn_math_f32_sqrt__neon_nr3rsqrts,
                    benchmark::utils::CheckNEON)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(SqrtError, avx512f_nr1fma,
                    xnn_math_f32_sqrt__avx512f_nr1fma,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, avx512f_nr2fma,
                    xnn_math_f32_sqrt__avx512f_nr2fma,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, avx512f_nr1fma1adj,
                    xnn_math_f32_sqrt__avx512f_nr1fma1adj,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(SqrtError, fma3_nr1fma,
                    xnn_math_f32_sqrt__fma3_nr1fma,
                    benchmark::utils::CheckFMA3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, fma3_nr2fma,
                    xnn_math_f32_sqrt__fma3_nr2fma,
                    benchmark::utils::CheckFMA3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, fma3_nr1fma1adj,
                    xnn_math_f32_sqrt__fma3_nr1fma1adj,
                    benchmark::utils::CheckFMA3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(SqrtError, sse_nr1mac,
                    xnn_math_f32_sqrt__sse_nr1mac)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, sse_nr2mac,
                    xnn_math_f32_sqrt__sse_nr2mac)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, sse_hh1mac,
                    xnn_math_f32_sqrt__sse_hh1mac)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
