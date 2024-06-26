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
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/math-stubs.h"
#include "xnnpack/math.h"

#include "bench/utils.h"
#include <benchmark/benchmark.h>
#include "pthreadpool.h"

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
    const double output_ref = std::exp(double(input[i]));
    const double abs_error = std::abs(output_ref - double(output[i]));
    const float output_abs = std::abs(output_ref);
    const float output_ulp = uint32_as_float(float_as_uint32(output_abs) + 1) - output_abs;
    error[i] = float(abs_error / output_ulp);
  }
}

static void ExpError(benchmark::State& state,
  xnn_f32_unary_math_fn exp,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  // The smallest x for which expf(x) is normalized (-0x1.5D589Ep6f).
  const uint32_t min_input = UINT32_C(0xC2AEAC4F);
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
    for (uint32_t n = min_input; int32_t(n) < 0; n -= block_size) {
      for (uint32_t i = 0; i < block_size; i++) {
        x[i] = uint32_as_float(std::max<uint32_t>(n - i, 0x80000000));
      }
      std::fill(y.begin(), y.end(), std::nanf(""));

      exp(block_size * sizeof(float), x.data(), y.data());

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
  BENCHMARK_CAPTURE(ExpError, neonfma_rr2_lut64_p2,
                    xnn_math_f32_expminus__neonfma_rr2_lut64_p2,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(ExpError, neonfma_rr2_lut2048_p1,
                    xnn_math_f32_expminus__neonfma_rr2_lut2048_p1,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(ExpError, neonfma_rr2_p5,
                    xnn_math_f32_expminus__neonfma_rr2_p5,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(ExpError, avx2_rr1_p5,
                    xnn_math_f32_expminus__avx2_rr1_p5,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(ExpError, avx2_rr2_p5,
                    xnn_math_f32_expminus__avx2_rr2_p5,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(ExpError, sse2_rr2_p5,
                    xnn_math_f32_expminus__sse2_rr2_p5,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

BENCHMARK_CAPTURE(ExpError, scalar_rr2_lut64_p2,
                  xnn_math_f32_expminus__scalar_rr2_lut64_p2)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(ExpError, scalar_rr2_lut2048_p1,
                  xnn_math_f32_expminus__scalar_rr2_lut2048_p1)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(ExpError, scalar_rr2_p5,
                  xnn_math_f32_expminus__scalar_rr2_p5)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
