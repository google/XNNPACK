// Copyright 2020 Google LLC
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
    const double output_ref = std::expm1(double(input[i]));
    const double abs_error = std::abs(output_ref - double(output[i]));
    const float output_abs = std::abs(output_ref);
    const float output_ulp = uint32_as_float(float_as_uint32(output_abs) + 1) - output_abs;
    error[i] = float(abs_error / output_ulp);
  }
}

static void ExpM1Error(benchmark::State& state,
  xnn_f32_unary_math_fn expm1,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  // The smallest x for which expm1f(x) is not saturated at -1 (-0x1.154244p+4f).
  const uint32_t min_input = UINT32_C(0xC18AA122);
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

      expm1(block_size * sizeof(float), x.data(), y.data());

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
  BENCHMARK_CAPTURE(ExpM1Error, neon_rr2_lut16_p3,
                    xnn_math_f32_expm1minus__neon_rr2_lut16_p3,
                    benchmark::utils::CheckNEON)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(ExpM1Error, neon_rr2_p6,
                    xnn_math_f32_expm1minus__neon_rr2_p6,
                    benchmark::utils::CheckNEON)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(ExpM1Error, neonfma_rr1_lut16_p3,
                    xnn_math_f32_expm1minus__neonfma_rr1_lut16_p3,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(ExpM1Error, neonfma_rr1_p6,
                    xnn_math_f32_expm1minus__neonfma_rr1_p6,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(ExpM1Error, avx512f_rr1_lut16_p3_perm,
                    xnn_math_f32_expm1minus__avx512f_rr1_lut16_p3_perm,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(ExpM1Error, avx512f_rr1_p6,
                    xnn_math_f32_expm1minus__avx512f_rr1_p6,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(ExpM1Error, avx2_rr1_lut4_p4_perm,
                    xnn_math_f32_expm1minus__avx2_rr1_lut4_p4_perm,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(ExpM1Error, avx2_rr1_lut8_p4_perm,
                    xnn_math_f32_expm1minus__avx2_rr1_lut8_p4_perm,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(ExpM1Error, avx2_rr1_lut16_p3_gather,
                    xnn_math_f32_expm1minus__avx2_rr1_lut16_p3_gather,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(ExpM1Error, avx2_rr1_p6,
                    xnn_math_f32_expm1minus__avx2_rr1_p6,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(ExpM1Error, avx_rr2_lut4_p4_perm,
                    xnn_math_f32_expm1minus__avx_rr2_lut4_p4_perm,
                    benchmark::utils::CheckAVX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(ExpM1Error, avx_rr2_lut16_p3,
                    xnn_math_f32_expm1minus__avx_rr2_lut16_p3,
                    benchmark::utils::CheckAVX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(ExpM1Error, avx_rr2_p6,
                    xnn_math_f32_expm1minus__avx_rr2_p6,
                    benchmark::utils::CheckAVX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(ExpM1Error, sse2_rr2_lut16_p3,
                    xnn_math_f32_expm1minus__sse2_rr2_lut16_p3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(ExpM1Error, sse2_rr2_p6,
                    xnn_math_f32_expm1minus__sse2_rr2_p6)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(ExpM1Error, wasmsimd_rr2_lut16_p3_andnot,
                    xnn_math_f32_expm1minus__wasmsimd_rr2_lut16_p3_andnot)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(ExpM1Error, wasmsimd_rr2_lut16_p3_max,
                    xnn_math_f32_expm1minus__wasmsimd_rr2_lut16_p3_max)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(ExpM1Error, wasmsimd_rr2_p6_andnot,
                    xnn_math_f32_expm1minus__wasmsimd_rr2_p6_andnot)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(ExpM1Error, wasmsimd_rr2_p6_max,
                    xnn_math_f32_expm1minus__wasmsimd_rr2_p6_max)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(ExpM1Error, scalar_rr2_lut4_p4,
                  xnn_math_f32_expm1minus__scalar_rr2_lut4_p4)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(ExpM1Error, scalar_rr2_lut8_p3,
                  xnn_math_f32_expm1minus__scalar_rr2_lut8_p3)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(ExpM1Error, scalar_rr2_lut8_p4,
                  xnn_math_f32_expm1minus__scalar_rr2_lut8_p4)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(ExpM1Error, scalar_rr2_lut16_p3,
                  xnn_math_f32_expm1minus__scalar_rr2_lut16_p3)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(ExpM1Error, scalar_rr2_lut16_p4,
                  xnn_math_f32_expm1minus__scalar_rr2_lut16_p4)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(ExpM1Error, scalar_rr2_p5,
                  xnn_math_f32_expm1minus__scalar_rr2_p5)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(ExpM1Error, scalar_rr2_p6,
                  xnn_math_f32_expm1minus__scalar_rr2_p6)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
