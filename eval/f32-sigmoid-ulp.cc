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
    const double input_val = input[i];
    double output_ref = 0.0;
    if (input_val < 0.0) {
      const double exp_val = std::exp(input_val);
      output_ref = exp_val / (1.0 + exp_val);
    } else {
      output_ref = 1.0 / (1.0 + std::exp(-input_val));
    }
    const double abs_error = std::abs(output_ref - double(output[i]));
    const float output_abs = std::abs(output_ref);
    const float output_ulp = uint32_as_float(float_as_uint32(output_abs) + 1) - output_abs;
    error[i] = float(abs_error / output_ulp);
  }
}

static void SigmoidError(benchmark::State& state,
  xnn_f32_unary_math_fn sigmoid,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  // The smallest x for which sigmoidf(x) is normalized (-0x1.5D589Ep+6f).
  const uint32_t min_input = 0xC2AEAC4F;
  // The largest x for which sigmoidf(x) is not 1.0f (0x1.154244p+4f).
  const uint32_t max_input = 0x418AA122;
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

      sigmoid(block_size * sizeof(float), x.data(), y.data());

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

      sigmoid(block_size * sizeof(float), x.data(), y.data());

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
  BENCHMARK_CAPTURE(SigmoidError, neonfma_rr1_lut64_p2_nr2recps,
                    xnn_math_f32_sigmoid__neonfma_rr1_lut64_p2_nr2recps,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfma_rr1_lut64_p2_nr1recps1fma,
                    xnn_math_f32_sigmoid__neonfma_rr1_lut64_p2_nr1recps1fma,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfma_rr1_lut64_p2_nr2fma,
                    xnn_math_f32_sigmoid__neonfma_rr1_lut64_p2_nr2fma,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfma_rr2_lut64_p2_nr2recps,
                    xnn_math_f32_sigmoid__neonfma_rr2_lut64_p2_nr2recps,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfma_rr2_lut64_p2_nr1recps1fma,
                    xnn_math_f32_sigmoid__neonfma_rr2_lut64_p2_nr1recps1fma,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfma_rr2_lut64_p2_nr2fma,
                    xnn_math_f32_sigmoid__neonfma_rr2_lut64_p2_nr2fma,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfma_rr1_lut2048_p1_nr2recps,
                    xnn_math_f32_sigmoid__neonfma_rr1_lut2048_p1_nr2recps,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfma_rr1_lut2048_p1_nr1recps1fma,
                    xnn_math_f32_sigmoid__neonfma_rr1_lut2048_p1_nr1recps1fma,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfma_rr1_lut2048_p1_nr2fma,
                    xnn_math_f32_sigmoid__neonfma_rr1_lut2048_p1_nr2fma,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfma_rr2_lut2048_p1_nr2recps,
                    xnn_math_f32_sigmoid__neonfma_rr2_lut2048_p1_nr2recps,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfma_rr2_lut2048_p1_nr1recps1fma,
                    xnn_math_f32_sigmoid__neonfma_rr2_lut2048_p1_nr1recps1fma,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfma_rr2_lut2048_p1_nr2fma,
                    xnn_math_f32_sigmoid__neonfma_rr2_lut2048_p1_nr2fma,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfma_rr1_p5_nr2recps,
                    xnn_math_f32_sigmoid__neonfma_rr1_p5_nr2recps,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfma_rr1_p5_nr1recps1fma,
                    xnn_math_f32_sigmoid__neonfma_rr1_p5_nr1recps1fma,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfma_rr1_p5_nr2fma,
                    xnn_math_f32_sigmoid__neonfma_rr1_p5_nr2fma,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfma_rr2_p5_nr2recps,
                    xnn_math_f32_sigmoid__neonfma_rr2_p5_nr2recps,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfma_rr2_p5_nr1recps1fma,
                    xnn_math_f32_sigmoid__neonfma_rr2_p5_nr1recps1fma,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfma_rr2_p5_nr2fma,
                    xnn_math_f32_sigmoid__neonfma_rr2_p5_nr2fma,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(SigmoidError, neon_rr2_lut64_p2_nr2recps,
                    xnn_math_f32_sigmoid__neon_rr2_lut64_p2_nr2recps,
                    benchmark::utils::CheckNEON)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neon_rr2_lut2048_p1_nr2recps,
                    xnn_math_f32_sigmoid__neon_rr2_lut2048_p1_nr2recps,
                    benchmark::utils::CheckNEON)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neon_rr2_p5_nr2recps,
                    xnn_math_f32_sigmoid__neon_rr2_p5_nr2recps,
                    benchmark::utils::CheckNEON)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(SigmoidError, aarch64_neonfma_rr1_lut2048_p1_div,
                    xnn_math_f32_sigmoid__aarch64_neonfma_rr1_lut2048_p1_div)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, aarch64_neonfma_rr2_lut2048_p1_div,
                    xnn_math_f32_sigmoid__aarch64_neonfma_rr2_lut2048_p1_div)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, aarch64_neonfma_rr1_lut64_p2_div,
                    xnn_math_f32_sigmoid__aarch64_neonfma_rr1_lut64_p2_div)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, aarch64_neonfma_rr2_lut64_p2_div,
                    xnn_math_f32_sigmoid__aarch64_neonfma_rr2_lut64_p2_div)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, aarch64_neonfma_rr1_p5_div,
                    xnn_math_f32_sigmoid__aarch64_neonfma_rr1_p5_div)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, aarch64_neonfma_rr2_p5_div,
                    xnn_math_f32_sigmoid__aarch64_neonfma_rr2_p5_div)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr1_lut16_p3_perm_scalef_nr1fma,
                    xnn_math_f32_sigmoid__avx512f_rr1_lut16_p3_perm_scalef_nr1fma,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr1_lut16_p3_perm_scalef_nr1fma1adj,
                    xnn_math_f32_sigmoid__avx512f_rr1_lut16_p3_perm_scalef_nr1fma1adj,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr1_lut16_p3_perm_scalef_div,
                    xnn_math_f32_sigmoid__avx512f_rr1_lut16_p3_perm_scalef_div,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr2_lut16_p3_perm_scalef_nr1fma,
                    xnn_math_f32_sigmoid__avx512f_rr2_lut16_p3_perm_scalef_nr1fma,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr2_lut16_p3_perm_scalef_nr1fma1adj,
                    xnn_math_f32_sigmoid__avx512f_rr2_lut16_p3_perm_scalef_nr1fma1adj,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr2_lut16_p3_perm_scalef_div,
                    xnn_math_f32_sigmoid__avx512f_rr2_lut16_p3_perm_scalef_div,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr1_lut32_p2_perm2_scalef_nr1fma,
                    xnn_math_f32_sigmoid__avx512f_rr1_lut32_p2_perm2_scalef_nr1fma,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr1_lut32_p2_perm2_scalef_nr1fma1adj,
                    xnn_math_f32_sigmoid__avx512f_rr1_lut32_p2_perm2_scalef_nr1fma1adj,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr1_lut32_p2_perm2_scalef_div,
                    xnn_math_f32_sigmoid__avx512f_rr1_lut32_p2_perm2_scalef_div,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr2_lut32_p2_perm2_scalef_nr1fma,
                    xnn_math_f32_sigmoid__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr2_lut32_p2_perm2_scalef_nr1fma1adj,
                    xnn_math_f32_sigmoid__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma1adj,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr2_lut32_p2_perm2_scalef_div,
                    xnn_math_f32_sigmoid__avx512f_rr2_lut32_p2_perm2_scalef_div,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr1_lut64_p2_gather_scalef_nr1fma,
                    xnn_math_f32_sigmoid__avx512f_rr1_lut64_p2_gather_scalef_nr1fma,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr1_lut64_p2_gather_scalef_nr1fma1adj,
                    xnn_math_f32_sigmoid__avx512f_rr1_lut64_p2_gather_scalef_nr1fma1adj,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr1_lut64_p2_gather_scalef_div,
                    xnn_math_f32_sigmoid__avx512f_rr1_lut64_p2_gather_scalef_div,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr2_lut64_p2_gather_scalef_nr1fma,
                    xnn_math_f32_sigmoid__avx512f_rr2_lut64_p2_gather_scalef_nr1fma,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr2_lut64_p2_gather_scalef_nr1fma1adj,
                    xnn_math_f32_sigmoid__avx512f_rr2_lut64_p2_gather_scalef_nr1fma1adj,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr2_lut64_p2_gather_scalef_div,
                    xnn_math_f32_sigmoid__avx512f_rr2_lut64_p2_gather_scalef_div,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr1_p5_scalef_nr1fma,
                    xnn_math_f32_sigmoid__avx512f_rr1_p5_scalef_nr1fma,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr1_p5_scalef_nr1fma1adj,
                    xnn_math_f32_sigmoid__avx512f_rr1_p5_scalef_nr1fma1adj,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr1_p5_scalef_div,
                    xnn_math_f32_sigmoid__avx512f_rr1_p5_scalef_div,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr2_p5_scalef_nr1fma,
                    xnn_math_f32_sigmoid__avx512f_rr2_p5_scalef_nr1fma,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr2_p5_scalef_nr1fma1adj,
                    xnn_math_f32_sigmoid__avx512f_rr2_p5_scalef_nr1fma1adj,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx512f_rr2_p5_scalef_div,
                    xnn_math_f32_sigmoid__avx512f_rr2_p5_scalef_div,
                    benchmark::utils::CheckAVX512F)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(SigmoidError, avx2_rr1_lut64_p2_gather_nr1fma,
                    xnn_math_f32_sigmoid__avx2_rr1_lut64_p2_gather_nr1fma,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx2_rr1_lut64_p2_gather_nr2fma,
                    xnn_math_f32_sigmoid__avx2_rr1_lut64_p2_gather_nr2fma,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx2_rr1_lut64_p2_gather_nr2fma1adj,
                    xnn_math_f32_sigmoid__avx2_rr1_lut64_p2_gather_nr2fma1adj,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx2_rr1_lut64_p2_gather_div,
                    xnn_math_f32_sigmoid__avx2_rr1_lut64_p2_gather_div,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx2_rr2_lut64_p2_gather_nr1fma,
                    xnn_math_f32_sigmoid__avx2_rr2_lut64_p2_gather_nr1fma,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx2_rr2_lut64_p2_gather_nr2fma,
                    xnn_math_f32_sigmoid__avx2_rr2_lut64_p2_gather_nr2fma,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx2_rr2_lut64_p2_gather_nr2fma1adj,
                    xnn_math_f32_sigmoid__avx2_rr2_lut64_p2_gather_nr2fma1adj,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx2_rr2_lut64_p2_gather_div,
                    xnn_math_f32_sigmoid__avx2_rr2_lut64_p2_gather_div,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx2_rr1_p5_nr1fma,
                    xnn_math_f32_sigmoid__avx2_rr1_p5_nr1fma,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx2_rr1_p5_nr2fma,
                    xnn_math_f32_sigmoid__avx2_rr1_p5_nr2fma,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx2_rr1_p5_div,
                    xnn_math_f32_sigmoid__avx2_rr1_p5_div,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx2_rr2_p5_nr1fma,
                    xnn_math_f32_sigmoid__avx2_rr2_p5_nr1fma,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx2_rr2_p5_nr2fma,
                    xnn_math_f32_sigmoid__avx2_rr2_p5_nr2fma,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx2_rr2_p5_div,
                    xnn_math_f32_sigmoid__avx2_rr2_p5_div,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(SigmoidError, avx_rr2_lut64_p2_div,
                    xnn_math_f32_sigmoid__avx_rr2_lut64_p2_div,
                    benchmark::utils::CheckAVX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx_rr2_p5_nr1,
                    xnn_math_f32_sigmoid__avx_rr2_p5_nr1,
                    benchmark::utils::CheckAVX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx_rr2_p5_nr2,
                    xnn_math_f32_sigmoid__avx_rr2_p5_nr2,
                    benchmark::utils::CheckAVX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx_rr2_p5_div,
                    xnn_math_f32_sigmoid__avx_rr2_p5_div,
                    benchmark::utils::CheckAVX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(SigmoidError, sse2_rr2_lut64_p2_nr1,
                    xnn_math_f32_sigmoid__sse2_rr2_lut64_p2_nr1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, sse2_rr2_lut64_p2_nr2,
                    xnn_math_f32_sigmoid__sse2_rr2_lut64_p2_nr2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, sse2_rr2_lut64_p2_div,
                    xnn_math_f32_sigmoid__sse2_rr2_lut64_p2_div)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, sse2_rr2_p5_nr1,
                    xnn_math_f32_sigmoid__sse2_rr2_p5_nr1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, sse2_rr2_p5_nr2,
                    xnn_math_f32_sigmoid__sse2_rr2_p5_nr2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, sse2_rr2_p5_div,
                    xnn_math_f32_sigmoid__sse2_rr2_p5_div)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(SigmoidError, wasmsimd_rr2_lut64_p2_div,
                    xnn_math_f32_sigmoid__wasmsimd_rr2_lut64_p2_div)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, wasmsimd_rr2_p5_div,
                    xnn_math_f32_sigmoid__wasmsimd_rr2_p5_div)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(SigmoidError, scalar_rr2_lut64_p2_div,
                  xnn_math_f32_sigmoid__scalar_rr2_lut64_p2_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(SigmoidError, scalar_rr2_lut2048_p1_div,
                  xnn_math_f32_sigmoid__scalar_rr2_lut2048_p1_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(SigmoidError, scalar_rr2_p5_div,
                  xnn_math_f32_sigmoid__scalar_rr2_p5_div)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
