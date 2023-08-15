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


constexpr uint32_t kNumSubnormalValues = 8388608;

struct ComputeErrorContext {
  const float* input;
  const float* output;
  float* error;
  uint32_t num_flush_to_zero_values;
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
    float input_val = input[i];
    float output_val = output[i];
#if XNN_ARCH_ARM || XNN_ARCH_ARM64 || XNN_ARCH_X86 || XNN_ARCH_X86_64
    const uint32_t num_flush_to_zero_values = context->num_flush_to_zero_values;
    const uint32_t abs_input_val = float_as_uint32(input_val) & UINT32_C(0x7FFFFFFF);
    if (abs_input_val < std::min<uint32_t>(num_flush_to_zero_values, kNumSubnormalValues)) {
      // Replace subnormal inputs with signed zeroes
      input_val = std::copysign(0.0f, input_val);
    } else if (abs_input_val < num_flush_to_zero_values) {
      // For the smallest normalized floating-point numbers the implementation is likely to produce 0
      // instead of the correct result (same as input) due to denormals in intermediate computations.
      if (std::abs(output_val) == 0.0f) {
        output_val = input_val;
      }
    }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64 || XNN_ARCH_X86 || XNN_ARCH_X86_64

    const double output_ref = std::tanh(double(input_val));
    const double abs_error = std::abs(output_ref - double(output_val));
    const float output_abs = std::abs(output_ref);
    const float output_ulp = uint32_as_float(float_as_uint32(output_abs) + 1) - output_abs;
    error[i] = float(abs_error / output_ulp);
  }
}

static void TanhError(
  benchmark::State& state,
  xnn_f32_unary_math_fn tanh,
  uint32_t num_flush_to_zero_values,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  // The smallest x for which tanh(x) is not -1.0f (-0x1.205966p+3f).
  const uint32_t min_input = 0xC1102CB3;
  // The largest x for which tanh(x) is not 1.0f (0x1.205966p+3f).
  const uint32_t max_input = 0x41102CB3;
  // Number of elements in one block of inputs/outputs.
  // Combining multiple elements in a block reduce function call overhead.
  const size_t block_size = 1048576;
  // Number of elements in one parallelization tile. Worker threads process this many elements in each task.
  const size_t tile_size = 1024;

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
  context.num_flush_to_zero_values = num_flush_to_zero_values;
  for (auto _ : state) {
    for (uint32_t n = min_input; int32_t(n) < 0; n -= block_size) {
      for (uint32_t i = 0; i < block_size; i++) {
        x[i] = uint32_as_float(std::max<uint32_t>(n - i, UINT32_C(0x80000000)));
      }
      std::fill(y.begin(), y.end(), std::nanf(""));

      pthreadpool_parallelize_1d_tile_1d(
        nullptr,
        [&](size_t offset, size_t size) {
          tanh(size * sizeof(float), x.data() + offset, y.data() + offset);
        },
        block_size, tile_size, /*flags=*/PTHREADPOOL_FLAG_DISABLE_DENORMALS);

      pthreadpool_parallelize_1d_tile_1d(
          threadpool.get(),
          reinterpret_cast<pthreadpool_task_1d_tile_1d_t>(ComputeError),
          static_cast<void*>(&context),
          block_size, tile_size, /*flags=*/0);

      max_ulp_error = std::accumulate(ulp_error.cbegin(), ulp_error.cend(), max_ulp_error,
        static_cast<const float& (*)(const float&, const float&)>(std::max<float>));
    }
    for (uint32_t n = 0; n < max_input; n += block_size) {
      for (uint32_t i = 0; i < block_size; i++) {
        x[i] = uint32_as_float(std::min<uint32_t>(n + i, max_input));
      }
      std::fill(y.begin(), y.end(), std::nanf(""));

      pthreadpool_parallelize_1d_tile_1d(
        nullptr,
        [&](size_t offset, size_t size) {
          tanh(size * sizeof(float), x.data() + offset, y.data() + offset);
        },
        block_size, tile_size, /*flags=*/PTHREADPOOL_FLAG_DISABLE_DENORMALS);

      pthreadpool_parallelize_1d_tile_1d(
          threadpool.get(),
          reinterpret_cast<pthreadpool_task_1d_tile_1d_t>(ComputeError),
          static_cast<void*>(&context),
          block_size, tile_size, /*flags=*/0);

      max_ulp_error = std::accumulate(ulp_error.cbegin(), ulp_error.cend(), max_ulp_error,
        static_cast<const float& (*)(const float&, const float&)>(std::max<float>));
    }
  }

  state.counters["ULPERROR"] = benchmark::Counter(max_ulp_error);
}

#if XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(TanhError, aarch64_neonfma_expm1minus_rr1_lut8_p4h3ps_div,
                    xnn_math_f32_tanh__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ps_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, aarch64_neonfma_expm1minus_rr1_p6h5ts_div,
                    xnn_math_f32_tanh__aarch64_neonfma_expm1minus_rr1_p6h5ts_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(TanhError, neonfma_expm1minus_rr1_lut8_p4h2ts_nr1recps1fma,
                    xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h2ts_nr1recps1fma,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfma_expm1minus_rr1_lut8_p4h2ts_nr2fma,
                    xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h2ts_nr2fma,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfma_expm1minus_rr1_lut8_p4h2ts_nr2recps,
                    xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h2ts_nr2recps,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfma_expm1minus_rr1_lut8_p4h3ps_nr1recps1fma,
                    xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr1recps1fma,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfma_expm1minus_rr1_lut8_p4h3ps_nr1recps1fmaadj,
                    xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr1recps1fmaadj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfma_expm1minus_rr1_lut8_p4h3ps_nr2fma,
                    xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr2fma,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfma_expm1minus_rr1_lut8_p4h3ps_nr2fmaadj,
                    xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr2fmaadj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfma_expm1minus_rr1_lut8_p4h3ps_nr2recps,
                    xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr2recps,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfma_expm1minus_rr1_lut8_p4h3ps_nr2recpsadj,
                    xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr2recpsadj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma,
                    xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfma_expm1minus_rr1_p6h5ts_nr1recps1fmaadj,
                    xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fmaadj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfma_expm1minus_rr1_p6h5ts_nr2fma,
                    xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr2fma,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfma_expm1minus_rr1_p6h5ts_nr2fmaadj,
                    xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr2fmaadj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfma_expm1minus_rr1_p6h5ts_nr2recps,
                    xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr2recps,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfma_expm1minus_rr1_p6h5ts_nr2recpsadj,
                    xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr2recpsadj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckNEONFMA)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(TanhError, neon_expm1minus_rr2_lut8_p4h2ts_nr2recps,
                    xnn_math_f32_tanh__neon_expm1minus_rr2_lut8_p4h2ts_nr2recps,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckNEON)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neon_expm1minus_rr2_lut8_p4h3ps_nr2recps,
                    xnn_math_f32_tanh__neon_expm1minus_rr2_lut8_p4h3ps_nr2recps,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckNEON)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neon_expm1minus_rr1_p6h5ts_nr2recps,
                    xnn_math_f32_tanh__neon_expm1minus_rr1_p6h5ts_nr2recps,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckNEON)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(TanhError, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div,
                    xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX512SKX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj,
                    xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX512SKX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx512skx_expm1minus_rr1_lut8_p4h3ps_perm_div,
                    xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_perm_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX512SKX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx512skx_expm1minus_rr1_lut8_p4h3ps_perm_nr1,
                    xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_perm_nr1,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX512SKX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx512skx_expm1minus_rr1_lut8_p4h3ps_perm_nr1adj,
                    xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_perm_nr1adj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX512SKX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx512skx_expm1minus_rr1_lut8_p4h3ps_gather_div,
                    xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_gather_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX512SKX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx512skx_expm1minus_rr1_lut8_p4h3ps_gather_nr1,
                    xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_gather_nr1,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX512SKX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx512skx_expm1minus_rr1_lut8_p4h3ps_gather_nr1adj,
                    xnn_math_f32_tanh__avx512skx_expm1minus_rr1_lut8_p4h3ps_gather_nr1adj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX512SKX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx512skx_expm1minus_rr1_p6h5ts_div,
                    xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5ts_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX512SKX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx512skx_expm1minus_rr1_p6h5ts_nr1,
                    xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5ts_nr1,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX512SKX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx512skx_expm1minus_rr1_p6h5ts_nr1adj,
                    xnn_math_f32_tanh__avx512skx_expm1minus_rr1_p6h5ts_nr1adj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX512SKX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(TanhError, avx2_expm1minus_rr1_lut4_p4h3ts_perm_div,
                    xnn_math_f32_tanh__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj,
                    xnn_math_f32_tanh__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx2_expm1minus_rr1_lut8_p4h3ps_perm_div,
                    xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_perm_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx2_expm1minus_rr1_lut8_p4h3ps_perm_nr1,
                    xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_perm_nr1,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx2_expm1minus_rr1_lut8_p4h3ps_perm_nr1adj,
                    xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_perm_nr1adj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx2_expm1minus_rr1_lut8_p4h3ps_gather_div,
                    xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_gather_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx2_expm1minus_rr1_lut8_p4h3ps_gather_nr1,
                    xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_gather_nr1,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx2_expm1minus_rr1_lut8_p4h3ps_gather_nr1adj,
                    xnn_math_f32_tanh__avx2_expm1minus_rr1_lut8_p4h3ps_gather_nr1adj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx2_expm1minus_rr1_p6h5ts_div,
                    xnn_math_f32_tanh__avx2_expm1minus_rr1_p6h5ts_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx2_expm1minus_rr1_p6h5ts_nr1,
                    xnn_math_f32_tanh__avx2_expm1minus_rr1_p6h5ts_nr1,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx2_expm1minus_rr1_p6h5ts_nr1adj,
                    xnn_math_f32_tanh__avx2_expm1minus_rr1_p6h5ts_nr1adj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(TanhError, fma3_expm1minus_rr1_lut4_p4h3ts_perm_div,
                    xnn_math_f32_tanh__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckFMA3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj,
                    xnn_math_f32_tanh__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckFMA3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, fma3_expm1minus_rr1_lut8_p4h3ps_div,
                    xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h3ps_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckFMA3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, fma3_expm1minus_rr1_lut8_p4h3ps_nr1,
                    xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h3ps_nr1,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckFMA3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, fma3_expm1minus_rr1_lut8_p4h3ps_nr1adj,
                    xnn_math_f32_tanh__fma3_expm1minus_rr1_lut8_p4h3ps_nr1adj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckFMA3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, fma3_expm1minus_rr1_p6h5ts_div,
                    xnn_math_f32_tanh__fma3_expm1minus_rr1_p6h5ts_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckFMA3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, fma3_expm1minus_rr1_p6h5ts_nr1,
                    xnn_math_f32_tanh__fma3_expm1minus_rr1_p6h5ts_nr1,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckFMA3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, fma3_expm1minus_rr1_p6h5ts_nr1adj,
                    xnn_math_f32_tanh__fma3_expm1minus_rr1_p6h5ts_nr1adj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckFMA3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(TanhError, avx_expm1minus_rr1_lut4_p4h2ts_perm_div,
                    xnn_math_f32_tanh__avx_expm1minus_rr1_lut4_p4h2ts_perm_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx_expm1minus_rr2_lut8_p4h2ts_nr1,
                    xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h2ts_nr1,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx_expm1minus_rr2_lut8_p4h2ts_nr2,
                    xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h2ts_nr2,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx_expm1minus_rr1_lut8_p4h3ps_div,
                    xnn_math_f32_tanh__avx_expm1minus_rr1_lut8_p4h3ps_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx_expm1minus_rr2_lut8_p4h3ps_nr1,
                    xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h3ps_nr1,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx_expm1minus_rr2_lut8_p4h3ps_nr2,
                    xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h3ps_nr2,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx_expm1minus_rr2_lut8_p4h3ts_nr1,
                    xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h3ts_nr1,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx_expm1minus_rr2_lut8_p4h3ts_nr2,
                    xnn_math_f32_tanh__avx_expm1minus_rr2_lut8_p4h3ts_nr2,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx_expm1minus_rr1_p6h5ts_div,
                    xnn_math_f32_tanh__avx_expm1minus_rr1_p6h5ts_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckAVX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx_expm1minus_rr1_p6h5ts_nr1,
                    xnn_math_f32_tanh__avx_expm1minus_rr1_p6h5ts_nr1,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckAVX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx_expm1minus_rr1_p6h5ts_nr2,
                    xnn_math_f32_tanh__avx_expm1minus_rr1_p6h5ts_nr2,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckAVX)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(TanhError, sse2_expm1minus_rr2_lut8_p4h2ts_nr1,
                    xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h2ts_nr1,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, sse2_expm1minus_rr2_lut8_p4h2ts_nr2,
                    xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h2ts_nr2,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, sse2_expm1minus_rr1_lut8_p4h3ps_div,
                    xnn_math_f32_tanh__sse2_expm1minus_rr1_lut8_p4h3ps_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, sse2_expm1minus_rr2_lut8_p4h3ps_nr1,
                    xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3ps_nr1,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, sse2_expm1minus_rr2_lut8_p4h3ps_nr2,
                    xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3ps_nr2,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, sse2_expm1minus_rr2_lut8_p4h3ts_nr1,
                    xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3ts_nr1,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, sse2_expm1minus_rr2_lut8_p4h3ts_nr2,
                    xnn_math_f32_tanh__sse2_expm1minus_rr2_lut8_p4h3ts_nr2,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(TanhError, sse2_expm1minus_rr1_p6h5ts_div,
                    xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5ts_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, sse2_expm1minus_rr1_p6h5ts_nr1,
                    xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5ts_nr1,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, sse2_expm1minus_rr1_p6h5ts_nr2,
                    xnn_math_f32_tanh__sse2_expm1minus_rr1_p6h5ts_nr2,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(TanhError, wasmsimd_expm1minus_rr1_lut8_p4h3ps_div_abs_min,
                    xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3ps_div_abs_min,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, wasmsimd_expm1minus_rr1_lut8_p4h3ps_div_abs_pmin,
                    xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3ps_div_abs_pmin,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, wasmsimd_expm1minus_rr1_lut8_p4h3ps_div_nabs_max,
                    xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3ps_div_nabs_max,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, wasmsimd_expm1minus_rr1_lut8_p4h3ps_div_nabs_pmax,
                    xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_lut8_p4h3ps_div_nabs_pmax,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(TanhError, wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min,
                    xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin,
                    xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max,
                    xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax,
                    xnn_math_f32_tanh__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(TanhError, wasm_expm1minus_rr1_lut8_p4h3ps_div,
                    xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3ps_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, wasm_expm1minus_rr1_p6h5ts_div,
                    xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5ts_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_lut4_p4h2ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_lut4_p4h2ts_rcp,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h2ts_rcp,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr2_lut4_p4h2ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr2_lut4_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_lut4_p4h3ps_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_lut4_p4h3ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_lut4_p4h3ps_rcp,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3ps_rcp,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_lut4_p4h3ts_rcp,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_lut4_p4h3ts_rcp,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr2_lut4_p4h3ps_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr2_lut4_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr2_lut4_p4h3ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr2_lut4_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_lut8_p3h1ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr2_lut8_p3h1ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_lut8_p4h2ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_lut8_p4h2ts_rcp,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h2ts_rcp,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr2_lut8_p4h2ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr2_lut8_p4h2ts_rcp,
                  xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h2ts_rcp,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_lut8_p4h3ps_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_lut8_p4h3ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_lut8_p4h3ps_rcp,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3ps_rcp,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_lut8_p4h3ts_rcp,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_lut8_p4h3ts_rcp,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr2_lut8_p4h3ps_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr2_lut8_p4h3ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr2_lut8_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_lut16_p3h1ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr2_lut16_p3h1ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_lut16_p4h2ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_lut16_p4h2ts_rcp,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h2ts_rcp,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr2_lut16_p4h2ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_lut16_p4h3ps_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_lut16_p4h3ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_lut16_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr2_lut16_p4h3ps_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr2_lut16_p4h3ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr2_lut16_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_lut32_p3h1ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_lut32_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr2_lut32_p3h1ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr2_lut32_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_lut64_p3h1ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_lut64_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr2_lut64_p3h1ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr2_lut64_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_p6h4ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_p6h4ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr2_p6h4ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr2_p6h4ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_p6h5ps_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_p6h5ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_p6h5ps_rcp,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5ps_rcp,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr1_p6h5ts_rcp,
                  xnn_math_f32_tanh__fma_expm1minus_rr1_p6h5ts_rcp,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr2_p6h5ps_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr2_p6h5ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1minus_rr2_p6h5ts_div,
                  xnn_math_f32_tanh__fma_expm1minus_rr2_p6h5ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr1_lut4_p4h2ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr1_lut4_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr2_lut4_p4h2ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr2_lut4_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr1_lut4_p4h3ps_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr1_lut4_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr1_lut4_p4h3ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr1_lut4_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr2_lut4_p4h3ps_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr2_lut4_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr2_lut4_p4h3ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr2_lut4_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr1_lut8_p3h1ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr2_lut8_p3h1ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr1_lut8_p4h2ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr2_lut8_p4h2ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr1_lut8_p4h3ps_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr1_lut8_p4h3ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr1_lut8_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr2_lut8_p4h3ps_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr2_lut8_p4h3ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr2_lut8_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr1_lut16_p3h1ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr2_lut16_p3h1ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr1_lut16_p4h2ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr2_lut16_p4h2ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr1_lut16_p4h3ps_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr1_lut16_p4h3ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr1_lut16_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr2_lut16_p4h3ps_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr2_lut16_p4h3ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr2_lut16_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr1_lut32_p3h1ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr1_lut32_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr2_lut32_p3h1ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr2_lut32_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr1_lut64_p3h1ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr1_lut64_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr2_lut64_p3h1ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr2_lut64_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr1_p6h4ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr1_p6h4ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr2_p6h4ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr2_p6h4ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr1_p6h5ps_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr1_p6h5ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr1_p6h5ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr1_p6h5ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr2_p6h5ps_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr2_p6h5ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, fma_expm1plus_rr2_p6h5ts_div,
                  xnn_math_f32_tanh__fma_expm1plus_rr2_p6h5ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);

BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_lut4_p4h2ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_lut4_p4h2ts_rcp,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h2ts_rcp,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr2_lut4_p4h2ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr2_lut4_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_lut4_p4h3ps_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_lut4_p4h3ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_lut4_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr2_lut4_p4h3ps_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr2_lut4_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr2_lut4_p4h3ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr2_lut4_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_lut8_p3h1ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr2_lut8_p3h1ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_lut8_p4h2ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_lut8_p4h2ts_rcp,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h2ts_rcp,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr2_lut8_p4h2ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr2_lut8_p4h2ts_rcp,
                  xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h2ts_rcp,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_lut8_p4h3ps_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_lut8_p4h3ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_lut8_p4h3ps_rcp,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3ps_rcp,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_lut8_p4h3ts_rcp,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_lut8_p4h3ts_rcp,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr2_lut8_p4h3ps_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr2_lut8_p4h3ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr2_lut8_p4h3ps_rcp,
                  xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3ps_rcp,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr2_lut8_p4h3ts_rcp,
                  xnn_math_f32_tanh__scalar_expm1minus_rr2_lut8_p4h3ts_rcp,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_lut16_p3h1ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr2_lut16_p3h1ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_lut16_p4h2ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_lut16_p4h2ts_rcp,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h2ts_rcp,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr2_lut16_p4h2ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_lut16_p4h3ps_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_lut16_p4h3ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_lut16_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr2_lut16_p4h3ps_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr2_lut16_p4h3ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr2_lut16_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_lut32_p3h1ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_lut32_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr2_lut32_p3h1ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr2_lut32_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_lut64_p3h1ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_lut64_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr2_lut64_p3h1ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr2_lut64_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_p6h4ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h4ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr2_p6h4ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr2_p6h4ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_p6h5ps_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_p6h5ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_p6h5ps_rcp,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5ps_rcp,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr1_p6h5ts_rcp,
                  xnn_math_f32_tanh__scalar_expm1minus_rr1_p6h5ts_rcp,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr2_p6h5ps_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr2_p6h5ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1minus_rr2_p6h5ts_div,
                  xnn_math_f32_tanh__scalar_expm1minus_rr2_p6h5ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr1_lut4_p4h2ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr1_lut4_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr2_lut4_p4h2ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr2_lut4_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr1_lut4_p4h3ps_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr1_lut4_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr1_lut4_p4h3ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr1_lut4_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr2_lut4_p4h3ps_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr2_lut4_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr2_lut4_p4h3ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr2_lut4_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr1_lut8_p3h1ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr2_lut8_p3h1ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr1_lut8_p4h2ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr2_lut8_p4h2ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr1_lut8_p4h3ps_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr1_lut8_p4h3ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr1_lut8_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr2_lut8_p4h3ps_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr2_lut8_p4h3ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr2_lut8_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr1_lut16_p3h1ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr2_lut16_p3h1ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr1_lut16_p4h2ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr2_lut16_p4h2ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p4h2ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr1_lut16_p4h3ps_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr1_lut16_p4h3ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr1_lut16_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr2_lut16_p4h3ps_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p4h3ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr2_lut16_p4h3ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr2_lut16_p4h3ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr1_lut32_p3h1ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr1_lut32_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr2_lut32_p3h1ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr2_lut32_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr1_lut64_p3h1ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr1_lut64_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr2_lut64_p3h1ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr2_lut64_p3h1ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr1_p6h4ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr1_p6h4ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr2_p6h4ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr2_p6h4ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr1_p6h5ps_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr1_p6h5ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr1_p6h5ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr1_p6h5ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr2_p6h5ps_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr2_p6h5ps_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);
BENCHMARK_CAPTURE(TanhError, scalar_expm1plus_rr2_p6h5ts_div,
                  xnn_math_f32_tanh__scalar_expm1plus_rr2_p6h5ts_div,
                  /*num_flush_to_zero_values=*/kNumSubnormalValues)
  ->Unit(benchmark::kMillisecond)
  ->Iterations(1);

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
