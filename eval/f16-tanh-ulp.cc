// Copyright 2023 Google LLC
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

#include <pthreadpool.h>

#include <benchmark/benchmark.h>
#include <fp16/fp16.h>

#include "bench/utils.h"
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>


constexpr uint16_t kNumSubnormalValues = 1024;

struct ComputeErrorContext {
  const uint16_t* input;
  const uint16_t* output;
  float* error;
  uint16_t num_flush_to_zero_values;
};

static void ComputeError(
  struct ComputeErrorContext* context,
  size_t start,
  size_t range)
{
  const uint16_t* input = context->input;
  const uint16_t* output = context->output;
  float* error = context->error;
  for (size_t i = start; i < start + range; i++) {
    uint16_t input_val = input[i];
    uint16_t output_val = output[i];
#if XNN_ARCH_ARM || XNN_ARCH_ARM64 || XNN_ARCH_X86 || XNN_ARCH_X86_64
    const uint16_t num_flush_to_zero_values = context->num_flush_to_zero_values;
    const uint16_t abs_input_val = input_val & UINT16_C(0x7FFF);
    if (abs_input_val < std::min<uint16_t>(num_flush_to_zero_values, kNumSubnormalValues)) {
      // Replace subnormal inputs with signed zeroes
      input_val = input_val & UINT16_C(0x8000);
    } else if (abs_input_val < num_flush_to_zero_values) {
      // For the smallest normalized floating-point numbers the implementation is likely to produce 0
      // instead of the correct result (same as input) due to denormals in intermediate computations.
      const uint16_t abs_output_val = output_val & UINT16_C(0x7FFF);
      if (abs_output_val == 0) {
        output_val = input_val;
      }
    }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64 || XNN_ARCH_X86 || XNN_ARCH_X86_64

    const float output_ref = std::tanh(fp16_ieee_to_fp32_value(input_val));
    const float abs_error = std::abs(output_ref - fp16_ieee_to_fp32_value(output_val));
    const uint16_t output_abs = fp16_ieee_from_fp32_value(std::abs(output_ref));
    const float output_ulp = fp16_ieee_to_fp32_value(output_abs + 1) - fp16_ieee_to_fp32_value(output_abs);
    error[i] = float(abs_error / output_ulp);
  }
}

static void TanhError(
  benchmark::State& state,
  xnn_f16_unary_math_fn tanh,
  uint16_t num_flush_to_zero_values,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  // The smallest x for which tanhh(x) is not -1.0h (-0x1.204p+2h).
  const uint16_t min_input = UINT16_C(0xC481);
  // The largest x for which tanhh(x) is not 1.0h (0x1.204p+2h).
  const uint16_t max_input = UINT16_C(0x4481);

  // Number of elements in one block of inputs/outputs.
  // Combining multiple elements in a block reduce function call overhead.
  const size_t block_size = 16384;
  // Number of elements in one parallelization tile. Worker threads process this many elements in each task.
  const size_t tile_size = 64;

  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> x(block_size);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> y(block_size);
  std::vector<float> ulp_error(block_size);
  float max_ulp_error = 0.0f;

  ComputeErrorContext context;
  context.input = x.data();
  context.output = y.data();
  context.error = ulp_error.data();
  context.num_flush_to_zero_values = num_flush_to_zero_values;
  for (auto _ : state) {
    for (uint16_t n = min_input; int16_t(n) < 0; n -= block_size) {
      for (uint16_t i = 0; i < block_size; i++) {
        x[i] = std::max<uint16_t>(n - i, UINT16_C(0x8000));
      }
      std::fill(y.begin(), y.end(), UINT16_C(0x7E00) /* NaN */);

      pthreadpool_parallelize_1d_tile_1d(
        nullptr,
        [&](size_t offset, size_t size) {
          tanh(size * sizeof(uint16_t), x.data() + offset, y.data() + offset);
        },
        block_size, tile_size, /*flags=*/PTHREADPOOL_FLAG_DISABLE_DENORMALS);

      pthreadpool_parallelize_1d_tile_1d(
          /*threadpool=*/nullptr,
          reinterpret_cast<pthreadpool_task_1d_tile_1d_t>(ComputeError),
          static_cast<void*>(&context),
          block_size, tile_size, /*flags=*/0);

      max_ulp_error = std::accumulate(ulp_error.cbegin(), ulp_error.cend(), max_ulp_error,
        static_cast<const float& (*)(const float&, const float&)>(std::max<float>));
    }
    for (uint16_t n = 0; n < max_input; n += block_size) {
      for (uint16_t i = 0; i < block_size; i++) {
        x[i] = std::min<uint16_t>(n + i, max_input);
      }
      std::fill(y.begin(), y.end(), UINT16_C(0x7E00) /* NaN */);

      pthreadpool_parallelize_1d_tile_1d(
        nullptr,
        [&](size_t offset, size_t size) {
          tanh(size * sizeof(uint16_t), x.data() + offset, y.data() + offset);
        },
        block_size, tile_size, /*flags=*/PTHREADPOOL_FLAG_DISABLE_DENORMALS);

      pthreadpool_parallelize_1d_tile_1d(
          /*threadpool=*/nullptr,
          reinterpret_cast<pthreadpool_task_1d_tile_1d_t>(ComputeError),
          static_cast<void*>(&context),
          block_size, tile_size, /*flags=*/0);

      max_ulp_error = std::accumulate(ulp_error.cbegin(), ulp_error.cend(), max_ulp_error,
        static_cast<const float& (*)(const float&, const float&)>(std::max<float>));
    }
  }

  state.counters["ULPERROR"] = benchmark::Counter(max_ulp_error);
}

#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(TanhError, aarch64_neonfp16arith_expm1minus_rr1_p3h1ts_div,
                    xnn_math_f16_tanh__aarch64_neonfp16arith_expm1minus_rr1_p3h1ts_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div,
                    xnn_math_f16_tanh__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(TanhError, neonfp16arith_expm1minus_rr1_p3h1ts_nr1fma,
                    xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_nr1fma,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfp16arith_expm1minus_rr1_p3h1ts_nr1fmaadj,
                    xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_nr1fmaadj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfp16arith_expm1minus_rr1_p3h1ts_nr1recps,
                    xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_nr1recps,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfp16arith_expm1minus_rr1_p3h1ts_nr1recpsadj,
                    xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_nr1recpsadj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 1,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfp16arith_expm1minus_rr1_p3h1ts_recpe,
                    xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_recpe,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 3,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfp16arith_expm1minus_rr1_p3h1ts_recpeadj,
                    xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_recpeadj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 3,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma,
                    xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfp16arith_expm1minus_rr1_p3h2ts_nr1fmaadj,
                    xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fmaadj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps,
                    xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfp16arith_expm1minus_rr1_p3h2ts_nr1recpsadj,
                    xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recpsadj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfp16arith_expm1minus_rr1_p3h2ts_recpe,
                    xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_recpe,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 3,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj,
                    xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj,
                    /*num_flush_to_zero_values=*/kNumSubnormalValues + 3,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(TanhError, avx2_expm1minus_rr1_p3h2ts_div,
                    xnn_math_f16_tanh__avx2_expm1minus_rr1_p3h2ts_div,
                    /*num_flush_to_zero_values=*/0,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, avx2_expm1minus_rr1_p3h2ts_rcp,
                    xnn_math_f16_tanh__avx2_expm1minus_rr1_p3h2ts_rcp,
                    /*num_flush_to_zero_values=*/0,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(TanhError, fma3_expm1minus_rr1_p3h2ts_div,
                    xnn_math_f16_tanh__fma3_expm1minus_rr1_p3h2ts_div,
                    /*num_flush_to_zero_values=*/0,
                    benchmark::utils::CheckFMA3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, fma3_expm1minus_rr1_p3h2ts_rcp,
                    xnn_math_f16_tanh__fma3_expm1minus_rr1_p3h2ts_rcp,
                    /*num_flush_to_zero_values=*/0,
                    benchmark::utils::CheckFMA3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, fma3_polynomial_p17h8t2,
                    xnn_math_f16_tanh__fma3_polynomial_p17h8t2,
                    /*num_flush_to_zero_values=*/0,
                    benchmark::utils::CheckFMA3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, fma3_polynomial_p19h9t2,
                    xnn_math_f16_tanh__fma3_polynomial_p19h9t2,
                    /*num_flush_to_zero_values=*/0,
                    benchmark::utils::CheckFMA3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

  BENCHMARK_CAPTURE(TanhError, f16c_expm1minus_rr1_p3h2ts_div,
                    xnn_math_f16_tanh__f16c_expm1minus_rr1_p3h2ts_div,
                    /*num_flush_to_zero_values=*/0,
                    benchmark::utils::CheckF16C)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, f16c_expm1minus_rr1_p3h2ts_rcp,
                    xnn_math_f16_tanh__f16c_expm1minus_rr1_p3h2ts_rcp,
                    /*num_flush_to_zero_values=*/0,
                    benchmark::utils::CheckF16C)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, f16c_polynomial_p17h8t2,
                    xnn_math_f16_tanh__f16c_polynomial_p17h8t2,
                    /*num_flush_to_zero_values=*/0,
                    benchmark::utils::CheckF16C)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(TanhError, f16c_polynomial_p19h9t2,
                    xnn_math_f16_tanh__f16c_polynomial_p19h9t2,
                    /*num_flush_to_zero_values=*/0,
                    benchmark::utils::CheckF16C)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
