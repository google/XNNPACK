// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include <fp16/fp16.h>

#include "bench/utils.h"
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>


static void SqrtError(benchmark::State& state,
  xnn_f32_unary_math_function sqrt,
  size_t tile_size,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }

  const uint32_t min_input = 0x3F800000;
  const uint32_t max_input = 0x41800000;
  // Number of tiles in one block of inputs/outputs. Combining multiple tiles in a block reduce function call overhead.
  const size_t num_tiles = 100;

  double max_ulp_error = 0.0;
  std::vector<float, AlignedAllocator<float, 64>> x(tile_size * num_tiles);
  std::vector<float, AlignedAllocator<float, 64>> y(tile_size * num_tiles);
  for (auto _ : state) {
    for (uint32_t n = min_input; n < max_input; n += tile_size * num_tiles) {
      for (uint32_t i = 0; i < tile_size * num_tiles; i++) {
        x[i] = fp32_from_bits(std::min<uint32_t>(n + i, max_input));
      }
      std::fill(y.begin(), y.end(), std::nanf(""));

      sqrt(tile_size * num_tiles * sizeof(float), x.data(), y.data());

      for (uint32_t i = 0; i < tile_size * num_tiles; i++) {
        const double y_ref = std::sqrt(double(x[i]));
        const double abs_error = std::abs(y_ref - double(y[i]));
        const float y_abs = std::abs(y_ref);
        const float y_ulp = fp32_from_bits(fp32_to_bits(y_abs) + 1) - y_abs;
        max_ulp_error = std::max<double>(max_ulp_error, abs_error / y_ulp);
      }
    }
  }

  state.counters["ULPERROR"] = benchmark::Counter(max_ulp_error);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(SqrtError, sse_nr1mac, xnn_math_f32_sqrt__sse_nr1mac, 4)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, sse_nr2mac, xnn_math_f32_sqrt__sse_nr2mac, 4)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, sse_hh1mac, xnn_math_f32_sqrt__sse_hh1mac, 4)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, fma3_nr1fma, xnn_math_f32_sqrt__fma3_nr1fma, 8, benchmark::utils::CheckFMA3)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, fma3_nr2fma, xnn_math_f32_sqrt__fma3_nr2fma, 8, benchmark::utils::CheckFMA3)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, fma3_nr1fma1adj, xnn_math_f32_sqrt__fma3_nr1fma1adj, 8, benchmark::utils::CheckFMA3)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, avx512f_nr1fma, xnn_math_f32_sqrt__avx512f_nr1fma, 16, benchmark::utils::CheckAVX512F)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, avx512f_nr2fma, xnn_math_f32_sqrt__avx512f_nr2fma, 16, benchmark::utils::CheckAVX512F)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, avx512f_nr1fma1adj, xnn_math_f32_sqrt__avx512f_nr1fma1adj, 16, benchmark::utils::CheckAVX512F)->Unit(benchmark::kMillisecond)->Iterations(1);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(SqrtError, neon_nr1rsqrts, xnn_math_f32_sqrt__neon_nr1rsqrts, 4, benchmark::utils::CheckNEON)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, neon_nr2rsqrts, xnn_math_f32_sqrt__neon_nr2rsqrts, 4, benchmark::utils::CheckNEON)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, neon_nr3rsqrts, xnn_math_f32_sqrt__neon_nr3rsqrts, 4, benchmark::utils::CheckNEON)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, neonfma_nr1fma, xnn_math_f32_sqrt__neonfma_nr1fma, 4, benchmark::utils::CheckNEONFMA)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, neonfma_nr2fma, xnn_math_f32_sqrt__neonfma_nr2fma, 4, benchmark::utils::CheckNEONFMA)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, neonfma_nr3fma, xnn_math_f32_sqrt__neonfma_nr3fma, 4, benchmark::utils::CheckNEONFMA)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, neonfma_nr2fma1adj, xnn_math_f32_sqrt__neonfma_nr2fma1adj, 4, benchmark::utils::CheckNEONFMA)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, neonfma_nr1rsqrts1fma1adj, xnn_math_f32_sqrt__neonfma_nr1rsqrts1fma1adj, 4, benchmark::utils::CheckNEONFMA)->Unit(benchmark::kMillisecond)->Iterations(1);
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
