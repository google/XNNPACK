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

#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>


static void ExpError(benchmark::State& state,
  xnn_f32_ext_unary_math_function extexp,
  size_t tile_size)
{
  // The smallest x for which exp(x) (double-precision) is normal (-0x1.6232BCp9f).
  const uint32_t min_input = 0xC431195E;
  // The largest x for which exp(x) (double-precision) is finite (0x1.62E42Ep9).
  const uint32_t max_input = 0x44317217;
  // Number of tiles in one block of inputs/outputs. Combining multiple tiles in a block reduce function call overhead.
  const size_t num_tiles = 100;

  double max_ulp_error = 0.0;
  std::vector<float, AlignedAllocator<float, 64>> x(tile_size * num_tiles);
  std::vector<float, AlignedAllocator<float, 64>> m(tile_size * num_tiles);
  std::vector<float, AlignedAllocator<float, 64>> e(tile_size * num_tiles);
  for (auto _ : state) {
    for (uint32_t n = min_input; int32_t(n) < 0; n -= tile_size * num_tiles) {
      for (uint32_t i = 0; i < tile_size * num_tiles; i++) {
        x[i] = fp32_from_bits(std::max<uint32_t>(n - i, 0x80000000));
      }
      std::fill(m.begin(), m.end(), std::nanf(""));
      std::fill(e.begin(), e.end(), std::nanf(""));

      extexp(tile_size * num_tiles * sizeof(float), x.data(), m.data(), e.data());

      for (uint32_t i = 0; i < tile_size * num_tiles; i++) {
        const double y_ref = std::exp(double(x[i]));
        int y_ref_e;
        const double y_ref_m = std::frexp(y_ref, &y_ref_e);
        const double ulp_error = std::abs(y_ref_m - std::ldexp(double(m[i]), int(e[i]) - y_ref_e)) * 0x1.0p+24;
        max_ulp_error = std::max<double>(max_ulp_error, ulp_error);
      }
    }
    for (uint32_t n = 0; n < max_input; n += tile_size * num_tiles) {
      for (uint32_t i = 0; i < tile_size * num_tiles; i++) {
        x[i] = fp32_from_bits(std::min<uint32_t>(n + i, max_input));
      }
      std::fill(m.begin(), m.end(), std::nanf(""));
      std::fill(e.begin(), e.end(), std::nanf(""));

      extexp(tile_size * num_tiles * sizeof(float), x.data(), m.data(), e.data());

      for (uint32_t i = 0; i < tile_size * num_tiles; i++) {
        const double y_ref = std::exp(double(x[i]));
        int y_ref_e;
        const double y_ref_m = std::frexp(y_ref, &y_ref_e);
        const double ulp_error = std::abs(y_ref_m - std::ldexp(double(m[i]), int(e[i]) - y_ref_e)) * 0x1.0p+24;
        max_ulp_error = std::max<double>(max_ulp_error, ulp_error);
      }
    }
  }

  state.counters["ULPERROR"] = benchmark::Counter(max_ulp_error);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_extexp__avx512f_p5(benchmark::State& state) {
    ExpError(state, xnn_math_f32_extexp__avx512f_p5, 16);
  }
  static void f32_extexp__avx2_p5(benchmark::State& state) {
    ExpError(state, xnn_math_f32_extexp__avx2_p5, 8);
  }

  BENCHMARK(f32_extexp__avx512f_p5)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_extexp__avx2_p5)->Unit(benchmark::kMillisecond)->Iterations(1);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
