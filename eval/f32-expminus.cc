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
  xnn_f32_unary_math_function exp,
  size_t tile_size)
{
  // The smallest x for which expf(x) is normalized (-0x1.5D589Ep6f).
  const uint32_t min_input = 0xC2AEAC4FL;
  const size_t num_tiles = 100;

  double max_ulp_error = 0.0;
  std::vector<float, AlignedAllocator<float, 64>> x(tile_size * num_tiles);
  std::vector<float, AlignedAllocator<float, 64>> y(tile_size * num_tiles);
  for (auto _ : state) {
    for (uint32_t n = min_input; int32_t(n) < 0; n -= tile_size * num_tiles) {
      for (uint32_t i = 0; i < tile_size * num_tiles; i++) {
        x[i] = fp32_from_bits(std::max<uint32_t>(n - i, 0x80000000));
      }
      std::fill(y.begin(), y.end(), std::nanf(""));

      exp(tile_size * num_tiles * sizeof(float), x.data(), y.data());

      for (uint32_t i = 0; i < tile_size * num_tiles; i++) {
        const double y_ref = std::exp(double(x[i]));
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
  static void f32_expminus__sse2_p5(benchmark::State& state) {
    ExpError(state, xnn_math_f32_expminus__sse2_p5, 4);
  }
  static void f32_expminus__avx2_p5(benchmark::State& state) {
    ExpError(state, xnn_math_f32_expminus__avx2_p5, 8);
  }

  BENCHMARK(f32_expminus__sse2_p5)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_expminus__avx2_p5)->Unit(benchmark::kMillisecond)->Iterations(1);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_expminus__neonfma_p5(benchmark::State& state) {
    ExpError(state, xnn_math_f32_expminus__neonfma_p5, 4);
  }
  static void f32_expminus__neonfma_lut64_p2(benchmark::State& state) {
    ExpError(state, xnn_math_f32_expminus__neonfma_lut64_p2, 4);
  }
  static void f32_expminus__neonfma_lut2048_p1(benchmark::State& state) {
    ExpError(state, xnn_math_f32_expminus__neonfma_lut2048_p1, 4);
  }

  BENCHMARK(f32_expminus__neonfma_p5)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_expminus__neonfma_lut64_p2)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_expminus__neonfma_lut2048_p1)->Unit(benchmark::kMillisecond)->Iterations(1);
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

static void f32_expminus__scalar_p5(benchmark::State& state) {
  ExpError(state, xnn_math_f32_expminus__scalar_p5, 1);
}
static void f32_expminus__scalar_lut64_p2(benchmark::State& state) {
  ExpError(state, xnn_math_f32_expminus__scalar_lut64_p2, 1);
}
static void f32_expminus__scalar_lut2048_p1(benchmark::State& state) {
  ExpError(state, xnn_math_f32_expminus__scalar_lut2048_p1, 1);
}

BENCHMARK(f32_expminus__scalar_p5)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK(f32_expminus__scalar_lut64_p2)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK(f32_expminus__scalar_lut2048_p1)->Unit(benchmark::kMillisecond)->Iterations(1);

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
