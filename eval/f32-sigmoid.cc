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


static void SigmoidError(benchmark::State& state,
  xnn_f32_unary_math_function sigmoid,
  size_t tile_size)
{
  // The smallest x for which sigmoidf(x) is normalized (-0x1.5D589Ep+6f).
  const uint32_t min_input = 0xC2AEAC4F;
  // The largest x for which sigmoidf(x) is not 1.0f (0x1.154244p+4f).
  const uint32_t max_input = 0x418AA122;
  // Number of tiles in one block of inputs/outputs. Combining multiple tiles in a block reduce function call overhead.
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

      sigmoid(tile_size * num_tiles * sizeof(float), x.data(), y.data());

      for (uint32_t i = 0; i < tile_size * num_tiles; i++) {
        const double e_ref = std::exp(double(x[i]));
        const double y_ref = e_ref / (e_ref + 1.0);
        const double abs_error = std::abs(y_ref - double(y[i]));
        const float y_abs = std::abs(y_ref);
        const float y_ulp = fp32_from_bits(fp32_to_bits(y_abs) + 1) - y_abs;
        max_ulp_error = std::max<double>(max_ulp_error, abs_error / y_ulp);
      }
    }
    for (uint32_t n = 0; n < max_input; n += tile_size * num_tiles) {
      for (uint32_t i = 0; i < tile_size * num_tiles; i++) {
        x[i] = fp32_from_bits(std::min<uint32_t>(n + i, max_input));
      }
      std::fill(y.begin(), y.end(), std::nanf(""));

      sigmoid(tile_size * num_tiles * sizeof(float), x.data(), y.data());

      for (uint32_t i = 0; i < tile_size * num_tiles; i++) {
        const double y_ref = 1.0 / (1.0 + std::exp(-double(x[i])));
        const double abs_error = std::abs(y_ref - double(y[i]));
        const float y_abs = std::abs(y_ref);
        const float y_ulp = fp32_from_bits(fp32_to_bits(y_abs) + 1) - y_abs;
        max_ulp_error = std::max<double>(max_ulp_error, abs_error / y_ulp);
      }
    }
  }

  state.counters["ULPERROR"] = benchmark::Counter(max_ulp_error);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_sigmoid__neon_frac_p9_p10_nr1recps(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neon_frac_p9_p10_nr1recps, 4);
  }

  static void f32_sigmoid__neon_rr2_lut2048_p1_nr2recps(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neon_rr2_lut2048_p1_nr2recps, 4);
  }
  static void f32_sigmoid__neonfma_rr2_lut2048_p1_nr2recps(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr2_lut2048_p1_nr2recps, 4);
  }
  static void f32_sigmoid__neonfma_rr2_lut2048_p1_nr1recps1fma(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr2_lut2048_p1_nr1recps1fma, 4);
  }
  static void f32_sigmoid__neonfma_rr2_lut2048_p1_nr2fma(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr2_lut2048_p1_nr2fma, 4);
  }

  static void f32_sigmoid__neon_rr2_lut64_p2_nr2recps(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neon_rr2_lut64_p2_nr2recps, 4);
  }
  static void f32_sigmoid__neonfma_rr2_lut64_p2_nr2recps(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr2_lut64_p2_nr2recps, 4);
  }
  static void f32_sigmoid__neonfma_rr2_lut64_p2_nr1recps1fma(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr2_lut64_p2_nr1recps1fma, 4);
  }
  static void f32_sigmoid__neonfma_rr2_lut64_p2_nr2fma(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr2_lut64_p2_nr2fma, 4);
  }

  static void f32_sigmoid__neon_rr2_p5_nr2recps(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neon_rr2_p5_nr2recps, 4);
  }
  static void f32_sigmoid__neonfma_rr2_p5_nr2recps(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr2_p5_nr2recps, 4);
  }
  static void f32_sigmoid__neonfma_rr2_p5_nr1recps1fma(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr2_p5_nr1recps1fma, 4);
  }
  static void f32_sigmoid__neonfma_rr2_p5_nr2fma(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr2_p5_nr2fma, 4);
  }

  static void f32_sigmoid__neon_rr1_lut2048_p1_nr2recps(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neon_rr1_lut2048_p1_nr2recps, 4);
  }
  static void f32_sigmoid__neonfma_rr1_lut2048_p1_nr2recps(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr1_lut2048_p1_nr2recps, 4);
  }
  static void f32_sigmoid__neonfma_rr1_lut2048_p1_nr1recps1fma(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr1_lut2048_p1_nr1recps1fma, 4);
  }
  static void f32_sigmoid__neonfma_rr1_lut2048_p1_nr2fma(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr1_lut2048_p1_nr2fma, 4);
  }

  static void f32_sigmoid__neon_rr1_lut64_p2_nr2recps(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neon_rr1_lut64_p2_nr2recps, 4);
  }
  static void f32_sigmoid__neonfma_rr1_lut64_p2_nr2recps(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr1_lut64_p2_nr2recps, 4);
  }
  static void f32_sigmoid__neonfma_rr1_lut64_p2_nr1recps1fma(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr1_lut64_p2_nr1recps1fma, 4);
  }
  static void f32_sigmoid__neonfma_rr1_lut64_p2_nr2fma(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr1_lut64_p2_nr2fma, 4);
  }

  static void f32_sigmoid__neon_rr1_p5_nr2recps(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neon_rr1_p5_nr2recps, 4);
  }
  static void f32_sigmoid__neonfma_rr1_p5_nr2recps(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr1_p5_nr2recps, 4);
  }
  static void f32_sigmoid__neonfma_rr1_p5_nr1recps1fma(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr1_p5_nr1recps1fma, 4);
  }
  static void f32_sigmoid__neonfma_rr1_p5_nr2fma(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr1_p5_nr2fma, 4);
  }

  BENCHMARK(f32_sigmoid__neon_frac_p9_p10_nr1recps)->Unit(benchmark::kMillisecond)->Iterations(1);

  BENCHMARK(f32_sigmoid__neon_rr2_lut2048_p1_nr2recps)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr2_lut2048_p1_nr2recps)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr2_lut2048_p1_nr1recps1fma)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr2_lut2048_p1_nr2fma)->Unit(benchmark::kMillisecond)->Iterations(1);

  BENCHMARK(f32_sigmoid__neon_rr2_lut64_p2_nr2recps)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr2_lut64_p2_nr2recps)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr2_lut64_p2_nr1recps1fma)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr2_lut64_p2_nr2fma)->Unit(benchmark::kMillisecond)->Iterations(1);

  BENCHMARK(f32_sigmoid__neon_rr2_p5_nr2recps)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr2_p5_nr2recps)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr2_p5_nr1recps1fma)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr2_p5_nr2fma)->Unit(benchmark::kMillisecond)->Iterations(1);

  BENCHMARK(f32_sigmoid__neon_rr1_lut2048_p1_nr2recps)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr1_lut2048_p1_nr2recps)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr1_lut2048_p1_nr1recps1fma)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr1_lut2048_p1_nr2fma)->Unit(benchmark::kMillisecond)->Iterations(1);

  BENCHMARK(f32_sigmoid__neon_rr1_lut64_p2_nr2recps)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr1_lut64_p2_nr2recps)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr1_lut64_p2_nr1recps1fma)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr1_lut64_p2_nr2fma)->Unit(benchmark::kMillisecond)->Iterations(1);

  BENCHMARK(f32_sigmoid__neon_rr1_p5_nr2recps)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr1_p5_nr2recps)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr1_p5_nr1recps1fma)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr1_p5_nr2fma)->Unit(benchmark::kMillisecond)->Iterations(1);
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM64
  static void f32_sigmoid__neonfma_rr2_lut2048_p1_div(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr2_lut2048_p1_div, 4);
  }
  static void f32_sigmoid__neonfma_rr2_lut64_p2_div(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr2_lut64_p2_div, 4);
  }
  static void f32_sigmoid__neonfma_rr2_p5_div(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr2_p5_div, 4);
  }
  static void f32_sigmoid__neonfma_rr1_lut2048_p1_div(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr1_lut2048_p1_div, 4);
  }
  static void f32_sigmoid__neonfma_rr1_lut64_p2_div(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr1_lut64_p2_div, 4);
  }
  static void f32_sigmoid__neonfma_rr1_p5_div(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__neonfma_rr1_p5_div, 4);
  }

  BENCHMARK(f32_sigmoid__neonfma_rr2_lut2048_p1_div)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr2_lut64_p2_div)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr2_p5_div)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr1_lut2048_p1_div)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr1_lut64_p2_div)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__neonfma_rr1_p5_div)->Unit(benchmark::kMillisecond)->Iterations(1);
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_sigmoid__avx2_rr2_p5_div(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__avx2_rr2_p5_div, 8);
  }
  static void f32_sigmoid__avx2_rr2_p5_nr2fma(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__avx2_rr2_p5_nr2fma, 8);
  }
  static void f32_sigmoid__avx2_rr2_p5_nr1fma(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__avx2_rr2_p5_nr1fma, 8);
  }
  static void f32_sigmoid__avx2_rr1_p5_div(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__avx2_rr1_p5_div, 8);
  }
  static void f32_sigmoid__avx2_rr1_p5_nr2fma(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__avx2_rr1_p5_nr2fma, 8);
  }
  static void f32_sigmoid__avx2_rr1_p5_nr1fma(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__avx2_rr1_p5_nr1fma, 8);
  }
  static void f32_sigmoid__sse2_p5_div(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__sse2_p5_div, 4);
  }

  BENCHMARK(f32_sigmoid__avx2_rr2_p5_div)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__avx2_rr2_p5_nr2fma)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__avx2_rr2_p5_nr1fma)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__avx2_rr1_p5_div)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__avx2_rr1_p5_nr2fma)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__avx2_rr1_p5_nr1fma)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_sigmoid__sse2_p5_div)->Unit(benchmark::kMillisecond)->Iterations(1);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  static void f32_sigmoid__psimd_p5_div(benchmark::State& state) {
    SigmoidError(state, xnn_math_f32_sigmoid__psimd_p5_div, 4);
  }

  BENCHMARK(f32_sigmoid__psimd_p5_div)->Unit(benchmark::kMillisecond)->Iterations(1);
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC

static void f32_sigmoid__scalar_lut2048_p1_div(benchmark::State& state) {
  SigmoidError(state, xnn_math_f32_sigmoid__scalar_lut2048_p1_div, 1);
}
static void f32_sigmoid__scalar_lut64_p2_div(benchmark::State& state) {
  SigmoidError(state, xnn_math_f32_sigmoid__scalar_lut64_p2_div, 1);
}
static void f32_sigmoid__scalar_p5_div(benchmark::State& state) {
  SigmoidError(state, xnn_math_f32_sigmoid__scalar_p5_div, 1);
}

BENCHMARK(f32_sigmoid__scalar_lut2048_p1_div)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK(f32_sigmoid__scalar_lut64_p2_div)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK(f32_sigmoid__scalar_p5_div)->Unit(benchmark::kMillisecond)->Iterations(1);

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
