// Copyright 2020 Google LLC
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


static void Expm1Error(benchmark::State& state,
  xnn_f32_unary_math_function expm1,
  size_t tile_size,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }

  // The smallest x for which expm1f(x) is not saturated at -1 (-0x1.154244p+4f).
  const uint32_t min_input = 0xC18AA122;
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

      expm1(tile_size * num_tiles * sizeof(float), x.data(), y.data());

      for (uint32_t i = 0; i < tile_size * num_tiles; i++) {
        const double y_ref = std::expm1(double(x[i]));
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
  static void f32_expm1minus__neon_rr2_lut16_p3(benchmark::State& state) {
    Expm1Error(state, xnn_math_f32_expm1minus__neon_rr2_lut16_p3, 4, benchmark::utils::CheckNEON);
  }
  static void f32_expm1minus__neon_rr2_p6(benchmark::State& state) {
    Expm1Error(state, xnn_math_f32_expm1minus__neon_rr2_p6, 4, benchmark::utils::CheckNEON);
  }

  static void f32_expm1minus__neonfma_rr1_lut16_p3(benchmark::State& state) {
    Expm1Error(state, xnn_math_f32_expm1minus__neonfma_rr1_lut16_p3, 4, benchmark::utils::CheckNEONFMA);
  }
  static void f32_expm1minus__neonfma_rr1_p6(benchmark::State& state) {
    Expm1Error(state, xnn_math_f32_expm1minus__neonfma_rr1_p6, 4, benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK(f32_expm1minus__neon_rr2_lut16_p3)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_expm1minus__neon_rr2_p6)->Unit(benchmark::kMillisecond)->Iterations(1);

  BENCHMARK(f32_expm1minus__neonfma_rr1_lut16_p3)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_expm1minus__neonfma_rr1_p6)->Unit(benchmark::kMillisecond)->Iterations(1);
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_expm1minus__avx512f_rr1_lut16_p3_perm(benchmark::State& state) {
    Expm1Error(state, xnn_math_f32_expm1minus__avx512f_rr1_lut16_p3_perm, 16, benchmark::utils::CheckAVX512F);
  }
  static void f32_expm1minus__avx512f_rr1_p6(benchmark::State& state) {
    Expm1Error(state, xnn_math_f32_expm1minus__avx512f_rr1_p6, 16, benchmark::utils::CheckAVX512F);
  }

  BENCHMARK(f32_expm1minus__avx512f_rr1_lut16_p3_perm)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_expm1minus__avx512f_rr1_p6)->Unit(benchmark::kMillisecond)->Iterations(1);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_expm1minus__avx2_rr1_lut4_p4_perm(benchmark::State& state) {
    Expm1Error(state, xnn_math_f32_expm1minus__avx2_rr1_lut4_p4_perm, 8, benchmark::utils::CheckAVX2);
  }
  static void f32_expm1minus__avx2_rr1_lut8_p4_perm(benchmark::State& state) {
    Expm1Error(state, xnn_math_f32_expm1minus__avx2_rr1_lut8_p4_perm, 8, benchmark::utils::CheckAVX2);
  }
  static void f32_expm1minus__avx2_rr1_lut16_p3_gather(benchmark::State& state) {
    Expm1Error(state, xnn_math_f32_expm1minus__avx2_rr1_lut16_p3_gather, 8, benchmark::utils::CheckAVX2);
  }
  static void f32_expm1minus__avx2_rr1_p6(benchmark::State& state) {
    Expm1Error(state, xnn_math_f32_expm1minus__avx2_rr1_p6, 8, benchmark::utils::CheckAVX2);
  }

  BENCHMARK(f32_expm1minus__avx2_rr1_lut4_p4_perm)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_expm1minus__avx2_rr1_lut8_p4_perm)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_expm1minus__avx2_rr1_lut16_p3_gather)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_expm1minus__avx2_rr1_p6)->Unit(benchmark::kMillisecond)->Iterations(1);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_expm1minus__avx_rr2_lut4_p4_perm(benchmark::State& state) {
    Expm1Error(state, xnn_math_f32_expm1minus__avx_rr2_lut4_p4_perm, 8, benchmark::utils::CheckAVX);
  }
  static void f32_expm1minus__avx_rr2_lut16_p3(benchmark::State& state) {
    Expm1Error(state, xnn_math_f32_expm1minus__avx_rr2_lut16_p3, 8, benchmark::utils::CheckAVX);
  }
  static void f32_expm1minus__avx_rr2_p6(benchmark::State& state) {
    Expm1Error(state, xnn_math_f32_expm1minus__avx_rr2_p6, 8, benchmark::utils::CheckAVX);
  }

  BENCHMARK(f32_expm1minus__avx_rr2_lut4_p4_perm)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_expm1minus__avx_rr2_lut16_p3)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_expm1minus__avx_rr2_p6)->Unit(benchmark::kMillisecond)->Iterations(1);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_expm1minus__sse2_rr2_lut16_p3(benchmark::State& state) {
    Expm1Error(state, xnn_math_f32_expm1minus__sse2_rr2_lut16_p3, 4);
  }
  static void f32_expm1minus__sse2_rr2_p6(benchmark::State& state) {
    Expm1Error(state, xnn_math_f32_expm1minus__sse2_rr2_p6, 4);
  }

  BENCHMARK(f32_expm1minus__sse2_rr2_lut16_p3)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_expm1minus__sse2_rr2_p6)->Unit(benchmark::kMillisecond)->Iterations(1);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD
  static void f32_expm1minus__wasmsimd_rr2_lut16_p3_andnot(benchmark::State& state) {
    Expm1Error(state, xnn_math_f32_expm1minus__wasmsimd_rr2_lut16_p3_andnot, 4);
  }
  static void f32_expm1minus__wasmsimd_rr2_lut16_p3_max(benchmark::State& state) {
    Expm1Error(state, xnn_math_f32_expm1minus__wasmsimd_rr2_lut16_p3_max, 4);
  }
  static void f32_expm1minus__wasmsimd_rr2_p6_andnot(benchmark::State& state) {
    Expm1Error(state, xnn_math_f32_expm1minus__wasmsimd_rr2_p6_andnot, 4);
  }
  static void f32_expm1minus__wasmsimd_rr2_p6_max(benchmark::State& state) {
    Expm1Error(state, xnn_math_f32_expm1minus__wasmsimd_rr2_p6_max, 4);
  }

  BENCHMARK(f32_expm1minus__wasmsimd_rr2_lut16_p3_andnot)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_expm1minus__wasmsimd_rr2_lut16_p3_max)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_expm1minus__wasmsimd_rr2_p6_andnot)->Unit(benchmark::kMillisecond)->Iterations(1);
  BENCHMARK(f32_expm1minus__wasmsimd_rr2_p6_max)->Unit(benchmark::kMillisecond)->Iterations(1);
#endif  // XNN_ARCH_WASMSIMD

static void f32_expm1minus__scalar_rr2_lut4_p4(benchmark::State& state) {
  Expm1Error(state, xnn_math_f32_expm1minus__scalar_rr2_lut4_p4, 1);
}
static void f32_expm1minus__scalar_rr2_lut8_p3(benchmark::State& state) {
  Expm1Error(state, xnn_math_f32_expm1minus__scalar_rr2_lut8_p3, 1);
}
static void f32_expm1minus__scalar_rr2_lut8_p4(benchmark::State& state) {
  Expm1Error(state, xnn_math_f32_expm1minus__scalar_rr2_lut8_p4, 1);
}
static void f32_expm1minus__scalar_rr2_lut16_p3(benchmark::State& state) {
  Expm1Error(state, xnn_math_f32_expm1minus__scalar_rr2_lut16_p3, 1);
}
static void f32_expm1minus__scalar_rr2_lut16_p4(benchmark::State& state) {
  Expm1Error(state, xnn_math_f32_expm1minus__scalar_rr2_lut16_p4, 1);
}
static void f32_expm1minus__scalar_rr2_p5(benchmark::State& state) {
  Expm1Error(state, xnn_math_f32_expm1minus__scalar_rr2_p5, 1);
}
static void f32_expm1minus__scalar_rr2_p6(benchmark::State& state) {
  Expm1Error(state, xnn_math_f32_expm1minus__scalar_rr2_p6, 1);
}

BENCHMARK(f32_expm1minus__scalar_rr2_lut4_p4)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK(f32_expm1minus__scalar_rr2_lut8_p3)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK(f32_expm1minus__scalar_rr2_lut8_p4)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK(f32_expm1minus__scalar_rr2_lut16_p3)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK(f32_expm1minus__scalar_rr2_lut16_p4)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK(f32_expm1minus__scalar_rr2_p5)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK(f32_expm1minus__scalar_rr2_p6)->Unit(benchmark::kMillisecond)->Iterations(1);

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
