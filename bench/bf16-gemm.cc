// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "gemm.h"
#include "utils.h"
#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pack.h"
#include "xnnpack/buffer.h"
#include <benchmark/benchmark.h>

static void bf16_gemm(benchmark::State& state,
  xnn_bf16_gemm_minmax_ukernel_fn gemm,
  size_t mr, size_t nr, size_t kr, size_t sr,
  xnn_init_bf16_minmax_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  const size_t nc_stride = benchmark::utils::RoundUp(nc, nr);
  const size_t kc_stride = benchmark::utils::RoundUp(kc, kr * sr);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  xnnpack::Buffer<xnn_bfloat16> a(mc * kc + XNN_EXTRA_BYTES / sizeof(xnn_bfloat16));
  std::generate(a.begin(), a.end(), [&] { return xnn_bfloat16_from_float(f32rng(rng)); });
  xnnpack::Buffer<xnn_bfloat16> k(nc * kc);
  std::generate(k.begin(), k.end(), [&] { return xnn_bfloat16_from_float(f32rng(rng)); });
  xnnpack::Buffer<xnn_bfloat16> b(nc);
  std::generate(b.begin(), b.end(), [&] { return xnn_bfloat16_from_float(f32rng(rng)); });

  const size_t w_elements = nc_stride * kc_stride + nc_stride;
  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(xnn_bfloat16) * (w_elements + c_elements));

  xnnpack::Buffer<xnn_bfloat16, XNN_ALLOCATION_ALIGNMENT> w(w_elements * num_buffers);
  xnn_pack_f16_gemm_goi_w(/*groups=*/1, nc, kc, nr, kr, sr,
                          reinterpret_cast<const uint16_t*>(k.data()),
                          reinterpret_cast<const uint16_t*>(b.data()), /*scale=*/nullptr,
                          reinterpret_cast<uint16_t*>(w.data()),
                          /*extra_bytes=*/0, /*params=*/nullptr);
  xnnpack::Buffer<xnn_bfloat16> c(c_elements * num_buffers);

  // Prepare minmax parameters.
  xnn_bf16_minmax_params params;
  init_params(&params,
              -std::numeric_limits<float>::infinity(),
              +std::numeric_limits<float>::infinity());

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(xnn_bfloat16));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      for (uint32_t n = 0; n < nc; n += nr) {
        const uint32_t nb = min(nc - n, nr);
        gemm(
          mb, nb, kc * sizeof(xnn_bfloat16),
          a.data() + m * kc, kc * sizeof(xnn_bfloat16),
          w.data() + (nc_stride * buffer_index + n) * (kc_stride + 1),
          c.data() + (mc * buffer_index + m) * nc + n, nc * sizeof(xnn_bfloat16), nr * sizeof(xnn_bfloat16),
          &params);
      }
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}


#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void bf16_gemm_1x8c2__neonbf16_bfdot_lane_ld128(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128, 1, 8, 2, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONBF16);
  }
  static void bf16_gemm_4x8c2__neonbf16_bfdot_lane_ld128(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128, 4, 8, 2, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONBF16);
  }
  static void bf16_gemm_5x8c2__neonbf16_bfdot_lane_ld128(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128, 5, 8, 2, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONBF16);
  }
  static void bf16_gemm_6x8c2__neonbf16_bfdot_lane_ld128(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128, 6, 8, 2, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONBF16);
  }

  static void bf16_gemm_1x4c8__neonbf16_bfdot(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot, 1, 4, 8, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONBF16);
  }
  static void bf16_gemm_2x4c8__neonbf16_bfdot(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot, 2, 4, 8, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONBF16);
  }
  static void bf16_gemm_3x4c8__neonbf16_bfdot(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot, 3, 4, 8, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONBF16);
  }
  static void bf16_gemm_4x4c8__neonbf16_bfdot(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot, 4, 4, 8, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONBF16);
  }
  static void bf16_gemm_5x4c8__neonbf16_bfdot(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot, 5, 4, 8, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONBF16);
  }

  static void bf16_gemm_1x4c8__neonbf16_bfmlal(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal, 1, 4, 8, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONBF16);
  }
  static void bf16_gemm_2x4c8__neonbf16_bfmlal(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal, 2, 4, 8, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONBF16);
  }
  static void bf16_gemm_3x4c8__neonbf16_bfmlal(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal, 3, 4, 8, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONBF16);
  }
  static void bf16_gemm_4x4c8__neonbf16_bfmlal(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal, 4, 4, 8, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONBF16);
  }
  static void bf16_gemm_5x4c8__neonbf16_bfmlal(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal, 5, 4, 8, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONBF16);
  }

  BENCHMARK_GEMM(bf16_gemm_1x8c2__neonbf16_bfdot_lane_ld128)
  BENCHMARK_GEMM(bf16_gemm_4x8c2__neonbf16_bfdot_lane_ld128)
  BENCHMARK_GEMM(bf16_gemm_5x8c2__neonbf16_bfdot_lane_ld128)
  BENCHMARK_GEMM(bf16_gemm_6x8c2__neonbf16_bfdot_lane_ld128)

  BENCHMARK_GEMM(bf16_gemm_1x4c8__neonbf16_bfdot)
  BENCHMARK_GEMM(bf16_gemm_2x4c8__neonbf16_bfdot)
  BENCHMARK_GEMM(bf16_gemm_3x4c8__neonbf16_bfdot)
  BENCHMARK_GEMM(bf16_gemm_4x4c8__neonbf16_bfdot)
  BENCHMARK_GEMM(bf16_gemm_5x4c8__neonbf16_bfdot)

  BENCHMARK_GEMM(bf16_gemm_1x4c8__neonbf16_bfmlal)
  BENCHMARK_GEMM(bf16_gemm_2x4c8__neonbf16_bfmlal)
  BENCHMARK_GEMM(bf16_gemm_3x4c8__neonbf16_bfmlal)
  BENCHMARK_GEMM(bf16_gemm_4x4c8__neonbf16_bfmlal)
  BENCHMARK_GEMM(bf16_gemm_5x4c8__neonbf16_bfmlal)
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void bf16_gemm_1x4c8__neonfma_zip(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip, 1, 4, 8, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void bf16_gemm_2x4c8__neonfma_zip(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip, 2, 4, 8, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void bf16_gemm_3x4c8__neonfma_zip(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip, 3, 4, 8, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void bf16_gemm_4x4c8__neonfma_zip(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip, 4, 4, 8, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void bf16_gemm_5x4c8__neonfma_zip(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip, 5, 4, 8, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }

  static void bf16_gemm_1x4c8__neonfma_shland(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland, 1, 4, 8, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void bf16_gemm_2x4c8__neonfma_shland(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland, 2, 4, 8, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void bf16_gemm_3x4c8__neonfma_shland(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland, 3, 4, 8, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void bf16_gemm_4x4c8__neonfma_shland(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland, 4, 4, 8, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void bf16_gemm_5x4c8__neonfma_shland(benchmark::State& state, const char* net) {
    bf16_gemm(state, xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland, 5, 4, 8, 1,
      xnn_init_bf16_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(bf16_gemm_1x4c8__neonfma_zip)
  BENCHMARK_GEMM(bf16_gemm_2x4c8__neonfma_zip)
  BENCHMARK_GEMM(bf16_gemm_3x4c8__neonfma_zip)
  BENCHMARK_GEMM(bf16_gemm_4x4c8__neonfma_zip)
  BENCHMARK_GEMM(bf16_gemm_5x4c8__neonfma_zip)

  BENCHMARK_GEMM(bf16_gemm_1x4c8__neonfma_shland)
  BENCHMARK_GEMM(bf16_gemm_2x4c8__neonfma_shland)
  BENCHMARK_GEMM(bf16_gemm_3x4c8__neonfma_shland)
  BENCHMARK_GEMM(bf16_gemm_4x4c8__neonfma_shland)
  BENCHMARK_GEMM(bf16_gemm_5x4c8__neonfma_shland)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
