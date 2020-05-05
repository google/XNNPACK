// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
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

#include <cpuinfo.h>

#include <benchmark/benchmark.h>
#include <fp16/fp16.h>
#include "bench/gemm.h"
#include "bench/utils.h"
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/common.h>
#include <xnnpack/gemm.h>
#include <xnnpack/pack.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


static void GEMMBenchmark(benchmark::State& state,
  xnn_f16_gemm_minmax_ukernel_function gemm,
  size_t mr, size_t nr, size_t kr, size_t sr)
{
  if (!cpuinfo_initialize()) {
    state.SkipWithError("cpuinfo initialization failed");
    return;
  }

  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  const size_t nc_stride = benchmark::utils::RoundUp(nc, nr);
  const size_t kc_stride = benchmark::utils::RoundUp(kc, kr);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

  std::vector<uint16_t> a(mc * kc);
  std::generate(a.begin(), a.end(), std::ref(f16rng));
  std::vector<uint16_t> k(nc * kc);
  std::generate(k.begin(), k.end(), std::ref(f16rng));
  std::vector<uint16_t> b(nc);
  std::generate(b.begin(), b.end(), std::ref(f16rng));

  const size_t w_elements = nc_stride * kc_stride + nc_stride;
  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(uint16_t) * (w_elements + c_elements));

  std::vector<uint16_t, AlignedAllocator<uint16_t, 32>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0);
  xnn_pack_f16_gemm_goi_w(1 /* groups */, nc, kc, nr, kr, sr, k.data(), b.data(), w.data());
  std::vector<uint16_t> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), UINT16_C(0x7E00) /* NaN */);

  // Prepare minmax parameters.
  xnn_f16_scaleminmax_params params;
  params = xnn_init_f16_scaleminmax_params(
    UINT16_C(0x3C00),  /* 1.0 */
    UINT16_C(0x7C00),  /* inf */
    UINT16_C(0xFC00)); /* -inf */

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(uint16_t));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      for (uint32_t n = 0; n < nc; n += nr) {
        const uint32_t nb = min(nc - n, nr);
        gemm(
          mb, nb, kc * sizeof(uint16_t),
          a.data() + m * kc, kc * sizeof(uint16_t),
          w.data() + (nc_stride * buffer_index + n) * (kc_stride + 1),
          c.data() + (mc * buffer_index + m) * nc + n, nc * sizeof(uint16_t), nr * sizeof(uint16_t),
          &params);
      }
    }
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM64
  static void f16_gemm_1x8__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64, 1, 8, 1, 1);
  }

  static void f16_gemm_4x8__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64, 4, 8, 1, 1);
  }

  static void f16_gemm_6x8__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64, 6, 8, 1, 1);
  }

  static void f16_gemm_8x8__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64, 8, 8, 1, 1);
  }

  BENCHMARK_GEMM(f16_gemm_1x8__neonfp16arith_ld64)
  BENCHMARK_GEMM(f16_gemm_4x8__neonfp16arith_ld64)
  BENCHMARK_GEMM(f16_gemm_6x8__neonfp16arith_ld64)
  BENCHMARK_GEMM(f16_gemm_8x8__neonfp16arith_ld64)
#endif

#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void f16_gemm_1x16__aarch64_neonfp16arith_ld32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f16_gemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, 1, 16, 1, 1);
  }

  static void f16_gemm_4x16__aarch64_neonfp16arith_ld32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f16_gemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, 4, 16, 1, 1);
  }

  static void f16_gemm_6x16__aarch64_neonfp16arith_ld32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f16_gemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, 6, 16, 1, 1);
  }

  static void f16_gemm_1x8__aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f16_gemm_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64, 1, 8, 1, 1);
  }

  static void f16_gemm_4x8__aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f16_gemm_minmax_ukernel_4x8__aarch64_neonfp16arith_ld64, 4, 8, 1, 1);
  }

  static void f16_gemm_6x8__aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f16_gemm_minmax_ukernel_6x8__aarch64_neonfp16arith_ld64, 6, 8, 1, 1);
  }

  static void f16_gemm_8x8__aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64, 8, 8, 1, 1);
  }

  BENCHMARK_GEMM(f16_gemm_1x16__aarch64_neonfp16arith_ld32)
  BENCHMARK_GEMM(f16_gemm_4x16__aarch64_neonfp16arith_ld32)
  BENCHMARK_GEMM(f16_gemm_6x16__aarch64_neonfp16arith_ld32)
  BENCHMARK_GEMM(f16_gemm_1x8__aarch64_neonfp16arith_ld64)
  BENCHMARK_GEMM(f16_gemm_4x8__aarch64_neonfp16arith_ld64)
  BENCHMARK_GEMM(f16_gemm_6x8__aarch64_neonfp16arith_ld64)
  BENCHMARK_GEMM(f16_gemm_8x8__aarch64_neonfp16arith_ld64)
#endif  // XNN_ARCH_ARM64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
