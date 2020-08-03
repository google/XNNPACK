// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <mutex>
#include <random>
#include <vector>

#include <cpuinfo.h>

#include <benchmark/benchmark.h>
#include "bench/gemm.h"
#include "bench/utils.h"
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/common.h>
#include <xnnpack/gemm.h>
#include <xnnpack/pack.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


static void GEMMBenchmark(benchmark::State& state,
  xnn_qs8_gemm_ukernel_function gemm,
  size_t mr, size_t nr, size_t kr, size_t sr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (!cpuinfo_initialize()) {
    state.SkipWithError("cpuinfo initialization failed");
    return;
  }
  if (isa_check && !isa_check(state)) {
    return;
  }

  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  const size_t nc_stride = benchmark::utils::RoundUp(nc, nr);
  const size_t kc_stride = benchmark::utils::RoundUp(kc, kr);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto s32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), std::ref(rng));
  auto s8rng = std::bind(
    std::uniform_int_distribution<uint32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()), std::ref(rng));

  std::vector<int8_t> a(mc * kc);
  std::generate(a.begin(), a.end(), std::ref(s8rng));
  std::vector<int8_t> k(nc * kc);
  std::generate(k.begin(), k.end(), std::ref(s8rng));
  std::vector<int32_t> b(nc);
  std::generate(b.begin(), b.end(), std::ref(s32rng));

  const size_t w_elements = kc_stride * nc_stride + nc_stride * sizeof(int32_t) / sizeof(int8_t);
  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(int8_t) * (w_elements + c_elements));

  std::vector<int8_t, AlignedAllocator<int8_t, 32>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0);
  const xnn_qs8_packing_params packing_params = { 127 };
  xnn_pack_qs8_gemm_goi_w(1 /* groups */, nc, kc, nr, kr, sr, k.data(), b.data(), w.data(), &packing_params);
  std::vector<int8_t> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), 0xA5);

  union xnn_qs8_gemm_params quantization_params = xnn_init_qs8_gemm_params(0.75f, 127, -127, 126);

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(int8_t));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      for (uint32_t n = 0; n < nc; n += nr) {
        const uint32_t nb = min(nc - n, nr);
        gemm(
          mb, nb, kc * sizeof(int8_t),
          a.data() + m * kc, kc * sizeof(int8_t),
          w.data() + (w_elements * buffer_index + n * (kc_stride + sizeof(int32_t))) / sizeof(int8_t),
          c.data() + (mc * buffer_index + m) * nc + n, nc * sizeof(int8_t), nr * sizeof(int8_t),
          &quantization_params);
      }
    }
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["OPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qs8_gemm_4x4c2__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_ukernel_4x4c2__sse2_ld64, 4, 4, 2, 1);
  }
  static void qs8_gemm_4x4c2__ssse3_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_ukernel_4x4c2__ssse3_ld64, 4, 4, 2, 1, benchmark::utils::CheckSSSE3);
  }
  static void qs8_gemm_4x4c2__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_ukernel_4x4c2__sse41_ld64, 4, 4, 2, 1, benchmark::utils::CheckSSE41);
  }
  static void qs8_gemm_4x4c2__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_ukernel_4x4c2__xop_ld64, 4, 4, 2, 1, benchmark::utils::CheckXOP);
  }

  static void qs8_gemm_4x4c2__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_ukernel_4x4c2__sse2_ld128, 4, 4, 2, 1);
  }
  static void qs8_gemm_4x4c2__ssse3_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_ukernel_4x4c2__ssse3_ld128, 4, 4, 2, 1, benchmark::utils::CheckSSSE3);
  }
  static void qs8_gemm_4x4c2__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_ukernel_4x4c2__sse41_ld128, 4, 4, 2, 1, benchmark::utils::CheckSSE41);
  }
  static void qs8_gemm_4x4c2__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_ukernel_4x4c2__xop_ld128, 4, 4, 2, 1, benchmark::utils::CheckXOP);
  }

  static void qs8_gemm_2x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_ukernel_2x4c8__sse2_ld64, 2, 4, 8, 1);
  }
  static void qs8_gemm_2x4c8__ssse3_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_ukernel_2x4c8__ssse3_ld64, 2, 4, 8, 1, benchmark::utils::CheckSSSE3);
  }
  static void qs8_gemm_2x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_ukernel_2x4c8__sse41_ld64, 2, 4, 8, 1, benchmark::utils::CheckSSE41);
  }
  static void qs8_gemm_2x4c8__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_ukernel_2x4c8__xop_ld64, 2, 4, 8, 1, benchmark::utils::CheckXOP);
  }

  static void qs8_gemm_2x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_ukernel_2x4c8__sse2_ld128, 2, 4, 8, 1);
  }
  static void qs8_gemm_2x4c8__ssse3_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_ukernel_2x4c8__ssse3_ld128, 2, 4, 8, 1, benchmark::utils::CheckSSSE3);
  }
  static void qs8_gemm_2x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_ukernel_2x4c8__sse41_ld128, 2, 4, 8, 1, benchmark::utils::CheckSSE41);
  }
  static void qs8_gemm_2x4c8__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_ukernel_2x4c8__xop_ld128, 2, 4, 8, 1, benchmark::utils::CheckXOP);
  }

  BENCHMARK_GEMM(qs8_gemm_4x4c2__sse2_ld64)
  BENCHMARK_GEMM(qs8_gemm_4x4c2__ssse3_ld64)
  BENCHMARK_GEMM(qs8_gemm_4x4c2__sse41_ld64)
  BENCHMARK_GEMM(qs8_gemm_4x4c2__xop_ld64)

  BENCHMARK_GEMM(qs8_gemm_4x4c2__sse2_ld128)
  BENCHMARK_GEMM(qs8_gemm_4x4c2__ssse3_ld128)
  BENCHMARK_GEMM(qs8_gemm_4x4c2__sse41_ld128)
  BENCHMARK_GEMM(qs8_gemm_4x4c2__xop_ld128)

  BENCHMARK_GEMM(qs8_gemm_2x4c8__sse2_ld64)
  BENCHMARK_GEMM(qs8_gemm_2x4c8__ssse3_ld64)
  BENCHMARK_GEMM(qs8_gemm_2x4c8__sse41_ld64)
  BENCHMARK_GEMM(qs8_gemm_2x4c8__xop_ld64)

  BENCHMARK_GEMM(qs8_gemm_2x4c8__sse2_ld128)
  BENCHMARK_GEMM(qs8_gemm_2x4c8__ssse3_ld128)
  BENCHMARK_GEMM(qs8_gemm_2x4c8__sse41_ld128)
  BENCHMARK_GEMM(qs8_gemm_2x4c8__xop_ld128)
#endif

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
