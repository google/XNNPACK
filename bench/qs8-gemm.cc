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
#ifdef BENCHMARK_RUY
#include "ruy/ruy.h"
#endif  // BENCHMARK_RUY
#include "bench/gemm.h"
#include "bench/utils.h"
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/common.h>
#include <xnnpack/gemm.h>
#include <xnnpack/pack.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>

static void GEMMBenchmark(benchmark::State& state,
  xnn_qs8_gemm_minmax_ukernel_function gemm,
  size_t mr, size_t nr, size_t kr, size_t sr,
  xnn_init_qs8_conv_minmax_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr,
  bool extended_weights = false)
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
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), std::ref(rng));
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(-std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max()), std::ref(rng));

  std::vector<int8_t> a(mc * kc + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::generate(a.begin(), a.end(), std::ref(i8rng));
  std::vector<int8_t> k(nc * kc);
  std::generate(k.begin(), k.end(), std::ref(i8rng));
  std::vector<int32_t> b(nc);
  std::generate(b.begin(), b.end(), std::ref(i32rng));

  const size_t w_element_size = extended_weights ? sizeof(int16_t) : sizeof(int8_t);
  const size_t w_size = nc_stride * sizeof(int32_t) + kc_stride * nc_stride * w_element_size;
  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(), w_size + c_elements * sizeof(int8_t));

  std::vector<char, AlignedAllocator<char, 64>> w(w_size * num_buffers);
  std::fill(w.begin(), w.end(), 0);
  const xnn_qs8_packing_params packing_params = { 127 };
  if (extended_weights) {
    xnn_pack_qs8_gemm_xw_goi_w(1 /* groups */, nc, kc, nr, kr, sr, k.data(), b.data(), w.data(), 0, &packing_params);
  } else {
    xnn_pack_qs8_gemm_goi_w(1 /* groups */, nc, kc, nr, kr, sr, k.data(), b.data(), w.data(), 0, &packing_params);
  }
  std::vector<int8_t> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), 0xA5);

  union xnn_qs8_conv_minmax_params quantization_params;
  init_params(&quantization_params, 0.75f, 127, -127, 126);

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
          w.data() + w_size * buffer_index + n * (kc_stride * w_element_size + sizeof(int32_t)),
          c.data() + (mc * buffer_index + m) * nc + n, nc * sizeof(int8_t), nr * sizeof(int8_t),
          &quantization_params);
      }
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["OPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}

#ifdef BENCHMARK_RUY
static void RuyBenchmark(benchmark::State& state, size_t threads)
{
  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), std::ref(rng));
  auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), std::ref(rng));

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      nc * (sizeof(int8_t) * (mc + kc) + sizeof(int32_t)));

  std::vector<int8_t> a(mc * kc);
  std::generate(a.begin(), a.end(), std::ref(u8rng));
  std::vector<int8_t> k(num_buffers * nc * kc);
  std::generate(k.begin(), k.end(), std::ref(u8rng));
  std::vector<int32_t> b(num_buffers * nc);
  std::generate(b.begin(), b.end(), std::ref(i32rng));
  std::vector<int8_t> c(num_buffers * nc * mc);
  std::fill(c.begin(), c.end(), std::nanf(""));

  // Note: context must be static to avoid the cost of re-creating it for each benchmark.
  static ruy::Context context;
  context.set_max_num_threads(threads);

  ruy::Matrix<int8_t> ruy_a;
  ruy::MakeSimpleLayout(nc, kc, ruy::Order::kRowMajor, ruy_a.mutable_layout());
  ruy_a.set_zero_point(127);
  ruy::Matrix<int8_t> ruy_b;
  ruy::MakeSimpleLayout(kc, mc, ruy::Order::kColMajor, ruy_b.mutable_layout());
  ruy_b.set_data(a.data());
  ruy_b.set_zero_point(127);
  ruy::Matrix<int8_t> ruy_c;
  ruy::MakeSimpleLayout(nc, mc, ruy::Order::kColMajor, ruy_c.mutable_layout());
  ruy_c.set_zero_point(127);

  ruy::MulParams<int32_t, int8_t> mul_params;
  mul_params.set_multiplier_fixedpoint(0x40000000);

  // ruy::Context uses deferred initialization, which affects percieved GEMM performance. Initialization happens during
  // the first GEMM calls, and per Benoit Jacob it takes up to ~250 milliseconds for performance to stabilize.
  // Thus, on the first benchmark, we compute GEMM for 500 milliseconds (to be safe) without recording performance, and
  // keep the ruy::Context object initialized (by being static) between subsequent benchmarks.
  static std::once_flag warmup;
  std::call_once(warmup, [&](){
    auto start = std::chrono::steady_clock::now();
    do {
      ruy_a.set_data(k.data());
      ruy_c.set_data(c.data());
      mul_params.set_bias(b.data());

      ruy::Mul(ruy_a, ruy_b, mul_params, &context, &ruy_c);
    } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < 0.5);
  });

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - K is not in cache (for any cache level)
    // - B is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(int8_t));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    ruy_a.set_data(k.data() + buffer_index * nc * kc);
    ruy_c.set_data(c.data() + buffer_index * mc * nc);
    mul_params.set_bias(b.data() + buffer_index * nc);

    ruy::Mul(ruy_a, ruy_b, mul_params, &context, &ruy_c);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["OPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}

static void ruy_st(benchmark::State& state, const char* net)
{
  RuyBenchmark(state, 1);
}
#endif  // BENCHMARK_RUY


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_gemm_1x8_gemmlowp__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_1x8__neon_mlal_lane, 1, 8, 1, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_2x8_gemmlowp__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_2x8__neon_mlal_lane, 2, 8, 1, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_3x8_gemmlowp__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_3x8__neon_mlal_lane, 3, 8, 1, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_4x8_gemmlowp__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x8__neon_mlal_lane, 4, 8, 1, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_6x8_gemmlowp__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_6x8__neon_mlal_lane, 6, 8, 1, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_1x16_gemmlowp__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_1x16__neon_mlal_lane, 1, 16, 1, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_2x16_gemmlowp__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_2x16__neon_mlal_lane, 2, 16, 1, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_3x16_gemmlowp__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_3x16__neon_mlal_lane, 3, 16, 1, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_4x16_gemmlowp__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x16__neon_mlal_lane, 4, 16, 1, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_6x16_gemmlowp__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_6x16__neon_mlal_lane, 6, 16, 1, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_1x8c2_gemmlowp__neon_mull_padal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_1x8c2__neon_mull_padal_dup, 1, 8, 2, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_2x8c2_gemmlowp__neon_mull_padal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_2x8c2__neon_mull_padal_dup, 2, 8, 2, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_3x8c2_gemmlowp__neon_mull_padal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_3x8c2__neon_mull_padal_dup, 3, 8, 2, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_4x8c2_gemmlowp__neon_mull_padal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x8c2__neon_mull_padal_dup, 4, 8, 2, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_1x16c2_gemmlowp__neon_mull_padal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_1x16c2__neon_mull_padal_dup, 1, 16, 2, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_2x16c2_gemmlowp__neon_mull_padal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_2x16c2__neon_mull_padal_dup, 2, 16, 2, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_3x16c2_gemmlowp__neon_mull_padal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_3x16c2__neon_mull_padal_dup, 3, 16, 2, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_4x16c2_gemmlowp__neon_mull_padal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x16c2__neon_mull_padal_dup, 4, 16, 2, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_1x8c2_gemmlowp__neon_mlal_padal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_1x8c2__neon_mlal_padal_dup, 1, 8, 2, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_2x8c2_gemmlowp__neon_mlal_padal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_2x8c2__neon_mlal_padal_dup, 2, 8, 2, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_3x8c2_gemmlowp__neon_mlal_padal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_3x8c2__neon_mlal_padal_dup, 3, 8, 2, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_4x8c2_gemmlowp__neon_mlal_padal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x8c2__neon_mlal_padal_dup, 4, 8, 2, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_1x16c2_gemmlowp__neon_mlal_padal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_1x16c2__neon_mlal_padal_dup, 1, 16, 2, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_2x16c2_gemmlowp__neon_mlal_padal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_2x16c2__neon_mlal_padal_dup, 2, 16, 2, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_3x16c2_gemmlowp__neon_mlal_padal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_3x16c2__neon_mlal_padal_dup, 3, 16, 2, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_4x16c2_gemmlowp__neon_mlal_padal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x16c2__neon_mlal_padal_dup, 4, 16, 2, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_1x8c8_gemmlowp__neon_mull_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_1x8c8__neon_mull_padal, 1, 8, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_2x8c8_gemmlowp__neon_mull_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_2x8c8__neon_mull_padal, 2, 8, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_3x8c8_gemmlowp__neon_mull_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_3x8c8__neon_mull_padal, 3, 8, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_4x8c8_gemmlowp__neon_mull_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x8c8__neon_mull_padal, 4, 8, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_1x16c8_gemmlowp__neon_mull_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_1x16c8__neon_mull_padal, 1, 16, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_2x16c8_gemmlowp__neon_mull_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_2x16c8__neon_mull_padal, 2, 16, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_3x16c8_gemmlowp__neon_mull_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_3x16c8__neon_mull_padal, 3, 16, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_4x16c8_gemmlowp__neon_mull_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x16c8__neon_mull_padal, 4, 16, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_1x8c8_gemmlowp__neon_mlal_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_1x8c8__neon_mlal_padal, 1, 8, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_2x8c8_gemmlowp__neon_mlal_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_2x8c8__neon_mlal_padal, 2, 8, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_3x8c8_gemmlowp__neon_mlal_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_3x8c8__neon_mlal_padal, 3, 8, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_4x8c8_gemmlowp__neon_mlal_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x8c8__neon_mlal_padal, 4, 8, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_1x16c8_gemmlowp__neon_mlal_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_1x16c8__neon_mlal_padal, 1, 16, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_2x16c8_gemmlowp__neon_mlal_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_2x16c8__neon_mlal_padal, 2, 16, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_3x16c8_gemmlowp__neon_mlal_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_3x16c8__neon_mlal_padal, 3, 16, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_4x16c8_gemmlowp__neon_mlal_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x16c8__neon_mlal_padal, 4, 16, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_1x8c16_gemmlowp__neon_mlal_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_1x8c16__neon_mlal_padal, 1, 8, 16, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_2x8c16_gemmlowp__neon_mlal_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_2x8c16__neon_mlal_padal, 2, 8, 16, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_3x8c16_gemmlowp__neon_mlal_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_3x8c16__neon_mlal_padal, 3, 8, 16, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_4x8c16_gemmlowp__neon_mlal_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x8c16__neon_mlal_padal, 4, 8, 16, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_1x16c16_gemmlowp__neon_mlal_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_1x16c16__neon_mlal_padal, 1, 16, 16, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_2x16c16_gemmlowp__neon_mlal_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_2x16c16__neon_mlal_padal, 2, 16, 16, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_3x16c16_gemmlowp__neon_mlal_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_3x16c16__neon_mlal_padal, 3, 16, 16, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_4x16c16_gemmlowp__neon_mlal_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x16c16__neon_mlal_padal, 4, 16, 16, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_1x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neondot, 1, 8, 4, 1,
      xnn_init_qs8_conv_minmax_rndnu_neon_params, benchmark::utils::CheckNEONDOT);
  }
  static void qs8_gemm_4x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neondot, 4, 8, 4, 1,
      xnn_init_qs8_conv_minmax_rndnu_neon_params, benchmark::utils::CheckNEONDOT);
  }
  static void qs8_gemm_6x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_rndnu_ukernel_6x8c4__neondot, 6, 8, 4, 1,
      xnn_init_qs8_conv_minmax_rndnu_neon_params, benchmark::utils::CheckNEONDOT);
  }
  static void qs8_gemm_8x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_rndnu_ukernel_8x8c4__neondot, 8, 8, 4, 1,
      xnn_init_qs8_conv_minmax_rndnu_neon_params, benchmark::utils::CheckNEONDOT);
  }
  static void qs8_gemm_1x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neondot, 1, 16, 4, 1,
      xnn_init_qs8_conv_minmax_rndnu_neon_params, benchmark::utils::CheckNEONDOT);
  }
  static void qs8_gemm_4x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neondot, 4, 16, 4, 1,
      xnn_init_qs8_conv_minmax_rndnu_neon_params, benchmark::utils::CheckNEONDOT);
  }
  static void qs8_gemm_6x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_rndnu_ukernel_6x16c4__neondot, 6, 16, 4, 1,
      xnn_init_qs8_conv_minmax_rndnu_neon_params, benchmark::utils::CheckNEONDOT);
  }
  static void qs8_gemm_8x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_rndnu_ukernel_8x16c4__neondot, 8, 16, 4, 1,
      xnn_init_qs8_conv_minmax_rndnu_neon_params, benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_GEMM(qs8_gemm_1x8_gemmlowp__neon_mlal_lane)
  BENCHMARK_GEMM(qs8_gemm_2x8_gemmlowp__neon_mlal_lane)
  BENCHMARK_GEMM(qs8_gemm_3x8_gemmlowp__neon_mlal_lane)
  BENCHMARK_GEMM(qs8_gemm_4x8_gemmlowp__neon_mlal_lane)
  BENCHMARK_GEMM(qs8_gemm_6x8_gemmlowp__neon_mlal_lane)
  BENCHMARK_GEMM(qs8_gemm_1x16_gemmlowp__neon_mlal_lane)
  BENCHMARK_GEMM(qs8_gemm_2x16_gemmlowp__neon_mlal_lane)
  BENCHMARK_GEMM(qs8_gemm_3x16_gemmlowp__neon_mlal_lane)
  BENCHMARK_GEMM(qs8_gemm_4x16_gemmlowp__neon_mlal_lane)
  BENCHMARK_GEMM(qs8_gemm_6x16_gemmlowp__neon_mlal_lane)
  BENCHMARK_GEMM(qs8_gemm_1x8c2_gemmlowp__neon_mull_padal_dup)
  BENCHMARK_GEMM(qs8_gemm_2x8c2_gemmlowp__neon_mull_padal_dup)
  BENCHMARK_GEMM(qs8_gemm_3x8c2_gemmlowp__neon_mull_padal_dup)
  BENCHMARK_GEMM(qs8_gemm_4x8c2_gemmlowp__neon_mull_padal_dup)
  BENCHMARK_GEMM(qs8_gemm_1x16c2_gemmlowp__neon_mull_padal_dup)
  BENCHMARK_GEMM(qs8_gemm_2x16c2_gemmlowp__neon_mull_padal_dup)
  BENCHMARK_GEMM(qs8_gemm_3x16c2_gemmlowp__neon_mull_padal_dup)
  BENCHMARK_GEMM(qs8_gemm_4x16c2_gemmlowp__neon_mull_padal_dup)
  BENCHMARK_GEMM(qs8_gemm_1x8c2_gemmlowp__neon_mlal_padal_dup)
  BENCHMARK_GEMM(qs8_gemm_2x8c2_gemmlowp__neon_mlal_padal_dup)
  BENCHMARK_GEMM(qs8_gemm_3x8c2_gemmlowp__neon_mlal_padal_dup)
  BENCHMARK_GEMM(qs8_gemm_4x8c2_gemmlowp__neon_mlal_padal_dup)
  BENCHMARK_GEMM(qs8_gemm_1x16c2_gemmlowp__neon_mlal_padal_dup)
  BENCHMARK_GEMM(qs8_gemm_2x16c2_gemmlowp__neon_mlal_padal_dup)
  BENCHMARK_GEMM(qs8_gemm_3x16c2_gemmlowp__neon_mlal_padal_dup)
  BENCHMARK_GEMM(qs8_gemm_4x16c2_gemmlowp__neon_mlal_padal_dup)
  BENCHMARK_GEMM(qs8_gemm_1x8c8_gemmlowp__neon_mull_padal)
  BENCHMARK_GEMM(qs8_gemm_2x8c8_gemmlowp__neon_mull_padal)
  BENCHMARK_GEMM(qs8_gemm_3x8c8_gemmlowp__neon_mull_padal)
  BENCHMARK_GEMM(qs8_gemm_4x8c8_gemmlowp__neon_mull_padal)
  BENCHMARK_GEMM(qs8_gemm_1x16c8_gemmlowp__neon_mull_padal)
  BENCHMARK_GEMM(qs8_gemm_2x16c8_gemmlowp__neon_mull_padal)
  BENCHMARK_GEMM(qs8_gemm_3x16c8_gemmlowp__neon_mull_padal)
  BENCHMARK_GEMM(qs8_gemm_4x16c8_gemmlowp__neon_mull_padal)
  BENCHMARK_GEMM(qs8_gemm_1x8c8_gemmlowp__neon_mlal_padal)
  BENCHMARK_GEMM(qs8_gemm_2x8c8_gemmlowp__neon_mlal_padal)
  BENCHMARK_GEMM(qs8_gemm_3x8c8_gemmlowp__neon_mlal_padal)
  BENCHMARK_GEMM(qs8_gemm_4x8c8_gemmlowp__neon_mlal_padal)
  BENCHMARK_GEMM(qs8_gemm_1x16c8_gemmlowp__neon_mlal_padal)
  BENCHMARK_GEMM(qs8_gemm_2x16c8_gemmlowp__neon_mlal_padal)
  BENCHMARK_GEMM(qs8_gemm_3x16c8_gemmlowp__neon_mlal_padal)
  BENCHMARK_GEMM(qs8_gemm_4x16c8_gemmlowp__neon_mlal_padal)
  BENCHMARK_GEMM(qs8_gemm_1x8c16_gemmlowp__neon_mlal_padal)
  BENCHMARK_GEMM(qs8_gemm_2x8c16_gemmlowp__neon_mlal_padal)
  BENCHMARK_GEMM(qs8_gemm_3x8c16_gemmlowp__neon_mlal_padal)
  BENCHMARK_GEMM(qs8_gemm_4x8c16_gemmlowp__neon_mlal_padal)
  BENCHMARK_GEMM(qs8_gemm_1x16c16_gemmlowp__neon_mlal_padal)
  BENCHMARK_GEMM(qs8_gemm_2x16c16_gemmlowp__neon_mlal_padal)
  BENCHMARK_GEMM(qs8_gemm_3x16c16_gemmlowp__neon_mlal_padal)
  BENCHMARK_GEMM(qs8_gemm_4x16c16_gemmlowp__neon_mlal_padal)

  BENCHMARK_GEMM(qs8_gemm_1x8c4__neondot)
  BENCHMARK_GEMM(qs8_gemm_4x8c4__neondot)
  BENCHMARK_GEMM(qs8_gemm_6x8c4__neondot)
  BENCHMARK_GEMM(qs8_gemm_8x8c4__neondot)
  BENCHMARK_GEMM(qs8_gemm_1x16c4__neondot)
  BENCHMARK_GEMM(qs8_gemm_4x16c4__neondot)
  BENCHMARK_GEMM(qs8_gemm_6x16c4__neondot)
  BENCHMARK_GEMM(qs8_gemm_8x16c4__neondot)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  static void qs8_gemm_4x16c4_gemmlowp__aarch64_neondot_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x16c4__aarch64_neondot_cortex_a55, 4, 16, 4, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEONDOT);
  }
  static void qs8_gemm_1x16c4_gemmlowp__aarch64_neondot_ld32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_1x16c4__aarch64_neondot_ld32, 1, 16, 4, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEONDOT);
  }
  static void qs8_gemm_1x16c4_gemmlowp__aarch64_neondot_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_1x16c4__aarch64_neondot_ld64, 1, 16, 4, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEONDOT);
  }
  static void qs8_gemm_4x16c4_gemmlowp__aarch64_neondot_ld32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x16c4__aarch64_neondot_ld32, 4, 16, 4, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEONDOT);
  }
  static void qs8_gemm_4x16c4_gemmlowp__aarch64_neondot_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x16c4__aarch64_neondot_ld64, 4, 16, 4, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEONDOT);
  }
  static void qs8_gemm_4x16c4_gemmlowp__aarch64_neondot_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x16c4__aarch64_neondot_ld128, 4, 16, 4, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEONDOT);
  }
  static void qs8_gemm_4x16_gemmlowp__aarch64_neon_mlal_lane_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53, 4, 16, 1, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_4x16_gemmlowp__aarch64_neon_mlal_lane_prfm_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a53, 4, 16, 1, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_1x8c8_gemmlowp__aarch64_neon_mlal_padal_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_1x8c8__aarch64_neon_mlal_padal_prfm, 1, 8, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_1x8c8_gemmlowp__aarch64_neon_mlal_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_1x8c8__aarch64_neon_mlal_padal, 1, 8, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_1x8c8_gemmlowp__aarch64_neon_mlal_padal_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_1x8c8__aarch64_neon_mlal_padal_cortex_a53, 1, 8, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_1x8c8_gemmlowp__aarch64_neon_mlal_padal_prfm_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_1x8c8__aarch64_neon_mlal_padal_prfm_cortex_a53, 1, 8, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_2x8c8_gemmlowp__aarch64_neon_mull_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_2x8c8__aarch64_neon_mull_padal, 2, 8, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_2x8c8_gemmlowp__aarch64_neon_mlal_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_2x8c8__aarch64_neon_mlal_padal, 2, 8, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_2x8c8_gemmlowp__aarch64_neon_mlal_padal_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_2x8c8__aarch64_neon_mlal_padal_prfm, 2, 8, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_2x8c8_gemmlowp__aarch64_neon_mlal_padal_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_2x8c8__aarch64_neon_mlal_padal_cortex_a53, 2, 8, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_2x8c8_gemmlowp__aarch64_neon_mlal_padal_prfm_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_2x8c8__aarch64_neon_mlal_padal_prfm_cortex_a53, 2, 8, 8, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_2x8c16_gemmlowp__aarch64_neon_mlal_padal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_2x8c16__aarch64_neon_mlal_padal, 2, 8, 16, 1,
      xnn_init_qs8_conv_minmax_gemmlowp_neon_params, benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(qs8_gemm_1x16c4_gemmlowp__aarch64_neondot_ld32)
  BENCHMARK_GEMM(qs8_gemm_1x16c4_gemmlowp__aarch64_neondot_ld64)
  BENCHMARK_GEMM(qs8_gemm_4x16c4_gemmlowp__aarch64_neondot_ld32)
  BENCHMARK_GEMM(qs8_gemm_4x16c4_gemmlowp__aarch64_neondot_ld64)
  BENCHMARK_GEMM(qs8_gemm_4x16c4_gemmlowp__aarch64_neondot_ld128)
  BENCHMARK_GEMM(qs8_gemm_4x16c4_gemmlowp__aarch64_neondot_cortex_a55)
  BENCHMARK_GEMM(qs8_gemm_4x16_gemmlowp__aarch64_neon_mlal_lane_cortex_a53)
  BENCHMARK_GEMM(qs8_gemm_4x16_gemmlowp__aarch64_neon_mlal_lane_prfm_cortex_a53)
  BENCHMARK_GEMM(qs8_gemm_1x8c8_gemmlowp__aarch64_neon_mlal_padal_prfm)
  BENCHMARK_GEMM(qs8_gemm_1x8c8_gemmlowp__aarch64_neon_mlal_padal)
  BENCHMARK_GEMM(qs8_gemm_1x8c8_gemmlowp__aarch64_neon_mlal_padal_prfm_cortex_a53)
  BENCHMARK_GEMM(qs8_gemm_1x8c8_gemmlowp__aarch64_neon_mlal_padal_cortex_a53)
  BENCHMARK_GEMM(qs8_gemm_2x8c8_gemmlowp__aarch64_neon_mull_padal)
  BENCHMARK_GEMM(qs8_gemm_2x8c8_gemmlowp__aarch64_neon_mlal_padal)
  BENCHMARK_GEMM(qs8_gemm_2x8c8_gemmlowp__aarch64_neon_mlal_padal_prfm)
  BENCHMARK_GEMM(qs8_gemm_2x8c8_gemmlowp__aarch64_neon_mlal_padal_cortex_a53)
  BENCHMARK_GEMM(qs8_gemm_2x8c8_gemmlowp__aarch64_neon_mlal_padal_prfm_cortex_a53)
  BENCHMARK_GEMM(qs8_gemm_2x8c16_gemmlowp__aarch64_neon_mlal_padal)
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qs8_gemm_2x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx, 2, 16, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_avx512_params, benchmark::utils::CheckAVX512SKX);
  }
  static void qs8_gemm_3x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx, 3, 16, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_avx512_params, benchmark::utils::CheckAVX512SKX);
  }
  static void qs8_gemm_4x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx, 4, 16, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_avx512_params, benchmark::utils::CheckAVX512SKX);
  }

  static void qs8_gemm_2x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x8c8__avx2, 2, 8, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_avx2_params, benchmark::utils::CheckAVX2);
  }
  static void qs8_gemm_3x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x8c8__avx2, 3, 8, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_avx2_params, benchmark::utils::CheckAVX2);
  }

  static void qs8_gemm_xw_2x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_2x8c8__avx2, 2, 8, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_avx2_params, benchmark::utils::CheckAVX2, true);
  }
  static void qs8_gemm_xw_3x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_3x8c8__avx2, 3, 8, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_avx2_params, benchmark::utils::CheckAVX2, true);
  }

  static void qs8_gemm_2x4c2__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x4c2__xop_ld64, 2, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckXOP);
  }
  static void qs8_gemm_3x4c2__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64, 3, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckXOP);
  }
  static void qs8_gemm_4x4c2__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_4x4c2__xop_ld64, 4, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckXOP);
  }

  static void qs8_gemm_2x4c2__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x4c2__xop_ld128, 2, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckXOP);
  }
  static void qs8_gemm_3x4c2__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x4c2__xop_ld128, 3, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckXOP);
  }
  static void qs8_gemm_4x4c2__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_4x4c2__xop_ld128, 4, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckXOP);
  }

  static void qs8_gemm_xw_2x4c2__xop(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_2x4c2__xop, 2, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckXOP, true);
  }
  static void qs8_gemm_xw_3x4c2__xop(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_3x4c2__xop, 3, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckXOP, true);
  }
  static void qs8_gemm_xw_4x4c2__xop(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_4x4c2__xop, 4, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckXOP, true);
  }

  static void qs8_gemm_2x4c8__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld64, 2, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckXOP);
  }
  static void qs8_gemm_3x4c8__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64, 3, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckXOP);
  }

  static void qs8_gemm_2x4c8__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128, 2, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckXOP);
  }
  static void qs8_gemm_3x4c8__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128, 3, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckXOP);
  }

  static void qs8_gemm_xw_2x4c8__xop(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_2x4c8__xop, 2, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckXOP, true);
  }
  static void qs8_gemm_xw_3x4c8__xop(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_3x4c8__xop, 3, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckXOP, true);
  }

  static void qs8_gemm_2x4c2__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x4c2__avx_ld64, 2, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckAVX);
  }
  static void qs8_gemm_3x4c2__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64, 3, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckAVX);
  }
  static void qs8_gemm_4x4c2__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_4x4c2__avx_ld64, 4, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckAVX);
  }

  static void qs8_gemm_2x4c2__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x4c2__avx_ld128, 2, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckAVX);
  }
  static void qs8_gemm_3x4c2__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x4c2__avx_ld128, 3, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckAVX);
  }
  static void qs8_gemm_4x4c2__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_4x4c2__avx_ld128, 4, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckAVX);
  }

  static void qs8_gemm_xw_2x4c2__avx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_2x4c2__avx, 2, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckAVX, true);
  }
  static void qs8_gemm_xw_3x4c2__avx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_3x4c2__avx, 3, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckAVX, true);
  }
  static void qs8_gemm_xw_4x4c2__avx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_4x4c2__avx, 4, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckAVX, true);
  }

  static void qs8_gemm_2x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld64, 2, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckAVX);
  }
  static void qs8_gemm_3x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64, 3, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckAVX);
  }

  static void qs8_gemm_2x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128, 2, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckAVX);
  }
  static void qs8_gemm_3x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x4c8__avx_ld128, 3, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckAVX);
  }

  static void qs8_gemm_xw_2x4c8__avx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_2x4c8__avx, 2, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckAVX, true);
  }
  static void qs8_gemm_xw_3x4c8__avx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_3x4c8__avx, 3, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckAVX, true);
  }

  static void qs8_gemm_2x4c2__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld64, 2, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckSSE41);
  }
  static void qs8_gemm_3x4c2__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld64, 3, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckSSE41);
  }
  static void qs8_gemm_4x4c2__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld64, 4, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckSSE41);
  }

  static void qs8_gemm_2x4c2__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128, 2, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckSSE41);
  }
  static void qs8_gemm_3x4c2__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128, 3, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckSSE41);
  }
  static void qs8_gemm_4x4c2__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld128, 4, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckSSE41);
  }

  static void qs8_gemm_xw_2x4c2__sse41(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_2x4c2__sse41, 2, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckSSE41, true);
  }
  static void qs8_gemm_xw_3x4c2__sse41(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_3x4c2__sse41, 3, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckSSE41, true);
  }
  static void qs8_gemm_xw_4x4c2__sse41(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_4x4c2__sse41, 4, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckSSE41, true);
  }

  static void qs8_gemm_2x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64, 2, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckSSE41);
  }
  static void qs8_gemm_3x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64, 3, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckSSE41);
  }

  static void qs8_gemm_2x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld128, 2, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckSSE41);
  }
  static void qs8_gemm_3x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld128, 3, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckSSE41);
  }

  static void qs8_gemm_xw_2x4c8__sse41(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_2x4c8__sse41, 2, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckSSE41, true);
  }
  static void qs8_gemm_xw_3x4c8__sse41(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_3x4c8__sse41, 3, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse4_params, benchmark::utils::CheckSSE41, true);
  }

  static void qs8_gemm_2x4c8__ssse3_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x4c8__ssse3_ld64, 2, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse2_params, benchmark::utils::CheckSSSE3);
  }
  static void qs8_gemm_3x4c8__ssse3_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x4c8__ssse3_ld64, 3, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse2_params, benchmark::utils::CheckSSSE3);
  }

  static void qs8_gemm_2x4c8__ssse3_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x4c8__ssse3_ld128, 2, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse2_params, benchmark::utils::CheckSSSE3);
  }
  static void qs8_gemm_3x4c8__ssse3_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x4c8__ssse3_ld128, 3, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse2_params, benchmark::utils::CheckSSSE3);
  }

  static void qs8_gemm_xw_2x4c8__ssse3(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_2x4c8__ssse3, 2, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse2_params, benchmark::utils::CheckSSSE3, true);
  }
  static void qs8_gemm_xw_3x4c8__ssse3(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_3x4c8__ssse3, 3, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse2_params, benchmark::utils::CheckSSSE3, true);
  }

  static void qs8_gemm_2x4c2__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld64, 2, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse2_params);
  }
  static void qs8_gemm_3x4c2__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64, 3, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse2_params);
  }
  static void qs8_gemm_4x4c2__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld64, 4, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse2_params);
  }

  static void qs8_gemm_2x4c2__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld128, 2, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse2_params);
  }
  static void qs8_gemm_3x4c2__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128, 3, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse2_params);
  }
  static void qs8_gemm_4x4c2__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld128, 4, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse2_params);
  }

  static void qs8_gemm_xw_2x4c2__sse2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_2x4c2__sse2, 2, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse2_params, nullptr, true);
  }
  static void qs8_gemm_xw_3x4c2__sse2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_3x4c2__sse2, 3, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse2_params, nullptr, true);
  }
  static void qs8_gemm_xw_4x4c2__sse2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_4x4c2__sse2, 4, 4, 2, 1,
      xnn_init_qs8_conv_minmax_fp32_sse2_params, nullptr, true);
  }

  static void qs8_gemm_2x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld64, 2, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse2_params);
  }
  static void qs8_gemm_3x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64, 3, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse2_params);
  }

  static void qs8_gemm_2x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128, 2, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse2_params);
  }
  static void qs8_gemm_3x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128, 3, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse2_params);
  }

  static void qs8_gemm_xw_2x4c8__sse2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_2x4c8__sse2, 2, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse2_params, nullptr, true);
  }
  static void qs8_gemm_xw_3x4c8__sse2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_3x4c8__sse2, 3, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_sse2_params, nullptr, true);
  }

  BENCHMARK_GEMM(qs8_gemm_2x16c8__avx512skx)
  BENCHMARK_GEMM(qs8_gemm_3x16c8__avx512skx)
  BENCHMARK_GEMM(qs8_gemm_4x16c8__avx512skx)

  BENCHMARK_GEMM(qs8_gemm_2x8c8__avx2)
  BENCHMARK_GEMM(qs8_gemm_3x8c8__avx2)
  BENCHMARK_GEMM(qs8_gemm_xw_2x8c8__avx2)
  BENCHMARK_GEMM(qs8_gemm_xw_3x8c8__avx2)

  BENCHMARK_GEMM(qs8_gemm_2x4c2__xop_ld64)
  BENCHMARK_GEMM(qs8_gemm_3x4c2__xop_ld64)
  BENCHMARK_GEMM(qs8_gemm_4x4c2__xop_ld64)
  BENCHMARK_GEMM(qs8_gemm_2x4c2__xop_ld128)
  BENCHMARK_GEMM(qs8_gemm_3x4c2__xop_ld128)
  BENCHMARK_GEMM(qs8_gemm_4x4c2__xop_ld128)
  BENCHMARK_GEMM(qs8_gemm_xw_2x4c2__xop)
  BENCHMARK_GEMM(qs8_gemm_xw_3x4c2__xop)
  BENCHMARK_GEMM(qs8_gemm_xw_4x4c2__xop)
  BENCHMARK_GEMM(qs8_gemm_2x4c8__xop_ld64)
  BENCHMARK_GEMM(qs8_gemm_3x4c8__xop_ld64)
  BENCHMARK_GEMM(qs8_gemm_2x4c8__xop_ld128)
  BENCHMARK_GEMM(qs8_gemm_3x4c8__xop_ld128)
  BENCHMARK_GEMM(qs8_gemm_xw_2x4c8__xop)
  BENCHMARK_GEMM(qs8_gemm_xw_3x4c8__xop)

  BENCHMARK_GEMM(qs8_gemm_2x4c2__avx_ld64)
  BENCHMARK_GEMM(qs8_gemm_3x4c2__avx_ld64)
  BENCHMARK_GEMM(qs8_gemm_4x4c2__avx_ld64)
  BENCHMARK_GEMM(qs8_gemm_2x4c2__avx_ld128)
  BENCHMARK_GEMM(qs8_gemm_3x4c2__avx_ld128)
  BENCHMARK_GEMM(qs8_gemm_4x4c2__avx_ld128)
  BENCHMARK_GEMM(qs8_gemm_xw_2x4c2__avx)
  BENCHMARK_GEMM(qs8_gemm_xw_3x4c2__avx)
  BENCHMARK_GEMM(qs8_gemm_xw_4x4c2__avx)
  BENCHMARK_GEMM(qs8_gemm_2x4c8__avx_ld64)
  BENCHMARK_GEMM(qs8_gemm_3x4c8__avx_ld64)
  BENCHMARK_GEMM(qs8_gemm_2x4c8__avx_ld128)
  BENCHMARK_GEMM(qs8_gemm_3x4c8__avx_ld128)
  BENCHMARK_GEMM(qs8_gemm_xw_2x4c8__avx)
  BENCHMARK_GEMM(qs8_gemm_xw_3x4c8__avx)

  BENCHMARK_GEMM(qs8_gemm_2x4c2__sse41_ld64)
  BENCHMARK_GEMM(qs8_gemm_3x4c2__sse41_ld64)
  BENCHMARK_GEMM(qs8_gemm_4x4c2__sse41_ld64)
  BENCHMARK_GEMM(qs8_gemm_2x4c2__sse41_ld128)
  BENCHMARK_GEMM(qs8_gemm_3x4c2__sse41_ld128)
  BENCHMARK_GEMM(qs8_gemm_4x4c2__sse41_ld128)
  BENCHMARK_GEMM(qs8_gemm_xw_2x4c2__sse41)
  BENCHMARK_GEMM(qs8_gemm_xw_3x4c2__sse41)
  BENCHMARK_GEMM(qs8_gemm_xw_4x4c2__sse41)
  BENCHMARK_GEMM(qs8_gemm_2x4c8__sse41_ld64)
  BENCHMARK_GEMM(qs8_gemm_3x4c8__sse41_ld64)
  BENCHMARK_GEMM(qs8_gemm_2x4c8__sse41_ld128)
  BENCHMARK_GEMM(qs8_gemm_3x4c8__sse41_ld128)
  BENCHMARK_GEMM(qs8_gemm_xw_2x4c8__sse41)
  BENCHMARK_GEMM(qs8_gemm_xw_3x4c8__sse41)

  BENCHMARK_GEMM(qs8_gemm_2x4c8__ssse3_ld64)
  BENCHMARK_GEMM(qs8_gemm_3x4c8__ssse3_ld64)
  BENCHMARK_GEMM(qs8_gemm_2x4c8__ssse3_ld128)
  BENCHMARK_GEMM(qs8_gemm_3x4c8__ssse3_ld128)
  BENCHMARK_GEMM(qs8_gemm_xw_2x4c8__ssse3)
  BENCHMARK_GEMM(qs8_gemm_xw_3x4c8__ssse3)

  BENCHMARK_GEMM(qs8_gemm_2x4c2__sse2_ld64)
  BENCHMARK_GEMM(qs8_gemm_3x4c2__sse2_ld64)
  BENCHMARK_GEMM(qs8_gemm_4x4c2__sse2_ld64)
  BENCHMARK_GEMM(qs8_gemm_2x4c2__sse2_ld128)
  BENCHMARK_GEMM(qs8_gemm_3x4c2__sse2_ld128)
  BENCHMARK_GEMM(qs8_gemm_4x4c2__sse2_ld128)
  BENCHMARK_GEMM(qs8_gemm_xw_2x4c2__sse2)
  BENCHMARK_GEMM(qs8_gemm_xw_3x4c2__sse2)
  BENCHMARK_GEMM(qs8_gemm_xw_4x4c2__sse2)
  BENCHMARK_GEMM(qs8_gemm_2x4c8__sse2_ld64)
  BENCHMARK_GEMM(qs8_gemm_3x4c8__sse2_ld64)
  BENCHMARK_GEMM(qs8_gemm_2x4c8__sse2_ld128)
  BENCHMARK_GEMM(qs8_gemm_3x4c8__sse2_ld128)
  BENCHMARK_GEMM(qs8_gemm_xw_2x4c8__sse2)
  BENCHMARK_GEMM(qs8_gemm_xw_3x4c8__sse2)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD
  static void qs8_gemm_2x4c8__wasmsimd_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_ld64, 2, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params);
  }
  static void qs8_gemm_3x4c8__wasmsimd_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_ld64, 3, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params);
  }

  static void qs8_gemm_2x4c8__wasmsimd_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_ld128, 2, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params);
  }
  static void qs8_gemm_3x4c8__wasmsimd_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_ld128, 3, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params);
  }

  static void qs8_gemm_xw_2x4c8__wasmsimd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_2x4c8__wasmsimd, 2, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params, nullptr, true);
  }
  static void qs8_gemm_xw_3x4c8__wasmsimd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_qs8_gemm_xw_minmax_fp32_ukernel_3x4c8__wasmsimd, 3, 4, 8, 1,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params, nullptr, true);
  }

  BENCHMARK_GEMM(qs8_gemm_2x4c8__wasmsimd_ld64)
  BENCHMARK_GEMM(qs8_gemm_3x4c8__wasmsimd_ld64)
  BENCHMARK_GEMM(qs8_gemm_2x4c8__wasmsimd_ld128)
  BENCHMARK_GEMM(qs8_gemm_3x4c8__wasmsimd_ld128)
  BENCHMARK_GEMM(qs8_gemm_xw_2x4c8__wasmsimd)
  BENCHMARK_GEMM(qs8_gemm_xw_3x4c8__wasmsimd)
#endif  // XNN_ARCH_WASMSIMD


static void qs8_gemm_2x2_gemmlowp__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_2x2__scalar, 2, 2, 1, 1,
    xnn_init_qs8_conv_minmax_gemmlowp_scalar_params);
}
static void qs8_gemm_3x2_gemmlowp__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_3x2__scalar, 3, 2, 1, 1,
    xnn_init_qs8_conv_minmax_gemmlowp_scalar_params);
}
static void qs8_gemm_4x2_gemmlowp__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x2__scalar, 4, 2, 1, 1,
    xnn_init_qs8_conv_minmax_gemmlowp_scalar_params);
}

static void qs8_gemm_2x4_gemmlowp__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_2x4__scalar, 2, 4, 1, 1,
    xnn_init_qs8_conv_minmax_gemmlowp_scalar_params);
}
static void qs8_gemm_3x4_gemmlowp__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_3x4__scalar, 3, 4, 1, 1,
    xnn_init_qs8_conv_minmax_gemmlowp_scalar_params);
}
static void qs8_gemm_4x4_gemmlowp__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state, xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x4__scalar, 4, 4, 1, 1,
    xnn_init_qs8_conv_minmax_gemmlowp_scalar_params);
}

BENCHMARK_GEMM(qs8_gemm_2x2_gemmlowp__scalar)
BENCHMARK_GEMM(qs8_gemm_3x2_gemmlowp__scalar)
BENCHMARK_GEMM(qs8_gemm_4x2_gemmlowp__scalar)

BENCHMARK_GEMM(qs8_gemm_2x4_gemmlowp__scalar)
BENCHMARK_GEMM(qs8_gemm_3x4_gemmlowp__scalar)
BENCHMARK_GEMM(qs8_gemm_4x4_gemmlowp__scalar)


#ifdef BENCHMARK_RUY
BENCHMARK_GEMM(ruy_st)
#endif  // BENCHMARK_RUY

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
