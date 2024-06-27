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

#include <benchmark/benchmark.h>
#include "bench/gemm-benchmark.h"
#include "bench/utils.h"
#ifdef BENCHMARK_RUY
#include "ruy/ruy.h"
#endif  // BENCHMARK_RUY

#include "xnnpack/isa-checks.h"
#include "xnnpack/gemm.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"


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
  ruy_b.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
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
BENCHMARK_GEMM(ruy_st)

#endif  // BENCHMARK_RUY

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
