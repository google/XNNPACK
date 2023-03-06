// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <functional>
#include <mutex>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#ifdef BENCHMARK_RUY
#include "ruy/ruy.h"
#endif  // BENCHMARK_RUY
#include "bench/bgemm.h"
#include "bench/utils.h"


#ifdef BENCHMARK_RUY
static void RuyBenchmark(benchmark::State& state, uint32_t threads)
{
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  const size_t batch = state.range(0);
  const size_t dim_m = state.range(1);
  const size_t dim_n = state.range(2);
  const size_t dim_k = state.range(3);

  std::vector<float> a(batch * dim_m * dim_k);
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::vector<float> b(batch * dim_n * dim_k);
  std::generate(b.begin(), b.end(), std::ref(f32rng));
  std::vector<float> c(batch * dim_m * dim_n);
  std::fill(c.begin(), c.end(), std::nanf(""));

  // Note: context must be static to avoid the cost of re-creating it for each benchmark.
  static ruy::Context context;
  context.set_max_num_threads(threads);

  ruy::Matrix<float> ruy_a;
  ruy::MakeSimpleLayout(dim_m, dim_k, ruy::Order::kRowMajor, ruy_a.mutable_layout());
  ruy::Matrix<float> ruy_b;
  ruy::MakeSimpleLayout(dim_k, dim_n, ruy::Order::kColMajor, ruy_b.mutable_layout());
  ruy::Matrix<float> ruy_c;
  ruy::MakeSimpleLayout(dim_m, dim_n, ruy::Order::kRowMajor, ruy_c.mutable_layout());

  ruy::MulParams<float, float> mul_params;

  // ruy::Context uses deferred initialization, which affects percieved GEMM performance. Initialization happens during
  // the first GEMM calls, and per Benoit Jacob it takes up to ~250 milliseconds for performance to stabilize.
  // Thus, on the first benchmark, we compute GEMM for 500 milliseconds (to be safe) without recording performance, and
  // keep the ruy::Context object initialized (by being static) between subsequent benchmarks.
  static std::once_flag warmup;
  std::call_once(warmup, [&](){
    auto start = std::chrono::steady_clock::now();
    do {
      for (size_t i = 0; i < batch; i++) {
        ruy_a.set_data(a.data() + i * dim_m * dim_k);
        ruy_b.set_data(b.data() + i * dim_n * dim_k);
        ruy_c.set_data(c.data() + i * dim_m * dim_n);

        ruy::Mul(ruy_a, ruy_b, mul_params, &context, &ruy_c);
      }
    } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < 0.5);
  });

  for (auto _ : state) {
    for (size_t i = 0; i < batch; i++) {
      ruy_a.set_data(a.data() + i * dim_m * dim_k);
      ruy_b.set_data(b.data() + i * dim_n * dim_k);
      ruy_c.set_data(c.data() + i * dim_m * dim_n);

      ruy::Mul(ruy_a, ruy_b, mul_params, &context, &ruy_c);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * batch * dim_m * dim_n * dim_k, benchmark::Counter::kIsRate);
}

static void ruy_st(benchmark::State& state, const char* net)
{
  RuyBenchmark(state, 1);
}
#endif  // BENCHMARK_RUY


#ifdef BENCHMARK_RUY
BENCHMARK_BGEMM(ruy_st)
#endif  // BENCHMARK_RUY

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
