// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
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
#ifdef BENCHMARK_RUY
#include "ruy/ruy.h"
#endif  // BENCHMARK_RUY
#include "bench/gemm.h"
#include "bench/utils.h"

#include "xnnpack/aligned-allocator.h"
#include "xnnpack/allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pack.h"
#include "xnnpack/packx.h"
#include "xnnpack/ppmm.h"


static void GEMMBenchmark(benchmark::State& state,
  xnn_f32_gemm_minmax_ukernel_fn gemm,
  xnn_init_f32_minmax_params_fn init_params,
  size_t mr, size_t nr, size_t kr, size_t sr,
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

  std::vector<float> a(mc * kc + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::vector<float> k(nc * kc);
  std::generate(k.begin(), k.end(), std::ref(f32rng));
  std::vector<float> b(nc);
  std::generate(b.begin(), b.end(), std::ref(f32rng));

  const size_t w_elements = nc_stride * kc_stride + nc_stride;
  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (w_elements + c_elements));

  std::vector<float, AlignedAllocator<float, 64>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0.0f);
  xnn_pack_f32_gemm_goi_w(/*groups=*/1, nc, kc, nr, kr, sr,
    k.data(), b.data(), /*scale=*/nullptr, w.data(), /*extra_bytes=*/0, /*params=*/nullptr);
  std::vector<float> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  xnn_f32_minmax_params params;
  init_params(&params,
    -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(float));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      gemm(
        mb, nc, kc * sizeof(float),
        a.data() + m * kc, kc * sizeof(float),
        w.data() + buffer_index * nc_stride * (kc_stride + 1),
        c.data() + (buffer_index * mc + m) * nc, nc * sizeof(float), nr * sizeof(float),
        &params);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}

static void GEMMGoiBenchmark(benchmark::State& state,
  xnn_f32_gemm_minmax_ukernel_fn gemm,
  xnn_init_f32_minmax_params_fn init_params,
  size_t mr, size_t nr, size_t kr, size_t sr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<float> a(mc * kc + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(a.begin(), a.end(), std::ref(f32rng));

  const size_t k_elements = nc * kc;
  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (k_elements + c_elements));

  std::vector<float> k(k_elements * num_buffers);
  std::vector<float> c(c_elements * num_buffers);
  std::generate(k.begin(), k.end(), std::ref(f32rng));
  std::fill(c.begin(), c.end(), std::nanf(""));

  xnn_f32_minmax_params params;
  init_params(&params,
    -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - K is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(float));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      gemm(
        mb, nc, kc * sizeof(float),
        a.data() + m * kc, kc * sizeof(float),
        k.data() + (buffer_index * k_elements),
        c.data() + (buffer_index * mc + m) * nc, nc * sizeof(float), nr * sizeof(float),
        &params);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}

static void PPMM1PBenchmark(benchmark::State& state,
  xnn_x32_packx_ukernel_fn packx,
  xnn_f32_ppmm_minmax_ukernel_fn ppmm,
  xnn_init_f32_minmax_params_fn init_params,
  size_t mr, size_t nr, size_t kr, size_t sr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  const size_t nc_stride = benchmark::utils::RoundUp(nc, nr);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<float> a(mc * kc + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::vector<float> k(nc * kc);
  std::generate(k.begin(), k.end(), std::ref(f32rng));
  std::vector<float> b(nc);
  std::generate(b.begin(), b.end(), std::ref(f32rng));

  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> t(mr * kc);

  const size_t w_elements = nc_stride * kc + nc_stride;
  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (w_elements + c_elements));

  std::vector<float, AlignedAllocator<float, 64>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0.0f);
  xnn_pack_f32_gemm_goi_w(/*groups=*/1, nc, kc, nr, /*kr=*/1, /*sr=*/1,
    k.data(), b.data(), /*scale=*/nullptr, w.data(), /*extra_bytes=*/0, /*params=*/nullptr);
  std::vector<float> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  xnn_f32_minmax_params params;
  init_params(&params,
    -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(float));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      packx(mb, kc, reinterpret_cast<const uint32_t*>(a.data() + m * kc), kc, t.data());
      ppmm(
        mb, nc, kc * sizeof(float),
        reinterpret_cast<const float*>(t.data()),
        w.data() + nc_stride * buffer_index * (kc + 1),
        c.data() + (mc * buffer_index + m) * nc, nc * sizeof(float), nr * sizeof(float),
        &params);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}

static void PPMM2PBenchmark(benchmark::State& state,
  xnn_x32_packx_ukernel_fn packx,
  xnn_f32_ppmm_minmax_ukernel_fn ppmm,
  xnn_init_f32_minmax_params_fn init_params,
  size_t mr, size_t nr, size_t kr, size_t sr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  const size_t mc_stride = benchmark::utils::RoundUp(mc, mr);
  const size_t nc_stride = benchmark::utils::RoundUp(nc, nr);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<float> a(mc * kc + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::vector<float> k(nc * kc);
  std::generate(k.begin(), k.end(), std::ref(f32rng));
  std::vector<float> b(nc);
  std::generate(b.begin(), b.end(), std::ref(f32rng));

  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> t(mc_stride * kc);

  const size_t w_elements = nc_stride * kc + nc_stride;
  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (w_elements + c_elements));

  std::vector<float, AlignedAllocator<float, 64>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0.0f);
  xnn_pack_f32_gemm_goi_w(/*groups=*/1, nc, kc, nr, /*kr=*/1, /*sr=*/1,
    k.data(), b.data(), /*scale=*/nullptr, w.data(), /*extra_bytes=*/0, /*params=*/nullptr);
  std::vector<float> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  xnn_f32_minmax_params params;
  init_params(&params,
    -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(float));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      packx(mb, kc, reinterpret_cast<const uint32_t*>(a.data() + m * kc), kc, t.data() + m * kc);
    }
    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      ppmm(
        mb, nc, kc * sizeof(float),
        reinterpret_cast<const float*>(t.data() + m * kc),
        w.data() + nc_stride * buffer_index * (kc + 1),
        c.data() + (mc * buffer_index + m) * nc, nc * sizeof(float), nr * sizeof(float),
        &params);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}

#ifdef BENCHMARK_RUY
static void RuyBenchmark(benchmark::State& state, uint32_t threads)
{
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (nc * (mc + kc + 1)));

  std::vector<float> a(mc * kc + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::vector<float> k(num_buffers * nc * kc);
  std::generate(k.begin(), k.end(), std::ref(f32rng));
  std::vector<float> b(num_buffers * nc);
  std::generate(b.begin(), b.end(), std::ref(f32rng));
  std::vector<float> c(num_buffers * nc * mc);
  std::fill(c.begin(), c.end(), std::nanf(""));

  // Note: context must be static to avoid the cost of re-creating it for each benchmark.
  static ruy::Context context;
  context.set_max_num_threads(threads);

  ruy::Matrix<float> ruy_a;
  ruy::MakeSimpleLayout(nc, kc, ruy::Order::kRowMajor, ruy_a.mutable_layout());
  ruy::Matrix<float> ruy_b;
  ruy::MakeSimpleLayout(kc, mc, ruy::Order::kColMajor, ruy_b.mutable_layout());
  ruy_b.set_data(a.data());
  ruy_b.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
  ruy::Matrix<float> ruy_c;
  ruy::MakeSimpleLayout(nc, mc, ruy::Order::kColMajor, ruy_c.mutable_layout());

  ruy::MulParams<float, float> mul_params;

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
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(float));
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

  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}

static void ruy_st(benchmark::State& state, const char* net)
{
  RuyBenchmark(state, 1);
}
#endif  // BENCHMARK_RUY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void f32_gemm_1x8__asm_aarch64_neon_ld128_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__asm_aarch64_neon_ld128_acc2_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__asm_aarch64_neonfma_ld64_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__asm_aarch64_neonfma_ld64_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__asm_aarch64_neonfma_ld64_acc2_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__asm_aarch64_neonfma_ld64_acc4(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__asm_aarch64_neonfma_ld64_acc4_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__asm_aarch64_neonfma_ld128_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__asm_aarch64_neonfma_ld128_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__asm_aarch64_neonfma_ld128_acc2_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__asm_aarch64_neonfma_ld128_acc4(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__asm_aarch64_neonfma_ld128_acc4_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x1__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/1, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x1__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/1, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x2__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x2__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x12__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x12__asm_aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/12, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__asm_aarch64_neonfma_cortex_a53_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x12__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x12__asm_aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/12, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_cortex_a53_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x2__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x2__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_cortex_a53_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_cortex_a73(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_4x8_unipass__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state,
      xnn_x32_packx_ukernel_4x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_4x8_twopass__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state,
      xnn_x32_packx_ukernel_4x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_4x8_unipass__asm_aarch64_neonfma_ld128_prfm(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state,
      xnn_x32_packx_ukernel_4x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_4x8_twopass__asm_aarch64_neonfma_ld128_prfm(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state,
      xnn_x32_packx_ukernel_4x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_4x8_unipass__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state,
      xnn_x32_packx_ukernel_4x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_4x8_twopass__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state,
      xnn_x32_packx_ukernel_4x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_4x8_unipass__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state,
      xnn_x32_packx_ukernel_4x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_4x8_twopass__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state,
      xnn_x32_packx_ukernel_4x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_8x8_unipass__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state,
      xnn_x32_packx_ukernel_8x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_8x8_twopass__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state,
      xnn_x32_packx_ukernel_8x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_8x8_unipass__asm_aarch64_neonfma_ld128_prfm(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state,
      xnn_x32_packx_ukernel_8x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_8x8_twopass__asm_aarch64_neonfma_ld128_prfm(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state,
      xnn_x32_packx_ukernel_8x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_8x8_unipass__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state,
      xnn_x32_packx_ukernel_8x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_8x8_twopass__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state,
      xnn_x32_packx_ukernel_8x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_8x8_unipass__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state,
      xnn_x32_packx_ukernel_8x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_8x8_twopass__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state,
      xnn_x32_packx_ukernel_8x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_goi_1x8__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    GEMMGoiBenchmark(state,
      xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_goi_1x8__asm_aarch64_neonfma_ld128_prfm(benchmark::State& state, const char* net) {
    GEMMGoiBenchmark(state,
      xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_goi_4x8__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    GEMMGoiBenchmark(state,
      xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }


  BENCHMARK_GEMM(f32_gemm_1x8__asm_aarch64_neon_ld128_acc2)
  BENCHMARK_GEMM(f32_gemm_1x8__asm_aarch64_neon_ld128_acc2_prfm)
  BENCHMARK_GEMM(f32_gemm_1x8__asm_aarch64_neonfma_ld64)
  BENCHMARK_GEMM(f32_gemm_1x8__asm_aarch64_neonfma_ld64_prfm)
  BENCHMARK_GEMM(f32_gemm_1x8__asm_aarch64_neonfma_ld64_acc2)
  BENCHMARK_GEMM(f32_gemm_1x8__asm_aarch64_neonfma_ld64_acc2_prfm)
  BENCHMARK_GEMM(f32_gemm_1x8__asm_aarch64_neonfma_ld64_acc4)
  BENCHMARK_GEMM(f32_gemm_1x8__asm_aarch64_neonfma_ld64_acc4_prfm)
  BENCHMARK_GEMM(f32_gemm_1x8__asm_aarch64_neonfma_ld128)
  BENCHMARK_GEMM(f32_gemm_1x8__asm_aarch64_neonfma_ld128_prfm)
  BENCHMARK_GEMM(f32_gemm_1x8__asm_aarch64_neonfma_ld128_acc2)
  BENCHMARK_GEMM(f32_gemm_1x8__asm_aarch64_neonfma_ld128_acc2_prfm)
  BENCHMARK_GEMM(f32_gemm_1x8__asm_aarch64_neonfma_ld128_acc4)
  BENCHMARK_GEMM(f32_gemm_1x8__asm_aarch64_neonfma_ld128_acc4_prfm)
  BENCHMARK_GEMM(f32_gemm_1x12__asm_aarch64_neonfma_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_1x8__asm_aarch64_neonfma_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_1x8__asm_aarch64_neonfma_cortex_a53_prfm)
  BENCHMARK_GEMM(f32_gemm_1x8__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_GEMM(f32_gemm_1x8__asm_aarch64_neonfma_cortex_a75_prfm)
  BENCHMARK_GEMM(f32_gemm_4x2__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_GEMM(f32_gemm_4x2__asm_aarch64_neonfma_cortex_a75_prfm)
  BENCHMARK_GEMM(f32_gemm_4x1__asm_aarch64_neonfma_ld64)
  BENCHMARK_GEMM(f32_gemm_4x1__asm_aarch64_neonfma_ld128)
  BENCHMARK_GEMM(f32_gemm_4x2__asm_aarch64_neonfma_ld64)
  BENCHMARK_GEMM(f32_gemm_4x2__asm_aarch64_neonfma_ld128)
  BENCHMARK_GEMM(f32_gemm_4x8__asm_aarch64_neonfma_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_4x8__asm_aarch64_neonfma_cortex_a53_prfm)
  BENCHMARK_GEMM(f32_gemm_4x8__asm_aarch64_neonfma_cortex_a55)
  BENCHMARK_GEMM(f32_gemm_4x8__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_GEMM(f32_gemm_4x8__asm_aarch64_neonfma_cortex_a75_prfm)
  BENCHMARK_GEMM(f32_gemm_4x8__asm_aarch64_neonfma_ld128)
  BENCHMARK_GEMM(f32_gemm_4x8__asm_aarch64_neonfma_ld64)
  BENCHMARK_GEMM(f32_gemm_4x12__asm_aarch64_neonfma_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_5x8__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_GEMM(f32_gemm_5x8__asm_aarch64_neonfma_cortex_a75_prfm)
  BENCHMARK_GEMM(f32_gemm_6x8__asm_aarch64_neonfma_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_6x8__asm_aarch64_neonfma_cortex_a53_prfm)
  BENCHMARK_GEMM(f32_gemm_6x8__asm_aarch64_neonfma_cortex_a55)
  BENCHMARK_GEMM(f32_gemm_6x8__asm_aarch64_neonfma_cortex_a73)
  BENCHMARK_GEMM(f32_gemm_6x8__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_GEMM(f32_gemm_6x8__asm_aarch64_neonfma_cortex_a75_prfm)
  BENCHMARK_GEMM(f32_gemm_6x8__asm_aarch64_neonfma_ld64)
  BENCHMARK_GEMM(f32_gemm_6x8__asm_aarch64_neonfma_ld128)
  BENCHMARK_GEMM(f32_ppmm_4x8_unipass__asm_aarch64_neonfma_ld128)
  BENCHMARK_GEMM(f32_ppmm_4x8_twopass__asm_aarch64_neonfma_ld128)
  BENCHMARK_GEMM(f32_ppmm_4x8_unipass__asm_aarch64_neonfma_ld128_prfm)
  BENCHMARK_GEMM(f32_ppmm_4x8_twopass__asm_aarch64_neonfma_ld128_prfm)
  BENCHMARK_GEMM(f32_ppmm_4x8_unipass__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_GEMM(f32_ppmm_4x8_twopass__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_GEMM(f32_ppmm_4x8_unipass__asm_aarch64_neonfma_cortex_a75_prfm)
  BENCHMARK_GEMM(f32_ppmm_4x8_twopass__asm_aarch64_neonfma_cortex_a75_prfm)
  BENCHMARK_GEMM(f32_ppmm_8x8_unipass__asm_aarch64_neonfma_ld128)
  BENCHMARK_GEMM(f32_ppmm_8x8_twopass__asm_aarch64_neonfma_ld128)
  BENCHMARK_GEMM(f32_ppmm_8x8_unipass__asm_aarch64_neonfma_ld128_prfm)
  BENCHMARK_GEMM(f32_ppmm_8x8_twopass__asm_aarch64_neonfma_ld128_prfm)
  BENCHMARK_GEMM(f32_ppmm_8x8_unipass__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_GEMM(f32_ppmm_8x8_twopass__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_GEMM(f32_ppmm_8x8_unipass__asm_aarch64_neonfma_cortex_a75_prfm)
  BENCHMARK_GEMM(f32_ppmm_8x8_twopass__asm_aarch64_neonfma_cortex_a75_prfm)
  BENCHMARK_GEMM(f32_gemm_goi_1x8__asm_aarch64_neonfma_ld128)
  BENCHMARK_GEMM(f32_gemm_goi_1x8__asm_aarch64_neonfma_ld128_prfm)
  BENCHMARK_GEMM(f32_gemm_goi_4x8__asm_aarch64_neonfma_ld128)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY

#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void f32_gemm_4x4__asm_aarch32_vfp_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x4__asm_aarch32_vfp_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckVFP);
  }

  static void f32_gemm_4x8__asm_aarch32_neon_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__asm_aarch32_neon_cortex_a7(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__asm_aarch32_neon_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__asm_aarch32_neon_cortex_a53_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__asm_aarch32_neon_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__asm_aarch32_neon_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__asm_aarch32_neon_cortex_a75_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_1x8__asm_aarch32_neon_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_1x8__asm_aarch32_neon_cortex_a53_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_4x4__asm_aarch32_vfp_ld64)
  BENCHMARK_GEMM(f32_gemm_4x8__asm_aarch32_neon_ld64)
  BENCHMARK_GEMM(f32_gemm_4x8__asm_aarch32_neon_cortex_a7)
  BENCHMARK_GEMM(f32_gemm_4x8__asm_aarch32_neon_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_4x8__asm_aarch32_neon_cortex_a53_prfm)
  BENCHMARK_GEMM(f32_gemm_4x8__asm_aarch32_neon_cortex_a55)
  BENCHMARK_GEMM(f32_gemm_4x8__asm_aarch32_neon_cortex_a75)
  BENCHMARK_GEMM(f32_gemm_4x8__asm_aarch32_neon_cortex_a75_prfm)
  BENCHMARK_GEMM(f32_gemm_1x8__asm_aarch32_neon_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_1x8__asm_aarch32_neon_cortex_a53_prfm)
#endif  // XNN_ARCH_ARM

#if XNN_ARCH_ARM64
  static void f32_gemm_1x8__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x2__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x2__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/2, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x16__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x16__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_4x8_unipass__aarch64_neonfma(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state,
      xnn_x32_packx_ukernel_4x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_4x8_twopass__aarch64_neonfma(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state,
      xnn_x32_packx_ukernel_4x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_8x8_unipass__aarch64_neonfma(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state,
      xnn_x32_packx_ukernel_8x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_8x8_twopass__aarch64_neonfma(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state,
      xnn_x32_packx_ukernel_8x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_4x8_unipass__aarch64_neonfma_prfm(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state,
      xnn_x32_packx_ukernel_4x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_4x8_twopass__aarch64_neonfma_prfm(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state,
      xnn_x32_packx_ukernel_4x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_8x8_unipass__aarch64_neonfma_prfm(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state,
      xnn_x32_packx_ukernel_8x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_8x8_twopass__aarch64_neonfma_prfm(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state,
      xnn_x32_packx_ukernel_8x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_4x16_unipass__aarch64_neonfma(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state,
      xnn_x32_packx_ukernel_4x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_4x16_twopass__aarch64_neonfma(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state,
      xnn_x32_packx_ukernel_4x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_4x16_unipass__aarch64_neonfma_prfm(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state,
      xnn_x32_packx_ukernel_4x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_4x16_twopass__aarch64_neonfma_prfm(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state,
      xnn_x32_packx_ukernel_4x__neon_st4_x8,
      xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1);
  }

  BENCHMARK_GEMM(f32_gemm_1x8__aarch64_neonfma_lane_ld64)
  BENCHMARK_GEMM(f32_gemm_4x2__aarch64_neonfma_lane_ld64)
  BENCHMARK_GEMM(f32_gemm_6x2__aarch64_neonfma_lane_ld64)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch64_neonfma_lane_ld64)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch64_neonfma_lane_ld128)
  BENCHMARK_GEMM(f32_gemm_5x8__aarch64_neonfma_lane_ld64)
  BENCHMARK_GEMM(f32_gemm_6x8__aarch64_neonfma_lane_ld64)
  BENCHMARK_GEMM(f32_gemm_6x8__aarch64_neonfma_lane_ld128)
  BENCHMARK_GEMM(f32_gemm_1x16__aarch64_neonfma_lane_ld128)
  BENCHMARK_GEMM(f32_gemm_4x16__aarch64_neonfma_lane_ld128)

  BENCHMARK_GEMM(f32_ppmm_4x8_unipass__aarch64_neonfma)
  BENCHMARK_GEMM(f32_ppmm_4x8_twopass__aarch64_neonfma)
  BENCHMARK_GEMM(f32_ppmm_8x8_unipass__aarch64_neonfma)
  BENCHMARK_GEMM(f32_ppmm_8x8_twopass__aarch64_neonfma)
  BENCHMARK_GEMM(f32_ppmm_4x8_unipass__aarch64_neonfma_prfm)
  BENCHMARK_GEMM(f32_ppmm_4x8_twopass__aarch64_neonfma_prfm)
  BENCHMARK_GEMM(f32_ppmm_8x8_unipass__aarch64_neonfma_prfm)
  BENCHMARK_GEMM(f32_ppmm_8x8_twopass__aarch64_neonfma_prfm)
  BENCHMARK_GEMM(f32_ppmm_4x16_unipass__aarch64_neonfma)
  BENCHMARK_GEMM(f32_ppmm_4x16_twopass__aarch64_neonfma)
  BENCHMARK_GEMM(f32_ppmm_4x16_unipass__aarch64_neonfma_prfm)
  BENCHMARK_GEMM(f32_ppmm_4x16_twopass__aarch64_neonfma_prfm)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_gemm_1x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x2__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x2__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_6x2__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x2__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__neon_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_5x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_6x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_6x8__neon_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__neon_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_1x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__neonfma_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_4x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__neonfma_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_4x8__neonfma_dup_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__neonfma_dup_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_6x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__neonfma_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_6x8__neonfma_dup_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__neonfma_dup_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_1x8s4__neon(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8s4__neon,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_1x8s4__neonfma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8s4__neonfma,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_4x8s4__neon(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8s4__neon,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8s4__neonfma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8s4__neonfma,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_6x8s4__neon(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8s4__neon,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_6x8s4__neonfma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8s4__neonfma,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_8x8s4__neon(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_8x8s4__neon,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_8x8s4__neonfma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_8x8s4__neonfma,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_1x8__neon_lane_ld64)
  BENCHMARK_GEMM(f32_gemm_4x2__neon_lane_ld64)
  BENCHMARK_GEMM(f32_gemm_6x2__neon_lane_ld64)
  BENCHMARK_GEMM(f32_gemm_4x8__neon_lane_ld64)
  BENCHMARK_GEMM(f32_gemm_4x8__neon_lane_ld128)
  BENCHMARK_GEMM(f32_gemm_5x8__neon_lane_ld64)
  BENCHMARK_GEMM(f32_gemm_6x8__neon_lane_ld64)
  BENCHMARK_GEMM(f32_gemm_6x8__neon_lane_ld128)

  BENCHMARK_GEMM(f32_gemm_1x8__neonfma_dup_ld64)
  BENCHMARK_GEMM(f32_gemm_4x8__neonfma_dup_ld64)
  BENCHMARK_GEMM(f32_gemm_4x8__neonfma_dup_ld128)
  BENCHMARK_GEMM(f32_gemm_6x8__neonfma_dup_ld64)
  BENCHMARK_GEMM(f32_gemm_6x8__neonfma_dup_ld128)

  BENCHMARK_GEMM(f32_gemm_1x8s4__neon)
  BENCHMARK_GEMM(f32_gemm_4x8s4__neon)
  BENCHMARK_GEMM(f32_gemm_6x8s4__neon)
  BENCHMARK_GEMM(f32_gemm_8x8s4__neon)

  BENCHMARK_GEMM(f32_gemm_1x8s4__neonfma)
  BENCHMARK_GEMM(f32_gemm_4x8s4__neonfma)
  BENCHMARK_GEMM(f32_gemm_6x8s4__neonfma)
  BENCHMARK_GEMM(f32_gemm_8x8s4__neonfma)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_gemm_1x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }
  static void f32_gemm_4x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }
  static void f32_gemm_5x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }
  static void f32_gemm_6x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }
  static void f32_gemm_7x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_7x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/7, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }
  static void f32_gemm_8x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_8x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }

  static void f32_gemm_1x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_4x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_5x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_6x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_7x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_7x8__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/7, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_8x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_8x8__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_1x16__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x16__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_3x16__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x16__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_4x16__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x16__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_5x16__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x16__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_6x16__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x16__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_1x16s4__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x16s4__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_3x16s4__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x16s4__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_4x16s4__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x16s4__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_5x16s4__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x16s4__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_6x16s4__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x16s4__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_1x8__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__avx_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }
  static void f32_gemm_4x8__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__avx_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }
  static void f32_gemm_5x8__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__avx_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }
  static void f32_gemm_6x8__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__avx_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }
  static void f32_gemm_7x8__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_7x8__avx_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/7, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }
  static void f32_gemm_1x16__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x16__avx_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }
  static void f32_gemm_3x16__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x16__avx_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }
  static void f32_gemm_4x16__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x16__avx_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }
  static void f32_gemm_5x16__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x16__avx_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }
  static void f32_gemm_6x16__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x16__avx_broadcast,
      xnn_init_f32_minmax_avx_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  static void f32_gemm_1x8__sse_load1(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__sse_load1,
      xnn_init_f32_minmax_sse_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_3x8__sse_load1(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8__sse_load1,
      xnn_init_f32_minmax_sse_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__sse_load1(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__sse_load1,
      xnn_init_f32_minmax_sse_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__sse_load1(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__sse_load1,
      xnn_init_f32_minmax_sse_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__sse_load1(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__sse_load1,
      xnn_init_f32_minmax_sse_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__sse_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__sse_dup,
      xnn_init_f32_minmax_sse_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_3x8__sse_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8__sse_dup,
      xnn_init_f32_minmax_sse_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__sse_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__sse_dup,
      xnn_init_f32_minmax_sse_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__sse_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__sse_dup,
      xnn_init_f32_minmax_sse_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__sse_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__sse_dup,
      xnn_init_f32_minmax_sse_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8s4__sse(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8s4__sse,
      xnn_init_f32_minmax_sse_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_3x8s4__sse(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8s4__sse,
      xnn_init_f32_minmax_sse_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_4x8s4__sse(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8s4__sse,
      xnn_init_f32_minmax_sse_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_5x8s4__sse(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8s4__sse,
      xnn_init_f32_minmax_sse_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_6x8s4__sse(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8s4__sse,
      xnn_init_f32_minmax_sse_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_ppmm_4x8_unipass__sse(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state,
      xnn_x32_packx_ukernel_4x__sse,
      xnn_f32_ppmm_minmax_ukernel_4x8__sse,
      xnn_init_f32_minmax_sse_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_4x8_twopass__sse(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state,
      xnn_x32_packx_ukernel_4x__sse,
      xnn_f32_ppmm_minmax_ukernel_4x8__sse,
      xnn_init_f32_minmax_sse_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  BENCHMARK_GEMM(f32_gemm_1x16__avx512f_broadcast)
  BENCHMARK_GEMM(f32_gemm_4x16__avx512f_broadcast)
  BENCHMARK_GEMM(f32_gemm_5x16__avx512f_broadcast)
  BENCHMARK_GEMM(f32_gemm_6x16__avx512f_broadcast)
  BENCHMARK_GEMM(f32_gemm_7x16__avx512f_broadcast)
  BENCHMARK_GEMM(f32_gemm_8x16__avx512f_broadcast)

  BENCHMARK_GEMM(f32_gemm_1x8__fma3_broadcast)
  BENCHMARK_GEMM(f32_gemm_4x8__fma3_broadcast)
  BENCHMARK_GEMM(f32_gemm_5x8__fma3_broadcast)
  BENCHMARK_GEMM(f32_gemm_6x8__fma3_broadcast)
  BENCHMARK_GEMM(f32_gemm_7x8__fma3_broadcast)
  BENCHMARK_GEMM(f32_gemm_8x8__fma3_broadcast)
  BENCHMARK_GEMM(f32_gemm_1x16__fma3_broadcast)
  BENCHMARK_GEMM(f32_gemm_3x16__fma3_broadcast)
  BENCHMARK_GEMM(f32_gemm_4x16__fma3_broadcast)
  BENCHMARK_GEMM(f32_gemm_5x16__fma3_broadcast)
  BENCHMARK_GEMM(f32_gemm_6x16__fma3_broadcast)

  BENCHMARK_GEMM(f32_gemm_1x16s4__fma3_broadcast)
  BENCHMARK_GEMM(f32_gemm_3x16s4__fma3_broadcast)
  BENCHMARK_GEMM(f32_gemm_4x16s4__fma3_broadcast)
  BENCHMARK_GEMM(f32_gemm_5x16s4__fma3_broadcast)
  BENCHMARK_GEMM(f32_gemm_6x16s4__fma3_broadcast)

  BENCHMARK_GEMM(f32_gemm_1x8__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_4x8__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_5x8__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_6x8__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_7x8__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_1x16__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_3x16__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_4x16__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_5x16__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_6x16__avx_broadcast)

  BENCHMARK_GEMM(f32_gemm_1x8__sse_load1)
  BENCHMARK_GEMM(f32_gemm_3x8__sse_load1)
  BENCHMARK_GEMM(f32_gemm_4x8__sse_load1)
  BENCHMARK_GEMM(f32_gemm_5x8__sse_load1)
  BENCHMARK_GEMM(f32_gemm_6x8__sse_load1)

  BENCHMARK_GEMM(f32_gemm_1x8__sse_dup)
  BENCHMARK_GEMM(f32_gemm_3x8__sse_dup)
  BENCHMARK_GEMM(f32_gemm_4x8__sse_dup)
  BENCHMARK_GEMM(f32_gemm_5x8__sse_dup)
  BENCHMARK_GEMM(f32_gemm_6x8__sse_dup)

  BENCHMARK_GEMM(f32_gemm_1x8s4__sse)
  BENCHMARK_GEMM(f32_gemm_3x8s4__sse)
  BENCHMARK_GEMM(f32_gemm_4x8s4__sse)
  BENCHMARK_GEMM(f32_gemm_5x8s4__sse)
  BENCHMARK_GEMM(f32_gemm_6x8s4__sse)

  BENCHMARK_GEMM(f32_ppmm_4x8_unipass__sse)
  BENCHMARK_GEMM(f32_ppmm_4x8_twopass__sse)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_gemm_1x8__wasmrelaxedsimd_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_3x8__wasmrelaxedsimd_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__wasmrelaxedsimd_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__wasmrelaxedsimd_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__wasmrelaxedsimd_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__wasmrelaxedsimd_fma_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_3x8__wasmrelaxedsimd_fma_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__wasmrelaxedsimd_fma_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__wasmrelaxedsimd_fma_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__wasmrelaxedsimd_fma_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__wasmrelaxedsimd_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_3x8__wasmrelaxedsimd_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__wasmrelaxedsimd_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__wasmrelaxedsimd_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__wasmrelaxedsimd_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__wasmrelaxedsimd_fma_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_3x8__wasmrelaxedsimd_fma_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__wasmrelaxedsimd_fma_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__wasmrelaxedsimd_fma_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__wasmrelaxedsimd_fma_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8s4__wasmrelaxedsimd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_3x8s4__wasmrelaxedsimd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_4x8s4__wasmrelaxedsimd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_5x8s4__wasmrelaxedsimd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_6x8s4__wasmrelaxedsimd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_1x8s4__wasmrelaxedsimd_fma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_3x8s4__wasmrelaxedsimd_fma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd_fma,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_4x8s4__wasmrelaxedsimd_fma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd_fma,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_5x8s4__wasmrelaxedsimd_fma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd_fma,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_6x8s4__wasmrelaxedsimd_fma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd_fma,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }

  BENCHMARK_GEMM(f32_gemm_1x8__wasmrelaxedsimd_loadsplat)
  BENCHMARK_GEMM(f32_gemm_3x8__wasmrelaxedsimd_loadsplat)
  BENCHMARK_GEMM(f32_gemm_4x8__wasmrelaxedsimd_loadsplat)
  BENCHMARK_GEMM(f32_gemm_5x8__wasmrelaxedsimd_loadsplat)
  BENCHMARK_GEMM(f32_gemm_6x8__wasmrelaxedsimd_loadsplat)

  BENCHMARK_GEMM(f32_gemm_1x8__wasmrelaxedsimd_fma_loadsplat)
  BENCHMARK_GEMM(f32_gemm_3x8__wasmrelaxedsimd_fma_loadsplat)
  BENCHMARK_GEMM(f32_gemm_4x8__wasmrelaxedsimd_fma_loadsplat)
  BENCHMARK_GEMM(f32_gemm_5x8__wasmrelaxedsimd_fma_loadsplat)
  BENCHMARK_GEMM(f32_gemm_6x8__wasmrelaxedsimd_fma_loadsplat)

  BENCHMARK_GEMM(f32_gemm_1x8__wasmrelaxedsimd_splat)
  BENCHMARK_GEMM(f32_gemm_3x8__wasmrelaxedsimd_splat)
  BENCHMARK_GEMM(f32_gemm_4x8__wasmrelaxedsimd_splat)
  BENCHMARK_GEMM(f32_gemm_5x8__wasmrelaxedsimd_splat)
  BENCHMARK_GEMM(f32_gemm_6x8__wasmrelaxedsimd_splat)

  BENCHMARK_GEMM(f32_gemm_1x8__wasmrelaxedsimd_fma_splat)
  BENCHMARK_GEMM(f32_gemm_3x8__wasmrelaxedsimd_fma_splat)
  BENCHMARK_GEMM(f32_gemm_4x8__wasmrelaxedsimd_fma_splat)
  BENCHMARK_GEMM(f32_gemm_5x8__wasmrelaxedsimd_fma_splat)
  BENCHMARK_GEMM(f32_gemm_6x8__wasmrelaxedsimd_fma_splat)

  BENCHMARK_GEMM(f32_gemm_1x8s4__wasmrelaxedsimd)
  BENCHMARK_GEMM(f32_gemm_3x8s4__wasmrelaxedsimd)
  BENCHMARK_GEMM(f32_gemm_4x8s4__wasmrelaxedsimd)
  BENCHMARK_GEMM(f32_gemm_5x8s4__wasmrelaxedsimd)
  BENCHMARK_GEMM(f32_gemm_6x8s4__wasmrelaxedsimd)

  BENCHMARK_GEMM(f32_gemm_1x8s4__wasmrelaxedsimd_fma)
  BENCHMARK_GEMM(f32_gemm_3x8s4__wasmrelaxedsimd_fma)
  BENCHMARK_GEMM(f32_gemm_4x8s4__wasmrelaxedsimd_fma)
  BENCHMARK_GEMM(f32_gemm_5x8s4__wasmrelaxedsimd_fma)
  BENCHMARK_GEMM(f32_gemm_6x8s4__wasmrelaxedsimd_fma)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_gemm_1x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_3x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_3x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_3x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_3x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_1x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_3x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_4x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_5x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_6x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_1x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_3x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_3x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_4x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_4x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_5x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_5x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_6x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_f32_gemm_minmax_ukernel_6x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }

  static void f32_ppmm_4x8_unipass__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state,
      xnn_x32_packx_ukernel_4x__wasmsimd,
      xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_4x8_unipass__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state,
      xnn_x32_packx_ukernel_4x__wasmsimd,
      xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void f32_ppmm_4x8_twopass__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state,
      xnn_x32_packx_ukernel_4x__wasmsimd,
      xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_ppmm_4x8_twopass__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state,
      xnn_x32_packx_ukernel_4x__wasmsimd,
      xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  BENCHMARK_GEMM(f32_gemm_1x8__wasmsimd_arm_loadsplat)
  BENCHMARK_GEMM(f32_gemm_3x8__wasmsimd_arm_loadsplat)
  BENCHMARK_GEMM(f32_gemm_4x8__wasmsimd_arm_loadsplat)
  BENCHMARK_GEMM(f32_gemm_5x8__wasmsimd_arm_loadsplat)
  BENCHMARK_GEMM(f32_gemm_6x8__wasmsimd_arm_loadsplat)

  BENCHMARK_GEMM(f32_gemm_1x8__wasmsimd_x86_loadsplat)
  BENCHMARK_GEMM(f32_gemm_3x8__wasmsimd_x86_loadsplat)
  BENCHMARK_GEMM(f32_gemm_4x8__wasmsimd_x86_loadsplat)
  BENCHMARK_GEMM(f32_gemm_5x8__wasmsimd_x86_loadsplat)
  BENCHMARK_GEMM(f32_gemm_6x8__wasmsimd_x86_loadsplat)

  BENCHMARK_GEMM(f32_gemm_1x8__wasmsimd_arm_splat)
  BENCHMARK_GEMM(f32_gemm_3x8__wasmsimd_arm_splat)
  BENCHMARK_GEMM(f32_gemm_4x8__wasmsimd_arm_splat)
  BENCHMARK_GEMM(f32_gemm_5x8__wasmsimd_arm_splat)
  BENCHMARK_GEMM(f32_gemm_6x8__wasmsimd_arm_splat)

  BENCHMARK_GEMM(f32_gemm_1x8__wasmsimd_x86_splat)
  BENCHMARK_GEMM(f32_gemm_3x8__wasmsimd_x86_splat)
  BENCHMARK_GEMM(f32_gemm_4x8__wasmsimd_x86_splat)
  BENCHMARK_GEMM(f32_gemm_5x8__wasmsimd_x86_splat)
  BENCHMARK_GEMM(f32_gemm_6x8__wasmsimd_x86_splat)

  BENCHMARK_GEMM(f32_gemm_1x8s4__wasmsimd_arm)
  BENCHMARK_GEMM(f32_gemm_3x8s4__wasmsimd_arm)
  BENCHMARK_GEMM(f32_gemm_4x8s4__wasmsimd_arm)
  BENCHMARK_GEMM(f32_gemm_5x8s4__wasmsimd_arm)
  BENCHMARK_GEMM(f32_gemm_6x8s4__wasmsimd_arm)

  BENCHMARK_GEMM(f32_gemm_1x8s4__wasmsimd_x86)
  BENCHMARK_GEMM(f32_gemm_3x8s4__wasmsimd_x86)
  BENCHMARK_GEMM(f32_gemm_4x8s4__wasmsimd_x86)
  BENCHMARK_GEMM(f32_gemm_5x8s4__wasmsimd_x86)
  BENCHMARK_GEMM(f32_gemm_6x8s4__wasmsimd_x86)

  BENCHMARK_GEMM(f32_ppmm_4x8_unipass__wasmsimd_arm_splat)
  BENCHMARK_GEMM(f32_ppmm_4x8_unipass__wasmsimd_x86_splat)

  BENCHMARK_GEMM(f32_ppmm_4x8_twopass__wasmsimd_arm_splat)
  BENCHMARK_GEMM(f32_ppmm_4x8_twopass__wasmsimd_x86_splat)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

static void f32_gemm_1x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_f32_gemm_minmax_ukernel_1x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1);
}
static void f32_gemm_2x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_f32_gemm_minmax_ukernel_2x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1);
}
static void f32_gemm_4x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_f32_gemm_minmax_ukernel_4x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1);
}

static void f32_ppmm_2x4_unipass__scalar(benchmark::State& state, const char* net) {
  PPMM1PBenchmark(state,
    xnn_x32_packx_ukernel_2x__scalar,
    xnn_f32_ppmm_minmax_ukernel_2x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1);
}
static void f32_ppmm_4x2_unipass__scalar(benchmark::State& state, const char* net) {
  PPMM1PBenchmark(state,
    xnn_x32_packx_ukernel_4x__scalar,
    xnn_f32_ppmm_minmax_ukernel_4x2__scalar,
    xnn_init_f32_minmax_scalar_params,
    /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1);
}
static void f32_ppmm_4x4_unipass__scalar(benchmark::State& state, const char* net) {
  PPMM1PBenchmark(state,
    xnn_x32_packx_ukernel_4x__scalar,
    xnn_f32_ppmm_minmax_ukernel_4x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1);
}
static void f32_ppmm_3x3_unipass__scalar(benchmark::State& state, const char* net) {
  PPMM1PBenchmark(state,
    xnn_x32_packx_ukernel_3x__scalar,
    xnn_f32_ppmm_minmax_ukernel_3x3__scalar,
    xnn_init_f32_minmax_scalar_params,
    /*mr=*/3, /*nr=*/3, /*kr=*/1, /*sr=*/1);
}

static void f32_ppmm_2x4_twopass__scalar(benchmark::State& state, const char* net) {
  PPMM2PBenchmark(state,
    xnn_x32_packx_ukernel_2x__scalar,
    xnn_f32_ppmm_minmax_ukernel_2x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1);
}
static void f32_ppmm_4x2_twopass__scalar(benchmark::State& state, const char* net) {
  PPMM2PBenchmark(state,
    xnn_x32_packx_ukernel_4x__scalar,
    xnn_f32_ppmm_minmax_ukernel_4x2__scalar,
    xnn_init_f32_minmax_scalar_params,
    /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1);
}
static void f32_ppmm_4x4_twopass__scalar(benchmark::State& state, const char* net) {
  PPMM2PBenchmark(state,
    xnn_x32_packx_ukernel_4x__scalar,
    xnn_f32_ppmm_minmax_ukernel_4x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1);
}
static void f32_ppmm_3x3_twopass__scalar(benchmark::State& state, const char* net) {
  PPMM2PBenchmark(state,
    xnn_x32_packx_ukernel_3x__scalar,
    xnn_f32_ppmm_minmax_ukernel_3x3__scalar,
    xnn_init_f32_minmax_scalar_params,
    /*mr=*/3, /*nr=*/3, /*kr=*/1, /*sr=*/1);
}

BENCHMARK_GEMM(f32_gemm_1x4__scalar)
BENCHMARK_GEMM(f32_gemm_2x4__scalar)
BENCHMARK_GEMM(f32_gemm_4x4__scalar)

BENCHMARK_GEMM(f32_ppmm_2x4_unipass__scalar)
BENCHMARK_GEMM(f32_ppmm_4x2_unipass__scalar)
BENCHMARK_GEMM(f32_ppmm_4x4_unipass__scalar)
BENCHMARK_GEMM(f32_ppmm_3x3_unipass__scalar)

BENCHMARK_GEMM(f32_ppmm_2x4_twopass__scalar)
BENCHMARK_GEMM(f32_ppmm_4x2_twopass__scalar)
BENCHMARK_GEMM(f32_ppmm_4x4_twopass__scalar)
BENCHMARK_GEMM(f32_ppmm_3x3_twopass__scalar)


#ifdef BENCHMARK_RUY
BENCHMARK_GEMM(ruy_st)
#endif  // BENCHMARK_RUY

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
