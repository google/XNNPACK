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
#include <xnnpack/packx.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>
#include <xnnpack/ppmm.h>


static void GEMMBenchmark(benchmark::State& state,
  xnn_f32_gemm_minmax_ukernel_function gemm,
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
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

  std::vector<float> a(mc * kc);
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

  std::vector<float, AlignedAllocator<float, 32>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0.0f);
  xnn_pack_f32_gemm_goi_w(1 /* groups */, nc, kc, nr, kr, sr, k.data(), b.data(), w.data());
  std::vector<float> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  xnn_f32_minmax_params minmax_params =
    xnn_init_f32_minmax_params(-std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

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
        &minmax_params);
    }
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}

static void PPMM1PBenchmark(benchmark::State& state,
  xnn_f32_ppmm_minmax_ukernel_function ppmm,
  xnn_x32_packx_ukernel_function packx,
  size_t mr, size_t nr,
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

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

  std::vector<float> a(mc * kc);
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::vector<float> k(nc * kc);
  std::generate(k.begin(), k.end(), std::ref(f32rng));
  std::vector<float> b(nc);
  std::generate(b.begin(), b.end(), std::ref(f32rng));

  std::vector<uint32_t, AlignedAllocator<uint32_t, 32>> t(mr * kc);

  const size_t w_elements = nc_stride * kc + nc_stride;
  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (w_elements + c_elements));

  std::vector<float, AlignedAllocator<float, 32>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0.0f);
  xnn_pack_f32_gemm_goi_w(1 /* groups */, nc, kc, nr, 1 /* kr */, 1 /* sr */, k.data(), b.data(), w.data());
  std::vector<float> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  xnn_f32_minmax_params minmax_params =
    xnn_init_f32_minmax_params(-std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

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
        &minmax_params);
    }
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}

static void PPMM2PBenchmark(benchmark::State& state,
  xnn_f32_ppmm_minmax_ukernel_function ppmm,
  xnn_x32_packx_ukernel_function packx,
  size_t mr, size_t nr,
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

  const size_t mc_stride = benchmark::utils::RoundUp(mc, mr);
  const size_t nc_stride = benchmark::utils::RoundUp(nc, nr);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

  std::vector<float> a(mc * kc);
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::vector<float> k(nc * kc);
  std::generate(k.begin(), k.end(), std::ref(f32rng));
  std::vector<float> b(nc);
  std::generate(b.begin(), b.end(), std::ref(f32rng));

  std::vector<uint32_t, AlignedAllocator<uint32_t, 32>> t(mc_stride * kc);

  const size_t w_elements = nc_stride * kc + nc_stride;
  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (w_elements + c_elements));

  std::vector<float, AlignedAllocator<float, 32>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0.0f);
  xnn_pack_f32_gemm_goi_w(1 /* groups */, nc, kc, nr, 1 /* kr */, 1 /* sr */, k.data(), b.data(), w.data());
  std::vector<float> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  xnn_f32_minmax_params minmax_params =
    xnn_init_f32_minmax_params(-std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

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
        &minmax_params);
    }
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}

#ifdef BENCHMARK_RUY
static void RuyBenchmark(benchmark::State& state, uint32_t threads)
{
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (nc * (mc + kc + 1)));

  std::vector<float> a(mc * kc);
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::vector<float> k(num_buffers * nc * kc);
  std::generate(k.begin(), k.end(), std::ref(f32rng));
  std::vector<float> b(num_buffers * nc);
  std::generate(b.begin(), b.end(), std::ref(f32rng));
  std::vector<float> c(num_buffers * nc * mc);
  std::fill(c.begin(), c.end(), std::nanf(""));

  // Note: context must be static to avoid the cost of re-creating it for each benchmark.
  static ruy::Context context;
  context.max_num_threads = threads;

  ruy::Matrix<float> ruy_a;
  ruy::MakeSimpleLayout(nc, kc, ruy::Order::kRowMajor, &ruy_a.layout);
  ruy::Matrix<float> ruy_b;
  ruy::MakeSimpleLayout(kc, mc, ruy::Order::kColMajor, &ruy_b.layout);
  ruy_b.data = a.data();
  ruy::Matrix<float> ruy_c;
  ruy::MakeSimpleLayout(nc, mc, ruy::Order::kColMajor, &ruy_c.layout);

  ruy::BasicSpec<float, float> spec;

  // ruy::Context uses deferred initialization, which affects percieved GEMM performance. Initialization happens during
  // the first GEMM calls, and per Benoit Jacob it takes up to ~250 milliseconds for performance to stabilize.
  // Thus, on the first benchmark, we compute GEMM for 500 milliseconds (to be safe) without recording performance, and
  // keep the ruy::Context object initialized (by being static) between subsequent benchmarks.
  static std::once_flag warmup;
  std::call_once(warmup, [&](){
    auto start = std::chrono::steady_clock::now();
    do {
      ruy_a.data = k.data();
      ruy_c.data = c.data();
      spec.bias = b.data();

      ruy::Mul<ruy::kAllPaths>(ruy_a, ruy_b, spec, &context, &ruy_c);
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

    ruy_a.data = k.data() + buffer_index * nc * kc;
    ruy_c.data = c.data() + buffer_index * mc * nc;
    spec.bias = b.data() + buffer_index * nc;

    ruy::Mul<ruy::kAllPaths>(ruy_a, ruy_b, spec, &context, &ruy_c);
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}

static void ruy_st(benchmark::State& state, const char* net)
{
  RuyBenchmark(state, 1);
}
#endif  // BENCHMARK_RUY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void f32_gemm_1x8__aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_ld64, 1, 8, 1, 1);
  }
  static void f32_gemm_1x12__aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x12__aarch64_neonfma_cortex_a53, 1, 12, 1, 1);
  }
  static void f32_gemm_1x8__aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53, 1, 8, 1, 1);
  }
  static void f32_gemm_1x8__aarch64_neonfma_cortex_a57(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a57, 1, 8, 1, 1);
  }
  static void f32_gemm_1x8__aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a75, 1, 8, 1, 1);
  }
  static void f32_gemm_4x12__aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x12__aarch64_neonfma_cortex_a53, 4, 12, 1, 1);
  }
  static void f32_gemm_4x8__aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a53, 4, 8, 1, 1);
  }
  static void f32_gemm_4x8__aarch64_neonfma_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a55, 4, 8, 1, 1);
  }
  static void f32_gemm_4x8__aarch64_neonfma_cortex_a57(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a57, 4, 8, 1, 1);
  }
  static void f32_gemm_4x8__aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a75, 4, 8, 1, 1);
  }
  static void f32_gemm_4x8__aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_ld64, 4, 8, 1, 1);
  }
  static void f32_gemm_4x8__aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_ld128, 4, 8, 1, 1);
  }
  static void f32_gemm_5x8__aarch64_neonfma_cortex_a57(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8__aarch64_neonfma_cortex_a57, 5, 8, 1, 1);
  }
  static void f32_gemm_5x8__aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8__aarch64_neonfma_cortex_a75, 5, 8, 1, 1);
  }
  static void f32_gemm_6x8__aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_ld64, 6, 8, 1, 1);
  }
  static void f32_gemm_6x8__aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_ld128, 6, 8, 1, 1);
  }
  static void f32_gemm_6x8__aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a53, 6, 8, 1, 1);
  }
  static void f32_gemm_6x8__aarch64_neonfma_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a55, 6, 8, 1, 1);
  }
  static void f32_gemm_6x8__aarch64_neonfma_cortex_a73(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a73, 6, 8, 1, 1);
  }
  static void f32_gemm_6x8__aarch64_neonfma_cortex_a57(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a57, 6, 8, 1, 1);
  }
  static void f32_gemm_6x8__aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a75, 6, 8, 1, 1);
  }
  static void f32_gemm_6x8__aarch64_neonfma_ios(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_ios, 6, 8, 1, 1);
  }
  static void f32_gemm_1x8__neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__neonfma_lane_ld64, 1, 8, 1, 1);
  }
  static void f32_gemm_4x8__neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__neonfma_lane_ld64, 4, 8, 1, 1);
  }
  static void f32_gemm_4x8__neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__neonfma_lane_ld128, 4, 8, 1, 1);
  }
  static void f32_gemm_5x8__neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8__neonfma_lane_ld64, 5, 8, 1, 1);
  }
  static void f32_gemm_6x8__neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__neonfma_lane_ld64, 6, 8, 1, 1);
  }
  static void f32_gemm_6x8__neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__neonfma_lane_ld128, 6, 8, 1, 1);
  }
  BENCHMARK_GEMM(f32_gemm_1x8__aarch64_neonfma_ld64)
  BENCHMARK_GEMM(f32_gemm_1x12__aarch64_neonfma_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_1x8__aarch64_neonfma_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_1x8__aarch64_neonfma_cortex_a57)
  BENCHMARK_GEMM(f32_gemm_1x8__aarch64_neonfma_cortex_a75)
  BENCHMARK_GEMM(f32_gemm_4x12__aarch64_neonfma_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch64_neonfma_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch64_neonfma_cortex_a55)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch64_neonfma_cortex_a57)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch64_neonfma_cortex_a75)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch64_neonfma_ld128)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch64_neonfma_ld64)
  BENCHMARK_GEMM(f32_gemm_5x8__aarch64_neonfma_cortex_a57)
  BENCHMARK_GEMM(f32_gemm_5x8__aarch64_neonfma_cortex_a75)
  BENCHMARK_GEMM(f32_gemm_6x8__aarch64_neonfma_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_6x8__aarch64_neonfma_cortex_a55)
  BENCHMARK_GEMM(f32_gemm_6x8__aarch64_neonfma_cortex_a73)
  BENCHMARK_GEMM(f32_gemm_6x8__aarch64_neonfma_cortex_a57)
  BENCHMARK_GEMM(f32_gemm_6x8__aarch64_neonfma_cortex_a75)
  BENCHMARK_GEMM(f32_gemm_6x8__aarch64_neonfma_ios)
  BENCHMARK_GEMM(f32_gemm_6x8__aarch64_neonfma_ld64)
  BENCHMARK_GEMM(f32_gemm_6x8__aarch64_neonfma_ld128)
  BENCHMARK_GEMM(f32_gemm_1x8__neonfma_lane_ld64)
  BENCHMARK_GEMM(f32_gemm_4x8__neonfma_lane_ld64)
  BENCHMARK_GEMM(f32_gemm_4x8__neonfma_lane_ld128)
  BENCHMARK_GEMM(f32_gemm_5x8__neonfma_lane_ld64)
  BENCHMARK_GEMM(f32_gemm_6x8__neonfma_lane_ld64)
  BENCHMARK_GEMM(f32_gemm_6x8__neonfma_lane_ld128)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void f32_gemm_4x8__aarch32_neon_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_ld64, 4, 8, 1, 1, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__aarch32_neon_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_cortex_a53, 4, 8, 1, 1, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__aarch32_neon_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_cortex_a55, 4, 8, 1, 1, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__aarch32_neon_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_cortex_a75, 4, 8, 1, 1, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__aarch32_neon_pld_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_pld_cortex_a75, 4, 8, 1, 1, benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_4x8__aarch32_neon_ld64)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch32_neon_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch32_neon_cortex_a55)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch32_neon_cortex_a75)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch32_neon_pld_cortex_a75)
#endif  // XNN_ARCH_ARM

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_gemm_1x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64, 1, 8, 1, 1, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld64, 4, 8, 1, 1, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__neon_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld128, 4, 8, 1, 1, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_5x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8__neon_lane_ld64, 5, 8, 1, 1, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_6x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__neon_lane_ld64, 6, 8, 1, 1, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_6x8__neon_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__neon_lane_ld128, 6, 8, 1, 1, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_1x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, 1, 8, 1, 1, benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_4x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, 4, 8, 1, 1, benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_4x8__neonfma_dup_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__neonfma_dup_ld128, 4, 8, 1, 1, benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_6x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, 6, 8, 1, 1, benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_6x8__neonfma_dup_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__neonfma_dup_ld128, 6, 8, 1, 1, benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_1x8s4__neon(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8s4__neon, 1, 8, 1, 4, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_1x8s4__neonfma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8s4__neonfma, 1, 8, 1, 4, benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_4x8s4__neon(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8s4__neon, 4, 8, 1, 4, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8s4__neonfma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8s4__neonfma, 4, 8, 1, 4, benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_6x8s4__neon(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8s4__neon, 6, 8, 1, 4, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_6x8s4__neonfma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8s4__neonfma, 6, 8, 1, 4, benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_8x8s4__neon(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_8x8s4__neon, 8, 8, 1, 4, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_8x8s4__neonfma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_8x8s4__neonfma, 8, 8, 1, 4, benchmark::utils::CheckNEONFMA);
  }
  static void f32_ppmm_4x8_unipass__neonfma(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x8__neonfma, xnn_x32_packx_ukernel_4x__neon_st4, 4, 8, benchmark::utils::CheckNEONFMA);
  }
  static void f32_ppmm_4x8_twopass__neonfma(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x8__neonfma, xnn_x32_packx_ukernel_4x__neon_st4, 4, 8, benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_GEMM(f32_gemm_1x8__neon_lane_ld64)
  BENCHMARK_GEMM(f32_gemm_4x8__neon_lane_ld64)
  BENCHMARK_GEMM(f32_gemm_4x8__neon_lane_ld128)
  BENCHMARK_GEMM(f32_gemm_5x8__neon_lane_ld64)
  BENCHMARK_GEMM(f32_gemm_6x8__neon_lane_ld64)
  BENCHMARK_GEMM(f32_gemm_6x8__neon_lane_ld128)
  BENCHMARK_GEMM(f32_gemm_1x8__neonfma_dup_ld64)
  BENCHMARK_GEMM(f32_gemm_4x8__neonfma_dup_ld128)
  BENCHMARK_GEMM(f32_gemm_4x8__neonfma_dup_ld64)
  BENCHMARK_GEMM(f32_gemm_6x8__neonfma_dup_ld64)
  BENCHMARK_GEMM(f32_gemm_6x8__neonfma_dup_ld128)
  BENCHMARK_GEMM(f32_gemm_1x8s4__neon)
  BENCHMARK_GEMM(f32_gemm_1x8s4__neonfma)
  BENCHMARK_GEMM(f32_gemm_4x8s4__neon)
  BENCHMARK_GEMM(f32_gemm_4x8s4__neonfma)
  BENCHMARK_GEMM(f32_gemm_6x8s4__neon)
  BENCHMARK_GEMM(f32_gemm_6x8s4__neonfma)
  BENCHMARK_GEMM(f32_gemm_8x8s4__neon)
  BENCHMARK_GEMM(f32_gemm_8x8s4__neonfma)
  BENCHMARK_GEMM(f32_ppmm_4x8_unipass__neonfma)
  BENCHMARK_GEMM(f32_ppmm_4x8_twopass__neonfma)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_gemm_1x8__sse_load1(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__sse_load1, 1, 8, 1, 1);
  }
  static void f32_gemm_4x8__sse_load1(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__sse_load1, 4, 8, 1, 1);
  }

  static void f32_gemm_1x8__sse_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__sse_dup, 1, 8, 1, 1);
  }
  static void f32_gemm_4x8__sse_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__sse_dup, 4, 8, 1, 1);
  }

  static void f32_gemm_1x8s4__sse(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8s4__sse, 1, 8, 1, 4);
  }
  static void f32_gemm_4x8s4__sse(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8s4__sse, 4, 8, 1, 4);
  }

  static void f32_ppmm_4x8_unipass__sse(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_x32_packx_ukernel_4x__sse, 4, 8);
  }
  static void f32_ppmm_4x8_twopass__sse(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_x32_packx_ukernel_4x__sse, 4, 8);
  }

  static void f32_gemm_1x8__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__avx_broadcast, 1, 8, 1, 1, benchmark::utils::CheckAVX);
  }
  static void f32_gemm_4x8__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__avx_broadcast, 4, 8, 1, 1, benchmark::utils::CheckAVX);
  }
  static void f32_gemm_5x8__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8__avx_broadcast, 5, 8, 1, 1, benchmark::utils::CheckAVX);
  }
  static void f32_gemm_6x8__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__avx_broadcast, 6, 8, 1, 1, benchmark::utils::CheckAVX);
  }
  static void f32_gemm_7x8__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_7x8__avx_broadcast, 7, 8, 1, 1, benchmark::utils::CheckAVX);
  }
  static void f32_gemm_1x16__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x16__avx_broadcast, 1, 16, 1, 1, benchmark::utils::CheckAVX);
  }
  static void f32_gemm_3x16__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_3x16__avx_broadcast, 4, 16, 1, 1, benchmark::utils::CheckAVX);
  }
  static void f32_gemm_4x16__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x16__avx_broadcast, 4, 16, 1, 1, benchmark::utils::CheckAVX);
  }
  static void f32_gemm_5x16__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x16__avx_broadcast, 5, 16, 1, 1, benchmark::utils::CheckAVX);
  }

  static void f32_gemm_1x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__fma3_broadcast, 1, 8, 1, 1, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_4x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__fma3_broadcast, 4, 8, 1, 1, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_5x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8__fma3_broadcast, 5, 8, 1, 1, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_6x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__fma3_broadcast, 6, 8, 1, 1, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_7x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_7x8__fma3_broadcast, 7, 8, 1, 1, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_8x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_8x8__fma3_broadcast, 8, 8, 1, 1, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_1x16__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x16__fma3_broadcast, 1, 16, 1, 1, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_3x16__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_3x16__fma3_broadcast, 4, 16, 1, 1, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_4x16__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x16__fma3_broadcast, 4, 16, 1, 1, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_5x16__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x16__fma3_broadcast, 5, 16, 1, 1, benchmark::utils::CheckFMA3);
  }

  static void f32_gemm_1x16s4__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x16s4__fma3_broadcast, 1, 16, 1, 4, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_3x16s4__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_3x16s4__fma3_broadcast, 4, 16, 1, 4, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_4x16s4__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x16s4__fma3_broadcast, 4, 16, 1, 4, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_5x16s4__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x16s4__fma3_broadcast, 5, 16, 1, 4, benchmark::utils::CheckFMA3);
  }

  static void f32_gemm_1x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x16__avx512f_broadcast, 1, 16, 1, 1, benchmark::utils::CheckAVX512F);
  }
  static void f32_gemm_4x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x16__avx512f_broadcast, 4, 16, 1, 1, benchmark::utils::CheckAVX512F);
  }
  static void f32_gemm_5x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x16__avx512f_broadcast, 5, 16, 1, 1, benchmark::utils::CheckAVX512F);
  }
  static void f32_gemm_6x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x16__avx512f_broadcast, 6, 16, 1, 1, benchmark::utils::CheckAVX512F);
  }
  static void f32_gemm_7x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_7x16__avx512f_broadcast, 7, 16, 1, 1, benchmark::utils::CheckAVX512F);
  }
  static void f32_gemm_8x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_8x16__avx512f_broadcast, 8, 16, 1, 1, benchmark::utils::CheckAVX512F);
  }

  BENCHMARK_GEMM(f32_gemm_1x8__sse_load1)
  BENCHMARK_GEMM(f32_gemm_4x8__sse_load1)

  BENCHMARK_GEMM(f32_gemm_1x8__sse_dup)
  BENCHMARK_GEMM(f32_gemm_4x8__sse_dup)

  BENCHMARK_GEMM(f32_gemm_1x8s4__sse)
  BENCHMARK_GEMM(f32_gemm_4x8s4__sse)

  BENCHMARK_GEMM(f32_ppmm_4x8_unipass__sse)
  BENCHMARK_GEMM(f32_ppmm_4x8_twopass__sse)

  BENCHMARK_GEMM(f32_gemm_1x8__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_4x8__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_5x8__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_6x8__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_7x8__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_1x16__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_3x16__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_4x16__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_5x16__avx_broadcast)

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

  BENCHMARK_GEMM(f32_gemm_1x16s4__fma3_broadcast)
  BENCHMARK_GEMM(f32_gemm_3x16s4__fma3_broadcast)
  BENCHMARK_GEMM(f32_gemm_4x16s4__fma3_broadcast)
  BENCHMARK_GEMM(f32_gemm_5x16s4__fma3_broadcast)

  BENCHMARK_GEMM(f32_gemm_1x16__avx512f_broadcast)
  BENCHMARK_GEMM(f32_gemm_4x16__avx512f_broadcast)
  BENCHMARK_GEMM(f32_gemm_5x16__avx512f_broadcast)
  BENCHMARK_GEMM(f32_gemm_6x16__avx512f_broadcast)
  BENCHMARK_GEMM(f32_gemm_7x16__avx512f_broadcast)
  BENCHMARK_GEMM(f32_gemm_8x16__avx512f_broadcast)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if !XNN_ARCH_WASM && !XNN_ARCH_ASMJS
  static void f32_gemm_4x8__psimd_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__psimd_loadsplat, 4, 8, 1, 1);
  }

  static void f32_gemm_6x8__psimd_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__psimd_loadsplat, 6, 8, 1, 1);
  }

  static void f32_gemm_4x8__psimd_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__psimd_splat, 4, 8, 1, 1);
  }

  static void f32_gemm_6x8__psimd_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__psimd_splat, 6, 8, 1, 1);
  }

  static void f32_gemm_4x8s4__psimd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8s4__psimd, 4, 8, 1, 4);
  }

  static void f32_gemm_6x8s4__psimd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8s4__psimd, 6, 8, 1, 4);
  }

  static void f32_ppmm_4x8_unipass__psimd(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x8__psimd, xnn_x32_packx_ukernel_4x__psimd, 4, 8);
  }

  static void f32_ppmm_4x8_twopass__psimd(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x8__psimd, xnn_x32_packx_ukernel_4x__psimd, 4, 8);
  }

  BENCHMARK_GEMM(f32_gemm_4x8__psimd_loadsplat)
  BENCHMARK_GEMM(f32_gemm_6x8__psimd_loadsplat)
  BENCHMARK_GEMM(f32_gemm_4x8__psimd_splat)
  BENCHMARK_GEMM(f32_gemm_6x8__psimd_splat)
  BENCHMARK_GEMM(f32_gemm_4x8s4__psimd)
  BENCHMARK_GEMM(f32_gemm_6x8s4__psimd)
  BENCHMARK_GEMM(f32_ppmm_4x8_unipass__psimd)
  BENCHMARK_GEMM(f32_ppmm_4x8_twopass__psimd)
#endif  // !XNN_ARCH_WASM && !XNN_ARCH_ASMJS

static void f32_gemm_1x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x4__scalar, 1, 4, 1, 1);
}

static void f32_gemm_2x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_2x4__scalar, 2, 4, 1, 1);
}

static void f32_gemm_4x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x4__scalar, 4, 4, 1, 1);
}

static void f32_ppmm_2x4_unipass__scalar(benchmark::State& state, const char* net) {
  PPMM1PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_x32_packx_ukernel_2x__scalar, 2, 4);
}

static void f32_ppmm_4x2_unipass__scalar(benchmark::State& state, const char* net) {
  PPMM1PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_x32_packx_ukernel_4x__scalar, 4, 2);
}

static void f32_ppmm_4x4_unipass__scalar(benchmark::State& state, const char* net) {
  PPMM1PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_x32_packx_ukernel_4x__scalar, 4, 4);
}

static void f32_ppmm_3x3_unipass__scalar(benchmark::State& state, const char* net) {
  PPMM1PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_x32_packx_ukernel_3x__scalar, 3, 3);
}

static void f32_ppmm_2x4_twopass__scalar(benchmark::State& state, const char* net) {
  PPMM2PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_x32_packx_ukernel_2x__scalar, 2, 4);
}

static void f32_ppmm_4x2_twopass__scalar(benchmark::State& state, const char* net) {
  PPMM2PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_x32_packx_ukernel_4x__scalar, 4, 2);
}

static void f32_ppmm_4x4_twopass__scalar(benchmark::State& state, const char* net) {
  PPMM2PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_x32_packx_ukernel_4x__scalar, 4, 4);
}

static void f32_ppmm_3x3_twopass__scalar(benchmark::State& state, const char* net) {
  PPMM2PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_x32_packx_ukernel_3x__scalar, 3, 3);
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
