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

#include <benchmark/benchmark.h>
#ifdef BENCHMARK_RUY
#include "ruy/ruy.h"
#endif  // BENCHMARK_RUY
#include "bench/gemm.h"
#include "bench/utils.h"
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/allocator.h>
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
  xnn_init_f32_minmax_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
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

  std::vector<float, AlignedAllocator<float, 64>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0.0f);
  xnn_pack_f32_gemm_goi_w(1 /* groups */, nc, kc, nr, kr, sr, k.data(), b.data(), w.data(), 0, nullptr);
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

static void PPMM1PBenchmark(benchmark::State& state,
  xnn_f32_ppmm_minmax_ukernel_function ppmm,
  xnn_x32_packx_ukernel_function packx,
  size_t mr, size_t nr,
  xnn_init_f32_minmax_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }

  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  const size_t nc_stride = benchmark::utils::RoundUp(nc, nr);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<float> a(mc * kc);
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
  xnn_pack_f32_gemm_goi_w(1 /* groups */, nc, kc, nr, 1 /* kr */, 1 /* sr */, k.data(), b.data(), w.data(), 0, nullptr);
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
  xnn_f32_ppmm_minmax_ukernel_function ppmm,
  xnn_x32_packx_ukernel_function packx,
  size_t mr, size_t nr,
  xnn_init_f32_minmax_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
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
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<float> a(mc * kc);
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
  xnn_pack_f32_gemm_goi_w(1 /* groups */, nc, kc, nr, 1 /* kr */, 1 /* sr */, k.data(), b.data(), w.data(), 0, nullptr);
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
  context.set_max_num_threads(threads);

  ruy::Matrix<float> ruy_a;
  ruy::MakeSimpleLayout(nc, kc, ruy::Order::kRowMajor, ruy_a.mutable_layout());
  ruy::Matrix<float> ruy_b;
  ruy::MakeSimpleLayout(kc, mc, ruy::Order::kColMajor, ruy_b.mutable_layout());
  ruy_b.set_data(a.data());
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

#if XNN_PLATFORM_JIT
static void GEMMBenchmark(benchmark::State& state,
  xnn_jit_gemm_code_generator_function generator,
  size_t mr, size_t nr, size_t kr, size_t sr,
  xnn_init_f32_minmax_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
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
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

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

  std::vector<float, AlignedAllocator<float, 64>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0.0f);
  xnn_pack_f32_gemm_goi_w(1 /* groups */, nc, kc, nr, kr, sr, k.data(), b.data(), w.data(), 0, nullptr);
  std::vector<float> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  xnn_f32_minmax_params params;
  init_params(&params,
    -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

  xnn_code_buffer code_buffer;
  xnn_allocate_code_memory(&code_buffer, XNN_DEFAULT_CODE_BUFFER_SIZE);
  jit_gemm_params jit_params = {
    .f32_minmax = {
      .min = -std::numeric_limits<float>::infinity(),
      .max = +std::numeric_limits<float>::infinity()
    }
  };
  generator(&code_buffer, nc, kc * sizeof(float), &jit_params);
  xnn_f32_gemm_minmax_ukernel_function gemm = reinterpret_cast<xnn_f32_gemm_minmax_ukernel_function>(code_buffer.code);

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

  xnn_release_code_memory(&code_buffer);

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}
#endif  // XNN_PLATFORM_JIT

#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void f32_gemm_1x8__aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_ld64, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_1x12__aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x12__aarch64_neonfma_cortex_a53, 1, 12, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_1x8__aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_1x8__aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a75, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_1x8__aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_4x12__aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x12__aarch64_neonfma_cortex_a53, 4, 12, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_4x8__aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a53, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_4x8__aarch64_neonfma_prfm_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_prfm_cortex_a53, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_4x8__aarch64_neonfma_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a55, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_4x8__aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a75, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_4x8__aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_prfm_cortex_a75, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_4x8__aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_ld64, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_4x8__aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_ld128, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_5x8__aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8__aarch64_neonfma_cortex_a75, 5, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_5x8__aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8__aarch64_neonfma_prfm_cortex_a75, 5, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_6x8__aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_ld64, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_6x8__aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_ld128, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_6x8__aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a53, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_6x8__aarch64_neonfma_prfm_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_prfm_cortex_a53, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_6x8__aarch64_neonfma_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a55, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_6x8__aarch64_neonfma_cortex_a73(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a73, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_6x8__aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a75, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_6x8__aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_1x8__neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__neonfma_lane_ld64, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_4x8__neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__neonfma_lane_ld64, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_4x8__neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__neonfma_lane_ld128, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_5x8__neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8__neonfma_lane_ld64, 5, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_6x8__neonfma_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__neonfma_lane_ld64, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_6x8__neonfma_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__neonfma_lane_ld128, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }

  BENCHMARK_GEMM(f32_gemm_1x8__aarch64_neonfma_ld64)
  BENCHMARK_GEMM(f32_gemm_1x12__aarch64_neonfma_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_1x8__aarch64_neonfma_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_1x8__aarch64_neonfma_cortex_a75)
  BENCHMARK_GEMM(f32_gemm_1x8__aarch64_neonfma_prfm_cortex_a75)
  BENCHMARK_GEMM(f32_gemm_4x12__aarch64_neonfma_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch64_neonfma_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch64_neonfma_prfm_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch64_neonfma_cortex_a55)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch64_neonfma_cortex_a75)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch64_neonfma_prfm_cortex_a75)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch64_neonfma_ld128)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch64_neonfma_ld64)
  BENCHMARK_GEMM(f32_gemm_5x8__aarch64_neonfma_cortex_a75)
  BENCHMARK_GEMM(f32_gemm_5x8__aarch64_neonfma_prfm_cortex_a75)
  BENCHMARK_GEMM(f32_gemm_6x8__aarch64_neonfma_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_6x8__aarch64_neonfma_prfm_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_6x8__aarch64_neonfma_cortex_a55)
  BENCHMARK_GEMM(f32_gemm_6x8__aarch64_neonfma_cortex_a73)
  BENCHMARK_GEMM(f32_gemm_6x8__aarch64_neonfma_cortex_a75)
  BENCHMARK_GEMM(f32_gemm_6x8__aarch64_neonfma_prfm_cortex_a75)
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
  static void f32_gemm_4x4__aarch32_vfp_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x4__aarch32_vfp_ld64, 4, 4, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckVFP);
  }
  static void f32_gemm_4x8__aarch32_neon_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_ld64, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__aarch32_neon_cortex_a7(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_cortex_a7, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__aarch32_neon_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_cortex_a53, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__aarch32_neon_prfm_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_prfm_cortex_a53, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__aarch32_neon_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_cortex_a55, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__aarch32_neon_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_cortex_a75, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__aarch32_neon_prfm_cortex_a75(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_prfm_cortex_a75, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(f32_gemm_4x4__aarch32_vfp_ld64)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch32_neon_ld64)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch32_neon_cortex_a7)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch32_neon_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch32_neon_prfm_cortex_a53)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch32_neon_cortex_a55)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch32_neon_cortex_a75)
  BENCHMARK_GEMM(f32_gemm_4x8__aarch32_neon_prfm_cortex_a75)
#endif  // XNN_ARCH_ARM

#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
  static void jit_f32_gemm_4x8__aarch32_neon_ld64(benchmark::State& state, const char* net)
  {
    GEMMBenchmark(state, xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_ld64, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void jit_f32_gemm_4x8__aarch32_neon_cortex_a7(benchmark::State& state, const char* net)
  {
    GEMMBenchmark(state, xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a7, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void jit_f32_gemm_4x8__aarch32_neon_cortex_a53(benchmark::State& state, const char* net)
  {
    GEMMBenchmark(state, xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a53, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void jit_f32_gemm_4x8__aarch32_neon_cortex_a55(benchmark::State& state, const char* net)
  {
    GEMMBenchmark(state, xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a55, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void jit_f32_gemm_4x8__aarch32_neon_cortex_a75(benchmark::State& state, const char* net)
  {
    GEMMBenchmark(state, xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a75, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void jit_f32_gemm_4x8__aarch32_neon_prfm_cortex_a75(benchmark::State& state, const char* net)
  {
    GEMMBenchmark(state, xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_prfm_cortex_a75, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }

  BENCHMARK_GEMM(jit_f32_gemm_4x8__aarch32_neon_ld64)
  BENCHMARK_GEMM(jit_f32_gemm_4x8__aarch32_neon_cortex_a7)
  BENCHMARK_GEMM(jit_f32_gemm_4x8__aarch32_neon_cortex_a53)
  BENCHMARK_GEMM(jit_f32_gemm_4x8__aarch32_neon_cortex_a55)
  BENCHMARK_GEMM(jit_f32_gemm_4x8__aarch32_neon_cortex_a75)
  BENCHMARK_GEMM(jit_f32_gemm_4x8__aarch32_neon_prfm_cortex_a75)
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT

#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  static void jit_f32_gemm_1x8__aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net)
  {
    GEMMBenchmark(state, xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void jit_f32_gemm_1x8__aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, const char* net)
  {
    GEMMBenchmark(state, xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void jit_f32_gemm_6x8__aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net)
  {
    GEMMBenchmark(state, xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void jit_f32_gemm_6x8__aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, const char* net)
  {
    GEMMBenchmark(state, xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  BENCHMARK_GEMM(jit_f32_gemm_1x8__aarch64_neonfma_cortex_a75)
  BENCHMARK_GEMM(jit_f32_gemm_1x8__aarch64_neonfma_prfm_cortex_a75)
  BENCHMARK_GEMM(jit_f32_gemm_6x8__aarch64_neonfma_cortex_a75)
  BENCHMARK_GEMM(jit_f32_gemm_6x8__aarch64_neonfma_prfm_cortex_a75)
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_gemm_1x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld64, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__neon_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld128, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_5x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8__neon_lane_ld64, 5, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_6x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__neon_lane_ld64, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_6x8__neon_lane_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__neon_lane_ld128, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_1x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__neonfma_dup_ld64, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_4x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__neonfma_dup_ld64, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_4x8__neonfma_dup_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__neonfma_dup_ld128, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_6x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__neonfma_dup_ld64, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_6x8__neonfma_dup_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__neonfma_dup_ld128, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_1x8s4__neon(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8s4__neon, 1, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_1x8s4__neonfma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8s4__neonfma, 1, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_4x8s4__neon(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8s4__neon, 4, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8s4__neonfma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8s4__neonfma, 4, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_6x8s4__neon(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8s4__neon, 6, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_6x8s4__neonfma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8s4__neonfma, 6, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_8x8s4__neon(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_8x8s4__neon, 8, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_gemm_8x8s4__neonfma(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_8x8s4__neonfma, 8, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void f32_ppmm_4x8_unipass__neonfma(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x8__neonfma, xnn_x32_packx_ukernel_4x__neon_st4, 4, 8,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void f32_ppmm_4x8_twopass__neonfma(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x8__neonfma, xnn_x32_packx_ukernel_4x__neon_st4, 4, 8,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
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
  BENCHMARK_GEMM(f32_gemm_4x8s4__neon)
  BENCHMARK_GEMM(f32_gemm_6x8s4__neon)
  BENCHMARK_GEMM(f32_gemm_8x8s4__neon)

  BENCHMARK_GEMM(f32_gemm_1x8s4__neonfma)
  BENCHMARK_GEMM(f32_gemm_4x8s4__neonfma)
  BENCHMARK_GEMM(f32_gemm_6x8s4__neonfma)
  BENCHMARK_GEMM(f32_gemm_8x8s4__neonfma)

  BENCHMARK_GEMM(f32_ppmm_4x8_unipass__neonfma)
  BENCHMARK_GEMM(f32_ppmm_4x8_twopass__neonfma)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_gemm_1x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x16__avx512f_broadcast, 1, 16, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckAVX512F);
  }
  static void f32_gemm_4x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x16__avx512f_broadcast, 4, 16, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckAVX512F);
  }
  static void f32_gemm_5x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x16__avx512f_broadcast, 5, 16, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckAVX512F);
  }
  static void f32_gemm_6x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x16__avx512f_broadcast, 6, 16, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckAVX512F);
  }
  static void f32_gemm_7x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_7x16__avx512f_broadcast, 7, 16, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckAVX512F);
  }
  static void f32_gemm_8x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_8x16__avx512f_broadcast, 8, 16, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckAVX512F);
  }

  static void f32_gemm_1x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__fma3_broadcast, 1, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_4x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__fma3_broadcast, 4, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_5x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8__fma3_broadcast, 5, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_6x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__fma3_broadcast, 6, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_7x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_7x8__fma3_broadcast, 7, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_8x8__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_8x8__fma3_broadcast, 8, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_1x16__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x16__fma3_broadcast, 1, 16, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_3x16__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_3x16__fma3_broadcast, 4, 16, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_4x16__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x16__fma3_broadcast, 4, 16, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_5x16__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x16__fma3_broadcast, 5, 16, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckFMA3);
  }

  static void f32_gemm_1x16s4__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x16s4__fma3_broadcast, 1, 16, 1, 4,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_3x16s4__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_3x16s4__fma3_broadcast, 4, 16, 1, 4,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_4x16s4__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x16s4__fma3_broadcast, 4, 16, 1, 4,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_5x16s4__fma3_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x16s4__fma3_broadcast, 5, 16, 1, 4,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckFMA3);
  }

  static void f32_gemm_1x8__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__avx_broadcast, 1, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckAVX);
  }
  static void f32_gemm_4x8__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__avx_broadcast, 4, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckAVX);
  }
  static void f32_gemm_5x8__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8__avx_broadcast, 5, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckAVX);
  }
  static void f32_gemm_6x8__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__avx_broadcast, 6, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckAVX);
  }
  static void f32_gemm_7x8__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_7x8__avx_broadcast, 7, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckAVX);
  }
  static void f32_gemm_1x16__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x16__avx_broadcast, 1, 16, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckAVX);
  }
  static void f32_gemm_3x16__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_3x16__avx_broadcast, 4, 16, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckAVX);
  }
  static void f32_gemm_4x16__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x16__avx_broadcast, 4, 16, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckAVX);
  }
  static void f32_gemm_5x16__avx_broadcast(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x16__avx_broadcast, 5, 16, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckAVX);
  }

  static void f32_gemm_1x8__sse2_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__sse2_dup, 1, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_gemm_3x8__sse2_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_3x8__sse2_dup, 3, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_gemm_4x8__sse2_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__sse2_dup, 4, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_gemm_5x8__sse2_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8__sse2_dup, 5, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }

  static void f32_gemm_1x8__sse_load1(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__sse_load1, 1, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_gemm_3x8__sse_load1(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_3x8__sse_load1, 3, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_gemm_4x8__sse_load1(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__sse_load1, 4, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_gemm_5x8__sse_load1(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8__sse_load1, 5, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }

  static void f32_gemm_1x8__sse_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8__sse_dup, 1, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_gemm_3x8__sse_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_3x8__sse_dup, 3, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_gemm_4x8__sse_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__sse_dup, 4, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_gemm_5x8__sse_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8__sse_dup, 5, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }

  static void f32_gemm_1x8s4__sse(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x8s4__sse, 1, 8, 1, 4,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_gemm_3x8s4__sse(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_3x8s4__sse, 3, 8, 1, 4,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_gemm_4x8s4__sse(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8s4__sse, 4, 8, 1, 4,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_gemm_5x8s4__sse(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8s4__sse, 5, 8, 1, 4,
      xnn_init_f32_minmax_sse_params);
  }

  static void f32_ppmm_4x8_unipass__sse(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_x32_packx_ukernel_4x__sse, 4, 8,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_ppmm_4x8_twopass__sse(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x8__sse, xnn_x32_packx_ukernel_4x__sse, 4, 8,
      xnn_init_f32_minmax_sse_params);
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

  BENCHMARK_GEMM(f32_gemm_1x16s4__fma3_broadcast)
  BENCHMARK_GEMM(f32_gemm_3x16s4__fma3_broadcast)
  BENCHMARK_GEMM(f32_gemm_4x16s4__fma3_broadcast)
  BENCHMARK_GEMM(f32_gemm_5x16s4__fma3_broadcast)

  BENCHMARK_GEMM(f32_gemm_1x8__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_4x8__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_5x8__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_6x8__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_7x8__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_1x16__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_3x16__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_4x16__avx_broadcast)
  BENCHMARK_GEMM(f32_gemm_5x16__avx_broadcast)

  BENCHMARK_GEMM(f32_gemm_1x8__sse2_dup)
  BENCHMARK_GEMM(f32_gemm_3x8__sse2_dup)
  BENCHMARK_GEMM(f32_gemm_4x8__sse2_dup)
  BENCHMARK_GEMM(f32_gemm_5x8__sse2_dup)

  BENCHMARK_GEMM(f32_gemm_1x8__sse_load1)
  BENCHMARK_GEMM(f32_gemm_3x8__sse_load1)
  BENCHMARK_GEMM(f32_gemm_4x8__sse_load1)
  BENCHMARK_GEMM(f32_gemm_5x8__sse_load1)

  BENCHMARK_GEMM(f32_gemm_1x8__sse_dup)
  BENCHMARK_GEMM(f32_gemm_3x8__sse_dup)
  BENCHMARK_GEMM(f32_gemm_4x8__sse_dup)
  BENCHMARK_GEMM(f32_gemm_5x8__sse_dup)

  BENCHMARK_GEMM(f32_gemm_1x8s4__sse)
  BENCHMARK_GEMM(f32_gemm_3x8s4__sse)
  BENCHMARK_GEMM(f32_gemm_4x8s4__sse)
  BENCHMARK_GEMM(f32_gemm_5x8s4__sse)

  BENCHMARK_GEMM(f32_ppmm_4x8_unipass__sse)
  BENCHMARK_GEMM(f32_ppmm_4x8_twopass__sse)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_gemm_3x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_arm_loadsplat, 3, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_4x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_arm_loadsplat, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_5x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_loadsplat, 5, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_6x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_arm_loadsplat, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_3x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_x86_loadsplat, 3, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_4x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_5x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_x86_loadsplat, 5, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_6x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_x86_loadsplat, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_3x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_arm_splat, 3, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_4x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_arm_splat, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_5x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_splat, 5, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_6x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_arm_splat, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_3x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_x86_splat, 3, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_4x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_splat, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_5x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_x86_splat, 5, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_6x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_x86_splat, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_3x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_3x8s4__wasmsimd_arm, 3, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_4x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8s4__wasmsimd_arm, 4, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_5x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8s4__wasmsimd_arm, 5, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_6x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8s4__wasmsimd_arm, 6, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_3x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_3x8s4__wasmsimd_x86, 3, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_4x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x8s4__wasmsimd_x86, 4, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_5x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_5x8s4__wasmsimd_x86, 5, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_gemm_6x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_6x8s4__wasmsimd_x86, 6, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_ppmm_4x8_unipass__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_x32_packx_ukernel_4x__wasmsimd, 4, 8,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_ppmm_4x8_unipass__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    PPMM1PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_x32_packx_ukernel_4x__wasmsimd, 4, 8,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_ppmm_4x8_twopass__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat, xnn_x32_packx_ukernel_4x__wasmsimd, 4, 8,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_ppmm_4x8_twopass__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    PPMM2PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat, xnn_x32_packx_ukernel_4x__wasmsimd, 4, 8,
      xnn_init_f32_minmax_scalar_params);
  }

  BENCHMARK_GEMM(f32_gemm_3x8__wasmsimd_arm_loadsplat)
  BENCHMARK_GEMM(f32_gemm_4x8__wasmsimd_arm_loadsplat)
  BENCHMARK_GEMM(f32_gemm_5x8__wasmsimd_arm_loadsplat)
  BENCHMARK_GEMM(f32_gemm_6x8__wasmsimd_arm_loadsplat)
  BENCHMARK_GEMM(f32_gemm_3x8__wasmsimd_x86_loadsplat)
  BENCHMARK_GEMM(f32_gemm_4x8__wasmsimd_x86_loadsplat)
  BENCHMARK_GEMM(f32_gemm_5x8__wasmsimd_x86_loadsplat)
  BENCHMARK_GEMM(f32_gemm_6x8__wasmsimd_x86_loadsplat)
  BENCHMARK_GEMM(f32_gemm_3x8__wasmsimd_arm_splat)
  BENCHMARK_GEMM(f32_gemm_4x8__wasmsimd_arm_splat)
  BENCHMARK_GEMM(f32_gemm_5x8__wasmsimd_arm_splat)
  BENCHMARK_GEMM(f32_gemm_6x8__wasmsimd_arm_splat)
  BENCHMARK_GEMM(f32_gemm_3x8__wasmsimd_x86_splat)
  BENCHMARK_GEMM(f32_gemm_4x8__wasmsimd_x86_splat)
  BENCHMARK_GEMM(f32_gemm_5x8__wasmsimd_x86_splat)
  BENCHMARK_GEMM(f32_gemm_6x8__wasmsimd_x86_splat)
  BENCHMARK_GEMM(f32_gemm_3x8s4__wasmsimd_arm)
  BENCHMARK_GEMM(f32_gemm_4x8s4__wasmsimd_arm)
  BENCHMARK_GEMM(f32_gemm_5x8s4__wasmsimd_arm)
  BENCHMARK_GEMM(f32_gemm_6x8s4__wasmsimd_arm)
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
  GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_1x4__scalar, 1, 4, 1, 1,
    xnn_init_f32_minmax_scalar_params);
}
static void f32_gemm_2x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_2x4__scalar, 2, 4, 1, 1,
    xnn_init_f32_minmax_scalar_params);
}
static void f32_gemm_4x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state, xnn_f32_gemm_minmax_ukernel_4x4__scalar, 4, 4, 1, 1,
    xnn_init_f32_minmax_scalar_params);
}

static void f32_ppmm_2x4_unipass__scalar(benchmark::State& state, const char* net) {
  PPMM1PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_x32_packx_ukernel_2x__scalar, 2, 4,
    xnn_init_f32_minmax_scalar_params);
}
static void f32_ppmm_4x2_unipass__scalar(benchmark::State& state, const char* net) {
  PPMM1PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_x32_packx_ukernel_4x__scalar, 4, 2,
    xnn_init_f32_minmax_scalar_params);
}
static void f32_ppmm_4x4_unipass__scalar(benchmark::State& state, const char* net) {
  PPMM1PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_x32_packx_ukernel_4x__scalar, 4, 4,
    xnn_init_f32_minmax_scalar_params);
}
static void f32_ppmm_3x3_unipass__scalar(benchmark::State& state, const char* net) {
  PPMM1PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_x32_packx_ukernel_3x__scalar, 3, 3,
    xnn_init_f32_minmax_scalar_params);
}

static void f32_ppmm_2x4_twopass__scalar(benchmark::State& state, const char* net) {
  PPMM2PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_2x4__scalar, xnn_x32_packx_ukernel_2x__scalar, 2, 4,
    xnn_init_f32_minmax_scalar_params);
}
static void f32_ppmm_4x2_twopass__scalar(benchmark::State& state, const char* net) {
  PPMM2PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x2__scalar, xnn_x32_packx_ukernel_4x__scalar, 4, 2,
    xnn_init_f32_minmax_scalar_params);
}
static void f32_ppmm_4x4_twopass__scalar(benchmark::State& state, const char* net) {
  PPMM2PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_4x4__scalar, xnn_x32_packx_ukernel_4x__scalar, 4, 4,
    xnn_init_f32_minmax_scalar_params);
}
static void f32_ppmm_3x3_twopass__scalar(benchmark::State& state, const char* net) {
  PPMM2PBenchmark(state, xnn_f32_ppmm_minmax_ukernel_3x3__scalar, xnn_x32_packx_ukernel_3x__scalar, 3, 3,
    xnn_init_f32_minmax_scalar_params);
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
