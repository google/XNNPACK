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

#include <xnnpack/aligned-allocator.h>
#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/gemm.h>
#include <xnnpack/math.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/pack.h>
#include <xnnpack/packw.h>


static void f32_gemm(benchmark::State& state,
  xnn_x32_packw_gemm_goi_ukernel_fn packw,
  xnn_f32_gemm_minmax_ukernel_fn gemm,
  xnn_init_f32_minmax_params_fn init_params,
  size_t mr, size_t nr, size_t kr, size_t sr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t batch = state.range(0);
  const size_t dim_m = state.range(1);
  const size_t dim_n = state.range(2);
  const size_t dim_k = state.range(3);

  const size_t stride_n = benchmark::utils::RoundUp(dim_n, nr);
  const size_t stride_k = benchmark::utils::RoundUp(dim_k, kr * sr);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<float> a(batch * dim_m * dim_k + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::vector<float> b(batch * dim_n * dim_k);
  std::generate(b.begin(), b.end(), std::ref(f32rng));
  std::vector<float> c(batch * dim_m * dim_n);
  std::fill(c.begin(), c.end(), std::nanf(""));

  const size_t w_elements = stride_n * stride_k + stride_n;
  std::vector<float, AlignedAllocator<float, 64>> w(batch * w_elements);
  std::fill(w.begin(), w.end(), 0.0f);

  xnn_f32_minmax_params params;
  init_params(&params,
    -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

  for (auto _ : state) {
    packw(batch, dim_n, dim_k, nr, kr, sr,
      reinterpret_cast<const uint32_t*>(b.data()), /*bias=*/nullptr,
      reinterpret_cast<uint32_t*>(w.data()),
      /*extra_bytes=*/0, nullptr);

    for (size_t i = 0; i < batch; i++) {
      for (size_t m = 0; m < dim_m; m += mr) {
        const size_t mb = min(dim_m - m, mr);
        gemm(
          mb, dim_n, dim_k * sizeof(float),
          a.data() + (i * dim_m + m) * dim_k, dim_k * sizeof(float),
          w.data() + i * stride_n * (stride_k + 1),
          c.data() + (i * dim_m + m) * dim_n, dim_n * sizeof(float), nr * sizeof(float),
          &params);
      }
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * batch * dim_m * dim_n * dim_k, benchmark::Counter::kIsRate);
}

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


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void f32_gemm_4x12__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_4x12__asm_aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/12, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_prfm_cortex_a53(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_prfm_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_cortex_a55(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_prfm_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__asm_aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4,
      xnn_f32_gemm_minmax_ukernel_5x8__asm_aarch64_neonfma_prfm_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_prfm_cortex_a53(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_prfm_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_cortex_a55(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_cortex_a73(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_prfm_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  BENCHMARK_BGEMM(f32_gemm_4x8__asm_aarch64_neonfma_cortex_a53)
  BENCHMARK_BGEMM(f32_gemm_4x8__asm_aarch64_neonfma_prfm_cortex_a53)
  BENCHMARK_BGEMM(f32_gemm_4x8__asm_aarch64_neonfma_cortex_a55)
  BENCHMARK_BGEMM(f32_gemm_4x8__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_BGEMM(f32_gemm_4x8__asm_aarch64_neonfma_prfm_cortex_a75)
  BENCHMARK_BGEMM(f32_gemm_4x8__asm_aarch64_neonfma_ld128)
  BENCHMARK_BGEMM(f32_gemm_4x8__asm_aarch64_neonfma_ld64)
  BENCHMARK_BGEMM(f32_gemm_4x12__asm_aarch64_neonfma_cortex_a53)
  BENCHMARK_BGEMM(f32_gemm_5x8__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_BGEMM(f32_gemm_5x8__asm_aarch64_neonfma_prfm_cortex_a75)
  BENCHMARK_BGEMM(f32_gemm_6x8__asm_aarch64_neonfma_cortex_a53)
  BENCHMARK_BGEMM(f32_gemm_6x8__asm_aarch64_neonfma_prfm_cortex_a53)
  BENCHMARK_BGEMM(f32_gemm_6x8__asm_aarch64_neonfma_cortex_a55)
  BENCHMARK_BGEMM(f32_gemm_6x8__asm_aarch64_neonfma_cortex_a73)
  BENCHMARK_BGEMM(f32_gemm_6x8__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_BGEMM(f32_gemm_6x8__asm_aarch64_neonfma_prfm_cortex_a75)
  BENCHMARK_BGEMM(f32_gemm_6x8__asm_aarch64_neonfma_ld64)
  BENCHMARK_BGEMM(f32_gemm_6x8__asm_aarch64_neonfma_ld128)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY

#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void f32_gemm_4x4__asm_aarch32_vfp_ld64(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_x4,
      xnn_f32_gemm_minmax_ukernel_4x4__asm_aarch32_vfp_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckVFP);
  }

  static void f32_gemm_4x8__asm_aarch32_neon_ld64(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__asm_aarch32_neon_cortex_a7(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__asm_aarch32_neon_cortex_a53(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__asm_aarch32_neon_prfm_cortex_a53(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_prfm_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__asm_aarch32_neon_cortex_a55(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__asm_aarch32_neon_cortex_a75(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__asm_aarch32_neon_prfm_cortex_a75(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_prfm_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_BGEMM(f32_gemm_4x4__asm_aarch32_vfp_ld64)
  BENCHMARK_BGEMM(f32_gemm_4x8__asm_aarch32_neon_ld64)
  BENCHMARK_BGEMM(f32_gemm_4x8__asm_aarch32_neon_cortex_a7)
  BENCHMARK_BGEMM(f32_gemm_4x8__asm_aarch32_neon_cortex_a53)
  BENCHMARK_BGEMM(f32_gemm_4x8__asm_aarch32_neon_prfm_cortex_a53)
  BENCHMARK_BGEMM(f32_gemm_4x8__asm_aarch32_neon_cortex_a55)
  BENCHMARK_BGEMM(f32_gemm_4x8__asm_aarch32_neon_cortex_a75)
  BENCHMARK_BGEMM(f32_gemm_4x8__asm_aarch32_neon_prfm_cortex_a75)
#endif  // XNN_ARCH_ARM

#if XNN_ARCH_ARM64
  static void f32_gemm_4x8__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  BENCHMARK_BGEMM(f32_gemm_4x8__aarch64_neonfma_lane_ld64)
  BENCHMARK_BGEMM(f32_gemm_4x8__aarch64_neonfma_lane_ld128)
  BENCHMARK_BGEMM(f32_gemm_5x8__aarch64_neonfma_lane_ld64)
  BENCHMARK_BGEMM(f32_gemm_6x8__aarch64_neonfma_lane_ld64)
  BENCHMARK_BGEMM(f32_gemm_6x8__aarch64_neonfma_lane_ld128)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_gemm_4x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__neon_lane_ld128(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_5x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_5x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_6x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_6x8__neon_lane_ld128(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__neon_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__neonfma_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_4x8__neonfma_dup_ld128(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__neonfma_dup_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_6x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__neonfma_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_6x8__neonfma_dup_ld128(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__neonfma_dup_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }

  static void f32_gemm_4x8s4__neon(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_4x8s4__neon,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_6x8s4__neon(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_6x8s4__neon,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_4x8s4__neonfma(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_4x8s4__neonfma,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_gemm_6x8s4__neonfma(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4,
      xnn_f32_gemm_minmax_ukernel_6x8s4__neonfma,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_BGEMM(f32_gemm_4x8__neon_lane_ld64)
  BENCHMARK_BGEMM(f32_gemm_4x8__neon_lane_ld128)
  BENCHMARK_BGEMM(f32_gemm_5x8__neon_lane_ld64)
  BENCHMARK_BGEMM(f32_gemm_6x8__neon_lane_ld64)
  BENCHMARK_BGEMM(f32_gemm_6x8__neon_lane_ld128)

  BENCHMARK_BGEMM(f32_gemm_4x8__neonfma_dup_ld64)
  BENCHMARK_BGEMM(f32_gemm_4x8__neonfma_dup_ld128)
  BENCHMARK_BGEMM(f32_gemm_6x8__neonfma_dup_ld64)
  BENCHMARK_BGEMM(f32_gemm_6x8__neonfma_dup_ld128)

  BENCHMARK_BGEMM(f32_gemm_4x8s4__neon)
  BENCHMARK_BGEMM(f32_gemm_6x8s4__neon)
  BENCHMARK_BGEMM(f32_gemm_4x8s4__neonfma)
  BENCHMARK_BGEMM(f32_gemm_6x8s4__neonfma)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_gemm_3x8__wasmrelaxedsimd_loadsplat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__wasmrelaxedsimd_loadsplat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__wasmrelaxedsimd_loadsplat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__wasmrelaxedsimd_loadsplat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_3x8__wasmrelaxedsimd_fma_loadsplat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__wasmrelaxedsimd_fma_loadsplat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__wasmrelaxedsimd_fma_loadsplat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__wasmrelaxedsimd_fma_loadsplat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_3x8__wasmrelaxedsimd_splat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__wasmrelaxedsimd_splat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__wasmrelaxedsimd_splat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__wasmrelaxedsimd_splat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_3x8__wasmrelaxedsimd_fma_splat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__wasmrelaxedsimd_fma_splat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__wasmrelaxedsimd_fma_splat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__wasmrelaxedsimd_fma_splat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_3x8s4__wasmrelaxedsimd(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_4x8s4__wasmrelaxedsimd(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_5x8s4__wasmrelaxedsimd(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_6x8s4__wasmrelaxedsimd(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_3x8s4__wasmrelaxedsimd_fma(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd_fma,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_4x8s4__wasmrelaxedsimd_fma(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd_fma,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_5x8s4__wasmrelaxedsimd_fma(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd_fma,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_6x8s4__wasmrelaxedsimd_fma(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd_fma,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }

  BENCHMARK_BGEMM(f32_gemm_3x8__wasmrelaxedsimd_loadsplat)
  BENCHMARK_BGEMM(f32_gemm_4x8__wasmrelaxedsimd_loadsplat)
  BENCHMARK_BGEMM(f32_gemm_5x8__wasmrelaxedsimd_loadsplat)
  BENCHMARK_BGEMM(f32_gemm_6x8__wasmrelaxedsimd_loadsplat)

  BENCHMARK_BGEMM(f32_gemm_3x8__wasmrelaxedsimd_fma_loadsplat)
  BENCHMARK_BGEMM(f32_gemm_4x8__wasmrelaxedsimd_fma_loadsplat)
  BENCHMARK_BGEMM(f32_gemm_5x8__wasmrelaxedsimd_fma_loadsplat)
  BENCHMARK_BGEMM(f32_gemm_6x8__wasmrelaxedsimd_fma_loadsplat)

  BENCHMARK_BGEMM(f32_gemm_3x8__wasmrelaxedsimd_splat)
  BENCHMARK_BGEMM(f32_gemm_4x8__wasmrelaxedsimd_splat)
  BENCHMARK_BGEMM(f32_gemm_5x8__wasmrelaxedsimd_splat)
  BENCHMARK_BGEMM(f32_gemm_6x8__wasmrelaxedsimd_splat)

  BENCHMARK_BGEMM(f32_gemm_3x8__wasmrelaxedsimd_fma_splat)
  BENCHMARK_BGEMM(f32_gemm_4x8__wasmrelaxedsimd_fma_splat)
  BENCHMARK_BGEMM(f32_gemm_5x8__wasmrelaxedsimd_fma_splat)
  BENCHMARK_BGEMM(f32_gemm_6x8__wasmrelaxedsimd_fma_splat)

  BENCHMARK_BGEMM(f32_gemm_3x8s4__wasmrelaxedsimd)
  BENCHMARK_BGEMM(f32_gemm_4x8s4__wasmrelaxedsimd)
  BENCHMARK_BGEMM(f32_gemm_5x8s4__wasmrelaxedsimd)
  BENCHMARK_BGEMM(f32_gemm_6x8s4__wasmrelaxedsimd)

  BENCHMARK_BGEMM(f32_gemm_3x8s4__wasmrelaxedsimd_fma)
  BENCHMARK_BGEMM(f32_gemm_4x8s4__wasmrelaxedsimd_fma)
  BENCHMARK_BGEMM(f32_gemm_5x8s4__wasmrelaxedsimd_fma)
  BENCHMARK_BGEMM(f32_gemm_6x8s4__wasmrelaxedsimd_fma)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_gemm_3x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_3x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_3x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_3x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_4x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_5x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_6x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_gemm_3x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_3x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_4x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_4x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_5x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_5x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_6x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_6x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_3x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_3x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_4x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_4x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_5x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_5x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_gemm_6x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    f32_gemm(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4,
      xnn_f32_gemm_minmax_ukernel_6x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }

  BENCHMARK_BGEMM(f32_gemm_3x8__wasmsimd_arm_loadsplat)
  BENCHMARK_BGEMM(f32_gemm_4x8__wasmsimd_arm_loadsplat)
  BENCHMARK_BGEMM(f32_gemm_5x8__wasmsimd_arm_loadsplat)
  BENCHMARK_BGEMM(f32_gemm_6x8__wasmsimd_arm_loadsplat)

  BENCHMARK_BGEMM(f32_gemm_3x8__wasmsimd_x86_loadsplat)
  BENCHMARK_BGEMM(f32_gemm_4x8__wasmsimd_x86_loadsplat)
  BENCHMARK_BGEMM(f32_gemm_5x8__wasmsimd_x86_loadsplat)
  BENCHMARK_BGEMM(f32_gemm_6x8__wasmsimd_x86_loadsplat)

  BENCHMARK_BGEMM(f32_gemm_3x8__wasmsimd_arm_splat)
  BENCHMARK_BGEMM(f32_gemm_4x8__wasmsimd_arm_splat)
  BENCHMARK_BGEMM(f32_gemm_5x8__wasmsimd_arm_splat)
  BENCHMARK_BGEMM(f32_gemm_6x8__wasmsimd_arm_splat)

  BENCHMARK_BGEMM(f32_gemm_3x8__wasmsimd_x86_splat)
  BENCHMARK_BGEMM(f32_gemm_4x8__wasmsimd_x86_splat)
  BENCHMARK_BGEMM(f32_gemm_5x8__wasmsimd_x86_splat)
  BENCHMARK_BGEMM(f32_gemm_6x8__wasmsimd_x86_splat)

  BENCHMARK_BGEMM(f32_gemm_3x8s4__wasmsimd_arm)
  BENCHMARK_BGEMM(f32_gemm_4x8s4__wasmsimd_arm)
  BENCHMARK_BGEMM(f32_gemm_5x8s4__wasmsimd_arm)
  BENCHMARK_BGEMM(f32_gemm_6x8s4__wasmsimd_arm)

  BENCHMARK_BGEMM(f32_gemm_3x8s4__wasmsimd_x86)
  BENCHMARK_BGEMM(f32_gemm_4x8s4__wasmsimd_x86)
  BENCHMARK_BGEMM(f32_gemm_5x8s4__wasmsimd_x86)
  BENCHMARK_BGEMM(f32_gemm_6x8s4__wasmsimd_x86)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

static void f32_gemm_4x2__scalar(benchmark::State& state, const char* net) {
  f32_gemm(state,
    xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_x4,
    xnn_f32_gemm_minmax_ukernel_4x2__scalar,
    xnn_init_f32_minmax_scalar_params,
    /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1);
}
static void f32_gemm_2x4__scalar(benchmark::State& state, const char* net) {
  f32_gemm(state,
    xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_x4,
    xnn_f32_gemm_minmax_ukernel_2x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1);
}
static void f32_gemm_4x4__scalar(benchmark::State& state, const char* net) {
  f32_gemm(state,
    xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_x4,
    xnn_f32_gemm_minmax_ukernel_4x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1);
}

BENCHMARK_BGEMM(f32_gemm_4x2__scalar)
BENCHMARK_BGEMM(f32_gemm_2x4__scalar)
BENCHMARK_BGEMM(f32_gemm_4x4__scalar)


#ifdef BENCHMARK_RUY
BENCHMARK_BGEMM(ruy_st)
#endif  // BENCHMARK_RUY

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
