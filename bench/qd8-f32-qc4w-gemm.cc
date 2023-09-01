// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
//
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "bench/gemm.h"
#include "bench/utils.h"

#include <xnnpack/aligned-allocator.h>
#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/gemm.h>
#include <xnnpack/math.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/pack.h>


void GEMMBenchmark(benchmark::State& state,
  xnn_qd8_f32_qc4w_gemm_minmax_ukernel_fn gemm,
  xnn_init_f32_qc4w_minmax_params_fn init_params,
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
  const size_t kc_stride = benchmark::utils::RoundUp(kc, kr * sr) / 2;

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(-std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max()), std::ref(rng));
  auto u8rng = std::bind(
    std::uniform_int_distribution<int32_t>(0, std::numeric_limits<uint8_t>::max()), std::ref(rng));

  std::vector<int8_t> a(mc * kc + XNN_EXTRA_BYTES);
  std::generate(a.begin(), a.end(), std::ref(i8rng));
  std::vector<uint8_t> k(nc * kc / 2);
  std::generate(k.begin(), k.end(), std::ref(u8rng));

  std::vector<xnn_qd8_quantization_params> quantization_params(mc + XNN_EXTRA_QUANTIZATION_PARAMS);
  const size_t w_elements = nc_stride * (sizeof(float) * 2 + sizeof(int32_t)) + kc_stride * nc_stride;

  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (w_elements + c_elements));

  std::vector<char, AlignedAllocator<char, 64>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0);

  const xnn_qs8_packing_params packing_params = { /*input_zero_point=*/1 };
  // Note that bias will be incorrect with qs8 pack.  Use qc4w variation when available
  xnn_pack_qs8_gemm_goi_w(1, nc, kc / 2, nr, kr, sr,
                          (const int8_t*) k.data(), /*bias=*/nullptr, /*scale=*/nullptr, w.data(), sizeof(float) * 2 * nr, &packing_params);
  std::vector<float> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  // Prepare parameters.
  xnn_f32_qc4w_minmax_params params;
  init_params(&params, std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max(), 0);

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size());
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      gemm(
        mb, nc, kc,
        a.data() + m * kc, kc * sizeof(int8_t),
        w.data() + w_elements * buffer_index,
        c.data() + (buffer_index * mc + m) * nc, nc * sizeof(float), nr * sizeof(float),
        &params, quantization_params.data() + m);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["OPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}

static void qd8_f32_qc4w_gemm_1x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1);
}

static void qd8_f32_qc4w_gemm_1x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1);
}

static void qd8_f32_qc4w_gemm_1x8__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
}

static void qd8_f32_qc4w_gemm_2x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1);
}

static void qd8_f32_qc4w_gemm_2x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1);
}

static void qd8_f32_qc4w_gemm_2x8__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1);
}

static void qd8_f32_qc4w_gemm_4x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1);
}

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc4w_gemm_1x2__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1);
  }

  static void qd8_f32_qc4w_gemm_1x4__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1);
  }

  static void qd8_f32_qc4w_gemm_1x8__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void qd8_f32_qc4w_gemm_2x2__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1);
  }

  static void qd8_f32_qc4w_gemm_2x4__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1);
  }

  static void qd8_f32_qc4w_gemm_2x8__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void qd8_f32_qc4w_gemm_4x4__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1);
  }
#endif // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_GEMM(qd8_f32_qc4w_gemm_1x2__scalar)
BENCHMARK_GEMM(qd8_f32_qc4w_gemm_1x4__scalar)
BENCHMARK_GEMM(qd8_f32_qc4w_gemm_1x8__scalar)
BENCHMARK_GEMM(qd8_f32_qc4w_gemm_2x2__scalar)
BENCHMARK_GEMM(qd8_f32_qc4w_gemm_2x4__scalar)
BENCHMARK_GEMM(qd8_f32_qc4w_gemm_2x8__scalar)
BENCHMARK_GEMM(qd8_f32_qc4w_gemm_4x4__scalar)

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_1x2__wasm)
  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_1x4__wasm)
  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_1x8__wasm)
  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_2x2__wasm)
  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_2x4__wasm)
  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_2x8__wasm)
  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_4x4__wasm)
#endif // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
