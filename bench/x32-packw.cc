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
#include "bench/bgemm.h"
#include "bench/utils.h"

#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/pack.h>
#include <xnnpack/packw.h>


static void x32_packw(benchmark::State& state,
  xnn_x32_packw_gemm_goi_ukernel_fn packw,
  size_t nr, size_t kr, size_t sr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t batch = state.range(0);
  const size_t dim_n = state.range(2);
  const size_t dim_k = state.range(3);

  const size_t stride_n = benchmark::utils::RoundUp(dim_n, nr);
  const size_t stride_k = benchmark::utils::RoundUp(dim_k, kr * sr);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<float> b(batch * dim_n * dim_k);
  std::generate(b.begin(), b.end(), std::ref(f32rng));

  const size_t w_elements = stride_n * stride_k + stride_n;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * batch * (w_elements + dim_n * dim_k));

  std::vector<float, AlignedAllocator<float, 64>> w(num_buffers * batch * w_elements);
  std::fill(w.begin(), w.end(), 0.0f);

  size_t buffer_index = 0;
  for (auto _ : state) {
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    packw(batch, dim_n, dim_k, nr, kr, sr,
      reinterpret_cast<const uint32_t*>(b.data()),
      /*bias=*/nullptr,
      reinterpret_cast<uint32_t*>(w.data() + buffer_index * batch * w_elements),
      /*extra_bytes=*/0, nullptr);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = batch * dim_n * dim_k;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = batch * (dim_n * dim_k + w_elements) * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x32_packw_x8__neon(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x32_packw_x12__neon(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x12__neon,
      /*nr=*/12, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_BGEMM(x32_packw_x8__neon)
  BENCHMARK_BGEMM(x32_packw_x12__neon)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

static void x32_packw_x2__scalar_float(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float,
    /*nr=*/2, /*kr=*/1, /*sr=*/1);
}
static void x32_packw_x2__scalar_int(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int,
    /*nr=*/2, /*kr=*/1, /*sr=*/1);
}
static void x32_packw_x4__scalar_float(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float,
    /*nr=*/4, /*kr=*/1, /*sr=*/1);
}
static void x32_packw_x4__scalar_int(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int,
    /*nr=*/4, /*kr=*/1, /*sr=*/1);
}

BENCHMARK_BGEMM(x32_packw_x2__scalar_float)
BENCHMARK_BGEMM(x32_packw_x2__scalar_int)
BENCHMARK_BGEMM(x32_packw_x4__scalar_float)
BENCHMARK_BGEMM(x32_packw_x4__scalar_int)


#ifdef BENCHMARK_RUY
BENCHMARK_BGEMM(ruy_st)
#endif  // BENCHMARK_RUY

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
