// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
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


static void x16_packw(benchmark::State& state,
  xnn_x16_packw_gemm_goi_ukernel_fn packw,
  size_t nr, size_t kr, size_t sr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t batch = state.range(0);  // batch is g parameter for packw
  const size_t dim_n = state.range(2);  // dim_n is nc parameter
  const size_t dim_k = state.range(3);  // dim_k is kc parameter

  const size_t rounded_n = benchmark::utils::RoundUp(dim_n, nr);
  const size_t rounded_k = benchmark::utils::RoundUp(dim_k, kr * sr);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto u16rng = std::bind(std::uniform_int_distribution<uint16_t>(), std::ref(rng));

  // Computer num_buffers that fit cache with source weights + packed_weights.
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(uint16_t) * batch * (dim_n * dim_k + rounded_n * rounded_k + rounded_n));

  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> weights(num_buffers * batch * dim_n * dim_k);
  std::generate(weights.begin(), weights.end(), std::ref(u16rng));
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> packed_weights(num_buffers * batch * (rounded_n * rounded_k + rounded_n));
  std::fill(packed_weights.begin(), packed_weights.end(), 0);

  size_t buffer_index = 0;
  for (auto _ : state) {
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    packw(batch, dim_n, dim_k, nr, kr, sr,
      reinterpret_cast<uint16_t*>(weights.data() + buffer_index * batch * dim_n * dim_k),
       /*bias=*/nullptr,
      reinterpret_cast<uint16_t*>(packed_weights.data() + buffer_index * batch * (rounded_n * rounded_k + rounded_n)),
      /*extra_bytes=*/0, /*params=*/nullptr);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = batch * dim_n * dim_k;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = (elements_per_iteration + batch * (rounded_n * rounded_k + rounded_n)) * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void x16_packw_x16__neon_ld4lane_x8(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_x8,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x16__neon_ld4lane_prfm_x8(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_prfm_x8,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x8__neon_ld4lane_x8(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_x8,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x8__neon_ld4lane_prfm_x8(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x8,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x16__neon_ld4lane_x4(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x16__neon_ld4lane_prfm_x4(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_prfm_x4,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x8__neon_ld4lane_x4(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x8__neon_ld4lane_prfm_x4(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_prfm_x4,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_BGEMM(x16_packw_x16__neon_ld4lane_x4)
  BENCHMARK_BGEMM(x16_packw_x16__neon_ld4lane_prfm_x4)
  BENCHMARK_BGEMM(x16_packw_x16__neon_ld4lane_x8)
  BENCHMARK_BGEMM(x16_packw_x16__neon_ld4lane_prfm_x8)
  BENCHMARK_BGEMM(x16_packw_x8__neon_ld4lane_x4)
  BENCHMARK_BGEMM(x16_packw_x8__neon_ld4lane_prfm_x4)
  BENCHMARK_BGEMM(x16_packw_x8__neon_ld4lane_x8)
  BENCHMARK_BGEMM(x16_packw_x8__neon_ld4lane_prfm_x8)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

static void x16_packw_x16__scalar_int_x4(benchmark::State& state, const char* net) {
  x16_packw(state,
    xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_x4,
    /*nr=*/16, /*kr=*/1, /*sr=*/1);
}
static void x16_packw_x8__scalar_int_x4(benchmark::State& state, const char* net) {
  x16_packw(state,
    xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_x4,
    /*nr=*/8, /*kr=*/1, /*sr=*/1);
}

BENCHMARK_BGEMM(x16_packw_x16__scalar_int_x4)
BENCHMARK_BGEMM(x16_packw_x8__scalar_int_x4)

void x16_packw__reference(
  size_t batch,
  size_t dim_n,
  size_t dim_k,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* weights,
  const uint16_t* bias,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  xnn_pack_f16_gemm_goi_w(batch, dim_n, dim_k, nr, kr, sr,
     weights,
     bias,
     packed_weights,
     extra_bytes, params);
}

static void x16_packw_x8__reference(benchmark::State& state, const char* net) {
  x16_packw(state,
    x16_packw__reference,
    /*nr=*/8, /*kr=*/1, /*sr=*/1);
}
BENCHMARK_BGEMM(x16_packw_x8__reference)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif