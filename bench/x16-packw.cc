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
      /*bias=*/nullptr, /*scale=*/nullptr,
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
  static void x16_packw_x16__neon_ld4lane_u4(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x16__neon_ld4lane_u4_prfm(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x8__neon_ld4lane_u4(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x8__neon_ld4lane_u4_prfm(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x16__neon_ld4lane_u8(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x16__neon_ld4lane_u8_prfm(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x8__neon_ld4lane_u8(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x8__neon_ld4lane_u8_prfm(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x16__neon_ld4lane_u12(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x16__neon_ld4lane_u12_prfm(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u12_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x8__neon_ld4lane_u12(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x8__neon_ld4lane_u12_prfm(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u12_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x16__neon_ld4lane_u16(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x16__neon_ld4lane_u16_prfm(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u16_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x8__neon_ld4lane_u16(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x16_packw_x8__neon_ld4lane_u16_prfm(benchmark::State& state, const char* net) {
    x16_packw(state,
      xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u16_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_BGEMM(x16_packw_x8__neon_ld4lane_u4)
  BENCHMARK_BGEMM(x16_packw_x8__neon_ld4lane_u4_prfm)
  BENCHMARK_BGEMM(x16_packw_x16__neon_ld4lane_u4)
  BENCHMARK_BGEMM(x16_packw_x16__neon_ld4lane_u4_prfm)
  BENCHMARK_BGEMM(x16_packw_x8__neon_ld4lane_u8)
  BENCHMARK_BGEMM(x16_packw_x8__neon_ld4lane_u8_prfm)
  BENCHMARK_BGEMM(x16_packw_x16__neon_ld4lane_u8)
  BENCHMARK_BGEMM(x16_packw_x16__neon_ld4lane_u8_prfm)
  BENCHMARK_BGEMM(x16_packw_x8__neon_ld4lane_u12)
  BENCHMARK_BGEMM(x16_packw_x8__neon_ld4lane_u12_prfm)
  BENCHMARK_BGEMM(x16_packw_x16__neon_ld4lane_u12)
  BENCHMARK_BGEMM(x16_packw_x16__neon_ld4lane_u12_prfm)
  BENCHMARK_BGEMM(x16_packw_x8__neon_ld4lane_u16)
  BENCHMARK_BGEMM(x16_packw_x8__neon_ld4lane_u16_prfm)
  BENCHMARK_BGEMM(x16_packw_x16__neon_ld4lane_u16)
  BENCHMARK_BGEMM(x16_packw_x16__neon_ld4lane_u16_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x16_packw_x8__avx2_int_u16(benchmark::State& state,
                                          const char* net) {
    x16_packw(state, xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16,
              /*nr=*/8, /*kr=*/1, /*sr=*/1, benchmark::utils::CheckAVX2);
  }
  static void x16_packw_x8__avx2_int_u16_prfm(benchmark::State& state,
                                               const char* net) {
    x16_packw(state, xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16_prfm,
              /*nr=*/8, /*kr=*/1, /*sr=*/1, benchmark::utils::CheckAVX2);
  }
  static void x16_packw_x16__avx2_int_u16(benchmark::State& state,
                                          const char* net) {
    x16_packw(state, xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16,
              /*nr=*/16, /*kr=*/1, /*sr=*/1, benchmark::utils::CheckAVX2);
  }
  static void x16_packw_x16__avx2_int_u16_prfm(benchmark::State& state,
                                               const char* net) {
    x16_packw(state, xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm,
              /*nr=*/16, /*kr=*/1, /*sr=*/1, benchmark::utils::CheckAVX2);
  }
  BENCHMARK_BGEMM(x16_packw_x8__avx2_int_u16)
  BENCHMARK_BGEMM(x16_packw_x8__avx2_int_u16_prfm)
  BENCHMARK_BGEMM(x16_packw_x16__avx2_int_u16)
  BENCHMARK_BGEMM(x16_packw_x16__avx2_int_u16_prfm)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

  static void x16_packw_x16__scalar_int_u4(benchmark::State& state,
                                           const char* net) {
    x16_packw(state, xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int_u4,
              /*nr=*/16, /*kr=*/1, /*sr=*/1);
  }
static void x16_packw_x8__scalar_int_u4(benchmark::State& state, const char* net) {
  x16_packw(state,
    xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int_u4,
    /*nr=*/8, /*kr=*/1, /*sr=*/1);
}

BENCHMARK_BGEMM(x16_packw_x16__scalar_int_u4)
BENCHMARK_BGEMM(x16_packw_x8__scalar_int_u4)

void x16_packw__reference(
  size_t batch,
  size_t dim_n,
  size_t dim_k,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* weights,
  const uint16_t* bias,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  xnn_pack_f16_gemm_goi_w(batch, dim_n, dim_k, nr, kr, sr,
     weights, bias, scale, packed_weights, extra_bytes, params);
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
