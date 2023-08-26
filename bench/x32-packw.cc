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

  const size_t batch = state.range(0);  // batch is g parameter for packw
  const size_t dim_n = state.range(2);  // dim_n is nc parameter
  const size_t dim_k = state.range(3);  // dim_k is kc parameter

  const size_t rounded_n = benchmark::utils::RoundUp(dim_n, nr);
  const size_t rounded_k = benchmark::utils::RoundUp(dim_k, kr * sr);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  // Computer num_buffers that fit cache with source weights + packed_weights.
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * batch * (dim_n * dim_k + rounded_n * rounded_k + rounded_n));

  std::vector<float, AlignedAllocator<float, 64>> weights(num_buffers * batch * dim_n * dim_k);
  std::generate(weights.begin(), weights.end(), std::ref(f32rng));
  std::vector<float, AlignedAllocator<float, 64>> packed_weights(num_buffers * batch * (rounded_n * rounded_k + rounded_n));
  std::fill(packed_weights.begin(), packed_weights.end(), 0.0f);

  size_t buffer_index = 0;
  for (auto _ : state) {
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    packw(batch, dim_n, dim_k, nr, kr, sr,
      reinterpret_cast<uint32_t*>(weights.data() + buffer_index * batch * dim_n * dim_k),
      /*bias=*/nullptr,
      /*scale=*/nullptr,
      reinterpret_cast<uint32_t*>(packed_weights.data() + buffer_index * batch * (rounded_n * rounded_k + rounded_n)),
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
  static void x32_packw_x2__neon_ld2lane_u2_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2_prfm,
      /*nr=*/2, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x32_packw_x2__neon_ld2lane_u2(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2,
      /*nr=*/2, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x32_packw_x8__neon_ld4lane_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x32_packw_x8__neon_ld4lane_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x32_packw_x8__neon_ld4lane_u8_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x32_packw_x8__neon_ld4lane_u8(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x32_packw_x8s4__neon_ld4lane_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u4_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }
  static void x32_packw_x8s4__neon_ld4lane_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u4,
      /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }
  static void x32_packw_x8s4__neon_ld4lane_u8_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u8_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }
  static void x32_packw_x8s4__neon_ld4lane_u8(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u8,
      /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }
  static void x32_packw_x12__neon_ld4lane_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u4_prfm,
      /*nr=*/12, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x32_packw_x12__neon_ld4lane_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u4,
      /*nr=*/12, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x32_packw_x12__neon_ld4lane_u8_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u8_prfm,
      /*nr=*/12, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x32_packw_x12__neon_ld4lane_u8(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u8,
      /*nr=*/12, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x32_packw_x16__neon_ld4lane_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x32_packw_x16__neon_ld4lane_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x32_packw_x16__neon_ld4lane_u8_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void x32_packw_x16__neon_ld4lane_u8(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_BGEMM(x32_packw_x2__neon_ld2lane_u2_prfm)
  BENCHMARK_BGEMM(x32_packw_x2__neon_ld2lane_u2)
  BENCHMARK_BGEMM(x32_packw_x8__neon_ld4lane_u4_prfm)
  BENCHMARK_BGEMM(x32_packw_x8__neon_ld4lane_u4)
  BENCHMARK_BGEMM(x32_packw_x8__neon_ld4lane_u8_prfm)
  BENCHMARK_BGEMM(x32_packw_x8__neon_ld4lane_u8)
  BENCHMARK_BGEMM(x32_packw_x8s4__neon_ld4lane_u4_prfm)
  BENCHMARK_BGEMM(x32_packw_x8s4__neon_ld4lane_u4)
  BENCHMARK_BGEMM(x32_packw_x8s4__neon_ld4lane_u8_prfm)
  BENCHMARK_BGEMM(x32_packw_x8s4__neon_ld4lane_u8)
  BENCHMARK_BGEMM(x32_packw_x12__neon_ld4lane_u4_prfm)
  BENCHMARK_BGEMM(x32_packw_x12__neon_ld4lane_u4)
  BENCHMARK_BGEMM(x32_packw_x12__neon_ld4lane_u8_prfm)
  BENCHMARK_BGEMM(x32_packw_x12__neon_ld4lane_u8)
  BENCHMARK_BGEMM(x32_packw_x16__neon_ld4lane_u4_prfm)
  BENCHMARK_BGEMM(x32_packw_x16__neon_ld4lane_u4)
  BENCHMARK_BGEMM(x32_packw_x16__neon_ld4lane_u8_prfm)
  BENCHMARK_BGEMM(x32_packw_x16__neon_ld4lane_u8)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void x32_packw_x2c4__sse2_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_u4,
      /*nr=*/2, /*kr=*/4, /*sr=*/1);
  }
  static void x32_packw_x16__avx512f_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_u4,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }
  static void x32_packw_x8__avx_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__avx_u4,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }
  static void x32_packw_x16__avx_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__avx_u4,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }
  static void x32_packw_x8s4__avx_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_u4,
      /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckAVX);
  }
  static void x32_packw_x16s4__avx_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_u4,
      /*nr=*/16, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckAVX);
  }
  static void x32_packw_x2c4__sse2_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_u4_prfm,
      /*nr=*/2, /*kr=*/4, /*sr=*/1);
  }
  static void x32_packw_x16__avx512f_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_u4_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }
  static void x32_packw_x8__avx_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__avx_u4_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }
  static void x32_packw_x16__avx_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__avx_u4_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }
  static void x32_packw_x8s4__avx_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_u4_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckAVX);
  }
  static void x32_packw_x16s4__avx_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_u4_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckAVX);
  }
  static void x32_packw_x8__sse2_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u4,
      /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void x32_packw_x8__sse2_u8(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u8,
      /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void x32_packw_x16__sse2_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__sse2_u4,
      /*nr=*/16, /*kr=*/1, /*sr=*/1);
  }
  static void x32_packw_x16__sse2_u8(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__sse2_u8,
      /*nr=*/16, /*kr=*/1, /*sr=*/1);
  }
  static void x32_packw_x8s4__sse2_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_u4,
      /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void x32_packw_x8s4__sse2_u8(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_u8,
      /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void x32_packw_x16s4__sse2_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_u4,
      /*nr=*/16, /*kr=*/1, /*sr=*/4);
  }
  static void x32_packw_x16s4__sse2_u8(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_u8,
      /*nr=*/16, /*kr=*/1, /*sr=*/4);
  }
  static void x32_packw_x8__sse2_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u4_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void x32_packw_x8__sse2_u8_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u8_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void x32_packw_x16__sse2_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__sse2_u4_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/1);
  }
  static void x32_packw_x16__sse2_u8_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16__sse2_u8_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/1);
  }
  static void x32_packw_x8s4__sse2_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_u4_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void x32_packw_x8s4__sse2_u8_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_u8_prfm,
      /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void x32_packw_x16s4__sse2_u4_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_u4_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/4);
  }
  static void x32_packw_x16s4__sse2_u8_prfm(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_u8_prfm,
      /*nr=*/16, /*kr=*/1, /*sr=*/4);
  }

  BENCHMARK_BGEMM(x32_packw_x2c4__sse2_u4_prfm)
  BENCHMARK_BGEMM(x32_packw_x2c4__sse2_u4)
  BENCHMARK_BGEMM(x32_packw_x8__avx_u4_prfm)
  BENCHMARK_BGEMM(x32_packw_x8__avx_u4)
  BENCHMARK_BGEMM(x32_packw_x8__sse2_u4_prfm)
  BENCHMARK_BGEMM(x32_packw_x8__sse2_u8_prfm)
  BENCHMARK_BGEMM(x32_packw_x8__sse2_u4)
  BENCHMARK_BGEMM(x32_packw_x8__sse2_u8)
  BENCHMARK_BGEMM(x32_packw_x8s4__avx_u4_prfm)
  BENCHMARK_BGEMM(x32_packw_x8s4__avx_u4)
  BENCHMARK_BGEMM(x32_packw_x8s4__sse2_u4_prfm)
  BENCHMARK_BGEMM(x32_packw_x8s4__sse2_u8_prfm)
  BENCHMARK_BGEMM(x32_packw_x8s4__sse2_u4)
  BENCHMARK_BGEMM(x32_packw_x8s4__sse2_u8)
  BENCHMARK_BGEMM(x32_packw_x16__avx512f_u4_prfm)
  BENCHMARK_BGEMM(x32_packw_x16__avx512f_u4)
  BENCHMARK_BGEMM(x32_packw_x16__avx_u4_prfm)
  BENCHMARK_BGEMM(x32_packw_x16__avx_u4)
  BENCHMARK_BGEMM(x32_packw_x16__sse2_u4_prfm)
  BENCHMARK_BGEMM(x32_packw_x16__sse2_u8_prfm)
  BENCHMARK_BGEMM(x32_packw_x16__sse2_u4)
  BENCHMARK_BGEMM(x32_packw_x16__sse2_u8)
  BENCHMARK_BGEMM(x32_packw_x16s4__avx_u4_prfm)
  BENCHMARK_BGEMM(x32_packw_x16s4__avx_u4)
  BENCHMARK_BGEMM(x32_packw_x16s4__sse2_u4_prfm)
  BENCHMARK_BGEMM(x32_packw_x16s4__sse2_u8_prfm)
  BENCHMARK_BGEMM(x32_packw_x16s4__sse2_u4)
  BENCHMARK_BGEMM(x32_packw_x16s4__sse2_u8)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void x32_packw_x2c4__wasmsimd_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_u4,
      /*nr=*/2, /*kr=*/4, /*sr=*/1);
  }
  static void x32_packw_x8__wasmsimd_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_u4,
      /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void x32_packw_x8s4__wasmsimd_u4(benchmark::State& state, const char* net) {
    x32_packw(state,
      xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_u4,
      /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }

  BENCHMARK_BGEMM(x32_packw_x2c4__wasmsimd_u4)
  BENCHMARK_BGEMM(x32_packw_x8__wasmsimd_u4)
  BENCHMARK_BGEMM(x32_packw_x8s4__wasmsimd_u4)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

static void x32_packw_x2__scalar_float_u4(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_u4,
    /*nr=*/2, /*kr=*/1, /*sr=*/1);
}
static void x32_packw_x2__scalar_int_u4(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int_u4,
    /*nr=*/2, /*kr=*/1, /*sr=*/1);
}
static void x32_packw_x4__scalar_float_u4(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_u4,
    /*nr=*/4, /*kr=*/1, /*sr=*/1);
}
static void x32_packw_x4__scalar_int_u4(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int_u4,
    /*nr=*/4, /*kr=*/1, /*sr=*/1);
}
static void x32_packw_x8__scalar_float_u4(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float_u4,
    /*nr=*/8, /*kr=*/1, /*sr=*/1);
}
static void x32_packw_x8__scalar_int_u4(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int_u4,
    /*nr=*/8, /*kr=*/1, /*sr=*/1);
}
static void x32_packw_x16__scalar_float_u4(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x16__scalar_float_u4,
    /*nr=*/16, /*kr=*/1, /*sr=*/1);
}
static void x32_packw_x16__scalar_int_u4(benchmark::State& state, const char* net) {
  x32_packw(state,
    xnn_x32_packw_gemm_goi_ukernel_x16__scalar_int_u4,
    /*nr=*/16, /*kr=*/1, /*sr=*/1);
}

BENCHMARK_BGEMM(x32_packw_x2__scalar_float_u4)
BENCHMARK_BGEMM(x32_packw_x2__scalar_int_u4)
BENCHMARK_BGEMM(x32_packw_x4__scalar_float_u4)
BENCHMARK_BGEMM(x32_packw_x4__scalar_int_u4)
BENCHMARK_BGEMM(x32_packw_x8__scalar_float_u4)
BENCHMARK_BGEMM(x32_packw_x8__scalar_int_u4)
BENCHMARK_BGEMM(x32_packw_x16__scalar_float_u4)
BENCHMARK_BGEMM(x32_packw_x16__scalar_int_u4)

void x32_packw__reference(
  size_t batch,
  size_t dim_n,
  size_t dim_k,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint32_t* weights,
  const uint32_t* bias,
  const void* scale,
  uint32_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  xnn_pack_f32_gemm_goi_w(batch, dim_n, dim_k, nr, kr, sr,
     reinterpret_cast<const float*>(weights),
     reinterpret_cast<const float*>(bias),
     scale,
     reinterpret_cast<float*>(packed_weights),
     extra_bytes, params);
}

static void x32_packw_x2c4__reference(benchmark::State& state, const char* net) {
  x32_packw(state,
    x32_packw__reference,
    /*nr=*/2, /*kr=*/4, /*sr=*/1);
}
static void x32_packw_x8__reference(benchmark::State& state, const char* net) {
  x32_packw(state,
    x32_packw__reference,
    /*nr=*/8, /*kr=*/1, /*sr=*/1);
}
static void x32_packw_x8s4__reference(benchmark::State& state, const char* net) {
  x32_packw(state,
    x32_packw__reference,
    /*nr=*/8, /*kr=*/1, /*sr=*/4);
}
static void x32_packw_x16__reference(benchmark::State& state, const char* net) {
  x32_packw(state,
    x32_packw__reference,
    /*nr=*/16, /*kr=*/1, /*sr=*/1);
}
static void x32_packw_x16s4__reference(benchmark::State& state, const char* net) {
  x32_packw(state,
    x32_packw__reference,
     /*nr=*/16, /*kr=*/1, /*sr=*/4);
}

BENCHMARK_BGEMM(x32_packw_x2c4__reference)
BENCHMARK_BGEMM(x32_packw_x8__reference)
BENCHMARK_BGEMM(x32_packw_x8s4__reference)
BENCHMARK_BGEMM(x32_packw_x16__reference)
BENCHMARK_BGEMM(x32_packw_x16s4__reference)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
