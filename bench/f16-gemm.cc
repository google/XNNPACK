// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "gemm.h"
#include "utils.h"
#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pack.h"
#include "xnnpack/buffer.h"
#include <benchmark/benchmark.h>

static void f16_gemm(benchmark::State& state,
  xnn_f16_gemm_minmax_ukernel_fn gemm,
  xnn_init_f16_minmax_params_fn init_params,
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
  
  xnnpack::Buffer<xnn_float16> a(mc * kc + XNN_EXTRA_BYTES / sizeof(xnn_float16));
  std::generate(a.begin(), a.end(), f32rng);
  xnnpack::Buffer<xnn_float16> k(nc * kc);
  std::generate(k.begin(), k.end(), f32rng);
  xnnpack::Buffer<xnn_float16> b(nc);
  std::generate(b.begin(), b.end(), f32rng);

  const size_t w_elements = nc_stride * kc_stride + nc_stride;
  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(xnn_float16) * (w_elements + c_elements));

  xnnpack::Buffer<xnn_float16, XNN_ALLOCATION_ALIGNMENT> w(w_elements * num_buffers);
  xnn_pack_f16_gemm_goi_w(/*groups=*/1, nc, kc, nr, kr, sr,
                          reinterpret_cast<const uint16_t*>(k.data()), 
                          reinterpret_cast<const uint16_t*>(b.data()), 
                          /*scale=*/nullptr, 
                          reinterpret_cast<uint16_t*>(w.data()), 
                          /*extra_bytes=*/0, /*params=*/nullptr);
  xnnpack::Buffer<xnn_float16> c(c_elements * num_buffers);

  // Prepare minmax parameters.
  xnn_f16_minmax_params params;
  init_params(&params,
    -INFINITY, INFINITY);

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(xnn_float16));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      for (uint32_t n = 0; n < nc; n += nr) {
        const uint32_t nb = min(nc - n, nr);
        gemm(
          mb, nb, kc * sizeof(xnn_float16),
          a.data() + m * kc, kc * sizeof(xnn_float16),
          w.data() + (nc_stride * buffer_index + n) * (kc_stride + 1),
          c.data() + (mc * buffer_index + m) * nc + n, nc * sizeof(xnn_float16), nr * sizeof(xnn_float16),
          &params);
      }
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void f16_gemm_1x16__asm_aarch64_neonfp16arith_ld32(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld32,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_1x16__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_4x16__asm_aarch64_neonfp16arith_ld32(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld32,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_4x16__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_6x16__asm_aarch64_neonfp16arith_cortex_a55(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_6x16__asm_aarch64_neonfp16arith_cortex_a55r0(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_6x16__asm_aarch64_neonfp16arith_cortex_a75(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_6x16__asm_aarch64_neonfp16arith_ld32(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld32,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_6x16__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_1x8__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_1x8__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_4x8__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_4x8__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_6x8__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_6x8__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_8x8__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_8x8__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_1x16__asm_aarch64_neonfp16arith_ld32)
  BENCHMARK_GEMM(f16_gemm_1x16__asm_aarch64_neonfp16arith_ld64)
  BENCHMARK_GEMM(f16_gemm_4x16__asm_aarch64_neonfp16arith_ld32)
  BENCHMARK_GEMM(f16_gemm_4x16__asm_aarch64_neonfp16arith_ld64)
  BENCHMARK_GEMM(f16_gemm_6x16__asm_aarch64_neonfp16arith_cortex_a55)
  BENCHMARK_GEMM(f16_gemm_6x16__asm_aarch64_neonfp16arith_cortex_a55r0)
  BENCHMARK_GEMM(f16_gemm_6x16__asm_aarch64_neonfp16arith_cortex_a75)
  BENCHMARK_GEMM(f16_gemm_6x16__asm_aarch64_neonfp16arith_ld32)
  BENCHMARK_GEMM(f16_gemm_6x16__asm_aarch64_neonfp16arith_ld64)
  BENCHMARK_GEMM(f16_gemm_1x8__asm_aarch64_neonfp16arith_ld64)
  BENCHMARK_GEMM(f16_gemm_4x8__asm_aarch64_neonfp16arith_ld64)
  BENCHMARK_GEMM(f16_gemm_6x8__asm_aarch64_neonfp16arith_ld64)
  BENCHMARK_GEMM(f16_gemm_8x8__asm_aarch64_neonfp16arith_ld64)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void f16_gemm_1x8__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_4x8__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_6x8__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_8x8__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_1x16__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_1x16__neonfp16arith_ld64,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_4x16__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_4x16__neonfp16arith_ld64,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_6x16__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_6x16__neonfp16arith_ld64,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_8x16__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_8x16__neonfp16arith_ld64,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/8, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_GEMM(f16_gemm_1x8__neonfp16arith_ld64)
  BENCHMARK_GEMM(f16_gemm_4x8__neonfp16arith_ld64)
  BENCHMARK_GEMM(f16_gemm_6x8__neonfp16arith_ld64)
  BENCHMARK_GEMM(f16_gemm_8x8__neonfp16arith_ld64)
  BENCHMARK_GEMM(f16_gemm_1x16__neonfp16arith_ld64)
  BENCHMARK_GEMM(f16_gemm_4x16__neonfp16arith_ld64)
  BENCHMARK_GEMM(f16_gemm_6x16__neonfp16arith_ld64)
  BENCHMARK_GEMM(f16_gemm_8x16__neonfp16arith_ld64)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f16_gemm_1x8__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_1x8__avx2_broadcast,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }
  static void f16_gemm_4x8__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_4x8__avx2_broadcast,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }
  static void f16_gemm_5x8__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_5x8__avx2_broadcast,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }
  static void f16_gemm_6x8__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_6x8__avx2_broadcast,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }
  static void f16_gemm_7x8__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_7x8__avx2_broadcast,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/7, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }
  static void f16_gemm_1x16__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_1x16__avx2_broadcast,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }
  static void f16_gemm_3x16__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_3x16__avx2_broadcast,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }
  static void f16_gemm_4x16__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_4x16__avx2_broadcast,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }
  static void f16_gemm_5x16__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_5x16__avx2_broadcast,
      xnn_init_f16_minmax_scalar_params,
      /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_GEMM(f16_gemm_1x8__avx2_broadcast)
  BENCHMARK_GEMM(f16_gemm_4x8__avx2_broadcast)
  BENCHMARK_GEMM(f16_gemm_5x8__avx2_broadcast)
  BENCHMARK_GEMM(f16_gemm_6x8__avx2_broadcast)
  BENCHMARK_GEMM(f16_gemm_7x8__avx2_broadcast)
  BENCHMARK_GEMM(f16_gemm_1x16__avx2_broadcast)
  BENCHMARK_GEMM(f16_gemm_3x16__avx2_broadcast)
  BENCHMARK_GEMM(f16_gemm_4x16__avx2_broadcast)
  BENCHMARK_GEMM(f16_gemm_5x16__avx2_broadcast)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
