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

#include <benchmark/benchmark.h>
#include <fp16/fp16.h>
#include "bench/gemm.h"
#include "bench/utils.h"

#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/math.h"
#include "xnnpack/pack.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"


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
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

  std::vector<uint16_t> a(mc * kc + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::generate(a.begin(), a.end(), std::ref(f16rng));
  std::vector<uint16_t> k(nc * kc);
  std::generate(k.begin(), k.end(), std::ref(f16rng));
  std::vector<uint16_t> b(nc);
  std::generate(b.begin(), b.end(), std::ref(f16rng));

  const size_t w_elements = nc_stride * kc_stride + nc_stride;
  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(uint16_t) * (w_elements + c_elements));

  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0);
  xnn_pack_f16_gemm_goi_w(/*groups=*/1, nc, kc, nr, kr, sr,
    k.data(), b.data(), /*scale=*/nullptr, w.data(), /*extra_bytes=*/0, /*params=*/nullptr);
  std::vector<uint16_t> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), UINT16_C(0x7E00) /* NaN */);

  // Prepare minmax parameters.
  xnn_f16_minmax_params params;
  init_params(&params,
    UINT16_C(0xFC00)  /* -inf */, UINT16_C(0x7C00)  /* inf */);

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(uint16_t));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      for (uint32_t n = 0; n < nc; n += nr) {
        const uint32_t nb = min(nc - n, nr);
        gemm(
          mb, nb, kc * sizeof(uint16_t),
          a.data() + m * kc, kc * sizeof(uint16_t),
          w.data() + (nc_stride * buffer_index + n) * (kc_stride + 1),
          c.data() + (mc * buffer_index + m) * nc + n, nc * sizeof(uint16_t), nr * sizeof(uint16_t),
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
      xnn_init_f16_minmax_fp16arith_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_1x16__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_4x16__asm_aarch64_neonfp16arith_ld32(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld32,
      xnn_init_f16_minmax_fp16arith_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_4x16__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_6x16__asm_aarch64_neonfp16arith_cortex_a55(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55,
      xnn_init_f16_minmax_fp16arith_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_6x16__asm_aarch64_neonfp16arith_cortex_a55r0(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0,
      xnn_init_f16_minmax_fp16arith_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_6x16__asm_aarch64_neonfp16arith_cortex_a75(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75,
      xnn_init_f16_minmax_fp16arith_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_6x16__asm_aarch64_neonfp16arith_ld32(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld32,
      xnn_init_f16_minmax_fp16arith_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_6x16__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_1x8__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_1x8__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_4x8__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_4x8__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_6x8__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_6x8__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_8x8__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_8x8__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
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
      xnn_init_f16_minmax_fp16arith_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_4x8__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_6x8__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_8x8__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_1x16__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_1x16__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_4x16__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_4x16__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_6x16__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_6x16__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_gemm_8x16__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_8x16__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
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
      xnn_init_f16_minmax_avx_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }
  static void f16_gemm_4x8__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_4x8__avx2_broadcast,
      xnn_init_f16_minmax_avx_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }
  static void f16_gemm_5x8__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_5x8__avx2_broadcast,
      xnn_init_f16_minmax_avx_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }
  static void f16_gemm_6x8__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_6x8__avx2_broadcast,
      xnn_init_f16_minmax_avx_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }
  static void f16_gemm_7x8__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_7x8__avx2_broadcast,
      xnn_init_f16_minmax_avx_params,
      /*mr=*/7, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }
  static void f16_gemm_1x16__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_1x16__avx2_broadcast,
      xnn_init_f16_minmax_avx_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }
  static void f16_gemm_3x16__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_3x16__avx2_broadcast,
      xnn_init_f16_minmax_avx_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }
  static void f16_gemm_4x16__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_4x16__avx2_broadcast,
      xnn_init_f16_minmax_avx_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }
  static void f16_gemm_5x16__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_gemm(state,
      xnn_f16_gemm_minmax_ukernel_5x16__avx2_broadcast,
      xnn_init_f16_minmax_avx_params,
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
