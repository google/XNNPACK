// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include <fp16/fp16.h>
#include "bench/spmm.h"
#include "bench/utils.h"

#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/spmm.h"

static inline bool is_fp16_zero(uint16_t x) {
  const uint16_t two_x = x + x;
  return two_x == 0;
}

static void f16_spmm(benchmark::State& state,
  xnn_f16_spmm_minmax_ukernel_fn spmm, uint32_t mr, uint32_t nr, float sparsity,
  xnn_init_f16_minmax_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }
  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_real_distribution<float> f32dist;
  std::uniform_real_distribution<float> pdist;

  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> input(kc * mc);
  // Think of b as (n/nr + n % nr) x k, expansion happens later.
  const size_t ncols = nc / nr + nc % nr;
  std::vector<uint16_t> b(ncols * kc);
  std::vector<uint16_t> bias(nc);
  // Number of non-zero weights per N (output channel).
  std::vector<uint32_t> nmap(nc);
  // Mapping from index of non-zero weight to increment of K (input channel) following this index.
  std::vector<int32_t> dmap(nc * kc);
  std::vector<uint16_t> w(nc * kc + nc);
  std::vector<uint16_t> output(nc * mc);

  std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::generate(b.begin(), b.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::generate(bias.begin(), bias.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);
  std::fill(nmap.begin(), nmap.end(), 0);
  std::fill(dmap.begin(), dmap.end(), 0);
  std::fill(w.begin(), w.end(), 0);

  for (uint16_t& b_value : b) {
    if (pdist(rng) <= sparsity) {
      b_value = 0;
    }
  }

  uint32_t nnz = 0;
  uint32_t wcnt = 0;
  size_t last_kk = 0;
  bool first_nzz = true;
  size_t first_kk = 0;
  for (size_t nn = 0; nn < nc / nr; nn++) {
    for (size_t i = 0; i < nr; ++i)
      w[wcnt++] = bias[nr * nn + i];
    for (size_t kk = 0; kk < kc; kk++) {
      if (!is_fp16_zero(b[nn * kc + kk])) {
        // Every non-zero actually corresponds to nr adjacent non-zeros.
        for (size_t i = 0; i < nr; ++i)
          w[wcnt++] = fp16_ieee_from_fp32_value(fp16_ieee_to_fp32_value(b[nn * kc + kk]) + static_cast<float>(i));
        // Skip the very first non-zero weight as we record only the difference.
        if (first_nzz) {
          first_kk = kk;
        } else {
          const int32_t increment = int32_t(kk - last_kk) * int32_t(mc * sizeof(uint16_t));
          dmap[nnz++] = increment;
        }
        last_kk = kk;
        first_nzz = false;
        nmap[nn] += 1;
      }
    }
  }

  // now we've constructed the matrix for the blocked part and switch to the
  // leftovers, which we do as nr=1 always.
  for (size_t nn = nc / nr; nn < ncols; nn++) {
    w[wcnt++] = bias[(nc / nr) * nr + (nn - nc / nr)];
    for (size_t kk = 0; kk < kc; kk++) {
      if (!is_fp16_zero(b[nn * kc + kk])) {
        // Every non-zero actually corresponds to nr adjacent non-zeros.
        w[wcnt++] = b[nn * kc + kk];
        // Skip the very first non-zero weight as we record only the difference.
        if (first_nzz) {
          first_kk = kk;
        } else {
          const int32_t increment = int32_t(kk - last_kk) * int32_t(mc * sizeof(uint16_t));
          dmap[nnz++] = increment;
        }
        last_kk = kk;
        first_nzz = false;
        nmap[nn] += 1;
      }
    }
  }
  // In the end, we must return input pointer to the initial value.
  const int64_t increment = int32_t(first_kk - last_kk) * int32_t(mc * sizeof(uint16_t));
  dmap[nnz++] = increment;

  // Generate expanded b which will be used in reference calculation.
  // Everywhere there is input non-zero in the original we copy it and add an
  // adjacent non-zero with incremented weight value.
  std::vector<uint16_t> b_full(nc * kc);
  if (nr == 1) {
     b_full = b;
  }
  else {
    for (size_t nn = 0; nn < nc / nr; nn++) {
      for (size_t kk = 0; kk < kc; kk++) {
        if (b[nn * kc + kk] != 0.0f) {
          for (size_t i = 0; i < nr; ++i)
            b_full[nr * nn * kc + i * kc + kk] = fp16_ieee_from_fp32_value(
              fp16_ieee_to_fp32_value(b[nn * kc + kk]) + static_cast<float>(i));
        }
      }
    }
    for (size_t nn = nc / nr; nn < ncols; nn++) {
      for (size_t kk = 0; kk < kc; kk++) {
        if (b[nn * kc + kk] != 0.0f) {
          b_full[nr * (nc / nr) * kc + (nn - nc / nr) * kc + kk] = b[nn * kc + kk];
        }
      }
    }
  }

  // Micro-kernel can access one element beyond w and dmap for software pipelining.
  w.resize(wcnt + 1);
  dmap.resize(nnz + 1);

  // Prepare parameters.
  xnn_f16_minmax_params params;
  init_params(&params, 0xFC00 /* -inf */, 0x7C00 /* inf */);

  for (auto _ : state) {

    spmm(mc * sizeof(uint16_t), nc,
      input.data() + first_kk * mc,
      w.data(), dmap.data(), nmap.data(),
      output.data(), mc * sizeof(uint16_t),
      &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nnz, benchmark::Counter::kIsRate);

  state.counters["EffFLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void spmm80_8x1__neonfp16arith(benchmark::State& state, const char* net) {
    f16_spmm(state, xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith, 8, 1, 0.8f,
      xnn_init_f16_minmax_fp16arith_params, benchmark::utils::CheckNEONFP16ARITH);
  }
  static void spmm80_8x1__neonfp16arith_pipelined(benchmark::State& state, const char* net) {
    f16_spmm(state, xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_pipelined, 8, 1, 0.8f,
      xnn_init_f16_minmax_fp16arith_params, benchmark::utils::CheckNEONFP16ARITH);
  }
  static void spmm80_8x1__neonfp16arith_x2(benchmark::State& state, const char* net) {
    f16_spmm(state, xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_x2, 8, 1, 0.8f,
      xnn_init_f16_minmax_fp16arith_params, benchmark::utils::CheckNEONFP16ARITH);
  }
  static void spmm80_16x1__neonfp16arith(benchmark::State& state, const char* net) {
    f16_spmm(state, xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith, 16, 1, 0.8f,
      xnn_init_f16_minmax_fp16arith_params, benchmark::utils::CheckNEONFP16ARITH);
  }
  static void spmm80_16x1__neonfp16arith_pipelined(benchmark::State& state, const char* net) {
    f16_spmm(state, xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_pipelined, 16, 1, 0.8f,
      xnn_init_f16_minmax_fp16arith_params, benchmark::utils::CheckNEONFP16ARITH);
  }
  static void spmm80_16x1__neonfp16arith_x2(benchmark::State& state, const char* net) {
    f16_spmm(state, xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_x2, 16, 1, 0.8f,
      xnn_init_f16_minmax_fp16arith_params, benchmark::utils::CheckNEONFP16ARITH);
  }
  static void spmm80_24x1__neonfp16arith(benchmark::State& state, const char* net) {
    f16_spmm(state, xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith, 24, 1, 0.8f,
      xnn_init_f16_minmax_fp16arith_params, benchmark::utils::CheckNEONFP16ARITH);
  }
  static void spmm80_24x1__neonfp16arith_pipelined(benchmark::State& state, const char* net) {
    f16_spmm(state, xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_pipelined, 24, 1, 0.8f,
      xnn_init_f16_minmax_fp16arith_params, benchmark::utils::CheckNEONFP16ARITH);
  }
  static void spmm80_24x1__neonfp16arith_x2(benchmark::State& state, const char* net) {
    f16_spmm(state, xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_x2, 24, 1, 0.8f,
      xnn_init_f16_minmax_fp16arith_params, benchmark::utils::CheckNEONFP16ARITH);
  }
  static void spmm80_32x1__neonfp16arith(benchmark::State& state, const char* net) {
    f16_spmm(state, xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith, 32, 1, 0.8f,
      xnn_init_f16_minmax_fp16arith_params, benchmark::utils::CheckNEONFP16ARITH);
  }
  static void spmm80_32x1__neonfp16arith_pipelined(benchmark::State& state, const char* net) {
    f16_spmm(state, xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_pipelined, 32, 1, 0.8f,
      xnn_init_f16_minmax_fp16arith_params, benchmark::utils::CheckNEONFP16ARITH);
  }
  static void spmm80_32x1__neonfp16arith_x2(benchmark::State& state, const char* net) {
    f16_spmm(state, xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_x2, 32, 1, 0.8f,
      xnn_init_f16_minmax_fp16arith_params, benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_SPMM(spmm80_8x1__neonfp16arith_pipelined)
  BENCHMARK_SPMM(spmm80_16x1__neonfp16arith_pipelined)
  BENCHMARK_SPMM(spmm80_24x1__neonfp16arith_pipelined)
  BENCHMARK_SPMM(spmm80_32x1__neonfp16arith_pipelined)
  BENCHMARK_SPMM(spmm80_8x1__neonfp16arith)
  BENCHMARK_SPMM(spmm80_16x1__neonfp16arith)
  BENCHMARK_SPMM(spmm80_24x1__neonfp16arith)
  BENCHMARK_SPMM(spmm80_32x1__neonfp16arith)
  BENCHMARK_SPMM(spmm80_8x1__neonfp16arith_x2)
  BENCHMARK_SPMM(spmm80_16x1__neonfp16arith_x2)
  BENCHMARK_SPMM(spmm80_24x1__neonfp16arith_x2)
  BENCHMARK_SPMM(spmm80_32x1__neonfp16arith_x2)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
