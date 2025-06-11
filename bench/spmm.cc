// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "bench/spmm.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "bench/utils.h"
#include "include/xnnpack.h"
#include "src/xnnpack/aligned-allocator.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/hardware-config.h"  // IWYU pragma: keep
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"  // IWYU pragma: keep
#include "src/xnnpack/spmm.h"  // IWYU pragma: keep
#include <benchmark/benchmark.h>

static void f16_spmm(benchmark::State& state, uint64_t arch_flags,
                     xnn_f16_spmm_minmax_ukernel_fn spmm, uint32_t mr,
                     uint32_t nr, float sparsity,
                     xnn_init_f16_minmax_params_fn init_params) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_real_distribution<float> f32dist;
  std::uniform_real_distribution<float> pdist;

  xnnpack::Buffer<xnn_float16, XNN_ALLOCATION_ALIGNMENT> input(kc * mc);
  // Think of b as (n/nr + n % nr) x k, expansion happens later.
  const size_t ncols = nc / nr + nc % nr;
  xnnpack::Buffer<xnn_float16> b(ncols * kc);
  xnnpack::Buffer<xnn_float16> bias(nc);
  // Number of non-zero weights per N (output channel).
  xnnpack::Buffer<uint32_t> nmap(nc);
  // Mapping from index of non-zero weight to increment of K (input channel)
  // following this index. Micro-kernel can access one element beyond w and dmap
  // for software pipelining.
  xnnpack::Buffer<int32_t> dmap(nc * kc + 1);
  xnnpack::Buffer<xnn_float16> w(nc * kc + nc + 1);
  xnnpack::Buffer<xnn_float16> output(nc * mc);

  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  std::fill(nmap.begin(), nmap.end(), 0);
  std::fill(dmap.begin(), dmap.end(), 0);
  std::fill(w.begin(), w.end(), 0.0f);

  for (xnn_float16& b_value : b) {
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
    for (size_t i = 0; i < nr; ++i) w[wcnt++] = bias[nr * nn + i];
    for (size_t kk = 0; kk < kc; kk++) {
      if (!xnn_float16_is_zero(b[nn * kc + kk])) {
        // Every non-zero actually corresponds to nr adjacent non-zeros.
        for (size_t i = 0; i < nr; ++i)
          w[wcnt++] = b[nn * kc + kk] + static_cast<xnn_float16>(i);
        // Skip the very first non-zero weight as we record only the difference.
        if (first_nzz) {
          first_kk = kk;
        } else {
          const int32_t increment =
              int32_t(kk - last_kk) * int32_t(mc * sizeof(xnn_float16));
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
      if (!xnn_float16_is_zero(b[nn * kc + kk])) {
        // Every non-zero actually corresponds to nr adjacent non-zeros.
        w[wcnt++] = b[nn * kc + kk];
        // Skip the very first non-zero weight as we record only the difference.
        if (first_nzz) {
          first_kk = kk;
        } else {
          const int32_t increment =
              int32_t(kk - last_kk) * int32_t(mc * sizeof(xnn_float16));
          dmap[nnz++] = increment;
        }
        last_kk = kk;
        first_nzz = false;
        nmap[nn] += 1;
      }
    }
  }
  // In the end, we must return input pointer to the initial value.
  const int64_t increment =
      int32_t(first_kk - last_kk) * int32_t(mc * sizeof(xnn_float16));
  dmap[nnz++] = increment;

  // Prepare parameters.
  xnn_f16_minmax_params params;
  init_params(&params, 0xFC00 /* -inf */, 0x7C00 /* inf */);

  for (auto _ : state) {
    spmm(mc * sizeof(xnn_float16), nc, input.data() + first_kk * mc, w.data(),
         dmap.data(), nmap.data(), output.data(), mc * sizeof(xnn_float16),
         &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
      uint64_t(state.iterations()) * 2 * mc * nnz, benchmark::Counter::kIsRate);

  state.counters["EffFLOPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * mc * nc * kc,
                         benchmark::Counter::kIsRate);
}

static void f32_spmm(benchmark::State& state, uint64_t arch_flags,
                     xnn_f32_spmm_minmax_ukernel_fn spmm, uint32_t mr,
                     uint32_t nr, float sparsity,
                     xnn_init_f32_minmax_params_fn init_params) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng =
      std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  // if using blocks, generate the reduced matrix first and then extrude along
  // the block dimension (n), to get the full matrix
  size_t ncols = nc / nr + nc % nr;
  std::vector<float> b(ncols * kc);
  std::vector<float> bias(nc);
  std::vector<float> w;
  std::vector<uint32_t> nmap;
  std::vector<int32_t> dmap;
  const size_t sparse_end =
      std::min(size_t(float(b.size()) * sparsity), b.size());
  const size_t num_nonzeroes = nr * (b.size() - sparse_end);

  const size_t w_elements = num_nonzeroes + nc;
  const size_t c_elements = mc * nc;
  const size_t dmap_elements = num_nonzeroes / nr;
  const size_t nmap_elements = nc;
  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(),
              sizeof(float) * (w_elements + c_elements) +
                  sizeof(uint32_t) * (dmap_elements + nmap_elements));

  // Micro-kernel can access one element beyond w and dmap for software
  // pipelining.
  w.reserve(num_buffers * w_elements + 1);
  dmap.reserve(num_buffers * dmap_elements + 1);
  nmap.resize(num_buffers * nmap_elements);

  std::vector<size_t> a_offsets(num_buffers);

  for (size_t buffer_index = 0; buffer_index < num_buffers; buffer_index++) {
    // Re-generate weights. Note: each re-generation produces the number of
    // non-zeroes.
    std::fill(b.begin(), b.begin() + sparse_end, 0.0f);
    std::generate(b.begin() + sparse_end, b.end(), std::ref(f32rng));
    std::shuffle(b.begin(), b.end(), rng);
    std::generate(bias.begin(), bias.end(), std::ref(f32rng));

    uint32_t first_j = 0, last_j = 0;
    bool is_first_nonzero = true;
    for (uint32_t i = 0; i < nc / nr; i++) {
      for (uint32_t n = 0; n < nr; n++) w.push_back(bias[nr * i + n]);
      for (uint32_t j = 0; j < kc; j++) {
        if (b[i * kc + j] != 0.0f) {
          for (size_t l = 0; l < nr; l++)
            w.push_back(b[i * kc + j] + static_cast<float>(i));
          if (is_first_nonzero) {
            first_j = j;
          } else {
            const ptrdiff_t increment =
                int32_t(j - last_j) * int32_t(mc) * int32_t(sizeof(float));
            dmap.push_back(increment);
          }
          last_j = j;
          is_first_nonzero = false;
          nmap[buffer_index * nmap_elements + i] += 1;
        }
      }
    }
    for (uint32_t i = nc / nr; i < ncols; i++) {
      w.push_back(bias[i]);
      for (uint32_t j = 0; j < kc; j++) {
        if (b[i * kc + j] != 0.0f) {
          w.push_back(b[i * kc + j]);
          if (is_first_nonzero) {
            first_j = j;
          } else {
            const ptrdiff_t increment =
                int32_t(j - last_j) * int32_t(mc) * int32_t(sizeof(float));
            dmap.push_back(increment);
          }
          last_j = j;
          is_first_nonzero = false;
          nmap[buffer_index * nmap_elements + i] += 1;
        }
      }
    }
    {
      const ptrdiff_t increment =
          int32_t(first_j - last_j) * int32_t(mc) * int32_t(sizeof(float));
      dmap.push_back(increment);
    }

    a_offsets[buffer_index] = first_j * mc;
  }

  // Micro-kernel can access one element beyond w and dmap for software
  // pipelining.
  w.resize(w.size() + 1);
  dmap.resize(dmap.size() + 1);

  std::vector<float, AlignedAllocator<float, 64>> a(kc * mc);
  std::vector<float, AlignedAllocator<float, 64>> c(num_buffers * c_elements);

  std::generate(a.begin(), a.end(), std::ref(f32rng));

  xnn_f32_minmax_params params;
  init_params(&params, -std::numeric_limits<float>::infinity(),
              +std::numeric_limits<float>::infinity());

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache
    // state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W, Kmap, and Nmap is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(float));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    spmm(mc * sizeof(float), nc, a.data() + a_offsets[buffer_index],
         w.data() + buffer_index * w_elements,
         dmap.data() + buffer_index * dmap_elements,
         nmap.data() + buffer_index * nmap_elements,
         c.data() + buffer_index * c_elements, mc * sizeof(float), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * mc * num_nonzeroes,
                         benchmark::Counter::kIsRate);

  state.counters["EffFLOPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * mc * nc * kc,
                         benchmark::Counter::kIsRate);
}

#define XNN_UKERNEL(arch_flags, ukernel, mr, nr, k_block, vector_tile,    \
                    pipelined, datatype, params_type, init_params)        \
  static void bench_##ukernel(benchmark::State& state, const char* net) { \
    f16_spmm(state, arch_flags, ukernel, mr, nr, /*sparsity=*/0.8f,       \
             init_params);                                                \
  }                                                                       \
  BENCHMARK_SPMM(bench_##ukernel)
#include "src/f16-spmm/f16-spmm-minmax.inc"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, ukernel, mr, nr, k_block, vector_tile,    \
                    pipelined, datatype, params_type, init_params)        \
  static void bench_##ukernel(benchmark::State& state, const char* net) { \
    f32_spmm(state, arch_flags, ukernel, mr, nr, /*sparsity=*/0.8f,       \
             init_params);                                                \
  }                                                                       \
  BENCHMARK_SPMM(bench_##ukernel)
#include "src/f32-spmm/f32-spmm-minmax.inc"
#undef XNN_UKERNEL

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
