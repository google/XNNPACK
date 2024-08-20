// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/config-types.h"
#include "xnnpack/gemm.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/pack.h"
#include "xnnpack/packq.h"
#include "xnnpack/packw.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <fp16/fp16.h>
#include "bench/utils.h"
#include <benchmark/benchmark.h>

void GEMMBenchmark(benchmark::State& state, xnn_qs8_gemm_minmax_ukernel_fn gemm,
                   xnn_init_qs8_conv_minmax_params_fn init_params,
                   xnn_pack_qs8_gemm_fn pack, size_t mr, size_t nr, size_t kr,
                   size_t sr, benchmark::utils::IsaCheckFunction isa_check) {
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
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000),
                          std::ref(rng));
  auto i8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             -std::numeric_limits<int8_t>::max(),
                             std::numeric_limits<int8_t>::max()),
                         std::ref(rng));

  std::vector<int8_t> a(mc * kc + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::generate(a.begin(), a.end(), std::ref(i8rng));
  std::vector<int8_t> k(nc * kc);
  std::generate(k.begin(), k.end(), std::ref(i8rng));
  std::vector<int32_t> b(nc);
  std::generate(b.begin(), b.end(), std::ref(i32rng));

  const size_t w_element_size = sizeof(int8_t);
  const size_t w_size = nc_stride * sizeof(int32_t) + kc_stride * nc_stride * w_element_size;
  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 + benchmark::utils::DivideRoundUp<size_t>(
                                     benchmark::utils::GetMaxCacheSize(),
                                     w_size + c_elements * sizeof(int8_t));

  std::vector<char, AlignedAllocator<char, 64>> w(w_size * num_buffers);
  std::fill(w.begin(), w.end(), 0);

  const xnn_qs8_packing_params packing_params = {127};
  pack(/*g=*/1, nc, kc, nr, kr, sr, k.data(), b.data(), /*scale=*/nullptr,
       w.data(),
       /*extra_bytes=*/0, &packing_params);
  std::vector<int8_t> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), INT8_C(0xA5));

  union xnn_qs8_conv_minmax_params quantization_params;
  init_params(&quantization_params,
              /*scale=*/0.75f,
              /*output_zero_point=*/127,
              /*output_min=*/-127,
              /*output_max=*/126);

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache
    // state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(int8_t));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      for (uint32_t n = 0; n < nc; n += nr) {
        const uint32_t nb = min(nc - n, nr);
        gemm(mb, nb, kc * sizeof(int8_t), a.data() + m * kc,
             kc * sizeof(int8_t),
             w.data() + w_size * buffer_index +
                 n * (kc_stride * w_element_size + sizeof(int32_t)),
             c.data() + (mc * buffer_index + m) * nc + n, nc * sizeof(int8_t),
             nr * sizeof(int8_t), &quantization_params);
      }
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["OPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * mc * nc * kc,
                         benchmark::Counter::kIsRate);
}

void GEMMBenchmark(benchmark::State& state,
                   xnn_qs8_qc8w_gemm_minmax_ukernel_fn gemm,
                   xnn_init_qs8_qc8w_conv_minmax_params_fn init_params,
                   xnn_pack_qs8_gemm_fn pack, size_t mr, size_t nr, size_t kr,
                   size_t sr, benchmark::utils::IsaCheckFunction isa_check) {
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
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000),
                          std::ref(rng));
  auto i8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             -std::numeric_limits<int8_t>::max(),
                             std::numeric_limits<int8_t>::max()),
                         std::ref(rng));

  std::vector<int8_t> a(mc * kc + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::generate(a.begin(), a.end(), std::ref(i8rng));
  std::vector<int8_t> k(nc * kc);
  std::generate(k.begin(), k.end(), std::ref(i8rng));
  std::vector<int32_t> b(nc);
  std::generate(b.begin(), b.end(), std::ref(i32rng));

  const size_t w_element_size = sizeof(int8_t);
  const size_t w_size = nc_stride * sizeof(int32_t) + kc_stride * nc_stride * w_element_size;
  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 + benchmark::utils::DivideRoundUp<size_t>(
                                     benchmark::utils::GetMaxCacheSize(),
                                     w_size + c_elements * sizeof(int8_t));

  std::vector<char, AlignedAllocator<char, 64>> w(w_size * num_buffers);
  std::fill(w.begin(), w.end(), 0);

  const xnn_qs8_packing_params packing_params = {int8_t(127 - 0x80)};
  pack(/*g=*/1, nc, kc, nr, kr, sr, k.data(), b.data(), /*scale=*/nullptr,
       w.data(), nr * sizeof(float), &packing_params);
  std::vector<int8_t> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), INT8_C(0xA5));

  union xnn_qs8_qc8w_conv_minmax_params quantization_params;
  init_params(&quantization_params,
              /*output_zero_point=*/127,
              /*output_min=*/-127,
              /*output_max=*/126);

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache
    // state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(int8_t));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      for (uint32_t n = 0; n < nc; n += nr) {
        const uint32_t nb = min(nc - n, nr);
        gemm(mb, nb, kc * sizeof(int8_t), a.data() + m * kc,
             kc * sizeof(int8_t),
             w.data() + w_size * buffer_index +
                 n * (kc_stride * w_element_size + sizeof(int32_t)),
             c.data() + (mc * buffer_index + m) * nc + n, nc * sizeof(int8_t),
             nr * sizeof(int8_t), &quantization_params);
      }
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["OPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * mc * nc * kc,
                         benchmark::Counter::kIsRate);
}

void GEMMBenchmark(benchmark::State& state,
                   xnn_qd8_f16_qc8w_gemm_ukernel_fn gemm,
                   xnn_init_f16_minmax_params_fn init_params,
                   xnn_pack_qs8_gemm_fn pack, size_t mr, size_t nr, size_t kr,
                   size_t sr, benchmark::utils::IsaCheckFunction isa_check) {
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
  auto i8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             -std::numeric_limits<int8_t>::max(),
                             std::numeric_limits<int8_t>::max()),
                         std::ref(rng));

  std::vector<int8_t> a(mc * kc + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::generate(a.begin(), a.end(), std::ref(i8rng));
  std::vector<int8_t> k(nc * kc);
  std::generate(k.begin(), k.end(), std::ref(i8rng));

  std::vector<xnn_qd8_quantization_params> quantization_params(
      mc + XNN_EXTRA_QUANTIZATION_PARAMS);
  const size_t w_elements =
      nc_stride * (sizeof(float) * 2 + sizeof(int32_t)) + kc_stride * nc_stride;

  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 + benchmark::utils::DivideRoundUp<size_t>(
                                     benchmark::utils::GetMaxCacheSize(),
                                     sizeof(float) * (w_elements + c_elements));

  std::vector<char, AlignedAllocator<char, 64>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0);

  const xnn_qs8_packing_params packing_params = {/*input_zero_point=*/1};
  pack(1, nc, kc, nr, kr, sr, k.data(), /*bias=*/nullptr, /*scale=*/nullptr,
       w.data(), sizeof(float) * 2 * nr, &packing_params);
  std::vector<uint16_t> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), UINT16_C(0x7E00) /* NaN */);

  // Prepare parameters.
  xnn_f16_minmax_params params;
  init_params(&params,
              fp16_ieee_from_fp32_value(std::numeric_limits<int8_t>::min()),
              fp16_ieee_from_fp32_value(std::numeric_limits<int8_t>::max()));

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache
    // state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size());
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      gemm(mb, nc, kc, a.data() + m * kc, kc * sizeof(int8_t),
           w.data() + w_elements * buffer_index,
           c.data() + (buffer_index * mc + m) * nc, nc * sizeof(uint16_t),
           nr * sizeof(uint16_t), &params, quantization_params.data() + m);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["OPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * mc * nc * kc,
                         benchmark::Counter::kIsRate);
}

void GEMMBenchmark(benchmark::State& state,
                   xnn_qd8_f32_qc8w_gemm_ukernel_fn gemm,
                   xnn_init_f32_minmax_params_fn init_params,
                   xnn_pack_qs8_gemm_fn pack, size_t mr, size_t nr, size_t kr,
                   size_t sr, benchmark::utils::IsaCheckFunction isa_check) {
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
  auto i8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             -std::numeric_limits<int8_t>::max(),
                             std::numeric_limits<int8_t>::max()),
                         std::ref(rng));

  std::vector<int8_t> a(mc * kc + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::generate(a.begin(), a.end(), std::ref(i8rng));
  std::vector<int8_t> k(nc * kc);
  std::generate(k.begin(), k.end(), std::ref(i8rng));

  std::vector<xnn_qd8_quantization_params> quantization_params(
      mc + XNN_EXTRA_QUANTIZATION_PARAMS);
  const size_t w_elements =
      nc_stride * (sizeof(float) * 2 + sizeof(int32_t)) + kc_stride * nc_stride;

  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 + benchmark::utils::DivideRoundUp<size_t>(
                                     benchmark::utils::GetMaxCacheSize(),
                                     sizeof(float) * (w_elements + c_elements));

  std::vector<char, AlignedAllocator<char, 64>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0);

  const xnn_qs8_packing_params packing_params = {/*input_zero_point=*/1};
  pack(1, nc, kc, nr, kr, sr, k.data(), /*bias=*/nullptr, /*scale=*/nullptr,
       w.data(), sizeof(float) * 2 * nr, &packing_params);
  std::vector<float> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  // Prepare parameters.
  xnn_f32_minmax_params params;
  init_params(&params, std::numeric_limits<int8_t>::min(),
              std::numeric_limits<int8_t>::max());

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache
    // state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size());
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      gemm(mb, nc, kc, a.data() + m * kc, kc * sizeof(int8_t),
           w.data() + w_elements * buffer_index,
           c.data() + (buffer_index * mc + m) * nc, nc * sizeof(float),
           nr * sizeof(float), &params, quantization_params.data() + m);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["OPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * mc * nc * kc,
                         benchmark::Counter::kIsRate);
}

void GEMMBenchmark(benchmark::State& state,
                   xnn_qd8_f16_qb4w_gemm_ukernel_fn gemm,
                   xnn_init_f16_qb4w_minmax_params_fn init_params,
                   xnn_pack_qs8_qb4w_gemm_fn pack, size_t mr, size_t nr,
                   size_t kr, size_t sr,
                   benchmark::utils::IsaCheckFunction isa_check) {
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t bl = state.range(3);
  const size_t kc = round_up(state.range(2), bl);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             -std::numeric_limits<int8_t>::max(),
                             std::numeric_limits<int8_t>::max()),
                         std::ref(rng));
  auto u8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             0, std::numeric_limits<uint8_t>::max()),
                         std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.5f, 2.f),
                            std::ref(rng));

  const size_t planes = 2;  // 4 bit is 2 planes - low nibbles and high nibbles
  const size_t k2 = round_up_po2(kc, 2);  // tester assumes byte aligned rows
  const size_t packed_k2 =
      round_up_po2(kc, kr * sr * planes);  // 2 blocks for nibbles

  const size_t packed_k_bytes = (packed_k2 + 1) / 2;
  const size_t num_blocks = packed_k2 / bl;
  const size_t packed_n = round_up_po2(nc, nr);

  std::vector<int8_t> a(mc * kc + XNN_EXTRA_BYTES);
  std::generate(a.begin(), a.end(), std::ref(i8rng));
  std::vector<uint8_t> k(nc * kc / 2);
  std::generate(k.begin(), k.end(), std::ref(u8rng));
  std::vector<uint16_t> kernel_scale2d(nc * k2 / bl);
  std::generate(k.begin(), k.end(), std::ref(u8rng));
  std::generate(kernel_scale2d.begin(), kernel_scale2d.end(),
                [&]() { return math_cvt_bf16_fp32(scalerng()); });

  std::vector<xnn_qd8_quantization_params> quantization_params(
      mc + XNN_EXTRA_QUANTIZATION_PARAMS);
  const size_t w_bytes = packed_n * packed_k_bytes +
                         /* vksum */ packed_n * sizeof(float) +
                         /* scales */ packed_n * num_blocks * sizeof(float) +
                         /* bias */ packed_n * sizeof(float);

  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 + benchmark::utils::DivideRoundUp<size_t>(
                                     benchmark::utils::GetMaxCacheSize(),
                                     w_bytes + sizeof(uint16_t) * c_elements);

  std::vector<char, AlignedAllocator<char, 64>> w(w_bytes * num_buffers);
  std::fill(w.begin(), w.end(), 0);

  const xnn_qs8_qc4w_packing_params packing_params = {/*input_zero_point=*/1,
                                                      /*kernel_zero_point=*/8};
  pack(1, nc, k2, nr, kr, sr, bl, k.data(), /*bias=*/nullptr,
       /*scale=*/kernel_scale2d.data(), w.data(), sizeof(float) * nr,
       sizeof(float) * nr, &packing_params);
  std::vector<uint16_t> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  // Prepare parameters.
  xnn_f16_qb4w_minmax_params params;
  init_params(
      &params, fp16_ieee_from_fp32_value(std::numeric_limits<int8_t>::min()),
      fp16_ieee_from_fp32_value(std::numeric_limits<int8_t>::max()), 8, bl);

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache
    // state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size());
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      gemm(mb, nc, kc, a.data() + m * kc, kc * sizeof(int8_t),
           w.data() + w_bytes * buffer_index,
           c.data() + (buffer_index * mc + m) * nc, nc * sizeof(uint16_t),
           nr * sizeof(uint16_t), &params, quantization_params.data() + m);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["OPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * mc * nc * kc,
                         benchmark::Counter::kIsRate);
}

void GEMMBenchmark(benchmark::State& state,
                   xnn_qd8_f16_qc4w_gemm_ukernel_fn gemm,
                   xnn_init_f16_qc4w_minmax_params_fn init_params,
                   xnn_pack_qs8_qc4w_gemm_fn pack, size_t mr, size_t nr,
                   size_t kr, size_t sr,
                   benchmark::utils::IsaCheckFunction isa_check) {
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
  auto i8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             -std::numeric_limits<int8_t>::max(),
                             std::numeric_limits<int8_t>::max()),
                         std::ref(rng));
  auto u8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             0, std::numeric_limits<uint8_t>::max()),
                         std::ref(rng));

  std::vector<int8_t> a(mc * kc + XNN_EXTRA_BYTES);
  std::generate(a.begin(), a.end(), std::ref(i8rng));
  std::vector<uint8_t> k(nc * kc / 2);
  std::generate(k.begin(), k.end(), std::ref(u8rng));

  std::vector<xnn_qd8_quantization_params> quantization_params(
      mc + XNN_EXTRA_QUANTIZATION_PARAMS);
  const size_t w_elements =
      nc_stride * (sizeof(float) * 2 + sizeof(int32_t)) + kc_stride * nc_stride;

  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 + benchmark::utils::DivideRoundUp<size_t>(
                                     benchmark::utils::GetMaxCacheSize(),
                                     sizeof(float) * (w_elements + c_elements));

  std::vector<char, AlignedAllocator<char, 64>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0);

  const xnn_qs8_qc4w_packing_params packing_params = {/*input_zero_point=*/1,
                                                      /*kernel_zero_point=*/8};
  pack(1, nc, kc, nr, kr, sr, k.data(), /*bias=*/nullptr, /*scale=*/nullptr,
       w.data(), sizeof(float) * 2 * nr, &packing_params);
  std::vector<uint16_t> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  // Prepare parameters.
  xnn_f16_qc4w_minmax_params params;
  init_params(&params,
              fp16_ieee_from_fp32_value(std::numeric_limits<int8_t>::min()),
              fp16_ieee_from_fp32_value(std::numeric_limits<int8_t>::max()), 8);

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache
    // state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size());
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      gemm(mb, nc, kc, a.data() + m * kc, kc * sizeof(int8_t),
           w.data() + w_elements * buffer_index,
           c.data() + (buffer_index * mc + m) * nc, nc * sizeof(uint16_t),
           nr * sizeof(uint16_t), &params, quantization_params.data() + m);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["OPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * mc * nc * kc,
                         benchmark::Counter::kIsRate);
}

void GEMMBenchmark(benchmark::State& state,
                   xnn_qd8_f32_qb4w_gemm_ukernel_fn gemm,
                   xnn_init_f32_qb4w_minmax_params_fn init_params,
                   xnn_pack_qs8_qb4w_gemm_fn pack, size_t mr, size_t nr,
                   size_t kr, size_t sr,
                   benchmark::utils::IsaCheckFunction isa_check) {
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t bl = state.range(3);
  const size_t kc = round_up(state.range(2), bl);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             -std::numeric_limits<int8_t>::max(),
                             std::numeric_limits<int8_t>::max()),
                         std::ref(rng));
  auto u8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             0, std::numeric_limits<uint8_t>::max()),
                         std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.5f, 2.f),
                            std::ref(rng));

  const size_t planes = 2;  // 4 bit is 2 planes - low nibbles and high nibbles
  const size_t k2 = round_up_po2(kc, 2);  // tester assumes byte aligned rows
  const size_t packed_k2 =
      round_up_po2(kc, kr * sr * planes);  // 2 blocks for nibbles

  const size_t packed_k_bytes = (packed_k2 + 1) / 2;
  const size_t num_blocks = packed_k2 / bl;
  const size_t packed_n = round_up_po2(nc, nr);

  std::vector<int8_t> a(mc * kc + XNN_EXTRA_BYTES);
  std::generate(a.begin(), a.end(), std::ref(i8rng));
  std::vector<uint8_t> k(nc * kc / 2);
  std::generate(k.begin(), k.end(), std::ref(u8rng));
  std::vector<uint16_t> kernel_scale2d(nc * k2 / bl);
  std::generate(k.begin(), k.end(), std::ref(u8rng));
  std::generate(kernel_scale2d.begin(), kernel_scale2d.end(),
                [&]() { return math_cvt_bf16_fp32(scalerng()); });

  std::vector<xnn_qd8_quantization_params> quantization_params(
      mc + XNN_EXTRA_QUANTIZATION_PARAMS);
  const size_t w_bytes = packed_n * packed_k_bytes +
                         /* vksum */ packed_n * sizeof(float) +
                         /* scales */ packed_n * num_blocks * sizeof(float) +
                         /* bias */ packed_n * sizeof(float);

  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 + benchmark::utils::DivideRoundUp<size_t>(
                                     benchmark::utils::GetMaxCacheSize(),
                                     w_bytes + sizeof(float) * c_elements);

  std::vector<char, AlignedAllocator<char, 64>> w(w_bytes * num_buffers);
  std::fill(w.begin(), w.end(), 0);

  const xnn_qs8_qc4w_packing_params packing_params = {/*input_zero_point=*/1,
                                                      /*kernel_zero_point=*/8};
  pack(1, nc, k2, nr, kr, sr, bl, k.data(), /*bias=*/nullptr,
       /*scale=*/kernel_scale2d.data(), w.data(), sizeof(float) * nr,
       sizeof(float) * nr, &packing_params);
  std::vector<float> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  // Prepare parameters.
  xnn_f32_qb4w_minmax_params params;
  init_params(&params, std::numeric_limits<int8_t>::min(),
              std::numeric_limits<int8_t>::max(), 8, bl);

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache
    // state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size());
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      gemm(mb, nc, kc, a.data() + m * kc, kc * sizeof(int8_t),
           w.data() + w_bytes * buffer_index,
           c.data() + (buffer_index * mc + m) * nc, nc * sizeof(float),
           nr * sizeof(float), &params, quantization_params.data() + m);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["OPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * mc * nc * kc,
                         benchmark::Counter::kIsRate);
}

void GEMMBenchmark(benchmark::State& state,
                   xnn_qd8_f32_qc4w_gemm_ukernel_fn gemm,
                   xnn_init_f32_qc4w_minmax_params_fn init_params,
                   xnn_pack_qs8_qc4w_gemm_fn pack, size_t mr, size_t nr,
                   size_t kr, size_t sr,
                   benchmark::utils::IsaCheckFunction isa_check) {
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  const size_t nc_stride = round_up_po2(nc, nr);
  const size_t kc_stride = (round_up_po2(kc, kr * sr * 2) + 1) / 2;

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             -std::numeric_limits<int8_t>::max(),
                             std::numeric_limits<int8_t>::max()),
                         std::ref(rng));
  auto u8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             0, std::numeric_limits<uint8_t>::max()),
                         std::ref(rng));

  std::vector<int8_t> a(mc * kc + XNN_EXTRA_BYTES);
  std::generate(a.begin(), a.end(), std::ref(i8rng));
  std::vector<uint8_t> k(nc * kc / 2);
  std::generate(k.begin(), k.end(), std::ref(u8rng));

  std::vector<xnn_qd8_quantization_params> quantization_params(
      mc + XNN_EXTRA_QUANTIZATION_PARAMS);
  const size_t w_elements =
      nc_stride * (sizeof(float) * 2 + sizeof(int32_t)) + kc_stride * nc_stride;

  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 + benchmark::utils::DivideRoundUp<size_t>(
                                     benchmark::utils::GetMaxCacheSize(),
                                     sizeof(float) * (w_elements + c_elements));

  std::vector<char, AlignedAllocator<char, 64>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0);

  const xnn_qs8_qc4w_packing_params packing_params = {/*input_zero_point=*/1,
                                                      /*kernel_zero_point=*/8};
  pack(1, nc, kc, nr, kr, sr, k.data(), /*bias=*/nullptr, /*scale=*/nullptr,
       w.data(), sizeof(float) * 2 * nr, &packing_params);
  std::vector<float> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  // Prepare parameters.
  xnn_f32_qc4w_minmax_params params;
  init_params(&params, std::numeric_limits<int8_t>::min(),
              std::numeric_limits<int8_t>::max(), 0);

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache
    // state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size());
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      gemm(mb, nc, kc, a.data() + m * kc, kc * sizeof(int8_t),
           w.data() + w_elements * buffer_index,
           c.data() + (buffer_index * mc + m) * nc, nc * sizeof(float),
           nr * sizeof(float), &params, quantization_params.data() + m);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["OPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * mc * nc * kc,
                         benchmark::Counter::kIsRate);
}

void GEMMBenchmark(benchmark::State& state,
                   xnn_qp8_f32_qc4w_gemm_minmax_ukernel_fn gemm,
                   xnn_init_f32_minmax_params_fn init_minmax_params,
                   xnn_pack_weights_and_biases_fn pack_weights,
                   xnn_packed_stride_weights_and_biases_fn packed_stride,
                   size_t mr, size_t nr, size_t kr, size_t sr, size_t mr_packed,
                   benchmark::utils::IsaCheckFunction isa_check) {
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = round_up(state.range(2), 2UL);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f),
                          std::ref(rng));
  auto u8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             0, std::numeric_limits<uint8_t>::max()),
                         std::ref(rng));

  std::vector<float> a(mc * kc + XNN_EXTRA_BYTES);
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::vector<uint8_t> k(nc * kc / 2);
  std::generate(k.begin(), k.end(), std::ref(u8rng));

  // Create a fake `gemm_config` for the packing functions.
  struct xnn_gemm_config gemm_config;
  gemm_config.mr = static_cast<uint8_t>(mr);
  gemm_config.mr_packed = static_cast<uint8_t>(mr_packed);
  gemm_config.nr = static_cast<uint8_t>(nr);
  gemm_config.log2_kr = static_cast<uint8_t>(31 - math_clz_nonzero_u32(kr));
  gemm_config.log2_sr = static_cast<uint8_t>(31 - math_clz_nonzero_u32(sr));

  const size_t packed_w_stride =
      packed_stride(&gemm_config, kc, /*k_stride=*/kc, /*extra_bytes=*/0);
  const size_t packed_w_size = packed_w_stride * round_up(nc, nr);

  const size_t c_elements = mc * nc;
  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(),
              sizeof(float) * (packed_w_size + c_elements));

  std::vector<char, AlignedAllocator<char, 64>> w(packed_w_size * num_buffers);
  std::fill(w.begin(), w.end(), 0);

  // Quantize the left-hand operand.
  const size_t input_packed_size =
      xnn_x8_packq_f32qp8_packed_size(mc, kc, mr_packed, kr, sr);
  std::vector<int8_t> input_qp8(input_packed_size);
  xnn_x8_packq_f32qp8_ukernel__scalar_u1(mc, kc, mr_packed, kr, sr,
                                         /*m_idx_start=*/0, a.data(),
                                         /*lhs_stride=*/kc * sizeof(float),
                                         input_qp8.data());

  // RHS packing
  std::vector<float> kernel_scale(nc, 1.0f);
  const xnn_qs8_qc4w_packing_params packing_params = {/*input_zero_point=*/1,
                                                      /*kernel_zero_point=*/8};
  pack_weights(/*flags=*/0, &gemm_config, kc, nc,
               /*groups=*/1, /*k_stride=*/kc,
               /*accumulator_init=*/nullptr,
               /*weights=*/k.data(),
               /*int_extra_data0_fn=*/nullptr,
               /*extra_data0=*/nullptr,
               /*extra_data0_size=*/0,
               /*init_extra_data1_fn=*/
               nullptr,
               /*extra_data1=*/kernel_scale.data(),
               /*extra_data1_size=*/sizeof(float),
               /*packed_weights_ptr=*/w.data(), &packing_params);

  std::vector<float> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  // Prepare parameters.
  xnn_f32_minmax_params minmax_params;
  init_minmax_params(&minmax_params, -std::numeric_limits<float>::infinity(),
                     std::numeric_limits<float>::infinity());

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache
    // state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size());
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      gemm(mb, nc, kc,
           input_qp8.data() +
               xnn_x8_packq_f32qp8_packed_offset(m, kc, mr, kr, sr),
           w.data() + packed_w_size * buffer_index,
           c.data() + (buffer_index * mc + m) * nc, nc * sizeof(float),
           sizeof(float), &minmax_params);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["OPS"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * 2 * mc * nc * kc,
      benchmark::Counter::kIsRate);
}

void GEMMBenchmark(benchmark::State& state, xnn_qu8_gemm_minmax_ukernel_fn gemm,
                   xnn_init_qu8_conv_minmax_params_fn init_params,
                   xnn_pack_qu8_gemm_fn pack, size_t mr, size_t nr, size_t kr,
                   size_t sr,
                   benchmark::utils::IsaCheckFunction isa_check = nullptr) {
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
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000),
                          std::ref(rng));
  auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(
                             0, std::numeric_limits<uint8_t>::max()),
                         std::ref(rng));

  std::vector<uint8_t> a(mc * kc + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::generate(a.begin(), a.end(), std::ref(u8rng));
  std::vector<uint8_t> k(nc * kc);
  std::generate(k.begin(), k.end(), std::ref(u8rng));
  std::vector<int32_t> b(nc);
  std::generate(b.begin(), b.end(), std::ref(i32rng));

  const size_t w_elements =
      kc_stride * nc_stride + nc_stride * sizeof(int32_t) / sizeof(uint8_t);
  const size_t c_elements = mc * nc;
  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(),
              sizeof(uint8_t) * (w_elements + c_elements));

  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> w(w_elements *
                                                        num_buffers);
  std::fill(w.begin(), w.end(), 0);
  const xnn_qu8_packing_params packing_params = {127, 127};
  pack(/*groups=*/1, nc, kc, nr, kr, sr, k.data(), b.data(), /*scale=*/nullptr,
       w.data(),
       /*extra_bytes=*/0, &packing_params);
  std::vector<uint8_t> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), 0xA5);

  union xnn_qu8_conv_minmax_params quantization_params;
  init_params(&quantization_params, 127, 0.75f, 127, 1, 254);

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache
    // state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(uint8_t));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      for (uint32_t n = 0; n < nc; n += nr) {
        const uint32_t nb = min(nc - n, nr);
        gemm(mb, nb, kc * sizeof(uint8_t), a.data() + m * kc,
             kc * sizeof(uint8_t),
             w.data() + (w_elements * buffer_index +
                         n * (kc_stride + sizeof(int32_t))) /
                            sizeof(uint8_t),
             c.data() + (mc * buffer_index + m) * nc + n, nc * sizeof(uint8_t),
             nr * sizeof(uint8_t), &quantization_params);
      }
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["OPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * mc * nc * kc,
                         benchmark::Counter::kIsRate);
}

void GEMMBenchmark(benchmark::State& state, xnn_f32_gemm_minmax_ukernel_fn gemm,
                   xnn_init_f32_minmax_params_fn init_params,
                   xnn_pack_f32_gemm_fn pack, size_t mr, size_t nr, size_t kr,
                   size_t sr,
                   benchmark::utils::IsaCheckFunction isa_check = nullptr) {
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
  auto f32rng =
      std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<float> a(mc * kc + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::vector<float> k(nc * kc);
  std::generate(k.begin(), k.end(), std::ref(f32rng));
  std::vector<float> b(nc);
  std::generate(b.begin(), b.end(), std::ref(f32rng));

  const size_t w_elements = nc_stride * kc_stride + nc_stride;
  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 + benchmark::utils::DivideRoundUp<size_t>(
                                     benchmark::utils::GetMaxCacheSize(),
                                     sizeof(float) * (w_elements + c_elements));

  std::vector<float, AlignedAllocator<float, 64>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0.0f);
  pack(/*groups=*/1, nc, kc, nr, kr, sr, k.data(), b.data(), /*scale=*/nullptr,
       w.data(), /*extra_bytes=*/0, /*params=*/nullptr);
  std::vector<float> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  xnn_f32_minmax_params params;
  init_params(&params, -std::numeric_limits<float>::infinity(),
              +std::numeric_limits<float>::infinity());

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache
    // state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(float));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      gemm(mb, nc, kc * sizeof(float), a.data() + m * kc, kc * sizeof(float),
           w.data() + buffer_index * nc_stride * (kc_stride + 1),
           c.data() + (buffer_index * mc + m) * nc, nc * sizeof(float),
           nr * sizeof(float), &params);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * mc * nc * kc,
                         benchmark::Counter::kIsRate);
}

void GEMMBenchmark(benchmark::State& state, xnn_f32_gemm_minmax_ukernel_fn gemm,
                   xnn_init_f32_minmax_params_fn init_params, size_t mr,
                   size_t nr, size_t kr, size_t sr,
                   benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng =
      std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<float> a(mc * kc + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(a.begin(), a.end(), std::ref(f32rng));

  const size_t k_elements = nc * kc;
  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 + benchmark::utils::DivideRoundUp<size_t>(
                                     benchmark::utils::GetMaxCacheSize(),
                                     sizeof(float) * (k_elements + c_elements));

  std::vector<float> k(k_elements * num_buffers);
  std::vector<float> c(c_elements * num_buffers);
  std::generate(k.begin(), k.end(), std::ref(f32rng));
  std::fill(c.begin(), c.end(), std::nanf(""));

  xnn_f32_minmax_params params;
  init_params(&params, -std::numeric_limits<float>::infinity(),
              +std::numeric_limits<float>::infinity());

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache
    // state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - K is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(float));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      gemm(mb, nc, kc * sizeof(float), a.data() + m * kc, kc * sizeof(float),
           k.data() + (buffer_index * k_elements),
           c.data() + (buffer_index * mc + m) * nc, nc * sizeof(float),
           nr * sizeof(float), &params);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * mc * nc * kc,
                         benchmark::Counter::kIsRate);
}

void GEMMBenchmark(benchmark::State& state, xnn_f16_gemm_minmax_ukernel_fn gemm,
                   xnn_init_f16_minmax_params_fn init_params,
                   xnn_pack_f16_gemm_fn pack, size_t mr, size_t nr, size_t kr,
                   size_t sr,
                   benchmark::utils::IsaCheckFunction isa_check = nullptr) {
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
  auto f32rng =
      std::bind(std::uniform_real_distribution<float>(), std::ref(rng));
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

  std::vector<uint16_t> a(mc * kc + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::generate(a.begin(), a.end(), std::ref(f16rng));
  std::vector<uint16_t> k(nc * kc);
  std::generate(k.begin(), k.end(), std::ref(f16rng));
  std::vector<uint16_t> b(nc);
  std::generate(b.begin(), b.end(), std::ref(f16rng));

  const size_t w_elements = nc_stride * kc_stride + nc_stride;
  const size_t c_elements = mc * nc;
  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(),
              sizeof(uint16_t) * (w_elements + c_elements));

  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> w(w_elements *
                                                          num_buffers);
  std::fill(w.begin(), w.end(), 0);
  pack(/*groups=*/1, nc, kc, nr, kr, sr, k.data(), b.data(), /*scale=*/nullptr,
       w.data(), /*extra_bytes=*/0, /*params=*/nullptr);
  std::vector<uint16_t> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), UINT16_C(0x7E00) /* NaN */);

  // Prepare minmax parameters.
  xnn_f16_minmax_params params;
  init_params(&params, UINT16_C(0xFC00) /* -inf */, UINT16_C(0x7C00) /* inf */);

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache
    // state:
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
        gemm(mb, nb, kc * sizeof(uint16_t), a.data() + m * kc,
             kc * sizeof(uint16_t),
             w.data() + (nc_stride * buffer_index + n) * (kc_stride + 1),
             c.data() + (mc * buffer_index + m) * nc + n, nc * sizeof(uint16_t),
             nr * sizeof(uint16_t), &params);
      }
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * mc * nc * kc,
                         benchmark::Counter::kIsRate);
}

