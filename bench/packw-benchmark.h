// Copyright 2023-2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>

#include "bench/bgemm.h"
#include "bench/utils.h"
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/pack.h"
#include "test/replicable_random_device.h"
#include <benchmark/benchmark.h>

static void x8_packw(benchmark::State& state,
                     xnn_x8_packw_gemm_goi_ukernel_fn packw, size_t nr,
                     size_t kr, size_t sr, uint64_t arch_flags = 0) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t batch = state.range(0);  // batch is g parameter for packw
  const size_t dim_n = state.range(2);  // dim_n is nc parameter
  const size_t dim_k = state.range(3);  // dim_k is kc parameter

  const size_t rounded_n = benchmark::utils::RoundUp(dim_n, nr);
  const size_t rounded_k = benchmark::utils::RoundUp(dim_k, kr * sr);
  const size_t rounded_size =
      rounded_n * rounded_k + rounded_n * sizeof(uint32_t);

  xnnpack::ReplicableRandomDevice rng;

  // Computer num_buffers that fit cache with source weights + packed_weights.
  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(),
              sizeof(int8_t) * batch * (dim_n * dim_k + rounded_size));

  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> weights(
      num_buffers * batch * dim_n * dim_k);
  xnnpack::fill_uniform_random_bits(weights.data(), weights.size(), rng);
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      num_buffers * batch * rounded_size);

  const xnn_qs8_packw_params params = {127};

  size_t buffer_index = 0;
  for (auto _ : state) {
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    packw(batch, dim_n, dim_k, nr, kr, sr,
          weights.data() + buffer_index * batch * dim_n * dim_k,
          /*bias=*/nullptr, /*scale=*/nullptr,
          packed_weights.data() + buffer_index * batch * rounded_size,
          /*extra_bytes=*/0, &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = batch * dim_n * dim_k;
  state.counters["elements"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * elements_per_iteration,
      benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration =
      (elements_per_iteration + batch * rounded_size);
  state.counters["bytes"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * bytes_per_iteration,
      benchmark::Counter::kIsRate);
}

static void x8_gio_packw(benchmark::State& state,
                         xnn_x8_packw_gemm_gio_ukernel_fn packw, size_t nr,
                         size_t kr, size_t sr, uint64_t arch_flags = 0) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t batch = state.range(0);  // batch is g parameter for packw
  const size_t dim_n = state.range(2);  // dim_n is nc parameter
  const size_t dim_k = state.range(3);  // dim_k is kc parameter

  const size_t rounded_n = benchmark::utils::RoundUp(dim_n, nr);
  const size_t rounded_k = benchmark::utils::RoundUp(dim_k, kr * sr);
  const size_t rounded_size =
      rounded_n * rounded_k + rounded_n * sizeof(uint32_t);

  xnnpack::ReplicableRandomDevice rng;

  // Computer num_buffers that fit cache with source weights + packed_weights.
  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(),
              sizeof(int8_t) * batch *
                  (dim_n * dim_k + rounded_n * rounded_k + rounded_n));

  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> weights(
      num_buffers * batch * dim_n * dim_k);
  xnnpack::fill_uniform_random_bits(weights.data(), weights.size(), rng);
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      num_buffers * batch * rounded_size);

  const xnn_qs8_packw_params params = {127};

  size_t buffer_index = 0;
  for (auto _ : state) {
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    packw(batch, dim_n, dim_k, nr, kr, sr, dim_n /* k_stride */,
          weights.data() + buffer_index * batch * dim_n * dim_k,
          /*bias=*/nullptr, /*scale=*/nullptr,
          packed_weights.data() + buffer_index * batch * rounded_size,
          /*extra_bytes=*/0, &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = batch * dim_n * dim_k;
  state.counters["elements"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * elements_per_iteration,
      benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration =
      elements_per_iteration + batch * rounded_size;
  state.counters["bytes"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * bytes_per_iteration,
      benchmark::Counter::kIsRate);
}

static void qb4_packw(benchmark::State& state,
                      xnn_qb4_packw_gemm_goi_ukernel_fn packw, size_t nr,
                      size_t kr, size_t sr, size_t bl, bool null_bias,
                      uint64_t arch_flags = 0) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t batch = state.range(0);  // batch is g parameter for packw
  const size_t dim_n = state.range(2);  // dim_n is nc parameter
  const size_t dim_k = round_up(state.range(3), bl);  // dim_k is kc parameter

  const size_t k2 = round_up_po2(dim_k, 2);  // tester assumes byte aligned rows
  size_t rounded_k2 = round_up_po2(k2, kr * sr * 2);
  const size_t rounded_n = round_up_po2(dim_n, nr);

  const size_t num_blocks = (rounded_k2 / bl);
  const size_t rounded_k_bytes = (rounded_k2 + 1);
  const size_t rounded_size =
      rounded_n * (rounded_k_bytes + sizeof(float) +
                   num_blocks * sizeof(uint16_t) + sizeof(float));

  xnnpack::ReplicableRandomDevice rng;

  // Computer num_buffers that fit cache with source weights + packed_weights.
  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(),
              sizeof(int8_t) * batch *
                  ((dim_n * dim_k + 1) / 2 + batch * rounded_size));

  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> weights(
      num_buffers * ((batch * (dim_n * dim_k + 1) / 2) + XNN_EXTRA_BYTES));
  xnnpack::fill_uniform_random_bits(weights.data(), weights.size(), rng);
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      num_buffers * batch * rounded_size);
  xnnpack::Buffer<int32_t, XNN_ALLOCATION_ALIGNMENT> bias(num_buffers * batch *
                                                          rounded_n);
  xnnpack::fill_uniform_random_bits(bias.data(), bias.size(), rng);
  xnnpack::Buffer<xnn_bfloat16, XNN_ALLOCATION_ALIGNMENT> bf16_scales(
      num_buffers * dim_n * num_blocks * batch);
  xnnpack::fill_uniform_random_bits(bf16_scales.data(), bf16_scales.size(),
                                    rng);

  const xnn_qs8_qc4w_packing_params packing_params = {1, 8};

  size_t buffer_index = 0;
  for (auto _ : state) {
    buffer_index = (buffer_index + 1) % num_buffers;
    int32_t* bias_ptr =
        null_bias ? nullptr : bias.data() + (buffer_index * batch * dim_n);

    packw(batch, dim_n, dim_k, nr, kr, sr, bl,
          weights.data() + buffer_index * (batch * (dim_n * dim_k + 1) / 2),
          /*bias=*/bias_ptr,
          /*scale=*/bf16_scales.data() +
              (buffer_index * batch * dim_n * num_blocks),
          packed_weights.data() + buffer_index * batch * rounded_size,
          /*extra_bytes_bl=*/sizeof(uint16_t) * nr, sizeof(float) * nr,
          &packing_params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = batch * dim_n * dim_k;
  state.counters["elements"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * elements_per_iteration,
      benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration =
      (elements_per_iteration + batch * rounded_size);
  state.counters["bytes"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * bytes_per_iteration,
      benchmark::Counter::kIsRate);
}

static void qs8_packw(benchmark::State& state,
                      xnn_qs8_packw_gemm_goi_ukernel_fn packw, size_t nr,
                      size_t kr, size_t sr, uint64_t arch_flags = 0) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t batch = state.range(0);  // batch is g parameter for packw
  const size_t dim_n = state.range(2);  // dim_n is nc parameter
  const size_t dim_k = state.range(3);  // dim_k is kc parameter

  const size_t rounded_n = benchmark::utils::RoundUp(dim_n, nr);
  const size_t rounded_k = benchmark::utils::RoundUp(dim_k, kr * sr);
  const size_t rounded_size =
      rounded_n * rounded_k + rounded_n * sizeof(uint32_t);

  xnnpack::ReplicableRandomDevice rng;

  // Computer num_buffers that fit cache with source weights + packed_weights.
  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(),
              sizeof(int8_t) * batch * (dim_n * dim_k + rounded_size));

  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> weights(
      num_buffers * batch * dim_n * dim_k);
  xnnpack::fill_uniform_random_bits(weights.data(), weights.size(), rng);
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      num_buffers * batch * rounded_size);

  const xnn_qs8_packw_params params = {127};

  size_t buffer_index = 0;
  for (auto _ : state) {
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    packw(batch, dim_n, dim_k, nr, kr, sr,
          weights.data() + buffer_index * batch * dim_n * dim_k,
          /*bias=*/nullptr, /*scale=*/nullptr,
          packed_weights.data() + buffer_index * batch * rounded_size,
          /*extra_bytes=*/0, &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = batch * dim_n * dim_k;
  state.counters["elements"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * elements_per_iteration,
      benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration =
      (elements_per_iteration + batch * rounded_size);
  state.counters["bytes"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * bytes_per_iteration,
      benchmark::Counter::kIsRate);
}

static void qs8_gio_packw(benchmark::State& state,
                          xnn_qs8_packw_gemm_gio_ukernel_fn packw, size_t nr,
                          size_t kr, size_t sr, uint64_t arch_flags = 0) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t batch = state.range(0);  // batch is g parameter for packw
  const size_t dim_n = state.range(2);  // dim_n is nc parameter
  const size_t dim_k = state.range(3);  // dim_k is kc parameter

  const size_t rounded_n = benchmark::utils::RoundUp(dim_n, nr);
  const size_t rounded_k = benchmark::utils::RoundUp(dim_k, kr * sr);
  const size_t rounded_size =
      rounded_n * rounded_k + rounded_n * sizeof(uint32_t);

  xnnpack::ReplicableRandomDevice rng;

  // Computer num_buffers that fit cache with source weights + packed_weights.
  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(),
              sizeof(int8_t) * batch *
                  (dim_n * dim_k + rounded_n * rounded_k + rounded_n));

  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> weights(
      num_buffers * batch * dim_n * dim_k);
  xnnpack::fill_uniform_random_bits(weights.data(), weights.size(), rng);
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      num_buffers * batch * rounded_size);

  const xnn_qs8_packw_params params = {127};

  size_t buffer_index = 0;
  for (auto _ : state) {
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    packw(batch, dim_n, dim_k, nr, kr, sr, dim_n,
          weights.data() + buffer_index * batch * dim_n * dim_k,
          /*bias=*/nullptr, /*scale=*/nullptr,
          packed_weights.data() + buffer_index * batch * rounded_size,
          /*extra_bytes=*/0, &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = batch * dim_n * dim_k;
  state.counters["elements"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * elements_per_iteration,
      benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration =
      (elements_per_iteration + batch * rounded_size);
  state.counters["bytes"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * bytes_per_iteration,
      benchmark::Counter::kIsRate);
}

static void qs8_qc4w_packw(benchmark::State& state,
                           xnn_qs8_qc4w_packw_gemm_goi_ukernel_fn packw,
                           size_t nr, size_t kr, size_t sr,
                           uint64_t arch_flags = 0) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t batch = state.range(0);  // batch is g parameter for packw
  const size_t dim_n = state.range(2);  // dim_n is nc parameter
  const size_t dim_k = state.range(3);  // dim_k is kc parameter

  const size_t rounded_n = benchmark::utils::RoundUp(dim_n, nr);
  const size_t rounded_k = benchmark::utils::RoundUp(dim_k, kr * sr);
  const size_t rounded_size =
      rounded_n * rounded_k / 2 + rounded_n * sizeof(uint32_t);

  xnnpack::ReplicableRandomDevice rng;

  // Computer num_buffers that fit cache with source weights + packed_weights.
  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(),
              sizeof(int8_t) * batch * (dim_n * dim_k + rounded_size));

  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> weights(
      num_buffers * batch * (dim_n * dim_k + 1) / 2);
  xnnpack::fill_uniform_random_bits(weights.data(), weights.size(), rng);
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      num_buffers * batch * rounded_size);

  const xnn_qs8_qc4w_packing_params params = {0, 0};

  size_t buffer_index = 0;
  for (auto _ : state) {
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    packw(batch, dim_n, dim_k, nr, kr, sr,
          weights.data() + buffer_index * batch * (dim_n * dim_k + 1) / 2,
          /*bias=*/nullptr, /*scale=*/nullptr,
          packed_weights.data() + buffer_index * batch * rounded_size,
          /*extra_bytes=*/0, &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = batch * dim_n * dim_k;
  state.counters["elements"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * elements_per_iteration,
      benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration =
      (elements_per_iteration + batch * rounded_size);
  state.counters["bytes"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * bytes_per_iteration,
      benchmark::Counter::kIsRate);
}

static void x16_packw(benchmark::State& state,
                      xnn_x16_packw_gemm_goi_ukernel_fn packw, size_t nr,
                      size_t kr, size_t sr, uint64_t arch_flags = 0) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t batch = state.range(0);  // batch is g parameter for packw
  const size_t dim_n = state.range(2);  // dim_n is nc parameter
  const size_t dim_k = state.range(3);  // dim_k is kc parameter

  const size_t rounded_n = benchmark::utils::RoundUp(dim_n, nr);
  const size_t rounded_k = benchmark::utils::RoundUp(dim_k, kr * sr);

  xnnpack::ReplicableRandomDevice rng;

  // Computer num_buffers that fit cache with source weights + packed_weights.
  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(),
              sizeof(uint16_t) * batch *
                  (dim_n * dim_k + rounded_n * rounded_k + rounded_n));

  xnnpack::Buffer<uint16_t, XNN_ALLOCATION_ALIGNMENT> weights(
      num_buffers * batch * dim_n * dim_k);
  xnnpack::fill_uniform_random_bits(weights.data(), weights.size(), rng);
  xnnpack::Buffer<uint16_t, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      num_buffers * batch * (rounded_n * rounded_k + rounded_n));

  size_t buffer_index = 0;
  for (auto _ : state) {
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    packw(batch, dim_n, dim_k, nr, kr, sr,
          reinterpret_cast<uint16_t*>(weights.data() +
                                      buffer_index * batch * dim_n * dim_k),
          /*bias=*/nullptr, /*scale=*/nullptr,
          reinterpret_cast<uint16_t*>(packed_weights.data() +
                                      buffer_index * batch *
                                          (rounded_n * rounded_k + rounded_n)),
          /*extra_bytes=*/0, /*params=*/nullptr);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = batch * dim_n * dim_k;
  state.counters["elements"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * elements_per_iteration,
      benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration =
      (elements_per_iteration + batch * (rounded_n * rounded_k + rounded_n)) *
      sizeof(uint16_t);
  state.counters["bytes"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * bytes_per_iteration,
      benchmark::Counter::kIsRate);
}

static void x16_x32_packw(benchmark::State& state,
                          xnn_x16_x32_packw_gemm_goi_ukernel_fn packw,
                          size_t nr, size_t kr, size_t sr,
                          uint64_t arch_flags = 0) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t batch = state.range(0);  // batch is g parameter for packw
  const size_t dim_n = state.range(2);  // dim_n is nc parameter
  const size_t dim_k = state.range(3);  // dim_k is kc parameter

  const size_t rounded_n = benchmark::utils::RoundUp(dim_n, nr);
  const size_t rounded_k = benchmark::utils::RoundUp(dim_k, kr * sr);

  xnnpack::ReplicableRandomDevice rng;

  // Computer num_buffers that fit cache with source weights + packed_weights.
  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(),
              sizeof(uint16_t) * batch *
                  (dim_n * dim_k + rounded_n * rounded_k + rounded_n));

  xnnpack::Buffer<uint16_t, XNN_ALLOCATION_ALIGNMENT> weights(
      num_buffers * batch * dim_n * dim_k);
  xnnpack::fill_uniform_random_bits(weights.data(), weights.size(), rng);
  xnnpack::Buffer<uint16_t, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      num_buffers * batch * (rounded_n * rounded_k + 2 * rounded_n));

  size_t buffer_index = 0;
  for (auto _ : state) {
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    packw(batch, dim_n, dim_k, nr, kr, sr,
          reinterpret_cast<uint16_t*>(weights.data() +
                                      buffer_index * batch * dim_n * dim_k),
          /*bias=*/nullptr, /*scale=*/nullptr,
          reinterpret_cast<uint16_t*>(packed_weights.data() +
                                      buffer_index * batch *
                                          (rounded_n * rounded_k + rounded_n)),
          /*extra_bytes=*/0, /*params=*/nullptr);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = batch * dim_n * dim_k;
  state.counters["elements"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * elements_per_iteration,
      benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration =
      (elements_per_iteration + batch * (rounded_n * rounded_k + rounded_n)) *
      sizeof(uint16_t);
  state.counters["bytes"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * bytes_per_iteration,
      benchmark::Counter::kIsRate);
}

static void x32_packw(benchmark::State& state,
                      xnn_x32_packw_gemm_goi_ukernel_fn packw, size_t nr,
                      size_t kr, size_t sr, uint64_t arch_flags = 0) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t batch = state.range(0);  // batch is g parameter for packw
  const size_t dim_n = state.range(2);  // dim_n is nc parameter
  const size_t dim_k = state.range(3);  // dim_k is kc parameter

  const size_t rounded_n = benchmark::utils::RoundUp(dim_n, nr);
  const size_t rounded_k = benchmark::utils::RoundUp(dim_k, kr * sr);

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng =
      std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  // Computer num_buffers that fit cache with source weights + packed_weights.
  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(),
              sizeof(float) * batch *
                  (dim_n * dim_k + rounded_n * rounded_k + rounded_n));

  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> weights(num_buffers * batch *
                                                           dim_n * dim_k);
  std::generate(weights.begin(), weights.end(), std::ref(f32rng));
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      num_buffers * batch * (rounded_n * rounded_k + rounded_n));

  size_t buffer_index = 0;
  for (auto _ : state) {
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    packw(batch, dim_n, dim_k, nr, kr, sr,
          reinterpret_cast<uint32_t*>(weights.data() +
                                      buffer_index * batch * dim_n * dim_k),
          /*bias=*/nullptr,
          /*scale=*/nullptr,
          reinterpret_cast<uint32_t*>(packed_weights.data() +
                                      buffer_index * batch *
                                          (rounded_n * rounded_k + rounded_n)),
          /*extra_bytes=*/0, /*params=*/nullptr);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = batch * dim_n * dim_k;
  state.counters["elements"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * elements_per_iteration,
      benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration =
      (elements_per_iteration + batch * (rounded_n * rounded_k + rounded_n)) *
      sizeof(float);
  state.counters["bytes"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * bytes_per_iteration,
      benchmark::Counter::kIsRate);
}

static void x32_gio_packw(benchmark::State& state,
                          xnn_x32_packw_gemm_gio_ukernel_fn packw, size_t nr,
                          size_t kr, size_t sr, uint64_t arch_flags = 0) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t batch = state.range(0);  // batch is g parameter for packw
  const size_t dim_n = state.range(2);  // dim_n is nc parameter
  const size_t dim_k = state.range(3);  // dim_k is kc parameter

  const size_t rounded_n = benchmark::utils::RoundUp(dim_n, nr);
  const size_t rounded_k = benchmark::utils::RoundUp(dim_k, kr * sr);

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng =
      std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  // Computer num_buffers that fit cache with source weights + packed_weights.
  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(),
              sizeof(float) * batch *
                  (dim_n * dim_k + rounded_n * rounded_k + rounded_n));

  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> weights(num_buffers * batch *
                                                           dim_n * dim_k);
  std::generate(weights.begin(), weights.end(), std::ref(f32rng));
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      num_buffers * batch * (rounded_n * rounded_k + rounded_n));

  size_t buffer_index = 0;
  for (auto _ : state) {
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    packw(batch, dim_n, dim_k, nr, kr, sr, dim_n /* k_stride */,
          reinterpret_cast<uint32_t*>(weights.data() +
                                      buffer_index * batch * dim_n * dim_k),
          /*bias=*/nullptr,
          /*scale=*/nullptr,
          reinterpret_cast<uint32_t*>(packed_weights.data() +
                                      buffer_index * batch *
                                          (rounded_n * rounded_k + rounded_n)),
          /*extra_bytes=*/0, /*params=*/nullptr);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = batch * dim_n * dim_k;
  state.counters["elements"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * elements_per_iteration,
      benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration =
      (elements_per_iteration + batch * (rounded_n * rounded_k + rounded_n)) *
      sizeof(float);
  state.counters["bytes"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * bytes_per_iteration,
      benchmark::Counter::kIsRate);
}

static void x8_packw__reference(size_t batch, size_t dim_n, size_t dim_k,
                                size_t nr, size_t kr, size_t sr,
                                const int8_t* weights, const uint32_t* bias,
                                const void* scale, int8_t* packed_weights,
                                size_t extra_bytes, const void* params) {
  xnn_pack_f32_qs8w_gemm_goi_w(
      batch, dim_n, dim_k, nr, kr, sr, reinterpret_cast<const int8_t*>(weights),
      reinterpret_cast<const float*>(bias), static_cast<const float*>(scale),
      static_cast<void*>(packed_weights), extra_bytes, params);
}

static void x8_packw_x2__reference(benchmark::State& state, const char* net) {
  x8_packw(state, x8_packw__reference,
           /*nr=*/2, /*kr=*/1, /*sr=*/1);
}
static void x8_packw_x4__reference(benchmark::State& state, const char* net) {
  x8_packw(state, x8_packw__reference,
           /*nr=*/4, /*kr=*/1, /*sr=*/1);
}
static void x8_packw_x8__reference(benchmark::State& state, const char* net) {
  x8_packw(state, x8_packw__reference,
           /*nr=*/8, /*kr=*/1, /*sr=*/1);
}
static void x8_packw_x16__reference(benchmark::State& state, const char* net) {
  x8_packw(state, x8_packw__reference,
           /*nr=*/16, /*kr=*/1, /*sr=*/1);
}
static void x8_packw_x32__reference(benchmark::State& state, const char* net) {
  x8_packw(state, x8_packw__reference,
           /*nr=*/32, /*kr=*/1, /*sr=*/1);
}

BENCHMARK_BGEMM(x8_packw_x2__reference)
BENCHMARK_BGEMM(x8_packw_x4__reference)
BENCHMARK_BGEMM(x8_packw_x8__reference)
BENCHMARK_BGEMM(x8_packw_x16__reference)
BENCHMARK_BGEMM(x8_packw_x32__reference)

static void x8_packw_gio__reference(size_t batch, size_t dim_n, size_t dim_k,
                                    size_t nr, size_t kr, size_t sr,
                                    const int8_t* weights, const uint32_t* bias,
                                    const void* scale, int8_t* packed_weights,
                                    size_t extra_bytes, const void* params) {
  xnn_pack_f32_qs8w_gemm_gio_w(
      batch, dim_n, dim_k, nr, kr, sr, dim_n,
      reinterpret_cast<const int8_t*>(weights),
      reinterpret_cast<const float*>(bias), static_cast<const float*>(scale),
      static_cast<void*>(packed_weights), extra_bytes, params);
}

static void x8_packw_gio_x2__reference(benchmark::State& state,
                                       const char* net) {
  x8_packw(state, x8_packw_gio__reference,
           /*nr=*/2, /*kr=*/1, /*sr=*/1);
}
static void x8_packw_gio_x4__reference(benchmark::State& state,
                                       const char* net) {
  x8_packw(state, x8_packw_gio__reference,
           /*nr=*/4, /*kr=*/1, /*sr=*/1);
}
static void x8_packw_gio_x8__reference(benchmark::State& state,
                                       const char* net) {
  x8_packw(state, x8_packw_gio__reference,
           /*nr=*/8, /*kr=*/1, /*sr=*/1);
}
static void x8_packw_gio_x16__reference(benchmark::State& state,
                                        const char* net) {
  x8_packw(state, x8_packw_gio__reference,
           /*nr=*/16, /*kr=*/1, /*sr=*/1);
}
static void x8_packw_gio_x32__reference(benchmark::State& state,
                                        const char* net) {
  x8_packw(state, x8_packw_gio__reference,
           /*nr=*/32, /*kr=*/1, /*sr=*/1);
}

static void x8_packw_gio_x8c8__reference(benchmark::State& state,
                                         const char* net) {
  x8_packw(state, x8_packw_gio__reference,
           /*nr=*/8, /*kr=*/8, /*sr=*/1);
}

BENCHMARK_BGEMM(x8_packw_gio_x2__reference)
BENCHMARK_BGEMM(x8_packw_gio_x4__reference)
BENCHMARK_BGEMM(x8_packw_gio_x8__reference)
BENCHMARK_BGEMM(x8_packw_gio_x16__reference)
BENCHMARK_BGEMM(x8_packw_gio_x32__reference)
BENCHMARK_BGEMM(x8_packw_gio_x8c8__reference)

static void qs8_packw__reference(size_t batch, size_t dim_n, size_t dim_k,
                                 size_t nr, size_t kr, size_t sr,
                                 const int8_t* weights, const int32_t* bias,
                                 const void* scale, int8_t* packed_weights,
                                 size_t extra_bytes, const void* params) {
  xnn_pack_qs8_gemm_goi_w(
      batch, dim_n, dim_k, nr, kr, sr, reinterpret_cast<const int8_t*>(weights),
      reinterpret_cast<const int32_t*>(bias), static_cast<const float*>(scale),
      static_cast<void*>(packed_weights), extra_bytes,
      reinterpret_cast<const struct xnn_qs8_packing_params*>(params));
}

static void qs8_packw_x2c4__reference(benchmark::State& state,
                                      const char* net) {
  qs8_packw(state, qs8_packw__reference,
            /*nr=*/2, /*kr=*/4, /*sr=*/1);
}
static void qs8_packw_x8c4__reference(benchmark::State& state,
                                      const char* net) {
  qs8_packw(state, qs8_packw__reference,
            /*nr=*/8, /*kr=*/4, /*sr=*/1);
}
static void qs8_packw_x16c4__reference(benchmark::State& state,
                                       const char* net) {
  qs8_packw(state, qs8_packw__reference,
            /*nr=*/16, /*kr=*/4, /*sr=*/1);
}
static void qs8_packw_x64c4__reference(benchmark::State& state,
                                       const char* net) {
  qs8_packw(state, qs8_packw__reference,
            /*nr=*/64, /*kr=*/4, /*sr=*/1);
}

BENCHMARK_BGEMM(qs8_packw_x2c4__reference)
BENCHMARK_BGEMM(qs8_packw_x8c4__reference)
BENCHMARK_BGEMM(qs8_packw_x16c4__reference)
BENCHMARK_BGEMM(qs8_packw_x64c4__reference)

static void qs8_packw_x8c8__reference(benchmark::State& state,
                                      const char* net) {
  qs8_packw(state, qs8_packw__reference,
            /*nr=*/8, /*kr=*/8, /*sr=*/1);
}
static void qs8_packw_x16c8__reference(benchmark::State& state,
                                       const char* net) {
  qs8_packw(state, qs8_packw__reference,
            /*nr=*/16, /*kr=*/8, /*sr=*/1);
}

BENCHMARK_BGEMM(qs8_packw_x8c8__reference)
BENCHMARK_BGEMM(qs8_packw_x16c8__reference)

static void qs8_packw_gio__reference(size_t batch, size_t dim_n, size_t dim_k,
                                     size_t nr, size_t kr, size_t sr,
                                     const int8_t* weights, const int32_t* bias,
                                     const void* scale, int8_t* packed_weights,
                                     size_t extra_bytes, const void* params) {
  xnn_pack_qs8_gemm_gio_w(
      batch, dim_n, dim_k, nr, kr, sr, dim_n,
      reinterpret_cast<const int8_t*>(weights),
      reinterpret_cast<const int32_t*>(bias), static_cast<const float*>(scale),
      static_cast<void*>(packed_weights), extra_bytes,
      reinterpret_cast<const struct xnn_qs8_packing_params*>(params));
}

static void qs8_packw_gio_x8c8__reference(benchmark::State& state,
                                          const char* net) {
  qs8_packw(state, qs8_packw_gio__reference,
            /*nr=*/8, /*kr=*/8, /*sr=*/1);
}
static void qs8_packw_gio_x16c8__reference(benchmark::State& state,
                                           const char* net) {
  qs8_packw(state, qs8_packw_gio__reference,
            /*nr=*/16, /*kr=*/8, /*sr=*/1);
}

BENCHMARK_BGEMM(qs8_packw_gio_x8c8__reference)
BENCHMARK_BGEMM(qs8_packw_gio_x16c8__reference)

static void qs8_qc4w_packw__reference(
    size_t batch, size_t dim_n, size_t dim_k, size_t nr, size_t kr, size_t sr,
    const uint8_t* weights, const int32_t* bias, const float* scale,
    void* packed_weights, size_t extra_bytes,
    const xnn_qs8_qc4w_packing_params* params) {
  xnn_pack_qs8_qc4w_gemm_goi_w(
      batch, dim_n, dim_k, nr, kr, sr,
      reinterpret_cast<const uint8_t*>(weights),
      reinterpret_cast<const int32_t*>(bias), static_cast<const float*>(scale),
      static_cast<void*>(packed_weights), extra_bytes,
      reinterpret_cast<const struct xnn_qs8_qc4w_packing_params*>(params));
}

static void qs8_qc4w_packw_x8c8__reference(benchmark::State& state,
                                           const char* net) {
  qs8_qc4w_packw(state, qs8_qc4w_packw__reference,
                 /*nr=*/8, /*kr=*/8, /*sr=*/1);
}
static void qs8_qc4w_packw_x16c8__reference(benchmark::State& state,
                                            const char* net) {
  qs8_qc4w_packw(state, qs8_qc4w_packw__reference,
                 /*nr=*/16, /*kr=*/8, /*sr=*/1);
}
static void qs8_qc4w_packw_x32c8__reference(benchmark::State& state,
                                            const char* net) {
  qs8_qc4w_packw(state, qs8_qc4w_packw__reference,
                 /*nr=*/32, /*kr=*/8, /*sr=*/1);
}

BENCHMARK_BGEMM(qs8_qc4w_packw_x8c8__reference)
BENCHMARK_BGEMM(qs8_qc4w_packw_x16c8__reference)
BENCHMARK_BGEMM(qs8_qc4w_packw_x32c8__reference)

static void x16_packw__reference(size_t batch, size_t dim_n, size_t dim_k,
                                 size_t nr, size_t kr, size_t sr,
                                 const uint16_t* weights, const uint16_t* bias,
                                 const void* scale, uint16_t* packed_weights,
                                 size_t extra_bytes, const void* params) {
  xnn_pack_f16_gemm_goi_w(batch, dim_n, dim_k, nr, kr, sr, weights, bias, scale,
                          packed_weights, extra_bytes, params);
}

static void x16_packw_x8__reference(benchmark::State& state, const char* net) {
  x16_packw(state, x16_packw__reference,
            /*nr=*/8, /*kr=*/1, /*sr=*/1);
}

BENCHMARK_BGEMM(x16_packw_x8__reference)

static void x32_packw__reference(size_t batch, size_t dim_n, size_t dim_k,
                                 size_t nr, size_t kr, size_t sr,
                                 const uint32_t* weights, const uint32_t* bias,
                                 const void* scale, uint32_t* packed_weights,
                                 size_t extra_bytes, const void* params) {
  xnn_pack_f32_gemm_goi_w(
      batch, dim_n, dim_k, nr, kr, sr, reinterpret_cast<const float*>(weights),
      reinterpret_cast<const float*>(bias), scale,
      reinterpret_cast<float*>(packed_weights), extra_bytes, params);
}

static void x32_packw_x2c4__reference(benchmark::State& state,
                                      const char* net) {
  x32_packw(state, x32_packw__reference,
            /*nr=*/2, /*kr=*/4, /*sr=*/1);
}
static void x32_packw_x8__reference(benchmark::State& state, const char* net) {
  x32_packw(state, x32_packw__reference,
            /*nr=*/8, /*kr=*/1, /*sr=*/1);
}
static void x32_packw_x8s4__reference(benchmark::State& state,
                                      const char* net) {
  x32_packw(state, x32_packw__reference,
            /*nr=*/8, /*kr=*/1, /*sr=*/4);
}
static void x32_packw_x16__reference(benchmark::State& state, const char* net) {
  x32_packw(state, x32_packw__reference,
            /*nr=*/16, /*kr=*/1, /*sr=*/1);
}
static void x32_packw_x16s4__reference(benchmark::State& state,
                                       const char* net) {
  x32_packw(state, x32_packw__reference,
            /*nr=*/16, /*kr=*/1, /*sr=*/4);
}
static void x32_packw_x32__reference(benchmark::State& state, const char* net) {
  x32_packw(state, x32_packw__reference,
            /*nr=*/32, /*kr=*/1, /*sr=*/1);
}
static void x32_packw_x64__reference(benchmark::State& state, const char* net) {
  x32_packw(state, x32_packw__reference,
            /*nr=*/64, /*kr=*/1, /*sr=*/1);
}

BENCHMARK_BGEMM(x32_packw_x2c4__reference)
BENCHMARK_BGEMM(x32_packw_x8__reference)
BENCHMARK_BGEMM(x32_packw_x8s4__reference)
BENCHMARK_BGEMM(x32_packw_x16__reference)
BENCHMARK_BGEMM(x32_packw_x16s4__reference)
BENCHMARK_BGEMM(x32_packw_x32__reference)
BENCHMARK_BGEMM(x32_packw_x64__reference)

static void x32_packw_gio__reference(size_t batch, size_t dim_n, size_t dim_k,
                                     size_t nr, size_t kr, size_t sr,
                                     const uint32_t* weights,
                                     const uint32_t* bias, const void* scale,
                                     uint32_t* packed_weights,
                                     size_t extra_bytes, const void* params) {
  xnn_pack_f32_gemm_gio_w(batch, dim_n, dim_k, nr, kr, sr, dim_n,
                          reinterpret_cast<const float*>(weights),
                          reinterpret_cast<const float*>(bias), scale,
                          reinterpret_cast<float*>(packed_weights), extra_bytes,
                          params);
}

static void x32_packw_x8_gio__reference(benchmark::State& state,
                                        const char* net) {
  x32_packw(state, x32_packw_gio__reference,
            /*nr=*/8, /*kr=*/1, /*sr=*/1);
}
static void x32_packw_x16_gio__reference(benchmark::State& state,
                                         const char* net) {
  x32_packw(state, x32_packw_gio__reference,
            /*nr=*/16, /*kr=*/1, /*sr=*/1);
}
static void x32_packw_x32_gio__reference(benchmark::State& state,
                                         const char* net) {
  x32_packw(state, x32_packw_gio__reference,
            /*nr=*/32, /*kr=*/1, /*sr=*/1);
}
static void x32_packw_x64_gio__reference(benchmark::State& state,
                                         const char* net) {
  x32_packw(state, x32_packw_gio__reference,
            /*nr=*/64, /*kr=*/1, /*sr=*/1);
}

BENCHMARK_BGEMM(x32_packw_x8_gio__reference)
BENCHMARK_BGEMM(x32_packw_x16_gio__reference)
BENCHMARK_BGEMM(x32_packw_x32_gio__reference)
BENCHMARK_BGEMM(x32_packw_x64_gio__reference)

static void qb4_packw_goi__reference(size_t batch, size_t dim_n, size_t dim_k,
                                     size_t nr, size_t kr, size_t sr, size_t bl,
                                     const uint32_t* weights,
                                     const uint32_t* bias, const void* scale,
                                     uint32_t* packed_weights,
                                     size_t extra_bytes_bl,
                                     size_t extra_bytes_n, const void* params) {
  xnn_pack_qs8_qb4w_gemm_goi_w(
      batch, dim_n, dim_k, nr, kr, sr, bl,
      reinterpret_cast<const uint8_t*>(weights),
      reinterpret_cast<const float*>(bias),
      reinterpret_cast<const xnn_bfloat16*>(scale), packed_weights,
      extra_bytes_bl, extra_bytes_n,
      reinterpret_cast<const struct xnn_qs8_qc4w_packing_params*>(params));
}

static void qb4_packw_x16c4_goi__reference(benchmark::State& state,
                                           const char* net) {
  qb4_packw(state, (xnn_qb4_packw_gemm_goi_ukernel_fn)qb4_packw_goi__reference,
            /*nr=*/16, /*kr=*/4, /*sr=*/1, /*bl=*/32, true);
}

static void qb4_packw_x16c8_goi__reference(benchmark::State& state,
                                           const char* net) {
  qb4_packw(state, (xnn_qb4_packw_gemm_goi_ukernel_fn)qb4_packw_goi__reference,
            /*nr=*/16, /*kr=*/8, /*sr=*/1, /*bl=*/32, true);
}

BENCHMARK_BGEMM(qb4_packw_x16c4_goi__reference)
BENCHMARK_BGEMM(qb4_packw_x16c8_goi__reference)
