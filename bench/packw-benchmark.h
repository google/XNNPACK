// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "bench/bgemm.h"
#include "bench/utils.h"

#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/pack.h"
#include "xnnpack/packw.h"


static void x8_packw(benchmark::State& state,
  xnn_x8_packw_gemm_goi_ukernel_fn packw,
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
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));

  // Computer num_buffers that fit cache with source weights + packed_weights.
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(int8_t) * batch * (dim_n * dim_k + rounded_n * rounded_k + rounded_n));

  std::vector<int8_t, AlignedAllocator<int8_t, 64>> weights(num_buffers * batch * dim_n * dim_k);
  std::generate(weights.begin(), weights.end(), std::ref(i8rng));
  std::vector<int8_t, AlignedAllocator<int8_t, 64>> packed_weights(num_buffers * batch * (rounded_n * rounded_k + rounded_n * sizeof(uint32_t)));
  std::fill(packed_weights.begin(), packed_weights.end(), 0);

  size_t buffer_index = 0;
  for (auto _ : state) {
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    packw(batch, dim_n, dim_k, nr, kr, sr,
      weights.data() + buffer_index * batch * dim_n * dim_k,
      /*bias=*/nullptr, /*scale=*/nullptr,
      packed_weights.data() + buffer_index * batch * (rounded_n * rounded_k + rounded_n),
      /*extra_bytes=*/0, /*params=*/nullptr);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = batch * dim_n * dim_k;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = (elements_per_iteration + batch * (rounded_n * rounded_k + rounded_n)) * sizeof(int8_t);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

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

static void x8_packw__reference(
  size_t batch,
  size_t dim_n,
  size_t dim_k,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* weights,
  const uint32_t* bias,
  const void* scale,
  int8_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  xnn_pack_f32_qs8w_gemm_goi_w(batch, dim_n, dim_k, nr, kr, sr,
     reinterpret_cast<const int8_t*>(weights),
     reinterpret_cast<const float*>(bias),
     static_cast<const float*>(scale),
     static_cast<void*>(packed_weights),
     extra_bytes, params);
}

static void x8_packw_x2__reference(benchmark::State& state, const char* net) {
  x8_packw(state,
    x8_packw__reference,
    /*nr=*/2, /*kr=*/1, /*sr=*/1);
}
static void x8_packw_x4__reference(benchmark::State& state, const char* net) {
  x8_packw(state,
    x8_packw__reference,
    /*nr=*/4, /*kr=*/1, /*sr=*/1);
}
static void x8_packw_x8__reference(benchmark::State& state, const char* net) {
  x8_packw(state,
    x8_packw__reference,
    /*nr=*/8, /*kr=*/1, /*sr=*/1);
}
static void x8_packw_x16__reference(benchmark::State& state, const char* net) {
  x8_packw(state,
    x8_packw__reference,
    /*nr=*/16, /*kr=*/1, /*sr=*/1);
}
static void x8_packw_x32__reference(benchmark::State& state, const char* net) {
  x8_packw(state,
    x8_packw__reference,
    /*nr=*/32, /*kr=*/1, /*sr=*/1);
}

BENCHMARK_BGEMM(x8_packw_x2__reference)
BENCHMARK_BGEMM(x8_packw_x4__reference)
BENCHMARK_BGEMM(x8_packw_x8__reference)
BENCHMARK_BGEMM(x8_packw_x16__reference)
BENCHMARK_BGEMM(x8_packw_x32__reference)

static void x16_packw__reference(
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

static void x32_packw__reference(
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
