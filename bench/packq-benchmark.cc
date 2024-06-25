// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "packq-benchmark.h"

#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/pack.h"
#include "xnnpack/packq.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "bench/utils.h"
#include <benchmark/benchmark.h>

void x8_packq(benchmark::State& state, xnn_x8_packq_f32qp8_ukernel_fn packq,
              size_t mr, size_t kr, size_t sr,
              benchmark::utils::IsaCheckFunction isa_check) {
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t batch = state.range(0);
  const size_t dim_m = state.range(2);
  const size_t dim_k = state.range(3);

  const size_t rounded_n = benchmark::utils::RoundUp(dim_m, mr);
  const size_t rounded_k = benchmark::utils::RoundUp(dim_k, kr);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = [&]() {
    return std::uniform_real_distribution<float>(-10, 10)(rng);
  };

  // Compute a num_buffers that fit cache with source weights + packed_weights.
  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(),
              sizeof(int8_t) * batch *
                  (dim_m * dim_k + rounded_n * rounded_k + rounded_n));

  std::vector<float, AlignedAllocator<float, 64>> input(num_buffers * batch *
                                                        dim_m * dim_k);
  std::generate(input.begin(), input.end(), f32rng);
  const size_t packed_size =
      xnn_x8_packq_f32qp8_packed_size(batch * dim_m, dim_k, mr, kr, sr);
  std::vector<int8_t, AlignedAllocator<int8_t, 64>> packed_weights(num_buffers *
                                                                   packed_size);

  size_t buffer_index = 0;
  for (auto _ : state) {
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    packq(batch * dim_m, dim_k, mr, kr, sr,
          /*m_idx_start=*/buffer_index * dim_m,
          input.data() + buffer_index * batch * dim_m * dim_k,
          dim_k * sizeof(float), packed_weights.data());
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = batch * dim_m * dim_k;
  state.counters["elements"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * elements_per_iteration,
      benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration =
      (elements_per_iteration + batch * (rounded_n * rounded_k + rounded_n)) *
      sizeof(int8_t);
  state.counters["bytes"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * bytes_per_iteration,
      benchmark::Counter::kIsRate);
}
