// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include "bench/rsum-benchmark.h"
#include "bench/utils.h"
#include <benchmark/benchmark.h>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/reduce.h>
#include <xnnpack/microfnptr.h>

namespace {
void f32_rsum_discontig(
    benchmark::State& state,
    xnn_f32_rdsum_ukernel_fn rdsum,
    xnn_init_f32_scale_params_fn init_params,
    benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }
  const size_t rows = state.range(0);
  const size_t channels = state.range(1);

  std::vector<float, AlignedAllocator<float, 64>> input(rows * channels + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> output(channels);
  std::vector<float> zero(channels + XNN_EXTRA_BYTES / sizeof(float), 0.f);
  std::iota(input.begin(), input.end(), 0.0f);

  // Prepare parameters.
  union xnn_f32_scale_params params;
  init_params(&params, /*scale=*/1.0f / rows);

  for (auto _ : state) {
    rdsum(rows, channels, input.data(), rows * sizeof(float), zero.data(), output.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

void f16_f32acc_rsum_discontig(
    benchmark::State& state,
    xnn_f16_f32acc_rdsum_ukernel_fn rdsum,
    xnn_init_f16_f32acc_scale_params_fn init_params,
    benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }
  const size_t rows = state.range(0);
  const size_t channels = state.range(1);

  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> input(rows * channels + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::vector<uint16_t> output(channels);
  std::vector<uint16_t> zero(channels + XNN_EXTRA_BYTES / sizeof(uint16_t), 0);
  std::iota(input.begin(), input.end(), 0.0f);

  // Prepare parameters.
  union xnn_f16_f32acc_scale_params params;
  init_params(&params, /*scale=*/1.0f / rows);

  for (auto _ : state) {
    rdsum(rows, channels, input.data(), rows * sizeof(uint16_t), zero.data(), output.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void BenchmarkBatch(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"rows", "channels"});
  b->Args({8, 1024});
  b->Args({16, 1024});
  b->Args({1024, 1024});
}

}  // namespace
