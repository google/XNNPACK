// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: f32-rwdsum
//   Generator: tools/generate-rwd-benchmark.py

#include <numeric>
#include <benchmark/benchmark.h>

#include "utils.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/reduce.h"

void f32_rwdsum(
  benchmark::State& state, 
  xnn_f32_rwd_ukernel_fn rwdsum,
  xnn_init_f32_default_params_fn init_params = nullptr,
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
  int64_t padding[2] = {0,0};
  int64_t base_dilation = 1;
  int64_t window_dilation = 1;
  int64_t window_dimensions = rows;
  int64_t window_stride = 1;
  float init_value = 0;
  struct xnn_f32_default_params params;
  if (init_params != nullptr) {
    init_params(&params);
  }

  for (auto _ : state) {
    rwdsum(rows, channels, input.data(), init_value, padding, base_dilation, window_dilation,
            window_dimensions, window_stride, output.data(), init_params != nullptr ? &params : nullptr);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void BenchmarkRWDSUM(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"rows", "channels"});
  b->Args({8, 1024});
  b->Args({16, 1024});
  b->Args({10240, 1024});
}

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile,\
                                datatype, params_type, init_params)          \
BENCHMARK_CAPTURE(f32_rwdsum, arch_flags, ukernel)                           \
  ->Apply(BenchmarkRWDSUM)                                                   \
  ->UseRealTime();
#include "f32-rwdsum/f32-rwdsum.h"
#undef XNN_UKERNEL_WITH_PARAMS


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
