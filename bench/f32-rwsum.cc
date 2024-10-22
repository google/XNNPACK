// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: f32-rwsum
//   Generator: tools/generate-rw-benchmark.py

#include <numeric>
#include <benchmark/benchmark.h>

#include "utils.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/reduce.h"

void f32_rwsum(
  benchmark::State& state, 
  xnn_f32_rw_ukernel_fn rwsum,
  xnn_init_f32_default_params_fn init_params = nullptr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }
  const size_t rows = state.range(0);
  const size_t batch = state.range(1);

  std::vector<float, AlignedAllocator<float, 64>> input(rows * batch + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> output(rows);
  std::iota(input.begin(), input.end(), 1);

  // Prepare parameters.
  int64_t padding[2] = {0,0};
  int64_t base_dilation = 1;
  int64_t window_dilation = 1;
  int64_t window_dimensions = batch;
  int64_t window_stride = 1;
  float init_value = 0;
  xnn_f32_default_params params;
  if (init_params != nullptr) {
    init_params(&params);
  }

  for (auto _ : state) {
    for (int64_t i = 0; i < rows; ++i) {
      rwsum(batch * sizeof(float), &input[i * batch], init_value, padding, base_dilation, window_dilation,
            window_dimensions, window_stride, &output[i], init_params != nullptr ? &params : nullptr);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void BenchmarkRWSUM(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"channels","rows"});
  b->Args({1, 512});
  b->Args({1, 1024});
  b->Args({1, 8000});
  b->Args({512, 512});
  b->Args({512, 1024});
  b->Args({512, 8000});
  b->Args({1024, 64});
  b->Args({32768, 1});
  b->Args({10240, 1024});
}

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile,\
                                datatype, params_type, init_params)          \
BENCHMARK_CAPTURE(f32_rwsum, arch_flags, ukernel)                            \
  ->Apply(BenchmarkRWSUM)                                                    \
  ->UseRealTime();
#include "f32-rwsum/f32-rwsum.h"
#undef XNN_UKERNEL_WITH_PARAMS


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
