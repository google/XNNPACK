// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "bench/subgraph/model_runtime.h"

#include <cassert>
#include <cstdint>
#include <functional>

#include "bench/utils.h"
#include "include/xnnpack.h"
#include <benchmark/benchmark.h>

namespace xnnpack {

void ModelRuntime::BenchmarkInvoke(
    benchmark::State& state, std::function<xnn_subgraph_t()> model_factory,
    uint32_t extra_flags) {
  if (xnn_initialize(nullptr /* allocator */) != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  ModelRuntime model_runtime(FLAGS_num_threads);
  if (!model_runtime.CreateModel(model_factory)) {
    state.SkipWithError("failed to create model");
    return;
  }

  // TODO(dsharlet): We should have benchmarks of these steps too.
  if (!model_runtime.CreateRuntime(FLAGS_xnn_runtime_flags | extra_flags)) {
    state.SkipWithError("failed to create runtime");
    return;
  }

  if (!model_runtime.ReshapeRuntime()) {
    state.SkipWithError("failed to reshape runtime");
    return;
  }

  if (!model_runtime.SetupRuntime()) {
    state.SkipWithError("failed to setup runtime");
    return;
  }

  int num_iters = FLAGS_benchmark_min_iters;
  while (state.KeepRunningBatch(num_iters)) {
    for (int iter = 0; iter < num_iters; iter++) {
      benchmark::utils::WipePthreadpoolL2Caches(state,
                                                model_runtime.threadpool);
      if (!model_runtime.Invoke()) {
        state.SkipWithError("failed to invoke runtime");
        return;
      }
    }
    num_iters = 1;
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

}  // namespace xnnpack
