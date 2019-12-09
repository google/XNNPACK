// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <xnnpack.h>

#include <benchmark/benchmark.h>

#include "bench/utils.h"
#include "models/models.h"


static void End2EndBenchmark(
  benchmark::State& state,
  models::ExecutionPlanFactory model_factory)
{
  if (xnn_initialize(nullptr /* allocator */) != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  const size_t num_threads = state.range(0);
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool(
    pthreadpool_create(num_threads), pthreadpool_destroy);

  auto execution_plan = model_factory(threadpool.get());
  if (execution_plan.empty()) {
    state.SkipWithError("failed to create a model");
    return;
  }

  for (auto _ : state) {
    for (const std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>& op : execution_plan) {
      xnn_status status = xnn_run_operator(op.get(), threadpool.get());
      if (status != xnn_status_success) {
        state.SkipWithError("failed to run a model");
        return;
      }
    }
  }
  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
}

static void MobileNetV1(benchmark::State& state) {
  End2EndBenchmark(state, models::MobileNetV1);
}

static void MobileNetV2(benchmark::State& state) {
  End2EndBenchmark(state, models::MobileNetV2);
}

static void MobileNetV3Large(benchmark::State& state) {
  End2EndBenchmark(state, models::MobileNetV3Large);
}

static void MobileNetV3Small(benchmark::State& state) {
  End2EndBenchmark(state, models::MobileNetV3Small);
}

BENCHMARK(MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(MobileNetV2)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(MobileNetV3Large)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(MobileNetV3Small)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif