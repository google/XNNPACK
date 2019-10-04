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

#include "models/models.h"


static void MobileNetV1(benchmark::State& state) {
  if (xnn_initialize() != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  auto execution_plan = models::MobileNetV1(nullptr);
  if (execution_plan.empty()) {
    state.SkipWithError("failed to create MobileNet v1");
    return;
  }

  for (auto _ : state) {
    for (const std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>& op : execution_plan) {
      xnn_status status = xnn_run_operator(op.get(), nullptr);
      if (status != xnn_status_success) {
        state.SkipWithError("failed to run MobileNet v1");
        return;
      }
    }
  }
}

static void MobileNetV2(benchmark::State& state) {
  if (xnn_initialize() != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  auto execution_plan = models::MobileNetV2(nullptr);
  if (execution_plan.empty()) {
    state.SkipWithError("failed to create MobileNet v2");
    return;
  }

  for (auto _ : state) {
    for (const std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>& op : execution_plan) {
      xnn_status status = xnn_run_operator(op.get(), nullptr);
      if (status != xnn_status_success) {
        state.SkipWithError("failed to run MobileNet v2");
        return;
      }
    }
  }
}

BENCHMARK(MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif