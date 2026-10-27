// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "bench/subgraph/benchmark.h"
#include "bench/subgraph/models.h"
#include "bench/utils.h"
#include <benchmark/benchmark.h>

static void FP32ResNet50(benchmark::State& state) {
  xnnpack::RunBenchmark(
      state, []() { return models::FP32ResNet50(FLAGS_batch_size); });
}

static void FP16ResNet50(benchmark::State& state) {
  xnnpack::RunBenchmark(
      state, []() { return models::FP16ResNet50(FLAGS_batch_size); });
}

static void QD8ResNet50(benchmark::State& state) {
  xnnpack::RunBenchmark(state,
                        []() { return models::QD8ResNet50(FLAGS_batch_size); });
}

static void QS8ResNet50(benchmark::State& state) {
  xnnpack::RunBenchmark(state,
                        []() { return models::QS8ResNet50(FLAGS_batch_size); });
}

BENCHMARK(FP32ResNet50)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

BENCHMARK(FP16ResNet50)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

BENCHMARK(QD8ResNet50)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

BENCHMARK(QS8ResNet50)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
