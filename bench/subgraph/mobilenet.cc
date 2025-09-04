// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <functional>

#include "bench/subgraph/benchmark.h"
#include "bench/subgraph/models.h"
#include "include/xnnpack.h"
#include <benchmark/benchmark.h>

static void FP32MobileNetV1(benchmark::State& state) {
  xnnpack::RunBenchmark(state, models::FP32MobileNetV1);
}

static void FP32MobileNetV2(benchmark::State& state) {
  xnnpack::RunBenchmark(state, models::FP32MobileNetV2);
}

static void FP32MobileNetV3Large(benchmark::State& state) {
  xnnpack::RunBenchmark(state, models::FP32MobileNetV3Large);
}

static void FP32MobileNetV3Small(benchmark::State& state) {
  xnnpack::RunBenchmark(state, models::FP32MobileNetV3Small);
}

static void FP16MobileNetV1(benchmark::State& state) {
  xnnpack::RunBenchmark(state, models::FP32MobileNetV1,
                        XNN_FLAG_FORCE_FP16_INFERENCE);
}

static void FP16MobileNetV2(benchmark::State& state) {
  xnnpack::RunBenchmark(state, models::FP32MobileNetV2,
                        XNN_FLAG_FORCE_FP16_INFERENCE);
}

static void FP16MobileNetV3Large(benchmark::State& state) {
  xnnpack::RunBenchmark(state, models::FP32MobileNetV3Large,
                        XNN_FLAG_FORCE_FP16_INFERENCE);
}

static void FP16MobileNetV3Small(benchmark::State& state) {
  xnnpack::RunBenchmark(state, models::FP32MobileNetV3Small,
                        XNN_FLAG_FORCE_FP16_INFERENCE);
}

static void QS8MobileNetV2(benchmark::State& state) {
  xnnpack::RunBenchmark(state, models::QS8MobileNetV2);
}

BENCHMARK(FP32MobileNetV1)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK(FP32MobileNetV2)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK(FP32MobileNetV3Large)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK(FP32MobileNetV3Small)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

BENCHMARK(FP16MobileNetV1)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK(FP16MobileNetV2)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK(FP16MobileNetV3Large)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK(FP16MobileNetV3Small)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

BENCHMARK(QS8MobileNetV2)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
