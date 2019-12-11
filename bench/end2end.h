// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "models/models.h"
#include <benchmark/benchmark.h>


#define BENCHMARK_END2END(benchmark_fn) \
  BENCHMARK_CAPTURE(benchmark_fn, mobilenet_v1, models::MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime(); \
  BENCHMARK_CAPTURE(benchmark_fn, mobilenet_v2, models::MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime(); \
  BENCHMARK_CAPTURE(benchmark_fn, mobilenet_v3_large, models::MobileNetV3Large)->Unit(benchmark::kMicrosecond)->UseRealTime(); \
  BENCHMARK_CAPTURE(benchmark_fn, mobilenet_v3_small, models::MobileNetV3Small)->Unit(benchmark::kMicrosecond)->UseRealTime();
