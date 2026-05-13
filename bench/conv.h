// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// clang-format off

#ifndef XNNPACK_BENCH_CONV_H_
#define XNNPACK_BENCH_CONV_H_

#include <cstddef>

#include <benchmark/benchmark.h>
#include "bench/utils.h"

inline void CmdlineConvArguments(benchmark::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  for (size_t i = 0; i + 9 < benchmark::utils::FLAGS_shapes.size(); i += 10) {
    b->Args({benchmark::utils::FLAGS_shapes[i],
             benchmark::utils::FLAGS_shapes[i + 1],
             benchmark::utils::FLAGS_shapes[i + 2],
             benchmark::utils::FLAGS_shapes[i + 3],
             benchmark::utils::FLAGS_shapes[i + 4],
             benchmark::utils::FLAGS_shapes[i + 5],
             benchmark::utils::FLAGS_shapes[i + 6],
             benchmark::utils::FLAGS_shapes[i + 7],
             benchmark::utils::FLAGS_shapes[i + 8],
             benchmark::utils::FLAGS_shapes[i + 9]});
  }
}

#define BENCHMARK_CONV(conv_fn) \
  BENCHMARK(conv_fn)->Apply([](benchmark::Benchmark* b) { \
    benchmark::utils::DeferArgs(b, CmdlineConvArguments); \
  })->UseRealTime();

#endif  // XNNPACK_BENCH_CONV_H_
