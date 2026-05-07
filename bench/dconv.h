// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// clang-format off

#ifndef XNNPACK_BENCH_DCONV_H_
#define XNNPACK_BENCH_DCONV_H_

#include <cstddef>

#include "bench/utils.h"
#include <benchmark/benchmark.h>

inline void CmdlineDConvArguments(benchmark::Benchmark* b) {
  b->ArgNames({"H", "W", "Cout"});

  for (size_t i = 0; i + 2 < benchmark::utils::FLAGS_shapes.size(); i += 3) {
    b->Args({benchmark::utils::FLAGS_shapes[i],
             benchmark::utils::FLAGS_shapes[i + 1],
             benchmark::utils::FLAGS_shapes[i + 2]});
  }
}

#define BENCHMARK_DCONV(conv_fn) \
  BENCHMARK(conv_fn)->Apply([](benchmark::Benchmark* b) { \
    benchmark::utils::DeferArgs(b, CmdlineDConvArguments); \
  })->UseRealTime();

#endif  // XNNPACK_BENCH_DCONV_H_
