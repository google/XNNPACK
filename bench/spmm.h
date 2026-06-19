// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// clang-format off

#ifndef XNNPACK_BENCH_SPMM_H_
#define XNNPACK_BENCH_SPMM_H_

#include <cstddef>

#include <benchmark/benchmark.h>
#include "bench/utils.h"

inline void CmdlineSpmmArguments(benchmark::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  for (size_t i = 0; i + 2 < benchmark::utils::FLAGS_shapes.size(); i += 3) {
    b->Args({benchmark::utils::FLAGS_shapes[i],
             benchmark::utils::FLAGS_shapes[i + 1],
             benchmark::utils::FLAGS_shapes[i + 2]});
  }
}

#define BENCHMARK_SPMM(spmm_fn) \
  BENCHMARK(spmm_fn)->Apply([](benchmark::Benchmark* b) { \
    benchmark::utils::DeferArgs(b, CmdlineSpmmArguments); \
  })->UseRealTime();

#endif  // XNNPACK_BENCH_SPMM_H_
