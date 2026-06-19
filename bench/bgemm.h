// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// clang-format off

#ifndef XNNPACK_BENCH_BGEMM_H_
#define XNNPACK_BENCH_BGEMM_H_

#include <cstddef>

#include <benchmark/benchmark.h>
#include "bench/utils.h"

inline void CmdlineBgemmArguments(benchmark::Benchmark* b) {
  b->ArgNames({"B", "M", "N", "K"});

  for (size_t i = 0; i + 3 < benchmark::utils::FLAGS_shapes.size(); i += 4) {
    b->Args({benchmark::utils::FLAGS_shapes[i],
             benchmark::utils::FLAGS_shapes[i + 1],
             benchmark::utils::FLAGS_shapes[i + 2],
             benchmark::utils::FLAGS_shapes[i + 3]});
  }
}

#define BENCHMARK_BGEMM(bgemm_fn) \
  BENCHMARK(bgemm_fn)->Apply([](benchmark::Benchmark* b) { \
    benchmark::utils::DeferArgs(b, CmdlineBgemmArguments); \
  })->UseRealTime();

#define BENCHMARK_CAPTURE_BGEMM(bgemm_fn, name_prefix, ...)      \
  BENCHMARK_CAPTURE(bgemm_fn, name_prefix##cmdline, __VA_ARGS__) \
      ->Apply([](benchmark::Benchmark* b) {                      \
    benchmark::utils::DeferArgs(b, CmdlineBgemmArguments);       \
  })->UseRealTime();

#endif  // XNNPACK_BENCH_BGEMM_H_
