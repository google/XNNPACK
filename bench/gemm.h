// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_BENCH_GEMM_H_
#define XNNPACK_BENCH_GEMM_H_

#include <cstdint>
#include <cstring>
#include <vector>

#include "bench/utils.h"
#include <benchmark/benchmark.h>

inline void CmdlineGemmArguments(benchmark::Benchmark* b) {
  b->UseRealTime();
  const bool blockwise = strstr(b->GetName(), "_qb") != nullptr;

  if (blockwise) {
    b->ArgNames({"M", "N", "K", "BL"});
  } else {
    b->ArgNames({"M", "N", "K"});
  }

  int64_t last_m = -1, last_n = -1, last_k = -1;
  for (size_t i = 0; i + 2 < benchmark::utils::FLAGS_shapes.size(); i += 3) {
    const int64_t m = benchmark::utils::FLAGS_shapes[i];
    const int64_t n = benchmark::utils::FLAGS_shapes[i + 1];
    const int64_t k = benchmark::utils::FLAGS_shapes[i + 2];

    if (!blockwise) {
      if (m == last_m && n == last_n && k == last_k) {
        continue;
      }
      b->Args({m, n, k});
    } else {
      b->Args({m, n, k, benchmark::utils::FLAGS_gemm_block_size});
    }

    last_m = m;
    last_n = n;
    last_k = k;
  }
}

#define BENCHMARK_GEMM(gemm_fn)                           \
  BENCHMARK(gemm_fn)->Apply([](benchmark::Benchmark* b) { \
    benchmark::utils::DeferArgs(b, CmdlineGemmArguments); \
  });

#endif  // XNNPACK_BENCH_GEMM_H_
