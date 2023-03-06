// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <benchmark/benchmark.h>

#define BENCHMARK_BGEMM(bgemm_fn) \
  BENCHMARK_CAPTURE(bgemm_fn, albert, "Albert")->Apply(AlbertBgemmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(bgemm_fn, mobilebert, "MobileBert")->Apply(MobilebertBgemmArguments)->UseRealTime();


static void AlbertBgemmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"B", "M", "N", "K"});

  /*        B   M    N    K  */
  b->Args({12, 384,  64, 384});
  b->Args({12, 384, 384,  64});
}

static void MobilebertBgemmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"B", "M", "N", "K"});

  /*       B   M    N    K  */
  b->Args({4, 384,  32, 384});
  b->Args({4, 384, 384,  32});
}
