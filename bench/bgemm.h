// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <benchmark/benchmark.h>

#define BENCHMARK_BGEMM(bgemm_fn) \
  BENCHMARK_CAPTURE(bgemm_fn, albert, "Albert")->Apply(AlbertBgemmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(bgemm_fn, mobilebert, "MobileBert")->Apply(MobilebertBgemmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(bgemm_fn, oddalbert, "OddAlbert")->Apply(OddAlbertBgemmArguments)->UseRealTime(); \


static void AlbertBgemmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"B", "M", "N", "K"});

  /*        B   M    N    K  */
  b->Args({12, 384,  64, 384});
  b->Args({12, 384, 384,  64});
}

// Albert but with each parameter reduced by 1 to benchmark remainder handling
// This is not a real model.  OddAlbert allows benchmarking microkernel
// remainder handling.
static void OddAlbertBgemmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"B", "M", "N", "K"});

  /*        B   M    N    K  */
  b->Args({12, 384,  64, 381});  // odd K values
  b->Args({12, 384,  64, 382});
  b->Args({12, 384,  64, 383});
  b->Args({12, 384,  64, 384});
  b->Args({12, 384,  64, 385});
  b->Args({12, 384, 384,  61});
  b->Args({12, 384, 384,  62});
  b->Args({12, 384, 384,  63});
  b->Args({12, 384, 384,  64});
  b->Args({12, 384, 384,  65});
  b->Args({12, 384,  61, 384});  // odd N values
  b->Args({12, 384,  62, 384});
  b->Args({12, 384,  63, 384});
  b->Args({12, 384,  65, 384});
  b->Args({12, 384, 381,  64});
  b->Args({12, 384, 382,  64});
  b->Args({12, 384, 383,  64});
  b->Args({12, 384, 385,  64});
}

static void MobilebertBgemmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"B", "M", "N", "K"});

  /*       B   M    N    K  */
  b->Args({4, 384,  32, 384});
  b->Args({4, 384, 384,  32});
}
