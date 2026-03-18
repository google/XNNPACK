// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// clang-format off

#ifndef XNNPACK_BENCH_BGEMM_H_
#define XNNPACK_BENCH_BGEMM_H_

#include <benchmark/benchmark.h>
#include "bench/utils.h"

#define BENCHMARK_BGEMM(bgemm_fn) \
  BENCHMARK_NAMED(bgemm_fn, attention)->Apply(QD8AttentionBgemmArguments)->UseRealTime(); \
  BENCHMARK_NAMED(bgemm_fn, albert)->Apply(AlbertBgemmArguments)->UseRealTime(); \
  BENCHMARK_NAMED(bgemm_fn, mobilebert)->Apply(MobilebertBgemmArguments)->UseRealTime(); \
  BENCHMARK_NAMED(bgemm_fn, sd1x_diffusion)->Apply(SD1XDiffusionBgemmArguments)->UseRealTime(); \
  BENCHMARK_NAMED(bgemm_fn, sd1x_encoder_decoder)->Apply(SD1XEncoderDecoderBgemmArguments)->UseRealTime(); \
  BENCHMARK_NAMED(bgemm_fn, sd1x_text_encoder)->Apply(SD1XTextEncoderBgemmArguments)->UseRealTime(); \
  BENCHMARK_NAMED(bgemm_fn, llm)->Apply(LLMGemmArguments)->UseRealTime();

#define BENCHMARK_CAPTURE_BGEMM(bgemm_fn, name_prefix, ...) \
  BENCHMARK_CAPTURE(bgemm_fn, name_prefix##attention, __VA_ARGS__)->Apply(QD8AttentionBgemmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(bgemm_fn, name_prefix##albert, __VA_ARGS__)->Apply(AlbertBgemmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(bgemm_fn, name_prefix##mobilebert, __VA_ARGS__)->Apply(MobilebertBgemmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(bgemm_fn, name_prefix##sd1x_diffusion, __VA_ARGS__)->Apply(SD1XDiffusionBgemmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(bgemm_fn, name_prefix##sd1x_encoder_decoder, __VA_ARGS__)->Apply(SD1XEncoderDecoderBgemmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(bgemm_fn, name_prefix##sd1x_text_encoder, __VA_ARGS__)->Apply(SD1XTextEncoderBgemmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(bgemm_fn, name_prefix##llm, __VA_ARGS__)->Apply(LLMGemmArguments)->UseRealTime();


inline void AlbertBgemmArguments(benchmark::Benchmark* b) {
  b->ArgNames({"B", "M", "N", "K"});

  /*        B   M    N    K  */
  b->Args({12, 384,  64, 384});
  b->Args({12, 384, 384,  64});
}

inline void MobilebertBgemmArguments(benchmark::Benchmark* b) {
  b->ArgNames({"B", "M", "N", "K"});

  /*       B   M    N    K  */
  b->Args({4, 384,  32, 384});
  b->Args({4, 384, 384,  32});
}

inline void SD1XDiffusionBgemmArguments(benchmark::Benchmark* b) {
  b->ArgNames({"B", "M", "N", "K"});

  /*       B    M     N     K */
  b->Args({8, 4096, 4096,   40});
  b->Args({8, 4096,   40, 4096});
  b->Args({8, 4096,   77,   40});
  b->Args({8, 4096,   40,   77});
  b->Args({8, 1024,  1024,  80});
  b->Args({8, 1024,   80, 1024});
  b->Args({8, 1024,   77,   80});
  b->Args({8, 1024,   80,   77});
  b->Args({8,  256,  256,  160});
  b->Args({8,  256,  160,  256});
  b->Args({8,  256,   77,  160});
  b->Args({8,  256,  160,   77});
  b->Args({8,   64,   64,  160});
  b->Args({8,   64,  160,   64});
  b->Args({8,   64,   77,  160});
  b->Args({8,   64,  160,   77});
}

inline void SD1XEncoderDecoderBgemmArguments(benchmark::Benchmark* b) {
  b->ArgNames({"B", "M", "N", "K"});

  /*       B    M     N     K */
  b->Args({1, 4096, 4096,  512});
  b->Args({1,  512, 4096, 4096});
}

inline void SD1XTextEncoderBgemmArguments(benchmark::Benchmark* b) {
  b->ArgNames({"B", "M", "N", "K"});

  /*       B   M    N   K */
  b->Args({12, 77, 77, 64});
  b->Args({12, 77, 64, 77});
}

inline void QD8AttentionBgemmArguments(benchmark::Benchmark* b) {
  b->ArgNames({"B", "M", "N", "K"});

  /*       B      M    N   K */
  b->Args({36186, 1,  4, 16});
  b->Args({9,     1,  4, 1792});
  b->Args({110,   1, 16, 1024});
  b->Args({50,    1, 16, 1536});
  b->Args({7,     1, 16, 2304});
  b->Args({18,    1, 16, 1792});
  b->Args({14,    1, 16, 2048});
  b->Args({3,     1, 16, 3072});
  b->Args({55,    1, 14, 1024});
  b->Args({36,    1,  6, 1536});
  b->Args({7,     1, 10, 2304});
  b->Args({3,     1, 12, 3072});
  b->Args({4,     1,  8, 2048});
  b->Args({14,    1,  2, 1536});
  b->Args({10,    1,  2, 2048});
}

// Large Language Model (Generic)
static void LLMGemmArguments(benchmark::Benchmark* b) {
  b->ArgNames({"B", "M", "N", "K"});

  b->Args({1, 128, 16, 1024});
  b->Args({1, 128, 128, 1024});
  b->Args({1, 128, 4096, 1024});
}

#endif  // XNNPACK_BENCH_BGEMM_H_
