// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <benchmark/benchmark.h>

#define BENCHMARK_BGEMM(bgemm_fn) \
  BENCHMARK_CAPTURE(bgemm_fn, albert, "Albert")->Apply(AlbertBgemmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(bgemm_fn, mobilebert, "MobileBert")->Apply(MobilebertBgemmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(bgemm_fn, sd1x_diffusion, "SD1.X Diffusion")->Apply(SD1XDiffusionBgemmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(bgemm_fn, sd1x_encoder_decoder, "SD1.X Encoder-Decoder")->Apply(SD1XEncoderDecoderBgemmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(bgemm_fn, sd1x_text_encoder, "SD1.X Text Encoder")->Apply(SD1XTextEncoderBgemmArguments)->UseRealTime();


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

static void SD1XDiffusionBgemmArguments(benchmark::internal::Benchmark* b) {
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

static void SD1XEncoderDecoderBgemmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"B", "M", "N", "K"});

  /*       B    M     N     K */
  b->Args({1, 4096, 4096,  512});
  b->Args({1,  512, 4096, 4096});
}

static void SD1XTextEncoderBgemmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"B", "M", "N", "K"});

  /*       B   M    N   K */
  b->Args({12, 77, 77, 64});
  b->Args({12, 77, 64, 77});
}
