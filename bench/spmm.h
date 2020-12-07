// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <benchmark/benchmark.h>

#define BENCHMARK_SPMM(spmm_fn) \
  BENCHMARK_CAPTURE(spmm_fn, mobilenet_v1, "MobileNet v1")->Apply(MobileNetV1SpmmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(spmm_fn, mobilenet_v2, "MobileNet v2")->Apply(MobileNetV2SpmmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(spmm_fn, mobilenet_v3_small, "MobileNet v3 Small")->Apply(MobileNetV3SmallSpmmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(spmm_fn, mobilenet_v3_large, "MobileNet v3 Large")->Apply(MobileNetV3LargeSpmmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(spmm_fn, shufflenet_v1_g1, "ShuffleNet v1 (1 group)")->Apply(ShuffleNetV1G1SpmmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(spmm_fn, shufflenet_v1_g2, "ShuffleNet v1 (2 groups)")->Apply(ShuffleNetV1G2SpmmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(spmm_fn, shufflenet_v1_g3, "ShuffleNet v1 (3 groups)")->Apply(ShuffleNetV1G3SpmmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(spmm_fn, shufflenet_v1_g4, "ShuffleNet v1 (4 groups)")->Apply(ShuffleNetV1G4SpmmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(spmm_fn, shufflenet_v1_g8, "ShuffleNet v1 (8 groups)")->Apply(ShuffleNetV1G8SpmmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(spmm_fn, shufflenet_v2_x05, "ShuffleNet v2 0.5X")->Apply(ShuffleNetV2X05SpmmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(spmm_fn, shufflenet_v2_x10, "ShuffleNet v2 1.0X")->Apply(ShuffleNetV2X10SpmmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(spmm_fn, shufflenet_v2_x15, "ShuffleNet v2 1.5X")->Apply(ShuffleNetV2X15SpmmArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(spmm_fn, shufflenet_v2_x20, "ShuffleNet v2 2.0X")->Apply(ShuffleNetV2X20SpmmArguments)->UseRealTime();


// ShuffleNet v1 with 1 group.
static void ShuffleNetV1G1SpmmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*          M      N    K */
  b->Args({56 * 56,  36,  24});
  b->Args({28 * 28, 120,  36});
  b->Args({28 * 28,  36, 144});
  b->Args({28 * 28, 144,  36});
  b->Args({28 * 28,  72, 144});
  b->Args({14 * 14, 144,  72});
  b->Args({14 * 14,  72, 288});
  b->Args({14 * 14, 288,  72});
  b->Args({14 * 14, 144, 288});
  b->Args({ 7 *  7, 288, 144});
  b->Args({ 7 *  7, 144, 576});
  b->Args({ 7 *  7, 576, 144});
}

// ShuffleNet v1 with 2 groups.
static void ShuffleNetV1G2SpmmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*          M      N    K */
  b->Args({56 * 56,  50,  24});
  b->Args({28 * 28,  88,  25});
  b->Args({28 * 28,  25, 100});
  b->Args({28 * 28, 100,  25});
  b->Args({28 * 28,  50, 100});
  b->Args({14 * 14, 100,  50});
  b->Args({14 * 14,  50, 200});
  b->Args({14 * 14, 200,  50});
  b->Args({14 * 14, 100, 200});
  b->Args({ 7 *  7, 200, 100});
  b->Args({ 7 *  7, 100, 400});
  b->Args({ 7 *  7, 400, 100});
}

// ShuffleNet v1 with 3 groups.
static void ShuffleNetV1G3SpmmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*          M      N    K */
  b->Args({56 * 56,  60,  24});
  b->Args({28 * 28,  72,  20});
  b->Args({28 * 28,  20,  80});
  b->Args({28 * 28,  80,  20});
  b->Args({28 * 28,  40,  80});
  b->Args({14 * 14,  80,  40});
  b->Args({14 * 14,  40, 160});
  b->Args({14 * 14, 160,  40});
  b->Args({14 * 14,  80, 160});
  b->Args({ 7 *  7, 160,  80});
  b->Args({ 7 *  7,  80, 320});
  b->Args({ 7 *  7, 320,  80});
}

// ShuffleNet v1 with 4 groups.
static void ShuffleNetV1G4SpmmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*          M      N    K */
  b->Args({56 * 56,  68,  24});
  b->Args({28 * 28,  62,  17});
  b->Args({28 * 28,  17,  68});
  b->Args({28 * 28,  68,  17});
  b->Args({28 * 28,  34,  68});
  b->Args({14 * 14,  68,  34});
  b->Args({14 * 14,  34, 136});
  b->Args({14 * 14, 136,  34});
  b->Args({14 * 14,  68, 136});
  b->Args({ 7 *  7, 136,  68});
  b->Args({ 7 *  7,  68, 272});
  b->Args({ 7 *  7, 272,  68});
}

// ShuffleNet v1 with 8 groups.
static void ShuffleNetV1G8SpmmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*          M      N    K */
  b->Args({56 * 56,  96,  24});
  b->Args({28 * 28,  45,  12});
  b->Args({28 * 28,  12,  48});
  b->Args({28 * 28,  48,  12});
  b->Args({28 * 28,  24,  48});
  b->Args({14 * 14,  48,  24});
  b->Args({14 * 14,  24,  96});
  b->Args({14 * 14,  96,  24});
  b->Args({14 * 14,  48,  96});
  b->Args({ 7 *  7,  96,  48});
  b->Args({ 7 *  7,  48, 192});
  b->Args({ 7 *  7, 192,  48});
}

// ShuffleNet v2 (0.5X scale)
static void ShuffleNetV2X05SpmmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*          M       N    K */
  b->Args({56 * 56,   24,  24});
  b->Args({28 * 28,   24,  24});
  b->Args({28 * 28,   48,  48});
  b->Args({14 * 14,   48,  48});
  b->Args({14 * 14,   96,  96});
  b->Args({ 7 *  7,   96,  96});
  b->Args({ 7 *  7, 1024, 192});
}

// ShuffleNet v2 (1.0X scale)
static void ShuffleNetV2X10SpmmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*          M       N    K */
  b->Args({56 * 56,   58,  24});
  b->Args({28 * 28,   58,  24});
  b->Args({28 * 28,   58,  58});
  b->Args({14 * 14,  116, 116});
  b->Args({14 * 14,  116, 116});
  b->Args({14 * 14,  232, 232});
  b->Args({ 7 *  7,  232, 232});
  b->Args({ 7 *  7, 1024, 464});
}

// ShuffleNet v2 (1.5X scale)
static void ShuffleNetV2X15SpmmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*          M       N    K */
  b->Args({56 * 56,   88,  24});
  b->Args({28 * 28,   88,  24});
  b->Args({28 * 28,   88,  88});
  b->Args({28 * 28,  176, 176});
  b->Args({14 * 14,  176, 176});
  b->Args({14 * 14,  352, 352});
  b->Args({ 7 *  7,  352, 352});
  b->Args({ 7 *  7, 1024, 704});
}

// ShuffleNet v2 (2.0X scale)
static void ShuffleNetV2X20SpmmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*          M       N    K */
  b->Args({56 * 56,  122,  24});
  b->Args({28 * 28,  122,  24});
  b->Args({28 * 28,  122, 122});
  b->Args({28 * 28,  244, 244});
  b->Args({14 * 14,  244, 244});
  b->Args({14 * 14,  488, 488});
  b->Args({ 7 *  7,  488, 488});
  b->Args({ 7 *  7, 2048, 976});
}

static void MobileNetV1SpmmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M        N     K */
  b->Args({112 * 112,   64,   32});
  b->Args({ 56 *  56,  128,   64});
  b->Args({ 56 *  56,  128,  128});
  b->Args({ 28 *  28,  256,  128});
  b->Args({ 28 *  28,  256,  256});
  b->Args({ 14 *  14,  512,  256});
  b->Args({ 14 *  14,  512,  512});
  b->Args({  7 *   7, 1024,  512});
  b->Args({  7 *   7, 1024, 1024});
}

static void MobileNetV2SpmmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /******** Bottleneck 1 *******/
  /*           M        N    K */
  b->Args({112 * 112,   16,  32});
  /******** Bottleneck 2 *******/
  /*           M        N    K */
  b->Args({112 * 112,   96,  16});
  b->Args({ 56 *  56,   24,  96});
  b->Args({ 56 *  56,  144,  24});
  b->Args({ 56 *  56,   24, 144});
  /******** Bottleneck 3 *******/
  /*           M        N    K */
  b->Args({ 28 *  28,   32, 144});
  b->Args({ 28 *  28,  192,  32});
  b->Args({ 28 *  28,   32, 192});
  /******** Bottleneck 4 *******/
  /*           M        N    K */
  b->Args({ 14 *  14,   64, 192});
  b->Args({ 14 *  14,  384,  64});
  b->Args({ 14 *  14,   64, 384});
  /******** Bottleneck 5 *******/
  /*           M        N    K */
  b->Args({ 14 *  14,   96, 384});
  b->Args({ 14 *  14,  576,  96});
  b->Args({ 14 *  14,   96, 576});
  /******** Bottleneck 6 *******/
  /*           M        N    K */
  b->Args({  7 *   7,  160, 576});
  b->Args({  7 *   7,  960, 160});
  b->Args({  7 *   7,  160, 960});
  /******** Bottleneck 7 *******/
  /*           M        N    K */
  b->Args({  7 *   7,  320, 960});
  /***** Pre-pooling Conv2D ****/
  /*           M        N    K */
  b->Args({  7 *   7, 1280, 320});
}

static void MobileNetV3SmallSpmmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /****** Bottleneck 1 ******/
  /*          M      N    K */
  b->Args({ 1 *  1,   8,  16});
  b->Args({ 1 *  1,  16,   8});
  b->Args({56 * 56,  16,  16});
  /****** Bottleneck 2 ******/
  /*          M      N    K */
  b->Args({56 * 56,  72,  16});
  b->Args({28 * 28,  24,  72});
  /****** Bottleneck 3 ******/
  /*          M      N    K */
  b->Args({28 * 28,  88,  24});
  b->Args({28 * 28,  24,  88});
  /****** Bottleneck 4 ******/
  /*          M      N    K */
  b->Args({28 * 28,  96,  24});
  b->Args({ 1 *  1,  24,  96});
  b->Args({ 1 *  1,  96,  24});
  b->Args({14 * 14,  40,  96});
  /****** Bottleneck 5 ******/
  /*          M      N    K */
  b->Args({14 * 14, 240,  40});
  b->Args({ 1 *  1,  64, 240});
  b->Args({ 1 *  1, 240,  64});
  b->Args({14 * 14,  40, 240});
  /****** Bottleneck 6 ******/
  /*          M      N    K */
//b->Args({14 * 14, 240,  40});
//b->Args({ 1 *  1,  64, 240});
//b->Args({ 1 *  1, 240,  64});
//b->Args({14 * 14,  40, 240});
  /****** Bottleneck 7 ******/
  /*          M      N    K */
  b->Args({14 * 14, 120,  40});
  b->Args({ 1 *  1,  32, 120});
  b->Args({ 1 *  1, 120,  32});
  b->Args({14 * 14,  48, 120});
  /****** Bottleneck 8 ******/
  /*          M      N    K */
  b->Args({14 * 14, 144,  48});
  b->Args({ 1 *  1,  40, 144});
  b->Args({ 1 *  1, 144,  40});
  b->Args({14 * 14,  48, 144});
  /****** Bottleneck 9 ******/
  /*          M      N    K */
  b->Args({14 * 14, 288,  48});
  b->Args({ 1 *  1,  72, 288});
  b->Args({ 1 *  1, 288,  72});
  b->Args({ 7 *  7,  96, 288});
  /****** Bottleneck 10 *****/
  /*          M      N     K */
  b->Args({ 7 *  7, 576,  96});
  b->Args({ 1 *  1, 144, 576});
  b->Args({ 1 *  1, 576, 144});
  b->Args({ 7 *  7,  96, 576});
  /****** Bottleneck 11 *****/
  /*          M      N    K */
//b->Args({ 7 *  7, 576,  96});
//b->Args({ 1 *  1, 144, 576});
//b->Args({ 1 *  1, 576, 144});
//b->Args({ 7 *  7,  96, 576});
  /******* Last Stage *******/
  /*          M      N    K */
//b->Args({ 7 *  7, 576,  96});
}

static void MobileNetV3LargeSpmmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /******* Bottleneck 1 *******/
  /*           M       N    K */
  b->Args({112 * 112,  16,  16});
  /******* Bottleneck 2 *******/
  /*           M       N    K */
  b->Args({112 * 112,  64,  16});
  b->Args({ 56 *  56,  24,  64});
  /******* Bottleneck 3 *******/
  /*           M       N    K */
  b->Args({ 56 *  56,  72,  24});
  b->Args({ 56 *  56,  24,  72});
  /******* Bottleneck 4 *******/
  /*           M       N    K */
//b->Args({ 56 *  56,  72,  24});
  b->Args({  1 *   1,  24,  72});
  b->Args({  1 *   1,  72,  24});
  b->Args({ 28 *  28,  40,  72});
  /******* Bottleneck 5 *******/
  /*           M       N    K */
  b->Args({ 28 *  28, 120,  40});
  b->Args({  1 *   1,  32, 120});
  b->Args({  1 *   1, 120,  32});
  b->Args({ 28 *  28,  40, 120});
  /******* Bottleneck 6 *******/
  /*           M       N    K */
//b->Args({ 28 *  28, 120,  40});
//b->Args({  1 *   1,  32, 120});
//b->Args({  1 *   1, 120,  32});
//b->Args({ 28 *  28,  40, 120});
  /******* Bottleneck 7 *******/
  /*           M       N    K */
  b->Args({ 28 *  28, 240,  40});
  b->Args({ 14 *  14,  80, 240});
  /******* Bottleneck 8 *******/
  /*           M       N    K */
  b->Args({ 14 *  14, 200,  80});
  b->Args({ 14 *  14,  80, 200});
  /******* Bottleneck 9 *******/
  /*           M       N    K */
  b->Args({ 14 *  14, 184,  80});
  b->Args({ 14 *  14,  80, 184});
  /******* Bottleneck 10 ******/
  /*           M       N    K */
  b->Args({ 14 *  14, 184,  80});
  b->Args({ 14 *  14,  80, 184});
  /******* Bottleneck 11 ******/
  /*           M       N    K */
  b->Args({ 14 *  14, 480,  80});
  b->Args({  1 *   1, 120, 480});
  b->Args({  1 *   1, 480, 120});
  b->Args({ 14 *  14, 112, 480});
  /******* Bottleneck 12 ******/
  /*           M       N    K */
  b->Args({ 14 *  14, 672, 112});
  b->Args({  1 *   1, 168, 672});
  b->Args({  1 *   1, 672, 168});
  b->Args({ 14 *  14, 112, 672});
  /******* Bottleneck 13 ******/
  /*           M       N    K */
//b->Args({ 14 *  14, 672, 112});
//b->Args({  1 *   1, 168, 672});
//b->Args({  1 *   1, 672, 168});
  b->Args({  7 *   7, 160, 672});
  /******* Bottleneck 14 ******/
  /*           M       N    K */
  b->Args({  7 *   7, 960, 160});
  b->Args({  1 *   1, 240, 960});
  b->Args({  1 *   1, 960, 240});
  b->Args({  7 *   7, 160, 960});
  /******* Bottleneck 15 ******/
  /*           M       N    K */
//b->Args({  7 *   7, 960, 160});
//b->Args({  1 *   1, 240, 960});
//b->Args({  1 *   1, 960, 240});
//b->Args({  7 *   7, 160, 960});
  /******** Last Stage  *******/
  /*           M       N    K */
//b->Args({  7 *   7, 960, 160});
}
