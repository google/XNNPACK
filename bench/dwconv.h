// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <benchmark/benchmark.h>


#define BENCHMARK_DWCONV(dwconv_fn) \
  BENCHMARK_CAPTURE(dwconv_fn, mobilenet_v1, "MobileNet v1")->Apply(MobileNetV1DWConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(dwconv_fn, mobilenet_v2, "MobileNet v2")->Apply(MobileNetV2DWConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(dwconv_fn, mobilenet_v3_small, "MobileNet v3 Small")->Apply(MobileNetV3SmallDWConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(dwconv_fn, mobilenet_v3_large, "MobileNet v3 Large")->Apply(MobileNetV3LargeDWConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(dwconv_fn, shufflenet_v1_g1, "ShuffleNet v1 (1 group)")->Apply(ShuffleNetV1G1DWConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(dwconv_fn, shufflenet_v1_g2, "ShuffleNet v1 (2 groups)")->Apply(ShuffleNetV1G2DWConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(dwconv_fn, shufflenet_v1_g3, "ShuffleNet v1 (3 groups)")->Apply(ShuffleNetV1G3DWConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(dwconv_fn, shufflenet_v1_g4, "ShuffleNet v1 (4 groups)")->Apply(ShuffleNetV1G4DWConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(dwconv_fn, shufflenet_v1_g8, "ShuffleNet v1 (8 groups)")->Apply(ShuffleNetV1G8DWConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(dwconv_fn, shufflenet_v2_x05, "ShuffleNet v2 0.5X")->Apply(ShuffleNetV2X05DWConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(dwconv_fn, shufflenet_v2_x10, "ShuffleNet v2 1.0X")->Apply(ShuffleNetV2X10DWConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(dwconv_fn, shufflenet_v2_x15, "ShuffleNet v2 1.5X")->Apply(ShuffleNetV2X15DWConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(dwconv_fn, shufflenet_v2_x20, "ShuffleNet v2 2.0X")->Apply(ShuffleNetV2X20DWConvArguments)->UseRealTime();


// ShuffleNet v1 with 1 group.
static void ShuffleNetV1G1DWConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "G"});

  /********* Stage 2: stride-2 unit *********/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({56, 56,  3,  3,  2,  2, 2, 1,  36});
  /********* Stage 2: stride-1 units ********/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({28, 28,  3,  3,  2,  2, 2, 1,  36});
  /********* Stage 3: stride-2 unit *********/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({28, 28,  3,  3,  2,  2, 2, 1,  72});
  /********* Stage 3: stride-1 units ********/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({14, 14,  3,  3,  2,  2, 2, 1,  72});
  /********* Stage 4: stride-2 unit *********/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({14, 14,  3,  3,  2,  2, 2, 1, 144});
  /********* Stage 4: stride-1 units ********/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({ 7,  7,  3,  3,  2,  2, 2, 1, 144});
}

// ShuffleNet v1 with 2 groups.
static void ShuffleNetV1G2DWConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "G"});

  /********* Stage 2: stride-2 unit *********/
  /*       H   W   KH  KW  PH  PW  S  D   G */
  b->Args({56, 56,  3,  3,  2,  2, 2, 1,  50});
  /********* Stage 2: stride-1 units ********/
  /*       H   W   KH  KW  PH  PW  S  D   G */
  b->Args({28, 28,  3,  3,  2,  2, 2, 1,  50});
  /********* Stage 3: stride-2 unit *********/
  /*       H   W   KH  KW  PH  PW  S  D   G */
  b->Args({28, 28,  3,  3,  2,  2, 2, 1, 100});
  /********* Stage 3: stride-1 units ********/
  /*       H   W   KH  KW  PH  PW  S  D   G */
  b->Args({14, 14,  3,  3,  2,  2, 2, 1, 100});
  /********* Stage 4: stride-2 unit *********/
  /*       H   W   KH  KW  PH  PW  S  D   G */
  b->Args({14, 14,  3,  3,  2,  2, 2, 1, 200});
  /********* Stage 4: stride-1 units ********/
  /*       H   W   KH  KW  PH  PW  S  D   G */
  b->Args({ 7,  7,  3,  3,  2,  2, 2, 1, 200});
}

// ShuffleNet v1 with 3 groups.
static void ShuffleNetV1G3DWConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "G"});

  /********* Stage 2: stride-2 unit **********/
  /*        H   W   KH  KW  PH  PW  S  D   G */
  b->Args({56, 56,  3,  3,  2,  2, 2, 1,  60});
  /********* Stage 2: stride-1 units *********/
  /*        H   W   KH  KW  PH  PW  S  D   G */
  b->Args({28, 28,  3,  3,  2,  2, 2, 1,  60});
  /********* Stage 3: stride-2 unit **********/
  /*        H   W   KH  KW  PH  PW  S  D   G */
  b->Args({28, 28,  3,  3,  2,  2, 2, 1, 120});
  /********* Stage 3: stride-1 units *********/
  /*        H   W   KH  KW  PH  PW  S  D   G */
  b->Args({14, 14,  3,  3,  2,  2, 2, 1, 120});
  /********* Stage 4: stride-2 unit **********/
  /*        H   W   KH  KW  PH  PW  S  D   G */
  b->Args({14, 14,  3,  3,  2,  2, 2, 1, 240});
  /********* Stage 4: stride-1 units *********/
  /*        H   W   KH  KW  PH  PW  S  D   G */
  b->Args({ 7,  7,  3,  3,  2,  2, 2, 1, 240});
}

// ShuffleNet v1 with 4 groups.
static void ShuffleNetV1G4DWConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "G"});

  /********* Stage 2: stride-2 unit *********/
  /*       H   W   KH  KW  PH  PW  S  D   G */
  b->Args({56, 56,  3,  3,  2,  2, 2, 1,  68});
  /********* Stage 2: stride-1 units ********/
  /*       H   W   KH  KW  PH  PW  S  D   G */
  b->Args({28, 28,  3,  3,  2,  2, 2, 1,  68});
  /********* Stage 3: stride-2 unit *********/
  /*       H   W   KH  KW  PH  PW  S  D   G */
  b->Args({28, 28,  3,  3,  2,  2, 2, 1, 136});
  /********* Stage 3: stride-1 units ********/
  /*       H   W   KH  KW  PH  PW  S  D   G */
  b->Args({14, 14,  3,  3,  2,  2, 2, 1, 136});
  /********* Stage 4: stride-2 unit *********/
  /*       H   W   KH  KW  PH  PW  S  D   G */
  b->Args({14, 14,  3,  3,  2,  2, 2, 1, 272});
  /********* Stage 4: stride-1 units ********/
  /*       H   W   KH  KW  PH  PW  S  D   G */
  b->Args({ 7,  7,  3,  3,  2,  2, 2, 1, 272});
}

// ShuffleNet v1 with 8 groups.
static void ShuffleNetV1G8DWConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "G"});

  /********* Stage 2: stride-2 unit *********/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({56, 56,  3,  3,  2,  2, 2, 1,  96});
  /********* Stage 2: stride-1 units ********/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({28, 28,  3,  3,  2,  2, 2, 1,  96});
  /********* Stage 3: stride-2 unit *********/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({28, 28,  3,  3,  2,  2, 2, 1, 192});
  /********* Stage 3: stride-1 units ********/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({14, 14,  3,  3,  2,  2, 2, 1, 192});
  /********* Stage 4: stride-2 unit *********/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({14, 14,  3,  3,  2,  2, 2, 1, 384});
  /********* Stage 4: stride-1 units ********/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({ 7,  7,  3,  3,  2,  2, 2, 1, 384});
}

// ShuffleNet v2 (0.5X scale)
static void ShuffleNetV2X05DWConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "G"});

  /**************** Stage 2 *****************/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({56, 56,  3,  3,  2,  2, 2, 1, 24});
  b->Args({28, 28,  3,  3,  2,  2, 1, 1, 24});
  /**************** Stage 3 *****************/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({28, 28,  3,  3,  2,  2, 2, 1, 48});
  b->Args({14, 14,  3,  3,  2,  2, 1, 1, 48});
  /**************** Stage 4 *****************/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({14, 14,  3,  3,  2,  2, 2, 1, 96});
  b->Args({ 7,  7,  3,  3,  2,  2, 1, 1, 96});
}

// ShuffleNet v2 (1.0X scale)
static void ShuffleNetV2X10DWConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "G"});

  /**************** Stage 2 *****************/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({56, 56,  3,  3,  2,  2, 2, 1,  24});
  b->Args({56, 56,  3,  3,  2,  2, 2, 1,  58});
  b->Args({28, 28,  3,  3,  2,  2, 1, 1,  58});
  /**************** Stage 3 *****************/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({28, 28,  3,  3,  2,  2, 2, 1, 116});
  b->Args({14, 14,  3,  3,  2,  2, 1, 1, 116});
  /**************** Stage 4 *****************/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({14, 14,  3,  3,  2,  2, 2, 1, 232});
  b->Args({ 7,  7,  3,  3,  2,  2, 1, 1, 232});
}

// ShuffleNet v2 (1.5X scale)
static void ShuffleNetV2X15DWConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "G"});

  /**************** Stage 2 *****************/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({56, 56,  3,  3,  2,  2, 2, 1,  24});
  b->Args({56, 56,  3,  3,  2,  2, 2, 1,  88});
  b->Args({28, 28,  3,  3,  2,  2, 1, 1,  88});
  /**************** Stage 3 *****************/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({28, 28,  3,  3,  2,  2, 2, 1, 176});
  b->Args({14, 14,  3,  3,  2,  2, 1, 1, 176});
  /**************** Stage 4 *****************/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({14, 14,  3,  3,  2,  2, 2, 1, 352});
  b->Args({ 7,  7,  3,  3,  2,  2, 1, 1, 352});
}

// ShuffleNet v2 (2.0X scale)
static void ShuffleNetV2X20DWConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "G"});

  /***************** Stage 2 ****************/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({56, 56,  3,  3,  2,  2, 2, 1,  24});
  b->Args({56, 56,  3,  3,  2,  2, 2, 1, 122});
  b->Args({28, 28,  3,  3,  2,  2, 1, 1, 122});
  /***************** Stage 3 ****************/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({28, 28,  3,  3,  2,  2, 2, 1, 244});
  b->Args({14, 14,  3,  3,  2,  2, 1, 1, 244});
  /***************** Stage 4 ****************/
  /*        H   W  KH  KW  PH  PW  S  D   G */
  b->Args({14, 14,  3,  3,  2,  2, 2, 1, 488});
  b->Args({ 7,  7,  3,  3,  2,  2, 1, 1, 488});
}

static void MobileNetV1DWConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "G"});

  /*        H    W   KH  KW  PH  PW  S  D    G */
  b->Args({112, 112,  3,  3,  2,  2, 1, 1,   32});
  b->Args({112, 112,  3,  3,  2,  2, 2, 1,   64});
  b->Args({ 56,  56,  3,  3,  2,  2, 1, 1,  128});
  b->Args({ 56,  56,  3,  3,  2,  2, 2, 1,  128});
  b->Args({ 28,  28,  3,  3,  2,  2, 1, 1,  256});
  b->Args({ 28,  28,  3,  3,  2,  2, 2, 1,  256});
  b->Args({ 14,  14,  3,  3,  2,  2, 1, 1,  512});
  b->Args({ 14,  14,  3,  3,  2,  2, 2, 1,  512});
  b->Args({  7,   7,  3,  3,  2,  2, 1, 1, 1024});
}

static void MobileNetV2DWConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "G"});

  /**************** Bottleneck 1 ***************/
  /*        H    W   KH  KW  PH  PW  S  D    G */
  b->Args({112, 112,  3,  3,  2,  2, 1, 1,  32});

  /**************** Bottleneck 2 ***************/
  /*        H    W   KH  KW  PH  PW  S  D    G */
  b->Args({112, 112,  3,  3,  2,  2, 2, 1,  96});
  b->Args({ 56,  56,  3,  3,  2,  2, 1, 1, 144});

  /**************** Bottleneck 3 ***************/
  /*        H    W   KH  KW  PH  PW  S  D    G */
  b->Args({ 56,  56,  3,  3,  2,  2, 2, 1, 144});
  b->Args({ 28,  28,  3,  3,  2,  2, 1, 1, 192});
//b->Args({ 28,  28,  3,  3,  2,  2, 1, 1, 192});

  /**************** Bottleneck 4 ***************/
  /*        H    W   KH  KW  PH  PW  S  D    G */
  b->Args({ 28,  28,  3,  3,  2,  2, 2, 1, 192});
  b->Args({ 14,  14,  3,  3,  2,  2, 1, 1, 384});
//b->Args({ 14,  14,  3,  3,  2,  2, 1, 1, 384});
//b->Args({ 14,  14,  3,  3,  2,  2, 1, 1, 384});

  /**************** Bottleneck 5 ***************/
  /*        H    W   KH  KW  PH  PW  S  D    G */
//b->Args({ 14,  14,  3,  3,  2,  2, 1, 1, 384});
  b->Args({ 14,  14,  3,  3,  2,  2, 1, 1, 576});
//b->Args({ 14,  14,  3,  3,  2,  2, 1, 1, 576});

  /**************** Bottleneck 6 ***************/
  /*        H    W   KH  KW  PH  PW  S  D    G */
  b->Args({ 14,  14,  3,  3,  2,  2, 2, 1, 576});
  b->Args({  7,   7,  3,  3,  2,  2, 1, 1, 960});
//b->Args({  7,   7,  3,  3,  2,  2, 1, 1, 960});

  /**************** Bottleneck 7 ***************/
  /*        H    W   KH  KW  PH  PW  S  D    G */
//b->Args({  7,   7,  3,  3,  2,  2, 1, 1, 960});
}

static void MobileNetV3SmallDWConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "G"});

  /*************** Bottleneck 1 ***************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
  b->Args({112, 112,  3,  3,  2,  2, 2, 1,  16});
  /*************** Bottleneck 2 ***************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
  b->Args({ 56,  56,  3,  3,  2,  2, 2, 1,  72});
  /*************** Bottleneck 3 ***************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
  b->Args({ 28,  28,  3,  3,  2,  2, 1, 1,  88});
  /*************** Bottleneck 4 ***************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
  b->Args({ 28,  28,  5,  5,  4,  4, 2, 1,  96});
  /*************** Bottleneck 5 ***************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
  b->Args({ 14,  14,  5,  5,  4,  4, 1, 1, 240});
  /*************** Bottleneck 6 ***************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
//b->Args({ 14,  14,  5,  5,  4,  4, 1, 1, 240});
  /*************** Bottleneck 7 ***************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
  b->Args({ 14,  14,  5,  5,  4,  4, 1, 1, 120});
  /*************** Bottleneck 8 ***************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
  b->Args({ 14,  14,  5,  5,  4,  4, 1, 1, 144});
  /*************** Bottleneck 9 ***************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
  b->Args({ 14,  14,  5,  5,  4,  4, 2, 1, 288});
  /*************** Bottleneck 10 **************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
  b->Args({  7,   7,  5,  5,  4,  4, 1, 1, 576});
  /*************** Bottleneck 11 **************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
//b->Args({  7,   7,  5,  5,  4,  4, 1, 1, 576});
}

static void MobileNetV3LargeDWConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "G"});

  /*************** Bottleneck 1 ***************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
  b->Args({112, 112,  3,  3,  2,  2, 1, 1,  16});
  /*************** Bottleneck 2 ***************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
  b->Args({112, 112,  3,  3,  2,  2, 2, 1,  64});
  /*************** Bottleneck 3 ***************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
  b->Args({ 56,  56,  3,  3,  2,  2, 1, 1,  72});
  /*************** Bottleneck 4 ***************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
  b->Args({ 56,  56,  5,  5,  4,  4, 2, 1,  72});
  /*************** Bottleneck 5 ***************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
  b->Args({ 28,  28,  5,  5,  4,  4, 1, 1, 120});
  /*************** Bottleneck 6 ***************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
//b->Args({ 28,  28,  5,  5,  4,  4, 1, 1, 120});
  /*************** Bottleneck 7 ***************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
  b->Args({ 28,  28,  3,  3,  2,  2, 2, 1, 240});
  /*************** Bottleneck 8 ***************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
  b->Args({ 14,  14,  3,  3,  2,  2, 1, 1, 200});
  /*************** Bottleneck 9 ***************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
  b->Args({ 14,  14,  3,  3,  2,  2, 1, 1, 184});
  /*************** Bottleneck 10 **************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
//b->Args({ 14,  14,  3,  3,  2,  2, 1, 1, 184});
  /*************** Bottleneck 11 **************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
  b->Args({ 14,  14,  3,  3,  2,  2, 1, 1, 480});
  /*************** Bottleneck 12 **************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
  b->Args({ 14,  14,  3,  3,  2,  2, 1, 1, 672});
  /*************** Bottleneck 13 **************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
  b->Args({ 14,  14,  5,  5,  4,  4, 2, 1, 672});
  /*************** Bottleneck 14 **************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
  b->Args({  7,   7,  5,  5,  4,  4, 1, 1, 960});
  /*************** Bottleneck 15 **************/
  /*        H    W   KH  KW  PH  PW  S  D   G */
//b->Args({  7,   7,  5,  5,  4,  4, 1, 1, 960});
}
