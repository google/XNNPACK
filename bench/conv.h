// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <benchmark/benchmark.h>


#define BENCHMARK_CONV(conv_fn) \
  BENCHMARK_CAPTURE(conv_fn, mobilenet_v1, "MobileNet v1")->Apply(MobileNetV1ConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, mobilenet_v2, "MobileNet v2")->Apply(MobileNetV2ConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, mobilenet_v3_small, "MobileNet v3 Small")->Apply(MobileNetV3SmallConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, mobilenet_v3_large, "MobileNet v3 Large")->Apply(MobileNetV3LargeConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, shufflenet_v1_g1, "ShuffleNet v1 (1 group)")->Apply(ShuffleNetV1G1ConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, shufflenet_v1_g2, "ShuffleNet v1 (2 groups)")->Apply(ShuffleNetV1G2ConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, shufflenet_v1_g3, "ShuffleNet v1 (3 groups)")->Apply(ShuffleNetV1G3ConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, shufflenet_v1_g4, "ShuffleNet v1 (4 groups)")->Apply(ShuffleNetV1G4ConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, shufflenet_v1_g8, "ShuffleNet v1 (8 groups)")->Apply(ShuffleNetV1G8ConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, shufflenet_v2_x05, "ShuffleNet v2 0.5X")->Apply(ShuffleNetV2X05ConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, shufflenet_v2_x10, "ShuffleNet v2 1.0X")->Apply(ShuffleNetV2X10ConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, shufflenet_v2_x15, "ShuffleNet v2 1.5X")->Apply(ShuffleNetV2X15ConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, shufflenet_v2_x20, "ShuffleNet v2 2.0X")->Apply(ShuffleNetV2X20ConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, inception_v3, "Inception v3")->Apply(InceptionV3ConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, resnet18, "ResNet-18")->Apply(ResNet18ConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, resnet50, "ResNet-50")->Apply(ResNet50ConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, squeezenet_v10, "SqueezeNet 1.0")->Apply(SqueezeNetV10ConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, squeezenet_v11, "SqueezeNet 1.1")->Apply(SqueezeNetV11ConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, vgg, "VGG")->Apply(VGGConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, srcnn915, "SRCNN (9-1-5)")->Apply(SRCNN915ConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, srcnn935, "SRCNN (9-3-5)")->Apply(SRCNN935ConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, srcnn955, "SRCNN (9-5-5)")->Apply(SRCNN955ConvArguments)->UseRealTime();


// ShuffleNet v1 with 1 group.
static void ShuffleNetV1G1ConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /*********************** Conv 1 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({224, 224,  3,  3,  2,  2, 2, 1,    3,   24});
  /*************** Stage 2: stride-2 unit **************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   24,   36});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   36,  120});
  /*************** Stage 2: stride-1 units *************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,  144,   36});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   36,  144});
  /*************** Stage 3: stride-2 unit **************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,  144,   72});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   72,  144});
  /*************** Stage 3: stride-1 units *************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  288,   72});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   72,  288});
  /*************** Stage 4: stride-2 unit **************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  288,  144});
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  144,  288});
  /*************** Stage 4: stride-1 units *************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  576,  144});
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  144,  576});
}

// ShuffleNet v1 with 2 groups.
static void ShuffleNetV1G2ConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /*********************** Conv 1 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({224, 224,  3,  3,  2,  2, 2, 1,    3,   24});
  /*************** Stage 2: stride-2 unit **************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   24,   50});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   25,   88});
  /*************** Stage 2: stride-1 units *************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,  100,   25});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   25,  100});
  /*************** Stage 3: stride-2 unit **************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,  100,   50});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   50,  100});
  /*************** Stage 3: stride-1 units *************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  200,   50});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   50,  200});
  /*************** Stage 4: stride-2 unit **************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  200,  100});
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  100,  200});
  /*************** Stage 4: stride-1 units *************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  400,  100});
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  100,  400});
}

// ShuffleNet v1 with 3 groups.
static void ShuffleNetV1G3ConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /*********************** Conv 1 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({224, 224,  3,  3,  2,  2, 2, 1,    3,   24});
  /*************** Stage 2: stride-2 unit **************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   24,   60});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   20,   72});
  /*************** Stage 2: stride-1 units *************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   80,   20});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   20,   80});
  /*************** Stage 3: stride-2 unit **************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   80,   40});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   40,   80});
  /*************** Stage 3: stride-1 units *************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  160,   40});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   40,  160});
  /*************** Stage 4: stride-2 unit **************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  160,   80});
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,   80,  160});
  /*************** Stage 4: stride-1 units *************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  320,   80});
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,   80,  320});
}

// ShuffleNet v1 with 4 groups.
static void ShuffleNetV1G4ConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /*********************** Conv 1 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({224, 224,  3,  3,  2,  2, 2, 1,    3,   24});
  /*************** Stage 2: stride-2 unit **************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   24,   68});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   17,   62});
  /*************** Stage 2: stride-1 units *************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   68,   17});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   17,   68});
  /*************** Stage 3: stride-2 unit **************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   68,   34});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   34,   68});
  /*************** Stage 3: stride-1 units *************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  136,   34});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   34,  136});
  /*************** Stage 4: stride-2 unit **************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  136,   68});
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,   68,  136});
  /*************** Stage 4: stride-1 units *************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  272,   68});
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,   68,  272});
}

// ShuffleNet v1 with 8 groups.
static void ShuffleNetV1G8ConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /*********************** Conv 1 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({224, 224,  3,  3,  2,  2, 2, 1,    3,   24});
  /*************** Stage 2: stride-2 unit **************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   24,   96});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   12,   45});
  /*************** Stage 2: stride-1 units *************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   48,   12});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   12,   48});
  /*************** Stage 3: stride-2 unit **************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   48,   24});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   24,   48});
  /*************** Stage 3: stride-1 units *************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   96,   24});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   24,   96});
  /*************** Stage 4: stride-2 unit **************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   96,   48});
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,   48,   96});
  /*************** Stage 4: stride-1 units *************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  192,   48});
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,   48,  192});
}

// ShuffleNet v2 (0.5X scale).
static void ShuffleNetV2X05ConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /*********************** Conv 1 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({224, 224,  3,  3,  2,  2, 2, 1,    3,   24});
  /********************** Stage 2 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   24,   24});
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   24,   24});
  /********************** Stage 3 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   48,   48});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   48,   48});
  /********************** Stage 4 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,   96,   96});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   96,   96});
  /*********************** Conv 5 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  192, 1024});
}

// ShuffleNet v2 (1.0X scale).
static void ShuffleNetV2X10ConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /*********************** Conv 1 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({224, 224,  3,  3,  2,  2, 2, 1,    3,   24});
  /********************** Stage 2 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   24,   58});
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   24,   58});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   58,   58});
  /********************** Stage 3 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  116,  116});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,  116,  116});
  /********************** Stage 4 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  232,  232});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  232,  232});
  /*********************** Conv 5 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  464, 1024});
}

// ShuffleNet v2 (1.5X scale).
static void ShuffleNetV2X15ConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /*********************** Conv 1 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({224, 224,  3,  3,  2,  2, 2, 1,    3,   24});
  /********************** Stage 2 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   24,   88});
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   24,   88});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   88,   88});
  /********************** Stage 3 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  176,  176});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,  176,  176});
  /********************** Stage 4 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  352,  352});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  352,  352});
  /*********************** Conv 5 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  704, 1024});
}

// ShuffleNet v2 (2.0X scale).
static void ShuffleNetV2X20ConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /*********************** Conv 1 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({224, 224,  3,  3,  2,  2, 2, 1,    3,   24});
  /********************** Stage 2 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   24,  122});
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   24,  122});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,  122,  122});
  /********************** Stage 3 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  244,  244});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,  244,  244});
  /********************** Stage 4 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  488,  488});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  488,  488});
  /*********************** Conv 5 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  976, 2048});
}

static void MobileNetV1ConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({224, 224,  3,  3,  2,  2, 2, 1,    3,   32});
  b->Args({112, 112,  1,  1,  0,  0, 1, 1,   32,   64});
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   64,  128});
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,  128,  128});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,  128,  256});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,  256,  256});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  256,  512});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  512,  512});
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  512, 1024});
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1, 1024, 1024});
}

static void MobileNetV2ConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({224, 224,  3,  3,  2,  2, 2, 1,    3,   32});

  /******************** Bottleneck 1 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({112, 112,  1,  1,  0,  0, 1, 1,   32,   16});

  /******************** Bottleneck 2 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({112, 112,  1,  1,  0,  0, 1, 1,   16,   96});
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   96,   24});
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   24,  144});
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,  144,   24});

  /******************** Bottleneck 3 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
//b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   24,  144});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,  144,   32});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   32,  192});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,  192,   32});
//b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   32,  192});
//b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,  192,   32});

  /******************** Bottleneck 4 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
//b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   32,  192});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  192,   64});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   64,  384});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  384,   64});
//b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   64,  384});
//b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  384,   64});
//b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   64,  384});
//b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  384,   64});

  /******************** Bottleneck 5 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
//b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   64,  384});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  384,   96});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   96,  576});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  576,   96});
//b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   96,  576});
//b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  576,   96});

  /******************** Bottleneck 6 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
//b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   96,  576});
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  576,  160});
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  160,  960});
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  960,  160});
//b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  160,  960});
//b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  960,  160});

  /******************** Bottleneck 7 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
//b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  160,  960});
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  960,  320});

  /**************** Pre-pooling Conv2D *****************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  320, 1280});
  /**************** Post-pooling Conv2D ****************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1, 1280, 1000});
}

static void MobileNetV3SmallConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /******************* Initial Stage *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({224, 224,  3,  3,  2,  2, 2, 1,    3,   16});
  /******************** Bottleneck 1 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,   16,    8});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,    8,   16});
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   16,   16});
  /******************** Bottleneck 2 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   16,   72});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   72,   24});
  /******************** Bottleneck 3 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   24,   88});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   88,   24});
  /******************** Bottleneck 4 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   24,   96});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,   96,   24});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,   24,   96});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   96,   40});
  /******************** Bottleneck 5 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   40,  240});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,  240,   64});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,   64,  240});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  240,   40});
  /******************** Bottleneck 6 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
//b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   40,  240});
//b->Args({  1,   1,  1,  1,  0,  0, 1, 1,  240,   64});
//b->Args({  1,   1,  1,  1,  0,  0, 1, 1,   64,  240});
//b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  240,   40});
  /******************** Bottleneck 7 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   40,  120});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,  120,   32});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,   32,  120});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  120,   48});
  /******************** Bottleneck 8 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   48,  144});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,  144,   40});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,   40,  144});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  144,   48});
  /******************** Bottleneck 9 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   48,  288});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,  288,   72});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,   72,  288});
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  288,   96});
  /******************* Bottleneck 10 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,   96,  576});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,  576,  144});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,  144,  576});
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  576,   96});
  /******************* Bottleneck 11 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
//b->Args({  7,   7,  1,  1,  0,  0, 1, 1,   96,  576});
//b->Args({  1,   1,  1,  1,  0,  0, 1, 1,  576,  144});
//b->Args({  1,   1,  1,  1,  0,  0, 1, 1,  144,  576});
//b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  576,   96});
  /********************* Last Stage ********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
//b->Args({  7,   7,  1,  1,  0,  0, 1, 1,   96,  576});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,  576, 1024});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1, 1024, 1001});
}

static void MobileNetV3LargeConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /******************* Initial Stage *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({224, 224,  3,  3,  2,  2, 2, 1,    3,   16});
  /******************** Bottleneck 1 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({112, 112,  1,  1,  0,  0, 1, 1,   16,   16});
  /******************** Bottleneck 2 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({112, 112,  1,  1,  0,  0, 1, 1,   16,   64});
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   64,   24});
  /******************** Bottleneck 3 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   24,   72});
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   72,   24});
  /******************** Bottleneck 4 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
//b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   24,   72});*/
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,   72,   24});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,   24,   72});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   72,   40});
  /******************** Bottleneck 5 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   40,  120});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,  120,   32});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,   32,  120});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,  120,   40});
  /******************** Bottleneck 6 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
//b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   40,  120});
//b->Args({  1,   1,  1,  1,  0,  0, 1, 1,  120,   32});
//b->Args({  1,   1,  1,  1,  0,  0, 1, 1,   32,  120});
//b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,  120,   40});
  /******************** Bottleneck 7 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,   40,  240});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  240,   80});
  /******************** Bottleneck 8 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   80,  200});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  200,   80});
  /******************** Bottleneck 9 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   80,  184});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  184,   80});
  /******************* Bottleneck 10 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
//b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   80,  184});
//b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  184,   80});
  /******************* Bottleneck 11 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,   80,  480});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,  480,  120});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,  120,  480});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  480,  112});
  /******************* Bottleneck 12 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  112,  672});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,  672,  168});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,  168,  672});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  672,  112});
  /******************* Bottleneck 13 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
//b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  112,  672});
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  672,  160});
  /******************* Bottleneck 14 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  160,  960});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,  960,  240});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,  240,  960});
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  960,  160});
  /******************* Bottleneck 15 *******************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
//b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  160,  960});
//b->Args({  1,   1,  1,  1,  0,  0, 1, 1,  960,  240});
//b->Args({  1,   1,  1,  1,  0,  0, 1, 1,  240,  960});
//b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  960,  160});
  /******************** Last Stage *********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
//b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  160,  960});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1,  960, 1280});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1, 1280, 1001});
}

// SqueezeNet 1.0
static void SqueezeNetV10ConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /*********************** Conv 1 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({224, 224,  7,  7,  6,  6, 2, 1,    3,   96});
  /*********************** Fire 2 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 55,  55,  1,  1,  0,  0, 1, 1,   96,   16});
  b->Args({ 55,  55,  1,  1,  0,  0, 1, 1,   16,   64});
  b->Args({ 55,  55,  3,  3,  2,  2, 1, 1,   16,   64});
  /*********************** Fire 3 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 56,  55,  1,  1,  0,  0, 1, 1,  128,   16});
//b->Args({ 55,  55,  1,  1,  0,  0, 1, 1,   16,   64});
//b->Args({ 55,  55,  3,  3,  2,  2, 1, 1,   16,   64});
  /*********************** Fire 4 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 55,  55,  1,  1,  0,  0, 1, 1,  128,   32});
  b->Args({ 55,  55,  1,  1,  0,  0, 1, 1,   32,  128});
  b->Args({ 55,  55,  3,  3,  2,  2, 1, 1,   32,  128});
  /*********************** Fire 5 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 27,  27,  1,  1,  0,  0, 1, 1,  256,   32});
  b->Args({ 27,  27,  1,  1,  0,  0, 1, 1,   32,  128});
  b->Args({ 27,  27,  3,  3,  2,  2, 1, 1,   32,  128});
  /*********************** Fire 6 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 27,  27,  1,  1,  0,  0, 1, 1,  256,   48});
  b->Args({ 27,  27,  1,  1,  0,  0, 1, 1,   48,  192});
  b->Args({ 27,  27,  3,  3,  2,  2, 1, 1,   48,  192});
  /*********************** Fire 7 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 27,  27,  1,  1,  0,  0, 1, 1,  384,   48});
//b->Args({ 27,  27,  1,  1,  0,  0, 1, 1,   48,  192});
//b->Args({ 27,  27,  3,  3,  2,  2, 1, 1,   48,  192});
  /*********************** Fire 8 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 27,  27,  1,  1,  0,  0, 1, 1,  384,   64});
  b->Args({ 27,  27,  1,  1,  0,  0, 1, 1,   64,  256});
  b->Args({ 27,  27,  3,  3,  2,  2, 1, 1,   64,  256});
  /*********************** Fire 9 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 13,  13,  1,  1,  0,  0, 1, 1,  512,   64});
  b->Args({ 13,  13,  1,  1,  0,  0, 1, 1,   64,  256});
  b->Args({ 13,  13,  3,  3,  2,  2, 1, 1,   64,  256});
  /********************** Conv 10 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 13,  13,  1,  1,  0,  0, 1, 1,  512, 1000});
}

// SqueezeNet 1.1
static void SqueezeNetV11ConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /*********************** Conv 1 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({224, 224,  3,  3,  2,  2, 2, 1,    3,   64});
  /*********************** Fire 2 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 55,  55,  1,  1,  0,  0, 1, 1,   64,   16});
  b->Args({ 55,  55,  1,  1,  0,  0, 1, 1,   16,   64});
  b->Args({ 55,  55,  3,  3,  2,  2, 1, 1,   16,   64});
  /*********************** Fire 3 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 55,  55,  1,  1,  0,  0, 1, 1,  128,   16});
//b->Args({ 55,  55,  1,  1,  0,  0, 1, 1,   16,   64});
//b->Args({ 55,  55,  3,  3,  2,  2, 1, 1,   16,   64});
  /*********************** Fire 4 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 27,  27,  1,  1,  0,  0, 1, 1,  128,   32});
  b->Args({ 27,  27,  1,  1,  0,  0, 1, 1,   32,  128});
  b->Args({ 27,  27,  3,  3,  2,  2, 1, 1,   32,  128});
  /*********************** Fire 5 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 27,  27,  1,  1,  0,  0, 1, 1,  256,   32});
//b->Args({ 27,  27,  1,  1,  0,  0, 1, 1,   32,  128});
//b->Args({ 27,  27,  3,  3,  2,  2, 1, 1,   32,  128});
  /*********************** Fire 6 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 13,  13,  1,  1,  0,  0, 1, 1,  256,   48});
  b->Args({ 13,  13,  1,  1,  0,  0, 1, 1,   48,  192});
  b->Args({ 13,  13,  3,  3,  2,  2, 1, 1,   48,  192});
  /*********************** Fire 7 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 13,  13,  1,  1,  0,  0, 1, 1,  384,   48});
//b->Args({ 13,  13,  1,  1,  0,  0, 1, 1,   48,  192});
//b->Args({ 13,  13,  3,  3,  2,  2, 1, 1,   48,  192});
  /*********************** Fire 8 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 13,  13,  1,  1,  0,  0, 1, 1,  384,   64});
  b->Args({ 13,  13,  1,  1,  0,  0, 1, 1,   64,  256});
  b->Args({ 13,  13,  3,  3,  2,  2, 1, 1,   64,  256});
  /*********************** Fire 9 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 13,  13,  1,  1,  0,  0, 1, 1,  512,   64});
//b->Args({ 13,  13,  1,  1,  0,  0, 1, 1,   64,  256});
//b->Args({ 13,  13,  3,  3,  2,  2, 1, 1,   64,  256});
  /********************** Conv 10 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 13,  13,  1,  1,  0,  0, 1, 1,  512, 1000});
}

static void InceptionV3ConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({299, 299,  3,  3,  0,  0, 2, 1,    3,   32});
  b->Args({149, 149,  3,  3,  0,  0, 1, 1,   32,   32});
  b->Args({147, 147,  3,  3,  2,  2, 1, 1,   32,   64});
  b->Args({ 73,  73,  1,  1,  0,  0, 1, 1,   64,   80});
  b->Args({ 73,  73,  3,  3,  0,  0, 1, 1,   80,  192});
  b->Args({ 35,  35,  1,  1,  0,  0, 1, 1,  192,   64});
  b->Args({ 35,  35,  1,  1,  0,  0, 1, 1,  192,   48});
  b->Args({ 35,  35,  5,  5,  4,  4, 1, 1,   48,   64});
  b->Args({ 35,  35,  3,  3,  2,  2, 1, 1,   64,   96});
  b->Args({ 35,  35,  3,  3,  2,  2, 1, 1,   96,   96});
  b->Args({ 35,  35,  1,  1,  0,  0, 1, 1,  192,   32});
  b->Args({ 35,  35,  1,  1,  0,  0, 1, 1,  256,   64});
  b->Args({ 35,  35,  1,  1,  0,  0, 1, 1,  256,   48});
  b->Args({ 35,  35,  1,  1,  0,  0, 1, 1,  288,   64});
  b->Args({ 35,  35,  1,  1,  0,  0, 1, 1,  288,   48});
  b->Args({ 35,  35,  3,  3,  0,  0, 2, 1,  288,  384});
  b->Args({ 35,  35,  3,  3,  0,  0, 2, 1,   96,   96});
  b->Args({ 17,  17,  1,  1,  0,  0, 1, 1,  768,  192});
  b->Args({ 17,  17,  1,  1,  0,  0, 1, 1,  768,  128});
  b->Args({ 17,  17,  1,  7,  0,  6, 1, 1,  128,  128});
  b->Args({ 17,  17,  7,  1,  6,  0, 1, 1,  128,  192});
  b->Args({ 17,  17,  7,  1,  6,  0, 1, 1,  128,  128});
  b->Args({ 17,  17,  1,  7,  0,  6, 1, 1,  128,  192});
  b->Args({ 17,  17,  1,  1,  0,  0, 1, 1,  768,  160});
  b->Args({ 17,  17,  1,  7,  0,  6, 1, 1,  160,  160});
  b->Args({ 17,  17,  7,  1,  6,  0, 1, 1,  160,  192});
  b->Args({ 17,  17,  7,  1,  6,  0, 1, 1,  160,  160});
  b->Args({ 17,  17,  1,  7,  0,  6, 1, 1,  160,  192});
  b->Args({ 17,  17,  1,  7,  0,  6, 1, 1,  192,  192});
  b->Args({ 17,  17,  7,  1,  6,  0, 1, 1,  192,  192});
  b->Args({ 17,  17,  3,  3,  0,  0, 2, 1,  192,  320});
  b->Args({ 17,  17,  3,  3,  0,  0, 2, 1,  192,  192});
  b->Args({  8,   8,  1,  1,  0,  0, 1, 1, 1280,  320});
  b->Args({  8,   8,  1,  1,  0,  0, 1, 1, 1280,  384});
  b->Args({  8,   8,  1,  3,  0,  2, 1, 1,  384,  384});
  b->Args({  8,   8,  3,  1,  2,  0, 1, 1,  384,  384});
  b->Args({  8,   8,  1,  1,  0,  0, 1, 1, 1280,  448});
  b->Args({  8,   8,  3,  3,  2,  2, 1, 1,  448,  384});
  b->Args({  8,   8,  1,  1,  0,  0, 1, 1, 1280,  192});
  b->Args({  8,   8,  1,  1,  0,  0, 1, 1, 2048,  320});
  b->Args({  8,   8,  1,  1,  0,  0, 1, 1, 2048,  384});
  b->Args({  8,   8,  1,  1,  0,  0, 1, 1, 2048,  448});
  b->Args({  8,   8,  1,  1,  0,  0, 1, 1, 2048,  192});
  b->Args({  1,   1,  1,  1,  0,  0, 1, 1, 2048, 1001});
}

static void ResNet18ConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /********************** Conv 1 ***********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({224, 224,  7,  7,  6,  6, 2, 1,    3,   64});
  /********************* Conv 2.X **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 56,  56,  3,  3,  2,  2, 1, 1,   64,   64});
  /********************* Conv 3.X **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 56,  56,  3,  3,  2,  2, 2, 1,   64,  128});
  b->Args({ 28,  28,  3,  3,  2,  2, 1, 1,  128,  128});
  b->Args({ 56,  56,  1,  1,  0,  0, 2, 1,   64,  128});
  /********************* Conv 4.X **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  3,  3,  2,  2, 2, 1,  128,  256});
  b->Args({ 14,  14,  3,  3,  2,  2, 1, 1,  256,  256});
  b->Args({ 28,  28,  1,  1,  0,  0, 2, 1,  128,  256});
  /********************* Conv 5.X **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  3,  3,  2,  2, 2, 1,  256,  512});
  b->Args({  7,   7,  3,  3,  2,  2, 1, 1,  512,  512});
  b->Args({ 14,  14,  1,  1,  0,  0, 2, 1,  256,  512});
}

static void ResNet50ConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /********************** Conv 1 ***********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({224, 224,  7,  7,  6,  6, 2, 1,    3,   64});
  /********************* Conv 2.1 **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   64,   64});
  b->Args({ 56,  56,  3,  3,  2,  2, 1, 1,   64,   64});
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   64,  256});
//b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   64,  256});
  /********************* Conv 2.X **********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,  256,   64});
//b->Args({ 56,  56,  3,  3,  2,  2, 1, 1,   64,   64});
//b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,   64,  256});
  /********************** Conv 3.1 *********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,  256,  128});
  b->Args({ 56,  56,  3,  3,  2,  2, 2, 1,  128,  128});
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,  128,  512});
  b->Args({ 56,  56,  1,  1,  0,  0, 2, 1,  256,  512});
  /********************** Conv 3.X *********************/
  /*         H    W   KH  KW PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,  512,  128});
  b->Args({ 28,  28,  3,  3,  2,  2, 1, 1,  128,  128});
//b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,  128,  512});
  /********************** Conv 4.1 *********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,  512,  256});
  b->Args({ 28,  28,  3,  3,  2,  2, 2, 1,  256,  256});
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  256, 1024});
  b->Args({ 28,  28,  1,  1,  0,  0, 2, 1,  512, 1024});
  /********************** Conv 4.X *********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1, 1024,  256});
  b->Args({ 14,  14,  3,  3,  2,  2, 1, 1,  256,  256});
//b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  256, 1024});
  /********************** Conv 5.1 *********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1, 1024,  512});
  b->Args({ 14,  14,  3,  3,  2,  2, 2, 1,  512,  512});
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  512, 2048});
  b->Args({ 14,  14,  1,  1,  0,  0, 2, 1, 1024, 2048});
  /********************** Conv 5.X *********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({  7,   7,  1,  1,  0,  0, 1, 1, 2048,  512});
  b->Args({  7,   7,  3,  3,  2,  2, 1, 1,  512,  512});
//b->Args({  7,   7,  1,  1,  0,  0, 1, 1,  512, 2048});
}

static void VGGConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /********************** Conv 1.1 *********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({224, 224,  3,  3,  2,  2, 1, 1,    3,   64});
  /********************** Conv 1.2 *********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({224, 224,  3,  3,  2,  2, 1, 1,   64,   64});

  /********************** Conv 2.1 *********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({112, 112,  3,  3,  2,  2, 1, 1,   64,  128});
  /********************** Conv 2.2 *********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({112, 112,  3,  3,  2,  2, 1, 1,  128,  128});

  /********************** Conv 3.1 *********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 56,  56,  3,  3,  2,  2, 1, 1,  128,  256});
  /********************** Conv 3.2 *********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 56,  56,  3,  3,  2,  2, 1, 1,  256,  256});
  /********************** Conv 3.3 *********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 56,  56,  1,  1,  0,  0, 1, 1,  256,  256});

  /********************** Conv 4.1 *********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  3,  3,  2,  2, 1, 1,  256,  512});
  /********************** Conv 4.2 *********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  3,  3,  2,  2, 1, 1,  512,  512});
  /********************** Conv 4.3 *********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 28,  28,  1,  1,  0,  0, 1, 1,  512,  512});

  /********************** Conv 5.X *********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  3,  3,  2,  2, 1, 1,  512,  512});
  /********************** Conv 5.3 *********************/
  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({ 14,  14,  1,  1,  0,  0, 1, 1,  512,  512});
}

// SRCNN (9-1-5)
static void SRCNN915ConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({384, 384,  9,  9,  0,  0, 1, 1,    1,   64});
  b->Args({376, 376,  1,  1,  0,  0, 1, 1,   64,   32});
  b->Args({376, 376,  5,  5,  0,  0, 1, 1,   32,    1});
}

// SRCNN (9-3-5)
static void SRCNN935ConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({384, 384,  9,  9,  0,  0, 1, 1,    1,   64});
  b->Args({376, 376,  3,  3,  0,  0, 1, 1,   64,   32});
  b->Args({374, 374,  5,  5,  0,  0, 1, 1,   32,    1});
}

// SRCNN (9-5-5)
static void SRCNN955ConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "KH", "KW", "PH", "PW", "S", "D", "GCin", "GCout"});

  /*        H    W   KH  KW  PH  PW  S  D  GCin  GCout */
  b->Args({384, 384,  9,  9,  0,  0, 1, 1,    1,   64});
  b->Args({376, 376,  5,  5,  0,  0, 1, 1,   64,   32});
  b->Args({372, 372,  5,  5,  0,  0, 1, 1,   32,    1});
}
