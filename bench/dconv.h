// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <benchmark/benchmark.h>


#define BENCHMARK_DCONV(conv_fn) \
  BENCHMARK_CAPTURE(conv_fn, mobilenet_v1, "MobileNet v1/v2")->Apply(MobileNetConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, mobilenet_v3, "MobileNet v3")->Apply(MobileNetV3ConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, shufflenet, "ShuffleNet v1/v2")->Apply(ShuffleNetConvArguments)->UseRealTime(); \
  BENCHMARK_CAPTURE(conv_fn, squeezenet_v11, "SqueezeNet 1.1")->Apply(SqueezeNetV11ConvArguments)->UseRealTime();


// ShuffleNet v1/v2.
static void ShuffleNetConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "Cout"});

  /********* Conv 1 ********/
  /*        H    W   GCout */
  b->Args({224, 224,   24});
}

// MobileNet v1/v2.
static void MobileNetConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "Cout"});

  /*        H    W   GCout */
  b->Args({224, 224,   32});
}

// MobileNet v3 Small/Large.
static void MobileNetV3ConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "Cout"});

  /******************* Initial Stage *******************/
  /*        H    W   GCout */
  b->Args({224, 224,   16});
}

// SqueezeNet 1.1
static void SqueezeNetV11ConvArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"H", "W", "GCout"});

  /*********************** Conv 1 **********************/
  /*        H    W   GCout */
  b->Args({224, 224,   64});
}
