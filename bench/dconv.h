// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// clang-format off

#ifndef XNNPACK_BENCH_DCONV_H_
#define XNNPACK_BENCH_DCONV_H_

#include "bench/utils.h"
#include <benchmark/benchmark.h>


#define BENCHMARK_DCONV(conv_fn) \
  BENCHMARK_NAMED(conv_fn, mobilenet_v1)->Apply(MobileNetConvArguments)->UseRealTime(); \
  BENCHMARK_NAMED(conv_fn, mobilenet_v3)->Apply(MobileNetV3ConvArguments)->UseRealTime(); \
  BENCHMARK_NAMED(conv_fn, shufflenet)->Apply(ShuffleNetConvArguments)->UseRealTime(); \
  BENCHMARK_NAMED(conv_fn, squeezenet_v11)->Apply(SqueezeNetV11ConvArguments)->UseRealTime();


// ShuffleNet v1/v2.
inline void ShuffleNetConvArguments(benchmark::Benchmark* b) {
  b->ArgNames({"H", "W", "Cout"});

  /********* Conv 1 ********/
  /*        H    W   GCout */
  b->Args({224, 224,   24});
}

// MobileNet v1/v2.
inline void MobileNetConvArguments(benchmark::Benchmark* b) {
  b->ArgNames({"H", "W", "Cout"});

  /*        H    W   GCout */
  b->Args({224, 224,   32});
}

// MobileNet v3 Small/Large.
inline void MobileNetV3ConvArguments(benchmark::Benchmark* b) {
  b->ArgNames({"H", "W", "Cout"});

  /******************* Initial Stage *******************/
  /*        H    W   GCout */
  b->Args({224, 224,   16});
}

// SqueezeNet 1.1
inline void SqueezeNetV11ConvArguments(benchmark::Benchmark* b) {
  b->ArgNames({"H", "W", "GCout"});

  /*********************** Conv 1 **********************/
  /*        H    W   GCout */
  b->Args({224, 224,   64});
}

#endif  // XNNPACK_BENCH_DCONV_H_
