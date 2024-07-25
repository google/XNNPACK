// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <benchmark/benchmark.h>
#include "xnnpack/math.h"

#include <cstdint>
#include <initializer_list>
#include <functional>
#include <string>
#include <vector>

/*
 * This is a lightweight wrapper around the underlying benchmark::internal::Benchmark
 * class, which handles registration of benchmark case args. It provides default values
 * for blocksize and skips blockwise-specific cases for non-blockwise kernels.
 */
class BenchmarkWrapper {
  public:
    BenchmarkWrapper(benchmark::internal::Benchmark* benchmark, bool blockwise)
      : benchmark_(benchmark), blockwise_(blockwise), last_k_(0), last_m_(0), last_n_(0) {}

    void ArgNames(const std::initializer_list<std::string>& names) {
      std::vector<std::string> vals(names.begin(), names.end());
      // Args are standardized as M, N, K, [BL] for gemm kernels.
      if (blockwise_ && names.size() < 4) {
        vals.push_back("BL");
      }
      else if (!blockwise_ && vals.size() == 4) {
        // Drop blocksize argument for non-blockwise kernels. This declutters the
        // benchmark output slightly.
        vals.pop_back();
      }
      benchmark_->ArgNames(vals);
    }

    void Args(const std::initializer_list<int64_t>& args) {
      std::vector<int64_t> vals(args.begin(), args.end());

      // Args are standardized as M, N, K, [BL] for gemm kernels.
      auto m = vals[0];
      auto n = vals[1];
      auto k = vals[2];

      // For non-blockwise kernels, skip test cases that differ only by blocksize.
      if (!blockwise_ && m == last_m_ && n == last_n_ && k == last_k_) {
        return;
      }

      if (blockwise_ && args.size() < 4) {
        // Use K as default blocksize for parity with non-blockwise kernels.
        // This is equivalent to per-channel quantization.
        vals.push_back(round_up_po2(k, 32));
      }
      else if (!blockwise_ && vals.size() == 4) {
        // Drop blocksize argument for non-blockwise kernels. This declutters the
        // benchmark output slightly.
        vals.pop_back();
      }

      last_m_ = m;
      last_n_ = n;
      last_k_ = k;
      benchmark_->Args(vals);
    }

  private:
    benchmark::internal::Benchmark* benchmark_;
    bool blockwise_;

    int64_t last_k_;
    int64_t last_m_;
    int64_t last_n_;
};

void Configure(benchmark::internal::Benchmark* benchmark, void (setup_method)(BenchmarkWrapper*), bool blockwise) {
    BenchmarkWrapper wrapper(benchmark, blockwise);
    setup_method(&wrapper);
    benchmark->UseRealTime();
}

#define BENCHMARK_CASE(gemm_fn, test_case_name, name, setup_method, blockwise) \
  BENCHMARK_CAPTURE(gemm_fn, test_case_name, name) \
    ->Apply([](benchmark::internal::Benchmark* b) { return Configure(b, &setup_method, blockwise); }) \
    ->UseRealTime();

#define BENCHMARK_GEMM_BASE(gemm_fn, blockwise) \
  BENCHMARK_CASE(gemm_fn, mobilenet_v1, "MobileNet v1", MobileNetV1GemmArguments, blockwise); \
  BENCHMARK_CASE(gemm_fn, mobilenet_v2, "MobileNet v2", MobileNetV2GemmArguments, blockwise); \
  BENCHMARK_CASE(gemm_fn, mobilenet_v3_small, "MobileNet v3 Small", MobileNetV3SmallGemmArguments, blockwise); \
  BENCHMARK_CASE(gemm_fn, mobilenet_v3_large, "MobileNet v3 Large", MobileNetV3LargeGemmArguments, blockwise); \
  BENCHMARK_CASE(gemm_fn, shufflenet_v1_g1, "ShuffleNet v1 (1 group)", ShuffleNetV1G1GemmArguments, blockwise); \
  BENCHMARK_CASE(gemm_fn, shufflenet_v1_g2, "ShuffleNet v1 (2 groups)", ShuffleNetV1G2GemmArguments, blockwise); \
  BENCHMARK_CASE(gemm_fn, shufflenet_v1_g3, "ShuffleNet v1 (3 groups)", ShuffleNetV1G3GemmArguments, blockwise); \
  BENCHMARK_CASE(gemm_fn, shufflenet_v1_g4, "ShuffleNet v1 (4 groups)", ShuffleNetV1G4GemmArguments, blockwise); \
  BENCHMARK_CASE(gemm_fn, shufflenet_v1_g8, "ShuffleNet v1 (8 groups)", ShuffleNetV1G8GemmArguments, blockwise); \
  BENCHMARK_CASE(gemm_fn, shufflenet_v2_x05, "ShuffleNet v2 0.5X", ShuffleNetV2X05GemmArguments, blockwise); \
  BENCHMARK_CASE(gemm_fn, shufflenet_v2_x10, "ShuffleNet v2 1.0X", ShuffleNetV2X10GemmArguments, blockwise); \
  BENCHMARK_CASE(gemm_fn, shufflenet_v2_x15, "ShuffleNet v2 1.5X", ShuffleNetV2X15GemmArguments, blockwise); \
  BENCHMARK_CASE(gemm_fn, shufflenet_v2_x20, "ShuffleNet v2 2.0X", ShuffleNetV2X20GemmArguments, blockwise); \
  BENCHMARK_CASE(gemm_fn, inception_v3, "Inception v3", InceptionV3GemmArguments, blockwise); \
  BENCHMARK_CASE(gemm_fn, resnet18, "ResNet-18", ResNet18GemmArguments, blockwise); \
  BENCHMARK_CASE(gemm_fn, resnet50, "ResNet-50", ResNet50GemmArguments, blockwise); \
  BENCHMARK_CASE(gemm_fn, squeezenet_v10, "SqueezeNet 1.0", SqueezeNetV10GemmArguments, blockwise); \
  BENCHMARK_CASE(gemm_fn, squeezenet_v11, "SqueezeNet 1.1", SqueezeNetV11GemmArguments, blockwise); \
  BENCHMARK_CASE(gemm_fn, vgg, "VGG", VGGGemmArguments, blockwise); \
  BENCHMARK_CASE(gemm_fn, srcnn915, "SRCNN (9-1-5)", SRCNN915GemmArguments, blockwise); \
  BENCHMARK_CASE(gemm_fn, srcnn935, "SRCNN (9-3-5)", SRCNN935GemmArguments, blockwise); \
  BENCHMARK_CASE(gemm_fn, llm, "LLM", LLMGemmArguments, blockwise);

// Removed due to OOM SEGFAULT on 32 bit ARM.
//  BENCHMARK_CAPTURE(gemm_fn, srcnn955, "SRCNN (9-5-5)")->Apply(SRCNN955GemmArguments)->UseRealTime();


#define BENCHMARK_GEMM(gemm_fn) BENCHMARK_GEMM_BASE(gemm_fn, false)
#define BENCHMARK_GEMM_BL(gemm_fn) BENCHMARK_GEMM_BASE(gemm_fn, true)


// ShuffleNet v1 with 1 group.
//static void ShuffleNetV1G1GemmArguments(benchmark::internal::Benchmark* b) {
static void ShuffleNetV1G1GemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M       N         K    */
  b->Args({112 * 112,  24,   3 * 3 * 3});
  b->Args({ 56 *  56,  36,  24 * 1 * 1});
  b->Args({ 28 *  28, 120,  36 * 1 * 1});
  b->Args({ 28 *  28,  36, 144 * 1 * 1});
  b->Args({ 28 *  28, 144,  36 * 1 * 1});
  b->Args({ 28 *  28,  72, 144 * 1 * 1});
  b->Args({ 14 *  14, 144,  72 * 1 * 1});
  b->Args({ 14 *  14,  72, 288 * 1 * 1});
  b->Args({ 14 *  14, 288,  72 * 1 * 1});
  b->Args({ 14 *  14, 144, 288 * 1 * 1});
  b->Args({  7 *   7, 288, 144 * 1 * 1});
  b->Args({  7 *   7, 144, 576 * 1 * 1});
  b->Args({  7 *   7, 576, 144 * 1 * 1});
}

// ShuffleNet v1 with 2 groups.
static void ShuffleNetV1G2GemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M       N         K    */
  b->Args({112 * 112,  24,   3 * 3 * 3});
  b->Args({ 56 *  56,  50,  24 * 1 * 1});
  b->Args({ 28 *  28,  88,  25 * 1 * 1});
  b->Args({ 28 *  28,  25, 100 * 1 * 1});
  b->Args({ 28 *  28, 100,  25 * 1 * 1});
  b->Args({ 28 *  28,  50, 100 * 1 * 1});
  b->Args({ 14 *  14, 100,  50 * 1 * 1});
  b->Args({ 14 *  14,  50, 200 * 1 * 1});
  b->Args({ 14 *  14, 200,  50 * 1 * 1});
  b->Args({ 14 *  14, 100, 200 * 1 * 1});
  b->Args({  7 *   7, 200, 100 * 1 * 1});
  b->Args({  7 *   7, 100, 400 * 1 * 1});
  b->Args({  7 *   7, 400, 100 * 1 * 1});
}

// ShuffleNet v1 with 3 groups.
static void ShuffleNetV1G3GemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M       N         K    */
  b->Args({112 * 112,  24,   3 * 3 * 3});
  b->Args({ 56 *  56,  60,  24 * 1 * 1});
  b->Args({ 28 *  28,  72,  20 * 1 * 1});
  b->Args({ 28 *  28,  20,  80 * 1 * 1});
  b->Args({ 28 *  28,  80,  20 * 1 * 1});
  b->Args({ 28 *  28,  40,  80 * 1 * 1});
  b->Args({ 14 *  14,  80,  40 * 1 * 1});
  b->Args({ 14 *  14,  40, 160 * 1 * 1});
  b->Args({ 14 *  14, 160,  40 * 1 * 1});
  b->Args({ 14 *  14,  80, 160 * 1 * 1});
  b->Args({  7 *   7, 160,  80 * 1 * 1});
  b->Args({  7 *   7,  80, 320 * 1 * 1});
  b->Args({  7 *   7, 320,  80 * 1 * 1});
}

// ShuffleNet v1 with 4 groups.
static void ShuffleNetV1G4GemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M       N         K    */
  b->Args({112 * 112,  24,   3 * 3 * 3});
  b->Args({ 56 *  56,  68,  24 * 1 * 1});
  b->Args({ 28 *  28,  62,  17 * 1 * 1});
  b->Args({ 28 *  28,  17,  68 * 1 * 1});
  b->Args({ 28 *  28,  68,  17 * 1 * 1});
  b->Args({ 28 *  28,  34,  68 * 1 * 1});
  b->Args({ 14 *  14,  68,  34 * 1 * 1});
  b->Args({ 14 *  14,  34, 136 * 1 * 1});
  b->Args({ 14 *  14, 136,  34 * 1 * 1});
  b->Args({ 14 *  14,  68, 136 * 1 * 1});
  b->Args({  7 *   7, 136,  68 * 1 * 1});
  b->Args({  7 *   7,  68, 272 * 1 * 1});
  b->Args({  7 *   7, 272,  68 * 1 * 1});
}

// ShuffleNet v1 with 8 groups.
static void ShuffleNetV1G8GemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M       N         K    */
  b->Args({112 * 112,  24,   3 * 3 * 3});
  b->Args({ 56 *  56,  96,  24 * 1 * 1});
  b->Args({ 28 *  28,  45,  12 * 1 * 1});
  b->Args({ 28 *  28,  12,  48 * 1 * 1});
  b->Args({ 28 *  28,  48,  12 * 1 * 1});
  b->Args({ 28 *  28,  24,  48 * 1 * 1});
  b->Args({ 14 *  14,  48,  24 * 1 * 1});
  b->Args({ 14 *  14,  24,  96 * 1 * 1});
  b->Args({ 14 *  14,  96,  24 * 1 * 1});
  b->Args({ 14 *  14,  48,  96 * 1 * 1});
  b->Args({  7 *   7,  96,  48 * 1 * 1});
  b->Args({  7 *   7,  48, 192 * 1 * 1});
  b->Args({  7 *   7, 192,  48 * 1 * 1});
}

// ShuffleNet v2 (0.5X scale)
static void ShuffleNetV2X05GemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M        N         K    */
  b->Args({112 * 112,   24,   3 * 3 * 3});
  b->Args({ 56 *  56,   24,  24 * 1 * 1});
  b->Args({ 28 *  28,   24,  24 * 1 * 1});
  b->Args({ 28 *  28,   48,  48 * 1 * 1});
  b->Args({ 14 *  14,   48,  48 * 1 * 1});
  b->Args({ 14 *  14,   96,  96 * 1 * 1});
  b->Args({  7 *   7,   96,  96 * 1 * 1});
  b->Args({  7 *   7, 1024, 192 * 1 * 1});
}

// ShuffleNet v2 (1.0X scale)
static void ShuffleNetV2X10GemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M        N         K    */
  b->Args({112 * 112,   24,   3 * 3 * 3});
  b->Args({ 56 *  56,   58,  24 * 1 * 1});
  b->Args({ 28 *  28,   58,  24 * 1 * 1});
  b->Args({ 28 *  28,   58,  58 * 1 * 1});
  b->Args({ 14 *  14,  116, 116 * 1 * 1});
  b->Args({ 14 *  14,  116, 116 * 1 * 1});
  b->Args({ 14 *  14,  232, 232 * 1 * 1});
  b->Args({  7 *   7,  232, 232 * 1 * 1});
  b->Args({  7 *   7, 1024, 464 * 1 * 1});
}

// ShuffleNet v2 (1.5X scale)
static void ShuffleNetV2X15GemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M        N         K    */
  b->Args({112 * 112,   24,   3 * 3 * 3});
  b->Args({ 56 *  56,   88,  24 * 1 * 1});
  b->Args({ 28 *  28,   88,  24 * 1 * 1});
  b->Args({ 28 *  28,   88,  88 * 1 * 1});
  b->Args({ 28 *  28,  176, 176 * 1 * 1});
  b->Args({ 14 *  14,  176, 176 * 1 * 1});
  b->Args({ 14 *  14,  352, 352 * 1 * 1});
  b->Args({  7 *   7,  352, 352 * 1 * 1});
  b->Args({  7 *   7, 1024, 704 * 1 * 1});
}

// ShuffleNet v2 (2.0X scale)
static void ShuffleNetV2X20GemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M        N         K    */
  b->Args({112 * 112,   24,   3 * 3 * 3});
  b->Args({ 56 *  56,  122,  24 * 1 * 1});
  b->Args({ 28 *  28,  122,  24 * 1 * 1});
  b->Args({ 28 *  28,  122, 122 * 1 * 1});
  b->Args({ 28 *  28,  244, 244 * 1 * 1});
  b->Args({ 14 *  14,  244, 244 * 1 * 1});
  b->Args({ 14 *  14,  488, 488 * 1 * 1});
  b->Args({  7 *   7,  488, 488 * 1 * 1});
  b->Args({  7 *   7, 2048, 976 * 1 * 1});
}

static void MobileNetV1GemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M        N          K    */
  b->Args({112 * 112,   32,    3 * 3 * 3});
  b->Args({112 * 112,   64,   32 * 1 * 1});
  b->Args({ 56 *  56,  128,   64 * 1 * 1});
  b->Args({ 56 *  56,  128,  128 * 1 * 1});
  b->Args({ 28 *  28,  256,  128 * 1 * 1});
  b->Args({ 28 *  28,  256,  256 * 1 * 1});
  b->Args({ 14 *  14,  512,  256 * 1 * 1});
  b->Args({ 14 *  14,  512,  512 * 1 * 1});
  b->Args({  7 *   7, 1024,  512 * 1 * 1});
  b->Args({  7 *   7, 1024, 1024 * 1 * 1});
}

static void MobileNetV2GemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K"});

  /*********** Initial Stage ************/
  /*           M        N          K    */
  b->Args({112 * 112,   32,    3 * 3 * 3});
  /************ Bottleneck 1 ************/
  /*           M        N          K    */
  b->Args({112 * 112,   16,   32 * 1 * 1});
  /************ Bottleneck 2 ************/
  /*           M        N          K    */
  b->Args({112 * 112,   96,   16 * 1 * 1});
  b->Args({ 56 *  56,   24,   96 * 1 * 1});
  b->Args({ 56 *  56,  144,   24 * 1 * 1});
  b->Args({ 56 *  56,   24,  144 * 1 * 1});
  /************ Bottleneck 3 ************/
  /*           M        N          K    */
  b->Args({ 28 *  28,   32,  144 * 1 * 1});
  b->Args({ 28 *  28,  192,   32 * 1 * 1});
  b->Args({ 28 *  28,   32,  192 * 1 * 1});
  /************ Bottleneck 4 ************/
  /*           M        N          K    */
  b->Args({ 14 *  14,   64,  192 * 1 * 1});
  b->Args({ 14 *  14,  384,   64 * 1 * 1});
  b->Args({ 14 *  14,   64,  384 * 1 * 1});
  /************ Bottleneck 5 ************/
  /*           M        N          K    */
  b->Args({ 14 *  14,   96,  384 * 1 * 1});
  b->Args({ 14 *  14,  576,   96 * 1 * 1});
  b->Args({ 14 *  14,   96,  576 * 1 * 1});
  /************ Bottleneck 6 ************/
  /*           M        N          K    */
  b->Args({  7 *   7,  160,  576 * 1 * 1});
  b->Args({  7 *   7,  960,  160 * 1 * 1});
  b->Args({  7 *   7,  160,  960 * 1 * 1});
  /************ Bottleneck 7 ************/
  /*           M        N          K    */
  b->Args({  7 *   7,  320,  960 * 1 * 1});
  /********* Pre-pooling Conv2D *********/
  /*           M        N          K    */
  b->Args({  7 *   7, 1280,  320 * 1 * 1});
  /******** Post-pooling Conv2D *********/
  /*           M        N          K    */
  b->Args({  1 *   1, 1000, 1280 * 1 * 1});
}

static void MobileNetV3SmallGemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K"});

  /************ Initial Stage ************/
  /*           M        N          K     */
  b->Args({112 * 112,   16,    3 * 3 * 3});
  /************* Bottleneck 1 ************/
  /*           M        N          K     */
  b->Args({  1 *   1,    8,   16 * 1 * 1});
  b->Args({  1 *   1,   16,    8 * 1 * 1});
  b->Args({ 56 *  56,   16,   16 * 1 * 1});
  /************* Bottleneck 2 ************/
  /*           M        N          K     */
  b->Args({ 56 *  56,   72,   16 * 1 * 1});
  b->Args({ 28 *  28,   24,   72 * 1 * 1});
  /************* Bottleneck 3 ************/
  /*           M        N          K     */
  b->Args({ 28 *  28,   88,   24 * 1 * 1});
  b->Args({ 28 *  28,   24,   88 * 1 * 1});
  /************* Bottleneck 4 ************/
  /*           M        N          K     */
  b->Args({ 28 *  28,   96,   24 * 1 * 1});
  b->Args({  1 *   1,   24,   96 * 1 * 1});
  b->Args({  1 *   1,   96,   24 * 1 * 1});
  b->Args({ 14 *  14,   40,   96 * 1 * 1});
  /************* Bottleneck 5 ************/
  /*           M        N          K     */
  b->Args({ 14 *  14,  240,   40 * 1 * 1});
  b->Args({  1 *   1,   64,  240 * 1 * 1});
  b->Args({  1 *   1,  240,   64 * 1 * 1});
  b->Args({ 14 *  14,   40,  240 * 1 * 1});
  /************* Bottleneck 6 ************/
  /*           M        N          K     */
//b->Args({ 14 *  14,  240,   40 * 1 * 1});
//b->Args({  1 *   1,   64,  240 * 1 * 1});
//b->Args({  1 *   1,  240,   64 * 1 * 1});
//b->Args({ 14 *  14,   40,  240 * 1 * 1});
  /************* Bottleneck 7 ************/
  /*           M        N          K     */
  b->Args({ 14 *  14,  120,   40 * 1 * 1});
  b->Args({  1 *   1,   32,  120 * 1 * 1});
  b->Args({  1 *   1,  120,   32 * 1 * 1});
  b->Args({ 14 *  14,   48,  120 * 1 * 1});
  /************* Bottleneck 8 ************/
  /*           M        N          K     */
  b->Args({ 14 *  14,  144,   48 * 1 * 1});
  b->Args({  1 *   1,   40,  144 * 1 * 1});
  b->Args({  1 *   1,  144,   40 * 1 * 1});
  b->Args({ 14 *  14,   48,  144 * 1 * 1});
  /************* Bottleneck 9 ************/
  /*           M        N          K     */
  b->Args({ 14 *  14,  288,   48 * 1 * 1});
  b->Args({  1 *   1,   72,  288 * 1 * 1});
  b->Args({  1 *   1,  288,   72 * 1 * 1});
  b->Args({  7 *   7,   96,  288 * 1 * 1});
  /************ Bottleneck 10 ************/
  /*           M        N          K     */
  b->Args({  7 *   7,  576,   96 * 1 * 1});
  b->Args({  1 *   1,  144,  576 * 1 * 1});
  b->Args({  1 *   1,  576,  144 * 1 * 1});
  b->Args({  7 *   7,   96,  576 * 1 * 1});
  /************ Bottleneck 11 ************/
  /*           M        N          K     */
//b->Args({  7 *   7,  576,   96 * 1 * 1});
//b->Args({  1 *   1,  144,  576 * 1 * 1});
//b->Args({  1 *   1,  576,  144 * 1 * 1});
//b->Args({  7 *   7,   96,  576 * 1 * 1});
  /************* Last Stage  *************/
  /*           M        N          K     */
//b->Args({  7 *   7,  576,   96 * 1 * 1});
  b->Args({  1 *   1, 1024,  576 * 1 * 1});
  b->Args({  1 *   1, 1001, 1024 * 1 * 1});
}

static void MobileNetV3LargeGemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K"});

  /************ Initial Stage ************/
  /*           M        N          K     */
  b->Args({112 * 112,   16,    3 * 3 * 3});
  /************* Bottleneck 1 ************/
  /*           M        N          K     */
  b->Args({112 * 112,   16,   16 * 1 * 1});
  /************* Bottleneck 2 ************/
  /*           M        N          K     */
  b->Args({112 * 112,   64,   16 * 1 * 1});
  b->Args({ 56 *  56,   24,   64 * 1 * 1});
  /************* Bottleneck 3 ************/
  /*           M        N          K     */
  b->Args({ 56 *  56,   72,   24 * 1 * 1});
  b->Args({ 56 *  56,   24,   72 * 1 * 1});
  /************* Bottleneck 4 ************/
  /*           M        N          K     */
//b->Args({ 56 *  56,   72,   24 * 1 * 1});
  b->Args({  1 *   1,   24,   72 * 1 * 1});
  b->Args({  1 *   1,   72,   24 * 1 * 1});
  b->Args({ 28 *  28,   40,   72 * 1 * 1});
  /************* Bottleneck 5 ************/
  /*           M        N          K     */
  b->Args({ 28 *  28,  120,   40 * 1 * 1});
  b->Args({  1 *   1,   32,  120 * 1 * 1});
  b->Args({  1 *   1,  120,   32 * 1 * 1});
  b->Args({ 28 *  28,   40,  120 * 1 * 1});
  /************* Bottleneck 6 ************/
  /*           M        N          K     */
//b->Args({ 28 *  28,  120,   40 * 1 * 1});
//b->Args({  1 *   1,   32,  120 * 1 * 1});
//b->Args({  1 *   1,  120,   32 * 1 * 1});
//b->Args({ 28 *  28,   40,  120 * 1 * 1});
  /************* Bottleneck 7 ************/
  /*           M        N          K     */
  b->Args({ 28 *  28,  240,   40 * 1 * 1});
  b->Args({ 14 *  14,   80,  240 * 1 * 1});
  /************* Bottleneck 8 ************/
  /*           M        N          K     */
  b->Args({ 14 *  14,  200,   80 * 1 * 1});
  b->Args({ 14 *  14,   80,  200 * 1 * 1});
  /************* Bottleneck 9 ************/
  /*           M        N          K     */
  b->Args({ 14 *  14,  184,   80 * 1 * 1});
  b->Args({ 14 *  14,   80,  184 * 1 * 1});
  /************ Bottleneck 10 ************/
  /*           M        N          K     */
  b->Args({ 14 *  14,  184,   80 * 1 * 1});
  b->Args({ 14 *  14,   80,  184 * 1 * 1});
  /************ Bottleneck 11 ************/
  /*           M        N          K     */
  b->Args({ 14 *  14,  480,   80 * 1 * 1});
  b->Args({  1 *   1,  120,  480 * 1 * 1});
  b->Args({  1 *   1,  480,  120 * 1 * 1});
  b->Args({ 14 *  14,  112,  480 * 1 * 1});
  /************ Bottleneck 12 ************/
  /*           M        N          K     */
  b->Args({ 14 *  14,  672,  112 * 1 * 1});
  b->Args({  1 *   1,  168,  672 * 1 * 1});
  b->Args({  1 *   1,  672,  168 * 1 * 1});
  b->Args({ 14 *  14,  112,  672 * 1 * 1});
  /************ Bottleneck 13 ************/
  /*           M        N          K     */
//b->Args({ 14 *  14,  672,  112 * 1 * 1});
//b->Args({  1 *   1,  168,  672 * 1 * 1});
//b->Args({  1 *   1,  672,  168 * 1 * 1});
  b->Args({  7 *   7,  160,  672 * 1 * 1});
  /************ Bottleneck 14 ************/
  /*           M        N          K     */
  b->Args({  7 *   7,  960,  160 * 1 * 1});
  b->Args({  1 *   1,  240,  960 * 1 * 1});
  b->Args({  1 *   1,  960,  240 * 1 * 1});
  b->Args({  7 *   7,  160,  960 * 1 * 1});
  /************ Bottleneck 15 ************/
  /*           M        N          K     */
//b->Args({  7 *   7,  960,  160 * 1 * 1});
//b->Args({  1 *   1,  240,  960 * 1 * 1});
//b->Args({  1 *   1,  960,  240 * 1 * 1});
//b->Args({  7 *   7,  160,  960 * 1 * 1});
  /************* Last Stage  *************/
  /*           M        N          K     */
//b->Args({  7 *   7,  960,  160 * 1 * 1});
  b->Args({  1 *   1, 1280,  960 * 1 * 1});
  b->Args({  1 *   1, 1001, 1280 * 1 * 1});
}

// SqueezeNet 1.0
static void SqueezeNetV10GemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K"});

  /************** Conv 1 ***************/
  /*           M        N         K    */
  b->Args({111 * 111,   96,   3 * 7 * 7});
  /************** Fire 2 ***************/
  /*           M        N         K    */
  b->Args({ 55 *  55,   16,  96 * 1 * 1});
  b->Args({ 55 *  55,   64,  16 * 1 * 1});
  b->Args({ 55 *  55,   64,  16 * 3 * 3});
  /************** Fire 3 ***************/
  /*           M        N         K    */
  b->Args({ 55 *  55,   16, 128 * 1 * 1});
  /************** Fire 4 ***************/
  /*           M        N         K    */
  b->Args({ 55 *  55,   32, 128 * 1 * 1});
  b->Args({ 55 *  55,  128,  32 * 1 * 1});
  b->Args({ 55 *  55,  128,  32 * 3 * 3});
  /************** Fire 5 ***************/
  /*           M        N         K    */
  b->Args({ 27 *  27,   32, 256 * 1 * 1});
  b->Args({ 27 *  27,  128,  32 * 1 * 1});
  b->Args({ 27 *  27,  128,  32 * 3 * 3});
  /************** Fire 6 ***************/
  /*           M        N         K    */
  b->Args({ 27 *  27,   48, 256 * 1 * 1});
  b->Args({ 27 *  27,  192,  48 * 1 * 1});
  b->Args({ 27 *  27,  192,  48 * 3 * 3});
  /************** Fire 7 ***************/
  /*           M        N         K    */
  b->Args({ 27 *  27,   48, 384 * 1 * 1});
  /************** Fire 8 ***************/
  /*           M        N         K    */
  b->Args({ 27 *  27,   64, 384 * 1 * 1});
  b->Args({ 27 *  27,  256,  64 * 1 * 1});
  b->Args({ 27 *  27,  256,  64 * 3 * 3});
  /************** Fire 9 ***************/
  /*           M        N         K    */
  b->Args({ 13 *  13,   64, 512 * 1 * 1});
  b->Args({ 13 *  13,  256,  64 * 1 * 1});
  b->Args({ 13 *  13,  256,  64 * 3 * 3});
  /************** Conv 10 **************/
  /*           M        N         K    */
  b->Args({ 13 *  13, 1000, 512 * 1 * 1});
}

// SqueezeNet 1.1
static void SqueezeNetV11GemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K"});

  /************** Conv 1 ***************/
  /*           M        N         K    */
  b->Args({111 * 111,   64,   3 * 3 * 3});
  /************** Fire 2 ***************/
  /*           M        N         K    */
  b->Args({ 55 *  55,   16,  64 * 1 * 1});
  b->Args({ 55 *  55,   64,  16 * 1 * 1});
  b->Args({ 55 *  55,   64,  16 * 3 * 3});
  /************** Fire 3 ***************/
  /*           M        N         K    */
  b->Args({ 55 *  55,   16, 128 * 1 * 1});
  /************** Fire 4 ***************/
  /*           M        N         K    */
  b->Args({ 27 *  27,   32, 128 * 1 * 1});
  b->Args({ 27 *  27,  128,  32 * 1 * 1});
  b->Args({ 27 *  27,  128,  32 * 3 * 3});
  /************** Fire 5 ***************/
  /*           M        N         K    */
  b->Args({ 27 *  27,   32, 256 * 1 * 1});
  /************** Fire 6 ***************/
  /*           M        N         K    */
  b->Args({ 13 *  13,   48, 256 * 1 * 1});
  b->Args({ 13 *  13,  192,  48 * 1 * 1});
  b->Args({ 13 *  13,  192,  48 * 3 * 3});
  /************** Fire 7 ***************/
  /*           M        N         K    */
  b->Args({ 13 *  13,   48, 384 * 1 * 1});
  /************** Fire 8 ***************/
  /*           M        N         K    */
  b->Args({ 13 *  13,   64, 384 * 1 * 1});
  b->Args({ 13 *  13,  256,  64 * 1 * 1});
  b->Args({ 13 *  13,  256,  64 * 3 * 3});
  /************** Fire 9 ***************/
  /*           M        N         K    */
  b->Args({ 13 *  13,   64, 512 * 1 * 1});
  /************** Conv 10 **************/
  /*           M        N         K    */
  b->Args({ 13 *  13, 1000, 512 * 1 * 1});
}

static void InceptionV3GemmArguments(BenchmarkWrapper* b) {
  /*           M        N          K   */
  b->Args({150 * 150,   32,    3 * 3 * 3});
  b->Args({149 * 149,   32,   32 * 3 * 3});
  b->Args({149 * 149,   64,   32 * 3 * 3});
  b->Args({ 75 *  75,   80,   64 * 1 * 1});
  b->Args({ 73 *  73,  192,   80 * 3 * 3});
  b->Args({ 37 *  37,   64,  192 * 1 * 1});
  b->Args({ 37 *  37,   48,  192 * 1 * 1});
  b->Args({ 37 *  37,   64,   48 * 5 * 5});
  b->Args({ 37 *  37,   96,   64 * 3 * 3});
  b->Args({ 37 *  37,   96,   96 * 3 * 3});
  b->Args({ 37 *  37,   32,  192 * 1 * 1});
  b->Args({ 37 *  37,   64,  256 * 1 * 1});
  b->Args({ 37 *  37,   48,  256 * 1 * 1});
  b->Args({ 37 *  37,   64,  288 * 1 * 1});
  b->Args({ 37 *  37,   48,  288 * 1 * 1});
  b->Args({ 18 *  18,  384,  288 * 3 * 3});
  b->Args({ 18 *  18,   96,   96 * 3 * 3});
  b->Args({ 19 *  19,  192,  768 * 1 * 1});
  b->Args({ 19 *  19,  128,  768 * 1 * 1});
  b->Args({ 19 *  19,  128,  128 * 1 * 7});
  b->Args({ 19 *  19,  192,  128 * 7 * 1});
  b->Args({ 19 *  19,  128,  128 * 7 * 1});
  b->Args({ 19 *  19,  192,  128 * 1 * 7});
  b->Args({ 19 *  19,  160,  768 * 1 * 1});
  b->Args({ 19 *  19,  160,  160 * 1 * 7});
  b->Args({ 19 *  19,  192,  160 * 7 * 1});
  b->Args({ 19 *  19,  160,  160 * 7 * 1});
  b->Args({ 19 *  19,  192,  160 * 1 * 7});
  b->Args({ 19 *  19,  192,  192 * 1 * 7});
  b->Args({ 19 *  19,  192,  192 * 7 * 1});
  b->Args({  9 *   9,  320,  192 * 3 * 3});
  b->Args({  9 *   9,  192,  192 * 3 * 3});
  b->Args({ 10 *  10,  320, 1280 * 1 * 1});
  b->Args({ 10 *  10,  384, 1280 * 1 * 1});
  b->Args({ 10 *  10,  384,  384 * 1 * 3});
  b->Args({ 10 *  10,  384,  384 * 3 * 1});
  b->Args({ 10 *  10,  448, 1280 * 1 * 1});
  b->Args({ 10 *  10,  384,  448 * 3 * 3});
  b->Args({ 10 *  10,  192, 1280 * 1 * 1});
  b->Args({ 10 *  10,  320, 2048 * 1 * 1});
  b->Args({ 10 *  10,  384, 2048 * 1 * 1});
  b->Args({ 10 *  10,  448, 2048 * 1 * 1});
  b->Args({ 10 *  10,  192, 2048 * 1 * 1});
  b->Args({  3 *   3, 1001, 2048 * 1 * 1});
}

static void ResNet18GemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M       N         K    */
  b->Args({112 * 112,  64,   3 * 7 * 7});
  b->Args({ 56 *  56,  64,  64 * 3 * 3});
  b->Args({ 28 *  28, 128,  64 * 3 * 3});
  b->Args({ 28 *  28, 128, 128 * 3 * 3});
  b->Args({ 28 *  28, 128,  64 * 1 * 1});
  b->Args({ 14 *  14, 256, 128 * 3 * 3});
  b->Args({ 14 *  14, 256, 256 * 3 * 3});
  b->Args({ 14 *  14, 256, 128 * 1 * 1});
  b->Args({  7 *   7, 512, 256 * 3 * 3});
  b->Args({  7 *   7, 512, 512 * 3 * 3});
  b->Args({  7 *   7, 512, 256 * 1 * 1});
}

static void ResNet50GemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K"});

  /*************** Conv 1 ***************/
  /*           M        N          K    */
  b->Args({112 * 112,   64,    3 * 7 * 7});
  /************** Conv 2.X **************/
  /*           M        N          K    */
  b->Args({ 56 *  56,   64,   64 * 1 * 1});
  b->Args({ 56 *  56,   64,   64 * 3 * 3});
  b->Args({ 56 *  56,  256,   64 * 1 * 1});
  b->Args({ 56 *  56,   64,  256 * 1 * 1});
  /************** Conv 3.X **************/
  /*           M        N          K    */
  b->Args({ 56 *  56,  128,  256 * 1 * 1});
  b->Args({ 28 *  28,  128,  128 * 3 * 3});
  b->Args({ 28 *  28,  512,  128 * 1 * 1});
  b->Args({ 28 *  28,  512,  256 * 1 * 1});
  b->Args({ 28 *  28,  128,  512 * 1 * 1});
  /************** Conv 4.X **************/
  /*           M        N          K    */
  b->Args({ 28 *  28,  256,  512 * 1 * 1});
  b->Args({ 14 *  14,  256,  256 * 3 * 3});
  b->Args({ 14 *  14, 1024,  256 * 1 * 1});
  b->Args({ 14 *  14, 1024,  512 * 1 * 1});
  b->Args({ 14 *  14,  256, 1024 * 1 * 1});
  /************** Conv 5.X **************/
  /*           M        N          K    */
  b->Args({ 14 *  14,  512, 1024 * 1 * 1});
  b->Args({  7 *   7,  512,  512 * 3 * 3});
  b->Args({  7 *   7, 2048,  512 * 1 * 1});
  b->Args({  7 *   7, 2048, 1024 * 1 * 1});
  b->Args({  7 *   7,  512, 2048 * 1 * 1});
}

static void VGGGemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K"});

  /************** Conv 1.1 *************/
  /*           M        N        K     */
  b->Args({224 * 224,  64,   3 * 3 * 3});
  /************** Conv 1.2 *************/
  /*           M        N        K     */
  b->Args({224 * 224,  64,  64 * 3 * 3});
  /************** Conv 2.1 *************/
  /*           M        N        K     */
  b->Args({112 * 112, 128,  64 * 3 * 3});
  /************** Conv 2.2 *************/
  /*           M        N        K     */
  b->Args({112 * 112, 128, 128 * 3 * 3});
  /************** Conv 3.1 *************/
  /*           M        N        K     */
  b->Args({ 56 *  56, 256, 128 * 3 * 3});
  /************** Conv 3.3 *************/
  /*           M        N        K     */
  b->Args({ 56 *  56, 256, 256 * 1 * 1});
  /************** Conv 4.1 *************/
  /*           M        N        K     */
  b->Args({ 28 *  28, 512, 256 * 3 * 3});
  /************** Conv 4.2 *************/
  /*           M        N        K     */
  b->Args({ 28 *  28, 512, 512 * 3 * 3});
  /************** Conv 4.3 *************/
  /*           M        N        K     */
  b->Args({ 28 *  28, 512, 512 * 1 * 1});
  /************** Conv 5.X *************/
  /*           M        N        K     */
  b->Args({ 14 *  14, 512, 512 * 3 * 3});
  /************** Conv 5.3 *************/
  /*           M        N        K     */
  b->Args({ 14 *  14, 512, 512 * 1 * 1});
}

// SRCNN (9-1-5)
static void SRCNN915GemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M       N       K    */
  b->Args({376 * 376, 64,  1 * 9 * 9});
  b->Args({376 * 376, 32, 64 * 1 * 1});
  b->Args({372 * 372,  1, 32 * 5 * 5});
}

// SRCNN (9-3-5)
static void SRCNN935GemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M       N       K    */
  b->Args({376 * 376, 64,  1 * 9 * 9});
  b->Args({374 * 374, 32, 64 * 3 * 3});
  b->Args({370 * 370,  1, 32 * 5 * 5});
}

// SRCNN (9-5-5)
static void SRCNN955GemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M       N       K    */
  b->Args({376 * 376, 64,  1 * 9 * 9});
  b->Args({372 * 372, 32, 64 * 5 * 5});
  b->Args({368 * 368,  1, 32 * 5 * 5});
}

// Large Language Model (Generic)
static void LLMGemmArguments(BenchmarkWrapper* b) {
  b->ArgNames({"M", "N", "K", "Bl"});

  b->Args({ 128, 16, 1024, 32 });
  b->Args({ 128, 16, 1024, 256 });

  b->Args({ 128, 128, 1024, 32 });
  b->Args({ 128, 128, 1024, 256 });

  b->Args({ 128, 4096, 1024, 32 });
  b->Args({ 128, 4096, 1024, 256 });

  b->Args({ 128, 11008, 4096, 32 });
  b->Args({ 128, 11008, 4096, 256 });

  b->Args({ 128, 32000, 4096, 32 });
  b->Args({ 128, 32000, 4096, 256 });
}
