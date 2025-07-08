// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <benchmark/benchmark.h>

#include <cassert>
#include <functional>
#include <vector>

#include "bench/subgraph/model_runtime.h"
#include "bench/subgraph/models.h"
#include "bench/utils.h"
#include "include/xnnpack.h"

static void FP32Attention(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(state, [&state]() {
    return models::FP32Attention(FLAGS_batch_size, state.range(0),
                                 state.range(1), state.range(2),
                                 state.range(3));
  });
}

static void FP16Attention(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(
      state,
      [&state]() {
        return models::FP32Attention(FLAGS_batch_size, state.range(0),
                                     state.range(1), state.range(2),
                                     state.range(3));
      },
      XNN_FLAG_FORCE_FP16_INFERENCE);
}

static void FP32MobileNetV1(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(state, models::FP32MobileNetV1);
}

static void FP32MobileNetV2(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(state, models::FP32MobileNetV2);
}

static void FP32MobileNetV3Large(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(state, models::FP32MobileNetV3Large);
}

static void FP32MobileNetV3Small(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(state, models::FP32MobileNetV3Small);
}

static void FP16MobileNetV1(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(state, models::FP32MobileNetV1,
                                         XNN_FLAG_FORCE_FP16_INFERENCE);
}

static void FP16MobileNetV2(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(state, models::FP32MobileNetV2,
                                         XNN_FLAG_FORCE_FP16_INFERENCE);
}

static void FP16MobileNetV3Large(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(state, models::FP32MobileNetV3Large,
                                         XNN_FLAG_FORCE_FP16_INFERENCE);
}

static void FP16MobileNetV3Small(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(state, models::FP32MobileNetV3Small,
                                         XNN_FLAG_FORCE_FP16_INFERENCE);
}

static void QD8Attention(benchmark::State& state) {
  models::QD8AttentionWeights weights;
  xnnpack::ModelRuntime::BenchmarkInvoke(state, [&state, &weights]() {
    return models::QD8Attention(FLAGS_batch_size, state.range(0),
                                state.range(1), state.range(2), state.range(3),
                                weights);
  });
}

static void QS8MobileNetV2(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(state, models::QS8MobileNetV2);
}

static void FP32Elementwise(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(state, [&state]() {
    return models::FP32Elementwise(/*batch_size=*/state.range(0),
                                   /*num_elements=*/state.range(1),
                                   /*depth=*/state.range(2));
  });
}

static void FP32LayerNorm(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(state, [&state]() {
    return models::FP32LayerNorm(state.range(0), state.range(1), state.range(2),
                                 state.range(3));
  });
}

static void FP32L2Norm(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(state, [&state]() {
    return models::FP32L2Norm(state.range(0), state.range(1), state.range(2),
                                 state.range(3));
  });
}

static void FP32SoftmaxDecomp(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(state, [&state]() {
    return models::FP32Softmax(state.range(0), state.range(1), state.range(2),
                               state.range(3), /*use_softmax=*/false);
  });
}

static void FP32SoftmaxFused(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(state, [&state]() {
    return models::FP32Softmax(state.range(0), state.range(1), state.range(2),
                               state.range(3), /*use_softmax=*/true);
  });
}

static void FP32DepthwiseSeparable(benchmark::State& state) {
  models::FP32DepthwiseSeparableWeights weights;
  xnnpack::ModelRuntime::BenchmarkInvoke(state, [&state, &weights]() {
    return models::FP32DepthwiseSeparable(state.range(0), state.range(1),
                                          state.range(2), state.range(3),
                                          state.range(4), weights);
  });
}

static void QD8TransformerBlock(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(state, [&state]() {
    return models::QD8TransformerBlock(
        FLAGS_batch_size, /*sequence_length=*/state.range(0),
        /*embedding_dim=*/state.range(1), /*num_heads=*/state.range(2),
        /*head_dim=*/state.range(3), /*hidden_dim=*/state.range(4));
  });
}

static void FP32TransformerBlock(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(state, [&state]() {
    return models::FP32TransformerBlock(
        FLAGS_batch_size, /*sequence_length=*/state.range(0),
        /*embedding_dim=*/state.range(1), /*num_heads=*/state.range(2),
        /*head_dim=*/state.range(3), /*hidden_dim=*/state.range(4));
  });
}

static void FP16TransformerBlock(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(
      state,
      [&state]() {
        return models::FP32TransformerBlock(
            FLAGS_batch_size, /*sequence_length=*/state.range(0),
            /*embedding_dim=*/state.range(1), /*num_heads=*/state.range(2),
            /*head_dim=*/state.range(3), /*hidden_dim=*/state.range(4));
      },
      XNN_FLAG_FORCE_FP16_INFERENCE);
}

static void AttentionArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"T", "H", "N", "S"});
  b->Args({16, 25, 24, 4});
  b->Args({1536, 128, 12, 18});
  b->Args({1024, 256, 4, 46});
  b->Args({1792, 256, 8, 36});
  b->Args({1536, 256, 6, 22});
  b->Args({2048, 256, 8, 18});
  b->Args({3072, 256, 16, 28});
  b->Args({2304, 256, 8, 26});
  b->Args({2048, 64, 32, 24});
}

static void LayerNormArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K", "NormMask"});
  for (int norm_mask = 1; norm_mask < 8; norm_mask++) {
    b->Args({128, 256, 512, norm_mask});
  }
}

static void L2NormArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K", "NormMask"});
  for (int norm_mask = 1; norm_mask < 8; norm_mask++) {
    b->Args({128, 256, 512, norm_mask});
  }
}

static void SoftmaxArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K", "NormMask"});
  for (int norm_mask = 1; norm_mask < 8; norm_mask++) {
    b->Args({128, 256, 512, norm_mask});
  }
}

static void DepthwiseSeparableArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"W", "H", "KW", "CI", "CO"});

  // Mobilenet v2-ish
  b->Args({112, 112, 3, 32, 16});
  b->Args({56, 56, 3, 96, 24});
  b->Args({28, 28, 3, 144, 32});
  b->Args({14, 14, 3, 192, 64});
  b->Args({14, 14, 3, 384, 96});
  b->Args({14, 14, 3, 576, 160});
  b->Args({7, 7, 3, 960, 320});

  // Bigger
  b->Args({512, 512, 3, 128, 128});
}

static void TransformerBlockArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"T", "D", "N", "H", "F"});

  // GeminiXXS parameters.
  b->Args({128, 1536, 6, 256, 8 * 1536});

  // GeminiV3- NanoV3 parameters.
  b->Args({128, 2048, 8, 256, 8 * 2048});

  // Gemma2-2B parameters.
  b->Args({128, 2304, 8, 256, 9216});

  // Gemma3-1B parameters.
  b->Args({128, 1152, 4, 256, 6 * 1152});
}

BENCHMARK(FP32Attention)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Apply(AttentionArguments);

BENCHMARK(FP16Attention)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Apply(AttentionArguments);

BENCHMARK(FP32MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP32MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP32MobileNetV3Large)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP32MobileNetV3Small)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK(FP16MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP16MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP16MobileNetV3Large)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP16MobileNetV3Small)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK(QD8Attention)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Apply(AttentionArguments);

BENCHMARK(QS8MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK(FP32Elementwise)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->ArgNames({"B", "N", "D"})
    ->Args({1024, 1024, 6})
    ->Args({1024, 1024, 10})
    ->Args({1024, 1024, 18})
    ->Args({1024, 1024, 34});

BENCHMARK(FP32LayerNorm)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Apply(LayerNormArguments);

BENCHMARK(FP32L2Norm)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Apply(L2NormArguments);

BENCHMARK(FP32SoftmaxDecomp)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Apply(SoftmaxArguments);

BENCHMARK(FP32SoftmaxFused)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Apply(SoftmaxArguments);

BENCHMARK(FP32DepthwiseSeparable)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Apply(DepthwiseSeparableArguments);

BENCHMARK(QD8TransformerBlock)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Apply(TransformerBlockArguments);

BENCHMARK(FP32TransformerBlock)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Apply(TransformerBlockArguments);

BENCHMARK(FP16TransformerBlock)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Apply(TransformerBlockArguments);

XNN_BENCHMARK_MAIN();
