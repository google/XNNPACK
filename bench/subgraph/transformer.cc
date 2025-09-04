// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "bench/subgraph/benchmark.h"
#include "bench/utils.h"
#include "include/xnnpack.h"
#include <benchmark/benchmark.h>

namespace models {

// Creates a single Gemma3-like Transformer block.
xnn_subgraph_t QD8TransformerBlock(size_t batch_size, size_t sequence_length,
                                   size_t embedding_dim, size_t num_heads,
                                   size_t head_dim, size_t hidden_dim);
xnn_subgraph_t FP32TransformerBlock(size_t batch_size, size_t sequence_length,
                                    size_t embedding_dim, size_t num_heads,
                                    size_t head_dim, size_t hidden_dim);

}  // namespace models

static void QD8TransformerBlock(benchmark::State& state) {
  xnnpack::RunBenchmark(state, [&state]() {
    return models::QD8TransformerBlock(
        FLAGS_batch_size, /*sequence_length=*/state.range(0),
        /*embedding_dim=*/state.range(1), /*num_heads=*/state.range(2),
        /*head_dim=*/state.range(3), /*hidden_dim=*/state.range(4));
  });
}

static void FP32TransformerBlock(benchmark::State& state) {
  xnnpack::RunBenchmark(state, [&state]() {
    return models::FP32TransformerBlock(
        FLAGS_batch_size, /*sequence_length=*/state.range(0),
        /*embedding_dim=*/state.range(1), /*num_heads=*/state.range(2),
        /*head_dim=*/state.range(3), /*hidden_dim=*/state.range(4));
  });
}

static void FP16TransformerBlock(benchmark::State& state) {
  xnnpack::RunBenchmark(
      state,
      [&state]() {
        return models::FP32TransformerBlock(
            FLAGS_batch_size, /*sequence_length=*/state.range(0),
            /*embedding_dim=*/state.range(1), /*num_heads=*/state.range(2),
            /*head_dim=*/state.range(3), /*hidden_dim=*/state.range(4));
      },
      XNN_FLAG_FORCE_FP16_INFERENCE);
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

BENCHMARK(QD8TransformerBlock)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(TransformerBlockArguments);

BENCHMARK(FP32TransformerBlock)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(TransformerBlockArguments);

BENCHMARK(FP16TransformerBlock)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(TransformerBlockArguments);
