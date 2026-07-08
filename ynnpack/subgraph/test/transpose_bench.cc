// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"
#include <benchmark/benchmark.h>

namespace ynn {
namespace {

template <typename T>
void BM_Transpose(benchmark::State& state) {
  const std::vector<size_t> input_shape = {
      static_cast<size_t>(state.range(0)),
      static_cast<size_t>(state.range(1)),
      static_cast<size_t>(state.range(2)),
      static_cast<size_t>(state.range(3)),
  };
  const std::vector<int32_t> perm = {0, 2, 1, 3};

  // Define subgraph: Input -> Transpose -> Reshape -> Output
  SubgraphBuilder subgraph(/*external_value_count=*/2);
  uint32_t input_id = 0;
  uint32_t output_id = 1;

  subgraph.AddInput(type_of<T>(), input_shape, input_id)
      .AddOutput(type_of<T>(), 4, output_id)
      .AddTranspose(perm, input_id, output_id);

  Runtime runtime(subgraph.GetSubgraph());
  if (runtime.Status() != ynn_status_success) {
    state.SkipWithError("Failed to create runtime");
    return;
  }

  Tensor<T> input(input_shape);
  Tensor<T> output({input.size()});
  input.fill(1);
  output.fill(0);

  runtime.ReshapeExternalTensor(input_shape, input.data(), input_id)
      .SetupExternalTensor(output.data(), output_id)
      .ReshapeRuntime();

  if (runtime.Status() != ynn_status_success) {
    state.SkipWithError("Failed to reshape runtime");
    return;
  }

  // Verify initial execution for correctness.
  runtime.InvokeRuntime();
  if (runtime.Status() != ynn_status_success) {
    state.SkipWithError("Failed to invoke runtime");
    return;
  }
  for (size_t i = 0; i < output.size(); ++i) {
    if (output[i] != static_cast<T>(1)) {
      state.SkipWithError("Incorrect result in benchmark execution");
      return;
    }
  }

  for (auto _ : state) {
    runtime.InvokeRuntime();
  }

  const size_t bytes = input.size_bytes() + output.size_bytes();
  state.counters["Bytes"] = benchmark::Counter(state.iterations() * bytes,
                                               benchmark::Counter::kIsRate);
}

BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({1, 1, 4096, 1})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({1, 2, 2048, 1})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({1, 4, 1024, 1})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({1, 8, 512, 1})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({1, 16, 256, 1})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({1, 32, 128, 1})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({1, 64, 64, 1})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({1, 128, 32, 1})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({1, 256, 16, 1})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({1, 512, 8, 1})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({1, 1024, 4, 1})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({1, 2048, 2, 1})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({1, 4096, 1, 1})->UseRealTime();

BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({1024, 2, 2, 1})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({512, 2, 2, 2})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({256, 2, 2, 4})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({128, 2, 2, 8})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({64, 2, 2, 16})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({32, 2, 2, 32})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({16, 2, 2, 64})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({8, 2, 2, 128})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({4, 2, 2, 256})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({2, 2, 2, 512})->UseRealTime();
BENCHMARK_TEMPLATE(BM_Transpose, uint8_t)->Args({1, 2, 2, 1024})->UseRealTime();

}  // namespace
}  // namespace ynn
