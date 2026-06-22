// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/test/buffer.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include <benchmark/benchmark.h>

namespace ynn {
namespace {

using subgraph_ptr =
    std::unique_ptr<ynn_subgraph, decltype(&ynn_delete_subgraph)>;
using runtime_ptr = std::unique_ptr<ynn_runtime, decltype(&ynn_delete_runtime)>;

template <typename Index, typename Elem>
void BM_Gather2D(benchmark::State& state) {
  const size_t output_dim0 = state.range(0);
  const size_t output_dim1 = state.range(1);
  const size_t gather_dim = state.range(2);
  const int32_t axis = state.range(3);

  ynn_subgraph_t sub_raw = nullptr;
  ynn_create_subgraph(3, 0, &sub_raw);
  subgraph_ptr subgraph(sub_raw, &ynn_delete_subgraph);

  uint32_t input_id = 0;
  uint32_t index_id = 1;
  uint32_t output_id = 2;

  size_t output_shape[2] = {output_dim0, output_dim1};
  size_t input_shape[2] = {output_dim0, output_dim1};
  size_t index_shape[2] = {output_dim0, output_dim1};
  input_shape[axis] = gather_dim;
  index_shape[1 - axis] = 1;

  ynn_define_tensor(subgraph.get(), type_of<Elem>(), 2, input_shape, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id);
  ynn_define_tensor(subgraph.get(), type_of<Index>(), 2, index_shape, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_INPUT, &index_id);
  ynn_define_tensor(subgraph.get(), type_of<Elem>(), 2, output_shape, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);

  ynn_define_gather(subgraph.get(), 1, &axis, 2, input_id, index_id, &output_id,
                    0);

  ynn_optimize_subgraph(subgraph.get(), nullptr, 0);

  ynn_runtime_t runtime_raw = nullptr;
  ynn_create_runtime(subgraph.get(), nullptr, 0, &runtime_raw);
  runtime_ptr runtime(runtime_raw, &ynn_delete_runtime);

  ynn_reshape_runtime(runtime.get());

  std::default_random_engine rng(42);
  Tensor<Elem> input({input_shape[0], input_shape[1]});
  Tensor<Elem> output({output_shape[0], output_shape[1]});
  Tensor<Index> index({index_shape[0], index_shape[1]});
  fill_random(input.data(), input.size(), rng);
  fill_random(index.data(), index.size(), rng, /*min=*/0,
              /*max=*/gather_dim - 1);

  ynn_set_external_value_data(runtime.get(), input_id, input.data());
  ynn_set_external_value_data(runtime.get(), index_id, index.data());
  ynn_set_external_value_data(runtime.get(), output_id, output.data());

  for (auto _ : state) {
    ynn_invoke_runtime(runtime.get());
  }

  state.SetBytesProcessed(
      state.iterations() *
      (index.size_bytes() + output.size_bytes() + input.size_bytes()));
}

void BM_Gather2D_U2_U8(benchmark::State& state) {
  BM_Gather2D<uint2x4, uint8_t>(state);
}
void BM_Gather2D_U2_BF16(benchmark::State& state) {
  BM_Gather2D<uint2x4, bfloat16>(state);
}
void BM_Gather2D_U4_U8(benchmark::State& state) {
  BM_Gather2D<uint4x2, uint8_t>(state);
}
void BM_Gather2D_U4_BF16(benchmark::State& state) {
  BM_Gather2D<uint4x2, bfloat16>(state);
}
void BM_Gather2D_U8_U8(benchmark::State& state) {
  BM_Gather2D<uint8_t, uint8_t>(state);
}
void BM_Gather2D_U8_BF16(benchmark::State& state) {
  BM_Gather2D<uint8_t, bfloat16>(state);
}
void BM_Gather2D_U8_F32(benchmark::State& state) {
  BM_Gather2D<uint8_t, float>(state);
}
void BM_Gather2D_S32_BF16(benchmark::State& state) {
  BM_Gather2D<int32_t, bfloat16>(state);
}
void BM_Gather2D_S32_F32(benchmark::State& state) {
  BM_Gather2D<int32_t, float>(state);
}

void Config2D(benchmark::Benchmark* b) {
  b->UseRealTime();
  b->MeasureProcessCPUTime();
  b->ArgNames({"M", "N", "K", "D"});
}

// We only support sub-byte lookups when the gather dimension is the trailing
// dimension.
BENCHMARK(BM_Gather2D_U2_U8)->Apply(Config2D)->Args({1024, 1024, 4, 1});
BENCHMARK(BM_Gather2D_U2_BF16)->Apply(Config2D)->Args({1024, 1024, 4, 1});
BENCHMARK(BM_Gather2D_U4_U8)->Apply(Config2D)->Args({1024, 1024, 16, 1});
BENCHMARK(BM_Gather2D_U4_BF16)->Apply(Config2D)->Args({1024, 1024, 16, 1});

BENCHMARK(BM_Gather2D_U8_U8)
    ->Apply(Config2D)
    ->Args({1024, 1024, 256, 0})
    ->Args({1024, 1024, 256, 1});
BENCHMARK(BM_Gather2D_U8_BF16)
    ->Apply(Config2D)
    ->Args({1024, 1024, 256, 0})
    ->Args({1024, 1024, 256, 1});
BENCHMARK(BM_Gather2D_U8_F32)
    ->Apply(Config2D)
    ->Args({1024, 1024, 256, 0})
    ->Args({1024, 1024, 256, 1});
BENCHMARK(BM_Gather2D_S32_BF16)
    ->Apply(Config2D)
    ->Args({1024, 1024, 512, 0})
    ->Args({1024, 1024, 512, 1});
BENCHMARK(BM_Gather2D_S32_F32)
    ->Apply(Config2D)
    ->Args({1024, 1024, 512, 0})
    ->Args({1024, 1024, 512, 1});

}  // namespace
}  // namespace ynn
