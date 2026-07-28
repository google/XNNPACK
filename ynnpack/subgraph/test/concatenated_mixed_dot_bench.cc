// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"
#include <benchmark/benchmark.h>

namespace ynn {
namespace {

template <typename T>
void BM_MixedDot(benchmark::State& state) {
  const int thread_count = state.range(0);
  const bool concat_a = state.range(1);
  const bool static_shape = state.range(2);

  constexpr size_t M = 8;
  constexpr size_t K0 = 256;
  constexpr size_t K1 = 256;
  constexpr size_t K = K0 + K1;
  constexpr size_t N = 4096;

  // Inputs
  const float scale = 0.5f;
  const int32_t zero_point = 2;

  // Build Subgraph
  SubgraphBuilder subgraph(concat_a ? 4 : 3, 0);

  const uint32_t W_quant_id = 0;
  const uint32_t Y_id = 1;
  uint32_t X_concat_id = concat_a ? YNN_INVALID_VALUE_ID : 2;
  const uint32_t X0_id = 2;
  const uint32_t X1_id = 3;

  if (concat_a) {
    TensorShape x0_shape = static_shape ? TensorShape({M, K0}) : TensorShape(2);
    TensorShape x1_shape = static_shape ? TensorShape({M, K1}) : TensorShape(2);
    TensorShape x_concat_shape = {M, K};
    subgraph.AddInput(type_of<T>(), x0_shape, X0_id)
        .AddInput(type_of<T>(), x1_shape, X1_id)
        .AddTensor(type_of<T>(), 2, X_concat_id);

    subgraph.AddConcatenate(1, {X0_id, X1_id}, X_concat_id);
  } else {
    TensorShape x_concat_shape =
        static_shape ? TensorShape({M, K}) : TensorShape(2);
    subgraph.AddInput(type_of<T>(), x_concat_shape, X_concat_id);
  }
  subgraph.AddOutput(type_of<T>(), 2, Y_id);

  TensorShape w_quant_shape =
      static_shape ? TensorShape({K, N}) : TensorShape(2);
  subgraph.AddInput(ynn_type_int8, w_quant_shape, W_quant_id);

  uint32_t W_dequant_id = YNN_INVALID_VALUE_ID;
  subgraph.AddTensor(type_of<T>(), 2, W_dequant_id);

  quantization_params quant;
  quant.scale = scale;
  quant.zero_point = zero_point;

  subgraph.AddDequantize(W_quant_id, quant, type_of<T>(), W_dequant_id);
  subgraph.AddDot(1, X_concat_id, W_dequant_id, YNN_INVALID_VALUE_ID, Y_id);

  // Run the subgraph
  // Run the subgraph. `thread_count` is the total number of threads that should
  // run the work; the runtime's invoking thread participates as a worker, so
  // the scheduler only needs `thread_count - 1` background threads.
  TestScheduler scheduler(thread_count - 1);
  Runtime runtime(subgraph.GetSubgraph(), &scheduler);
  if (runtime.Status() != ynn_status_success) {
    state.SkipWithError("Failed to create runtime");
    return;
  }

  Tensor<T> in_x0;
  Tensor<T> in_x1;
  Tensor<T> in_x;
  if (concat_a) {
    in_x0 = Tensor<T>({M, K0});
    in_x1 = Tensor<T>({M, K1});
    std::fill(in_x0.begin(), in_x0.end(), static_cast<T>(1.0f));
    std::fill(in_x1.begin(), in_x1.end(), static_cast<T>(1.0f));
    runtime.ReshapeExternalTensor({M, K0}, in_x0.data(), X0_id);
    runtime.ReshapeExternalTensor({M, K1}, in_x1.data(), X1_id);
  } else {
    in_x = Tensor<T>({M, K});
    std::fill(in_x.begin(), in_x.end(), static_cast<T>(1.0f));
    runtime.ReshapeExternalTensor({M, K}, in_x.data(), X_concat_id);
  }

  const int8_t quant_w_val = static_cast<int8_t>(1.0f / scale + zero_point);

  Tensor<int8_t> in_w_quant({K, N});
  std::fill(in_w_quant.begin(), in_w_quant.end(), quant_w_val);
  runtime.ReshapeExternalTensor({K, N}, in_w_quant.data(), W_quant_id);

  runtime.ReshapeRuntime();
  if (runtime.Status() != ynn_status_success) {
    state.SkipWithError("Failed to reshape runtime");
    return;
  }

  Tensor<T> out_y({M, N});
  runtime.SetupExternalTensor(out_y.data(), Y_id);

  for (auto _ : state) {
    runtime.InvokeRuntime();
  }

  if (runtime.Status() != ynn_status_success) {
    state.SkipWithError("Failed to invoke runtime");
  }

  const T* out_y_ptr = out_y.data();
  for (size_t i = 0; i < M * N; ++i) {
    float val = static_cast<float>(out_y_ptr[i]);
    if (val != K) {
      state.SkipWithError("Incorrect result");
      return;
    }
  }

  const size_t ops = M * N * K * 2;
  state.counters["OP"] =
      benchmark::Counter(state.iterations() * ops, benchmark::Counter::kIsRate);
}

void params(benchmark::Benchmark* state) {
  state->ArgNames({"thread_count", "concat_a", "static_shape"});
  for (int thread_count : {1, 2, 4, 8}) {
    for (bool concat_a : {true, false}) {
      for (bool static_shape : {true, false}) {
        state->Args({thread_count, concat_a, static_shape});
      }
    }
  }
}

void BM_MixedDotF32(benchmark::State& state) { BM_MixedDot<float>(state); }
void BM_MixedDotF16(benchmark::State& state) { BM_MixedDot<half>(state); }
void BM_MixedDotBF16(benchmark::State& state) {
  BM_MixedDot<bfloat16>(state);
}

BENCHMARK(BM_MixedDotF32)->Apply(params);
BENCHMARK(BM_MixedDotF16)->Apply(params);
BENCHMARK(BM_MixedDotBF16)->Apply(params);

}  // namespace
}  // namespace ynn
