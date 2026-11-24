// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include <benchmark/benchmark.h>

namespace ynn {
namespace {

using subgraph_ptr =
    std::unique_ptr<ynn_subgraph, decltype(&ynn_delete_subgraph)>;
using runtime_ptr = std::unique_ptr<ynn_runtime, decltype(&ynn_delete_runtime)>;
using threadpool_ptr =
    std::unique_ptr<ynn_threadpool, decltype(&ynn_delete_threadpool)>;

subgraph_ptr create_subgraph(size_t num_external_values, uint32_t flags = 0) {
  ynn_subgraph_t subgraph = nullptr;
  ynn_create_subgraph(num_external_values, flags, &subgraph);
  return subgraph_ptr(subgraph, &ynn_delete_subgraph);
}

runtime_ptr create_runtime(ynn_subgraph_t subgraph,
                           ynn_threadpool_t threadpool = nullptr,
                           uint32_t flags = 0) {
  ynn_runtime_t runtime = nullptr;
  ynn_create_runtime(subgraph, threadpool, flags, &runtime);
  return runtime_ptr(runtime, &ynn_delete_runtime);
}

threadpool_ptr create_threadpool(ynn_scheduler_t scheduler,
                                 void* scheduler_context, uint32_t flags = 0) {
  ynn_threadpool_t threadpool = nullptr;
  ynn_create_threadpool(scheduler, scheduler_context, flags, &threadpool);
  return threadpool_ptr(threadpool, &ynn_delete_threadpool);
}

// Reproduces the runtime multiply + reduce_sum steps from blockwise dot
// (QD8F32QB4WFullyConnected):
//   mul = multiply(a[1, num_blocks, M, N] (int32),
//                  b[num_blocks, 1, N] (fp32))
//   out = reduce_sum(mul, axis=1) -> fp32[1, M, N]
void BM_MultiplyReduceInt32Fp32(benchmark::State& state) {
  constexpr size_t kBatch = 1;
  const size_t m = state.range(0);
  const size_t num_blocks = state.range(1);
  const size_t n = state.range(2);
  const bool dynamic_a = state.range(3) != 0;
  const int thread_count = state.range(4);

  TestScheduler scheduler(thread_count - 1);
  threadpool_ptr threadpool =
      thread_count > 1 ? create_threadpool(scheduler.scheduler(), &scheduler)
                       : threadpool_ptr(nullptr, &ynn_delete_threadpool);

  subgraph_ptr subgraph = create_subgraph(2);

  const size_t a_dims[4] = {kBatch, num_blocks, m, n};
  const size_t b_dims[3] = {num_blocks, 1, n};
  const size_t out_dims[3] = {kBatch, m, n};

  uint32_t a_id = 0;
  uint32_t out_id = 1;
  if (ynn_define_tensor(subgraph.get(), type_of<int32_t>(), 4,
                        dynamic_a ? nullptr : a_dims, nullptr,
                        YNN_VALUE_FLAG_EXTERNAL_INPUT,
                        &a_id) != ynn_status_success ||
      ynn_define_tensor(subgraph.get(), type_of<float>(), 3,
                        dynamic_a ? nullptr : out_dims, nullptr,
                        YNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                        &out_id) != ynn_status_success) {
    state.SkipWithError("Failed to define external tensors");
    return;
  }

  std::vector<float> b_data(num_blocks * n, 1.0f);
  uint32_t b_id = YNN_INVALID_VALUE_ID;
  if (ynn_define_tensor(subgraph.get(), type_of<float>(), 3, b_dims,
                        b_data.data(), YNN_VALUE_FLAG_COPY_DATA,
                        &b_id) != ynn_status_success) {
    state.SkipWithError("Failed to define b_scale tensor");
    return;
  }

  uint32_t mul_id = YNN_INVALID_VALUE_ID;
  if (ynn_define_binary(subgraph.get(), ynn_binary_multiply, a_id, b_id,
                        &mul_id, /*flags=*/0) != ynn_status_success) {
    state.SkipWithError("Failed to define multiply node");
    return;
  }

  const int32_t axis = 1;
  if (ynn_define_reduce(subgraph.get(), ynn_reduce_sum, 1, &axis, mul_id,
                        YNN_INVALID_VALUE_ID, &out_id,
                        /*flags=*/0) != ynn_status_success) {
    state.SkipWithError("Failed to define reduce node");
    return;
  }

  if (ynn_optimize_subgraph(subgraph.get(), threadpool.get(), /*flags=*/0) !=
      ynn_status_success) {
    state.SkipWithError("Failed to optimize subgraph");
    return;
  }

  runtime_ptr runtime = create_runtime(subgraph.get(), threadpool.get());
  if (!runtime) {
    state.SkipWithError("Failed to create runtime");
    return;
  }

  if (ynn_set_external_value_shape(runtime.get(), a_id, 4, a_dims) !=
          ynn_status_success ||
      ynn_reshape_runtime(runtime.get()) != ynn_status_success) {
    state.SkipWithError("Failed to reshape runtime");
    return;
  }

  const size_t a_size = kBatch * num_blocks * m * n;
  const size_t out_size = kBatch * m * n;
  std::vector<int32_t> a_data(a_size, 1);
  std::vector<float> out_data(out_size, 0.0f);

  if (ynn_set_external_value_data(runtime.get(), a_id, a_data.data()) !=
          ynn_status_success ||
      ynn_set_external_value_data(runtime.get(), out_id, out_data.data()) !=
          ynn_status_success) {
    state.SkipWithError("Failed to set external tensor data");
    return;
  }

  // Warm up before timing so output pages are materialized.
  if (ynn_invoke_runtime(runtime.get()) != ynn_status_success) {
    state.SkipWithError("Failed to invoke runtime");
    return;
  }

  for (auto _ : state) {
    if (ynn_invoke_runtime(runtime.get()) != ynn_status_success) {
      state.SkipWithError("Failed to invoke runtime");
      return;
    }
  }

  const float expected = static_cast<float>(num_blocks);
  for (size_t i = 0; i < out_size; ++i) {
    if (out_data[i] != expected) {
      state.SkipWithError("Incorrect result");
      return;
    }
  }

  const size_t bytes = a_size * sizeof(int32_t) +
                       num_blocks * n * sizeof(float) +
                       out_size * sizeof(float);
  state.counters["Bytes"] = benchmark::Counter(state.iterations() * bytes,
                                               benchmark::Counter::kIsRate);
  state.counters["OP"] = benchmark::Counter(state.iterations() * a_size * 2,
                                            benchmark::Counter::kIsRate);
}

void MultiplyReduceArgs(benchmark::Benchmark* b) {
  b->ArgNames({"M", "blocks", "N", "dynamic", "threads"});
  constexpr int kM = 512;
  constexpr int kNumBlocks = 32;  // K=1024 with block_size=32
  for (int n : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}) {
    for (int dynamic_a : {0, 1}) {
      b->Args({kM, kNumBlocks, n, dynamic_a, 1});
    }
  }
}

BENCHMARK(BM_MultiplyReduceInt32Fp32)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(MultiplyReduceArgs);

}  // namespace
}  // namespace ynn
