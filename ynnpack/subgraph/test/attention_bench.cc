// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/attention_graph.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include <benchmark/benchmark.h>

namespace ynn {
namespace {

using threadpool_ptr =
    std::unique_ptr<ynn_threadpool, decltype(&ynn_delete_threadpool)>;

// Layout: Q, O are [b, n, t, h]; K, V are [b, n, s, h].
//
// `query_len` == 0 is the prefill / self-attention case (t == s == range(0)).
// A non-zero `query_len` fixes the query length and takes range(0) as the KV
// context length s: `query_len` == 1 is the autoregressive decoding case (a
// single query token attending over the whole KV cache).
// When `transpose_io` is set the external tensors are sequence-major
// (Q/O [b, t, n, h], K/V [b, s, n, h]) and `define_attention` inserts the
// transposes to head-major, mirroring XNNPACK's layout.
void BenchAttention(benchmark::State& state, size_t b, size_t query_len = 0,
                    bool transpose_io = false, bool decode1 = false) {
  const bool dynamic = state.range(0);
  const size_t s = state.range(1);
  const size_t t = query_len == 0 ? s : query_len;
  const size_t h = state.range(2);
  const size_t n = state.range(3);
  const int num_threads = static_cast<int>(state.range(4));
  const int s_active = std::min<int>(s, static_cast<int>(state.range(5)));
  const float scale = 1.0f / std::sqrt(static_cast<float>(h));

  // The `threads` argument is the total number of threads that should run the
  // work. The runtime's invoking thread participates as a worker, so the
  // scheduler only needs `num_threads - 1` background threads.
  TestScheduler scheduler(num_threads - 1);

  ynn_threadpool_t threadpool_raw = nullptr;
  ynn_create_threadpool(TestScheduler::scheduler(), &scheduler, 0,
                        &threadpool_raw);
  threadpool_ptr threadpool(threadpool_raw, &ynn_delete_threadpool);

  subgraph_ptr subgraph = create_subgraph(s_active != s ? 5 : 4, 0);
  if (!subgraph) {
    state.SkipWithError("failed to create subgraph");
    return;
  }

  // Define the external tensors.
  const size_t qo_dims[] = {b, transpose_io ? t : n, transpose_io ? n : t, h};
  const size_t kv_dims[] = {b, transpose_io ? s : n, transpose_io ? n : s, h};
  const size_t kv_active_dims[] = {b, transpose_io ? s_active : n,
                                   transpose_io ? n : s_active, h};
  uint32_t q_id = 0, k_id = 1, v_id = 2, o_id = 3, dummy_kv_id = 4;
  ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4,
                    dynamic ? nullptr : qo_dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_INPUT, &q_id);
  ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4,
                    dynamic ? nullptr : kv_dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_INPUT, &k_id);
  ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4,
                    dynamic ? nullptr : kv_dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_INPUT, &v_id);
  ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4,
                    dynamic ? nullptr : qo_dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &o_id);

  uint32_t actual_k_id = k_id;
  uint32_t actual_v_id = v_id;
  uint32_t actual_o_id = (s_active != s) ? YNN_INVALID_VALUE_ID : o_id;

  if (s_active != s) {
    // Slice the K and V values to the active sequence length in the reduction
    // dimension.
    ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4,
                      dynamic ? nullptr : kv_active_dims, nullptr,
                      YNN_VALUE_FLAG_EXTERNAL_INPUT, &dummy_kv_id);

    int32_t seq_axis = transpose_io ? 1 : 2;
    uint32_t sliced_k_id = YNN_INVALID_VALUE_ID;
    uint32_t sliced_v_id = YNN_INVALID_VALUE_ID;

    if (ynn_define_slice_like(subgraph.get(), /*num_axes=*/1, &seq_axis, k_id,
                              dummy_kv_id, &sliced_k_id,
                              /*flags=*/0) != ynn_status_success ||
        ynn_define_slice_like(subgraph.get(), /*num_axes=*/1, &seq_axis, v_id,
                              dummy_kv_id, &sliced_v_id,
                              /*flags=*/0) != ynn_status_success) {
      state.SkipWithError("failed to define slice_like for KV cache");
      return;
    }
    actual_k_id = sliced_k_id;
    actual_v_id = sliced_v_id;
  }

  ynn_status status;
  if (decode1) {
    status =
        define_attention_decode1(subgraph.get(), q_id, actual_k_id, actual_v_id,
                                 scale, actual_o_id, transpose_io);
  } else {
    status = define_attention(subgraph.get(), q_id, actual_k_id, actual_v_id,
                              scale, actual_o_id, transpose_io);
  }

  if (status != ynn_status_success) {
    state.SkipWithError("failed to define attention");
    return;
  }

  if (s_active != s) {
    // Slice the output to the active sequence length. Note that this keeps the
    // original shape, it just limits the range of the computation to the active
    // sequence length.
    int32_t seq_axis = transpose_io ? 1 : 2;
    if (ynn_define_slice_like(subgraph.get(), /*num_axes=*/1, &seq_axis,
                              actual_o_id, dummy_kv_id, &o_id,
                              YNN_NODE_FLAG_KEEP_SHAPE) != ynn_status_success) {
      state.SkipWithError("failed to define slice_like for attention output");
      return;
    }
  }

  if (ynn_optimize_subgraph(subgraph.get(), threadpool.get(), 0) !=
      ynn_status_success) {
    state.SkipWithError("failed to optimize subgraph");
    return;
  }

  runtime_ptr runtime = create_runtime(subgraph, threadpool.get(), 0);
  if (!runtime) {
    state.SkipWithError("failed to create runtime");
    return;
  }

  std::vector<float> q(b * n * t * h, 0.01f);
  std::vector<float> k(b * n * s * h, 0.01f);
  std::vector<float> v(b * n * s * h, 0.01f);
  std::vector<float> o(b * n * t * h);

  if (ynn_set_external_value_shape(runtime.get(), q_id, 4, qo_dims) !=
          ynn_status_success ||
      ynn_set_external_value_shape(runtime.get(), k_id, 4, kv_dims) !=
          ynn_status_success ||
      ynn_set_external_value_shape(runtime.get(), v_id, 4, kv_dims) !=
          ynn_status_success ||
      (s_active != s &&
       ynn_set_external_value_shape(runtime.get(), dummy_kv_id, 4,
                                    kv_active_dims) != ynn_status_success) ||
      ynn_set_external_value_data(runtime.get(), q_id, q.data()) !=
          ynn_status_success ||
      ynn_set_external_value_data(runtime.get(), k_id, k.data()) !=
          ynn_status_success ||
      ynn_set_external_value_data(runtime.get(), v_id, v.data()) !=
          ynn_status_success ||
      ynn_set_external_value_data(runtime.get(), o_id, o.data()) !=
          ynn_status_success) {
    state.SkipWithError("failed to set external values");
    return;
  }

  if (ynn_reshape_runtime(runtime.get()) != ynn_status_success) {
    state.SkipWithError("failed to reshape runtime");
    return;
  }

  for (auto _ : state) {
    if (ynn_invoke_runtime(runtime.get()) != ynn_status_success) {
      state.SkipWithError("failed to invoke runtime");
      return;
    }
  }

  const size_t flops = 2ull * b * n * t * s_active * h * 2;  // QK^T and P@V
  state.counters["FLOP"] =
      benchmark::Counter(static_cast<double>(state.iterations() * flops),
                         benchmark::Counter::kIsRate);
}

void Attention(benchmark::State& state) { BenchAttention(state, /*b=*/1); }

void AttentionTransposed(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/0,
                 /*transpose_io=*/true);
}

void AttentionDecodeTransposed(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1,
                 /*transpose_io=*/true);
}

// Decoding case: a single query token attends over a range(0)-long KV cache.
// The score slab is 1 x s, so vanilla never materializes a large scores matrix
// and the workload is dominated by streaming K and V once each (memory-bound).
void AttentionDecode(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1);
}

// Same as AttentionDecode, but keeps K as the dot's `A` operand (natural
// layout, no transpose/pack) and makes Q the `B` operand instead (a free
// size-1-axis swap), avoiding the O(s * h) K-transpose/pack that
// AttentionDecode pays on every decode step. See define_attention_decode1.
void AttentionDecode1Transposed(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1,
                 /*transpose_io=*/true, /*decode1=*/true);
}

void AttentionDecode1(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1,
                 /*transpose_io=*/false, /*decode1=*/true);
}

void AttentionArguments(benchmark::Benchmark* b) {
  b->ArgNames({"dynamic", "seq", "head", "heads", "threads", "seq_active"});
  b->UseRealTime();
  b->MeasureProcessCPUTime();
  std::vector<std::vector<int>> shapes = {{256, 64, 8},
                                          {512, 64, 8},
                                          {1024, 64, 8},
                                          {1024, 64, 32},
                                          {4096, 64, 32}};
  for (bool dynamic : {false, true}) {
    for (const auto& shape : shapes) {
      for (int threads : {1, 2, 4}) {
        for (float s_fraction : {0.01f, 0.5f, 0.99f, 1.0f}) {
          const int s_active = std::ceil(s_fraction * shape[0]);
          b->Args({dynamic, shape[0], shape[1], shape[2], threads, s_active});
        }
      }
    }
  }
}

BENCHMARK(Attention)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(AttentionTransposed)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(AttentionDecodeTransposed)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(AttentionDecode)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(AttentionDecode1Transposed)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(AttentionDecode1)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

}  // namespace
}  // namespace ynn
