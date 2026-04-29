// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include <benchmark/benchmark.h>

namespace ynn {

// Helper aliases for YNNPACK C objects.
using subgraph_ptr =
    std::unique_ptr<ynn_subgraph, decltype(&ynn_delete_subgraph)>;
using runtime_ptr = std::unique_ptr<ynn_runtime, decltype(&ynn_delete_runtime)>;
using threadpool_ptr =
    std::unique_ptr<ynn_threadpool, decltype(&ynn_delete_threadpool)>;

template <typename T>
void bench_pi(benchmark::State& state) {
  const int n = state.range(0);
  const int thread_count = state.range(1);

  const size_t n_size = static_cast<size_t>(n);
  const T a = static_cast<T>(0.5);
  const T b = static_cast<T>(0.25);
  const T c = static_cast<T>(0.75);

  // pi = sum_{i=0}^{n-1} (a / ((i + b) * (i + c)))

  TestScheduler scheduler(thread_count);
  ynn_threadpool_t threadpool_raw = nullptr;
  ynn_create_threadpool(scheduler.scheduler(), &scheduler, 0, &threadpool_raw);
  threadpool_ptr threadpool(threadpool_raw, &ynn_delete_threadpool);

  ynn_subgraph_t sub_raw = nullptr;
  ynn_create_subgraph(2, 0, &sub_raw);
  subgraph_ptr subgraph(sub_raw, &ynn_delete_subgraph);

  auto define_constant = [&](T value) {
    uint32_t id = YNN_INVALID_VALUE_ID;
    ynn_define_tensor(subgraph.get(), type_of<T>(), 0, nullptr, &value,
                      YNN_VALUE_FLAG_COPY_DATA, &id);
    return id;
  };

  // output[i] = begin + i * stride
  uint32_t i_id = YNN_INVALID_VALUE_ID;
  ynn_define_iota(subgraph.get(), type_of<T>(), 1, &n_size,
                  YNN_INVALID_VALUE_ID, define_constant(1.0f), &i_id, 0);

  uint32_t i_plus_b_id = YNN_INVALID_VALUE_ID;
  ynn_define_binary(subgraph.get(), ynn_binary_add, i_id, define_constant(b),
                    &i_plus_b_id, 0);

  uint32_t i_plus_c_id = YNN_INVALID_VALUE_ID;
  ynn_define_binary(subgraph.get(), ynn_binary_add, i_id, define_constant(c),
                    &i_plus_c_id, 0);

  uint32_t denom_id = YNN_INVALID_VALUE_ID;
  ynn_define_binary(subgraph.get(), ynn_binary_multiply, i_plus_b_id,
                    i_plus_c_id, &denom_id, 0);

  // Make a non-constant so the operation cannot constant fold away.
  uint32_t a_id = 1;
  ynn_define_tensor(subgraph.get(), type_of<T>(), 0, nullptr, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_INPUT, &a_id);

  uint32_t term_id = YNN_INVALID_VALUE_ID;
  ynn_define_binary(subgraph.get(), ynn_binary_divide, a_id,
                    denom_id, &term_id, 0);

  uint32_t output_id = 0;
  ynn_define_tensor(subgraph.get(), type_of<T>(), 0, nullptr, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);

  int32_t axis = 0;
  ynn_define_reduce(subgraph.get(), ynn_reduce_sum, 1, &axis, term_id,
                    YNN_INVALID_VALUE_ID, &output_id, 0);

  // Optimize and create runtime.
  ynn_optimize_subgraph(subgraph.get(), threadpool.get(), 0);

  ynn_runtime_t runtime_raw = nullptr;
  ynn_create_runtime(subgraph.get(), threadpool.get(), 0, &runtime_raw);
  runtime_ptr runtime(runtime_raw, &ynn_delete_runtime);

  ynn_reshape_runtime(runtime.get());

  T output = 0;
  ynn_set_external_value_data(runtime.get(), a_id, const_cast<T*>(&a));
  ynn_set_external_value_data(runtime.get(), output_id, &output);

  for (auto _ : state) {
    ynn_invoke_runtime(runtime.get());
  }

  // Check that the result is approximately pi.
  const double pi = 3.14159265358979323846;
  const double actual = static_cast<double>(output);
  if (std::abs(actual - pi) > 1e-4) {
    std::cerr << "Incorrect result: " << actual << " (expected " << pi << ")"
              << std::endl;
    state.SkipWithError("Incorrect result");
  }
}

static void BM_pi_fp32(benchmark::State& state) { bench_pi<float>(state); }

static void config(benchmark::Benchmark* b) {
  b->ArgNames({"n", "thread_count"});
  b->UseRealTime();
  b->MeasureProcessCPUTime();
  for (int n : {10000, 100000, 1000000}) {
    for (int thread_count : {1, 2, 4, 8, 16}) {
      b->Args({n, thread_count});
    }
  }
}

BENCHMARK(BM_pi_fp32)->Apply(config);

}  // namespace ynn
