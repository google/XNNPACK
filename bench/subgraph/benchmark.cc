// Copyright 2019-2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "bench/subgraph/benchmark.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include "bench/utils.h"
#include "include/xnnpack.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/subgraph.h"
#include "test/replicable_random_device.h"
#include <benchmark/benchmark.h>
#include <pthreadpool.h>

namespace xnnpack {

namespace {

struct ModelRuntime {
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> model;
  pthreadpool_t threadpool = nullptr;
  xnn_runtime_t runtime = nullptr;
  std::vector<xnn_external_value> external_values;

  explicit ModelRuntime(int num_threads) : model(nullptr, xnn_delete_subgraph) {
    xnn_delete_runtime(runtime);
    threadpool = pthreadpool_create(num_threads);
  }

  ~ModelRuntime() {
    if (runtime) {
      xnn_delete_runtime(runtime);
    }
    if (threadpool) {
      pthreadpool_destroy(threadpool);
    }
    for (xnn_external_value& i : external_values) {
      free(i.data);
    }
  }

  bool CreateModel(std::function<xnn_subgraph_t()> model_factory) {
    model.reset(model_factory());
    if (!model) {
      return false;
    }
    return model != nullptr;
  }

  bool CreateRuntime(uint32_t flags) {
    assert(!runtime);
    return xnn_status_success == xnn_create_runtime_v4(model.get(), nullptr,
                                                       nullptr, threadpool,
                                                       flags, &runtime);
  }
  bool ReshapeRuntime() {
    return xnn_status_success == xnn_reshape_runtime(runtime);
  }

  bool SetupRuntime() {
    ReplicableRandomDevice rng;
    for (uint32_t i = 0; i < xnn_subgraph_get_num_external_values(model.get());
         ++i) {
      uint32_t flags = xnn_subgraph_get_value_flags(model.get(), i);
      if ((flags & (XNN_VALUE_FLAG_EXTERNAL_INPUT |
                    XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) == 0) {
        continue;
      }
      // Make a buffer for this external value.
      size_t num_dims = 0;
      size_t dims[XNN_MAX_TENSOR_DIMS];
      xnn_get_external_value_shape(runtime, i, &num_dims, &dims[0]);
      xnn_datatype type = xnn_subgraph_get_value_datatype(model.get(), i);
      size_t size = xnn_datatype_size_bytes(type);
      for (size_t i = 0; i < num_dims; ++i) {
        size *= dims[i];
      }
      void* data = malloc(size + XNN_EXTRA_BYTES);
      switch (type) {
        case xnn_datatype_fp32: {
          std::generate((float*)data, (float*)((uintptr_t)data + size),
                        [&] { return rng.NextFloat(); });
        } break;
        case xnn_datatype_fp16: {
          std::generate((xnn_float16*)data,
                        (xnn_float16*)((uintptr_t)data + size),
                        [&] { return rng.NextFloat(); });
        } break;
        default: {
          std::generate((uint8_t*)data, (uint8_t*)((uintptr_t)data + size),
                        [&] { return rng.NextUInt32() & 0xFF; });
        } break;
      }
      external_values.push_back(xnn_external_value{i, data});
    }
    return xnn_status_success == xnn_setup_runtime_v2(runtime,
                                                      external_values.size(),
                                                      external_values.data());
  }

  bool Invoke() { return xnn_status_success == xnn_invoke_runtime(runtime); }
};

}  // namespace

void RunBenchmark(benchmark::State& state,
                  std::function<xnn_subgraph_t()> model_factory,
                  uint32_t extra_flags) {
  if (xnn_initialize(nullptr /* allocator */) != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  ModelRuntime model_runtime(FLAGS_num_threads);
  if (!model_runtime.CreateModel(model_factory)) {
    state.SkipWithError("failed to create model");
    return;
  }

  // TODO(dsharlet): We should have benchmarks of these steps too.
  if (!model_runtime.CreateRuntime(FLAGS_xnn_runtime_flags | extra_flags)) {
    state.SkipWithError("failed to create runtime");
    return;
  }

  if (!model_runtime.ReshapeRuntime()) {
    state.SkipWithError("failed to reshape runtime");
    return;
  }

  if (!model_runtime.SetupRuntime()) {
    state.SkipWithError("failed to setup runtime");
    return;
  }

  int num_iters = FLAGS_benchmark_min_iters;
  while (state.KeepRunningBatch(num_iters)) {
    for (int iter = 0; iter < num_iters; iter++) {
      benchmark::utils::WipePthreadpoolL2Caches(state,
                                                model_runtime.threadpool);
      if (!model_runtime.Invoke()) {
        state.SkipWithError("failed to invoke runtime");
        return;
      }
    }
    num_iters = 1;
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

}  // namespace xnnpack

XNN_BENCHMARK_MAIN();
