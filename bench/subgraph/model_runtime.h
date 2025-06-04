// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "include/xnnpack.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/subgraph.h"
#include <benchmark/benchmark.h>
#include <pthreadpool.h>

namespace xnnpack {

class ModelRuntime {
 public:
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
      xnn_release_simd_memory(i.data);
    }
  }

  bool CreateModel(std::function<xnn_subgraph_t()> model_factory) {
    model.reset(model_factory());
    if (!model) {
      return false;
    }
    for (uint32_t i = 0; i < model->num_values; ++i) {
      if ((model->values[i].flags & (XNN_VALUE_FLAG_EXTERNAL_INPUT |
                                     XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) == 0) {
        continue;
      }
      // Make a buffer for this external value.
      size_t size = xnn_tensor_get_size(&model->values[i]) + XNN_EXTRA_BYTES;
      external_values.push_back(
          xnn_external_value{i, xnn_allocate_zero_simd_memory(size)});
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
    return xnn_status_success == xnn_setup_runtime_v2(runtime,
                                                      external_values.size(),
                                                      external_values.data());
  }

  bool Invoke() { return xnn_status_success == xnn_invoke_runtime(runtime); }

  static void BenchmarkInvoke(benchmark::State& state,
                              std::function<xnn_subgraph_t()> model_factory,
                              uint32_t extra_flags = 0);

 private:
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> model;
  pthreadpool_t threadpool = nullptr;
  xnn_runtime_t runtime = nullptr;
  std::vector<xnn_external_value> external_values;
};

}  // namespace xnnpack
