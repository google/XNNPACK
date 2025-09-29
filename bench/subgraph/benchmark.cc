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
#include <vector>

#include "bench/subgraph/simple_scheduler.h"
#include "bench/utils.h"
#include "include/experimental.h"
#include "include/xnnpack.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/subgraph.h"
#include "test/replicable_random_device.h"
#include <benchmark/benchmark.h>
#include <pthreadpool.h>

namespace xnnpack {

namespace {

// Base class for loading and running models with XNNPACK.
class ModelRuntimeBase {
 public:
  ModelRuntimeBase() : model_(nullptr, xnn_delete_subgraph) {};

  virtual ~ModelRuntimeBase() {
    if (runtime_) {
      xnn_delete_runtime(runtime_);
    }
    for (xnn_external_value& i : external_values_) {
      free(i.data);
    }
  }

  bool CreateModel(std::function<xnn_subgraph_t()> model_factory) {
    model_.reset(model_factory());
    return model_ != nullptr;
  }

  bool CreateRuntime(uint32_t flags) {
    assert(!runtime_);
    return CreateRuntimeImpl(model_.get(), flags, &runtime_);
  }

  bool ReshapeRuntime() {
    return xnn_status_success == xnn_reshape_runtime(runtime_);
  }

  bool SetupRuntime() {
    ReplicableRandomDevice rng;
    for (uint32_t i = 0; i < xnn_subgraph_get_num_external_values(model_.get());
         ++i) {
      uint32_t flags = xnn_subgraph_get_value_flags(model_.get(), i);
      if ((flags & (XNN_VALUE_FLAG_EXTERNAL_INPUT |
                    XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) == 0) {
        continue;
      }
      // Make a buffer for this external value.
      size_t num_dims = 0;
      size_t dims[XNN_MAX_TENSOR_DIMS];
      xnn_get_external_value_shape(runtime_, i, &num_dims, &dims[0]);
      xnn_datatype type = xnn_subgraph_get_value_datatype(model_.get(), i);
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
      external_values_.push_back(xnn_external_value{i, data});
    }
    return xnn_status_success == xnn_setup_runtime_v2(runtime_,
                                                      external_values_.size(),
                                                      external_values_.data());
  }

  bool Invoke() { return xnn_status_success == xnn_invoke_runtime(runtime_); }

  virtual void WipeL2Caches(benchmark::State& state) {
    benchmark::utils::WipePthreadpoolL2Caches(state, /*threadpool=*/nullptr);
  }

 protected:
  // This function needs to be overridden by subclasses.
  virtual bool CreateRuntimeImpl(xnn_subgraph_t subgraph, uint32_t flags,
                                 xnn_runtime_t* runtime) = 0;

 private:
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> model_;
  xnn_runtime_t runtime_ = nullptr;
  std::vector<xnn_external_value> external_values_;
};

// Create and run a model in XNNPACK using a `pthreadpool` for parallelism.
class ModelRuntimePthreadpool : public ModelRuntimeBase {
 public:
  explicit ModelRuntimePthreadpool(int num_threads)
      : threadpool_(pthreadpool_create(num_threads), pthreadpool_destroy) {}

  void WipeL2Caches(benchmark::State& state) override {
    benchmark::utils::WipePthreadpoolL2Caches(state, threadpool_.get());
  }

 protected:
  bool CreateRuntimeImpl(xnn_subgraph_t subgraph, uint32_t flags,
                         xnn_runtime_t* runtime) override {
    return (xnn_status_success ==
            xnn_create_runtime_v4(subgraph, /*weights_cache=*/nullptr,
                                  /*workspace=*/nullptr, threadpool_.get(),
                                  flags, runtime));
  }

 private:
  std::unique_ptr<struct pthreadpool, decltype(&pthreadpool_destroy)>
      threadpool_;
};

// Create and run a model in XNNPACK using an `xnn_scheduler_v2` for
// parallelism.
class ModelRuntimeXnnThreadpool : public ModelRuntimeBase {
 public:
  explicit ModelRuntimeXnnThreadpool(int num_threads)
      : scheduler_(
            std::make_unique<xnnpack::SimpleScheduler>(num_threads - 1)) {
    enum xnn_status status = xnn_create_threadpool_v2(
        scheduler_->GetXnnSchedulerV2(), scheduler_->GetContext(), /*flags=*/0,
        &threadpool_);
    assert(status == xnn_status_success);
    (void)status;
  }
  ~ModelRuntimeXnnThreadpool() override {
    if (threadpool_) {
      xnn_delete_threadpool(threadpool_);
    }
  }

  void WipeL2Caches(benchmark::State& state) override {
    benchmark::utils::WipeSchedulerL2Caches(
        state, scheduler_->GetXnnSchedulerV2(), scheduler_->GetContext());
  }

 protected:
  bool CreateRuntimeImpl(xnn_subgraph_t subgraph, uint32_t flags,
                         xnn_runtime_t* runtime) override {
    return (xnn_status_success == xnn_create_runtime_with_threadpool(
                                      subgraph, /*weights_cache=*/nullptr,
                                      threadpool_, flags, runtime));
  }

 private:
  std::unique_ptr<xnnpack::SimpleScheduler> scheduler_;
  xnn_threadpool_t threadpool_;
};

}  // namespace

template <class M>
void RunBenchmarkImpl(benchmark::State& state,
                      std::function<xnn_subgraph_t()> model_factory,
                      uint32_t extra_flags) {
  if (xnn_initialize(nullptr /* allocator */) != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  M model_runtime(FLAGS_num_threads);
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
      model_runtime.WipeL2Caches(state);
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

void RunBenchmark(benchmark::State& state,
                  std::function<xnn_subgraph_t()> model_factory,
                  uint32_t extra_flags) {
#ifdef XNN_BENCHMARK_USE_THREADPOOL
  RunBenchmarkImpl<ModelRuntimeXnnThreadpool>(state, model_factory,
                                              extra_flags);
#else
  RunBenchmarkImpl<ModelRuntimePthreadpool>(state, model_factory, extra_flags);
#endif
}

}  // namespace xnnpack

XNN_BENCHMARK_MAIN();
