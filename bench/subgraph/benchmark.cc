// Copyright 2019-2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "bench/subgraph/benchmark.h"

#include <algorithm>
#include <atomic>
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

unique_subgraph_ptr CreateUniqueSubgraph(uint32_t num_external_values,
                                         uint32_t external_value_flags) {
  xnn_subgraph_t subgraph = nullptr;
  xnn_status status =
      xnn_create_subgraph(num_external_values, external_value_flags, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgraph" << std::endl;
    assert(!subgraph);
  }
  return unique_subgraph_ptr(subgraph, xnn_delete_subgraph);
}

namespace {

class TrackingAllocator {
 public:
  TrackingAllocator()
      : current_allocated_(0), peak_allocated_(0), baseline_allocated_(0) {}

  void RecordAlloc(size_t size) {
    size_t current = current_allocated_.fetch_add(size) + size;
    size_t peak = peak_allocated_.load();
    while (current > peak &&
           !peak_allocated_.compare_exchange_weak(peak, current)) {
      // loop until we successfully update or peak becomes larger than current
    }
  }

  void RecordFree(size_t size) {
    size_t current = current_allocated_.load();
    if (current >= size) {
      current_allocated_.fetch_sub(size);
    } else {
      current_allocated_.store(0);
    }
  }

  void Reset(size_t baseline) {
    baseline_allocated_.store(baseline);
    peak_allocated_.store(baseline);
  }

  size_t current_allocated() const { return current_allocated_.load(); }
  size_t peak_allocated() const { return peak_allocated_.load(); }
  size_t baseline_allocated() const { return baseline_allocated_.load(); }

 private:
  std::atomic<size_t> current_allocated_;
  std::atomic<size_t> peak_allocated_;
  std::atomic<size_t> baseline_allocated_;
};

static TrackingAllocator global_tracking_allocator;

static void TrackingDeallocate(void* context, void* pointer) {
  if (!pointer) return;
  auto* self = static_cast<TrackingAllocator*>(context);
  void* original_ptr = static_cast<char*>(pointer) - sizeof(size_t);
  size_t size = *static_cast<size_t*>(original_ptr);
  self->RecordFree(size);
  std::free(original_ptr);
}

static void* TrackingAllocate(void* context, size_t size) {
  auto* self = static_cast<TrackingAllocator*>(context);
  size_t total_size = size + sizeof(size_t);
  void* ptr = std::malloc(total_size);
  if (!ptr) return nullptr;
  *static_cast<size_t*>(ptr) = size;
  self->RecordAlloc(size);
  return static_cast<char*>(ptr) + sizeof(size_t);
}

static void* TrackingReallocate(void* context, void* pointer, size_t size) {
  if (!pointer) {
    return TrackingAllocate(context, size);
  }
  void* old_original_ptr = static_cast<char*>(pointer) - sizeof(size_t);
  size_t old_size = *static_cast<size_t*>(old_original_ptr);

  void* new_payload = TrackingAllocate(context, size);
  if (!new_payload) return nullptr;

  size_t copy_size = std::min(old_size, size);
  std::memcpy(new_payload, pointer, copy_size);
  TrackingDeallocate(context, pointer);
  return new_payload;
}

static void* TrackingAlignedAllocate(void* context, size_t alignment,
                                     size_t size) {
  auto* self = static_cast<TrackingAllocator*>(context);
  size_t header_size = sizeof(void*) + sizeof(size_t);
  size_t total_size = size + alignment + header_size;
  void* raw_ptr = std::malloc(total_size);
  if (!raw_ptr) return nullptr;

  uintptr_t raw_addr = reinterpret_cast<uintptr_t>(raw_ptr);
  uintptr_t payload_addr =
      (raw_addr + header_size + alignment - 1) & ~(alignment - 1);

  void* payload_ptr = reinterpret_cast<void*>(payload_addr);
  void* header_ptr = static_cast<char*>(payload_ptr) - header_size;

  *static_cast<void**>(header_ptr) = raw_ptr;
  *reinterpret_cast<size_t*>(static_cast<char*>(header_ptr) + sizeof(void*)) =
      size;

  self->RecordAlloc(size);
  return payload_ptr;
}

static void TrackingAlignedDeallocate(void* context, void* pointer) {
  if (!pointer) return;
  auto* self = static_cast<TrackingAllocator*>(context);
  size_t header_size = sizeof(void*) + sizeof(size_t);
  void* header_ptr = static_cast<char*>(pointer) - header_size;

  void* raw_ptr = *static_cast<void**>(header_ptr);
  size_t size = *reinterpret_cast<size_t*>(static_cast<char*>(header_ptr) +
                                           sizeof(void*));

  self->RecordFree(size);
  std::free(raw_ptr);
}

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
    if (FLAGS_wipe_caches) {
      benchmark::utils::WipePthreadpoolL2Caches(state, /*threadpool=*/nullptr);
    }
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
    if (FLAGS_wipe_caches) {
      benchmark::utils::WipePthreadpoolL2Caches(state, threadpool_.get());
    }
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
    if (FLAGS_wipe_caches) {
      benchmark::utils::WipeSchedulerL2Caches(
          state, scheduler_->GetXnnSchedulerV2(), scheduler_->GetContext());
    }
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
  xnn_allocator struct_allocator = {
      &global_tracking_allocator, TrackingAllocate,
      TrackingReallocate,         TrackingDeallocate,
      TrackingAlignedAllocate,    TrackingAlignedDeallocate};

  if (xnn_initialize(&struct_allocator) != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  global_tracking_allocator.Reset(
      global_tracking_allocator.current_allocated());
  const size_t baseline_allocated =
      global_tracking_allocator.baseline_allocated();

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

  const size_t peak_allocated = global_tracking_allocator.peak_allocated();
  const double peak_allocated_mb =
      (double)(peak_allocated - baseline_allocated) / (1024.0 * 1024.0);
  state.counters["PeakAllocated_MB"] = peak_allocated_mb;

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
