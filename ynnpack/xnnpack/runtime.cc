// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/runtime.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "include/experimental.h"
#include "include/xnnpack.h"
#include "ynnpack/base/log.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/xnnpack/utils.h"
#include "ynnpack/xnnpack/xnnpack.h"
#include <pthreadpool.h>
#include "slinky/base/ref_count.h"
#include "slinky/base/thread_pool.h"

namespace {

// XNNPACK sets the FPU state. YNNPACK allows the application to control this.
// Therefore, we need to set the FPU state in the compatibility shim.
template <typename Fn>
void run_with_xnn_fpu_state(Fn fn) {
  // TODO: It would be nice to find a simple (not 10s of lines of platform
  // specific assembly) reliable implementation of controlling the FPU state.
  // In absence of that, let's just use XNNPACK's helpers for this, even though
  // it costs a few extra function calls.
  pthreadpool_parallelize_1d(
      nullptr, [fn = std::move(fn)](int) { fn(); }, 1,
      PTHREADPOOL_FLAG_DISABLE_DENORMALS);
}

}  // namespace

extern "C" {

xnn_status xnn_get_runtime_profiling_info(xnn_runtime_t runtime,
                                          xnn_profile_info param_name,
                                          size_t param_value_size,
                                          void* param_value,
                                          size_t* param_value_size_ret) {
  // TODO: I think the best we can do is a kind of sampling profiler. It
  // wouldn't be hard to do: every func implementation atomically sets a global
  // identifier to something that identifies that func, and a background thread
  // reads that global periodically. Histogram that data, and you have a
  // profiler. However, it cannot tell you the number of times anything ran,
  // only a sampling of how much time was spent in that thing.
  return xnn_status_deprecated;
}

// This object needs to be reference counted because scheduled tasks try to use
// it, potentially after it has been released by the user.
// This wrapper only exists to run tasks under `run_with_xnn_fpu_state`,
// otherwise we could just forward calls to the xnn_scheduler_v2 without adding
// a new layer to the stack.
struct xnn_threadpool : public slinky::ref_counted<xnn_threadpool> {
  ynn_threadpool_t ynn;
  ynn_scheduler scheduler;
  void* scheduler_context;

  xnn_threadpool(xnn_scheduler_v2 scheduler, void* scheduler_context) {
    this->scheduler_context = scheduler_context;
    this->scheduler.num_threads = scheduler.num_threads;
    this->scheduler.schedule = scheduler.schedule;
  }

  ~xnn_threadpool() final {
    ynn_delete_threadpool(ynn);
  }

  // This layer of `ynn_scheduler` exists to wrap the tasks in code to save,
  // modify, and restore the FPU state to what XNNPACK clients expected
  // (denormals flushed to zero).
  static int num_threads_impl(void* scheduler_context) {
    auto this_ = reinterpret_cast<xnn_threadpool*>(scheduler_context);
    return this_->scheduler.num_threads(this_->scheduler_context);
  }

  struct wrapped_task {
    slinky::ref_count<xnn_threadpool> pool;
    void* context;
    void (*task)(void*);
  };

  static void task_impl(void* context) {
    auto task = reinterpret_cast<wrapped_task*>(context);
    run_with_xnn_fpu_state([task]() { task->task(task->context); });
    delete task;
  }

  static void schedule_impl(void* scheduler_context, void* context,
                            void (*task)(void* context)) {
    auto this_ = reinterpret_cast<xnn_threadpool*>(scheduler_context);
    wrapped_task* wrapped_context = new wrapped_task();
    wrapped_context->pool = this_;
    wrapped_context->context = context;
    wrapped_context->task = task;
    this_->scheduler.schedule(this_->scheduler_context, wrapped_context,
                              task_impl);
  }

  static void destroy(xnn_threadpool* pool) {
    delete pool;
  }
};

static ynn_scheduler xnn_threadpool_scheduler = {
    xnn_threadpool::num_threads_impl, xnn_threadpool::schedule_impl};

static xnn_status create_runtime_impl(xnn_subgraph_t subgraph,
                                      xnn_weights_cache_t weights_cache,
                                      xnn_workspace_t workspace,
                                      pthreadpool_t threadpool,
                                      xnn_threadpool_t xnn_threadpool,
                                      uint32_t flags,
                                      xnn_runtime_t* runtime_out) {
  uint32_t ynn_flags = 0;
  if (flags & XNN_FLAG_SLINKY_NO_SCHEDULE) {
    ynn_flags |= YNN_RUNTIME_FLAG_NO_SCHEDULE;
  }
  if (flags & XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC) {
    YNN_LOG_WARNING()
        << "XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC flag is not supported by the "
           "YNNPACK compatibility shim. Pass YNN_FLAG_CONSISTENT_ARITHMETIC to "
           "`ynn_create_subgraph` instead.";
  }

  ynn_threadpool_t ynn_threadpool =
      xnn_threadpool ? xnn_threadpool->ynn : nullptr;

  ynn_status status =
      ynn_optimize_subgraph(subgraph->ynn, ynn_threadpool, ynn_flags);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  return ynn::xnn_status_from_ynn(ynn_create_runtime(
      subgraph->ynn, ynn_threadpool, ynn_flags, (ynn_runtime_t*)runtime_out));
}

xnn_status xnn_create_runtime_with_threadpool(xnn_subgraph_t subgraph,
                                              xnn_weights_cache_t weights_cache,
                                              xnn_threadpool_t threadpool,
                                              uint32_t flags,
                                              xnn_runtime_t* runtime_out) {
  return create_runtime_impl(subgraph, weights_cache, /*workspace=*/nullptr,
                             /*threadpool=*/nullptr,
                             /*xnn_threadpool=*/threadpool, flags, runtime_out);
}

xnn_status xnn_update_runtime_with_threadpool(xnn_runtime_t runtime,
                                              xnn_threadpool_t threadpool) {
  ((ynn_runtime_t)runtime)->eval_config.thread_pool =
      reinterpret_cast<slinky::thread_pool*>(threadpool->ynn);
  return xnn_status_success;
}

// TODO(dsharlet): Find a way to make this flag visible.
#define XNN_FLAG_SLINKY_USE_XLA_THREAD_POOL 1

xnn_status xnn_create_threadpool_v2(xnn_scheduler_v2 scheduler,
                                    void* scheduler_context, uint32_t flags,
                                    xnn_threadpool_t* threadpool_out) {
  uint32_t ynn_flags = 0;

  // This is a little bit confusing, there are two layers of schedulers here:
  // - The inner layer is a wrapper around the xnn_scheduler_v2 we got from the
  // XNNPACK client.
  // - The outer layer is a wrapper to call that scheduler, with calls to
  // pthreadpool_parallelize around the scheduler, to get the XNNPACK default
  // FPU state.
  *threadpool_out = new xnn_threadpool(scheduler, scheduler_context);

  (*threadpool_out)->add_ref();

  return ynn::xnn_status_from_ynn(
      ynn_create_threadpool(&xnn_threadpool_scheduler, *threadpool_out,
                            ynn_flags, &(*threadpool_out)->ynn));
}

xnn_status xnn_delete_threadpool(xnn_threadpool_t threadpool) {
  threadpool->release();
  return xnn_status_success;
}

xnn_status xnn_create_runtime_v4(xnn_subgraph_t subgraph,
                                 xnn_weights_cache_t weights_cache,
                                 xnn_workspace_t workspace,
                                 pthreadpool_t threadpool, uint32_t flags,
                                 xnn_runtime_t* runtime_out) {
  return create_runtime_impl(subgraph, weights_cache, /*workspace=*/nullptr,
                             threadpool, /*xnn_threadpool=*/nullptr, flags,
                             runtime_out);
}

xnn_status xnn_create_runtime(xnn_subgraph_t subgraph,
                              xnn_runtime_t* runtime_out) {
  return xnn_create_runtime_v2(subgraph, /*threadpool=*/nullptr, /*flags=*/0,
                               runtime_out);
}

xnn_status xnn_create_runtime_v2(xnn_subgraph_t subgraph,
                                 pthreadpool_t threadpool, uint32_t flags,
                                 xnn_runtime_t* runtime_out) {
  return xnn_create_runtime_v3(subgraph, /*weights_cache=*/nullptr, threadpool,
                               flags, runtime_out);
}

xnn_status xnn_create_runtime_v3(xnn_subgraph_t subgraph,
                                 xnn_weights_cache_t weights_cache,
                                 pthreadpool_t threadpool, uint32_t flags,
                                 xnn_runtime_t* runtime_out) {
  xnn_workspace_t workspace;
  xnn_status status = xnn_create_workspace(&workspace);
  if (status != xnn_status_success) {
    return status;
  }
  status = xnn_create_runtime_v4(subgraph, weights_cache, workspace, threadpool,
                                 flags, runtime_out);
  xnn_release_workspace(workspace);
  return status;
}

xnn_status xnn_reshape_external_value(xnn_runtime_t runtime,
                                      uint32_t external_id, size_t num_dims,
                                      const size_t* dims) {
  return ynn::xnn_status_from_ynn(ynn_set_external_value_shape(
      (ynn_runtime_t)runtime, external_id, num_dims, dims));
}

xnn_status xnn_get_external_value_shape(xnn_runtime_t runtime,
                                        uint32_t external_id, size_t* num_dims,
                                        size_t* dims) {
  // YNNPACK uses the input value of num_dims to know if it's going to write too
  // much to `dims`, but XNNPACK just assumes it's big enough. To avoid hitting
  // the check in YNNPACK, implement XNNPACK's assumption.
  *num_dims = XNN_MAX_TENSOR_DIMS;
  return ynn::xnn_status_from_ynn(ynn_get_external_value_shape(
      (ynn_runtime_t)runtime, external_id, num_dims, dims));
}

xnn_status xnn_reshape_runtime(xnn_runtime_t runtime) {
  return ynn::xnn_status_from_ynn(ynn_reshape_runtime((ynn_runtime_t)runtime));
}

static enum ynn_status set_external_values(
    ynn_runtime_t runtime, size_t num_external_values,
    const xnn_external_value* external_values) {
  for (size_t i = 0; i < num_external_values; ++i) {
    enum ynn_status status = ynn_set_external_value_data(
        (ynn_runtime_t)runtime, external_values[i].id, external_values[i].data);
    if (status != ynn_status_success) {
      return status;
    }
  }
  return ynn_status_success;
}

xnn_status xnn_setup_runtime(xnn_runtime_t runtime, size_t num_external_values,
                             const xnn_external_value* external_values) {
  enum ynn_status status = set_external_values(
      (ynn_runtime_t)runtime, num_external_values, external_values);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }
  return ynn::xnn_status_from_ynn(ynn_reshape_runtime((ynn_runtime_t)runtime));
}

xnn_status xnn_setup_runtime_v2(xnn_runtime_t runtime,
                                size_t num_external_values,
                                const xnn_external_value* external_values) {
  return ynn::xnn_status_from_ynn(set_external_values(
      (ynn_runtime_t)runtime, num_external_values, external_values));
}

xnn_status xnn_invoke_runtime(xnn_runtime_t runtime) {
  xnn_status result;
  run_with_xnn_fpu_state([&result, runtime]() {
    result =
        ynn::xnn_status_from_ynn(ynn_invoke_runtime((ynn_runtime_t)runtime));
  });
  return result;
}

xnn_status xnn_delete_runtime(xnn_runtime_t runtime) {
  ynn_delete_runtime((ynn_runtime_t)runtime);
  return xnn_status_success;
}

}  // extern "C"
