// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_SLINKY_THREAD_POOL_H_
#define XNNPACK_YNNPACK_SUBGRAPH_SLINKY_THREAD_POOL_H_

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "ynnpack/include/ynnpack.h"
#include "slinky/base/function_ref.h"
#include "slinky/base/ref_count.h"
#include "slinky/base/thread_pool.h"
#include "slinky/base/thread_pool_impl.h"

namespace ynn {

// This is a wrapper for `slinky::thread_pool` to use `xnn_scheduler` to provide
// the parallelism.
class slinky_thread_pool : public slinky::thread_pool {
 public:
  explicit slinky_thread_pool(const ynn_scheduler* scheduler,
                              void* scheduler_context);
  ~slinky_thread_pool() override;

  int thread_count() const override;
  slinky::ref_count<task> enqueue(size_t n, task_body t,
                                  int32_t max_workers) override;
  void wait_for(task* t) override;
  void wait_for(predicate_ref condition) override;
  void atomic_call(slinky::function_ref<void()> t) override;

 private:
  slinky::thread_pool_impl impl_;
  const ynn_scheduler* scheduler_ = nullptr;
  void* scheduler_context_ = nullptr;
  std::atomic<int> idle_workers_{0};
  int num_threads_ = 0;
};

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_SLINKY_THREAD_POOL_H_
