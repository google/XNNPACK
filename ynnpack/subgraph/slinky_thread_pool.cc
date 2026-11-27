// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/slinky_thread_pool.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <thread>  // NOLINT(build/c++11)

#include "ynnpack/include/ynnpack.h"
#include "slinky/base/function_ref.h"
#include "slinky/base/ref_count.h"
#include "slinky/base/thread_pool.h"

namespace ynn {

slinky_thread_pool::slinky_thread_pool(const ynn_scheduler* scheduler,
                                       void* scheduler_context)
    : impl_(/*workers=*/0),
      scheduler_(scheduler),
      scheduler_context_(scheduler_context) {
  num_threads_ =
      scheduler_ ? std::max(0, scheduler_->num_threads(scheduler_context_)) : 0;
  idle_workers_.store(num_threads_, std::memory_order_relaxed);
  impl_.expect_workers(num_threads_);
}

slinky_thread_pool::~slinky_thread_pool() {
  // We should wait until all our scheduled workers run before returning. If we
  // don't do this, the workers could get executed (and access this object)
  // after this destructor runs.
  // TODO: Try to find a way to do this without spinning, and without adding
  // overhead to the steady state (b/429222328).
  while (idle_workers_.load(std::memory_order_acquire) < num_threads_) {
    std::this_thread::yield();
  }
}

int slinky_thread_pool::thread_count() const { return num_threads_; }

slinky::ref_count<slinky::thread_pool::task> slinky_thread_pool::enqueue(
    size_t n, task_body t, int max_workers) {
  auto result = impl_.enqueue(n, t, max_workers);
  if (scheduler_ && max_workers > 0) {
    // Limit the number of workers we enqueue to the most workers we could use,
    // and atomically claim up to that many idle workers so concurrent enqueues
    // never over-subscribe the thread pool or drive `idle_workers_` negative.
    const int desired_workers =
        static_cast<int>(std::min<size_t>(max_workers, n));
    int idle = idle_workers_.load(std::memory_order_relaxed);
    int workers_to_schedule = 0;
    while (idle > 0 && desired_workers > 0) {
      workers_to_schedule = std::min(desired_workers, idle);
      if (idle_workers_.compare_exchange_weak(idle, idle - workers_to_schedule,
                                              std::memory_order_acq_rel,
                                              std::memory_order_relaxed)) {
        break;
      }
      workers_to_schedule = 0;
    }
    for (int32_t i = 0; i < workers_to_schedule; ++i) {
      // Note that here, every worker is identical, so we can re-use the same
      // context for all scheduled tasks!
      scheduler_->schedule(scheduler_context_, this, [](void* context) {
        auto pool = reinterpret_cast<slinky_thread_pool*>(context);
        pool->impl_.work_until_idle();
        pool->idle_workers_.fetch_add(1, std::memory_order_acq_rel);
      });
    }
  }
  return result;
}

void slinky_thread_pool::wait_for(task* t) { impl_.wait_for(t); }

void slinky_thread_pool::wait_for(predicate_ref condition) {
  impl_.wait_for(condition);
}

void slinky_thread_pool::atomic_call(slinky::function_ref<void()> t) {
  impl_.atomic_call(t);
}

}  // namespace ynn
