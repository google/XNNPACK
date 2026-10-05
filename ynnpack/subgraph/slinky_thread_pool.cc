// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/slinky_thread_pool.h"

#include <algorithm>
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
  num_threads_ = scheduler_ ? scheduler_->num_threads(scheduler_context_) : 0;
  idle_workers_ = num_threads_;
  impl_.expect_workers(idle_workers_);
}

slinky_thread_pool::~slinky_thread_pool() {
  // We should wait until all our scheduled workers run before returning. If we
  // don't do this, the workers could get executed (and access this object)
  // after this destructor runs.
  // TODO: Try to find a way to do this without spinning, and without adding
  // overhead to the steady state (b/429222328).
  while (idle_workers_.load() < num_threads_) {
    std::this_thread::yield();
  }
}

int slinky_thread_pool::thread_count() const { return num_threads_; }

slinky::ref_count<slinky::thread_pool::task> slinky_thread_pool::enqueue(
    size_t n, task_body t, int max_workers) {
  auto result = impl_.enqueue(n, t, max_workers);
  if (scheduler_) {
    // Claim idle workers for this enqueue, so we know how many workers to
    // schedule. We account for the new workers here, before they actually
    // exist, so that concurrent enqueues don't schedule many workers based on
    // the same idle workers.

    // Limit the number of workers we enqueue to the most workers we could use,
    // and the number of threads in the thread pool. Since these workers are
    // generic, we don't want more live workers than there are threads.
    max_workers = std::min<size_t>(max_workers, n);

    // This is a saturating atomic subtraction: claim min(max_workers,
    // idle_workers_) workers, never claiming a negative number. The invariant
    // we must maintain is that `idle_workers_` plus the number of scheduled
    // workers that have not yet finished is equal to `num_threads_`, which the
    // destructor relies on to wait for all scheduled workers. A naive
    // `idle_workers_ -= max_workers` with `max_workers` computed from a
    // transiently negative `idle_workers_` would *add* to the counter without
    // scheduling anything, breaking this invariant.
    int claimed = 0;
    int idle = idle_workers_.load();
    do {
      claimed = std::min<int>(max_workers, idle);
      if (claimed <= 0) {
        claimed = 0;
        break;
      }
    } while (!idle_workers_.compare_exchange_weak(idle, idle - claimed));

    for (int i = 0; i < claimed; ++i) {
      // Note that here, every worker is identical, so we can re-use the same
      // context for all scheduled tasks!
      scheduler_->schedule(scheduler_context_, this, [](void* context) {
        auto pool = reinterpret_cast<slinky_thread_pool*>(context);
        pool->impl_.work_until_idle();
        // This must be the last access to `pool`, the destructor may return as
        // soon as this is incremented.
        ++pool->idle_workers_;
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
