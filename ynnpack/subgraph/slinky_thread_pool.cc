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
    size_t n, task_body t, int32_t max_workers) {
  auto result = impl_.enqueue(n, t, max_workers);
  if (scheduler_) {
    // Atomically increment the expected workers, so we know how many workers
    // this enqueue should schedule.

    // Limit the number of workers we enqueue to the most workers we could use,
    // and the number of threads in the thread pool. Since these workers are
    // generic, we don't want more live workers than there are threads.
    max_workers = std::min<size_t>(max_workers, n);
    max_workers = std::min<int>(max_workers, idle_workers_);

    // The logic here is a bit racy, because the number of workers we compute we
    // need is based on this shared state that might be changed by another
    // thread. It seems best to account for the new workers here, before they
    // actually exist, so we don't end up enqueuing many instances based on
    // accounting for the same idle workers many times. Doing this here doesn't
    // fix this race, it just makes it less bad. Ideally we'd do it atomically,
    // unfortunately there is no saturating atomic add.
    idle_workers_ -= max_workers;
    for (int32_t i = 0; i < max_workers; ++i) {
      // Note that here, every worker is identical, so we can re-use the same
      // context for all scheduled tasks!
      scheduler_->schedule(scheduler_context_, this, [](void* context) {
        auto pool = reinterpret_cast<slinky_thread_pool*>(context);
        pool->impl_.work_until_idle();
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
