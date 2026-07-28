#ifndef XNNPACK_YNNPACK_SUBGRAPH_TEST_SCHEDULER_H_
#define XNNPACK_YNNPACK_SUBGRAPH_TEST_SCHEDULER_H_

#include <atomic>
#include <cstddef>

#include "ynnpack/include/ynnpack.h"
#include "slinky/base/thread_pool_impl.h"

namespace ynn {

// This scheduler is only meant for use in tests.
class TestScheduler {
 public:
  explicit TestScheduler(int thread_count);
  ~TestScheduler();

  static int num_threads_impl(void* self);
  static void schedule_impl(void* self, void* context,
                            void (*task)(void* context));

  static const ynn_scheduler* scheduler();

  // The number of tasks scheduled so far. Allows tests to check that work was
  // actually parallelized.
  size_t task_count() const { return task_count_; }

 private:
  // To implement this scheduler, we use the slinky thread pool, but it could be
  // any thread pool.
  slinky::thread_pool_impl impl_;
  std::atomic<size_t> task_count_{0};
};

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_TEST_SCHEDULER_H_
