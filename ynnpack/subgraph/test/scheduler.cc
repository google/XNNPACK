#include "ynnpack/subgraph/test/scheduler.h"

#include "ynnpack/include/ynnpack.h"
#include "slinky/base/thread_pool_impl.h"

namespace ynn {

TestScheduler::TestScheduler(int thread_count) : impl_(thread_count) {}

TestScheduler::~TestScheduler() { impl_.work_until_idle(); }

int TestScheduler::num_threads_impl(void* self) {
  return reinterpret_cast<TestScheduler*>(self)->impl_.thread_count();
}

void TestScheduler::schedule_impl(void* self, void* context,
                                  void (*task)(void* context)) {
  reinterpret_cast<TestScheduler*>(self)->impl_.enqueue(
      [task, context]() { (*task)(context); });
}

const ynn_scheduler* TestScheduler::scheduler() {
  static const ynn_scheduler s = {num_threads_impl, schedule_impl};
  return &s;
}

}  // namespace ynn
