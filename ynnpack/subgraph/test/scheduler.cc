#include "ynnpack/subgraph/test/scheduler.h"

#include "ynnpack/include/ynnpack.h"
#include "slinky/base/thread_pool_impl.h"

namespace ynn {

// Emscripten behavior:
// - Without pthreads, we must not instantiate `std::thread` objects because the
// target
//   does not natively support threading, otherwise it aborts. Thus, thread pool
//   size drops to 0.
// - With pthreads, we limit the thread pool to 8 to avoid deadlocks: tests
// invoke pthread_create
//   synchronously, and we must match the pre-allocated PTHREAD_POOL_SIZE linked
//   during build.
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
TestScheduler::TestScheduler(int) : impl_(0) {}
#elif defined(__EMSCRIPTEN__) && defined(__EMSCRIPTEN_PTHREADS__)
TestScheduler::TestScheduler(int thread_count)
    : impl_(std::min(thread_count, 8)) {}
#else
TestScheduler::TestScheduler(int thread_count) : impl_(thread_count) {}
#endif

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
