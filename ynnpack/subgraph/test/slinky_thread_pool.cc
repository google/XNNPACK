// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/slinky_thread_pool.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/subgraph/test/scheduler.h"

namespace ynn {

TEST(thread_pool, inline_scheduling) {
  slinky_thread_pool thread_pool(nullptr, nullptr);
  static constexpr size_t size = 10000;

  std::vector<int32_t> data(size, 0);
  auto inc = [&](size_t i) { data[i]++; };

  thread_pool.parallel_for(size, inc);

  std::vector<int32_t> expected(size, 1);
  EXPECT_EQ(data, expected);
}

TEST(thread_pool, single_loop) {
  auto threads = std::make_unique<TestScheduler>(3);
  slinky_thread_pool thread_pool(threads->scheduler(), threads.get());

  static constexpr size_t size = 10000;

  std::vector<int32_t> data(size, 0);
  auto inc = [&](size_t i) { data[i]++; };

  thread_pool.parallel_for(size, inc);

  std::vector<int32_t> expected(size, 1);
  EXPECT_EQ(data, expected);
}

TEST(thread_pool, loop_chain) {
  auto threads = std::make_unique<TestScheduler>(3);
  slinky_thread_pool thread_pool(threads->scheduler(), threads.get());

  static constexpr size_t size = 10000;

  std::vector<int32_t> data(size, 0);
  auto inc = [&](size_t i) { data[i]++; };

  thread_pool.parallel_for(size, inc);
  thread_pool.parallel_for(size, inc);
  thread_pool.parallel_for(size, inc);
  thread_pool.parallel_for(size, inc);
  thread_pool.parallel_for(size, inc);

  std::vector<int32_t> expected(size, 5);
  EXPECT_EQ(data, expected);
}

TEST(thread_pool, nested_loops) {
  auto threads = std::make_unique<TestScheduler>(3);
  slinky_thread_pool thread_pool(threads->scheduler(), threads.get());

  static constexpr size_t size = 100;

  std::array<std::atomic<int32_t>, size> data = {{0}};
  auto inc = [&](size_t i) { data[i]++; };

  thread_pool.parallel_for(
      size, [&](size_t i) { thread_pool.parallel_for(size, inc); });

  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(data[i], size);
  }
}

// The following test needs more real threads than WASM supports (by default
// WASM has no thread support at all, and with pthreads the thread pool is
// pre-allocated with a fixed, small size).
#if defined(__EMSCRIPTEN__)
TEST(thread_pool, destructor_waits_for_scheduled_workers) {}
#else

namespace {

// A scheduler that can be switched into a mode where scheduled tasks are held
// back, and then released one at a time. This lets a test control exactly when
// scheduled workers run relative to other events (e.g. destroying the pool).
class HoldingScheduler {
 public:
  explicit HoldingScheduler(int thread_count) : impl_(thread_count) {}
  ~HoldingScheduler() {
    release(held());
    wait_for_in_flight(0);
    impl_.work_until_idle();
  }

  static int num_threads_impl(void* self) {
    return reinterpret_cast<HoldingScheduler*>(self)->impl_.thread_count();
  }

  static void schedule_impl(void* self, void* context,
                            void (*task)(void* context)) {
    HoldingScheduler* scheduler = reinterpret_cast<HoldingScheduler*>(self);
    std::lock_guard<std::mutex> lock(scheduler->mutex_);  // NOLINT(build/c++11)
    ++scheduler->in_flight_;
    if (scheduler->hold_) {
      scheduler->held_.push_back({task, context});
    } else {
      scheduler->run(task, context);
    }
  }

  static const ynn_scheduler* scheduler() {
    static const ynn_scheduler s = {num_threads_impl, schedule_impl};
    return &s;
  }

  // Start holding back newly scheduled tasks.
  void hold() {
    std::lock_guard<std::mutex> lock(mutex_);  // NOLINT(build/c++11)
    hold_ = true;
  }

  // The number of tasks currently held back.
  size_t held() {
    std::lock_guard<std::mutex> lock(mutex_);  // NOLINT(build/c++11)
    return held_.size();
  }

  // Run up to `count` held tasks.
  void release(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);  // NOLINT(build/c++11)
    for (size_t i = 0; i < count && !held_.empty(); ++i) {
      HeldTask t = held_.front();
      held_.erase(held_.begin());
      run(t.task, t.context);
    }
  }

  // Wait until the number of scheduled tasks that have not finished (including
  // held tasks) is `count`.
  void wait_for_in_flight(int count) {
    while (in_flight_.load() != count) {
      std::this_thread::yield();
    }
  }

 private:
  struct HeldTask {
    void (*task)(void* context);
    void* context;
  };

  void run(void (*task)(void* context), void* context) {
    impl_.enqueue([this, task, context]() {
      (*task)(context);
      --in_flight_;
    });
  }

  slinky::thread_pool_impl impl_;
  std::mutex mutex_;  // NOLINT(build/c++11)
  bool hold_ = false;
  std::vector<HeldTask> held_;
  std::atomic<int> in_flight_{0};
};

}  // namespace

// `~slinky_thread_pool` must not return while any worker it scheduled has not
// finished, otherwise the worker accesses the destroyed pool. The number of
// outstanding workers is tracked by `idle_workers_`, which is updated by
// concurrent `enqueue` calls (nested loops enqueue from worker threads). This
// test stresses that accounting, and then checks that the destructor waits
// for the last scheduled worker.
TEST(thread_pool, destructor_waits_for_scheduled_workers) {
  constexpr int kThreads = 8;
  HoldingScheduler scheduler(kThreads);
  auto thread_pool = std::make_unique<slinky_thread_pool>(
      HoldingScheduler::scheduler(), &scheduler);

  // Hammer the pool with concurrent nested loops from several threads.
  std::atomic<int64_t> sink{0};
  auto body = [&](size_t i) { sink.fetch_add(i, std::memory_order_relaxed); };
  std::vector<std::thread> callers;
  for (int t = 0; t < 4; ++t) {
    callers.emplace_back([&]() {
      for (int i = 0; i < 500; ++i) {
        thread_pool->parallel_for(
            16, [&](size_t) { thread_pool->parallel_for(16, body); });
      }
    });
  }
  for (std::thread& t : callers) t.join();
  // Let any workers scheduled by the above finish.
  scheduler.wait_for_in_flight(0);

  // Now schedule workers, but hold them back. The calling thread does all the
  // work itself, so this returns with the scheduled workers still pending.
  scheduler.hold();
  thread_pool->parallel_for(1024, body);
  const size_t held = scheduler.held();
  ASSERT_GT(held, 0);
  // The pool should not schedule more workers than it has threads. If it did,
  // its accounting of idle workers has drifted.
  EXPECT_LE(held, kThreads);

  // Destroy the pool on another thread. It must block until all held workers
  // have run.
  std::atomic<bool> destroyed{false};
  std::thread destroyer([&]() {
    thread_pool.reset();
    destroyed.store(true);
  });

  // Release all but one of the held workers, and wait for them to finish.
  scheduler.release(held - 1);
  scheduler.wait_for_in_flight(1);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(destroyed.load())
      << "~slinky_thread_pool returned with a scheduled worker still pending";

  // Release the last worker, now the destructor can finish.
  scheduler.release(1);
  destroyer.join();
  EXPECT_TRUE(destroyed.load());
}

#endif  // defined(__EMSCRIPTEN__)

}  // namespace ynn
