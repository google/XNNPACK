// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/slinky_thread_pool.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
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

TEST(thread_pool, concurrent_nested_enqueue_and_destroy) {
  struct LifecycleScheduler {
    explicit LifecycleScheduler(int thread_count) : inner(thread_count) {}

    static int num_threads_impl(void* self) {
      auto* s = static_cast<LifecycleScheduler*>(self);
      return TestScheduler::num_threads_impl(&s->inner);
    }

    static void schedule_impl(void* self, void* context,
                              void (*task)(void* context)) {
      auto* s = static_cast<LifecycleScheduler*>(self);
      struct TaskCall {
        std::shared_ptr<std::atomic<bool>> pool_destroyed;
        void* context;
        void (*task)(void*);
      };
      auto* call = new TaskCall{s->pool_destroyed, context, task};
      TestScheduler::schedule_impl(&s->inner, call, [](void* raw) {
        std::unique_ptr<TaskCall> c(static_cast<TaskCall*>(raw));
        EXPECT_FALSE(c->pool_destroyed->load(std::memory_order_acquire));
        c->task(c->context);
      });
    }

    static const ynn_scheduler* scheduler() {
      static const ynn_scheduler s = {num_threads_impl, schedule_impl};
      return &s;
    }

    TestScheduler inner;
    std::shared_ptr<std::atomic<bool>> pool_destroyed;
  };

  // Use outer_size < num_threads so idle_workers_ > 0 when multiple outer
  // iterations concurrently enqueue inner parallel loops.
  static constexpr int num_threads = 32;
  static constexpr size_t outer_size = 8;
  static constexpr size_t inner_size = 16;

  LifecycleScheduler scheduler(num_threads);
  const bool can_synchronize = LifecycleScheduler::num_threads_impl(
                                   &scheduler) >= static_cast<int>(outer_size);

  for (int iter = 0; iter < 200; ++iter) {
    auto pool_destroyed = std::make_shared<std::atomic<bool>>(false);
    scheduler.pool_destroyed = pool_destroyed;
    auto thread_pool =
        std::make_unique<slinky_thread_pool>(scheduler.scheduler(), &scheduler);

    std::array<std::atomic<int32_t>, inner_size> data = {{0}};
    for (int round = 0; round < 20; ++round) {
      std::atomic<size_t> started{0};
      thread_pool->parallel_for(outer_size, [&](size_t) {
        if (can_synchronize) {
          started.fetch_add(1, std::memory_order_acq_rel);
          while (started.load(std::memory_order_acquire) < outer_size) {
            std::this_thread::yield();
          }
        }
        thread_pool->parallel_for(inner_size, [&](size_t j) { data[j]++; });
      });
    }

    thread_pool.reset();
    pool_destroyed->store(true, std::memory_order_release);

    for (size_t j = 0; j < inner_size; ++j) {
      EXPECT_EQ(data[j], static_cast<int32_t>(20 * outer_size))
          << "at index " << j;
    }
  }
}

}  // namespace ynn
