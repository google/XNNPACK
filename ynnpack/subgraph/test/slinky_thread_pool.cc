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
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/include/ynnpack.h"
#include "external/+_repo_rules+slinky/base/thread_pool.h"
#include "external/+_repo_rules+slinky/base/thread_pool_impl.h"

namespace ynn {

// This code looks silly: we make a slinky thread pool, wrap it in
// xnn_scheduler, just to make it look like slinky::thread_pool. We
// just need any old thread pool with a `schedule(std::function<void()>)`
// function, and we want to avoid new dependencies.
class TestThreadPool {
 public:
  explicit TestThreadPool(int n) : impl_(n) {}

  static int num_threads_impl(void* self) {
    return reinterpret_cast<TestThreadPool*>(self)->impl_.thread_count();
  }

  static void schedule_impl(void* self, void* context,
                            void (*task)(void* context)) {
    reinterpret_cast<TestThreadPool*>(self)->impl_.thread_pool::enqueue(
        [task, context]() { (*task)(context); });
  }

  static const ynn_scheduler* scheduler() {
    static const ynn_scheduler scheduler = {
        TestThreadPool::num_threads_impl, TestThreadPool::schedule_impl};
    return &scheduler;
  }

 private:
  slinky::thread_pool_impl impl_;
};

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
  std::unique_ptr<TestThreadPool> threads = std::make_unique<TestThreadPool>(3);
  slinky_thread_pool thread_pool(threads->scheduler(), threads.get());

  static constexpr size_t size = 10000;

  std::vector<int32_t> data(size, 0);
  auto inc = [&](size_t i) { data[i]++; };

  thread_pool.parallel_for(size, inc);

  std::vector<int32_t> expected(size, 1);
  EXPECT_EQ(data, expected);
}

TEST(thread_pool, loop_chain) {
  std::unique_ptr<TestThreadPool> threads = std::make_unique<TestThreadPool>(3);
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
  std::unique_ptr<TestThreadPool> threads = std::make_unique<TestThreadPool>(3);
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

}  // namespace ynn
