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

}  // namespace ynn
