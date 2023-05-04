// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <random>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <xnnpack.h>
#include <xnnpack/common.h>
#include <xnnpack/mutex.h>

TEST(MUTEX, init_lock_unlock_destroy) {

  xnn_mutex m;
  ASSERT_EQ(xnn_status_success, xnn_mutex_init(&m));
  ASSERT_EQ(xnn_status_success, xnn_mutex_lock(&m));
  ASSERT_EQ(xnn_status_success, xnn_mutex_unlock(&m));
  ASSERT_EQ(xnn_status_success, xnn_mutex_destroy(&m));
}

TEST(MUTEX, counter) {
  // Skip if we are not targeting pthread.
#if XNN_PLATFORM_WEB && !defined(__EMSCRIPTEN_PTHREADS__)
  GTEST_SKIP();
#endif

  xnn_mutex m;
  constexpr size_t num_threads = 50;
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  volatile size_t counter = 0;

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto dist = std::uniform_int_distribution<int>(100, 200);

  ASSERT_EQ(xnn_status_success, xnn_mutex_init(&m));

  for (size_t i = 0; i < num_threads; i++) {
    threads.emplace_back(([&] () {
      ASSERT_EQ(xnn_status_success, xnn_mutex_lock(&m));
      std::this_thread::sleep_for(std::chrono::milliseconds(dist(rng)));
      counter += 1;
      ASSERT_EQ(xnn_status_success, xnn_mutex_unlock(&m));
    }));
  }

  for (int i = num_threads - 1; i >= 0; i--) {
    threads[i].join();
  }

  ASSERT_EQ(counter, num_threads);
  ASSERT_EQ(xnn_status_success, xnn_mutex_destroy(&m));
}
