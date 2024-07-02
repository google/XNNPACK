// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>  // For std::rotate.
#include <cstdint>    // For uintptr_t.
#include <cstring>    // For memcpy.
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/cache.h"
#include "xnnpack/common.h"
#include "xnnpack/memory.h"

static void* cache_end(const struct xnn_internal_weights_cache* cache) {
  return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(cache->cache.weights.start) + cache->cache.weights.size);
}

static void write_weights(struct xnn_internal_weights_cache* cache, const std::string& str) {
  ASSERT_NE(nullptr, xnn_internal_reserve_space_in_weights_cache(cache, str.length()));
  std::memcpy(cache_end(cache), str.data(), str.length());
};

TEST(WEIGHTS_CACHE, init_and_release)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  struct xnn_internal_weights_cache cache;
  EXPECT_EQ(xnn_status_success, xnn_internal_init_weights_cache_with_size(&cache, XNN_DEFAULT_WEIGHTS_BUFFER_SIZE));
  EXPECT_EQ(xnn_status_success, xnn_internal_release_weights_cache(&cache));
}

TEST(WEIGHTS_CACHE, init_with_size_and_release)
{
  constexpr size_t four_mb = 4194304;
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  struct xnn_internal_weights_cache cache;
  EXPECT_EQ(xnn_status_success, xnn_internal_init_weights_cache_with_size(&cache, four_mb));
  // Allocation can be rounded up to alignment, so check GE instead of EQ.
  ASSERT_GE(cache.cache.weights.capacity, four_mb);
  EXPECT_EQ(xnn_status_success, xnn_internal_release_weights_cache(&cache));
}

TEST(WEIGHTS_CACHE, release_null)
{
  EXPECT_EQ(xnn_status_success, xnn_internal_release_weights_cache(nullptr));
}

TEST(WEIGHTS_CACHE, get_or_insert)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  struct xnn_internal_weights_cache cache;
  EXPECT_EQ(xnn_status_success, xnn_internal_init_weights_cache_with_size(&cache, XNN_DEFAULT_WEIGHTS_BUFFER_SIZE));

  write_weights(&cache, "1234");
  ASSERT_EQ(0, xnn_internal_get_or_insert_weights_cache(&cache, nullptr, cache.cache.weights.start, 4));
  ASSERT_EQ(0, cache.cache.hits);
  ASSERT_EQ(1, cache.cache.misses);
  ASSERT_EQ(4, cache.cache.weights.size);

  void* span2_weights = cache_end(&cache);
  // Simulate a cache hit.
  write_weights(&cache, "1234");
  ASSERT_EQ(0, xnn_internal_get_or_insert_weights_cache(&cache, nullptr, span2_weights, 4));
  ASSERT_EQ(1, cache.cache.hits);
  ASSERT_EQ(1, cache.cache.misses);
  ASSERT_EQ(4, cache.cache.weights.size);

  void* span3_weights = cache_end(&cache);
  // Simulate a cache miss.
  write_weights(&cache, "5678");
  ASSERT_EQ(4, xnn_internal_get_or_insert_weights_cache(&cache, nullptr, span3_weights, 4));
  ASSERT_EQ(1, cache.cache.hits);
  ASSERT_EQ(2, cache.cache.misses);
  ASSERT_EQ(2, cache.cache.num_entries);
  ASSERT_EQ(8, cache.cache.weights.size);

  EXPECT_EQ(xnn_status_success, xnn_internal_release_weights_cache(&cache));
}

TEST(WEIGHTS_CACHE, grow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  struct xnn_internal_weights_cache cache;
  EXPECT_EQ(xnn_status_success, xnn_internal_init_weights_cache_with_size(&cache, XNN_DEFAULT_WEIGHTS_BUFFER_SIZE));
  size_t old_num_buckets = cache.cache.num_buckets;
  for (size_t i = 0, expected_offset = 0; i < old_num_buckets; i++) {
    // Add many entries to force cache to grow.
    const std::string s = std::to_string(i);
    write_weights(&cache, s);
    ASSERT_EQ(expected_offset, xnn_internal_get_or_insert_weights_cache(&cache, nullptr, cache_end(&cache), s.length()));
    expected_offset += s.length();
  }

  ASSERT_EQ(0, cache.cache.hits);
  ASSERT_EQ(old_num_buckets, cache.cache.num_entries);
  // Check that cache has grown.
  ASSERT_LT(old_num_buckets, cache.cache.num_buckets);
  // Check that all the entries are still in cache.
  for (size_t i = 0, expected_offset = 0; i < old_num_buckets; i++) {
    const std::string s = std::to_string(i);
    write_weights(&cache, s);
    ASSERT_EQ(expected_offset, xnn_internal_get_or_insert_weights_cache(&cache, nullptr, cache_end(&cache), s.length()));
    expected_offset += s.length();
  }
  // And now all of the lookups should be cache hits.
  ASSERT_EQ(old_num_buckets, cache.cache.hits);

  EXPECT_EQ(xnn_status_success, xnn_internal_release_weights_cache(&cache));
}

TEST(WEIGHTS_MEMORY, allocate_and_release) {
  xnn_weights_buffer b;
  ASSERT_EQ(xnn_status_success, xnn_allocate_weights_memory(&b, XNN_DEFAULT_WEIGHTS_BUFFER_SIZE));
  ASSERT_EQ(xnn_status_success, xnn_release_weights_memory(&b));
}

TEST(WEIGHTS_MEMORY, grow) {
  xnn_weights_buffer b;
  ASSERT_EQ(xnn_status_success, xnn_allocate_weights_memory(&b, 8));
  // Allocations rounded to page size, so it might not be 8.
  size_t old_capacity = b.capacity;

  std::string junk = "1234";
  std::memcpy(b.start, junk.data(), junk.length());
  b.size += junk.length();
  ASSERT_EQ(b.size, 4);
  const uintptr_t old_weights = reinterpret_cast<uintptr_t>(b.start);

  // This should be a no-op, since we have enough space.
  ASSERT_EQ(xnn_status_success, xnn_reserve_weights_memory(&b, 4));
  ASSERT_EQ(old_weights, reinterpret_cast<uintptr_t>(b.start));

  // Simulate copying bytes until we are full.
  b.size += (old_capacity - b.size);

  const size_t old_size = b.size;
  ASSERT_EQ(xnn_status_success, xnn_reserve_weights_memory(&b, 4));

  // After growing, the new capacity should be bigger than the old one.
  ASSERT_LT(old_capacity, b.capacity);
  // At least 4 bytes free.
  ASSERT_GE(b.capacity, b.size + 4);
  // But size stays the same.
  ASSERT_EQ(old_size, b.size);

  // Check that after growing, the contents remain.
  std::string actual = std::string(static_cast<char*>(b.start), static_cast<char*>(b.start) + junk.length());
  ASSERT_EQ(junk, actual);

  ASSERT_EQ(xnn_status_success, xnn_release_weights_memory(&b));
}

// Checks for a bug in mremap using the wrong value for old_size.
TEST(WEIGHTS_MEMORY, grow_from_zero_size) {
  xnn_weights_buffer b;
  ASSERT_EQ(xnn_status_success, xnn_allocate_weights_memory(&b, 8));
  size_t old_capacity = b.capacity;
  ASSERT_GE(old_capacity, 0);
  // Allocate weights, but don't use it, so size is still 0.
  ASSERT_EQ(b.size, 0);

  // Reserve an absurd amount of memory to force growth.
  // It is not certain that we will grow (due to rounding to page size), but if
  // we do, this catches a bug where we pass the wrong arguments to mremap.
  size_t large = 32 * 1024 * 1024;  // 32MB
  ASSERT_EQ(xnn_status_success, xnn_reserve_weights_memory(&b, large));

  size_t new_capacity = b.capacity;
  EXPECT_GE(new_capacity, old_capacity);
}

TEST(WEIGHTS_CACHE, finalize_empty) {
  xnn_weights_buffer b;
  const size_t initial_capacity = 1024 * 1024;  // 1MB.
  ASSERT_EQ(xnn_status_success, xnn_allocate_weights_memory(&b, initial_capacity));

  ASSERT_EQ(0, b.size);
  ASSERT_EQ(initial_capacity, b.capacity);

  ASSERT_EQ(xnn_status_success, xnn_finalize_weights_memory(&b));
  ASSERT_EQ(0, b.size);
  ASSERT_EQ(0, b.capacity);

  ASSERT_EQ(xnn_status_success, xnn_release_weights_memory(&b));
}

TEST(WEIGHTS_CACHE, finalize) {
  xnn_weights_buffer b;
  const size_t initial_capacity = 1024 * 1024;  // 1MB.
  ASSERT_EQ(xnn_status_success, xnn_allocate_weights_memory(&b, initial_capacity));
  const size_t actual_capacity = b.capacity;

  const std::string junk = "1234";
  std::memcpy(b.start, junk.data(), junk.length());
  b.size += junk.length();
  ASSERT_EQ(4, b.size);

  ASSERT_EQ(xnn_status_success, xnn_finalize_weights_memory(&b));
  #if XNN_PLATFORM_WEB
    // Web does not support partial unmapping.
    ASSERT_EQ(actual_capacity, b.capacity);
  #else
    // The actual capacity depends on page size, since it is aligned, just check that it shrunk.
    ASSERT_GE(actual_capacity, b.capacity);
  #endif
  ASSERT_EQ(4, b.size);

  ASSERT_EQ(xnn_status_success, xnn_release_weights_memory(&b));
}

TEST(WEIGHTS_CACHE, finalize_twice) {
  xnn_weights_buffer b;
  ASSERT_EQ(xnn_status_success, xnn_allocate_weights_memory(&b, XNN_DEFAULT_WEIGHTS_BUFFER_SIZE));

  const std::string junk = "1234";
  std::memcpy(b.start, junk.data(), junk.length());
  b.size += junk.length();

  ASSERT_EQ(xnn_status_success, xnn_finalize_weights_memory(&b));
  const size_t capacity = b.capacity;
  // Finalizing twice does not error.
  ASSERT_EQ(xnn_status_success, xnn_finalize_weights_memory(&b));
  // Capacity does not change.
  ASSERT_EQ(capacity, b.capacity);
  ASSERT_EQ(4, b.size);

  ASSERT_EQ(xnn_status_success, xnn_release_weights_memory(&b));
}

TEST(WEIGHTS_CACHE, finalize_capacity_smaller_than_page_aligned_size) {
  xnn_weights_buffer b;
  // Small capacity that is smaller than page sizes on all platforms.
  ASSERT_EQ(xnn_status_success, xnn_allocate_weights_memory(&b, 8));

  const std::string junk = "1234";
  std::memcpy(b.start, junk.data(), junk.length());
  b.size += junk.length();
  ASSERT_EQ(xnn_status_success, xnn_finalize_weights_memory(&b));
  ASSERT_EQ(4, b.size);
  ASSERT_EQ(xnn_status_success, xnn_release_weights_memory(&b));
}

TEST(WEIGHTS_CACHE, write_many_cache_hits) {
#if XNN_PLATFORM_WEB && !defined(__EMSCRIPTEN_PTHREADS__)
  GTEST_SKIP();
#endif
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  struct xnn_internal_weights_cache cache;
  EXPECT_EQ(xnn_status_success, xnn_internal_init_weights_cache_with_size(&cache, XNN_DEFAULT_WEIGHTS_BUFFER_SIZE));
  const std::string weights = "0123456789abcdefghij";
  const size_t weights_size = weights.size();
  auto write = [&] {
    write_weights(&cache, weights);
    xnn_internal_get_or_insert_weights_cache(&cache, nullptr, cache_end(&cache), weights_size);
  };
  constexpr size_t num_threads = 20;
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (size_t i = 0; i < num_threads; i++) {
    threads.emplace_back(write);
  }
  for (size_t i = 0; i < num_threads; i++) {
    threads[i].join();
  }

  ASSERT_EQ(num_threads - 1, cache.cache.hits);
  ASSERT_EQ(1, cache.cache.num_entries);
  ASSERT_EQ(weights_size, cache.cache.weights.size);
  EXPECT_EQ(xnn_status_success, xnn_internal_release_weights_cache(&cache));
}

TEST(WEIGHTS_CACHE, write_many_cache_misses) {
#if XNN_PLATFORM_WEB && !defined(__EMSCRIPTEN_PTHREADS__)
  GTEST_SKIP();
#endif
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  struct xnn_internal_weights_cache cache;
  EXPECT_EQ(xnn_status_success, xnn_internal_init_weights_cache_with_size(&cache, XNN_DEFAULT_WEIGHTS_BUFFER_SIZE));
  const std::string weights = "0123456789abcdefghij";
  const size_t weights_size = weights.size();
  auto write = [&](size_t i) {
    std::string rotated_weights = weights;
    std::rotate(rotated_weights.begin(), rotated_weights.begin() + i,
                rotated_weights.end());
    write_weights(&cache, rotated_weights);
    xnn_internal_get_or_insert_weights_cache(&cache, nullptr, cache_end(&cache), weights_size);
  };
  constexpr size_t num_threads = 20;
  ASSERT_LE(num_threads, weights_size);
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (size_t i = 0; i < num_threads; i++) {
    threads.emplace_back(write, i);
  }
  for (size_t i = 0; i < num_threads; i++) {
    threads[i].join();
  }

  ASSERT_EQ(0, cache.cache.hits);
  ASSERT_EQ(num_threads, cache.cache.num_entries);
  ASSERT_EQ(weights_size * num_threads, cache.cache.weights.size);
  EXPECT_EQ(xnn_status_success, xnn_internal_release_weights_cache(&cache));
}

TEST(WEIGHTS_CACHE, operations_on_finalized_cache_hard) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  struct xnn_internal_weights_cache cache;
  ASSERT_EQ(xnn_status_success, xnn_internal_init_weights_cache_with_size(&cache, XNN_DEFAULT_WEIGHTS_BUFFER_SIZE));

  ASSERT_EQ(xnn_status_success, xnn_internal_finalize_weights_cache(&cache, xnn_weights_cache_finalization_kind_hard));
  // Finalizing a finalized cache is an error.
  ASSERT_NE(xnn_status_success, xnn_internal_finalize_weights_cache(&cache, xnn_weights_cache_finalization_kind_hard));
  // Trying to reserve is an error.
  ASSERT_EQ(nullptr, xnn_internal_reserve_space_in_weights_cache(&cache, 1));

  // We should not be able to insert into the weights cache, and also this shouldn't timeout by unlocking a mutex which
  // has not been locked (since xnn_internal_reserve_space_in_weights_cache above failed).
  ASSERT_EQ(XNN_CACHE_NOT_FOUND, xnn_internal_get_or_insert_weights_cache(&cache, nullptr, cache.cache.weights.start, 4));

  ASSERT_EQ(xnn_status_success, xnn_internal_release_weights_cache(&cache));
}

TEST(WEIGHTS_CACHE, operations_on_finalized_cache_soft) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  struct xnn_internal_weights_cache cache;
  ASSERT_EQ(xnn_status_success, xnn_internal_init_weights_cache_with_size(&cache, XNN_DEFAULT_WEIGHTS_BUFFER_SIZE));

  ASSERT_EQ(xnn_status_success, xnn_internal_finalize_weights_cache(&cache, xnn_weights_cache_finalization_kind_soft));
  // Finalizing a finalized cache is an error.
  ASSERT_NE(xnn_status_success, xnn_internal_finalize_weights_cache(&cache, xnn_weights_cache_finalization_kind_soft));
  // Trying to reserve too much is an error.
  ASSERT_EQ(nullptr, xnn_internal_reserve_space_in_weights_cache(&cache, cache.cache.weights.capacity + 1));

  ASSERT_EQ(xnn_status_success, xnn_internal_release_weights_cache(&cache));
}

TEST(WEIGHTS_CACHE, insert_into_finalized_cache_soft) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  struct xnn_internal_weights_cache cache;
  ASSERT_EQ(xnn_status_success, xnn_internal_init_weights_cache_with_size(&cache, XNN_DEFAULT_WEIGHTS_BUFFER_SIZE));

  write_weights(&cache, "1234");
  ASSERT_EQ(0, xnn_internal_get_or_insert_weights_cache(&cache, nullptr, cache.cache.weights.start, 4));
  ASSERT_EQ(xnn_status_success, xnn_internal_finalize_weights_cache(&cache, xnn_weights_cache_finalization_kind_soft));

  // Inserting into a finalized cache is okay as long as cache memory has space and it is a cache hit.
  ASSERT_LT(cache.cache.weights.size + 4, cache.cache.weights.capacity);
  write_weights(&cache, "1234");
  void* cached_weights = cache_end(&cache);
  ASSERT_EQ(0, xnn_internal_get_or_insert_weights_cache(&cache, nullptr, cached_weights, 4));
  ASSERT_EQ(4, cache.cache.weights.size);

  // Sufficient space, but Cache miss.
  write_weights(&cache, "4567");
  ASSERT_EQ(XNN_CACHE_NOT_FOUND, xnn_internal_get_or_insert_weights_cache(&cache, nullptr, cached_weights, 4));

  // Not enough space in the finalized weights cache.
  std::string big_string(cache.cache.weights.capacity, '5');
  // Don't use write_weights here as it asserts xnn_internal_reserve_space_in_weights_cache does not return nullptr.
  ASSERT_EQ(nullptr, xnn_internal_reserve_space_in_weights_cache(&cache, big_string.length()));

  ASSERT_EQ(xnn_status_success, xnn_internal_release_weights_cache(&cache));
}
