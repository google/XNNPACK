// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint> // For uintptr_t.
#include <cstring> // For memcpy.

#include <xnnpack.h>
#include <xnnpack/cache.h>

#include <gtest/gtest.h>

static void* cache_end(const xnn_weights_cache* cache) {
  return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(cache->cache.weights.start) + cache->cache.weights.size);
}

static void write_weights(xnn_weights_cache* cache, const std::string& str) {
  ASSERT_GE(cache->cache.weights.capacity - cache->cache.weights.size, str.length());
  std::memcpy(cache_end(cache), str.data(), str.length());
};

TEST(WEIGHTS_CACHE, init_and_release)
{
  xnn_initialize(/*allocator=*/nullptr);
  struct xnn_weights_cache cache;
  EXPECT_EQ(xnn_status_success, xnn_init_weights_cache(&cache));
  EXPECT_EQ(xnn_status_success, xnn_release_weights_cache(&cache));
}

TEST(WEIGHTS_CACHE, release_null)
{
  EXPECT_EQ(xnn_status_success, xnn_release_weights_cache(NULL));
}

TEST(WEIGHTS_CACHE, get_or_insert)
{
  xnn_initialize(/*allocator=*/nullptr);
  struct xnn_weights_cache cache;
  EXPECT_EQ(xnn_status_success, xnn_init_weights_cache(&cache));

  write_weights(&cache, "1234");
  ASSERT_EQ(0, xnn_get_or_insert_weights_cache(&cache, cache.cache.weights.start, 4));
  ASSERT_EQ(0, cache.cache.hits);
  ASSERT_EQ(1, cache.cache.misses);
  ASSERT_EQ(4, cache.cache.weights.size);

  void* span2_weights = cache_end(&cache);
  // Simulate a cache hit.
  write_weights(&cache, "1234");
  ASSERT_EQ(0, xnn_get_or_insert_weights_cache(&cache, span2_weights, 4));
  ASSERT_EQ(1, cache.cache.hits);
  ASSERT_EQ(1, cache.cache.misses);
  ASSERT_EQ(4, cache.cache.weights.size);

  void* span3_weights = cache_end(&cache);
  // Simulate a cache miss.
  write_weights(&cache, "5678");
  ASSERT_EQ(4, xnn_get_or_insert_weights_cache(&cache, span3_weights, 4));
  ASSERT_EQ(1, cache.cache.hits);
  ASSERT_EQ(2, cache.cache.misses);
  ASSERT_EQ(2, cache.cache.num_entries);
  ASSERT_EQ(8, cache.cache.weights.size);

  EXPECT_EQ(xnn_status_success, xnn_release_weights_cache(&cache));
}

TEST(WEIGHTS_CACHE, grow) {
  xnn_initialize(/*allocator=*/nullptr);
  xnn_weights_cache cache;
  EXPECT_EQ(xnn_status_success, xnn_init_weights_cache(&cache));
  size_t old_num_buckets = cache.cache.num_buckets;
  for (size_t i = 0, expected_offset = 0; i < old_num_buckets; i++) {
    // Add many entries to force cache to grow.
    const std::string s = std::to_string(i);
    write_weights(&cache, s);
    ASSERT_EQ(expected_offset, xnn_get_or_insert_weights_cache(&cache, cache_end(&cache), s.length()));
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
    ASSERT_EQ(expected_offset, xnn_get_or_insert_weights_cache(&cache, cache_end(&cache), s.length()));
    expected_offset += s.length();
  }
  // And now all of the lookups should be cache hits.
  ASSERT_EQ(old_num_buckets, cache.cache.hits);

  EXPECT_EQ(xnn_status_success, xnn_release_weights_cache(&cache));
}

TEST(WEIGHTS_MEMORY, allocate_and_release) {
  xnn_weights_buffer b;
  ASSERT_EQ(xnn_status_success, xnn_allocate_weights_memory(&b, XNN_DEFAULT_WEIGHTS_BUFFER_SIZE));
  ASSERT_EQ(xnn_status_success, xnn_release_weights_memory(&b));
}

TEST(WEIGHTS_MEMORY, grow) {
  xnn_weights_buffer b;
  ASSERT_EQ(xnn_status_success, xnn_allocate_weights_memory(&b, 8));

  std::string junk = "1234";
  std::memcpy(b.start, junk.data(), junk.length());
  b.size += junk.length();
  ASSERT_EQ(b.size, 4);
  const uintptr_t old_weights = reinterpret_cast<uintptr_t>(b.start);

  // This should be a no-op, since we have enough space.
  ASSERT_EQ(xnn_status_success, xnn_reserve_weights_memory(&b, 4));
  ASSERT_EQ(old_weights, reinterpret_cast<uintptr_t>(b.start));

  // Copy 4 more bytes, now we are full.
  memcpy(b.start, junk.data(), junk.length());
  b.size += junk.length();

  const size_t old_size = b.size;
  ASSERT_EQ(xnn_status_success, xnn_reserve_weights_memory(&b, 4));

  // After growing, the new capacity should be bigger than the old one.
  ASSERT_EQ(12, b.capacity);
  // At least 4 bytes free.
  ASSERT_GE(b.capacity, b.size + 4);
  // But size stays the same.
  ASSERT_EQ(old_size, b.size);

  ASSERT_EQ(xnn_status_success, xnn_release_weights_memory(&b));
}
