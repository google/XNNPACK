// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint> // For uintptr_t.
#include <cstring> // For memcpy.

#include <xnnpack.h>
#include <xnnpack/codecache.h>

#include <gtest/gtest.h>

static void* cache_end(const xnn_weights_cache* cache) {
  return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(cache->cache.weights.start) + cache->cache.weights.size);
}

static void write_weights(xnn_weights_cache* cache, const std::string& str) {
  ASSERT_GE(cache->cache.weights.capacity - cache->cache.weights.size, str.length());
  std::memcpy(cache_end(cache), str.data(), str.length());
  cache->cache.weights.size += str.length();
};

TEST(WEIGHTS_CACHE, init_and_release)
{
  xnn_initialize(/*allocator=*/nullptr);
  struct xnn_weights_cache cache;
  EXPECT_EQ(xnn_status_success, xnn_init_weights_cache(&cache));
  EXPECT_EQ(xnn_status_success, xnn_release_weights_cache(&cache));
}

TEST(WEIGHTS_CACHE, get_or_insert)
{
  xnn_initialize(/*allocator=*/nullptr);
  struct xnn_weights_cache cache;
  EXPECT_EQ(xnn_status_success, xnn_init_weights_cache(&cache));

  write_weights(&cache, "1234");
  const xnn_byte_span span1 = {
    .start = cache.cache.code.start,
    .size = 4
  };
  ASSERT_EQ(0, xnn_get_or_insert_weights_cache(&cache, span1));
  ASSERT_EQ(0, cache.cache.hits);
  ASSERT_EQ(1, cache.cache.misses);

  void* span2_weights = cache_end(&cache);
  // Simulate a cache hit.
  write_weights(&cache, "1234");
  const xnn_byte_span span2 = {
    .start = span2_weights,
    .size = 4
  };
  ASSERT_EQ(0, xnn_get_or_insert_weights_cache(&cache, span2));
  ASSERT_EQ(1, cache.cache.hits);
  ASSERT_EQ(1, cache.cache.misses);

  void* span3_weights = cache_end(&cache);
  // Simulate a cache miss.
  write_weights(&cache, "5678");
  const xnn_byte_span span3 = {
    .start = span3_weights,
    .size = 4
  };
  ASSERT_EQ(4, xnn_get_or_insert_weights_cache(&cache, span3));
  ASSERT_EQ(1, cache.cache.hits);
  ASSERT_EQ(2, cache.cache.misses);
  ASSERT_EQ(2, cache.cache.num_entries);

  EXPECT_EQ(xnn_status_success, xnn_release_weights_cache(&cache));
}
