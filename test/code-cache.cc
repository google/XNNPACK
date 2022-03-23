// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint> // For uintptr_t.
#include <cstring> // For memcpy.

#include <xnnpack.h>
#include <xnnpack/codecache.h>

#include <gtest/gtest.h>

static void* cache_end(const xnn_code_cache* cache) {
  return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(cache->cache.code.start) + cache->cache.code.size);
}

static void write_code(xnn_code_cache* cache, const std::string& str) {
  ASSERT_GE(cache->cache.code.capacity - cache->cache.code.size, str.length());
  std::memcpy(cache_end(cache), str.data(), str.length());
  cache->cache.code.size += str.length();
};

TEST(JIT_CACHE, init_and_release)
{
  xnn_initialize(/*allocator=*/nullptr);
  xnn_code_cache cache;
  EXPECT_EQ(xnn_status_success, xnn_init_code_cache(&cache));
  EXPECT_EQ(xnn_status_success, xnn_release_code_cache(&cache));
}


TEST(JIT_CACHE, release_null)
{
  EXPECT_EQ(xnn_status_success, xnn_release_code_cache(NULL));
}

TEST(JIT_CACHE, get_or_insert)
{
  xnn_initialize(/*allocator=*/nullptr);
  xnn_code_cache cache;
  EXPECT_EQ(xnn_status_success, xnn_init_code_cache(&cache));

  write_code(&cache, "1234");
  const xnn_byte_span span1 = {
    .start = cache.cache.code.start,
    .size = 4
  };
  ASSERT_EQ(0, xnn_code_cache_get_or_insert(&cache, span1));
  ASSERT_EQ(0, cache.cache.hits);
  ASSERT_EQ(1, cache.cache.misses);

  void* span2_code = cache_end(&cache);
  // Simulate a cache hit.
  write_code(&cache, "1234");
  const xnn_byte_span span2 = {
    .start = span2_code,
    .size = 4
  };
  ASSERT_EQ(0, xnn_code_cache_get_or_insert(&cache, span2));
  ASSERT_EQ(1, cache.cache.hits);
  ASSERT_EQ(1, cache.cache.misses);

  void* span3_code = cache_end(&cache);
  // Simulate a cache miss.
  write_code(&cache, "5678");
  const xnn_byte_span span3 = {
    .start = span3_code,
    .size = 4
  };
  ASSERT_EQ(4, xnn_code_cache_get_or_insert(&cache, span3));
  ASSERT_EQ(1, cache.cache.hits);
  ASSERT_EQ(2, cache.cache.misses);
  ASSERT_EQ(2, cache.cache.num_entries);

  EXPECT_EQ(xnn_status_success, xnn_release_code_cache(&cache));
}
