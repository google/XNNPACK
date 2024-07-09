// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>  // For uintptr_t.
#include <cstring>  // For memcpy.
#include <string>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/cache.h"

static void* cache_end(const xnn_code_cache* cache) {
  return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(cache->cache.code.start) + cache->cache.code.size);
}

static void write_code(xnn_code_cache* cache, const std::string& str) {
  ASSERT_GE(cache->cache.code.capacity - cache->cache.code.size, str.length());
  std::memcpy(cache_end(cache), str.data(), str.length());
  cache->cache.code.size += str.length();
};

TEST(CODE_CACHE, init_and_release)
{
  xnn_initialize(/*allocator=*/nullptr);
  xnn_code_cache cache;
  EXPECT_EQ(xnn_status_success, xnn_init_code_cache(&cache));
  EXPECT_EQ(xnn_status_success, xnn_release_code_cache(&cache));
}


TEST(CODE_CACHE, release_null)
{
  EXPECT_EQ(xnn_status_success, xnn_release_code_cache(NULL));
}

TEST(CODE_CACHE, get_or_insert)
{
  xnn_initialize(/*allocator=*/nullptr);
  xnn_code_cache cache;
  EXPECT_EQ(xnn_status_success, xnn_init_code_cache(&cache));

  write_code(&cache, "1234");
  ASSERT_EQ(0, xnn_get_or_insert_code_cache(&cache, cache.cache.code.start, 4));
  ASSERT_EQ(0, cache.cache.hits);
  ASSERT_EQ(1, cache.cache.misses);

  void* span2_code = cache_end(&cache);
  // Simulate a cache hit.
  write_code(&cache, "1234");
  ASSERT_EQ(0, xnn_get_or_insert_code_cache(&cache, span2_code, 4));
  ASSERT_EQ(1, cache.cache.hits);
  ASSERT_EQ(1, cache.cache.misses);

  void* span3_code = cache_end(&cache);
  // Simulate a cache miss.
  write_code(&cache, "5678");
  ASSERT_EQ(4, xnn_get_or_insert_code_cache(&cache, span3_code, 4));
  ASSERT_EQ(1, cache.cache.hits);
  ASSERT_EQ(2, cache.cache.misses);
  ASSERT_EQ(2, cache.cache.num_entries);

  EXPECT_EQ(xnn_status_success, xnn_release_code_cache(&cache));
}

TEST(CODE_CACHE, grow) {
  xnn_initialize(/*allocator=*/nullptr);
  xnn_code_cache cache;
  EXPECT_EQ(xnn_status_success, xnn_init_code_cache(&cache));
  size_t old_num_buckets = cache.cache.num_buckets;
  for (size_t i = 0, expected_offset = 0; i < old_num_buckets; i++) {
    // Add many entries to force cache to grow.
    const std::string s = std::to_string(i);
    // write_code will update cache size, so get the code offset first.
    void* code_ptr = cache_end(&cache);
    write_code(&cache, s);
    ASSERT_EQ(expected_offset, xnn_get_or_insert_code_cache(&cache, code_ptr, s.length()));
    expected_offset += s.length();
  }

  ASSERT_EQ(0, cache.cache.hits);
  ASSERT_EQ(old_num_buckets, cache.cache.num_entries);
  // Check that cache has grown.
  ASSERT_LT(old_num_buckets, cache.cache.num_buckets);
  // Check that all the entries are still in cache.
  for (size_t i = 0, expected_offset = 0; i < old_num_buckets; i++) {
    const std::string s = std::to_string(i);
    // write_code will update cache size, so get the code offset first.
    void* code_ptr = cache_end(&cache);
    write_code(&cache, s);
    ASSERT_EQ(expected_offset, xnn_get_or_insert_code_cache(&cache, code_ptr, s.length()));
    expected_offset += s.length();
  }
  // And now all of the lookups should be cache hits.
  ASSERT_EQ(old_num_buckets, cache.cache.hits);

  EXPECT_EQ(xnn_status_success, xnn_release_code_cache(&cache));
}
