// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint> // For uintptr_t.

#include <xnnpack.h>
#include <xnnpack/codecache.h>

#include <gtest/gtest.h>

static void write_string(xnn_code_cache* cache, std::string str) {
  ASSERT_GE(cache->code_buffer.capacity - cache->code_buffer.size, str.length());
  std::memcpy((void*) ((uintptr_t)cache->code_buffer.code + cache->code_buffer.size), str.data(), str.length());
  cache->code_buffer.size += str.length();
};

TEST(JIT_CACHE, init_and_release)
{
  xnn_initialize(/*allocator=*/nullptr);
  xnn_code_cache cache;
  EXPECT_EQ(xnn_status_success, xnn_init_code_cache(&cache));
  EXPECT_EQ(xnn_status_success, xnn_release_code_cache(&cache));
}

TEST(JIT_CACHE, get_or_insert)
{
  xnn_initialize(/*allocator=*/nullptr);
  xnn_code_cache cache;
  EXPECT_EQ(xnn_status_success, xnn_init_code_cache(&cache));

  write_string(&cache, "1234");
  const xnn_code_span span1 = {
    .code = cache.code_buffer.code,
    .size = 4
  };
  ASSERT_EQ(0, xnn_code_cache_get_or_insert(&cache, span1));
  ASSERT_EQ(0, cache.hits);
  ASSERT_EQ(1, cache.misses);

  void* span2_code = (void*) ((uintptr_t) cache.code_buffer.code + cache.code_buffer.size);
  // Simulate a cache hit.
  write_string(&cache, "1234");
  const xnn_code_span span2 = {
    .code = span2_code,
    .size = 4
  };
  ASSERT_EQ(0, xnn_code_cache_get_or_insert(&cache, span2));
  ASSERT_EQ(1, cache.hits);
  ASSERT_EQ(1, cache.misses);

  void* span3_code = (void*) ((uintptr_t) cache.code_buffer.code + cache.code_buffer.size);
  // Simulate a cache miss.
  write_string(&cache, "5678");
  const xnn_code_span span3 = {
    .code = span3_code,
    .size = 4
  };
  ASSERT_EQ(4, xnn_code_cache_get_or_insert(&cache, span3));
  ASSERT_EQ(1, cache.hits);
  ASSERT_EQ(2, cache.misses);
  ASSERT_EQ(2, cache.num_entries);

  EXPECT_EQ(xnn_status_success, xnn_release_code_cache(&cache));
}
