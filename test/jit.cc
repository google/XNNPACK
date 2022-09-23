// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstring>

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/memory.h>


TEST(JIT_MEMORY, allocate_and_release_empty_code) {
  xnn_code_buffer b;
  ASSERT_EQ(xnn_status_success, xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE));
#if XNN_PLATFORM_JIT
  ASSERT_EQ(xnn_status_success, xnn_finalize_code_memory(&b));
#endif  // XNN_PLATFORM_JIT
  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

TEST(JIT_MEMORY, allocate_and_release_junk_code) {
  xnn_code_buffer b;
  ASSERT_EQ(xnn_status_success, xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE));
  std::string junk = "1234";
  std::memcpy(b.start, junk.data(), junk.length());
  b.size = junk.length();
#if XNN_PLATFORM_JIT
  ASSERT_EQ(xnn_status_success, xnn_finalize_code_memory(&b));
  ASSERT_GT(XNN_DEFAULT_CODE_BUFFER_SIZE, b.capacity);
#endif  // XNN_PLATFORM_JIT
  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
  ASSERT_EQ(nullptr, b.start);
  ASSERT_EQ(0, b.size);
  ASSERT_EQ(0, b.capacity);
}

TEST(JIT_MEMORY, allocate_and_release_code_buffer_with_no_capacity) {
  xnn_code_buffer b;
  b.capacity = 0;
  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

TEST(JIT_MEMORY, grow_memory) {
  xnn_code_buffer b;
  ASSERT_EQ(xnn_status_success, xnn_allocate_code_memory(&b, 8));
  size_t original_capacity = b.capacity;
  constexpr size_t junk_len = 4;
  b.size += junk_len;
  ASSERT_EQ(b.size, 4);
  const uintptr_t old_code = reinterpret_cast<uintptr_t>(b.start);

  // This should be a no-op, since we have enough space.
  ASSERT_EQ(xnn_status_success, xnn_reserve_code_memory(&b, 4));
  ASSERT_EQ(old_code, reinterpret_cast<uintptr_t>(b.start));
  ASSERT_EQ(original_capacity, b.capacity);

  // Simulate copying bytes until the memory is full.
  b.size += original_capacity - junk_len;
  ASSERT_EQ(b.size, b.capacity);

  const size_t old_size = b.size;
  ASSERT_EQ(xnn_status_success, xnn_reserve_code_memory(&b, 4));

  // After growing, the new capacity should be bigger than the old one.
  ASSERT_LT(original_capacity, b.capacity);
  // At least 4 bytes free.
  ASSERT_GE(b.capacity, b.size + 4);
  // But size stays the same.
  ASSERT_EQ(old_size, b.size);

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

TEST(JIT_MEMORY, finalize_twice) {
  xnn_code_buffer b;
  ASSERT_EQ(xnn_status_success, xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE));
  const std::string junk = "1234";
  std::memcpy(b.start, junk.data(), junk.length());
  b.size += junk.length();
  ASSERT_EQ(b.size, 4);

#if XNN_PLATFORM_JIT
  ASSERT_EQ(xnn_status_success, xnn_finalize_code_memory(&b));
#endif
  const size_t capacity = b.capacity;
  // Finalizing twice does not error.
#if XNN_PLATFORM_JIT
  ASSERT_EQ(xnn_status_success, xnn_finalize_code_memory(&b));
#endif
  // Capacity does not change.
  ASSERT_EQ(capacity, b.capacity);
  ASSERT_EQ(4, b.size);

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}
