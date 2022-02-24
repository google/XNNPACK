// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/allocator.h>
#include <xnnpack/common.h>

#include <gtest/gtest.h>

TEST(JitMemory, AllocateAndReleaseEmptyCode) {
  xnn_code_buffer b;
  ASSERT_EQ(xnn_status_success, xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE));
#if XNN_PLATFORM_JIT
  ASSERT_EQ(xnn_status_success, xnn_finalize_code_memory(&b));
#endif  // XNN_PLATFORM_JIT
  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

TEST(JitMemory, AllocateAndReleaseJunkCode) {
  xnn_code_buffer b;
  ASSERT_EQ(xnn_status_success, xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE));
  std::string junk = "1234";
  memcpy(b.code, junk.data(), junk.length());
  b.size = junk.length();
#if XNN_PLATFORM_JIT
  ASSERT_EQ(xnn_status_success, xnn_finalize_code_memory(&b));
#endif  // XNN_PLATFORM_JIT
  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
  ASSERT_EQ(nullptr, b.code);
  ASSERT_EQ(0, b.size);
  ASSERT_EQ(0, b.capacity);
}

TEST(JitMemory, AllocateAndReleaseCodeBufferWithNoCapacity) {
  xnn_code_buffer b;
  b.capacity = 0;
  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

TEST(JitMemory, GrowMemory) {
  xnn_code_buffer b;
  ASSERT_EQ(xnn_status_success, xnn_allocate_code_memory(&b, 8));
  std::string junk = "1234";
  memcpy(b.code, junk.data(), junk.length());
  b.size += junk.length();
  ASSERT_EQ(b.size, 4);
  uintptr_t old_code = (uintptr_t) b.code;

  // This should be a no-op, since we have enough space.
  ASSERT_EQ(xnn_status_success, xnn_grow_code_memory(&b, 4));
  ASSERT_EQ(old_code, (uintptr_t) b.code);

  // Copy 4 more bytes, now we are full.
  memcpy(b.code, junk.data(), junk.length());
  b.size += junk.length();

  size_t old_size = b.size;
  ASSERT_EQ(xnn_status_success, xnn_grow_code_memory(&b, 4));

  // After growing, the new capacity should be bigger than the old one.
  ASSERT_EQ(12, b.capacity);
  // But size stays the same.
  ASSERT_EQ(old_size, b.size);

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}
