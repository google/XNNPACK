// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/common.h>
#include <xnnpack/memory.h>

#include <cstring>
#include <string>

#include <gtest/gtest.h>

#if !XNN_PLATFORM_WEB  // the behavior of xnn_finalize_code_memory on WEB is
                       // different and tested separately.
TEST(JIT_MEMORY, allocate_and_release_empty_code) {
  xnn_code_buffer b;
  ASSERT_EQ(xnn_status_success,
            xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE));
#if XNN_PLATFORM_JIT && !XNN_PLATFORM_WEB
  ASSERT_EQ(xnn_status_success, xnn_finalize_code_memory(&b));
#endif  // XNN_PLATFORM_JIT && !XNN_PLATFORM_WEB
  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

TEST(JIT_MEMORY, allocate_and_release_junk_code) {
  xnn_code_buffer b;
  ASSERT_EQ(xnn_status_success,
            xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE));
  std::string junk = "1234";
  std::memcpy(b.start, junk.data(), junk.length());
  b.size = junk.length();
#if XNN_PLATFORM_JIT && !XNN_PLATFORM_WEB
  ASSERT_EQ(xnn_status_success, xnn_finalize_code_memory(&b));
  ASSERT_GT(XNN_DEFAULT_CODE_BUFFER_SIZE, b.capacity);
#endif  // XNN_PLATFORM_JIT && !XNN_PLATFORM_WEB
  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
  ASSERT_EQ(nullptr, b.start);
  ASSERT_EQ(0, b.size);
  ASSERT_EQ(0, b.capacity);
}
#endif  // !XNN_PLATFORM_WEB

#if XNN_PLATFORM_WEB
TEST(JIT_MEMORY, allocate_and_release_junk_code_web) {
  xnn_code_buffer b;
  ASSERT_EQ(xnn_status_success,
            xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE));
  std::string junk = "1234";
  std::memcpy(b.start, junk.data(), junk.length());
  b.size = junk.length();
  ASSERT_EQ(XNN_INVALID_FUNCTION_INDEX, xnn_first_function_in_chunk_ptr(&b, 0, junk.length()));
  ASSERT_EQ(xnn_status_success, xnn_finalize_code_memory(&b));
  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
  ASSERT_EQ(nullptr, b.start);
  ASSERT_EQ(0, b.size);
  ASSERT_EQ(0, b.capacity);
}
#endif  // XNN_PLATFORM_WEB

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

#if !XNN_PLATFORM_WEB  // the behavior of xnn_finalize_code_memory on WEB is
                       // different and tested separately.
TEST(JIT_MEMORY, finalize_twice) {
  xnn_code_buffer b;
  ASSERT_EQ(xnn_status_success,
            xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE));
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
#endif  // !XNN_PLATFORM_WEB

#if XNN_PLATFORM_JIT && XNN_PLATFORM_WEB

constexpr size_t kDotCodeSize = 159;
// The code below a slightly modified output of
// emcc  -Os dot.cc  -s SIDE_MODULE=1  -o dot.wasm
// The Export section is edited to include only `dot` and `add`.
//
// dot.cc:
// extern "C" {
// int dot(int* a, int* b, int size) {
//   int res = 0;
//   for (int i = 0; i < size; i++) {
//     res += a[i] * b[i];
//   }
//   return res;
// }
// int add(int a, int b) {
//   return a + b;
// }
// }
constexpr std::array<uint8_t, kDotCodeSize> kDotCode = {
    0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00, 0x00, 0x0F, 0x08, 0x64,
    0x79, 0x6C, 0x69, 0x6E, 0x6B, 0x2E, 0x30, 0x01, 0x04, 0x00, 0x00, 0x00,
    0x00, 0x01, 0x11, 0x03, 0x60, 0x00, 0x00, 0x60, 0x03, 0x7F, 0x7F, 0x7F,
    0x01, 0x7F, 0x60, 0x02, 0x7F, 0x7F, 0x01, 0x7F, 0x02, 0x0F, 0x01, 0x03,
    0x65, 0x6E, 0x76, 0x06, 0x6D, 0x65, 0x6D, 0x6F, 0x72, 0x79, 0x02, 0x00,
    0x00, 0x03, 0x04, 0x03, 0x00, 0x01, 0x02, 0x07, 0x0d, 0x02, 0x03, 0x64,
    0x6F, 0x74, 0x00, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x02, 0x0A, 0x4B,
    0x03, 0x03, 0x00, 0x01, 0x0B, 0x3D, 0x01, 0x02, 0x7F, 0x02, 0x40, 0x20,
    0x02, 0x41, 0x00, 0x4C, 0x04, 0x40, 0x0C, 0x01, 0x0B, 0x03, 0x40, 0x20,
    0x03, 0x20, 0x01, 0x20, 0x04, 0x41, 0x02, 0x74, 0x22, 0x03, 0x6A, 0x28,
    0x02, 0x00, 0x20, 0x00, 0x20, 0x03, 0x6A, 0x28, 0x02, 0x00, 0x6C, 0x6A,
    0x21, 0x03, 0x20, 0x04, 0x41, 0x01, 0x6A, 0x22, 0x04, 0x20, 0x02, 0x47,
    0x0D, 0x00, 0x0B, 0x0B, 0x20, 0x03, 0x0B, 0x07, 0x00, 0x20, 0x00, 0x20,
    0x01, 0x6A, 0x0B};

// the same code as `kDotCode`, but "Export" section is removed
constexpr size_t kNoExportsCodeSize = 129;
constexpr std::array<uint8_t, kNoExportsCodeSize> kNoExportsCode = {
    0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00, 0x00, 0x0F, 0x08, 0x64,
    0x79, 0x6C, 0x69, 0x6E, 0x6B, 0x2E, 0x30, 0x01, 0x04, 0x00, 0x00, 0x00,
    0x00, 0x01, 0x0B, 0x02, 0x60, 0x00, 0x00, 0x60, 0x03, 0x7F, 0x7F, 0x7F,
    0x01, 0x7F, 0x02, 0x0F, 0x01, 0x03, 0x65, 0x6E, 0x76, 0x06, 0x6D, 0x65,
    0x6D, 0x6F, 0x72, 0x79, 0x02, 0x00, 0x00, 0x03, 0x03, 0x02, 0x00, 0x01,
    0x0A, 0x43, 0x02, 0x03, 0x00, 0x01, 0x0B, 0x3D, 0x01, 0x02, 0x7F, 0x02,
    0x40, 0x20, 0x02, 0x41, 0x00, 0x4C, 0x04, 0x40, 0x0C, 0x01, 0x0B, 0x03,
    0x40, 0x20, 0x03, 0x20, 0x01, 0x20, 0x04, 0x41, 0x02, 0x74, 0x22, 0x03,
    0x6A, 0x28, 0x02, 0x00, 0x20, 0x00, 0x20, 0x03, 0x6A, 0x28, 0x02, 0x00,
    0x6C, 0x6A, 0x21, 0x03, 0x20, 0x04, 0x41, 0x01, 0x6A, 0x22, 0x04, 0x20,
    0x02, 0x47, 0x0D, 0x00, 0x0B, 0x0B, 0x20, 0x03, 0x0B};

constexpr size_t kVecSize = 5;
constexpr std::array<int, kVecSize> kVecA = {1, 2, 3, 4, 5};
constexpr std::array<int, kVecSize> kVecB = {10, 20, 30, 40, 50};
const int kExpectedDotProduct = [] {
  int result = 0;
  for (int i = 0; i < kVecSize; i++) {
    result += kVecA[i] * kVecB[i];
  }
  return result;
}();

constexpr int kA = 2;
constexpr int kB = 5;
constexpr int kSum = kA + kB;

using DotPtr = int (*)(const int*, const int*, int);
using AddPtr = int (*)(int, int);

template <typename Array>
void MakeBuffer(Array&& array, xnn_code_buffer* b) {
  ASSERT_EQ(xnn_status_success, xnn_allocate_code_memory(b, array.size()));
  b->size = array.size();
  std::memcpy(b->start, array.data(), array.size());
}

template <typename Array>
void MakeBufferWithTwoModules(Array&& array, xnn_code_buffer* b) {
  const size_t buffer_size = 2 * array.size();
  ASSERT_EQ(xnn_status_success, xnn_allocate_code_memory(b, array.size()));
  b->size = array.size();
  std::memcpy(b->start, array.data(), array.size());
  xnn_reserve_code_memory(b, buffer_size);
  b->size += array.size();
  std::memcpy((void*)((uintptr_t)b->start + array.size()), array.data(), array.size());
}

TEST(JIT_MEMORY, finalize_code_twice_with_dot_and_add_exported) {
  xnn_code_buffer buffer;
  MakeBuffer(kDotCode, &buffer);
  const auto status = xnn_finalize_code_memory(&buffer);
  const int dot = xnn_first_function_ptr(&buffer);
  ASSERT_EQ(status, xnn_status_success);
  EXPECT_NE(dot, XNN_INVALID_FUNCTION_INDEX);
  const int add = dot + 1;
  EXPECT_EQ(((DotPtr)dot)(kVecA.data(), kVecB.data(), kVecSize),
            kExpectedDotProduct);
  EXPECT_EQ(((AddPtr)add)(kA, kB), kSum);
  ASSERT_EQ(xnn_finalize_code_memory(&buffer), xnn_status_success);
  EXPECT_NE(dot, xnn_first_function_ptr(&buffer));

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&buffer));
}

TEST(JIT_MEMORY, finalize_code_for_multiple_modules_in_buffer) {
  xnn_code_buffer buffer;
  MakeBufferWithTwoModules(kDotCode, &buffer);
  const auto status = xnn_finalize_code_memory(&buffer);
  ASSERT_EQ(status, xnn_status_success);
  {
    // for the first module
    const int dot = xnn_first_function_in_chunk_ptr(&buffer, 0, kDotCode.size());
    EXPECT_NE(dot, XNN_INVALID_FUNCTION_INDEX);
    const int add = dot + 1;
    EXPECT_EQ(((DotPtr)dot)(kVecA.data(), kVecB.data(), kVecSize),
              kExpectedDotProduct);
    EXPECT_EQ(((AddPtr)add)(kA, kB), kSum);
  }
  {
    // for the second module
    const int dot = xnn_first_function_in_chunk_ptr(&buffer, kDotCode.size(), buffer.size);
    EXPECT_NE(dot, XNN_INVALID_FUNCTION_INDEX);
    const int add = dot + 1;
    EXPECT_EQ(((DotPtr)dot)(kVecA.data(), kVecB.data(), kVecSize),
              kExpectedDotProduct);
    EXPECT_EQ(((AddPtr)add)(kA, kB), kSum);
  }

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&buffer));
}

TEST(JIT_MEMORY, finalize_code_with_no_exports) {
  xnn_code_buffer buffer;
  MakeBuffer(kNoExportsCode, &buffer);
  const auto status = xnn_finalize_code_memory(&buffer);
  EXPECT_EQ(status, xnn_status_success);
  EXPECT_EQ(xnn_first_function_ptr(&buffer), XNN_INVALID_FUNCTION_INDEX);
  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&buffer));
}
#endif  // XNN_PLATFORM_JIT && XNN_PLATFORM_WEB
