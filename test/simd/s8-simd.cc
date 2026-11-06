// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// This header needs to go first for the arch test macros.
#include "src/xnnpack/common.h"

#if defined(XNN_SIMD_SCALAR)
#include "src/xnnpack/simd/s8-scalar.h"
#elif defined(XNN_SIMD_SSE41) && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
#include "src/xnnpack/simd/s8-sse41.h"
#define XNN_SIMD_TEST_ARCH_FLAGS xnn_arch_x86_sse4_1
#elif defined(XNN_SIMD_NEON) && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
#include "src/xnnpack/simd/s8-neon.h"
#define XNN_SIMD_TEST_ARCH_FLAGS xnn_arch_arm_neon
#elif defined(XNN_SIMD_WASMSIMD) && \
    (XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD)
#include "src/xnnpack/simd/s8-wasmsimd.h"
#elif defined(XNN_SIMD_HVX) && (XNN_ENABLE_HVX && XNN_ARCH_HEXAGON)
#include "src/xnnpack/simd/s8-hvx.h"
#define XNN_SIMD_TEST_ARCH_FLAGS xnn_arch_hvx
#endif

#ifdef xnn_simd_size_s8

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "src/xnnpack/isa-checks.h"
#include "test/replicable_random_device.h"

namespace xnnpack {

class S8SimdTest : public ::testing::Test {
 protected:
  void SetUp() override {
#ifdef XNN_SIMD_TEST_ARCH_FLAGS
    TEST_REQUIRES_ARCH_FLAGS(XNN_SIMD_TEST_ARCH_FLAGS);
#endif  // XNN_SIMD_TEST_ARCH_FLAGS
    inputs_.resize(3 * xnn_simd_size_s8);
    output_.resize(xnn_simd_size_s8);
    std::uniform_int_distribution<int> s8(std::numeric_limits<int8_t>::min(),
                                          std::numeric_limits<int8_t>::max());
    std::generate(inputs_.begin(), inputs_.end(), [&]() { return s8(rng_); });
  }

  xnnpack::ReplicableRandomDevice rng_;
  std::vector<int8_t> inputs_;
  std::vector<int8_t> output_;
};

TEST_F(S8SimdTest, SimdSize) {
  ASSERT_GT(xnn_simd_size_s8, 0);
  ASSERT_EQ(xnn_simd_bytes_s8, xnn_simd_size_s8 * sizeof(int8_t));
#ifdef xnn_simd_log2_size_s8
  ASSERT_EQ(xnn_simd_size_s8, static_cast<size_t>(1) << xnn_simd_log2_size_s8);
#endif  // xnn_simd_log2_size_s8
}

TEST_F(S8SimdTest, Set1) {
  for (int8_t value : {int8_t{-128}, int8_t{-1}, int8_t{0}, int8_t{42},
                       int8_t{127}, inputs_[0]}) {
    const xnn_simd_s8_t v = xnn_set1_s8(value);
    xnn_storeu_s8(output_.data(), v);
    for (size_t k = 0; k < xnn_simd_size_s8; k++) {
      ASSERT_EQ(output_[k], value);
    }
  }
  XNN_SIMD_CONST_S8(vconst, -37);
  xnn_storeu_s8(output_.data(), vconst);
  for (size_t k = 0; k < xnn_simd_size_s8; k++) {
    ASSERT_EQ(output_[k], -37);
  }
}

TEST_F(S8SimdTest, LoadStoreAligned) {
  const size_t alignment = std::max<size_t>(alignof(int8_t), xnn_simd_bytes_s8);
  std::vector<uint8_t> input_storage(xnn_simd_bytes_s8 + alignment);
  std::vector<uint8_t> output_storage(xnn_simd_bytes_s8 + alignment);
  auto align_ptr = [](uint8_t* ptr, size_t align) -> int8_t* {
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t rem = addr % align;
    return reinterpret_cast<int8_t*>(rem == 0 ? ptr : ptr + (align - rem));
  };
  int8_t* aligned_input = align_ptr(input_storage.data(), alignment);
  int8_t* aligned_output = align_ptr(output_storage.data(), alignment);
  std::copy_n(inputs_.begin(), xnn_simd_size_s8, aligned_input);
  const xnn_simd_s8_t v = xnn_load_s8(aligned_input);
  xnn_store_s8(aligned_output, v);
  for (size_t k = 0; k < xnn_simd_size_s8; k++) {
    ASSERT_EQ(aligned_output[k], aligned_input[k]);
  }
}

TEST_F(S8SimdTest, Add) {
  const xnn_simd_s8_t a = xnn_loadu_s8(inputs_.data());
  const xnn_simd_s8_t b = xnn_loadu_s8(inputs_.data() + xnn_simd_size_s8);
  const xnn_simd_s8_t res = xnn_add_s8(a, b);
  xnn_storeu_s8(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_s8; k++) {
    ASSERT_EQ(output_[k],
              static_cast<int8_t>(inputs_[k] + inputs_[k + xnn_simd_size_s8]));
  }
}

TEST_F(S8SimdTest, Min) {
  const xnn_simd_s8_t a = xnn_loadu_s8(inputs_.data());
  const xnn_simd_s8_t b = xnn_loadu_s8(inputs_.data() + xnn_simd_size_s8);
  const xnn_simd_s8_t res = xnn_min_s8(a, b);
  xnn_storeu_s8(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_s8; k++) {
    ASSERT_EQ(output_[k],
              std::min<int8_t>(inputs_[k], inputs_[k + xnn_simd_size_s8]));
  }
}

TEST_F(S8SimdTest, Max) {
  const xnn_simd_s8_t a = xnn_loadu_s8(inputs_.data());
  const xnn_simd_s8_t b = xnn_loadu_s8(inputs_.data() + xnn_simd_size_s8);
  const xnn_simd_s8_t res = xnn_max_s8(a, b);
  xnn_storeu_s8(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_s8; k++) {
    ASSERT_EQ(output_[k],
              std::max<int8_t>(inputs_[k], inputs_[k + xnn_simd_size_s8]));
  }
}

TEST_F(S8SimdTest, ReduceMin) {
  const xnn_simd_s8_t a = xnn_loadu_s8(inputs_.data());
  int8_t expected = std::numeric_limits<int8_t>::max();
  for (size_t i = 0; i < xnn_simd_size_s8; ++i) {
    expected = std::min<int8_t>(expected, inputs_[i]);
  }
  ASSERT_EQ(xnn_reduce_min_s8(a), expected);
}

TEST_F(S8SimdTest, ReduceMax) {
  const xnn_simd_s8_t a = xnn_loadu_s8(inputs_.data());
  int8_t expected = std::numeric_limits<int8_t>::min();
  for (size_t i = 0; i < xnn_simd_size_s8; ++i) {
    expected = std::max<int8_t>(expected, inputs_[i]);
  }
  ASSERT_EQ(xnn_reduce_max_s8(a), expected);

  std::uniform_int_distribution<int> neg_s8(-128, -1);
  std::generate(inputs_.begin(), inputs_.begin() + xnn_simd_size_s8,
                [&]() { return neg_s8(rng_); });
  const xnn_simd_s8_t neg_a = xnn_loadu_s8(inputs_.data());
  expected = std::numeric_limits<int8_t>::min();
  for (size_t i = 0; i < xnn_simd_size_s8; ++i) {
    expected = std::max<int8_t>(expected, inputs_[i]);
  }
  ASSERT_EQ(xnn_reduce_max_s8(neg_a), expected);
}

TEST_F(S8SimdTest, Xor) {
  const xnn_simd_s8_t a = xnn_loadu_s8(inputs_.data());
  const xnn_simd_s8_t b = xnn_loadu_s8(inputs_.data() + xnn_simd_size_s8);
  const xnn_simd_s8_t res = xnn_xor_s8(a, b);
  xnn_storeu_s8(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_s8; k++) {
    ASSERT_EQ(output_[k],
              static_cast<int8_t>(inputs_[k] ^ inputs_[k + xnn_simd_size_s8]));
  }
}

TEST_F(S8SimdTest, LoadTail) {
  for (size_t num_elements = 1; num_elements < xnn_simd_size_s8;
       num_elements++) {
    const xnn_simd_s8_t a = xnn_load_tail_s8(inputs_.data(), num_elements);
    xnn_storeu_s8(output_.data(), a);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_[k], inputs_[k]) << " " << k;
    }
  }
}

TEST_F(S8SimdTest, LoadTailSafe) {
  for (size_t num_elements = 1; num_elements < xnn_simd_size_s8;
       num_elements++) {
    const xnn_simd_s8_t a = xnn_load_tail_safe_s8(inputs_.data(), num_elements);
    xnn_storeu_s8(output_.data(), a);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_[k], inputs_[k]) << " " << k;
    }
  }
}

TEST_F(S8SimdTest, StoreTail) {
  const xnn_simd_s8_t a = xnn_loadu_s8(inputs_.data());
  for (size_t num_elements = 1; num_elements < xnn_simd_size_s8;
       num_elements++) {
    std::fill(output_.begin(), output_.end(), int8_t{0});
    xnn_store_tail_s8(output_.data(), a, num_elements);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_[k], inputs_[k]) << " " << k;
    }
    for (size_t k = num_elements; k < xnn_simd_size_s8; k++) {
      ASSERT_EQ(output_[k], 0);
    }
  }
}

}  // namespace xnnpack

#endif  // xnn_simd_size_s8
