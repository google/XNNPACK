// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// This header needs to go first for the arch test macros.
#include "src/xnnpack/common.h"

#if defined(XNN_SIMD_SCALAR)
#include "src/xnnpack/simd/u8-scalar.h"
#elif defined(XNN_SIMD_SSE2) && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
#include "src/xnnpack/simd/u8-sse2.h"
#elif defined(XNN_SIMD_NEON) && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
#include "src/xnnpack/simd/u8-neon.h"
#define XNN_SIMD_TEST_ARCH_FLAGS xnn_arch_arm_neon
#elif defined(XNN_SIMD_WASMSIMD) && \
    (XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD)
#include "src/xnnpack/simd/u8-wasmsimd.h"
#elif defined(XNN_SIMD_HVX) && (XNN_ENABLE_HVX && XNN_ARCH_HEXAGON)
#include "src/xnnpack/simd/u8-hvx.h"
#define XNN_SIMD_TEST_ARCH_FLAGS xnn_arch_hvx
#endif

#ifdef xnn_simd_size_u8

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

class U8SimdTest : public ::testing::Test {
 protected:
  void SetUp() override {
#ifdef XNN_SIMD_TEST_ARCH_FLAGS
    TEST_REQUIRES_ARCH_FLAGS(XNN_SIMD_TEST_ARCH_FLAGS);
#endif  // XNN_SIMD_TEST_ARCH_FLAGS
    inputs_.resize(3 * xnn_simd_size_u8);
    output_.resize(xnn_simd_size_u8);
    std::uniform_int_distribution<int> u8(std::numeric_limits<uint8_t>::min(),
                                          std::numeric_limits<uint8_t>::max());
    std::generate(inputs_.begin(), inputs_.end(), [&]() { return u8(rng_); });
  }

  xnnpack::ReplicableRandomDevice rng_;
  std::vector<uint8_t> inputs_;
  std::vector<uint8_t> output_;
};

TEST_F(U8SimdTest, SimdSize) {
  ASSERT_GT(xnn_simd_size_u8, 0);
  ASSERT_EQ(xnn_simd_bytes_u8, xnn_simd_size_u8 * sizeof(uint8_t));
#ifdef xnn_simd_log2_size_u8
  ASSERT_EQ(xnn_simd_size_u8, static_cast<size_t>(1) << xnn_simd_log2_size_u8);
#endif  // xnn_simd_log2_size_u8
}

TEST_F(U8SimdTest, Set1) {
  for (uint8_t value : {uint8_t{0}, uint8_t{1}, uint8_t{42}, uint8_t{128},
                        uint8_t{255}, inputs_[0]}) {
    const xnn_simd_u8_t v = xnn_set1_u8(value);
    xnn_storeu_u8(output_.data(), v);
    for (size_t k = 0; k < xnn_simd_size_u8; k++) {
      ASSERT_EQ(output_[k], value);
    }
  }
  XNN_SIMD_CONST_U8(vconst, 219);
  xnn_storeu_u8(output_.data(), vconst);
  for (size_t k = 0; k < xnn_simd_size_u8; k++) {
    ASSERT_EQ(output_[k], 219);
  }
}

TEST_F(U8SimdTest, LoadStoreAligned) {
  const size_t alignment =
      std::max<size_t>(alignof(uint8_t), xnn_simd_bytes_u8);
  std::vector<uint8_t> input_storage(xnn_simd_bytes_u8 + alignment);
  std::vector<uint8_t> output_storage(xnn_simd_bytes_u8 + alignment);
  auto align_ptr = [](uint8_t* ptr, size_t align) -> uint8_t* {
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t rem = addr % align;
    return rem == 0 ? ptr : ptr + (align - rem);
  };
  uint8_t* aligned_input = align_ptr(input_storage.data(), alignment);
  uint8_t* aligned_output = align_ptr(output_storage.data(), alignment);
  std::copy_n(inputs_.begin(), xnn_simd_size_u8, aligned_input);
  const xnn_simd_u8_t v = xnn_load_u8(aligned_input);
  xnn_store_u8(aligned_output, v);
  for (size_t k = 0; k < xnn_simd_size_u8; k++) {
    ASSERT_EQ(aligned_output[k], aligned_input[k]);
  }
}

TEST_F(U8SimdTest, Add) {
  const xnn_simd_u8_t a = xnn_loadu_u8(inputs_.data());
  const xnn_simd_u8_t b = xnn_loadu_u8(inputs_.data() + xnn_simd_size_u8);
  const xnn_simd_u8_t res = xnn_add_u8(a, b);
  xnn_storeu_u8(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_u8; k++) {
    ASSERT_EQ(output_[k],
              static_cast<uint8_t>(inputs_[k] + inputs_[k + xnn_simd_size_u8]));
  }
}

TEST_F(U8SimdTest, Min) {
  const xnn_simd_u8_t a = xnn_loadu_u8(inputs_.data());
  const xnn_simd_u8_t b = xnn_loadu_u8(inputs_.data() + xnn_simd_size_u8);
  const xnn_simd_u8_t res = xnn_min_u8(a, b);
  xnn_storeu_u8(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_u8; k++) {
    ASSERT_EQ(output_[k],
              std::min<uint8_t>(inputs_[k], inputs_[k + xnn_simd_size_u8]));
  }
}

TEST_F(U8SimdTest, Max) {
  const xnn_simd_u8_t a = xnn_loadu_u8(inputs_.data());
  const xnn_simd_u8_t b = xnn_loadu_u8(inputs_.data() + xnn_simd_size_u8);
  const xnn_simd_u8_t res = xnn_max_u8(a, b);
  xnn_storeu_u8(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_u8; k++) {
    ASSERT_EQ(output_[k],
              std::max<uint8_t>(inputs_[k], inputs_[k + xnn_simd_size_u8]));
  }
}

TEST_F(U8SimdTest, Xor) {
  const xnn_simd_u8_t a = xnn_loadu_u8(inputs_.data());
  const xnn_simd_u8_t b = xnn_loadu_u8(inputs_.data() + xnn_simd_size_u8);
  const xnn_simd_u8_t res = xnn_xor_u8(a, b);
  xnn_storeu_u8(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_u8; k++) {
    ASSERT_EQ(output_[k],
              static_cast<uint8_t>(inputs_[k] ^ inputs_[k + xnn_simd_size_u8]));
  }
}

TEST_F(U8SimdTest, ReduceMin) {
  const xnn_simd_u8_t a = xnn_loadu_u8(inputs_.data());
  uint8_t expected = std::numeric_limits<uint8_t>::max();
  for (size_t i = 0; i < xnn_simd_size_u8; ++i) {
    expected = std::min<uint8_t>(expected, inputs_[i]);
  }
  ASSERT_EQ(xnn_reduce_min_u8(a), expected);
}

TEST_F(U8SimdTest, ReduceMax) {
  const xnn_simd_u8_t a = xnn_loadu_u8(inputs_.data());
  uint8_t expected = std::numeric_limits<uint8_t>::min();
  for (size_t i = 0; i < xnn_simd_size_u8; ++i) {
    expected = std::max<uint8_t>(expected, inputs_[i]);
  }
  ASSERT_EQ(xnn_reduce_max_u8(a), expected);
}

TEST_F(U8SimdTest, LoadTail) {
  for (size_t num_elements = 1; num_elements < xnn_simd_size_u8;
       num_elements++) {
    const xnn_simd_u8_t a = xnn_load_tail_u8(inputs_.data(), num_elements);
    xnn_storeu_u8(output_.data(), a);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_[k], inputs_[k]) << " " << k;
    }
  }
}

TEST_F(U8SimdTest, LoadTailSafe) {
  for (size_t num_elements = 1; num_elements < xnn_simd_size_u8;
       num_elements++) {
    const xnn_simd_u8_t a = xnn_load_tail_safe_u8(inputs_.data(), num_elements);
    xnn_storeu_u8(output_.data(), a);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_[k], inputs_[k]) << " " << k;
    }
  }
}

TEST_F(U8SimdTest, StoreTail) {
  const xnn_simd_u8_t a = xnn_loadu_u8(inputs_.data());
  for (size_t num_elements = 1; num_elements < xnn_simd_size_u8;
       num_elements++) {
    std::fill(output_.begin(), output_.end(), uint8_t{0});
    xnn_store_tail_u8(output_.data(), a, num_elements);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_[k], inputs_[k]) << " " << k;
    }
    for (size_t k = num_elements; k < xnn_simd_size_u8; k++) {
      ASSERT_EQ(output_[k], 0);
    }
  }
}

}  // namespace xnnpack

#endif  // xnn_simd_size_u8
