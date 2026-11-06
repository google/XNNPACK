// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// This header needs to go first for the arch test macros.
#include "src/xnnpack/common.h"

#if defined(XNN_SIMD_SCALAR)
#include "src/xnnpack/simd/s32-scalar.h"
#elif defined(XNN_SIMD_SSE41) && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
#include "src/xnnpack/simd/s32-sse41.h"
#define XNN_SIMD_TEST_ARCH_FLAGS xnn_arch_x86_sse4_1
#elif defined(XNN_SIMD_AVX2) && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
#include "src/xnnpack/simd/s32-avx2.h"
#define XNN_SIMD_TEST_ARCH_FLAGS xnn_arch_x86_avx2
#elif defined(XNN_SIMD_AVX512F) && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
#define XNN_SIMD_TEST_AVX512F 1
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx512f"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx512f")
#endif
#include "src/xnnpack/simd/s32-avx512f.h"
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#define XNN_SIMD_TEST_ARCH_FLAGS xnn_arch_x86_avx512f
#elif defined(XNN_SIMD_NEON) && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
#include "src/xnnpack/simd/s32-neon.h"
#define XNN_SIMD_TEST_ARCH_FLAGS xnn_arch_arm_neon
#elif defined(XNN_SIMD_WASMSIMD) && \
    (XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD)
#include "src/xnnpack/simd/s32-wasmsimd.h"
#elif defined(XNN_SIMD_HVX) && (XNN_ENABLE_HVX && XNN_ARCH_HEXAGON)
#include "src/xnnpack/simd/s32-hvx.h"
#define XNN_SIMD_TEST_ARCH_FLAGS xnn_arch_hvx
#endif

#ifdef xnn_simd_size_s32

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "src/xnnpack/isa-checks.h"
#include "test/replicable_random_device.h"

namespace xnnpack {

class S32SimdTest : public ::testing::Test {
 protected:
  void SetUp() override {
#ifdef XNN_SIMD_TEST_ARCH_FLAGS
    TEST_REQUIRES_ARCH_FLAGS(XNN_SIMD_TEST_ARCH_FLAGS);
#endif  // XNN_SIMD_TEST_ARCH_FLAGS
    inputs_.resize(3 * xnn_simd_size_s32);
    output_.resize(xnn_simd_size_s32);
    std::uniform_int_distribution<int> s32dist(-10000, 10000);
    std::generate(inputs_.begin(), inputs_.end(),
                  [&]() { return s32dist(rng_); });
  }

  xnnpack::ReplicableRandomDevice rng_;
  std::vector<int32_t> inputs_;
  std::vector<int32_t> output_;
};

#ifdef XNN_SIMD_TEST_AVX512F
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx512f"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx512f")
#endif
#endif  // XNN_SIMD_TEST_AVX512F

TEST_F(S32SimdTest, SimdSize) {
  ASSERT_GT(xnn_simd_size_s32, 0);
  ASSERT_EQ(xnn_simd_bytes_s32, xnn_simd_size_s32 * sizeof(int32_t));
#ifdef xnn_simd_log2_size_s32
  ASSERT_EQ(xnn_simd_size_s32, static_cast<size_t>(1)
                                   << xnn_simd_log2_size_s32);
#endif  // xnn_simd_log2_size_s32
}

TEST_F(S32SimdTest, Set1) {
  for (int32_t value :
       {std::numeric_limits<int32_t>::min(), int32_t{-1}, int32_t{0},
        int32_t{42}, std::numeric_limits<int32_t>::max(), inputs_[0]}) {
    const xnn_simd_s32_t v = xnn_set1_s32(value);
    xnn_storeu_s32(output_.data(), v);
    for (size_t k = 0; k < xnn_simd_size_s32; k++) {
      ASSERT_EQ(output_[k], value);
    }
  }
  XNN_SIMD_CONST_S32(vconst, -123456789);
  xnn_storeu_s32(output_.data(), vconst);
  for (size_t k = 0; k < xnn_simd_size_s32; k++) {
    ASSERT_EQ(output_[k], -123456789);
  }
}

TEST_F(S32SimdTest, LoadStoreAligned) {
  const size_t alignment =
      std::max<size_t>(alignof(int32_t), xnn_simd_bytes_s32);
  std::vector<uint8_t> input_storage(xnn_simd_bytes_s32 + alignment);
  std::vector<uint8_t> output_storage(xnn_simd_bytes_s32 + alignment);
  auto align_ptr = [](uint8_t* ptr, size_t align) -> int32_t* {
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t rem = addr % align;
    return reinterpret_cast<int32_t*>(rem == 0 ? ptr : ptr + (align - rem));
  };
  int32_t* aligned_input = align_ptr(input_storage.data(), alignment);
  int32_t* aligned_output = align_ptr(output_storage.data(), alignment);
  std::copy_n(inputs_.begin(), xnn_simd_size_s32, aligned_input);
  const xnn_simd_s32_t v = xnn_load_s32(aligned_input);
  xnn_store_s32(aligned_output, v);
  for (size_t k = 0; k < xnn_simd_size_s32; k++) {
    ASSERT_EQ(aligned_output[k], aligned_input[k]);
  }
}

TEST_F(S32SimdTest, Mul) {
  const xnn_simd_s32_t a = xnn_loadu_s32(inputs_.data());
  const xnn_simd_s32_t b = xnn_loadu_s32(inputs_.data() + xnn_simd_size_s32);
  const xnn_simd_s32_t res = xnn_mul_s32(a, b);
  xnn_storeu_s32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_s32; k++) {
    ASSERT_EQ(output_[k], inputs_[k] * inputs_[k + xnn_simd_size_s32]);
  }

  // Test negative and wrapping 32-bit products.
  for (const auto& pair :
       {std::pair<int32_t, int32_t>{-46341, 46341},
        std::pair<int32_t, int32_t>{46341, 46341},
        std::pair<int32_t, int32_t>{std::numeric_limits<int32_t>::max(), 2}}) {
    const xnn_simd_s32_t va = xnn_set1_s32(pair.first);
    const xnn_simd_s32_t vb = xnn_set1_s32(pair.second);
    xnn_storeu_s32(output_.data(), xnn_mul_s32(va, vb));
    const int32_t expected = static_cast<int32_t>(
        static_cast<uint32_t>(pair.first) * static_cast<uint32_t>(pair.second));
    for (size_t k = 0; k < xnn_simd_size_s32; k++) {
      ASSERT_EQ(output_[k], expected);
    }
  }
}

TEST_F(S32SimdTest, Sub) {
  const xnn_simd_s32_t a = xnn_loadu_s32(inputs_.data());
  const xnn_simd_s32_t b = xnn_loadu_s32(inputs_.data() + xnn_simd_size_s32);
  const xnn_simd_s32_t res = xnn_sub_s32(a, b);
  xnn_storeu_s32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_s32; k++) {
    ASSERT_EQ(output_[k], inputs_[k] - inputs_[k + xnn_simd_size_s32]);
  }
}

TEST_F(S32SimdTest, Min) {
  const xnn_simd_s32_t a = xnn_loadu_s32(inputs_.data());
  const xnn_simd_s32_t b = xnn_loadu_s32(inputs_.data() + xnn_simd_size_s32);
  const xnn_simd_s32_t res = xnn_min_s32(a, b);
  xnn_storeu_s32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_s32; k++) {
    ASSERT_EQ(output_[k], std::min(inputs_[k], inputs_[k + xnn_simd_size_s32]));
  }
}

TEST_F(S32SimdTest, Max) {
  const xnn_simd_s32_t a = xnn_loadu_s32(inputs_.data());
  const xnn_simd_s32_t b = xnn_loadu_s32(inputs_.data() + xnn_simd_size_s32);
  const xnn_simd_s32_t res = xnn_max_s32(a, b);
  xnn_storeu_s32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_s32; k++) {
    ASSERT_EQ(output_[k], std::max(inputs_[k], inputs_[k + xnn_simd_size_s32]));
  }
}

TEST_F(S32SimdTest, LoadTail) {
  for (size_t num_elements = 1; num_elements < xnn_simd_size_s32;
       num_elements++) {
    const xnn_simd_s32_t a = xnn_load_tail_s32(inputs_.data(), num_elements);
    xnn_storeu_s32(output_.data(), a);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_[k], inputs_[k]) << " " << k;
    }
  }
}

TEST_F(S32SimdTest, LoadTailSafe) {
  for (size_t num_elements = 0; num_elements <= xnn_simd_size_s32;
       num_elements++) {
    std::fill(output_.begin(), output_.end(), int32_t{-1});
    const xnn_simd_s32_t a =
        xnn_load_tail_safe_s32(inputs_.data(), num_elements);
    xnn_storeu_s32(output_.data(), a);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_[k], inputs_[k]) << " " << k;
    }
    for (size_t k = num_elements; k < xnn_simd_size_s32; k++) {
      ASSERT_EQ(output_[k], 0) << " " << k;
    }
  }
}

TEST_F(S32SimdTest, StoreTail) {
  const xnn_simd_s32_t a = xnn_loadu_s32(inputs_.data());
  for (size_t num_elements = 1; num_elements < xnn_simd_size_s32;
       num_elements++) {
    std::fill(output_.begin(), output_.end(), int32_t{0});
    xnn_store_tail_s32(output_.data(), a, num_elements);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_[k], inputs_[k]);
    }
    for (size_t k = num_elements; k < xnn_simd_size_s32; k++) {
      ASSERT_EQ(output_[k], 0);
    }
  }
}

#ifdef XNN_SIMD_TEST_AVX512F
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif  // XNN_SIMD_TEST_AVX512F

}  // namespace xnnpack

#endif  // xnn_simd_size_s32
