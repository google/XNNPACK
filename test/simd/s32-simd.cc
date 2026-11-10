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
// Scope AVX-512 target attributes to the SIMD header and test bodies so the
// test fixture SetUp() (which checks ISA support) can run on older CPUs.
#define XNN_SIMD_USES_AVX512F_TARGET 1
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
#if defined(XNN_SIMD_HVX)
#include <cstring>
#endif
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

#if defined(XNN_SIMD_USES_AVX512F_TARGET)
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx512f"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx512f")
#endif
#endif  // defined(XNN_SIMD_USES_AVX512F_TARGET)

TEST_F(S32SimdTest, Mul) {
  const xnn_simd_s32_t a = xnn_loadu_s32(inputs_.data());
  const xnn_simd_s32_t b = xnn_loadu_s32(inputs_.data() + xnn_simd_size_s32);
  const xnn_simd_s32_t res = xnn_mul_s32(a, b);
  xnn_storeu_s32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_s32; k++) {
    ASSERT_EQ(output_[k], inputs_[k] * inputs_[k + xnn_simd_size_s32]);
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

TEST_F(S32SimdTest, StoreTail) {
  const xnn_simd_s32_t a = xnn_loadu_s32(inputs_.data());
  for (size_t num_elements = 1; num_elements < xnn_simd_size_s32;
       num_elements++) {
    xnn_store_tail_s32(output_.data(), a, num_elements);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_[k], inputs_[k]);
    }
    for (size_t k = num_elements; k < xnn_simd_size_s32; k++) {
      ASSERT_EQ(output_[k], 0.0f);
    }
  }
}

#if defined(XNN_SIMD_HVX)
TEST_F(S32SimdTest, CvtF32S32) {
  std::vector<int32_t> values = {
      0, 1, -1, 2, -2, 255, -256, 65535, 65536, -65537,
      (1 << 24) - 1, 1 << 24, (1 << 24) + 1, (1 << 24) + 3, -(1 << 24) - 1,
      (1 << 25) + 2, (1 << 25) + 6, 0x7FFFFF80, 0x7FFFFFC0,
      std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::min(),
      std::numeric_limits<int32_t>::min() + 1};
  std::uniform_int_distribution<int32_t> s32dist(
      std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());
  while (values.size() % xnn_simd_size_s32 != 0 || values.size() < 1024) {
    values.push_back(s32dist(rng_));
  }
  std::vector<float> result(xnn_simd_size_s32);
  for (size_t i = 0; i < values.size(); i += xnn_simd_size_s32) {
    const xnn_simd_s32_t a = xnn_loadu_s32(values.data() + i);
    const HVX_Vector res = xnn_cvt_f32_s32(a);
    std::memcpy(result.data(), &res, sizeof(res));
    for (size_t k = 0; k < xnn_simd_size_s32; k++) {
      ASSERT_EQ(result[k], static_cast<float>(values[i + k]))
          << "input " << values[i + k];
    }
  }
}
#endif  // defined(XNN_SIMD_HVX)

#if defined(XNN_SIMD_USES_AVX512F_TARGET)
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif  // defined(XNN_SIMD_USES_AVX512F_TARGET)

}  // namespace xnnpack

#endif  // xnn_simd_size_s32
