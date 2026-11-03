// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// This header needs to go first for the arch test macros.
#include "src/xnnpack/common.h"

#if defined(XNN_SIMD_SCALAR)
#include "src/xnnpack/simd/s16-scalar.h"
#elif defined(XNN_SIMD_SSE41) && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
#include "src/xnnpack/simd/s16-sse41.h"
#define XNN_SIMD_TEST_ARCH_FLAGS xnn_arch_x86_sse4_1
#elif defined(XNN_SIMD_AVX2) && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
#include "src/xnnpack/simd/s16-avx2.h"
#define XNN_SIMD_TEST_ARCH_FLAGS xnn_arch_x86_avx2
#elif defined(XNN_SIMD_AVX512SKX) && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
#define XNN_SIMD_TEST_AVX512SKX 1
#if defined(__clang__)
#pragma clang attribute push(                                             \
    __attribute__((                                                       \
        target("avx512f,avx512cd,avx512bw,avx512dq,avx512vl,fma,f16c"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx512f,avx512cd,avx512bw,avx512dq,avx512vl,fma,f16c")
#endif
#include "src/xnnpack/simd/s16-avx512skx.h"
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#define XNN_SIMD_TEST_ARCH_FLAGS xnn_arch_x86_avx512skx
#elif defined(XNN_SIMD_NEON) && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
#include "src/xnnpack/simd/s16-neon.h"
#define XNN_SIMD_TEST_ARCH_FLAGS xnn_arch_arm_neon
#elif defined(XNN_SIMD_WASMSIMD) && \
    (XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD)
#include "src/xnnpack/simd/s16-wasmsimd.h"
#endif

#ifdef xnn_simd_size_s16

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/math.h"
#include "test/replicable_random_device.h"

namespace xnnpack {

class S16SimdTest : public ::testing::Test {
 protected:
  void SetUp() override {
#ifdef XNN_SIMD_TEST_ARCH_FLAGS
    TEST_REQUIRES_ARCH_FLAGS(XNN_SIMD_TEST_ARCH_FLAGS);
#endif  // XNN_SIMD_TEST_ARCH_FLAGS
    inputs_.resize(3 * xnn_simd_size_s16);
    output_.resize(xnn_simd_size_s16);
    std::uniform_int_distribution<int> s16(-100, 100);
    std::generate(inputs_.begin(), inputs_.end(), [&]() { return s16(rng_); });
  }

  xnnpack::ReplicableRandomDevice rng_;
  std::vector<int16_t> inputs_;
  std::vector<int16_t> output_;
};

#ifdef XNN_SIMD_TEST_AVX512SKX
#if defined(__clang__)
#pragma clang attribute push(                                             \
    __attribute__((                                                       \
        target("avx512f,avx512cd,avx512bw,avx512dq,avx512vl,fma,f16c"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx512f,avx512cd,avx512bw,avx512dq,avx512vl,fma,f16c")
#endif
#endif  // XNN_SIMD_TEST_AVX512SKX

TEST_F(S16SimdTest, SimdSize) {
  ASSERT_GT(xnn_simd_size_s16, 0);
  ASSERT_EQ(xnn_simd_bytes_s16, xnn_simd_size_s16 * sizeof(int16_t));
#ifdef xnn_simd_log2_size_s16
  ASSERT_EQ(xnn_simd_size_s16, static_cast<size_t>(1)
                                   << xnn_simd_log2_size_s16);
#endif  // xnn_simd_log2_size_s16
}

TEST_F(S16SimdTest, Set1) {
  for (int16_t value : {int16_t{-32768}, int16_t{-1}, int16_t{0}, int16_t{42},
                        int16_t{32767}, inputs_[0]}) {
    const xnn_simd_s16_t v = xnn_set1_s16(value);
    xnn_storeu_s16(output_.data(), v);
    for (size_t k = 0; k < xnn_simd_size_s16; k++) {
      ASSERT_EQ(output_[k], value);
    }
  }
  XNN_SIMD_CONST_S16(vconst, -1234);
  xnn_storeu_s16(output_.data(), vconst);
  for (size_t k = 0; k < xnn_simd_size_s16; k++) {
    ASSERT_EQ(output_[k], -1234);
  }
}

TEST_F(S16SimdTest, LoadStoreAligned) {
  const size_t alignment =
      std::max<size_t>(alignof(int16_t), xnn_simd_bytes_s16);
  std::vector<uint8_t> input_storage(xnn_simd_bytes_s16 + alignment);
  std::vector<uint8_t> output_storage(xnn_simd_bytes_s16 + alignment);
  auto align_ptr = [](uint8_t* ptr, size_t align) -> int16_t* {
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t rem = addr % align;
    return reinterpret_cast<int16_t*>(rem == 0 ? ptr : ptr + (align - rem));
  };
  int16_t* aligned_input = align_ptr(input_storage.data(), alignment);
  int16_t* aligned_output = align_ptr(output_storage.data(), alignment);
  std::copy_n(inputs_.begin(), xnn_simd_size_s16, aligned_input);
  const xnn_simd_s16_t v = xnn_load_s16(aligned_input);
  xnn_store_s16(aligned_output, v);
  for (size_t k = 0; k < xnn_simd_size_s16; k++) {
    ASSERT_EQ(aligned_output[k], aligned_input[k]);
  }
}

TEST_F(S16SimdTest, Min) {
  const xnn_simd_s16_t a = xnn_loadu_s16(inputs_.data());
  const xnn_simd_s16_t b = xnn_loadu_s16(inputs_.data() + xnn_simd_size_s16);
  const xnn_simd_s16_t res = xnn_min_s16(a, b);
  xnn_storeu_s16(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_s16; k++) {
    ASSERT_EQ(output_[k],
              std::min<int16_t>(inputs_[k], inputs_[k + xnn_simd_size_s16]));
  }
}

TEST_F(S16SimdTest, Max) {
  const xnn_simd_s16_t a = xnn_loadu_s16(inputs_.data());
  const xnn_simd_s16_t b = xnn_loadu_s16(inputs_.data() + xnn_simd_size_s16);
  const xnn_simd_s16_t res = xnn_max_s16(a, b);
  xnn_storeu_s16(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_s16; k++) {
    ASSERT_EQ(output_[k],
              std::max<int16_t>(inputs_[k], inputs_[k + xnn_simd_size_s16]));
  }
}

TEST_F(S16SimdTest, SignComplement) {
  const xnn_simd_s16_t a = xnn_loadu_s16(inputs_.data());
  const xnn_simd_s16_t res = xnn_signcomplement_s16(a);
  xnn_storeu_s16(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_s16; k++) {
    ASSERT_EQ(output_[k], math_signcomplement_f16((uint16_t)inputs_[k]));
  }
}

TEST_F(S16SimdTest, LoadTail) {
  for (size_t num_elements = 1; num_elements < xnn_simd_size_s16;
       num_elements++) {
    const xnn_simd_s16_t a = xnn_load_tail_s16(inputs_.data(), num_elements);
    xnn_storeu_s16(output_.data(), a);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_[k], inputs_[k]) << " " << k;
    }
    // The rest of the lanes are undefined.
  }
}

TEST_F(S16SimdTest, LoadTailSafe) {
  for (size_t num_elements = 1; num_elements < xnn_simd_size_s16;
       num_elements++) {
    const xnn_simd_s16_t a =
        xnn_load_tail_safe_s16(inputs_.data(), num_elements);
    xnn_storeu_s16(output_.data(), a);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_[k], inputs_[k]) << " " << k;
    }
  }
}

TEST_F(S16SimdTest, StoreTail) {
  const xnn_simd_s16_t a = xnn_loadu_s16(inputs_.data());
  for (size_t num_elements = 1; num_elements < xnn_simd_size_s16;
       num_elements++) {
    std::fill(output_.begin(), output_.end(), int16_t{0});
    xnn_store_tail_s16(output_.data(), a, num_elements);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_[k], inputs_[k]) << " " << k;
    }
    for (size_t k = num_elements; k < xnn_simd_size_s16; k++) {
      ASSERT_EQ(output_[k], 0);
    }
  }
}

#ifdef XNN_SIMD_TEST_AVX512SKX
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif  // XNN_SIMD_TEST_AVX512SKX

}  // namespace xnnpack

#endif  // xnn_simd_size_s16
