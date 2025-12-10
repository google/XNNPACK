// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_TEST_GENERIC_H_
#define XNNPACK_YNNPACK_BASE_SIMD_TEST_GENERIC_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"

namespace ynn {

namespace simd {

template <typename vector>
void test_load_store(uint32_t arch_flags) {
  if (!is_arch_supported(arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  using scalar = typename vector::value_type;
  static constexpr size_t N = vector::N;

  scalar src[N];
  for (size_t i = 0; i < N; ++i) {
    src[i] = static_cast<scalar>(i);
  }
  vector v = load(src, vector{});

  scalar dst[N];
  store(dst, v);
  for (size_t i = 0; i < N; ++i) {
    ASSERT_EQ(dst[i], src[i]);
  }
}

#define TEST_LOAD_STORE(test_class, type, arch_flags) \
  TEST(test_class, load_store_##type) { test_load_store<type>(arch_flags); }

template <typename vector>
void test_aligned_load_store(uint32_t arch_flags) {
  if (!is_arch_supported(arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  using scalar = typename vector::value_type;
  static constexpr size_t N = vector::N;

  alignas(vector) scalar src[N];
  for (size_t i = 0; i < N; ++i) {
    src[i] = static_cast<scalar>(i);
  }
  vector v = load_aligned(src, vector{});

  alignas(vector) scalar dst[N];
  store_aligned(dst, v);
  for (size_t i = 0; i < N; ++i) {
    ASSERT_EQ(dst[i], src[i]);
  }
}

#define TEST_ALIGNED_LOAD_STORE(test_class, type, arch_flags) \
  TEST(test_class, aligned_load_store_##type) {               \
    test_aligned_load_store<type>(arch_flags);                \
  }

template <typename vector>
void test_partial_load_store(uint32_t arch_flags) {
  if (!is_arch_supported(arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  using scalar = typename vector::value_type;
  static constexpr size_t N = vector::N;

  for (size_t n = 1; n < N; ++n) {
    std::vector<scalar> src(n);
    for (size_t i = 0; i < n; ++i) {
      src[i] = static_cast<scalar>(i);
    }
    vector v = load(src.data(), vector{}, n);

    std::vector<scalar> dst(n);
    store(dst.data(), v, n);
    for (size_t i = 0; i < n; ++i) {
      ASSERT_EQ(dst[i], src[i]);
    }
  }
}

#define TEST_PARTIAL_LOAD_STORE(test_class, type, arch_flags) \
  TEST(test_class, partial_load_store_##type) {               \
    test_partial_load_store<type>(arch_flags);                \
  }

template <typename T, template <typename> typename Op>
void test_op(uint32_t arch_flags) {
  if (!is_arch_supported(arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  typename T::value_type a[T::N];
  typename T::value_type b[T::N];
  for (size_t i = 0; i < T::N; ++i) {
    a[i] = i * 2;
    b[i] = T::N / 2 - i;
  }

  typename T::value_type result[T::N];

  Op<T> vector_op;
  Op<typename T::value_type> scalar_op;

  store(result, vector_op(load(a, T{}), load(b, T{})));
  for (size_t i = 0; i < T::N; ++i) {
    ASSERT_EQ(result[i], scalar_op(a[i], b[i]));
  }
}

template <typename T>
struct min_op {
  T operator()(T a, T b) {
    using std::min;
    return min(a, b);
  }
};

template <typename T>
struct max_op {
  T operator()(T a, T b) {
    using std::max;
    return max(a, b);
  }
};

#define TEST_ADD(test_class, type, arch_flags) \
  TEST(test_class, add_##type) { test_op<type, std::plus>(arch_flags); }
#define TEST_SUBTRACT(test_class, type, arch_flags) \
  TEST(test_class, subtract_##type) { test_op<type, std::minus>(arch_flags); }
#define TEST_MULTIPLY(test_class, type, arch_flags) \
  TEST(test_class, multiply_##type) {               \
    test_op<type, std::multiplies>(arch_flags);     \
  }
#define TEST_MIN(test_class, type, arch_flags) \
  TEST(test_class, min_##type) { test_op<type, min_op>(arch_flags); }
#define TEST_MAX(test_class, type, arch_flags) \
  TEST(test_class, max_##type) { test_op<type, max_op>(arch_flags); }
#define TEST_AND(test_class, type, arch_flags) \
  TEST(test_class, and_##type) { test_op<type, min_op>(arch_flags); }
#define TEST_OR(test_class, type, arch_flags) \
  TEST(test_class, or_##type) { test_op<type, max_op>(arch_flags); }

// This function has a max of n at n, and descends to 0 at either 0 or 2*n - 1.
// This allows us to test a horizontal min/max reduction where any one of n
// lanes is the min or max.
inline int tent(int x, int n) { return std::min(x, 2 * n - 1 - x); }

template <typename vector>
void test_horizontal_min(uint32_t arch_flags) {
  if (!is_arch_supported(arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  constexpr int N = vector::N;
  typename vector::value_type a[N * 2 - 1];
  for (int i = 0; i < N * 2 - 1; ++i) {
    a[i] = N - tent(i, N);
  }
  for (int i = 0; i < N - 1; ++i) {
    ASSERT_EQ(N - tent(N, N), horizontal_min(load(&a[i], vector{})));
  }
}

template <typename vector>
void test_horizontal_max(uint32_t arch_flags) {
  if (!is_arch_supported(arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  constexpr int N = vector::N;
  typename vector::value_type a[N * 2 - 1];
  for (int i = 0; i < N * 2 - 1; ++i) {
    a[i] = tent(i, N);
  }
  for (int i = 0; i < N - 1; ++i) {
    ASSERT_EQ(tent(N, N), horizontal_max(load(&a[i], vector{})));
  }
}

#define TEST_HORIZONTAL_MIN(test_class, type, arch_flags) \
  TEST(test_class, horizontal_min_##type) {               \
    test_horizontal_min<type>(arch_flags);                \
  }
#define TEST_HORIZONTAL_MAX(test_class, type, arch_flags) \
  TEST(test_class, horizontal_max_##type) {               \
    test_horizontal_max<type>(arch_flags);                \
  }

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_TEST_GENERIC_H_
