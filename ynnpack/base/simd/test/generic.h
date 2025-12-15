// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_TEST_GENERIC_H_
#define XNNPACK_YNNPACK_BASE_SIMD_TEST_GENERIC_H_

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"

namespace ynn {

namespace simd {

template <typename vector>
void test_broadcast(uint32_t arch_flags) {
  if (!is_arch_supported(arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  using scalar = typename vector::value_type;

  for (scalar value : {1, 2, 3}) {
    scalar dst[vector::N];
    store(dst, vector{value});
    for (size_t i = 0; i < vector::N; ++i) {
      ASSERT_EQ(dst[i], value);
    }
  }
}

#define TEST_BROADCAST(test_class, type, arch_flags) \
  TEST(test_class, broadcast_##type) { test_broadcast<type>(arch_flags); }

template <typename vector>
void test_load_store(uint32_t arch_flags) {
  if (!is_arch_supported(arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  using scalar = typename vector::value_type;
  static constexpr size_t N = vector::N;

  scalar src_aligned[N * 2];
  scalar dst_aligned[N * 2];
  for (size_t i = 0; i < N * 2; ++i) {
    src_aligned[i] = static_cast<scalar>(i);
  }
  for (size_t align = 0; align < N; ++align) {
    // Use a different alignment for src and dst.
    scalar* src = &src_aligned[align];
    scalar* dst = &dst_aligned[N - align];
    vector v = load(src, vector{});

    store(dst, v);
    for (size_t i = 0; i < N; ++i) {
      ASSERT_EQ(dst[i], src[i]);
    }
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
void test_partial_load(uint32_t arch_flags) {
  if (!is_arch_supported(arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  using scalar = typename vector::value_type;
  static constexpr size_t N = vector::N;

  scalar dst[N];
  scalar init[N];
  scalar src_aligned[N * 2];
  for (size_t i = 0; i < N; ++i) {
    init[i] = static_cast<scalar>(i * 2 + 1);
  }
  for (size_t i = 0; i < N * 2; ++i) {
    src_aligned[i] = static_cast<scalar>(i);
  }
  for (int align = 0; align < N; ++align) {
    for (size_t n = 1; n < N; ++n) {
      scalar* src = &src_aligned[N + align - n];
      vector v = load(src, load(init, vector{}), n);

      store(dst, v);
      for (size_t i = 0; i < n; ++i) {
        ASSERT_EQ(dst[i], src[i]);
      }
      for (size_t i = n; i < N; ++i) {
        ASSERT_EQ(dst[i], init[i]);
      }
    }
  }
}

template <typename vector>
void test_partial_store(uint32_t arch_flags) {
  if (!is_arch_supported(arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  using scalar = typename vector::value_type;
  static constexpr size_t N = vector::N;

  scalar src[N];
  for (size_t i = 0; i < N; ++i) {
    src[i] = static_cast<scalar>(i);
  }
  scalar dst_aligned[N * 2];
  for (size_t align = 0; align < N; ++align) {
    scalar* dst = &dst_aligned[align];
    for (size_t i = 0; i < N; ++i) {
      dst[i] = static_cast<scalar>(i + 5);
    }
    vector v = load(src, vector{});
    for (size_t n = 1; n < N; ++n) {
      store(dst, v, n);
      for (size_t i = 0; i < n; ++i) {
        ASSERT_EQ(dst[i], src[i]);
      }
      for (size_t i = n; i < N; ++i) {
        ASSERT_EQ(dst[i], static_cast<scalar>(i + 5));
      }
    }
  }
}

#define TEST_PARTIAL_LOAD_STORE(test_class, type, arch_flags) \
  TEST(test_class, partial_load_##type) {                     \
    test_partial_load<type>(arch_flags);                      \
  }                                                           \
  TEST(test_class, partial_store_##type) {                    \
    test_partial_store<type>(arch_flags);                     \
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

template <typename T>
struct copysign_op {
  T operator()(T a, T b) {
    using std::copysign;
    return copysign(a, b);
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
#define TEST_COPYSIGN(test_class, type, arch_flags) \
  TEST(test_class, copysign_##type) { test_op<type, copysign_op>(arch_flags); }
#define TEST_MIN(test_class, type, arch_flags) \
  TEST(test_class, min_##type) { test_op<type, min_op>(arch_flags); }
#define TEST_MAX(test_class, type, arch_flags) \
  TEST(test_class, max_##type) { test_op<type, max_op>(arch_flags); }
#define TEST_AND(test_class, type, arch_flags) \
  TEST(test_class, and_##type) { test_op<type, min_op>(arch_flags); }
#define TEST_OR(test_class, type, arch_flags) \
  TEST(test_class, or_##type) { test_op<type, max_op>(arch_flags); }

template <typename To, typename From, size_t... Is>
void test_extract_impl(std::index_sequence<Is...>, From from_v,
                       typename From::value_type* src) {
  (([&]() {
    constexpr size_t i = Is;
    To to_v = extract<i>(from_v, To{});
    typename To::value_type dst[To::N];
    store(dst, to_v);
    for (size_t j = 0; j < To::N; ++j) {
      ASSERT_EQ(dst[j], src[i * To::N + j]);
    }
  }()), ...);
}

template <typename To, typename From>
void test_extract(uint32_t arch_flags) {
  if (!is_arch_supported(arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  ASSERT_EQ(From::N % To::N, 0);
  using FromScalar = typename From::value_type;

  FromScalar src[From::N];
  for (size_t i = 0; i < From::N; ++i) {
    src[i] = static_cast<FromScalar>(i);
  }
  From from_v = load(src, From{});

  test_extract_impl<To, From>(std::make_index_sequence<From::N / To::N>{},
                              from_v, src);
}

#define TEST_EXTRACT(test_class, to, from, arch_flags) \
  TEST(test_class, extract_##to##_##from) {            \
    test_extract<to, from>(arch_flags);                \
  }

template <typename vector>
void test_concat(uint32_t arch_flags) {
  if (!is_arch_supported(arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  using scalar = typename vector::value_type;
  constexpr size_t N = vector::N;

  scalar src[N * 2];
  for (size_t i = 0; i < N * 2; ++i) {
    src[i] = static_cast<scalar>(i);
  }
  scalar dst[N * 2];
  store(dst, concat(load(src, vector{}), load(src + N, vector{})));

  for (size_t i = 0; i < N * 2; ++i) {
    ASSERT_EQ(dst[i], src[i]);
  }
}

#define TEST_CONCAT(test_class, vector, arch_flags) \
  TEST(test_class, concat_##vector) { test_concat<vector>(arch_flags); }

template <typename To, typename From>
void test_convert(uint32_t arch_flags) {
  if (!is_arch_supported(arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }

  using FromScalar = typename From::value_type;
  using ToScalar = typename To::value_type;
  static constexpr size_t N = To::N;

  FromScalar src[N];
  for (size_t i = 0; i < N; ++i) {
    src[i] = static_cast<FromScalar>(i);
  }
  From from_v = load(src, From{});
  To to_v = convert(from_v, ToScalar{});

  ToScalar dst[N];
  store(dst, to_v);
  for (size_t i = 0; i < N; ++i) {
    ASSERT_EQ(dst[i], static_cast<ToScalar>(src[i]));
  }
}

#define TEST_CONVERT(test_class, to, from, arch_flags) \
  TEST(test_class, convert_##to##_##from) {            \
    test_convert<to, from>(arch_flags);                \
  }

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

template <typename vector>
void test_fma(uint32_t arch_flags) {
  if (!is_arch_supported(arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  using scalar = typename vector::value_type;
  constexpr size_t N = vector::N;

  ReplicableRandomDevice rng;
  TypeGenerator<scalar> gen;
  for (auto _ : FuzzTest(std::chrono::milliseconds(100))) {
    scalar a[N];
    scalar b[N];
    scalar acc[N];
    scalar expected[N];
    for (size_t i = 0; i < N; ++i) {
      a[i] = gen(rng);
      b[i] = gen(rng);
      acc[i] = gen(rng);
      expected[i] = std::fma(a[i], b[i], acc[i]);
    }
    scalar result[N];
    store(result,
          fma(load(a, vector{}), load(b, vector{}), load(acc, vector{})));
    for (size_t i = 0; i < N; ++i) {
#ifdef YNN_ARCH_ARM32
      if (std::abs(expected[i]) < type_info<scalar>::smallest_normal()) {
        // ARM32 flushes denormals to 0(?).
        continue;
      }
#endif  // YNN_ARCH_ARM32
      ASSERT_EQ(result[i], expected[i]);
    }
  }
}

#define TEST_FMA(test_class, type, arch_flags) \
  TEST(test_class, fma_##type) { test_fma<type>(arch_flags); }

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_TEST_GENERIC_H_
