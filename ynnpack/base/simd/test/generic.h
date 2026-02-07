// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_TEST_GENERIC_H_
#define XNNPACK_YNNPACK_BASE_SIMD_TEST_GENERIC_H_

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>
#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"

namespace ynn {

namespace simd {

template <typename scalar, size_t N>
void test_broadcast() {
  for (scalar value : {1, 2, 3}) {
    scalar dst[N];
    store(dst, broadcast<N>(value));
    for (size_t i = 0; i < N; ++i) {
      ASSERT_EQ(dst[i], value);
    }
  }
}

#define TEST_BROADCAST(test_class, type, N) \
  TEST_F(test_class, broadcast_##type##x##N) { test_broadcast<type, N>(); }

template <typename scalar, size_t N>
void test_load_store() {
  scalar src_aligned[N * 2];
  scalar dst_aligned[N * 2];
  for (size_t i = 0; i < N * 2; ++i) {
    src_aligned[i] = static_cast<scalar>(i);
  }
  for (size_t align = 0; align < N; ++align) {
    // Use a different alignment for src and dst.
    const scalar* src = &src_aligned[align];
    scalar* dst = &dst_aligned[N - align];
    auto v = load(src, std::integral_constant<size_t, N>{});

    store(dst, v);
    for (size_t i = 0; i < N; ++i) {
      ASSERT_EQ(dst[i], src[i]);
    }
  }
}

#define TEST_LOAD_STORE(test_class, type, N) \
  TEST_F(test_class, load_store_##type##x##N) { test_load_store<type, N>(); }

template <typename scalar, size_t N>
void test_aligned_load_store() {
  using vector = vec<scalar, N>;

  alignas(vector) scalar src[N];
  for (size_t i = 0; i < N; ++i) {
    src[i] = static_cast<scalar>(i);
  }
  vector v = load_aligned(src, vector::N);

  alignas(vector) scalar dst[N];
  store_aligned(dst, v);
  for (size_t i = 0; i < N; ++i) {
    ASSERT_EQ(dst[i], src[i]);
  }
}

#define TEST_ALIGNED_LOAD_STORE(test_class, type, N)  \
  TEST_F(test_class, aligned_load_store_##type##x##N) { \
    test_aligned_load_store<type, N>();               \
  }

template <typename scalar, size_t N>
void test_partial_load() {
  using vector = vec<scalar, N>;

  scalar init[N];
  alignas(vector) scalar src_aligned[N * 2];
  for (size_t i = 0; i < N; ++i) {
    init[i] = static_cast<scalar>(N - 1 - i);
  }
  for (size_t i = 0; i < N * 2; ++i) {
    src_aligned[i] = static_cast<scalar>(i);
  }
  for (int align : {0, static_cast<int>(N) - 1}) {
    for (size_t n = 1; n < N; ++n) {
      scalar* src = &src_aligned[N + align - n];
      vector v = load(src, n, load(init, vector::N));

      scalar dst[N];
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

template <typename scalar, size_t N>
void test_partial_store() {
  using vector = vec<scalar, N>;

  scalar src[N];
  for (size_t i = 0; i < N; ++i) {
    src[i] = static_cast<scalar>(i);
  }
  alignas(vector) scalar dst_aligned[N * 2];
  for (int align : {0, static_cast<int>(N) - 1}) {
    scalar* dst = &dst_aligned[align];
    for (size_t i = 0; i < N; ++i) {
      dst[i] = static_cast<scalar>(i + 5);
    }
    vector v = load(src, vector::N);
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

#define TEST_PARTIAL_LOAD_STORE(test_class, type, N) \
  TEST_F(test_class, partial_load_##type##x##N) {      \
    test_partial_load<type, N>();                    \
  }                                                  \
  TEST_F(test_class, partial_store_##type##x##N) {     \
    test_partial_store<type, N>();                   \
  }

template <typename scalar, size_t N, template <typename> typename Op>
void test_op() {
  using vector = vec<scalar, N>;

  scalar a[vector::N];
  scalar b[vector::N];
  for (size_t i = 0; i < vector::N; ++i) {
    a[i] = i * 2;
    b[i] = vector::N / 2 - i;
  }

  scalar result[vector::N];

  Op<vector> vector_op;
  Op<scalar> scalar_op;

  store(result, vector_op(load(a, vector::N), load(b, vector::N)));
  for (size_t i = 0; i < vector::N; ++i) {
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

#define TEST_ADD(test_class, type, N) \
  TEST_F(test_class, add_##type##x##N) { test_op<type, N, std::plus>(); }
#define TEST_SUBTRACT(test_class, type, N) \
  TEST_F(test_class, subtract_##type##x##N) { test_op<type, N, std::minus>(); }
#define TEST_MULTIPLY(test_class, type, N)  \
  TEST_F(test_class, multiply_##type##x##N) { \
    test_op<type, N, std::multiplies>();    \
  }
#define TEST_COPYSIGN(test_class, type, N) \
  TEST_F(test_class, copysign_##type##x##N) { test_op<type, N, copysign_op>(); }
#define TEST_MIN(test_class, type, N) \
  TEST_F(test_class, min_##type##x##N) { test_op<type, N, min_op>(); }
#define TEST_MAX(test_class, type, N) \
  TEST_F(test_class, max_##type##x##N) { test_op<type, N, max_op>(); }
#define TEST_AND(test_class, type, N) \
  TEST_F(test_class, and_##type##x##N) { test_op<type, N, min_op>(); }
#define TEST_OR(test_class, type, N) \
  TEST_F(test_class, or_##type##x##N) { test_op<type, N, max_op>(); }

template <size_t Lanes, typename From, size_t... Is>
void test_extract_impl(std::index_sequence<Is...>, From from_v,
                       typename From::value_type* src) {
  (([&]() {
    constexpr size_t i = Is;
    auto to_v = extract<i>(from_v, std::integral_constant<size_t, Lanes>());
    typename From::value_type dst[Lanes];
    store(dst, to_v);
    for (size_t j = 0; j < Lanes; ++j) {
      ASSERT_EQ(dst[j], src[i * Lanes + j]);
    }
  }()), ...);
}

template <typename From, size_t Lanes>
void test_extract() {
  ASSERT_EQ(From::N % Lanes, 0);
  using FromScalar = typename From::value_type;

  FromScalar src[From::N];
  for (size_t i = 0; i < From::N; ++i) {
    src[i] = static_cast<FromScalar>(i);
  }
  From from_v = load(src, From::N);

  test_extract_impl<Lanes, From>(std::make_index_sequence<From::N / Lanes>{},
                                 from_v, src);
}

#define TEST_EXTRACT(test_class, from, lanes) \
  TEST_F(test_class, extract_##from##_##lanes) { test_extract<from, lanes>(); }

template <typename vector>
void test_concat() {
  using scalar = typename vector::value_type;
  constexpr size_t N = vector::N;

  scalar src[N * 2];
  for (size_t i = 0; i < N * 2; ++i) {
    src[i] = static_cast<scalar>(i);
  }
  scalar dst[N * 2];
  store(dst, concat(load(src, vector::N), load(src + N, vector::N)));

  for (size_t i = 0; i < N * 2; ++i) {
    ASSERT_EQ(dst[i], src[i]);
  }
}

#define TEST_CONCAT(test_class, vector) \
  TEST_F(test_class, concat_##vector) { test_concat<vector>(); }

template <typename To, typename From>
void test_convert() {
  using FromScalar = typename From::value_type;
  static constexpr size_t N = From::N;

  FromScalar src[N];
  for (size_t i = 0; i < N; ++i) {
    src[i] = static_cast<FromScalar>(i);
  }
  From from_v = load(src, From::N);
  auto to_v = convert(from_v, To{});

  To dst[N];
  store(dst, to_v);
  for (size_t i = 0; i < N; ++i) {
    ASSERT_EQ(dst[i], static_cast<To>(src[i]));
  }
}

#define TEST_CONVERT(test_class, to, from) \
  TEST_F(test_class, convert_##to##_##from) { test_convert<to, from>(); }

template <typename scalar, size_t N>
void test_horizontal_sum() {
  using vector = vec<scalar, N>;

  scalar a[N * 2] = {};
  std::fill_n(&a[N], N, static_cast<scalar>(1));
  // Note we test N + 1 values here, so we cover both the all 0 and all 1 cases.
  for (int i = 0; i <= N; ++i) {
#ifdef YNN_ARCH_HEXAGON
    // Hexagon does this reduction with qfloats, which do not compute this
    // exactly.
    ASSERT_NEAR(i, horizontal_sum(load(&a[i], vector::N)), 1e-5f);
#else
    ASSERT_EQ(i, horizontal_sum(load(&a[i], vector::N)));
#endif  // YNN_ARCH_HEXAGON
  }
}

// This function has a max of n at n, and descends to 0 at either 0 or 2*n - 1.
// This allows us to test a horizontal min/max reduction where any one of n
// lanes is the min or max.
inline int tent(int x, int n) { return std::min(x, 2 * n - 1 - x); }

template <typename scalar, size_t N>
void test_horizontal_min() {
  using vector = vec<scalar, N>;

  scalar a[N * 2 - 1];
  for (int i = 0; i < N * 2 - 1; ++i) {
    a[i] = N - tent(i, N);
  }
  for (int i = 0; i < N - 1; ++i) {
    ASSERT_EQ(N - tent(N, N), horizontal_min(load(&a[i], vector::N)));
  }
}

template <typename scalar, size_t N>
void test_horizontal_max() {
  using vector = vec<scalar, N>;

  scalar a[N * 2 - 1];
  for (int i = 0; i < N * 2 - 1; ++i) {
    a[i] = tent(i, N);
  }
  for (int i = 0; i < N - 1; ++i) {
    ASSERT_EQ(tent(N, N), horizontal_max(load(&a[i], vector::N)));
  }
}

#define TEST_HORIZONTAL_SUM(test_class, type, N)  \
  TEST_F(test_class, horizontal_sum_##type##x##N) { \
    test_horizontal_sum<type, N>();               \
  }
#define TEST_HORIZONTAL_MIN(test_class, type, N)  \
  TEST_F(test_class, horizontal_min_##type##x##N) { \
    test_horizontal_min<type, N>();               \
  }
#define TEST_HORIZONTAL_MAX(test_class, type, N)  \
  TEST_F(test_class, horizontal_max_##type##x##N) { \
    test_horizontal_max<type, N>();               \
  }

template <typename scalar, size_t N>
void test_fma() {
  using vector = vec<scalar, N>;

  ReplicableRandomDevice rng;
  for (auto _ : FuzzTest(std::chrono::milliseconds(100))) {
    scalar a[N];
    scalar b[N];
    scalar acc[N];
    scalar expected[N];
    for (size_t i = 0; i < N; ++i) {
      a[i] = random_normal_float<scalar>(rng);
      b[i] = random_normal_float<scalar>(rng);
      acc[i] = random_normal_float<scalar>(rng);
      expected[i] = std::fma(a[i], b[i], acc[i]);
    }
    scalar result[N];
    store(result,
          fma(load(a, vector::N), load(b, vector::N), load(acc, vector::N)));
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

#define TEST_FMA(test_class, type, N) \
  TEST_F(test_class, fma_##type##x##N) { test_fma<type, N>(); }

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_TEST_GENERIC_H_
