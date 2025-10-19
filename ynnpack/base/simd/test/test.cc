// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/build_config.h"
#if defined(YNN_ARCH_X86_AVX512F) || defined(YNN_ARCH_X86_AVX512BW) || \
    defined(YNN_ARCH_X86_AVX512BF16) || defined(YNN_ARCH_X86_AVX512FP16)
#include "ynnpack/base/simd/x86_avx512.h"
#elif defined(YNN_ARCH_X86_AVX2)
#include "ynnpack/base/simd/x86_avx2.h"
#elif defined(YNN_ARCH_X86_AVX) || defined(YNN_ARCH_X86_F16C)
#include "ynnpack/base/simd/x86_avx.h"
#elif defined(YNN_ARCH_X86_SSE41)
#include "ynnpack/base/simd/x86_sse41.h"
#elif defined(YNN_ARCH_X86_SSE2)
#include "ynnpack/base/simd/x86_sse.h"
#endif
#if defined(YNN_ARCH_ARM_NEON)
#include "ynnpack/base/simd/arm.h"
#endif


namespace ynn {

namespace simd {

#ifdef YNN_ARCH_X86_SSE2
using types = testing::Types<f32x4>;
constexpr int required_arch_flags = arch_flag::sse2;
#endif
#ifdef YNN_ARCH_X86_SSE41
using types = testing::Types<s32x4, u8x16, s8x16>;
constexpr int required_arch_flags = arch_flag::sse41;
#endif
#ifdef YNN_ARCH_X86_AVX
using types = testing::Types<f32x8>;
constexpr int required_arch_flags = arch_flag::avx;
#endif
#ifdef YNN_ARCH_X86_AVX2
using types = testing::Types<s32x8, u8x32, s8x32>;
constexpr int required_arch_flags = arch_flag::avx2;
#endif
#ifdef YNN_ARCH_X86_AVX512F
using types = testing::Types<f32x16, s32x16>;
constexpr int required_arch_flags = arch_flag::avx512f;
#endif
#ifdef YNN_ARCH_X86_AVX512BW
using types = testing::Types<u8x64, s8x64>;
constexpr int required_arch_flags = arch_flag::avx512f | arch_flag::avx512bw;
#endif
#ifdef YNN_ARCH_ARM_NEON
using types = testing::Types<f32x4, s32x4, u8x16, s8x16>;
constexpr int required_arch_flags = arch_flag::neon;
#endif

template <typename T>
class ARCH : public testing::Test {};

TYPED_TEST_SUITE(ARCH, types);

TYPED_TEST(ARCH, load_store) {
  if (!is_arch_supported(required_arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  using vector = TypeParam;
  using scalar = typename TypeParam::value_type;
  static constexpr size_t N = TypeParam::N;

  scalar src[N];
  std::iota(src, src + N, static_cast<scalar>(0));
  vector v = load(src, vector{});

  scalar dst[N];
  store(dst, v);
  for (size_t i = 0; i < N; ++i) {
    ASSERT_EQ(dst[i], src[i]);
  }
}

TYPED_TEST(ARCH, aligned_load_store) {
  if (!is_arch_supported(required_arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  using vector = TypeParam;
  using scalar = typename TypeParam::value_type;
  static constexpr size_t N = TypeParam::N;

  alignas(vector) scalar src[N];
  std::iota(src, src + N, static_cast<scalar>(0));
  vector v = load_aligned(src, vector{});

  alignas(vector) scalar dst[N];
  store_aligned(dst, v);
  for (size_t i = 0; i < N; ++i) {
    ASSERT_EQ(dst[i], src[i]);
  }
}

TYPED_TEST(ARCH, partial_load_store) {
  if (!is_arch_supported(required_arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  using vector = TypeParam;
  using scalar = typename TypeParam::value_type;
  static constexpr size_t N = TypeParam::N;

  for (size_t n = 1; n < N; ++n) {
    std::vector<scalar> src(n);
    std::iota(src.begin(), src.end(), 0);
    vector v = load(src.data(), vector{}, n);

    std::vector<scalar> dst(n);
    store(dst.data(), v, n);
    for (size_t i = 0; i < n; ++i) {
      ASSERT_EQ(dst[i], src[i]);
    }
  }
}

template <typename T, template <typename> typename Op>
void test_op() {
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

TYPED_TEST(ARCH, add) {
  if (!is_arch_supported(required_arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  test_op<TypeParam, std::plus>();
}

TYPED_TEST(ARCH, subtract) {
  if (!is_arch_supported(required_arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  test_op<TypeParam, std::minus>();
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

TYPED_TEST(ARCH, min) {
  if (!is_arch_supported(required_arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  test_op<TypeParam, min_op>();
}

TYPED_TEST(ARCH, max) {
  if (!is_arch_supported(required_arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  test_op<TypeParam, max_op>();
}

namespace {

// This function has a max of n at n, and descends to 0 at either 0 or 2*n - 1.
// This allows us to test a horizontal min/max reduction where any one of n
// lanes is the min or max.
int tent(int x, int n) { return std::min(x, 2 * n - 1 - x); }

}  // namespace

TYPED_TEST(ARCH, horizontal_min) {
  if (!is_arch_supported(required_arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  constexpr int N = TypeParam::N;
  typename TypeParam::value_type a[N * 2 - 1];
  for (int i = 0; i < N * 2 - 1; ++i) {
    a[i] = N - tent(i, N);
  }
  for (int i = 0; i < N - 1; ++i) {
    ASSERT_EQ(N - tent(N, N), horizontal_min(load(&a[i], TypeParam{})));
  }
}

TYPED_TEST(ARCH, horizontal_max) {
  if (!is_arch_supported(required_arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  constexpr int N = TypeParam::N;
  typename TypeParam::value_type a[N * 2 - 1];
  for (int i = 0; i < N * 2 - 1; ++i) {
    a[i] = tent(i, N);
  }
  for (int i = 0; i < N - 1; ++i) {
    ASSERT_EQ(tent(N, N), horizontal_max(load(&a[i], TypeParam{})));
  }
}

}  // namespace simd

}  // namespace ynn
