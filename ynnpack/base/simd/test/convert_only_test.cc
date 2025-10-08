// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/build_config.h"
#if defined(YNN_ARCH_X86_AVX512F) || defined(YNN_ARCH_X86_AVX512BW) || \
    defined(YNN_ARCH_X86_AVX512BF16) || defined(YNN_ARCH_X86_AVX512FP16)
#include "ynnpack/base/simd/x86_avx.h"
#include "ynnpack/base/simd/x86_avx512.h"
#include "ynnpack/base/simd/x86_sse.h"
#elif defined(YNN_ARCH_X86_AVX) || defined(YNN_ARCH_X86_AVX2) || \
    defined(YNN_ARCH_X86_F16C)
#include "ynnpack/base/simd/x86_avx.h"
#include "ynnpack/base/simd/x86_sse.h"
#elif defined(YNN_ARCH_X86_SSE2) || defined(YNN_ARCH_X86_SSE41)
#include "ynnpack/base/simd/x86_sse.h"
#endif
#if defined(YNN_ARCH_ARM_NEON)
#include "ynnpack/base/simd/arm.h"
#endif

namespace ynn {

namespace simd {

#ifdef YNN_ARCH_X86_SSE2
using types = testing::Types<bf16x8, f16x8, s16x8>;
constexpr int required_arch_flags = arch_flag::sse2;
#endif
#ifdef YNN_ARCH_X86_AVX
using types = testing::Types<bf16x8, f16x8, s16x8>;
constexpr int required_arch_flags = arch_flag::avx;
#endif
#ifdef YNN_ARCH_X86_AVX2
using types = testing::Types<bf16x16, f16x16, s16x16, f32x8>;
constexpr int required_arch_flags = arch_flag::avx2;
#endif
#ifdef YNN_ARCH_X86_F16C
using types = testing::Types<f16x8>;
constexpr int required_arch_flags = arch_flag::f16c;
#endif
#ifdef YNN_ARCH_X86_AVX512
using types = testing::Types<bf16x32, f16x32, s16x32>;
constexpr int required_arch_flags = arch_flag::avx512;
#endif
#ifdef YNN_ARCH_X86_AVX512BF16
using types = testing::Types<bf16x32>;
constexpr int required_arch_flags = arch_flag::avx512bf16;
#endif
#ifdef YNN_ARCH_X86_AVX512FP16
using types = testing::Types<f16x16>;
constexpr int required_arch_flags = arch_flag::avx512fp16;
#endif
#ifdef YNN_ARCH_ARM_NEON
using types = testing::Types<bf16x8, f16x8, s16x8, f32x4>;
constexpr int required_arch_flags = arch_flag::neon;
#endif

#define PASTE_IMPL(prefix, suffix) prefix##suffix
#define PASTE(prefix, suffix) PASTE_IMPL(prefix, suffix)
#define ARCH_CONVERT_ONLY PASTE(ARCH, _CONVERT_ONLY)

template <typename T>
class ARCH_CONVERT_ONLY : public testing::Test {};

TYPED_TEST_SUITE(ARCH_CONVERT_ONLY, types);

TYPED_TEST(ARCH_CONVERT_ONLY, load_store) {
  if (!is_arch_supported(required_arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  using vector = TypeParam;
  using scalar = typename TypeParam::value_type;
  static constexpr size_t N = TypeParam::N;

  scalar src[N];
  std::iota(src, src + N, 0);
  vector v = load(src, vector{});

  scalar dst[N];
  store(dst, v);
  for (size_t i = 0; i < N; ++i) {
    ASSERT_EQ(dst[i], src[i]);
  }
}

TYPED_TEST(ARCH_CONVERT_ONLY, aligned_load_store) {
  if (!is_arch_supported(required_arch_flags)) {
    GTEST_SKIP() << "Unsupported architecture";
  }
  using vector = TypeParam;
  using scalar = typename TypeParam::value_type;
  static constexpr size_t N = TypeParam::N;

  alignas(vector) scalar src[N];
  std::iota(src, src + N, 0);
  vector v = load_aligned(src, vector{});

  alignas(vector) scalar dst[N];
  store_aligned(dst, v);
  for (size_t i = 0; i < N; ++i) {
    ASSERT_EQ(dst[i], src[i]);
  }
}

TYPED_TEST(ARCH_CONVERT_ONLY, partial_load_store) {
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

}  // namespace simd

}  // namespace ynn
