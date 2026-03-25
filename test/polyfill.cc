// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/common.h"

#if XNN_ARCH_X86 || XNN_ARCH_X86_64

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/intrinsics-polyfill.h"

namespace xnnpack {

TEST(polyfill, _mm256_dpbusd_epi32_madd_kzp2) {
#if defined(__AVX2__)
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_x86_avx2);

  for (uint32_t u8_val = 0; u8_val < 256; ++u8_val) {
    for (uint32_t u2_val = 0; u2_val < 4; ++u2_val) {
      __m256i vacc = _mm256_set1_epi32(100);
      __m256i vu8 = _mm256_set1_epi8(static_cast<uint8_t>(u8_val));
      __m256i vu2 = _mm256_set1_epi8(static_cast<uint8_t>(u2_val));

      __m256i vres = _mm256_dpbusd_epi32_madd_kzp2(vacc, vu8, vu2);

      int32_t res[8];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(res), vres);

      // Reference formula: result = acc + 4 * (u8 * (u2 - 2))
      // Since all 4 elements in a lane are the same.
      int32_t expected = 100 + 4 * (static_cast<int32_t>(u8_val) * (static_cast<int32_t>(u2_val) - 2));

      for (int i = 0; i < 8; ++i) {
        ASSERT_EQ(res[i], expected) << "u8=" << u8_val << ", u2=" << u2_val << ", index=" << i;
      }
    }
  }
#else
  GTEST_SKIP();
#endif
}

TEST(polyfill, _mm512_dpbusd_epi32_madd) {
#if defined(__AVX512BW__)
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_x86_avx512skx);

  for (uint32_t u8_val = 0; u8_val < 256; ++u8_val) {
    for (uint32_t u4_val = 0; u4_val < 16; ++u4_val) {
      __m512i vacc = _mm512_set1_epi32(100);
      __m512i vu8 = _mm512_set1_epi8(static_cast<uint8_t>(u8_val));
      __m512i vu4 = _mm512_set1_epi8(static_cast<uint8_t>(u4_val));

      __m512i vres = _mm512_dpbusd_epi32_madd(vacc, vu8, vu4);

      int32_t res[16];
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(res), vres);

      // Reference formula: result = acc + 4 * (u8 * (u4 - 8))
      int32_t expected = 100 + 4 * (static_cast<int32_t>(u8_val) * (static_cast<int32_t>(u4_val) - 8));

      for (int i = 0; i < 16; ++i) {
        ASSERT_EQ(res[i], expected) << "u8=" << u8_val << ", u4=" << u4_val << ", index=" << i;
      }
    }
  }
#else
  GTEST_SKIP();
#endif
}

TEST(polyfill, _mm256_dpbusd_offset_epi32_madd) {
#if defined(__AVX2__)
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_x86_avx2);

  for (uint32_t u8_val = 0; u8_val < 256; ++u8_val) {
    for (uint32_t u4_val = 0; u4_val < 16; ++u4_val) {
      __m256i vacc = _mm256_set1_epi32(100);
      __m256i vu8 = _mm256_set1_epi8(static_cast<uint8_t>(u8_val));
      __m256i vu4 = _mm256_set1_epi8(static_cast<uint8_t>(u4_val));

      __m256i vres = _mm256_dpbusd_offset_epi32_madd(vacc, vu8, vu4, 8);

      int32_t res[8];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(res), vres);

      // Reference formula: result = acc + 4 * (u8 * (u4 - 8))
      int32_t expected = 100 + 4 * (static_cast<int32_t>(u8_val) * (static_cast<int32_t>(u4_val) - 8));

      for (int i = 0; i < 8; ++i) {
        ASSERT_EQ(res[i], expected) << "u8=" << u8_val << ", u4=" << u4_val << ", index=" << i;
      }
    }
  }
#else
  GTEST_SKIP();
#endif
}

TEST(polyfill, _mm_dpbusd_epi32_madd) {
#if defined(__SSSE3__)
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_x86_ssse3);

  for (uint32_t u8_val = 0; u8_val < 256; ++u8_val) {
    for (uint32_t u4_val = 0; u4_val < 16; ++u4_val) {
      __m128i vacc = _mm_set1_epi32(100);
      __m128i vu8 = _mm_set1_epi8(static_cast<uint8_t>(u8_val));
      __m128i vu4 = _mm_set1_epi8(static_cast<uint8_t>(u4_val));

      __m128i vres = _mm_dpbusd_epi32_madd(vacc, vu8, vu4);

      int32_t res[4];
      _mm_storeu_si128(reinterpret_cast<__m128i*>(res), vres);

      // Reference formula: result = acc + 4 * (u8 * (u4 - 8))
      int32_t expected = 100 + 4 * (static_cast<int32_t>(u8_val) * (static_cast<int32_t>(u4_val) - 8));

      for (int i = 0; i < 4; ++i) {
        ASSERT_EQ(res[i], expected) << "u8=" << u8_val << ", u4=" << u4_val << ", index=" << i;
      }
    }
  }
#else
  GTEST_SKIP();
#endif
}

}  // namespace xnnpack

#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
