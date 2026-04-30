// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x16-packw/avx.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/packw.h"
#include "src/xnnpack/unaligned.h"


void xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* weights,
  const uint16_t* bias,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 8);   // This kernel is for NR=8
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const uint16_t* b = bias;
  uint16_t* packed_w = packed_weights;
  do {
    // NC main loop multiple of 8
    const uint16_t* w0 = weights;
    size_t n = nc;

    for (; n >= 8; n -= 8) {
      if XNN_LIKELY(b != NULL) {
        const __m128i vb0 = _mm_loadu_si128((const __m128i*) (b + 0));
        _mm_storeu_si128((__m128i*) (packed_w + 0), vb0);
        b += 8;
      } else {
        const __m128i vzero = _mm_setzero_si128();
        _mm_storeu_si128((__m128i*) (packed_w + 0), vzero);
      }
      packed_w += 8;

      const uint16_t* w1 = w0 + kc;
      const uint16_t* w2 = w1 + kc;
      const uint16_t* w3 = w2 + kc;
      const uint16_t* w4 = w3 + kc;
      const uint16_t* w5 = w4 + kc;
      const uint16_t* w6 = w5 + kc;
      const uint16_t* w7 = w6 + kc;

      // KC main loop multiple of 16
      size_t k = kc;
      for (; k >= 16; k -= 16) {
        __m256i v0 = _mm256_loadu_si256((const __m256i*) w0);
        w0 += 16;
        __m256i v1 = _mm256_loadu_si256((const __m256i*) w1);
        w1 += 16;
        __m256i v2 = _mm256_loadu_si256((const __m256i*) w2);
        w2 += 16;
        __m256i v3 = _mm256_loadu_si256((const __m256i*) w3);
        w3 += 16;
        __m256i v4 = _mm256_loadu_si256((const __m256i*) w4);
        w4 += 16;
        __m256i v5 = _mm256_loadu_si256((const __m256i*) w5);
        w5 += 16;
        __m256i v6 = _mm256_loadu_si256((const __m256i*) w6);
        w6 += 16;
        __m256i v7 = _mm256_loadu_si256((const __m256i*) w7);
        w7 += 16;

        const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
        const __m256i t1 = _mm256_unpackhi_epi16(v0, v1);
        const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
        const __m256i t3 = _mm256_unpackhi_epi16(v2, v3);
        const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
        const __m256i t5 = _mm256_unpackhi_epi16(v4, v5);
        const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);
        const __m256i t7 = _mm256_unpackhi_epi16(v6, v7);

        const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
        const __m256i u1 = _mm256_unpackhi_epi32(t0, t2);
        const __m256i u2 = _mm256_unpacklo_epi32(t1, t3);
        const __m256i u3 = _mm256_unpackhi_epi32(t1, t3);
        const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);
        const __m256i u5 = _mm256_unpackhi_epi32(t4, t6);
        const __m256i u6 = _mm256_unpacklo_epi32(t5, t7);
        const __m256i u7 = _mm256_unpackhi_epi32(t5, t7);

        const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
        const __m256i w1 = _mm256_unpackhi_epi64(u0, u4);
        const __m256i w2 = _mm256_unpacklo_epi64(u1, u5);
        const __m256i w3 = _mm256_unpackhi_epi64(u1, u5);
        const __m256i w4 = _mm256_unpacklo_epi64(u2, u6);
        const __m256i w5 = _mm256_unpackhi_epi64(u2, u6);
        const __m256i w6 = _mm256_unpacklo_epi64(u3, u7);
        const __m256i w7 = _mm256_unpackhi_epi64(u3, u7);

        _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
        _mm_storeu_si128((__m128i*) (packed_w + 64), _mm256_extracti128_si256(w0, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w1));
        _mm_storeu_si128((__m128i*) (packed_w + 72), _mm256_extracti128_si256(w1, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w2));
        _mm_storeu_si128((__m128i*) (packed_w + 80), _mm256_extracti128_si256(w2, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w3));
        _mm_storeu_si128((__m128i*) (packed_w + 88), _mm256_extracti128_si256(w3, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w4));
        _mm_storeu_si128((__m128i*) (packed_w + 96), _mm256_extracti128_si256(w4, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w5));
        _mm_storeu_si128((__m128i*) (packed_w + 104), _mm256_extracti128_si256(w5, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w6));
        _mm_storeu_si128((__m128i*) (packed_w + 112), _mm256_extracti128_si256(w6, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w7));
        _mm_storeu_si128((__m128i*) (packed_w + 120), _mm256_extracti128_si256(w7, 1));
        packed_w += 128;
      }

      // KC remainder (1..15)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 15);

        if (k & 8) {
          __m256i v0 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w0));
          w0 += 8;
          __m256i v1 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w1));
          w1 += 8;
          __m256i v2 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w2));
          w2 += 8;
          __m256i v3 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w3));
          w3 += 8;
          __m256i v4 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w4));
          w4 += 8;
          __m256i v5 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w5));
          w5 += 8;
          __m256i v6 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w6));
          w6 += 8;
          __m256i v7 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w7));
          w7 += 8;

          const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
          const __m256i t1 = _mm256_unpackhi_epi16(v0, v1);
          const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
          const __m256i t3 = _mm256_unpackhi_epi16(v2, v3);
          const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
          const __m256i t5 = _mm256_unpackhi_epi16(v4, v5);
          const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);
          const __m256i t7 = _mm256_unpackhi_epi16(v6, v7);

          const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
          const __m256i u1 = _mm256_unpackhi_epi32(t0, t2);
          const __m256i u2 = _mm256_unpacklo_epi32(t1, t3);
          const __m256i u3 = _mm256_unpackhi_epi32(t1, t3);
          const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);
          const __m256i u5 = _mm256_unpackhi_epi32(t4, t6);
          const __m256i u6 = _mm256_unpacklo_epi32(t5, t7);
          const __m256i u7 = _mm256_unpackhi_epi32(t5, t7);

          const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
          const __m256i w1 = _mm256_unpackhi_epi64(u0, u4);
          const __m256i w2 = _mm256_unpacklo_epi64(u1, u5);
          const __m256i w3 = _mm256_unpackhi_epi64(u1, u5);
          const __m256i w4 = _mm256_unpacklo_epi64(u2, u6);
          const __m256i w5 = _mm256_unpackhi_epi64(u2, u6);
          const __m256i w6 = _mm256_unpacklo_epi64(u3, u7);
          const __m256i w7 = _mm256_unpackhi_epi64(u3, u7);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w2));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w3));
          _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w4));
          _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w5));
          _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w6));
          _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w7));
          packed_w += 64;
        }

        if (k & 4) {
          __m256i v0 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w0));
          w0 += 4;
          __m256i v1 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w1));
          w1 += 4;
          __m256i v2 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w2));
          w2 += 4;
          __m256i v3 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w3));
          w3 += 4;
          __m256i v4 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w4));
          w4 += 4;
          __m256i v5 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w5));
          w5 += 4;
          __m256i v6 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w6));
          w6 += 4;
          __m256i v7 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w7));
          w7 += 4;

          const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
          const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
          const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
          const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);

          const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
          const __m256i u1 = _mm256_unpackhi_epi32(t0, t2);
          const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);
          const __m256i u5 = _mm256_unpackhi_epi32(t4, t6);

          const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
          const __m256i w1 = _mm256_unpackhi_epi64(u0, u4);
          const __m256i w2 = _mm256_unpacklo_epi64(u1, u5);
          const __m256i w3 = _mm256_unpackhi_epi64(u1, u5);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w2));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w3));
          packed_w += 32;
        }

        if (k & 2) {
          __m256i v0 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w0)));
          w0 += 2;
          __m256i v1 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w1)));
          w1 += 2;
          __m256i v2 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w2)));
          w2 += 2;
          __m256i v3 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w3)));
          w3 += 2;
          __m256i v4 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w4)));
          w4 += 2;
          __m256i v5 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w5)));
          w5 += 2;
          __m256i v6 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w6)));
          w6 += 2;
          __m256i v7 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w7)));
          w7 += 2;

          const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
          const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
          const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
          const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);

          const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
          const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);

          const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
          const __m256i w1 = _mm256_unpackhi_epi64(u0, u4);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w1));
          packed_w += 16;
        }

        if (k & 1) {
          __m256i v0 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w0, 0));
          w0 += 1;
          __m256i v1 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w1, 0));
          w1 += 1;
          __m256i v2 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w2, 0));
          w2 += 1;
          __m256i v3 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w3, 0));
          w3 += 1;
          __m256i v4 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w4, 0));
          w4 += 1;
          __m256i v5 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w5, 0));
          w5 += 1;
          __m256i v6 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w6, 0));
          w6 += 1;
          __m256i v7 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w7, 0));
          w7 += 1;

          const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
          const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
          const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
          const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);

          const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
          const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);

          const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          packed_w += 8;
        }
      }
      packed_w = (uint16_t*) ((uintptr_t) packed_w + extra_bytes);
      w0 = w7;
    }

    // NC remainder (1..7)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 7);
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *packed_w++ = *b++;
        } while (--nb != 0);
        packed_w += (8 - n);
      } else {
        const __m128i vzero = _mm_setzero_si128();
        _mm_storeu_si128((__m128i*) (packed_w + 0), vzero);
        packed_w += 8;
      }

      // NR remainder has less than 8 rows so last row is not loaded
      const uint16_t* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const uint16_t* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const uint16_t* w3 = w2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const uint16_t* w4 = w3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const uint16_t* w5 = w4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const uint16_t* w6 = w5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }

      // KC main loop multiple of 16
      size_t k = kc;
      for (; k >= 16; k -= 16) {
        __m256i v0 = _mm256_loadu_si256((const __m256i*) w0);
        w0 += 16;
        __m256i v1 = _mm256_loadu_si256((const __m256i*) w1);
        w1 += 16;
        __m256i v2 = _mm256_loadu_si256((const __m256i*) w2);
        w2 += 16;
        __m256i v3 = _mm256_loadu_si256((const __m256i*) w3);
        w3 += 16;
        __m256i v4 = _mm256_loadu_si256((const __m256i*) w4);
        w4 += 16;
        __m256i v5 = _mm256_loadu_si256((const __m256i*) w5);
        w5 += 16;
        __m256i v6 = _mm256_loadu_si256((const __m256i*) w6);
        w6 += 16;
        __m256i v7 = _mm256_setzero_si256();

        const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
        const __m256i t1 = _mm256_unpackhi_epi16(v0, v1);
        const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
        const __m256i t3 = _mm256_unpackhi_epi16(v2, v3);
        const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
        const __m256i t5 = _mm256_unpackhi_epi16(v4, v5);
        const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);
        const __m256i t7 = _mm256_unpackhi_epi16(v6, v7);

        const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
        const __m256i u1 = _mm256_unpackhi_epi32(t0, t2);
        const __m256i u2 = _mm256_unpacklo_epi32(t1, t3);
        const __m256i u3 = _mm256_unpackhi_epi32(t1, t3);
        const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);
        const __m256i u5 = _mm256_unpackhi_epi32(t4, t6);
        const __m256i u6 = _mm256_unpacklo_epi32(t5, t7);
        const __m256i u7 = _mm256_unpackhi_epi32(t5, t7);

        const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
        const __m256i w1 = _mm256_unpackhi_epi64(u0, u4);
        const __m256i w2 = _mm256_unpacklo_epi64(u1, u5);
        const __m256i w3 = _mm256_unpackhi_epi64(u1, u5);
        const __m256i w4 = _mm256_unpacklo_epi64(u2, u6);
        const __m256i w5 = _mm256_unpackhi_epi64(u2, u6);
        const __m256i w6 = _mm256_unpacklo_epi64(u3, u7);
        const __m256i w7 = _mm256_unpackhi_epi64(u3, u7);

        _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
        _mm_storeu_si128((__m128i*) (packed_w + 64), _mm256_extracti128_si256(w0, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w1));
        _mm_storeu_si128((__m128i*) (packed_w + 72), _mm256_extracti128_si256(w1, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w2));
        _mm_storeu_si128((__m128i*) (packed_w + 80), _mm256_extracti128_si256(w2, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w3));
        _mm_storeu_si128((__m128i*) (packed_w + 88), _mm256_extracti128_si256(w3, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w4));
        _mm_storeu_si128((__m128i*) (packed_w + 96), _mm256_extracti128_si256(w4, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w5));
        _mm_storeu_si128((__m128i*) (packed_w + 104), _mm256_extracti128_si256(w5, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w6));
        _mm_storeu_si128((__m128i*) (packed_w + 112), _mm256_extracti128_si256(w6, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w7));
        _mm_storeu_si128((__m128i*) (packed_w + 120), _mm256_extracti128_si256(w7, 1));
        packed_w += 128;
      }

      // KC remainder (1..15)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 15);

        if (k & 8) {
          __m256i v0 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w0));
          w0 += 8;
          __m256i v1 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w1));
          w1 += 8;
          __m256i v2 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w2));
          w2 += 8;
          __m256i v3 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w3));
          w3 += 8;
          __m256i v4 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w4));
          w4 += 8;
          __m256i v5 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w5));
          w5 += 8;
          __m256i v6 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w6));
          w6 += 8;
          __m256i v7 = _mm256_setzero_si256();

          const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
          const __m256i t1 = _mm256_unpackhi_epi16(v0, v1);
          const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
          const __m256i t3 = _mm256_unpackhi_epi16(v2, v3);
          const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
          const __m256i t5 = _mm256_unpackhi_epi16(v4, v5);
          const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);
          const __m256i t7 = _mm256_unpackhi_epi16(v6, v7);

          const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
          const __m256i u1 = _mm256_unpackhi_epi32(t0, t2);
          const __m256i u2 = _mm256_unpacklo_epi32(t1, t3);
          const __m256i u3 = _mm256_unpackhi_epi32(t1, t3);
          const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);
          const __m256i u5 = _mm256_unpackhi_epi32(t4, t6);
          const __m256i u6 = _mm256_unpacklo_epi32(t5, t7);
          const __m256i u7 = _mm256_unpackhi_epi32(t5, t7);

          const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
          const __m256i w1 = _mm256_unpackhi_epi64(u0, u4);
          const __m256i w2 = _mm256_unpacklo_epi64(u1, u5);
          const __m256i w3 = _mm256_unpackhi_epi64(u1, u5);
          const __m256i w4 = _mm256_unpacklo_epi64(u2, u6);
          const __m256i w5 = _mm256_unpackhi_epi64(u2, u6);
          const __m256i w6 = _mm256_unpacklo_epi64(u3, u7);
          const __m256i w7 = _mm256_unpackhi_epi64(u3, u7);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w2));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w3));
          _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w4));
          _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w5));
          _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w6));
          _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w7));
          packed_w += 64;
        }

        if (k & 4) {
          __m256i v0 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w0));
          w0 += 4;
          __m256i v1 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w1));
          w1 += 4;
          __m256i v2 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w2));
          w2 += 4;
          __m256i v3 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w3));
          w3 += 4;
          __m256i v4 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w4));
          w4 += 4;
          __m256i v5 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w5));
          w5 += 4;
          __m256i v6 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w6));
          w6 += 4;
          __m256i v7 = _mm256_setzero_si256();

          const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
          const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
          const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
          const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);

          const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
          const __m256i u1 = _mm256_unpackhi_epi32(t0, t2);
          const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);
          const __m256i u5 = _mm256_unpackhi_epi32(t4, t6);

          const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
          const __m256i w1 = _mm256_unpackhi_epi64(u0, u4);
          const __m256i w2 = _mm256_unpacklo_epi64(u1, u5);
          const __m256i w3 = _mm256_unpackhi_epi64(u1, u5);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w2));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w3));
          packed_w += 32;
        }

        if (k & 2) {
          __m256i v0 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w0)));
          w0 += 2;
          __m256i v1 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w1)));
          w1 += 2;
          __m256i v2 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w2)));
          w2 += 2;
          __m256i v3 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w3)));
          w3 += 2;
          __m256i v4 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w4)));
          w4 += 2;
          __m256i v5 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w5)));
          w5 += 2;
          __m256i v6 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w6)));
          w6 += 2;
          __m256i v7 = _mm256_setzero_si256();

          const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
          const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
          const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
          const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);

          const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
          const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);

          const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
          const __m256i w1 = _mm256_unpackhi_epi64(u0, u4);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w1));
          packed_w += 16;
        }

        if (k & 1) {
          __m256i v0 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w0, 0));
          w0 += 1;
          __m256i v1 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w1, 0));
          w1 += 1;
          __m256i v2 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w2, 0));
          w2 += 1;
          __m256i v3 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w3, 0));
          w3 += 1;
          __m256i v4 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w4, 0));
          w4 += 1;
          __m256i v5 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w5, 0));
          w5 += 1;
          __m256i v6 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w6, 0));
          w6 += 1;
          __m256i v7 = _mm256_setzero_si256();

          const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
          const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
          const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
          const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);

          const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
          const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);

          const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          packed_w += 8;
        }
      }
      packed_w = (uint16_t*) ((uintptr_t) packed_w + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
