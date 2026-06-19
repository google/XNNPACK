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


void xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16(
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
  assert(nr == 16);   // This kernel is for NR=16
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const uint16_t* b = bias;
  uint16_t* packed_w = packed_weights;
  do {
    // NC main loop multiple of 16
    const uint16_t* w0 = weights;
    size_t n = nc;

    for (; n >= 16; n -= 16) {
      if XNN_LIKELY(b != NULL) {
        const __m128i vb0 = _mm_loadu_si128((const __m128i*) (b + 0));
        _mm_storeu_si128((__m128i*) (packed_w + 0), vb0);
        const __m128i vb8 = _mm_loadu_si128((const __m128i*) (b + 8));
        _mm_storeu_si128((__m128i*) (packed_w + 8), vb8);
        b += 16;
      } else {
        const __m128i vzero = _mm_setzero_si128();
        _mm_storeu_si128((__m128i*) (packed_w + 0), vzero);
        _mm_storeu_si128((__m128i*) (packed_w + 8), vzero);
      }
      packed_w += 16;

      const uint16_t* w1 = w0 + kc;
      const uint16_t* w2 = w1 + kc;
      const uint16_t* w3 = w2 + kc;
      const uint16_t* w4 = w3 + kc;
      const uint16_t* w5 = w4 + kc;
      const uint16_t* w6 = w5 + kc;
      const uint16_t* w7 = w6 + kc;
      const uint16_t* w8 = w7 + kc;
      const uint16_t* w9 = w8 + kc;
      const uint16_t* w10 = w9 + kc;
      const uint16_t* w11 = w10 + kc;
      const uint16_t* w12 = w11 + kc;
      const uint16_t* w13 = w12 + kc;
      const uint16_t* w14 = w13 + kc;
      const uint16_t* w15 = w14 + kc;

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
        __m256i v8 = _mm256_loadu_si256((const __m256i*) w8);
        w8 += 16;
        __m256i v9 = _mm256_loadu_si256((const __m256i*) w9);
        w9 += 16;
        __m256i v10 = _mm256_loadu_si256((const __m256i*) w10);
        w10 += 16;
        __m256i v11 = _mm256_loadu_si256((const __m256i*) w11);
        w11 += 16;
        __m256i v12 = _mm256_loadu_si256((const __m256i*) w12);
        w12 += 16;
        __m256i v13 = _mm256_loadu_si256((const __m256i*) w13);
        w13 += 16;
        __m256i v14 = _mm256_loadu_si256((const __m256i*) w14);
        w14 += 16;
        __m256i v15 = _mm256_loadu_si256((const __m256i*) w15);
        w15 += 16;

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
        const __m256i t8 = _mm256_unpacklo_epi16(v8, v9);
        const __m256i t9 = _mm256_unpackhi_epi16(v8, v9);
        const __m256i t10 = _mm256_unpacklo_epi16(v10, v11);
        const __m256i t11 = _mm256_unpackhi_epi16(v10, v11);
        const __m256i t12 = _mm256_unpacklo_epi16(v12, v13);
        const __m256i t13 = _mm256_unpackhi_epi16(v12, v13);
        const __m256i t14 = _mm256_unpacklo_epi16(v14, v15);
        const __m256i t15 = _mm256_unpackhi_epi16(v14, v15);

        const __m256i u8 = _mm256_unpacklo_epi32(t8, t10);
        const __m256i u9 = _mm256_unpackhi_epi32(t8, t10);
        const __m256i u10 = _mm256_unpacklo_epi32(t9, t11);
        const __m256i u11 = _mm256_unpackhi_epi32(t9, t11);
        const __m256i u12 = _mm256_unpacklo_epi32(t12, t14);
        const __m256i u13 = _mm256_unpackhi_epi32(t12, t14);
        const __m256i u14 = _mm256_unpacklo_epi32(t13, t15);
        const __m256i u15 = _mm256_unpackhi_epi32(t13, t15);

        const __m256i w8 = _mm256_unpacklo_epi64(u8, u12);
        const __m256i w9 = _mm256_unpackhi_epi64(u8, u12);
        const __m256i w10 = _mm256_unpacklo_epi64(u9, u13);
        const __m256i w11 = _mm256_unpackhi_epi64(u9, u13);
        const __m256i w12 = _mm256_unpacklo_epi64(u10, u14);
        const __m256i w13 = _mm256_unpackhi_epi64(u10, u14);
        const __m256i w14 = _mm256_unpacklo_epi64(u11, u15);
        const __m256i w15 = _mm256_unpackhi_epi64(u11, u15);

        _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
        _mm_storeu_si128((__m128i*) (packed_w + 128), _mm256_extracti128_si256(w0, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
        _mm_storeu_si128((__m128i*) (packed_w + 136), _mm256_extracti128_si256(w8, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w1));
        _mm_storeu_si128((__m128i*) (packed_w + 144), _mm256_extracti128_si256(w1, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w9));
        _mm_storeu_si128((__m128i*) (packed_w + 152), _mm256_extracti128_si256(w9, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w2));
        _mm_storeu_si128((__m128i*) (packed_w + 160), _mm256_extracti128_si256(w2, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w10));
        _mm_storeu_si128((__m128i*) (packed_w + 168), _mm256_extracti128_si256(w10, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w3));
        _mm_storeu_si128((__m128i*) (packed_w + 176), _mm256_extracti128_si256(w3, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w11));
        _mm_storeu_si128((__m128i*) (packed_w + 184), _mm256_extracti128_si256(w11, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 64), _mm256_castsi256_si128(w4));
        _mm_storeu_si128((__m128i*) (packed_w + 192), _mm256_extracti128_si256(w4, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 72), _mm256_castsi256_si128(w12));
        _mm_storeu_si128((__m128i*) (packed_w + 200), _mm256_extracti128_si256(w12, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 80), _mm256_castsi256_si128(w5));
        _mm_storeu_si128((__m128i*) (packed_w + 208), _mm256_extracti128_si256(w5, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 88), _mm256_castsi256_si128(w13));
        _mm_storeu_si128((__m128i*) (packed_w + 216), _mm256_extracti128_si256(w13, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 96), _mm256_castsi256_si128(w6));
        _mm_storeu_si128((__m128i*) (packed_w + 224), _mm256_extracti128_si256(w6, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 104), _mm256_castsi256_si128(w14));
        _mm_storeu_si128((__m128i*) (packed_w + 232), _mm256_extracti128_si256(w14, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 112), _mm256_castsi256_si128(w7));
        _mm_storeu_si128((__m128i*) (packed_w + 240), _mm256_extracti128_si256(w7, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 120), _mm256_castsi256_si128(w15));
        _mm_storeu_si128((__m128i*) (packed_w + 248), _mm256_extracti128_si256(w15, 1));
        packed_w += 256;
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
          __m256i v8 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w8));
          w8 += 8;
          __m256i v9 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w9));
          w9 += 8;
          __m256i v10 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w10));
          w10 += 8;
          __m256i v11 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w11));
          w11 += 8;
          __m256i v12 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w12));
          w12 += 8;
          __m256i v13 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w13));
          w13 += 8;
          __m256i v14 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w14));
          w14 += 8;
          __m256i v15 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w15));
          w15 += 8;

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
          const __m256i t8 = _mm256_unpacklo_epi16(v8, v9);
          const __m256i t9 = _mm256_unpackhi_epi16(v8, v9);
          const __m256i t10 = _mm256_unpacklo_epi16(v10, v11);
          const __m256i t11 = _mm256_unpackhi_epi16(v10, v11);
          const __m256i t12 = _mm256_unpacklo_epi16(v12, v13);
          const __m256i t13 = _mm256_unpackhi_epi16(v12, v13);
          const __m256i t14 = _mm256_unpacklo_epi16(v14, v15);
          const __m256i t15 = _mm256_unpackhi_epi16(v14, v15);

          const __m256i u8 = _mm256_unpacklo_epi32(t8, t10);
          const __m256i u9 = _mm256_unpackhi_epi32(t8, t10);
          const __m256i u10 = _mm256_unpacklo_epi32(t9, t11);
          const __m256i u11 = _mm256_unpackhi_epi32(t9, t11);
          const __m256i u12 = _mm256_unpacklo_epi32(t12, t14);
          const __m256i u13 = _mm256_unpackhi_epi32(t12, t14);
          const __m256i u14 = _mm256_unpacklo_epi32(t13, t15);
          const __m256i u15 = _mm256_unpackhi_epi32(t13, t15);

          const __m256i w8 = _mm256_unpacklo_epi64(u8, u12);
          const __m256i w9 = _mm256_unpackhi_epi64(u8, u12);
          const __m256i w10 = _mm256_unpacklo_epi64(u9, u13);
          const __m256i w11 = _mm256_unpackhi_epi64(u9, u13);
          const __m256i w12 = _mm256_unpacklo_epi64(u10, u14);
          const __m256i w13 = _mm256_unpackhi_epi64(u10, u14);
          const __m256i w14 = _mm256_unpacklo_epi64(u11, u15);
          const __m256i w15 = _mm256_unpackhi_epi64(u11, u15);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w9));
          _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w2));
          _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w10));
          _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w3));
          _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w11));
          _mm_storeu_si128((__m128i*) (packed_w + 64), _mm256_castsi256_si128(w4));
          _mm_storeu_si128((__m128i*) (packed_w + 72), _mm256_castsi256_si128(w12));
          _mm_storeu_si128((__m128i*) (packed_w + 80), _mm256_castsi256_si128(w5));
          _mm_storeu_si128((__m128i*) (packed_w + 88), _mm256_castsi256_si128(w13));
          _mm_storeu_si128((__m128i*) (packed_w + 96), _mm256_castsi256_si128(w6));
          _mm_storeu_si128((__m128i*) (packed_w + 104), _mm256_castsi256_si128(w14));
          _mm_storeu_si128((__m128i*) (packed_w + 112), _mm256_castsi256_si128(w7));
          _mm_storeu_si128((__m128i*) (packed_w + 120), _mm256_castsi256_si128(w15));
          packed_w += 128;
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
          __m256i v8 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w8));
          w8 += 4;
          __m256i v9 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w9));
          w9 += 4;
          __m256i v10 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w10));
          w10 += 4;
          __m256i v11 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w11));
          w11 += 4;
          __m256i v12 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w12));
          w12 += 4;
          __m256i v13 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w13));
          w13 += 4;
          __m256i v14 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w14));
          w14 += 4;
          __m256i v15 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w15));
          w15 += 4;

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
          const __m256i t8 = _mm256_unpacklo_epi16(v8, v9);
          const __m256i t10 = _mm256_unpacklo_epi16(v10, v11);
          const __m256i t12 = _mm256_unpacklo_epi16(v12, v13);
          const __m256i t14 = _mm256_unpacklo_epi16(v14, v15);

          const __m256i u8 = _mm256_unpacklo_epi32(t8, t10);
          const __m256i u9 = _mm256_unpackhi_epi32(t8, t10);
          const __m256i u12 = _mm256_unpacklo_epi32(t12, t14);
          const __m256i u13 = _mm256_unpackhi_epi32(t12, t14);

          const __m256i w8 = _mm256_unpacklo_epi64(u8, u12);
          const __m256i w9 = _mm256_unpackhi_epi64(u8, u12);
          const __m256i w10 = _mm256_unpacklo_epi64(u9, u13);
          const __m256i w11 = _mm256_unpackhi_epi64(u9, u13);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w9));
          _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w2));
          _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w10));
          _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w3));
          _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w11));
          packed_w += 64;
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
          __m256i v8 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w8)));
          w8 += 2;
          __m256i v9 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w9)));
          w9 += 2;
          __m256i v10 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w10)));
          w10 += 2;
          __m256i v11 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w11)));
          w11 += 2;
          __m256i v12 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w12)));
          w12 += 2;
          __m256i v13 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w13)));
          w13 += 2;
          __m256i v14 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w14)));
          w14 += 2;
          __m256i v15 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w15)));
          w15 += 2;

          const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
          const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
          const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
          const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);

          const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
          const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);

          const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
          const __m256i w1 = _mm256_unpackhi_epi64(u0, u4);
          const __m256i t8 = _mm256_unpacklo_epi16(v8, v9);
          const __m256i t10 = _mm256_unpacklo_epi16(v10, v11);
          const __m256i t12 = _mm256_unpacklo_epi16(v12, v13);
          const __m256i t14 = _mm256_unpacklo_epi16(v14, v15);

          const __m256i u8 = _mm256_unpacklo_epi32(t8, t10);
          const __m256i u12 = _mm256_unpacklo_epi32(t12, t14);

          const __m256i w8 = _mm256_unpacklo_epi64(u8, u12);
          const __m256i w9 = _mm256_unpackhi_epi64(u8, u12);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w9));
          packed_w += 32;
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
          __m256i v8 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w8, 0));
          w8 += 1;
          __m256i v9 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w9, 0));
          w9 += 1;
          __m256i v10 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w10, 0));
          w10 += 1;
          __m256i v11 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w11, 0));
          w11 += 1;
          __m256i v12 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w12, 0));
          w12 += 1;
          __m256i v13 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w13, 0));
          w13 += 1;
          __m256i v14 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w14, 0));
          w14 += 1;
          __m256i v15 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w15, 0));
          w15 += 1;

          const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
          const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
          const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
          const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);

          const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
          const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);

          const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
          const __m256i t8 = _mm256_unpacklo_epi16(v8, v9);
          const __m256i t10 = _mm256_unpacklo_epi16(v10, v11);
          const __m256i t12 = _mm256_unpacklo_epi16(v12, v13);
          const __m256i t14 = _mm256_unpacklo_epi16(v14, v15);

          const __m256i u8 = _mm256_unpacklo_epi32(t8, t10);
          const __m256i u12 = _mm256_unpacklo_epi32(t12, t14);

          const __m256i w8 = _mm256_unpacklo_epi64(u8, u12);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          packed_w += 16;
        }
      }
      packed_w = (uint16_t*) ((uintptr_t) packed_w + extra_bytes);
      w0 = w15;
    }

    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 15);
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *packed_w++ = *b++;
        } while (--nb != 0);
        packed_w += (16 - n);
      } else {
        const __m128i vzero = _mm_setzero_si128();
        _mm_storeu_si128((__m128i*) (packed_w + 0), vzero);
        _mm_storeu_si128((__m128i*) (packed_w + 8), vzero);
        packed_w += 16;
      }

      // NR remainder has less than 16 rows so last row is not loaded
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
      const uint16_t* w7 = w6 + kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const uint16_t* w8 = w7 + kc;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const uint16_t* w9 = w8 + kc;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const uint16_t* w10 = w9 + kc;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const uint16_t* w11 = w10 + kc;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const uint16_t* w12 = w11 + kc;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const uint16_t* w13 = w12 + kc;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const uint16_t* w14 = w13 + kc;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
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
        __m256i v7 = _mm256_loadu_si256((const __m256i*) w7);
        w7 += 16;
        __m256i v8 = _mm256_loadu_si256((const __m256i*) w8);
        w8 += 16;
        __m256i v9 = _mm256_loadu_si256((const __m256i*) w9);
        w9 += 16;
        __m256i v10 = _mm256_loadu_si256((const __m256i*) w10);
        w10 += 16;
        __m256i v11 = _mm256_loadu_si256((const __m256i*) w11);
        w11 += 16;
        __m256i v12 = _mm256_loadu_si256((const __m256i*) w12);
        w12 += 16;
        __m256i v13 = _mm256_loadu_si256((const __m256i*) w13);
        w13 += 16;
        __m256i v14 = _mm256_loadu_si256((const __m256i*) w14);
        w14 += 16;
        __m256i v15 = _mm256_setzero_si256();

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
        const __m256i t8 = _mm256_unpacklo_epi16(v8, v9);
        const __m256i t9 = _mm256_unpackhi_epi16(v8, v9);
        const __m256i t10 = _mm256_unpacklo_epi16(v10, v11);
        const __m256i t11 = _mm256_unpackhi_epi16(v10, v11);
        const __m256i t12 = _mm256_unpacklo_epi16(v12, v13);
        const __m256i t13 = _mm256_unpackhi_epi16(v12, v13);
        const __m256i t14 = _mm256_unpacklo_epi16(v14, v15);
        const __m256i t15 = _mm256_unpackhi_epi16(v14, v15);

        const __m256i u8 = _mm256_unpacklo_epi32(t8, t10);
        const __m256i u9 = _mm256_unpackhi_epi32(t8, t10);
        const __m256i u10 = _mm256_unpacklo_epi32(t9, t11);
        const __m256i u11 = _mm256_unpackhi_epi32(t9, t11);
        const __m256i u12 = _mm256_unpacklo_epi32(t12, t14);
        const __m256i u13 = _mm256_unpackhi_epi32(t12, t14);
        const __m256i u14 = _mm256_unpacklo_epi32(t13, t15);
        const __m256i u15 = _mm256_unpackhi_epi32(t13, t15);

        const __m256i w8 = _mm256_unpacklo_epi64(u8, u12);
        const __m256i w9 = _mm256_unpackhi_epi64(u8, u12);
        const __m256i w10 = _mm256_unpacklo_epi64(u9, u13);
        const __m256i w11 = _mm256_unpackhi_epi64(u9, u13);
        const __m256i w12 = _mm256_unpacklo_epi64(u10, u14);
        const __m256i w13 = _mm256_unpackhi_epi64(u10, u14);
        const __m256i w14 = _mm256_unpacklo_epi64(u11, u15);
        const __m256i w15 = _mm256_unpackhi_epi64(u11, u15);

        _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
        _mm_storeu_si128((__m128i*) (packed_w + 128), _mm256_extracti128_si256(w0, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
        _mm_storeu_si128((__m128i*) (packed_w + 136), _mm256_extracti128_si256(w8, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w1));
        _mm_storeu_si128((__m128i*) (packed_w + 144), _mm256_extracti128_si256(w1, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w9));
        _mm_storeu_si128((__m128i*) (packed_w + 152), _mm256_extracti128_si256(w9, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w2));
        _mm_storeu_si128((__m128i*) (packed_w + 160), _mm256_extracti128_si256(w2, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w10));
        _mm_storeu_si128((__m128i*) (packed_w + 168), _mm256_extracti128_si256(w10, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w3));
        _mm_storeu_si128((__m128i*) (packed_w + 176), _mm256_extracti128_si256(w3, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w11));
        _mm_storeu_si128((__m128i*) (packed_w + 184), _mm256_extracti128_si256(w11, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 64), _mm256_castsi256_si128(w4));
        _mm_storeu_si128((__m128i*) (packed_w + 192), _mm256_extracti128_si256(w4, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 72), _mm256_castsi256_si128(w12));
        _mm_storeu_si128((__m128i*) (packed_w + 200), _mm256_extracti128_si256(w12, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 80), _mm256_castsi256_si128(w5));
        _mm_storeu_si128((__m128i*) (packed_w + 208), _mm256_extracti128_si256(w5, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 88), _mm256_castsi256_si128(w13));
        _mm_storeu_si128((__m128i*) (packed_w + 216), _mm256_extracti128_si256(w13, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 96), _mm256_castsi256_si128(w6));
        _mm_storeu_si128((__m128i*) (packed_w + 224), _mm256_extracti128_si256(w6, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 104), _mm256_castsi256_si128(w14));
        _mm_storeu_si128((__m128i*) (packed_w + 232), _mm256_extracti128_si256(w14, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 112), _mm256_castsi256_si128(w7));
        _mm_storeu_si128((__m128i*) (packed_w + 240), _mm256_extracti128_si256(w7, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 120), _mm256_castsi256_si128(w15));
        _mm_storeu_si128((__m128i*) (packed_w + 248), _mm256_extracti128_si256(w15, 1));
        packed_w += 256;
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
          __m256i v8 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w8));
          w8 += 8;
          __m256i v9 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w9));
          w9 += 8;
          __m256i v10 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w10));
          w10 += 8;
          __m256i v11 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w11));
          w11 += 8;
          __m256i v12 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w12));
          w12 += 8;
          __m256i v13 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w13));
          w13 += 8;
          __m256i v14 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w14));
          w14 += 8;
          __m256i v15 = _mm256_setzero_si256();

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
          const __m256i t8 = _mm256_unpacklo_epi16(v8, v9);
          const __m256i t9 = _mm256_unpackhi_epi16(v8, v9);
          const __m256i t10 = _mm256_unpacklo_epi16(v10, v11);
          const __m256i t11 = _mm256_unpackhi_epi16(v10, v11);
          const __m256i t12 = _mm256_unpacklo_epi16(v12, v13);
          const __m256i t13 = _mm256_unpackhi_epi16(v12, v13);
          const __m256i t14 = _mm256_unpacklo_epi16(v14, v15);
          const __m256i t15 = _mm256_unpackhi_epi16(v14, v15);

          const __m256i u8 = _mm256_unpacklo_epi32(t8, t10);
          const __m256i u9 = _mm256_unpackhi_epi32(t8, t10);
          const __m256i u10 = _mm256_unpacklo_epi32(t9, t11);
          const __m256i u11 = _mm256_unpackhi_epi32(t9, t11);
          const __m256i u12 = _mm256_unpacklo_epi32(t12, t14);
          const __m256i u13 = _mm256_unpackhi_epi32(t12, t14);
          const __m256i u14 = _mm256_unpacklo_epi32(t13, t15);
          const __m256i u15 = _mm256_unpackhi_epi32(t13, t15);

          const __m256i w8 = _mm256_unpacklo_epi64(u8, u12);
          const __m256i w9 = _mm256_unpackhi_epi64(u8, u12);
          const __m256i w10 = _mm256_unpacklo_epi64(u9, u13);
          const __m256i w11 = _mm256_unpackhi_epi64(u9, u13);
          const __m256i w12 = _mm256_unpacklo_epi64(u10, u14);
          const __m256i w13 = _mm256_unpackhi_epi64(u10, u14);
          const __m256i w14 = _mm256_unpacklo_epi64(u11, u15);
          const __m256i w15 = _mm256_unpackhi_epi64(u11, u15);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w9));
          _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w2));
          _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w10));
          _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w3));
          _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w11));
          _mm_storeu_si128((__m128i*) (packed_w + 64), _mm256_castsi256_si128(w4));
          _mm_storeu_si128((__m128i*) (packed_w + 72), _mm256_castsi256_si128(w12));
          _mm_storeu_si128((__m128i*) (packed_w + 80), _mm256_castsi256_si128(w5));
          _mm_storeu_si128((__m128i*) (packed_w + 88), _mm256_castsi256_si128(w13));
          _mm_storeu_si128((__m128i*) (packed_w + 96), _mm256_castsi256_si128(w6));
          _mm_storeu_si128((__m128i*) (packed_w + 104), _mm256_castsi256_si128(w14));
          _mm_storeu_si128((__m128i*) (packed_w + 112), _mm256_castsi256_si128(w7));
          _mm_storeu_si128((__m128i*) (packed_w + 120), _mm256_castsi256_si128(w15));
          packed_w += 128;
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
          __m256i v8 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w8));
          w8 += 4;
          __m256i v9 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w9));
          w9 += 4;
          __m256i v10 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w10));
          w10 += 4;
          __m256i v11 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w11));
          w11 += 4;
          __m256i v12 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w12));
          w12 += 4;
          __m256i v13 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w13));
          w13 += 4;
          __m256i v14 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w14));
          w14 += 4;
          __m256i v15 = _mm256_setzero_si256();

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
          const __m256i t8 = _mm256_unpacklo_epi16(v8, v9);
          const __m256i t10 = _mm256_unpacklo_epi16(v10, v11);
          const __m256i t12 = _mm256_unpacklo_epi16(v12, v13);
          const __m256i t14 = _mm256_unpacklo_epi16(v14, v15);

          const __m256i u8 = _mm256_unpacklo_epi32(t8, t10);
          const __m256i u9 = _mm256_unpackhi_epi32(t8, t10);
          const __m256i u12 = _mm256_unpacklo_epi32(t12, t14);
          const __m256i u13 = _mm256_unpackhi_epi32(t12, t14);

          const __m256i w8 = _mm256_unpacklo_epi64(u8, u12);
          const __m256i w9 = _mm256_unpackhi_epi64(u8, u12);
          const __m256i w10 = _mm256_unpacklo_epi64(u9, u13);
          const __m256i w11 = _mm256_unpackhi_epi64(u9, u13);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w9));
          _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w2));
          _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w10));
          _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w3));
          _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w11));
          packed_w += 64;
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
          __m256i v8 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w8)));
          w8 += 2;
          __m256i v9 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w9)));
          w9 += 2;
          __m256i v10 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w10)));
          w10 += 2;
          __m256i v11 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w11)));
          w11 += 2;
          __m256i v12 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w12)));
          w12 += 2;
          __m256i v13 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w13)));
          w13 += 2;
          __m256i v14 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w14)));
          w14 += 2;
          __m256i v15 = _mm256_setzero_si256();

          const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
          const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
          const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
          const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);

          const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
          const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);

          const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
          const __m256i w1 = _mm256_unpackhi_epi64(u0, u4);
          const __m256i t8 = _mm256_unpacklo_epi16(v8, v9);
          const __m256i t10 = _mm256_unpacklo_epi16(v10, v11);
          const __m256i t12 = _mm256_unpacklo_epi16(v12, v13);
          const __m256i t14 = _mm256_unpacklo_epi16(v14, v15);

          const __m256i u8 = _mm256_unpacklo_epi32(t8, t10);
          const __m256i u12 = _mm256_unpacklo_epi32(t12, t14);

          const __m256i w8 = _mm256_unpacklo_epi64(u8, u12);
          const __m256i w9 = _mm256_unpackhi_epi64(u8, u12);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w9));
          packed_w += 32;
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
          __m256i v8 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w8, 0));
          w8 += 1;
          __m256i v9 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w9, 0));
          w9 += 1;
          __m256i v10 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w10, 0));
          w10 += 1;
          __m256i v11 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w11, 0));
          w11 += 1;
          __m256i v12 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w12, 0));
          w12 += 1;
          __m256i v13 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w13, 0));
          w13 += 1;
          __m256i v14 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w14, 0));
          w14 += 1;
          __m256i v15 = _mm256_setzero_si256();

          const __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
          const __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
          const __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
          const __m256i t6 = _mm256_unpacklo_epi16(v6, v7);

          const __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
          const __m256i u4 = _mm256_unpacklo_epi32(t4, t6);

          const __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
          const __m256i t8 = _mm256_unpacklo_epi16(v8, v9);
          const __m256i t10 = _mm256_unpacklo_epi16(v10, v11);
          const __m256i t12 = _mm256_unpacklo_epi16(v12, v13);
          const __m256i t14 = _mm256_unpacklo_epi16(v14, v15);

          const __m256i u8 = _mm256_unpacklo_epi32(t8, t10);
          const __m256i u12 = _mm256_unpacklo_epi32(t12, t14);

          const __m256i w8 = _mm256_unpacklo_epi64(u8, u12);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          packed_w += 16;
        }
      }
      packed_w = (uint16_t*) ((uintptr_t) packed_w + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
