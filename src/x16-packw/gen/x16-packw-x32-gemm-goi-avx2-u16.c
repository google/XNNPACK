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


void xnn_x16_packw_gemm_goi_ukernel_x32__avx2_u16(
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
  assert(nr == 32);   // This kernel is for NR=32
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const uint16_t* b = bias;
  uint16_t* packed_w = packed_weights;
  do {
    // NC main loop multiple of 32
    const uint16_t* w0 = weights;
    size_t n = nc;

    for (; n >= 32; n -= 32) {
      if XNN_LIKELY(b != NULL) {
        const __m128i vb0 = _mm_loadu_si128((const __m128i*) (b + 0));
        _mm_storeu_si128((__m128i*) (packed_w + 0), vb0);
        const __m128i vb8 = _mm_loadu_si128((const __m128i*) (b + 8));
        _mm_storeu_si128((__m128i*) (packed_w + 8), vb8);
        const __m128i vb16 = _mm_loadu_si128((const __m128i*) (b + 16));
        _mm_storeu_si128((__m128i*) (packed_w + 16), vb16);
        const __m128i vb24 = _mm_loadu_si128((const __m128i*) (b + 24));
        _mm_storeu_si128((__m128i*) (packed_w + 24), vb24);
        b += 32;
      } else {
        const __m128i vzero = _mm_setzero_si128();
        _mm_storeu_si128((__m128i*) (packed_w + 0), vzero);
        _mm_storeu_si128((__m128i*) (packed_w + 8), vzero);
        _mm_storeu_si128((__m128i*) (packed_w + 16), vzero);
        _mm_storeu_si128((__m128i*) (packed_w + 24), vzero);
      }
      packed_w += 32;

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
      const uint16_t* w16 = w15 + kc;
      const uint16_t* w17 = w16 + kc;
      const uint16_t* w18 = w17 + kc;
      const uint16_t* w19 = w18 + kc;
      const uint16_t* w20 = w19 + kc;
      const uint16_t* w21 = w20 + kc;
      const uint16_t* w22 = w21 + kc;
      const uint16_t* w23 = w22 + kc;
      const uint16_t* w24 = w23 + kc;
      const uint16_t* w25 = w24 + kc;
      const uint16_t* w26 = w25 + kc;
      const uint16_t* w27 = w26 + kc;
      const uint16_t* w28 = w27 + kc;
      const uint16_t* w29 = w28 + kc;
      const uint16_t* w30 = w29 + kc;
      const uint16_t* w31 = w30 + kc;

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
        __m256i v16 = _mm256_loadu_si256((const __m256i*) w16);
        w16 += 16;
        __m256i v17 = _mm256_loadu_si256((const __m256i*) w17);
        w17 += 16;
        __m256i v18 = _mm256_loadu_si256((const __m256i*) w18);
        w18 += 16;
        __m256i v19 = _mm256_loadu_si256((const __m256i*) w19);
        w19 += 16;
        __m256i v20 = _mm256_loadu_si256((const __m256i*) w20);
        w20 += 16;
        __m256i v21 = _mm256_loadu_si256((const __m256i*) w21);
        w21 += 16;
        __m256i v22 = _mm256_loadu_si256((const __m256i*) w22);
        w22 += 16;
        __m256i v23 = _mm256_loadu_si256((const __m256i*) w23);
        w23 += 16;
        __m256i v24 = _mm256_loadu_si256((const __m256i*) w24);
        w24 += 16;
        __m256i v25 = _mm256_loadu_si256((const __m256i*) w25);
        w25 += 16;
        __m256i v26 = _mm256_loadu_si256((const __m256i*) w26);
        w26 += 16;
        __m256i v27 = _mm256_loadu_si256((const __m256i*) w27);
        w27 += 16;
        __m256i v28 = _mm256_loadu_si256((const __m256i*) w28);
        w28 += 16;
        __m256i v29 = _mm256_loadu_si256((const __m256i*) w29);
        w29 += 16;
        __m256i v30 = _mm256_loadu_si256((const __m256i*) w30);
        w30 += 16;
        __m256i v31 = _mm256_loadu_si256((const __m256i*) w31);
        w31 += 16;

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
        const __m256i t16 = _mm256_unpacklo_epi16(v16, v17);
        const __m256i t17 = _mm256_unpackhi_epi16(v16, v17);
        const __m256i t18 = _mm256_unpacklo_epi16(v18, v19);
        const __m256i t19 = _mm256_unpackhi_epi16(v18, v19);
        const __m256i t20 = _mm256_unpacklo_epi16(v20, v21);
        const __m256i t21 = _mm256_unpackhi_epi16(v20, v21);
        const __m256i t22 = _mm256_unpacklo_epi16(v22, v23);
        const __m256i t23 = _mm256_unpackhi_epi16(v22, v23);

        const __m256i u16 = _mm256_unpacklo_epi32(t16, t18);
        const __m256i u17 = _mm256_unpackhi_epi32(t16, t18);
        const __m256i u18 = _mm256_unpacklo_epi32(t17, t19);
        const __m256i u19 = _mm256_unpackhi_epi32(t17, t19);
        const __m256i u20 = _mm256_unpacklo_epi32(t20, t22);
        const __m256i u21 = _mm256_unpackhi_epi32(t20, t22);
        const __m256i u22 = _mm256_unpacklo_epi32(t21, t23);
        const __m256i u23 = _mm256_unpackhi_epi32(t21, t23);

        const __m256i w16 = _mm256_unpacklo_epi64(u16, u20);
        const __m256i w17 = _mm256_unpackhi_epi64(u16, u20);
        const __m256i w18 = _mm256_unpacklo_epi64(u17, u21);
        const __m256i w19 = _mm256_unpackhi_epi64(u17, u21);
        const __m256i w20 = _mm256_unpacklo_epi64(u18, u22);
        const __m256i w21 = _mm256_unpackhi_epi64(u18, u22);
        const __m256i w22 = _mm256_unpacklo_epi64(u19, u23);
        const __m256i w23 = _mm256_unpackhi_epi64(u19, u23);
        const __m256i t24 = _mm256_unpacklo_epi16(v24, v25);
        const __m256i t25 = _mm256_unpackhi_epi16(v24, v25);
        const __m256i t26 = _mm256_unpacklo_epi16(v26, v27);
        const __m256i t27 = _mm256_unpackhi_epi16(v26, v27);
        const __m256i t28 = _mm256_unpacklo_epi16(v28, v29);
        const __m256i t29 = _mm256_unpackhi_epi16(v28, v29);
        const __m256i t30 = _mm256_unpacklo_epi16(v30, v31);
        const __m256i t31 = _mm256_unpackhi_epi16(v30, v31);

        const __m256i u24 = _mm256_unpacklo_epi32(t24, t26);
        const __m256i u25 = _mm256_unpackhi_epi32(t24, t26);
        const __m256i u26 = _mm256_unpacklo_epi32(t25, t27);
        const __m256i u27 = _mm256_unpackhi_epi32(t25, t27);
        const __m256i u28 = _mm256_unpacklo_epi32(t28, t30);
        const __m256i u29 = _mm256_unpackhi_epi32(t28, t30);
        const __m256i u30 = _mm256_unpacklo_epi32(t29, t31);
        const __m256i u31 = _mm256_unpackhi_epi32(t29, t31);

        const __m256i w24 = _mm256_unpacklo_epi64(u24, u28);
        const __m256i w25 = _mm256_unpackhi_epi64(u24, u28);
        const __m256i w26 = _mm256_unpacklo_epi64(u25, u29);
        const __m256i w27 = _mm256_unpackhi_epi64(u25, u29);
        const __m256i w28 = _mm256_unpacklo_epi64(u26, u30);
        const __m256i w29 = _mm256_unpackhi_epi64(u26, u30);
        const __m256i w30 = _mm256_unpacklo_epi64(u27, u31);
        const __m256i w31 = _mm256_unpackhi_epi64(u27, u31);

        _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
        _mm_storeu_si128((__m128i*) (packed_w + 256), _mm256_extracti128_si256(w0, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
        _mm_storeu_si128((__m128i*) (packed_w + 264), _mm256_extracti128_si256(w8, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w16));
        _mm_storeu_si128((__m128i*) (packed_w + 272), _mm256_extracti128_si256(w16, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w24));
        _mm_storeu_si128((__m128i*) (packed_w + 280), _mm256_extracti128_si256(w24, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w1));
        _mm_storeu_si128((__m128i*) (packed_w + 288), _mm256_extracti128_si256(w1, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w9));
        _mm_storeu_si128((__m128i*) (packed_w + 296), _mm256_extracti128_si256(w9, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w17));
        _mm_storeu_si128((__m128i*) (packed_w + 304), _mm256_extracti128_si256(w17, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w25));
        _mm_storeu_si128((__m128i*) (packed_w + 312), _mm256_extracti128_si256(w25, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 64), _mm256_castsi256_si128(w2));
        _mm_storeu_si128((__m128i*) (packed_w + 320), _mm256_extracti128_si256(w2, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 72), _mm256_castsi256_si128(w10));
        _mm_storeu_si128((__m128i*) (packed_w + 328), _mm256_extracti128_si256(w10, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 80), _mm256_castsi256_si128(w18));
        _mm_storeu_si128((__m128i*) (packed_w + 336), _mm256_extracti128_si256(w18, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 88), _mm256_castsi256_si128(w26));
        _mm_storeu_si128((__m128i*) (packed_w + 344), _mm256_extracti128_si256(w26, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 96), _mm256_castsi256_si128(w3));
        _mm_storeu_si128((__m128i*) (packed_w + 352), _mm256_extracti128_si256(w3, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 104), _mm256_castsi256_si128(w11));
        _mm_storeu_si128((__m128i*) (packed_w + 360), _mm256_extracti128_si256(w11, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 112), _mm256_castsi256_si128(w19));
        _mm_storeu_si128((__m128i*) (packed_w + 368), _mm256_extracti128_si256(w19, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 120), _mm256_castsi256_si128(w27));
        _mm_storeu_si128((__m128i*) (packed_w + 376), _mm256_extracti128_si256(w27, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 128), _mm256_castsi256_si128(w4));
        _mm_storeu_si128((__m128i*) (packed_w + 384), _mm256_extracti128_si256(w4, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 136), _mm256_castsi256_si128(w12));
        _mm_storeu_si128((__m128i*) (packed_w + 392), _mm256_extracti128_si256(w12, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 144), _mm256_castsi256_si128(w20));
        _mm_storeu_si128((__m128i*) (packed_w + 400), _mm256_extracti128_si256(w20, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 152), _mm256_castsi256_si128(w28));
        _mm_storeu_si128((__m128i*) (packed_w + 408), _mm256_extracti128_si256(w28, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 160), _mm256_castsi256_si128(w5));
        _mm_storeu_si128((__m128i*) (packed_w + 416), _mm256_extracti128_si256(w5, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 168), _mm256_castsi256_si128(w13));
        _mm_storeu_si128((__m128i*) (packed_w + 424), _mm256_extracti128_si256(w13, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 176), _mm256_castsi256_si128(w21));
        _mm_storeu_si128((__m128i*) (packed_w + 432), _mm256_extracti128_si256(w21, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 184), _mm256_castsi256_si128(w29));
        _mm_storeu_si128((__m128i*) (packed_w + 440), _mm256_extracti128_si256(w29, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 192), _mm256_castsi256_si128(w6));
        _mm_storeu_si128((__m128i*) (packed_w + 448), _mm256_extracti128_si256(w6, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 200), _mm256_castsi256_si128(w14));
        _mm_storeu_si128((__m128i*) (packed_w + 456), _mm256_extracti128_si256(w14, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 208), _mm256_castsi256_si128(w22));
        _mm_storeu_si128((__m128i*) (packed_w + 464), _mm256_extracti128_si256(w22, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 216), _mm256_castsi256_si128(w30));
        _mm_storeu_si128((__m128i*) (packed_w + 472), _mm256_extracti128_si256(w30, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 224), _mm256_castsi256_si128(w7));
        _mm_storeu_si128((__m128i*) (packed_w + 480), _mm256_extracti128_si256(w7, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 232), _mm256_castsi256_si128(w15));
        _mm_storeu_si128((__m128i*) (packed_w + 488), _mm256_extracti128_si256(w15, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 240), _mm256_castsi256_si128(w23));
        _mm_storeu_si128((__m128i*) (packed_w + 496), _mm256_extracti128_si256(w23, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 248), _mm256_castsi256_si128(w31));
        _mm_storeu_si128((__m128i*) (packed_w + 504), _mm256_extracti128_si256(w31, 1));
        packed_w += 512;
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
          __m256i v16 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w16));
          w16 += 8;
          __m256i v17 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w17));
          w17 += 8;
          __m256i v18 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w18));
          w18 += 8;
          __m256i v19 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w19));
          w19 += 8;
          __m256i v20 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w20));
          w20 += 8;
          __m256i v21 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w21));
          w21 += 8;
          __m256i v22 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w22));
          w22 += 8;
          __m256i v23 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w23));
          w23 += 8;
          __m256i v24 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w24));
          w24 += 8;
          __m256i v25 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w25));
          w25 += 8;
          __m256i v26 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w26));
          w26 += 8;
          __m256i v27 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w27));
          w27 += 8;
          __m256i v28 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w28));
          w28 += 8;
          __m256i v29 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w29));
          w29 += 8;
          __m256i v30 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w30));
          w30 += 8;
          __m256i v31 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w31));
          w31 += 8;

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
          const __m256i t16 = _mm256_unpacklo_epi16(v16, v17);
          const __m256i t17 = _mm256_unpackhi_epi16(v16, v17);
          const __m256i t18 = _mm256_unpacklo_epi16(v18, v19);
          const __m256i t19 = _mm256_unpackhi_epi16(v18, v19);
          const __m256i t20 = _mm256_unpacklo_epi16(v20, v21);
          const __m256i t21 = _mm256_unpackhi_epi16(v20, v21);
          const __m256i t22 = _mm256_unpacklo_epi16(v22, v23);
          const __m256i t23 = _mm256_unpackhi_epi16(v22, v23);

          const __m256i u16 = _mm256_unpacklo_epi32(t16, t18);
          const __m256i u17 = _mm256_unpackhi_epi32(t16, t18);
          const __m256i u18 = _mm256_unpacklo_epi32(t17, t19);
          const __m256i u19 = _mm256_unpackhi_epi32(t17, t19);
          const __m256i u20 = _mm256_unpacklo_epi32(t20, t22);
          const __m256i u21 = _mm256_unpackhi_epi32(t20, t22);
          const __m256i u22 = _mm256_unpacklo_epi32(t21, t23);
          const __m256i u23 = _mm256_unpackhi_epi32(t21, t23);

          const __m256i w16 = _mm256_unpacklo_epi64(u16, u20);
          const __m256i w17 = _mm256_unpackhi_epi64(u16, u20);
          const __m256i w18 = _mm256_unpacklo_epi64(u17, u21);
          const __m256i w19 = _mm256_unpackhi_epi64(u17, u21);
          const __m256i w20 = _mm256_unpacklo_epi64(u18, u22);
          const __m256i w21 = _mm256_unpackhi_epi64(u18, u22);
          const __m256i w22 = _mm256_unpacklo_epi64(u19, u23);
          const __m256i w23 = _mm256_unpackhi_epi64(u19, u23);
          const __m256i t24 = _mm256_unpacklo_epi16(v24, v25);
          const __m256i t25 = _mm256_unpackhi_epi16(v24, v25);
          const __m256i t26 = _mm256_unpacklo_epi16(v26, v27);
          const __m256i t27 = _mm256_unpackhi_epi16(v26, v27);
          const __m256i t28 = _mm256_unpacklo_epi16(v28, v29);
          const __m256i t29 = _mm256_unpackhi_epi16(v28, v29);
          const __m256i t30 = _mm256_unpacklo_epi16(v30, v31);
          const __m256i t31 = _mm256_unpackhi_epi16(v30, v31);

          const __m256i u24 = _mm256_unpacklo_epi32(t24, t26);
          const __m256i u25 = _mm256_unpackhi_epi32(t24, t26);
          const __m256i u26 = _mm256_unpacklo_epi32(t25, t27);
          const __m256i u27 = _mm256_unpackhi_epi32(t25, t27);
          const __m256i u28 = _mm256_unpacklo_epi32(t28, t30);
          const __m256i u29 = _mm256_unpackhi_epi32(t28, t30);
          const __m256i u30 = _mm256_unpacklo_epi32(t29, t31);
          const __m256i u31 = _mm256_unpackhi_epi32(t29, t31);

          const __m256i w24 = _mm256_unpacklo_epi64(u24, u28);
          const __m256i w25 = _mm256_unpackhi_epi64(u24, u28);
          const __m256i w26 = _mm256_unpacklo_epi64(u25, u29);
          const __m256i w27 = _mm256_unpackhi_epi64(u25, u29);
          const __m256i w28 = _mm256_unpacklo_epi64(u26, u30);
          const __m256i w29 = _mm256_unpackhi_epi64(u26, u30);
          const __m256i w30 = _mm256_unpacklo_epi64(u27, u31);
          const __m256i w31 = _mm256_unpackhi_epi64(u27, u31);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w16));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w24));
          _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w9));
          _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w17));
          _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w25));
          _mm_storeu_si128((__m128i*) (packed_w + 64), _mm256_castsi256_si128(w2));
          _mm_storeu_si128((__m128i*) (packed_w + 72), _mm256_castsi256_si128(w10));
          _mm_storeu_si128((__m128i*) (packed_w + 80), _mm256_castsi256_si128(w18));
          _mm_storeu_si128((__m128i*) (packed_w + 88), _mm256_castsi256_si128(w26));
          _mm_storeu_si128((__m128i*) (packed_w + 96), _mm256_castsi256_si128(w3));
          _mm_storeu_si128((__m128i*) (packed_w + 104), _mm256_castsi256_si128(w11));
          _mm_storeu_si128((__m128i*) (packed_w + 112), _mm256_castsi256_si128(w19));
          _mm_storeu_si128((__m128i*) (packed_w + 120), _mm256_castsi256_si128(w27));
          _mm_storeu_si128((__m128i*) (packed_w + 128), _mm256_castsi256_si128(w4));
          _mm_storeu_si128((__m128i*) (packed_w + 136), _mm256_castsi256_si128(w12));
          _mm_storeu_si128((__m128i*) (packed_w + 144), _mm256_castsi256_si128(w20));
          _mm_storeu_si128((__m128i*) (packed_w + 152), _mm256_castsi256_si128(w28));
          _mm_storeu_si128((__m128i*) (packed_w + 160), _mm256_castsi256_si128(w5));
          _mm_storeu_si128((__m128i*) (packed_w + 168), _mm256_castsi256_si128(w13));
          _mm_storeu_si128((__m128i*) (packed_w + 176), _mm256_castsi256_si128(w21));
          _mm_storeu_si128((__m128i*) (packed_w + 184), _mm256_castsi256_si128(w29));
          _mm_storeu_si128((__m128i*) (packed_w + 192), _mm256_castsi256_si128(w6));
          _mm_storeu_si128((__m128i*) (packed_w + 200), _mm256_castsi256_si128(w14));
          _mm_storeu_si128((__m128i*) (packed_w + 208), _mm256_castsi256_si128(w22));
          _mm_storeu_si128((__m128i*) (packed_w + 216), _mm256_castsi256_si128(w30));
          _mm_storeu_si128((__m128i*) (packed_w + 224), _mm256_castsi256_si128(w7));
          _mm_storeu_si128((__m128i*) (packed_w + 232), _mm256_castsi256_si128(w15));
          _mm_storeu_si128((__m128i*) (packed_w + 240), _mm256_castsi256_si128(w23));
          _mm_storeu_si128((__m128i*) (packed_w + 248), _mm256_castsi256_si128(w31));
          packed_w += 256;
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
          __m256i v16 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w16));
          w16 += 4;
          __m256i v17 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w17));
          w17 += 4;
          __m256i v18 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w18));
          w18 += 4;
          __m256i v19 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w19));
          w19 += 4;
          __m256i v20 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w20));
          w20 += 4;
          __m256i v21 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w21));
          w21 += 4;
          __m256i v22 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w22));
          w22 += 4;
          __m256i v23 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w23));
          w23 += 4;
          __m256i v24 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w24));
          w24 += 4;
          __m256i v25 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w25));
          w25 += 4;
          __m256i v26 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w26));
          w26 += 4;
          __m256i v27 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w27));
          w27 += 4;
          __m256i v28 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w28));
          w28 += 4;
          __m256i v29 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w29));
          w29 += 4;
          __m256i v30 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w30));
          w30 += 4;
          __m256i v31 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w31));
          w31 += 4;

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
          const __m256i t16 = _mm256_unpacklo_epi16(v16, v17);
          const __m256i t18 = _mm256_unpacklo_epi16(v18, v19);
          const __m256i t20 = _mm256_unpacklo_epi16(v20, v21);
          const __m256i t22 = _mm256_unpacklo_epi16(v22, v23);

          const __m256i u16 = _mm256_unpacklo_epi32(t16, t18);
          const __m256i u17 = _mm256_unpackhi_epi32(t16, t18);
          const __m256i u20 = _mm256_unpacklo_epi32(t20, t22);
          const __m256i u21 = _mm256_unpackhi_epi32(t20, t22);

          const __m256i w16 = _mm256_unpacklo_epi64(u16, u20);
          const __m256i w17 = _mm256_unpackhi_epi64(u16, u20);
          const __m256i w18 = _mm256_unpacklo_epi64(u17, u21);
          const __m256i w19 = _mm256_unpackhi_epi64(u17, u21);
          const __m256i t24 = _mm256_unpacklo_epi16(v24, v25);
          const __m256i t26 = _mm256_unpacklo_epi16(v26, v27);
          const __m256i t28 = _mm256_unpacklo_epi16(v28, v29);
          const __m256i t30 = _mm256_unpacklo_epi16(v30, v31);

          const __m256i u24 = _mm256_unpacklo_epi32(t24, t26);
          const __m256i u25 = _mm256_unpackhi_epi32(t24, t26);
          const __m256i u28 = _mm256_unpacklo_epi32(t28, t30);
          const __m256i u29 = _mm256_unpackhi_epi32(t28, t30);

          const __m256i w24 = _mm256_unpacklo_epi64(u24, u28);
          const __m256i w25 = _mm256_unpackhi_epi64(u24, u28);
          const __m256i w26 = _mm256_unpacklo_epi64(u25, u29);
          const __m256i w27 = _mm256_unpackhi_epi64(u25, u29);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w16));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w24));
          _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w9));
          _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w17));
          _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w25));
          _mm_storeu_si128((__m128i*) (packed_w + 64), _mm256_castsi256_si128(w2));
          _mm_storeu_si128((__m128i*) (packed_w + 72), _mm256_castsi256_si128(w10));
          _mm_storeu_si128((__m128i*) (packed_w + 80), _mm256_castsi256_si128(w18));
          _mm_storeu_si128((__m128i*) (packed_w + 88), _mm256_castsi256_si128(w26));
          _mm_storeu_si128((__m128i*) (packed_w + 96), _mm256_castsi256_si128(w3));
          _mm_storeu_si128((__m128i*) (packed_w + 104), _mm256_castsi256_si128(w11));
          _mm_storeu_si128((__m128i*) (packed_w + 112), _mm256_castsi256_si128(w19));
          _mm_storeu_si128((__m128i*) (packed_w + 120), _mm256_castsi256_si128(w27));
          packed_w += 128;
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
          __m256i v16 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w16)));
          w16 += 2;
          __m256i v17 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w17)));
          w17 += 2;
          __m256i v18 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w18)));
          w18 += 2;
          __m256i v19 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w19)));
          w19 += 2;
          __m256i v20 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w20)));
          w20 += 2;
          __m256i v21 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w21)));
          w21 += 2;
          __m256i v22 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w22)));
          w22 += 2;
          __m256i v23 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w23)));
          w23 += 2;
          __m256i v24 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w24)));
          w24 += 2;
          __m256i v25 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w25)));
          w25 += 2;
          __m256i v26 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w26)));
          w26 += 2;
          __m256i v27 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w27)));
          w27 += 2;
          __m256i v28 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w28)));
          w28 += 2;
          __m256i v29 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w29)));
          w29 += 2;
          __m256i v30 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w30)));
          w30 += 2;
          __m256i v31 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w31)));
          w31 += 2;

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
          const __m256i t16 = _mm256_unpacklo_epi16(v16, v17);
          const __m256i t18 = _mm256_unpacklo_epi16(v18, v19);
          const __m256i t20 = _mm256_unpacklo_epi16(v20, v21);
          const __m256i t22 = _mm256_unpacklo_epi16(v22, v23);

          const __m256i u16 = _mm256_unpacklo_epi32(t16, t18);
          const __m256i u20 = _mm256_unpacklo_epi32(t20, t22);

          const __m256i w16 = _mm256_unpacklo_epi64(u16, u20);
          const __m256i w17 = _mm256_unpackhi_epi64(u16, u20);
          const __m256i t24 = _mm256_unpacklo_epi16(v24, v25);
          const __m256i t26 = _mm256_unpacklo_epi16(v26, v27);
          const __m256i t28 = _mm256_unpacklo_epi16(v28, v29);
          const __m256i t30 = _mm256_unpacklo_epi16(v30, v31);

          const __m256i u24 = _mm256_unpacklo_epi32(t24, t26);
          const __m256i u28 = _mm256_unpacklo_epi32(t28, t30);

          const __m256i w24 = _mm256_unpacklo_epi64(u24, u28);
          const __m256i w25 = _mm256_unpackhi_epi64(u24, u28);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w16));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w24));
          _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w9));
          _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w17));
          _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w25));
          packed_w += 64;
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
          __m256i v16 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w16, 0));
          w16 += 1;
          __m256i v17 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w17, 0));
          w17 += 1;
          __m256i v18 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w18, 0));
          w18 += 1;
          __m256i v19 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w19, 0));
          w19 += 1;
          __m256i v20 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w20, 0));
          w20 += 1;
          __m256i v21 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w21, 0));
          w21 += 1;
          __m256i v22 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w22, 0));
          w22 += 1;
          __m256i v23 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w23, 0));
          w23 += 1;
          __m256i v24 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w24, 0));
          w24 += 1;
          __m256i v25 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w25, 0));
          w25 += 1;
          __m256i v26 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w26, 0));
          w26 += 1;
          __m256i v27 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w27, 0));
          w27 += 1;
          __m256i v28 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w28, 0));
          w28 += 1;
          __m256i v29 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w29, 0));
          w29 += 1;
          __m256i v30 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w30, 0));
          w30 += 1;
          __m256i v31 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w31, 0));
          w31 += 1;

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
          const __m256i t16 = _mm256_unpacklo_epi16(v16, v17);
          const __m256i t18 = _mm256_unpacklo_epi16(v18, v19);
          const __m256i t20 = _mm256_unpacklo_epi16(v20, v21);
          const __m256i t22 = _mm256_unpacklo_epi16(v22, v23);

          const __m256i u16 = _mm256_unpacklo_epi32(t16, t18);
          const __m256i u20 = _mm256_unpacklo_epi32(t20, t22);

          const __m256i w16 = _mm256_unpacklo_epi64(u16, u20);
          const __m256i t24 = _mm256_unpacklo_epi16(v24, v25);
          const __m256i t26 = _mm256_unpacklo_epi16(v26, v27);
          const __m256i t28 = _mm256_unpacklo_epi16(v28, v29);
          const __m256i t30 = _mm256_unpacklo_epi16(v30, v31);

          const __m256i u24 = _mm256_unpacklo_epi32(t24, t26);
          const __m256i u28 = _mm256_unpacklo_epi32(t28, t30);

          const __m256i w24 = _mm256_unpacklo_epi64(u24, u28);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w16));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w24));
          packed_w += 32;
        }
      }
      packed_w = (uint16_t*) ((uintptr_t) packed_w + extra_bytes);
      w0 = w31;
    }

    // NC remainder (1..31)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 31);
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *packed_w++ = *b++;
        } while (--nb != 0);
        packed_w += (32 - n);
      } else {
        const __m128i vzero = _mm_setzero_si128();
        _mm_storeu_si128((__m128i*) (packed_w + 0), vzero);
        _mm_storeu_si128((__m128i*) (packed_w + 8), vzero);
        _mm_storeu_si128((__m128i*) (packed_w + 16), vzero);
        _mm_storeu_si128((__m128i*) (packed_w + 24), vzero);
        packed_w += 32;
      }

      // NR remainder has less than 32 rows so last row is not loaded
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
      const uint16_t* w15 = w14 + kc;
      if XNN_UNPREDICTABLE(n < 16) {
        w15 = w14;
      }
      const uint16_t* w16 = w15 + kc;
      if XNN_UNPREDICTABLE(n <= 16) {
        w16 = w15;
      }
      const uint16_t* w17 = w16 + kc;
      if XNN_UNPREDICTABLE(n < 18) {
        w17 = w16;
      }
      const uint16_t* w18 = w17 + kc;
      if XNN_UNPREDICTABLE(n <= 18) {
        w18 = w17;
      }
      const uint16_t* w19 = w18 + kc;
      if XNN_UNPREDICTABLE(n < 20) {
        w19 = w18;
      }
      const uint16_t* w20 = w19 + kc;
      if XNN_UNPREDICTABLE(n <= 20) {
        w20 = w19;
      }
      const uint16_t* w21 = w20 + kc;
      if XNN_UNPREDICTABLE(n < 22) {
        w21 = w20;
      }
      const uint16_t* w22 = w21 + kc;
      if XNN_UNPREDICTABLE(n <= 22) {
        w22 = w21;
      }
      const uint16_t* w23 = w22 + kc;
      if XNN_UNPREDICTABLE(n < 24) {
        w23 = w22;
      }
      const uint16_t* w24 = w23 + kc;
      if XNN_UNPREDICTABLE(n <= 24) {
        w24 = w23;
      }
      const uint16_t* w25 = w24 + kc;
      if XNN_UNPREDICTABLE(n < 26) {
        w25 = w24;
      }
      const uint16_t* w26 = w25 + kc;
      if XNN_UNPREDICTABLE(n <= 26) {
        w26 = w25;
      }
      const uint16_t* w27 = w26 + kc;
      if XNN_UNPREDICTABLE(n < 28) {
        w27 = w26;
      }
      const uint16_t* w28 = w27 + kc;
      if XNN_UNPREDICTABLE(n <= 28) {
        w28 = w27;
      }
      const uint16_t* w29 = w28 + kc;
      if XNN_UNPREDICTABLE(n < 30) {
        w29 = w28;
      }
      const uint16_t* w30 = w29 + kc;
      if XNN_UNPREDICTABLE(n <= 30) {
        w30 = w29;
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
        __m256i v15 = _mm256_loadu_si256((const __m256i*) w15);
        w15 += 16;
        __m256i v16 = _mm256_loadu_si256((const __m256i*) w16);
        w16 += 16;
        __m256i v17 = _mm256_loadu_si256((const __m256i*) w17);
        w17 += 16;
        __m256i v18 = _mm256_loadu_si256((const __m256i*) w18);
        w18 += 16;
        __m256i v19 = _mm256_loadu_si256((const __m256i*) w19);
        w19 += 16;
        __m256i v20 = _mm256_loadu_si256((const __m256i*) w20);
        w20 += 16;
        __m256i v21 = _mm256_loadu_si256((const __m256i*) w21);
        w21 += 16;
        __m256i v22 = _mm256_loadu_si256((const __m256i*) w22);
        w22 += 16;
        __m256i v23 = _mm256_loadu_si256((const __m256i*) w23);
        w23 += 16;
        __m256i v24 = _mm256_loadu_si256((const __m256i*) w24);
        w24 += 16;
        __m256i v25 = _mm256_loadu_si256((const __m256i*) w25);
        w25 += 16;
        __m256i v26 = _mm256_loadu_si256((const __m256i*) w26);
        w26 += 16;
        __m256i v27 = _mm256_loadu_si256((const __m256i*) w27);
        w27 += 16;
        __m256i v28 = _mm256_loadu_si256((const __m256i*) w28);
        w28 += 16;
        __m256i v29 = _mm256_loadu_si256((const __m256i*) w29);
        w29 += 16;
        __m256i v30 = _mm256_loadu_si256((const __m256i*) w30);
        w30 += 16;
        __m256i v31 = _mm256_setzero_si256();

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
        const __m256i t16 = _mm256_unpacklo_epi16(v16, v17);
        const __m256i t17 = _mm256_unpackhi_epi16(v16, v17);
        const __m256i t18 = _mm256_unpacklo_epi16(v18, v19);
        const __m256i t19 = _mm256_unpackhi_epi16(v18, v19);
        const __m256i t20 = _mm256_unpacklo_epi16(v20, v21);
        const __m256i t21 = _mm256_unpackhi_epi16(v20, v21);
        const __m256i t22 = _mm256_unpacklo_epi16(v22, v23);
        const __m256i t23 = _mm256_unpackhi_epi16(v22, v23);

        const __m256i u16 = _mm256_unpacklo_epi32(t16, t18);
        const __m256i u17 = _mm256_unpackhi_epi32(t16, t18);
        const __m256i u18 = _mm256_unpacklo_epi32(t17, t19);
        const __m256i u19 = _mm256_unpackhi_epi32(t17, t19);
        const __m256i u20 = _mm256_unpacklo_epi32(t20, t22);
        const __m256i u21 = _mm256_unpackhi_epi32(t20, t22);
        const __m256i u22 = _mm256_unpacklo_epi32(t21, t23);
        const __m256i u23 = _mm256_unpackhi_epi32(t21, t23);

        const __m256i w16 = _mm256_unpacklo_epi64(u16, u20);
        const __m256i w17 = _mm256_unpackhi_epi64(u16, u20);
        const __m256i w18 = _mm256_unpacklo_epi64(u17, u21);
        const __m256i w19 = _mm256_unpackhi_epi64(u17, u21);
        const __m256i w20 = _mm256_unpacklo_epi64(u18, u22);
        const __m256i w21 = _mm256_unpackhi_epi64(u18, u22);
        const __m256i w22 = _mm256_unpacklo_epi64(u19, u23);
        const __m256i w23 = _mm256_unpackhi_epi64(u19, u23);
        const __m256i t24 = _mm256_unpacklo_epi16(v24, v25);
        const __m256i t25 = _mm256_unpackhi_epi16(v24, v25);
        const __m256i t26 = _mm256_unpacklo_epi16(v26, v27);
        const __m256i t27 = _mm256_unpackhi_epi16(v26, v27);
        const __m256i t28 = _mm256_unpacklo_epi16(v28, v29);
        const __m256i t29 = _mm256_unpackhi_epi16(v28, v29);
        const __m256i t30 = _mm256_unpacklo_epi16(v30, v31);
        const __m256i t31 = _mm256_unpackhi_epi16(v30, v31);

        const __m256i u24 = _mm256_unpacklo_epi32(t24, t26);
        const __m256i u25 = _mm256_unpackhi_epi32(t24, t26);
        const __m256i u26 = _mm256_unpacklo_epi32(t25, t27);
        const __m256i u27 = _mm256_unpackhi_epi32(t25, t27);
        const __m256i u28 = _mm256_unpacklo_epi32(t28, t30);
        const __m256i u29 = _mm256_unpackhi_epi32(t28, t30);
        const __m256i u30 = _mm256_unpacklo_epi32(t29, t31);
        const __m256i u31 = _mm256_unpackhi_epi32(t29, t31);

        const __m256i w24 = _mm256_unpacklo_epi64(u24, u28);
        const __m256i w25 = _mm256_unpackhi_epi64(u24, u28);
        const __m256i w26 = _mm256_unpacklo_epi64(u25, u29);
        const __m256i w27 = _mm256_unpackhi_epi64(u25, u29);
        const __m256i w28 = _mm256_unpacklo_epi64(u26, u30);
        const __m256i w29 = _mm256_unpackhi_epi64(u26, u30);
        const __m256i w30 = _mm256_unpacklo_epi64(u27, u31);
        const __m256i w31 = _mm256_unpackhi_epi64(u27, u31);

        _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
        _mm_storeu_si128((__m128i*) (packed_w + 256), _mm256_extracti128_si256(w0, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
        _mm_storeu_si128((__m128i*) (packed_w + 264), _mm256_extracti128_si256(w8, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w16));
        _mm_storeu_si128((__m128i*) (packed_w + 272), _mm256_extracti128_si256(w16, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w24));
        _mm_storeu_si128((__m128i*) (packed_w + 280), _mm256_extracti128_si256(w24, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w1));
        _mm_storeu_si128((__m128i*) (packed_w + 288), _mm256_extracti128_si256(w1, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w9));
        _mm_storeu_si128((__m128i*) (packed_w + 296), _mm256_extracti128_si256(w9, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w17));
        _mm_storeu_si128((__m128i*) (packed_w + 304), _mm256_extracti128_si256(w17, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w25));
        _mm_storeu_si128((__m128i*) (packed_w + 312), _mm256_extracti128_si256(w25, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 64), _mm256_castsi256_si128(w2));
        _mm_storeu_si128((__m128i*) (packed_w + 320), _mm256_extracti128_si256(w2, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 72), _mm256_castsi256_si128(w10));
        _mm_storeu_si128((__m128i*) (packed_w + 328), _mm256_extracti128_si256(w10, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 80), _mm256_castsi256_si128(w18));
        _mm_storeu_si128((__m128i*) (packed_w + 336), _mm256_extracti128_si256(w18, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 88), _mm256_castsi256_si128(w26));
        _mm_storeu_si128((__m128i*) (packed_w + 344), _mm256_extracti128_si256(w26, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 96), _mm256_castsi256_si128(w3));
        _mm_storeu_si128((__m128i*) (packed_w + 352), _mm256_extracti128_si256(w3, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 104), _mm256_castsi256_si128(w11));
        _mm_storeu_si128((__m128i*) (packed_w + 360), _mm256_extracti128_si256(w11, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 112), _mm256_castsi256_si128(w19));
        _mm_storeu_si128((__m128i*) (packed_w + 368), _mm256_extracti128_si256(w19, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 120), _mm256_castsi256_si128(w27));
        _mm_storeu_si128((__m128i*) (packed_w + 376), _mm256_extracti128_si256(w27, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 128), _mm256_castsi256_si128(w4));
        _mm_storeu_si128((__m128i*) (packed_w + 384), _mm256_extracti128_si256(w4, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 136), _mm256_castsi256_si128(w12));
        _mm_storeu_si128((__m128i*) (packed_w + 392), _mm256_extracti128_si256(w12, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 144), _mm256_castsi256_si128(w20));
        _mm_storeu_si128((__m128i*) (packed_w + 400), _mm256_extracti128_si256(w20, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 152), _mm256_castsi256_si128(w28));
        _mm_storeu_si128((__m128i*) (packed_w + 408), _mm256_extracti128_si256(w28, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 160), _mm256_castsi256_si128(w5));
        _mm_storeu_si128((__m128i*) (packed_w + 416), _mm256_extracti128_si256(w5, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 168), _mm256_castsi256_si128(w13));
        _mm_storeu_si128((__m128i*) (packed_w + 424), _mm256_extracti128_si256(w13, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 176), _mm256_castsi256_si128(w21));
        _mm_storeu_si128((__m128i*) (packed_w + 432), _mm256_extracti128_si256(w21, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 184), _mm256_castsi256_si128(w29));
        _mm_storeu_si128((__m128i*) (packed_w + 440), _mm256_extracti128_si256(w29, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 192), _mm256_castsi256_si128(w6));
        _mm_storeu_si128((__m128i*) (packed_w + 448), _mm256_extracti128_si256(w6, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 200), _mm256_castsi256_si128(w14));
        _mm_storeu_si128((__m128i*) (packed_w + 456), _mm256_extracti128_si256(w14, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 208), _mm256_castsi256_si128(w22));
        _mm_storeu_si128((__m128i*) (packed_w + 464), _mm256_extracti128_si256(w22, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 216), _mm256_castsi256_si128(w30));
        _mm_storeu_si128((__m128i*) (packed_w + 472), _mm256_extracti128_si256(w30, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 224), _mm256_castsi256_si128(w7));
        _mm_storeu_si128((__m128i*) (packed_w + 480), _mm256_extracti128_si256(w7, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 232), _mm256_castsi256_si128(w15));
        _mm_storeu_si128((__m128i*) (packed_w + 488), _mm256_extracti128_si256(w15, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 240), _mm256_castsi256_si128(w23));
        _mm_storeu_si128((__m128i*) (packed_w + 496), _mm256_extracti128_si256(w23, 1));
        _mm_storeu_si128((__m128i*) (packed_w + 248), _mm256_castsi256_si128(w31));
        _mm_storeu_si128((__m128i*) (packed_w + 504), _mm256_extracti128_si256(w31, 1));
        packed_w += 512;
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
          __m256i v16 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w16));
          w16 += 8;
          __m256i v17 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w17));
          w17 += 8;
          __m256i v18 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w18));
          w18 += 8;
          __m256i v19 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w19));
          w19 += 8;
          __m256i v20 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w20));
          w20 += 8;
          __m256i v21 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w21));
          w21 += 8;
          __m256i v22 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w22));
          w22 += 8;
          __m256i v23 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w23));
          w23 += 8;
          __m256i v24 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w24));
          w24 += 8;
          __m256i v25 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w25));
          w25 += 8;
          __m256i v26 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w26));
          w26 += 8;
          __m256i v27 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w27));
          w27 += 8;
          __m256i v28 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w28));
          w28 += 8;
          __m256i v29 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w29));
          w29 += 8;
          __m256i v30 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) w30));
          w30 += 8;
          __m256i v31 = _mm256_setzero_si256();

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
          const __m256i t16 = _mm256_unpacklo_epi16(v16, v17);
          const __m256i t17 = _mm256_unpackhi_epi16(v16, v17);
          const __m256i t18 = _mm256_unpacklo_epi16(v18, v19);
          const __m256i t19 = _mm256_unpackhi_epi16(v18, v19);
          const __m256i t20 = _mm256_unpacklo_epi16(v20, v21);
          const __m256i t21 = _mm256_unpackhi_epi16(v20, v21);
          const __m256i t22 = _mm256_unpacklo_epi16(v22, v23);
          const __m256i t23 = _mm256_unpackhi_epi16(v22, v23);

          const __m256i u16 = _mm256_unpacklo_epi32(t16, t18);
          const __m256i u17 = _mm256_unpackhi_epi32(t16, t18);
          const __m256i u18 = _mm256_unpacklo_epi32(t17, t19);
          const __m256i u19 = _mm256_unpackhi_epi32(t17, t19);
          const __m256i u20 = _mm256_unpacklo_epi32(t20, t22);
          const __m256i u21 = _mm256_unpackhi_epi32(t20, t22);
          const __m256i u22 = _mm256_unpacklo_epi32(t21, t23);
          const __m256i u23 = _mm256_unpackhi_epi32(t21, t23);

          const __m256i w16 = _mm256_unpacklo_epi64(u16, u20);
          const __m256i w17 = _mm256_unpackhi_epi64(u16, u20);
          const __m256i w18 = _mm256_unpacklo_epi64(u17, u21);
          const __m256i w19 = _mm256_unpackhi_epi64(u17, u21);
          const __m256i w20 = _mm256_unpacklo_epi64(u18, u22);
          const __m256i w21 = _mm256_unpackhi_epi64(u18, u22);
          const __m256i w22 = _mm256_unpacklo_epi64(u19, u23);
          const __m256i w23 = _mm256_unpackhi_epi64(u19, u23);
          const __m256i t24 = _mm256_unpacklo_epi16(v24, v25);
          const __m256i t25 = _mm256_unpackhi_epi16(v24, v25);
          const __m256i t26 = _mm256_unpacklo_epi16(v26, v27);
          const __m256i t27 = _mm256_unpackhi_epi16(v26, v27);
          const __m256i t28 = _mm256_unpacklo_epi16(v28, v29);
          const __m256i t29 = _mm256_unpackhi_epi16(v28, v29);
          const __m256i t30 = _mm256_unpacklo_epi16(v30, v31);
          const __m256i t31 = _mm256_unpackhi_epi16(v30, v31);

          const __m256i u24 = _mm256_unpacklo_epi32(t24, t26);
          const __m256i u25 = _mm256_unpackhi_epi32(t24, t26);
          const __m256i u26 = _mm256_unpacklo_epi32(t25, t27);
          const __m256i u27 = _mm256_unpackhi_epi32(t25, t27);
          const __m256i u28 = _mm256_unpacklo_epi32(t28, t30);
          const __m256i u29 = _mm256_unpackhi_epi32(t28, t30);
          const __m256i u30 = _mm256_unpacklo_epi32(t29, t31);
          const __m256i u31 = _mm256_unpackhi_epi32(t29, t31);

          const __m256i w24 = _mm256_unpacklo_epi64(u24, u28);
          const __m256i w25 = _mm256_unpackhi_epi64(u24, u28);
          const __m256i w26 = _mm256_unpacklo_epi64(u25, u29);
          const __m256i w27 = _mm256_unpackhi_epi64(u25, u29);
          const __m256i w28 = _mm256_unpacklo_epi64(u26, u30);
          const __m256i w29 = _mm256_unpackhi_epi64(u26, u30);
          const __m256i w30 = _mm256_unpacklo_epi64(u27, u31);
          const __m256i w31 = _mm256_unpackhi_epi64(u27, u31);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w16));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w24));
          _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w9));
          _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w17));
          _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w25));
          _mm_storeu_si128((__m128i*) (packed_w + 64), _mm256_castsi256_si128(w2));
          _mm_storeu_si128((__m128i*) (packed_w + 72), _mm256_castsi256_si128(w10));
          _mm_storeu_si128((__m128i*) (packed_w + 80), _mm256_castsi256_si128(w18));
          _mm_storeu_si128((__m128i*) (packed_w + 88), _mm256_castsi256_si128(w26));
          _mm_storeu_si128((__m128i*) (packed_w + 96), _mm256_castsi256_si128(w3));
          _mm_storeu_si128((__m128i*) (packed_w + 104), _mm256_castsi256_si128(w11));
          _mm_storeu_si128((__m128i*) (packed_w + 112), _mm256_castsi256_si128(w19));
          _mm_storeu_si128((__m128i*) (packed_w + 120), _mm256_castsi256_si128(w27));
          _mm_storeu_si128((__m128i*) (packed_w + 128), _mm256_castsi256_si128(w4));
          _mm_storeu_si128((__m128i*) (packed_w + 136), _mm256_castsi256_si128(w12));
          _mm_storeu_si128((__m128i*) (packed_w + 144), _mm256_castsi256_si128(w20));
          _mm_storeu_si128((__m128i*) (packed_w + 152), _mm256_castsi256_si128(w28));
          _mm_storeu_si128((__m128i*) (packed_w + 160), _mm256_castsi256_si128(w5));
          _mm_storeu_si128((__m128i*) (packed_w + 168), _mm256_castsi256_si128(w13));
          _mm_storeu_si128((__m128i*) (packed_w + 176), _mm256_castsi256_si128(w21));
          _mm_storeu_si128((__m128i*) (packed_w + 184), _mm256_castsi256_si128(w29));
          _mm_storeu_si128((__m128i*) (packed_w + 192), _mm256_castsi256_si128(w6));
          _mm_storeu_si128((__m128i*) (packed_w + 200), _mm256_castsi256_si128(w14));
          _mm_storeu_si128((__m128i*) (packed_w + 208), _mm256_castsi256_si128(w22));
          _mm_storeu_si128((__m128i*) (packed_w + 216), _mm256_castsi256_si128(w30));
          _mm_storeu_si128((__m128i*) (packed_w + 224), _mm256_castsi256_si128(w7));
          _mm_storeu_si128((__m128i*) (packed_w + 232), _mm256_castsi256_si128(w15));
          _mm_storeu_si128((__m128i*) (packed_w + 240), _mm256_castsi256_si128(w23));
          _mm_storeu_si128((__m128i*) (packed_w + 248), _mm256_castsi256_si128(w31));
          packed_w += 256;
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
          __m256i v16 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w16));
          w16 += 4;
          __m256i v17 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w17));
          w17 += 4;
          __m256i v18 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w18));
          w18 += 4;
          __m256i v19 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w19));
          w19 += 4;
          __m256i v20 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w20));
          w20 += 4;
          __m256i v21 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w21));
          w21 += 4;
          __m256i v22 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w22));
          w22 += 4;
          __m256i v23 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w23));
          w23 += 4;
          __m256i v24 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w24));
          w24 += 4;
          __m256i v25 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w25));
          w25 += 4;
          __m256i v26 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w26));
          w26 += 4;
          __m256i v27 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w27));
          w27 += 4;
          __m256i v28 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w28));
          w28 += 4;
          __m256i v29 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w29));
          w29 += 4;
          __m256i v30 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) w30));
          w30 += 4;
          __m256i v31 = _mm256_setzero_si256();

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
          const __m256i t16 = _mm256_unpacklo_epi16(v16, v17);
          const __m256i t18 = _mm256_unpacklo_epi16(v18, v19);
          const __m256i t20 = _mm256_unpacklo_epi16(v20, v21);
          const __m256i t22 = _mm256_unpacklo_epi16(v22, v23);

          const __m256i u16 = _mm256_unpacklo_epi32(t16, t18);
          const __m256i u17 = _mm256_unpackhi_epi32(t16, t18);
          const __m256i u20 = _mm256_unpacklo_epi32(t20, t22);
          const __m256i u21 = _mm256_unpackhi_epi32(t20, t22);

          const __m256i w16 = _mm256_unpacklo_epi64(u16, u20);
          const __m256i w17 = _mm256_unpackhi_epi64(u16, u20);
          const __m256i w18 = _mm256_unpacklo_epi64(u17, u21);
          const __m256i w19 = _mm256_unpackhi_epi64(u17, u21);
          const __m256i t24 = _mm256_unpacklo_epi16(v24, v25);
          const __m256i t26 = _mm256_unpacklo_epi16(v26, v27);
          const __m256i t28 = _mm256_unpacklo_epi16(v28, v29);
          const __m256i t30 = _mm256_unpacklo_epi16(v30, v31);

          const __m256i u24 = _mm256_unpacklo_epi32(t24, t26);
          const __m256i u25 = _mm256_unpackhi_epi32(t24, t26);
          const __m256i u28 = _mm256_unpacklo_epi32(t28, t30);
          const __m256i u29 = _mm256_unpackhi_epi32(t28, t30);

          const __m256i w24 = _mm256_unpacklo_epi64(u24, u28);
          const __m256i w25 = _mm256_unpackhi_epi64(u24, u28);
          const __m256i w26 = _mm256_unpacklo_epi64(u25, u29);
          const __m256i w27 = _mm256_unpackhi_epi64(u25, u29);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w16));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w24));
          _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w9));
          _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w17));
          _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w25));
          _mm_storeu_si128((__m128i*) (packed_w + 64), _mm256_castsi256_si128(w2));
          _mm_storeu_si128((__m128i*) (packed_w + 72), _mm256_castsi256_si128(w10));
          _mm_storeu_si128((__m128i*) (packed_w + 80), _mm256_castsi256_si128(w18));
          _mm_storeu_si128((__m128i*) (packed_w + 88), _mm256_castsi256_si128(w26));
          _mm_storeu_si128((__m128i*) (packed_w + 96), _mm256_castsi256_si128(w3));
          _mm_storeu_si128((__m128i*) (packed_w + 104), _mm256_castsi256_si128(w11));
          _mm_storeu_si128((__m128i*) (packed_w + 112), _mm256_castsi256_si128(w19));
          _mm_storeu_si128((__m128i*) (packed_w + 120), _mm256_castsi256_si128(w27));
          packed_w += 128;
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
          __m256i v16 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w16)));
          w16 += 2;
          __m256i v17 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w17)));
          w17 += 2;
          __m256i v18 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w18)));
          w18 += 2;
          __m256i v19 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w19)));
          w19 += 2;
          __m256i v20 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w20)));
          w20 += 2;
          __m256i v21 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w21)));
          w21 += 2;
          __m256i v22 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w22)));
          w22 += 2;
          __m256i v23 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w23)));
          w23 += 2;
          __m256i v24 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w24)));
          w24 += 2;
          __m256i v25 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w25)));
          w25 += 2;
          __m256i v26 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w26)));
          w26 += 2;
          __m256i v27 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w27)));
          w27 += 2;
          __m256i v28 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w28)));
          w28 += 2;
          __m256i v29 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w29)));
          w29 += 2;
          __m256i v30 = _mm256_castsi128_si256(_mm_cvtsi32_si128((int) unaligned_load_u32(w30)));
          w30 += 2;
          __m256i v31 = _mm256_setzero_si256();

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
          const __m256i t16 = _mm256_unpacklo_epi16(v16, v17);
          const __m256i t18 = _mm256_unpacklo_epi16(v18, v19);
          const __m256i t20 = _mm256_unpacklo_epi16(v20, v21);
          const __m256i t22 = _mm256_unpacklo_epi16(v22, v23);

          const __m256i u16 = _mm256_unpacklo_epi32(t16, t18);
          const __m256i u20 = _mm256_unpacklo_epi32(t20, t22);

          const __m256i w16 = _mm256_unpacklo_epi64(u16, u20);
          const __m256i w17 = _mm256_unpackhi_epi64(u16, u20);
          const __m256i t24 = _mm256_unpacklo_epi16(v24, v25);
          const __m256i t26 = _mm256_unpacklo_epi16(v26, v27);
          const __m256i t28 = _mm256_unpacklo_epi16(v28, v29);
          const __m256i t30 = _mm256_unpacklo_epi16(v30, v31);

          const __m256i u24 = _mm256_unpacklo_epi32(t24, t26);
          const __m256i u28 = _mm256_unpacklo_epi32(t28, t30);

          const __m256i w24 = _mm256_unpacklo_epi64(u24, u28);
          const __m256i w25 = _mm256_unpackhi_epi64(u24, u28);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w16));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w24));
          _mm_storeu_si128((__m128i*) (packed_w + 32), _mm256_castsi256_si128(w1));
          _mm_storeu_si128((__m128i*) (packed_w + 40), _mm256_castsi256_si128(w9));
          _mm_storeu_si128((__m128i*) (packed_w + 48), _mm256_castsi256_si128(w17));
          _mm_storeu_si128((__m128i*) (packed_w + 56), _mm256_castsi256_si128(w25));
          packed_w += 64;
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
          __m256i v16 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w16, 0));
          w16 += 1;
          __m256i v17 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w17, 0));
          w17 += 1;
          __m256i v18 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w18, 0));
          w18 += 1;
          __m256i v19 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w19, 0));
          w19 += 1;
          __m256i v20 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w20, 0));
          w20 += 1;
          __m256i v21 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w21, 0));
          w21 += 1;
          __m256i v22 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w22, 0));
          w22 += 1;
          __m256i v23 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w23, 0));
          w23 += 1;
          __m256i v24 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w24, 0));
          w24 += 1;
          __m256i v25 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w25, 0));
          w25 += 1;
          __m256i v26 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w26, 0));
          w26 += 1;
          __m256i v27 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w27, 0));
          w27 += 1;
          __m256i v28 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w28, 0));
          w28 += 1;
          __m256i v29 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w29, 0));
          w29 += 1;
          __m256i v30 = _mm256_castsi128_si256(_mm_insert_epi16(_mm_setzero_si128(), *w30, 0));
          w30 += 1;
          __m256i v31 = _mm256_setzero_si256();

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
          const __m256i t16 = _mm256_unpacklo_epi16(v16, v17);
          const __m256i t18 = _mm256_unpacklo_epi16(v18, v19);
          const __m256i t20 = _mm256_unpacklo_epi16(v20, v21);
          const __m256i t22 = _mm256_unpacklo_epi16(v22, v23);

          const __m256i u16 = _mm256_unpacklo_epi32(t16, t18);
          const __m256i u20 = _mm256_unpacklo_epi32(t20, t22);

          const __m256i w16 = _mm256_unpacklo_epi64(u16, u20);
          const __m256i t24 = _mm256_unpacklo_epi16(v24, v25);
          const __m256i t26 = _mm256_unpacklo_epi16(v26, v27);
          const __m256i t28 = _mm256_unpacklo_epi16(v28, v29);
          const __m256i t30 = _mm256_unpacklo_epi16(v30, v31);

          const __m256i u24 = _mm256_unpacklo_epi32(t24, t26);
          const __m256i u28 = _mm256_unpacklo_epi32(t28, t30);

          const __m256i w24 = _mm256_unpacklo_epi64(u24, u28);

          _mm_storeu_si128((__m128i*) (packed_w + 0), _mm256_castsi256_si128(w0));
          _mm_storeu_si128((__m128i*) (packed_w + 8), _mm256_castsi256_si128(w8));
          _mm_storeu_si128((__m128i*) (packed_w + 16), _mm256_castsi256_si128(w16));
          _mm_storeu_si128((__m128i*) (packed_w + 24), _mm256_castsi256_si128(w24));
          packed_w += 32;
        }
      }
      packed_w = (uint16_t*) ((uintptr_t) packed_w + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
