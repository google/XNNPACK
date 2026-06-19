// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x32-packw/avx512c2.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/packw.h"
#include "src/xnnpack/prefetch.h"


void xnn_x32_packw_gemm_goi_ukernel_x32c2__avx512f_u4_prfm(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint32_t* weights,
  const uint32_t* bias,
  const void* scale,
  uint32_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 32);
  assert(kr == 2);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const float* b = (const float*) bias;
  float* packed_w = (float*) packed_weights;
  do {
    // NC main loop multiple of 32
    const float* w0 = (const float*) weights;
    size_t n = nc;

    for (; n >= 32; n -= 32) {
      if XNN_LIKELY(b != NULL) {
        const __m512 vb0 = _mm512_loadu_ps(b);
        const __m512 vb16 = _mm512_loadu_ps(b + 16);
        _mm512_store_ps(packed_w, vb0);
        _mm512_store_ps(packed_w + 16, vb16);
        b += 32;
      } else {
        const __m512 vzero = _mm512_setzero_ps();
        _mm512_store_ps(packed_w, vzero);
        _mm512_store_ps(packed_w + 16, vzero);
      }
      packed_w += 32;

      const float* w1 = w0 + kc;
      const float* w2 = w1 + kc;
      const float* w3 = w2 + kc;
      const float* w4 = w3 + kc;
      const float* w5 = w4 + kc;
      const float* w6 = w5 + kc;
      const float* w7 = w6 + kc;
      const float* w8 = w7 + kc;
      const float* w9 = w8 + kc;
      const float* w10 = w9 + kc;
      const float* w11 = w10 + kc;
      const float* w12 = w11 + kc;
      const float* w13 = w12 + kc;
      const float* w14 = w13 + kc;
      const float* w15 = w14 + kc;
      const float* w16 = w15 + kc;
      const float* w17 = w16 + kc;
      const float* w18 = w17 + kc;
      const float* w19 = w18 + kc;
      const float* w20 = w19 + kc;
      const float* w21 = w20 + kc;
      const float* w22 = w21 + kc;
      const float* w23 = w22 + kc;
      const float* w24 = w23 + kc;
      const float* w25 = w24 + kc;
      const float* w26 = w25 + kc;
      const float* w27 = w26 + kc;
      const float* w28 = w27 + kc;
      const float* w29 = w28 + kc;
      const float* w30 = w29 + kc;
      const float* w31 = w30 + kc;
      xnn_prefetch_to_l1((const int8_t*) w0);
      xnn_prefetch_to_l1((const int8_t*) w0 + 64);
      xnn_prefetch_to_l1((const int8_t*) w1);
      xnn_prefetch_to_l1((const int8_t*) w1 + 64);
      xnn_prefetch_to_l1((const int8_t*) w2);
      xnn_prefetch_to_l1((const int8_t*) w2 + 64);
      xnn_prefetch_to_l1((const int8_t*) w3);
      xnn_prefetch_to_l1((const int8_t*) w3 + 64);
      xnn_prefetch_to_l1((const int8_t*) w4);
      xnn_prefetch_to_l1((const int8_t*) w4 + 64);
      xnn_prefetch_to_l1((const int8_t*) w5);
      xnn_prefetch_to_l1((const int8_t*) w5 + 64);
      xnn_prefetch_to_l1((const int8_t*) w6);
      xnn_prefetch_to_l1((const int8_t*) w6 + 64);
      xnn_prefetch_to_l1((const int8_t*) w7);
      xnn_prefetch_to_l1((const int8_t*) w7 + 64);
      xnn_prefetch_to_l1((const int8_t*) w8);
      xnn_prefetch_to_l1((const int8_t*) w8 + 64);
      xnn_prefetch_to_l1((const int8_t*) w9);
      xnn_prefetch_to_l1((const int8_t*) w9 + 64);
      xnn_prefetch_to_l1((const int8_t*) w10);
      xnn_prefetch_to_l1((const int8_t*) w10 + 64);
      xnn_prefetch_to_l1((const int8_t*) w11);
      xnn_prefetch_to_l1((const int8_t*) w11 + 64);
      xnn_prefetch_to_l1((const int8_t*) w12);
      xnn_prefetch_to_l1((const int8_t*) w12 + 64);
      xnn_prefetch_to_l1((const int8_t*) w13);
      xnn_prefetch_to_l1((const int8_t*) w13 + 64);
      xnn_prefetch_to_l1((const int8_t*) w14);
      xnn_prefetch_to_l1((const int8_t*) w14 + 64);
      xnn_prefetch_to_l1((const int8_t*) w15);
      xnn_prefetch_to_l1((const int8_t*) w15 + 64);
      xnn_prefetch_to_l1((const int8_t*) w16);
      xnn_prefetch_to_l1((const int8_t*) w16 + 64);
      xnn_prefetch_to_l1((const int8_t*) w17);
      xnn_prefetch_to_l1((const int8_t*) w17 + 64);
      xnn_prefetch_to_l1((const int8_t*) w18);
      xnn_prefetch_to_l1((const int8_t*) w18 + 64);
      xnn_prefetch_to_l1((const int8_t*) w19);
      xnn_prefetch_to_l1((const int8_t*) w19 + 64);
      xnn_prefetch_to_l1((const int8_t*) w20);
      xnn_prefetch_to_l1((const int8_t*) w20 + 64);
      xnn_prefetch_to_l1((const int8_t*) w21);
      xnn_prefetch_to_l1((const int8_t*) w21 + 64);
      xnn_prefetch_to_l1((const int8_t*) w22);
      xnn_prefetch_to_l1((const int8_t*) w22 + 64);
      xnn_prefetch_to_l1((const int8_t*) w23);
      xnn_prefetch_to_l1((const int8_t*) w23 + 64);
      xnn_prefetch_to_l1((const int8_t*) w24);
      xnn_prefetch_to_l1((const int8_t*) w24 + 64);
      xnn_prefetch_to_l1((const int8_t*) w25);
      xnn_prefetch_to_l1((const int8_t*) w25 + 64);
      xnn_prefetch_to_l1((const int8_t*) w26);
      xnn_prefetch_to_l1((const int8_t*) w26 + 64);
      xnn_prefetch_to_l1((const int8_t*) w27);
      xnn_prefetch_to_l1((const int8_t*) w27 + 64);
      xnn_prefetch_to_l1((const int8_t*) w28);
      xnn_prefetch_to_l1((const int8_t*) w28 + 64);
      xnn_prefetch_to_l1((const int8_t*) w29);
      xnn_prefetch_to_l1((const int8_t*) w29 + 64);
      xnn_prefetch_to_l1((const int8_t*) w30);
      xnn_prefetch_to_l1((const int8_t*) w30 + 64);
      xnn_prefetch_to_l1((const int8_t*) w31);
      xnn_prefetch_to_l1((const int8_t*) w31 + 64);

      // KC main loop multiple of 4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        // Read blocks of 4x4
        // Load first 4 rows of N into low part of each register
        __m512 v0x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w0));
        w0 += 4;
        __m512 v1x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w1));
        w1 += 4;
        __m512 v8x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w8));
        w8 += 4;
        __m512 v9x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w9));
        w9 += 4;
        __m512 v16x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w16));
        w16 += 4;
        __m512 v17x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w17));
        w17 += 4;
        __m512 v24x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w24));
        w24 += 4;
        __m512 v25x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w25));
        w25 += 4;
        // Load next 4 rows of N into the high part of each register
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w2), 1);
        w2 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w3), 1);
        w3 += 4;
        v8x0123 = _mm512_insertf32x4(v8x0123, _mm_loadu_ps(w10), 1);
        w10 += 4;
        v9x0123 = _mm512_insertf32x4(v9x0123, _mm_loadu_ps(w11), 1);
        w11 += 4;
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w4), 2);
        w4 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w5), 2);
        w5 += 4;
        v8x0123 = _mm512_insertf32x4(v8x0123, _mm_loadu_ps(w12), 2);
        w12 += 4;
        v9x0123 = _mm512_insertf32x4(v9x0123, _mm_loadu_ps(w13), 2);
        w13 += 4;
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w6), 3);
        w6 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w7), 3);
        w7 += 4;
        v8x0123 = _mm512_insertf32x4(v8x0123, _mm_loadu_ps(w14), 3);
        w14 += 4;
        v9x0123 = _mm512_insertf32x4(v9x0123, _mm_loadu_ps(w15), 3);
        w15 += 4;
        v16x0123 = _mm512_insertf32x4(v16x0123, _mm_loadu_ps(w18), 1);
        w18 += 4;
        v17x0123 = _mm512_insertf32x4(v17x0123, _mm_loadu_ps(w19), 1);
        w19 += 4;
        v24x0123 = _mm512_insertf32x4(v24x0123, _mm_loadu_ps(w26), 1);
        w26 += 4;
        v25x0123 = _mm512_insertf32x4(v25x0123, _mm_loadu_ps(w27), 1);
        w27 += 4;
        v16x0123 = _mm512_insertf32x4(v16x0123, _mm_loadu_ps(w20), 2);
        w20 += 4;
        v17x0123 = _mm512_insertf32x4(v17x0123, _mm_loadu_ps(w21), 2);
        w21 += 4;
        v24x0123 = _mm512_insertf32x4(v24x0123, _mm_loadu_ps(w28), 2);
        w28 += 4;
        v25x0123 = _mm512_insertf32x4(v25x0123, _mm_loadu_ps(w29), 2);
        w29 += 4;
        v16x0123 = _mm512_insertf32x4(v16x0123, _mm_loadu_ps(w22), 3);
        w22 += 4;
        v17x0123 = _mm512_insertf32x4(v17x0123, _mm_loadu_ps(w23), 3);
        w23 += 4;
        v24x0123 = _mm512_insertf32x4(v24x0123, _mm_loadu_ps(w30), 3);
        w30 += 4;
        v25x0123 = _mm512_insertf32x4(v25x0123, _mm_loadu_ps(w31), 3);
        w31 += 4;

        // Transpose 2x2
        const __m512 vres0x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(v0x0123), _mm512_castps_pd(v1x0123)));
        const __m512 vres16x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(v8x0123), _mm512_castps_pd(v9x0123)));
        const __m512 vres64x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(v0x0123), _mm512_castps_pd(v1x0123)));
        const __m512 vres80x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(v8x0123), _mm512_castps_pd(v9x0123)));
        const __m512 vres32x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(v16x0123), _mm512_castps_pd(v17x0123)));
        const __m512 vres48x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(v24x0123), _mm512_castps_pd(v25x0123)));
        const __m512 vres96x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(v16x0123), _mm512_castps_pd(v17x0123)));
        const __m512 vres112x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(v24x0123), _mm512_castps_pd(v25x0123)));
        xnn_prefetch_to_l1((const int8_t*) w0 + 128);
        xnn_prefetch_to_l1((const int8_t*) w1 + 128);
        xnn_prefetch_to_l1((const int8_t*) w2 + 128);
        xnn_prefetch_to_l1((const int8_t*) w3 + 128);
        xnn_prefetch_to_l1((const int8_t*) w4 + 128);
        xnn_prefetch_to_l1((const int8_t*) w5 + 128);
        xnn_prefetch_to_l1((const int8_t*) w6 + 128);
        xnn_prefetch_to_l1((const int8_t*) w7 + 128);
        xnn_prefetch_to_l1((const int8_t*) w8 + 128);
        xnn_prefetch_to_l1((const int8_t*) w9 + 128);
        xnn_prefetch_to_l1((const int8_t*) w10 + 128);
        xnn_prefetch_to_l1((const int8_t*) w11 + 128);
        xnn_prefetch_to_l1((const int8_t*) w12 + 128);
        xnn_prefetch_to_l1((const int8_t*) w13 + 128);
        xnn_prefetch_to_l1((const int8_t*) w14 + 128);
        xnn_prefetch_to_l1((const int8_t*) w15 + 128);
        xnn_prefetch_to_l1((const int8_t*) w16 + 128);
        xnn_prefetch_to_l1((const int8_t*) w17 + 128);
        xnn_prefetch_to_l1((const int8_t*) w18 + 128);
        xnn_prefetch_to_l1((const int8_t*) w19 + 128);
        xnn_prefetch_to_l1((const int8_t*) w20 + 128);
        xnn_prefetch_to_l1((const int8_t*) w21 + 128);
        xnn_prefetch_to_l1((const int8_t*) w22 + 128);
        xnn_prefetch_to_l1((const int8_t*) w23 + 128);
        xnn_prefetch_to_l1((const int8_t*) w24 + 128);
        xnn_prefetch_to_l1((const int8_t*) w25 + 128);
        xnn_prefetch_to_l1((const int8_t*) w26 + 128);
        xnn_prefetch_to_l1((const int8_t*) w27 + 128);
        xnn_prefetch_to_l1((const int8_t*) w28 + 128);
        xnn_prefetch_to_l1((const int8_t*) w29 + 128);
        xnn_prefetch_to_l1((const int8_t*) w30 + 128);
        xnn_prefetch_to_l1((const int8_t*) w31 + 128);

        _mm512_store_ps(packed_w, vres0x0123);
        _mm512_store_ps(packed_w + 16, vres16x0123);
        _mm512_store_ps(packed_w + 32, vres32x0123);
        _mm512_store_ps(packed_w + 48, vres48x0123);
        _mm512_store_ps(packed_w + 64, vres64x0123);
        _mm512_store_ps(packed_w + 80, vres80x0123);
        _mm512_store_ps(packed_w + 96, vres96x0123);
        _mm512_store_ps(packed_w + 112, vres112x0123);
        packed_w += 128;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        if (k & 2) {
          // Read blocks of 4x2
          __m128 v0 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w0));
          w0 += 2;
          __m128 v1 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w1));
          w1 += 2;
          __m128 v2 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w2));
          w2 += 2;
          __m128 v3 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w3));
          w3 += 2;
          __m128 v4 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w4));
          w4 += 2;
          __m128 v5 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w5));
          w5 += 2;
          __m128 v6 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w6));
          w6 += 2;
          __m128 v7 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w7));
          w7 += 2;
          __m128 v8 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w8));
          w8 += 2;
          __m128 v9 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w9));
          w9 += 2;
          __m128 v10 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w10));
          w10 += 2;
          __m128 v11 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w11));
          w11 += 2;
          __m128 v12 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w12));
          w12 += 2;
          __m128 v13 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w13));
          w13 += 2;
          __m128 v14 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w14));
          w14 += 2;
          __m128 v15 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w15));
          w15 += 2;
          __m128 v16 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w16));
          w16 += 2;
          __m128 v17 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w17));
          w17 += 2;
          __m128 v18 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w18));
          w18 += 2;
          __m128 v19 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w19));
          w19 += 2;
          __m128 v20 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w20));
          w20 += 2;
          __m128 v21 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w21));
          w21 += 2;
          __m128 v22 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w22));
          w22 += 2;
          __m128 v23 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w23));
          w23 += 2;
          __m128 v24 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w24));
          w24 += 2;
          __m128 v25 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w25));
          w25 += 2;
          __m128 v26 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w26));
          w26 += 2;
          __m128 v27 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w27));
          w27 += 2;
          __m128 v28 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w28));
          w28 += 2;
          __m128 v29 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w29));
          w29 += 2;
          __m128 v30 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w30));
          w30 += 2;
          __m128 v31 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w31));
          w31 += 2;

          // Transpose 2x2
          const __m128 vres0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v0), _mm_castps_pd(v1)));
          const __m128 vres1 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v2), _mm_castps_pd(v3)));
          // Transpose 2x2
          const __m128 vres2 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v4), _mm_castps_pd(v5)));
          const __m128 vres3 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v6), _mm_castps_pd(v7)));
          // Transpose 2x2
          const __m128 vres4 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v8), _mm_castps_pd(v9)));
          const __m128 vres5 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v10), _mm_castps_pd(v11)));
          // Transpose 2x2
          const __m128 vres6 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v12), _mm_castps_pd(v13)));
          const __m128 vres7 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v14), _mm_castps_pd(v15)));
          // Transpose 2x2
          const __m128 vres8 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v16), _mm_castps_pd(v17)));
          const __m128 vres9 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v18), _mm_castps_pd(v19)));
          // Transpose 2x2
          const __m128 vres10 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v20), _mm_castps_pd(v21)));
          const __m128 vres11 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v22), _mm_castps_pd(v23)));
          // Transpose 2x2
          const __m128 vres12 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v24), _mm_castps_pd(v25)));
          const __m128 vres13 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v26), _mm_castps_pd(v27)));
          // Transpose 2x2
          const __m128 vres14 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v28), _mm_castps_pd(v29)));
          const __m128 vres15 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v30), _mm_castps_pd(v31)));

          _mm_store_ps(packed_w, vres0);
          _mm_store_ps(packed_w + 4, vres1);
          _mm_store_ps(packed_w + 8, vres2);
          _mm_store_ps(packed_w + 12, vres3);
          _mm_store_ps(packed_w + 16, vres4);
          _mm_store_ps(packed_w + 20, vres5);
          _mm_store_ps(packed_w + 24, vres6);
          _mm_store_ps(packed_w + 28, vres7);
          _mm_store_ps(packed_w + 32, vres8);
          _mm_store_ps(packed_w + 36, vres9);
          _mm_store_ps(packed_w + 40, vres10);
          _mm_store_ps(packed_w + 44, vres11);
          _mm_store_ps(packed_w + 48, vres12);
          _mm_store_ps(packed_w + 52, vres13);
          _mm_store_ps(packed_w + 56, vres14);
          _mm_store_ps(packed_w + 60, vres15);
          packed_w += 64;
        }
        if (k & 1) {
          // Read blocks of 4x1
          __m128 v0 = _mm_load_ss(w0);
          w0 += 1;
          __m128 v1 = _mm_load_ss(w1);
          w1 += 1;
          __m128 v2 = _mm_load_ss(w2);
          w2 += 1;
          __m128 v3 = _mm_load_ss(w3);
          w3 += 1;
          __m128 v4 = _mm_load_ss(w4);
          w4 += 1;
          __m128 v5 = _mm_load_ss(w5);
          w5 += 1;
          __m128 v6 = _mm_load_ss(w6);
          w6 += 1;
          __m128 v7 = _mm_load_ss(w7);
          w7 += 1;
          __m128 v8 = _mm_load_ss(w8);
          w8 += 1;
          __m128 v9 = _mm_load_ss(w9);
          w9 += 1;
          __m128 v10 = _mm_load_ss(w10);
          w10 += 1;
          __m128 v11 = _mm_load_ss(w11);
          w11 += 1;
          __m128 v12 = _mm_load_ss(w12);
          w12 += 1;
          __m128 v13 = _mm_load_ss(w13);
          w13 += 1;
          __m128 v14 = _mm_load_ss(w14);
          w14 += 1;
          __m128 v15 = _mm_load_ss(w15);
          w15 += 1;
          __m128 v16 = _mm_load_ss(w16);
          w16 += 1;
          __m128 v17 = _mm_load_ss(w17);
          w17 += 1;
          __m128 v18 = _mm_load_ss(w18);
          w18 += 1;
          __m128 v19 = _mm_load_ss(w19);
          w19 += 1;
          __m128 v20 = _mm_load_ss(w20);
          w20 += 1;
          __m128 v21 = _mm_load_ss(w21);
          w21 += 1;
          __m128 v22 = _mm_load_ss(w22);
          w22 += 1;
          __m128 v23 = _mm_load_ss(w23);
          w23 += 1;
          __m128 v24 = _mm_load_ss(w24);
          w24 += 1;
          __m128 v25 = _mm_load_ss(w25);
          w25 += 1;
          __m128 v26 = _mm_load_ss(w26);
          w26 += 1;
          __m128 v27 = _mm_load_ss(w27);
          w27 += 1;
          __m128 v28 = _mm_load_ss(w28);
          w28 += 1;
          __m128 v29 = _mm_load_ss(w29);
          w29 += 1;
          __m128 v30 = _mm_load_ss(w30);
          w30 += 1;
          __m128 v31 = _mm_load_ss(w31);
          w31 += 1;

          // Transpose 2x2
          const __m128 vres0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v0), _mm_castps_pd(v1)));  // a b  from row 0, 1
          // Transpose 2x2
          const __m128 vres1 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v2), _mm_castps_pd(v3)));  // a b  from row 0, 1
          // Transpose 2x2
          const __m128 vres2 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v4), _mm_castps_pd(v5)));  // a b  from row 0, 1
          // Transpose 2x2
          const __m128 vres3 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v6), _mm_castps_pd(v7)));  // a b  from row 0, 1
          // Transpose 2x2
          const __m128 vres4 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v8), _mm_castps_pd(v9)));  // a b  from row 0, 1
          // Transpose 2x2
          const __m128 vres5 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v10), _mm_castps_pd(v11)));  // a b  from row 0, 1
          // Transpose 2x2
          const __m128 vres6 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v12), _mm_castps_pd(v13)));  // a b  from row 0, 1
          // Transpose 2x2
          const __m128 vres7 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v14), _mm_castps_pd(v15)));  // a b  from row 0, 1
          // Transpose 2x2
          const __m128 vres8 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v16), _mm_castps_pd(v17)));  // a b  from row 0, 1
          // Transpose 2x2
          const __m128 vres9 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v18), _mm_castps_pd(v19)));  // a b  from row 0, 1
          // Transpose 2x2
          const __m128 vres10 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v20), _mm_castps_pd(v21)));  // a b  from row 0, 1
          // Transpose 2x2
          const __m128 vres11 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v22), _mm_castps_pd(v23)));  // a b  from row 0, 1
          // Transpose 2x2
          const __m128 vres12 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v24), _mm_castps_pd(v25)));  // a b  from row 0, 1
          // Transpose 2x2
          const __m128 vres13 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v26), _mm_castps_pd(v27)));  // a b  from row 0, 1
          // Transpose 2x2
          const __m128 vres14 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v28), _mm_castps_pd(v29)));  // a b  from row 0, 1
          // Transpose 2x2
          const __m128 vres15 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v30), _mm_castps_pd(v31)));  // a b  from row 0, 1

          _mm_store_ps(packed_w, vres0);
          _mm_store_ps(packed_w + 4, vres1);
          _mm_store_ps(packed_w + 8, vres2);
          _mm_store_ps(packed_w + 12, vres3);
          _mm_store_ps(packed_w + 16, vres4);
          _mm_store_ps(packed_w + 20, vres5);
          _mm_store_ps(packed_w + 24, vres6);
          _mm_store_ps(packed_w + 28, vres7);
          _mm_store_ps(packed_w + 32, vres8);
          _mm_store_ps(packed_w + 36, vres9);
          _mm_store_ps(packed_w + 40, vres10);
          _mm_store_ps(packed_w + 44, vres11);
          _mm_store_ps(packed_w + 48, vres12);
          _mm_store_ps(packed_w + 52, vres13);
          _mm_store_ps(packed_w + 56, vres14);
          _mm_store_ps(packed_w + 60, vres15);
          packed_w += 32 * 2;
        }
      }
      packed_w = (float*) ((uintptr_t) packed_w + extra_bytes);
      w0 = w31;
    }

    // NC remainder (1..31)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 31);
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *packed_w++  = *b++;
        } while (--nb != 0);
        packed_w += (32 - n);
      } else {
        const __m512 vzero = _mm512_setzero_ps();
        _mm512_store_ps(packed_w, vzero);
        _mm512_store_ps(packed_w + 16, vzero);
        packed_w += 32;
      }

      // NR remainder has less than 32 rows so last row is not loaded
      // For SR=4 the
      const float* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const float* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const float* w3 = w2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const float* w4 = w3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const float* w5 = w4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const float* w6 = w5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const float* w7 = w6 + kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const float* w8 = w7 + kc;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const float* w9 = w8 + kc;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const float* w10 = w9 + kc;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const float* w11 = w10 + kc;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const float* w12 = w11 + kc;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const float* w13 = w12 + kc;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const float* w14 = w13 + kc;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }
      const float* w15 = w14 + kc;
      if XNN_UNPREDICTABLE(n < 16) {
        w15 = w14;
      }
      const float* w16 = w15 + kc;
      if XNN_UNPREDICTABLE(n <= 16) {
        w16 = w15;
      }
      const float* w17 = w16 + kc;
      if XNN_UNPREDICTABLE(n < 18) {
        w17 = w16;
      }
      const float* w18 = w17 + kc;
      if XNN_UNPREDICTABLE(n <= 18) {
        w18 = w17;
      }
      const float* w19 = w18 + kc;
      if XNN_UNPREDICTABLE(n < 20) {
        w19 = w18;
      }
      const float* w20 = w19 + kc;
      if XNN_UNPREDICTABLE(n <= 20) {
        w20 = w19;
      }
      const float* w21 = w20 + kc;
      if XNN_UNPREDICTABLE(n < 22) {
        w21 = w20;
      }
      const float* w22 = w21 + kc;
      if XNN_UNPREDICTABLE(n <= 22) {
        w22 = w21;
      }
      const float* w23 = w22 + kc;
      if XNN_UNPREDICTABLE(n < 24) {
        w23 = w22;
      }
      const float* w24 = w23 + kc;
      if XNN_UNPREDICTABLE(n <= 24) {
        w24 = w23;
      }
      const float* w25 = w24 + kc;
      if XNN_UNPREDICTABLE(n < 26) {
        w25 = w24;
      }
      const float* w26 = w25 + kc;
      if XNN_UNPREDICTABLE(n <= 26) {
        w26 = w25;
      }
      const float* w27 = w26 + kc;
      if XNN_UNPREDICTABLE(n < 28) {
        w27 = w26;
      }
      const float* w28 = w27 + kc;
      if XNN_UNPREDICTABLE(n <= 28) {
        w28 = w27;
      }
      const float* w29 = w28 + kc;
      if XNN_UNPREDICTABLE(n < 30) {
        w29 = w28;
      }
      const float* w30 = w29 + kc;
      if XNN_UNPREDICTABLE(n <= 30) {
        w30 = w29;
      }

      // KC main loop multiple of 4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        // Read blocks of 4x4
        // Load first 4 rows of N into low part of each register
        __m512 v0x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w0));
        w0 += 4;
        __m512 v1x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w1));
        w1 += 4;
        __m512 v8x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w8));
        w8 += 4;
        __m512 v9x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w9));
        w9 += 4;
        __m512 v16x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w16));
        w16 += 4;
        __m512 v17x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w17));
        w17 += 4;
        __m512 v24x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w24));
        w24 += 4;
        __m512 v25x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w25));
        w25 += 4;
        // Load next 4 rows of N into the high part of each register
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w2), 1);
        w2 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w3), 1);
        w3 += 4;
        v8x0123 = _mm512_insertf32x4(v8x0123, _mm_loadu_ps(w10), 1);
        w10 += 4;
        v9x0123 = _mm512_insertf32x4(v9x0123, _mm_loadu_ps(w11), 1);
        w11 += 4;
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w4), 2);
        w4 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w5), 2);
        w5 += 4;
        v8x0123 = _mm512_insertf32x4(v8x0123, _mm_loadu_ps(w12), 2);
        w12 += 4;
        v9x0123 = _mm512_insertf32x4(v9x0123, _mm_loadu_ps(w13), 2);
        w13 += 4;
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w6), 3);
        w6 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w7), 3);
        w7 += 4;
        v8x0123 = _mm512_insertf32x4(v8x0123, _mm_loadu_ps(w14), 3);
        w14 += 4;
        v9x0123 = _mm512_insertf32x4(v9x0123, _mm_loadu_ps(w15), 3);
        w15 += 4;
        v16x0123 = _mm512_insertf32x4(v16x0123, _mm_loadu_ps(w18), 1);
        w18 += 4;
        v17x0123 = _mm512_insertf32x4(v17x0123, _mm_loadu_ps(w19), 1);
        w19 += 4;
        v24x0123 = _mm512_insertf32x4(v24x0123, _mm_loadu_ps(w26), 1);
        w26 += 4;
        v25x0123 = _mm512_insertf32x4(v25x0123, _mm_loadu_ps(w27), 1);
        w27 += 4;
        v16x0123 = _mm512_insertf32x4(v16x0123, _mm_loadu_ps(w20), 2);
        w20 += 4;
        v17x0123 = _mm512_insertf32x4(v17x0123, _mm_loadu_ps(w21), 2);
        w21 += 4;
        v24x0123 = _mm512_insertf32x4(v24x0123, _mm_loadu_ps(w28), 2);
        w28 += 4;
        v25x0123 = _mm512_insertf32x4(v25x0123, _mm_loadu_ps(w29), 2);
        w29 += 4;
        v16x0123 = _mm512_insertf32x4(v16x0123, _mm_loadu_ps(w22), 3);
        w22 += 4;
        v17x0123 = _mm512_insertf32x4(v17x0123, _mm_loadu_ps(w23), 3);
        w23 += 4;
        v24x0123 = _mm512_insertf32x4(v24x0123, _mm_loadu_ps(w30), 3);
        w30 += 4;

        // Transpose 2x2
        const __m512 vres0x0123 =  _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(v0x0123), _mm512_castps_pd(v1x0123)));
        const __m512 vres16x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(v8x0123), _mm512_castps_pd(v9x0123)));
        const __m512 vres64x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(v0x0123), _mm512_castps_pd(v1x0123)));
        const __m512 vres80x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(v8x0123), _mm512_castps_pd(v9x0123)));
        const __m512 vres32x0123 =  _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(v16x0123), _mm512_castps_pd(v17x0123)));
        const __m512 vres48x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(v24x0123), _mm512_castps_pd(v25x0123)));
        const __m512 vres96x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(v16x0123), _mm512_castps_pd(v17x0123)));
        const __m512 vres112x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(v24x0123), _mm512_castps_pd(v25x0123)));
        xnn_prefetch_to_l1((const int8_t*) w0 + 128);
        xnn_prefetch_to_l1((const int8_t*) w1 + 128);
        xnn_prefetch_to_l1((const int8_t*) w2 + 128);
        xnn_prefetch_to_l1((const int8_t*) w3 + 128);
        xnn_prefetch_to_l1((const int8_t*) w4 + 128);
        xnn_prefetch_to_l1((const int8_t*) w5 + 128);
        xnn_prefetch_to_l1((const int8_t*) w6 + 128);
        xnn_prefetch_to_l1((const int8_t*) w7 + 128);
        xnn_prefetch_to_l1((const int8_t*) w8 + 128);
        xnn_prefetch_to_l1((const int8_t*) w9 + 128);
        xnn_prefetch_to_l1((const int8_t*) w10 + 128);
        xnn_prefetch_to_l1((const int8_t*) w11 + 128);
        xnn_prefetch_to_l1((const int8_t*) w12 + 128);
        xnn_prefetch_to_l1((const int8_t*) w13 + 128);
        xnn_prefetch_to_l1((const int8_t*) w14 + 128);
        xnn_prefetch_to_l1((const int8_t*) w15 + 128);
        xnn_prefetch_to_l1((const int8_t*) w16 + 128);
        xnn_prefetch_to_l1((const int8_t*) w17 + 128);
        xnn_prefetch_to_l1((const int8_t*) w18 + 128);
        xnn_prefetch_to_l1((const int8_t*) w19 + 128);
        xnn_prefetch_to_l1((const int8_t*) w20 + 128);
        xnn_prefetch_to_l1((const int8_t*) w21 + 128);
        xnn_prefetch_to_l1((const int8_t*) w22 + 128);
        xnn_prefetch_to_l1((const int8_t*) w23 + 128);
        xnn_prefetch_to_l1((const int8_t*) w24 + 128);
        xnn_prefetch_to_l1((const int8_t*) w25 + 128);
        xnn_prefetch_to_l1((const int8_t*) w26 + 128);
        xnn_prefetch_to_l1((const int8_t*) w27 + 128);
        xnn_prefetch_to_l1((const int8_t*) w28 + 128);
        xnn_prefetch_to_l1((const int8_t*) w29 + 128);
        xnn_prefetch_to_l1((const int8_t*) w30 + 128);

        _mm512_store_ps(packed_w, vres0x0123);
        _mm512_store_ps(packed_w + 16, vres16x0123);
        _mm512_store_ps(packed_w + 32, vres32x0123);
        _mm512_store_ps(packed_w + 48, vres48x0123);
        _mm512_store_ps(packed_w + 64, vres64x0123);
        _mm512_store_ps(packed_w + 80, vres80x0123);
        _mm512_store_ps(packed_w + 96, vres96x0123);
        _mm512_store_ps(packed_w + 112, vres112x0123);
        packed_w += 128;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        if (k & 2) {
          // Read blocks of 4x2
          __m128 v0 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w0));
          w0 += 2;
          __m128 v1 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w1));
          w1 += 2;
          __m128 v2 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w2));
          w2 += 2;
          __m128 v3 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w3));
          w3 += 2;
          __m128 v4 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w4));
          w4 += 2;
          __m128 v5 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w5));
          w5 += 2;
          __m128 v6 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w6));
          w6 += 2;
          __m128 v7 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w7));
          w7 += 2;
          __m128 v8 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w8));
          w8 += 2;
          __m128 v9 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w9));
          w9 += 2;
          __m128 v10 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w10));
          w10 += 2;
          __m128 v11 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w11));
          w11 += 2;
          __m128 v12 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w12));
          w12 += 2;
          __m128 v13 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w13));
          w13 += 2;
          __m128 v14 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w14));
          w14 += 2;
          __m128 v15 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w15));
          w15 += 2;
          __m128 v16 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w16));
          w16 += 2;
          __m128 v17 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w17));
          w17 += 2;
          __m128 v18 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w18));
          w18 += 2;
          __m128 v19 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w19));
          w19 += 2;
          __m128 v20 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w20));
          w20 += 2;
          __m128 v21 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w21));
          w21 += 2;
          __m128 v22 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w22));
          w22 += 2;
          __m128 v23 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w23));
          w23 += 2;
          __m128 v24 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w24));
          w24 += 2;
          __m128 v25 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w25));
          w25 += 2;
          __m128 v26 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w26));
          w26 += 2;
          __m128 v27 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w27));
          w27 += 2;
          __m128 v28 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w28));
          w28 += 2;
          __m128 v29 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w29));
          w29 += 2;
          __m128 v30 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w30));
          w30 += 2;

          __m128 v31 = _mm_setzero_ps();
          // Transpose 2x2
          const __m128 vres0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v0), _mm_castps_pd(v1)));
          const __m128 vres1 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v2), _mm_castps_pd(v3)));
          // Transpose 2x2
          const __m128 vres2 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v4), _mm_castps_pd(v5)));
          const __m128 vres3 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v6), _mm_castps_pd(v7)));
          // Transpose 2x2
          const __m128 vres4 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v8), _mm_castps_pd(v9)));
          const __m128 vres5 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v10), _mm_castps_pd(v11)));
          // Transpose 2x2
          const __m128 vres6 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v12), _mm_castps_pd(v13)));
          const __m128 vres7 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v14), _mm_castps_pd(v15)));
          // Transpose 2x2
          const __m128 vres8 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v16), _mm_castps_pd(v17)));
          const __m128 vres9 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v18), _mm_castps_pd(v19)));
          // Transpose 2x2
          const __m128 vres10 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v20), _mm_castps_pd(v21)));
          const __m128 vres11 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v22), _mm_castps_pd(v23)));
          // Transpose 2x2
          const __m128 vres12 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v24), _mm_castps_pd(v25)));
          const __m128 vres13 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v26), _mm_castps_pd(v27)));
          // Transpose 2x2
          const __m128 vres14 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v28), _mm_castps_pd(v29)));
          const __m128 vres15 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v30), _mm_castps_pd(v31)));

          _mm_store_ps(packed_w, vres0);
          _mm_store_ps(packed_w + 4, vres1);
          _mm_store_ps(packed_w + 8, vres2);
          _mm_store_ps(packed_w + 12, vres3);
          _mm_store_ps(packed_w + 16, vres4);
          _mm_store_ps(packed_w + 20, vres5);
          _mm_store_ps(packed_w + 24, vres6);
          _mm_store_ps(packed_w + 28, vres7);
          _mm_store_ps(packed_w + 32, vres8);
          _mm_store_ps(packed_w + 36, vres9);
          _mm_store_ps(packed_w + 40, vres10);
          _mm_store_ps(packed_w + 44, vres11);
          _mm_store_ps(packed_w + 48, vres12);
          _mm_store_ps(packed_w + 52, vres13);
          _mm_store_ps(packed_w + 56, vres14);
          _mm_store_ps(packed_w + 60, vres15);
          packed_w += 64;
        }
        if (k & 1) {
          // Read blocks of 4x1
          __m128 v0 = _mm_load_ss(w0);
          w0 += 1;
          __m128 v1 = _mm_load_ss(w1);
          w1 += 1;
          __m128 v2 = _mm_load_ss(w2);
          w2 += 1;
          __m128 v3 = _mm_load_ss(w3);
          w3 += 1;
          __m128 v4 = _mm_load_ss(w4);
          w4 += 1;
          __m128 v5 = _mm_load_ss(w5);
          w5 += 1;
          __m128 v6 = _mm_load_ss(w6);
          w6 += 1;
          __m128 v7 = _mm_load_ss(w7);
          w7 += 1;
          __m128 v8 = _mm_load_ss(w8);
          w8 += 1;
          __m128 v9 = _mm_load_ss(w9);
          w9 += 1;
          __m128 v10 = _mm_load_ss(w10);
          w10 += 1;
          __m128 v11 = _mm_load_ss(w11);
          w11 += 1;
          __m128 v12 = _mm_load_ss(w12);
          w12 += 1;
          __m128 v13 = _mm_load_ss(w13);
          w13 += 1;
          __m128 v14 = _mm_load_ss(w14);
          w14 += 1;
          __m128 v15 = _mm_load_ss(w15);
          w15 += 1;
          __m128 v16 = _mm_load_ss(w16);
          w16 += 1;
          __m128 v17 = _mm_load_ss(w17);
          w17 += 1;
          __m128 v18 = _mm_load_ss(w18);
          w18 += 1;
          __m128 v19 = _mm_load_ss(w19);
          w19 += 1;
          __m128 v20 = _mm_load_ss(w20);
          w20 += 1;
          __m128 v21 = _mm_load_ss(w21);
          w21 += 1;
          __m128 v22 = _mm_load_ss(w22);
          w22 += 1;
          __m128 v23 = _mm_load_ss(w23);
          w23 += 1;
          __m128 v24 = _mm_load_ss(w24);
          w24 += 1;
          __m128 v25 = _mm_load_ss(w25);
          w25 += 1;
          __m128 v26 = _mm_load_ss(w26);
          w26 += 1;
          __m128 v27 = _mm_load_ss(w27);
          w27 += 1;
          __m128 v28 = _mm_load_ss(w28);
          w28 += 1;
          __m128 v29 = _mm_load_ss(w29);
          w29 += 1;
          __m128 v30 = _mm_load_ss(w30);
          w30 += 1;

          __m128 v31 = _mm_setzero_ps();
          // Transpose 2x2
          const __m128 vres0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v0), _mm_castps_pd(v1)));
          // Transpose 2x2
          const __m128 vres1 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v2), _mm_castps_pd(v3)));
          // Transpose 2x2
          const __m128 vres2 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v4), _mm_castps_pd(v5)));
          // Transpose 2x2
          const __m128 vres3 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v6), _mm_castps_pd(v7)));
          // Transpose 2x2
          const __m128 vres4 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v8), _mm_castps_pd(v9)));
          // Transpose 2x2
          const __m128 vres5 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v10), _mm_castps_pd(v11)));
          // Transpose 2x2
          const __m128 vres6 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v12), _mm_castps_pd(v13)));
          // Transpose 2x2
          const __m128 vres7 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v14), _mm_castps_pd(v15)));
          // Transpose 2x2
          const __m128 vres8 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v16), _mm_castps_pd(v17)));
          // Transpose 2x2
          const __m128 vres9 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v18), _mm_castps_pd(v19)));
          // Transpose 2x2
          const __m128 vres10 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v20), _mm_castps_pd(v21)));
          // Transpose 2x2
          const __m128 vres11 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v22), _mm_castps_pd(v23)));
          // Transpose 2x2
          const __m128 vres12 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v24), _mm_castps_pd(v25)));
          // Transpose 2x2
          const __m128 vres13 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v26), _mm_castps_pd(v27)));
          // Transpose 2x2
          const __m128 vres14 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v28), _mm_castps_pd(v29)));
          // Transpose 2x2
          const __m128 vres15 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(v30), _mm_castps_pd(v31)));

          _mm_store_ps(packed_w, vres0);
          _mm_store_ps(packed_w + 4, vres1);
          _mm_store_ps(packed_w + 8, vres2);
          _mm_store_ps(packed_w + 12, vres3);
          _mm_store_ps(packed_w + 16, vres4);
          _mm_store_ps(packed_w + 20, vres5);
          _mm_store_ps(packed_w + 24, vres6);
          _mm_store_ps(packed_w + 28, vres7);
          _mm_store_ps(packed_w + 32, vres8);
          _mm_store_ps(packed_w + 36, vres9);
          _mm_store_ps(packed_w + 40, vres10);
          _mm_store_ps(packed_w + 44, vres11);
          _mm_store_ps(packed_w + 48, vres12);
          _mm_store_ps(packed_w + 52, vres13);
          _mm_store_ps(packed_w + 56, vres14);
          _mm_store_ps(packed_w + 60, vres15);
          packed_w += 32 * 2;
        }
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}
