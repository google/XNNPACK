// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x32-packw/avx512.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/packw.h"


void xnn_x32_packw_gemm_goi_ukernel_x32__avx512f_u4(
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
  assert(nr == 32);   // This kernel is for NR=32
  assert(kr == 1);
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

      // KC main loop multiple of 4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        // Read blocks of 4x4
        // a b c d
        // e f g h
        // i j k l
        // m n o p
        // Load first 4 rows of N into low part of each register
        __m512 v0x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w0));
        w0 += 4;
        __m512 v1x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w1));
        w1 += 4;
        __m512 v2x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w2));
        w2 += 4;
        __m512 v3x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w3));
        w3 += 4;
        __m512 v16x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w16));
        w16 += 4;
        __m512 v17x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w17));
        w17 += 4;
        __m512 v18x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w18));
        w18 += 4;
        __m512 v19x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w19));
        w19 += 4;
        // Load next 4 rows of N into the high part of each register
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w4), 1);
        w4 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w5), 1);
        w5 += 4;
        v2x0123 = _mm512_insertf32x4(v2x0123, _mm_loadu_ps(w6), 1);
        w6 += 4;
        v3x0123 = _mm512_insertf32x4(v3x0123, _mm_loadu_ps(w7), 1);
        w7 += 4;
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w8), 2);
        w8 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w9), 2);
        w9 += 4;
        v2x0123 = _mm512_insertf32x4(v2x0123, _mm_loadu_ps(w10), 2);
        w10 += 4;
        v3x0123 = _mm512_insertf32x4(v3x0123, _mm_loadu_ps(w11), 2);
        w11 += 4;
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w12), 3);
        w12 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w13), 3);
        w13 += 4;
        v2x0123 = _mm512_insertf32x4(v2x0123, _mm_loadu_ps(w14), 3);
        w14 += 4;
        v3x0123 = _mm512_insertf32x4(v3x0123, _mm_loadu_ps(w15), 3);
        w15 += 4;
        v16x0123 = _mm512_insertf32x4(v16x0123, _mm_loadu_ps(w20), 1);
        w20 += 4;
        v17x0123 = _mm512_insertf32x4(v17x0123, _mm_loadu_ps(w21), 1);
        w21 += 4;
        v18x0123 = _mm512_insertf32x4(v18x0123, _mm_loadu_ps(w22), 1);
        w22 += 4;
        v19x0123 = _mm512_insertf32x4(v19x0123, _mm_loadu_ps(w23), 1);
        w23 += 4;
        v16x0123 = _mm512_insertf32x4(v16x0123, _mm_loadu_ps(w24), 2);
        w24 += 4;
        v17x0123 = _mm512_insertf32x4(v17x0123, _mm_loadu_ps(w25), 2);
        w25 += 4;
        v18x0123 = _mm512_insertf32x4(v18x0123, _mm_loadu_ps(w26), 2);
        w26 += 4;
        v19x0123 = _mm512_insertf32x4(v19x0123, _mm_loadu_ps(w27), 2);
        w27 += 4;
        v16x0123 = _mm512_insertf32x4(v16x0123, _mm_loadu_ps(w28), 3);
        w28 += 4;
        v17x0123 = _mm512_insertf32x4(v17x0123, _mm_loadu_ps(w29), 3);
        w29 += 4;
        v18x0123 = _mm512_insertf32x4(v18x0123, _mm_loadu_ps(w30), 3);
        w30 += 4;
        v19x0123 = _mm512_insertf32x4(v19x0123, _mm_loadu_ps(w31), 3);
        w31 += 4;

        // Transpose 2x2
        const __m512 vtmp0x0123 = _mm512_unpacklo_ps(v0x0123, v1x0123);  // a e b f   from row 0, 1
        const __m512 vtmp1x0123 = _mm512_unpacklo_ps(v2x0123, v3x0123);  // i m j n   from row 2, 3
        const __m512 vtmp2x0123 = _mm512_unpackhi_ps(v0x0123, v1x0123);  // c g d h   from row 0, 1
        const __m512 vtmp3x0123 = _mm512_unpackhi_ps(v2x0123, v3x0123);  // k o l p   from row 2, 3
        const __m512 vtmp16x0123 = _mm512_unpacklo_ps(v16x0123, v17x0123);  // a e b f   from row 0, 1
        const __m512 vtmp17x0123 = _mm512_unpacklo_ps(v18x0123, v19x0123);  // i m j n   from row 2, 3
        const __m512 vtmp18x0123 = _mm512_unpackhi_ps(v16x0123, v17x0123);  // c g d h   from row 0, 1
        const __m512 vtmp19x0123 = _mm512_unpackhi_ps(v18x0123, v19x0123);  // k o l p   from row 2, 3
         // Transpose 4x4
        v0x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(vtmp0x0123), _mm512_castps_pd(vtmp1x0123)));  // a e i m   from row 0, 1
        v1x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(vtmp0x0123), _mm512_castps_pd(vtmp1x0123)));  // b f j n   from row 0, 1
        v2x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(vtmp2x0123), _mm512_castps_pd(vtmp3x0123)));  // c g k o   from row 2, 3
        v3x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(vtmp2x0123), _mm512_castps_pd(vtmp3x0123)));  // d h l p   from row 2, 3
        v16x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(vtmp16x0123), _mm512_castps_pd(vtmp17x0123)));  // a e i m   from row 0, 1
        v17x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(vtmp16x0123), _mm512_castps_pd(vtmp17x0123)));  // b f j n   from row 0, 1
        v18x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(vtmp18x0123), _mm512_castps_pd(vtmp19x0123)));  // c g k o   from row 2, 3
        v19x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(vtmp18x0123), _mm512_castps_pd(vtmp19x0123)));  // d h l p   from row 2, 3

        _mm512_store_ps(packed_w, v0x0123);
        _mm512_store_ps(packed_w + 16, v16x0123);
        _mm512_store_ps(packed_w + 32, v1x0123);
        _mm512_store_ps(packed_w + 48, v17x0123);
        _mm512_store_ps(packed_w + 64, v2x0123);
        _mm512_store_ps(packed_w + 80, v18x0123);
        _mm512_store_ps(packed_w + 96, v3x0123);
        _mm512_store_ps(packed_w + 112, v19x0123);
        packed_w += 128;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        if (k & 2) {
          // Read blocks of 4x2
          // a b
          // c d
          // e f
          // g h
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
          const __m128 vtmp0 = _mm_unpacklo_ps(v0, v1);  // a c b d   from row 0, 1
          const __m128 vtmp1 = _mm_unpacklo_ps(v2, v3);  // e g f h   from row 2, 3
          // Transpose 4x4
          v0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp0), _mm_castps_pd(vtmp1)));  // a c e g   from row 0, 1
          v1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp0), _mm_castps_pd(vtmp1)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp4 = _mm_unpacklo_ps(v4, v5);  // a c b d   from row 0, 1
          const __m128 vtmp5 = _mm_unpacklo_ps(v6, v7);  // e g f h   from row 2, 3
          // Transpose 4x4
          v4 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp4), _mm_castps_pd(vtmp5)));  // a c e g   from row 0, 1
          v5 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp4), _mm_castps_pd(vtmp5)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp8 = _mm_unpacklo_ps(v8, v9);  // a c b d   from row 0, 1
          const __m128 vtmp9 = _mm_unpacklo_ps(v10, v11);  // e g f h   from row 2, 3
          // Transpose 4x4
          v8 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp8), _mm_castps_pd(vtmp9)));  // a c e g   from row 0, 1
          v9 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp8), _mm_castps_pd(vtmp9)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp12 = _mm_unpacklo_ps(v12, v13);  // a c b d   from row 0, 1
          const __m128 vtmp13 = _mm_unpacklo_ps(v14, v15);  // e g f h   from row 2, 3
          // Transpose 4x4
          v12 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp12), _mm_castps_pd(vtmp13)));  // a c e g   from row 0, 1
          v13 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp12), _mm_castps_pd(vtmp13)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp16 = _mm_unpacklo_ps(v16, v17);  // a c b d   from row 0, 1
          const __m128 vtmp17 = _mm_unpacklo_ps(v18, v19);  // e g f h   from row 2, 3
          // Transpose 4x4
          v16 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp16), _mm_castps_pd(vtmp17)));  // a c e g   from row 0, 1
          v17 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp16), _mm_castps_pd(vtmp17)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp20 = _mm_unpacklo_ps(v20, v21);  // a c b d   from row 0, 1
          const __m128 vtmp21 = _mm_unpacklo_ps(v22, v23);  // e g f h   from row 2, 3
          // Transpose 4x4
          v20 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp20), _mm_castps_pd(vtmp21)));  // a c e g   from row 0, 1
          v21 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp20), _mm_castps_pd(vtmp21)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp24 = _mm_unpacklo_ps(v24, v25);  // a c b d   from row 0, 1
          const __m128 vtmp25 = _mm_unpacklo_ps(v26, v27);  // e g f h   from row 2, 3
          // Transpose 4x4
          v24 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp24), _mm_castps_pd(vtmp25)));  // a c e g   from row 0, 1
          v25 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp24), _mm_castps_pd(vtmp25)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp28 = _mm_unpacklo_ps(v28, v29);  // a c b d   from row 0, 1
          const __m128 vtmp29 = _mm_unpacklo_ps(v30, v31);  // e g f h   from row 2, 3
          // Transpose 4x4
          v28 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp28), _mm_castps_pd(vtmp29)));  // a c e g   from row 0, 1
          v29 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp28), _mm_castps_pd(vtmp29)));  // b d f h   from row 0, 1

          _mm_store_ps(packed_w, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v8);
          _mm_store_ps(packed_w + 12, v12);
          _mm_store_ps(packed_w + 16, v16);
          _mm_store_ps(packed_w + 20, v20);
          _mm_store_ps(packed_w + 24, v24);
          _mm_store_ps(packed_w + 28, v28);
          _mm_store_ps(packed_w + 32, v1);
          _mm_store_ps(packed_w + 36, v5);
          _mm_store_ps(packed_w + 40, v9);
          _mm_store_ps(packed_w + 44, v13);
          _mm_store_ps(packed_w + 48, v17);
          _mm_store_ps(packed_w + 52, v21);
          _mm_store_ps(packed_w + 56, v25);
          _mm_store_ps(packed_w + 60, v29);
          packed_w += 64;
        }
        if (k & 1) {
          // Read blocks of 4x1
          // a
          // b
          // c
          // d
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
          const __m128 vtmp0 = _mm_unpacklo_ps(v0, v1);  // a b  from row 0, 1
          const __m128 vtmp1 = _mm_unpacklo_ps(v2, v3);  // c d  from row 2, 3
          // Transpose 4x4
          v0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp0), _mm_castps_pd(vtmp1)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp4 = _mm_unpacklo_ps(v4, v5);  // a b  from row 0, 1
          const __m128 vtmp5 = _mm_unpacklo_ps(v6, v7);  // c d  from row 2, 3
          // Transpose 4x4
          v4 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp4), _mm_castps_pd(vtmp5)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp8 = _mm_unpacklo_ps(v8, v9);  // a b  from row 0, 1
          const __m128 vtmp9 = _mm_unpacklo_ps(v10, v11);  // c d  from row 2, 3
          // Transpose 4x4
          v8 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp8), _mm_castps_pd(vtmp9)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp12 = _mm_unpacklo_ps(v12, v13);  // a b  from row 0, 1
          const __m128 vtmp13 = _mm_unpacklo_ps(v14, v15);  // c d  from row 2, 3
          // Transpose 4x4
          v12 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp12), _mm_castps_pd(vtmp13)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp16 = _mm_unpacklo_ps(v16, v17);  // a b  from row 0, 1
          const __m128 vtmp17 = _mm_unpacklo_ps(v18, v19);  // c d  from row 2, 3
          // Transpose 4x4
          v16 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp16), _mm_castps_pd(vtmp17)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp20 = _mm_unpacklo_ps(v20, v21);  // a b  from row 0, 1
          const __m128 vtmp21 = _mm_unpacklo_ps(v22, v23);  // c d  from row 2, 3
          // Transpose 4x4
          v20 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp20), _mm_castps_pd(vtmp21)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp24 = _mm_unpacklo_ps(v24, v25);  // a b  from row 0, 1
          const __m128 vtmp25 = _mm_unpacklo_ps(v26, v27);  // c d  from row 2, 3
          // Transpose 4x4
          v24 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp24), _mm_castps_pd(vtmp25)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp28 = _mm_unpacklo_ps(v28, v29);  // a b  from row 0, 1
          const __m128 vtmp29 = _mm_unpacklo_ps(v30, v31);  // c d  from row 2, 3
          // Transpose 4x4
          v28 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp28), _mm_castps_pd(vtmp29)));  // a b c d   from row 0, 1

          _mm_store_ps(packed_w, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v8);
          _mm_store_ps(packed_w + 12, v12);
          _mm_store_ps(packed_w + 16, v16);
          _mm_store_ps(packed_w + 20, v20);
          _mm_store_ps(packed_w + 24, v24);
          _mm_store_ps(packed_w + 28, v28);
          packed_w += 32;
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
        // a b c d
        // e f g h
        // i j k l
        // m n o p
        // Load first 4 rows of N into low part of each register
        __m512 v0x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w0));
        w0 += 4;
        __m512 v1x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w1));
        w1 += 4;
        __m512 v2x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w2));
        w2 += 4;
        __m512 v3x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w3));
        w3 += 4;
        __m512 v16x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w16));
        w16 += 4;
        __m512 v17x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w17));
        w17 += 4;
        __m512 v18x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w18));
        w18 += 4;
        // castps leaves upper 128 bits undefined, so zero them.
        __m512 v19x0123 = _mm512_zextps128_ps512(_mm_loadu_ps(w19));
        w19 += 4;
        // Load next 4 rows of N into the high part of each register
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w4), 1);
        w4 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w5), 1);
        w5 += 4;
        v2x0123 = _mm512_insertf32x4(v2x0123, _mm_loadu_ps(w6), 1);
        w6 += 4;
        v3x0123 = _mm512_insertf32x4(v3x0123, _mm_loadu_ps(w7), 1);
        w7 += 4;
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w8), 2);
        w8 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w9), 2);
        w9 += 4;
        v2x0123 = _mm512_insertf32x4(v2x0123, _mm_loadu_ps(w10), 2);
        w10 += 4;
        v3x0123 = _mm512_insertf32x4(v3x0123, _mm_loadu_ps(w11), 2);
        w11 += 4;
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w12), 3);
        w12 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w13), 3);
        w13 += 4;
        v2x0123 = _mm512_insertf32x4(v2x0123, _mm_loadu_ps(w14), 3);
        w14 += 4;
        v3x0123 = _mm512_insertf32x4(v3x0123, _mm_loadu_ps(w15), 3);
        w15 += 4;
        v16x0123 = _mm512_insertf32x4(v16x0123, _mm_loadu_ps(w20), 1);
        w20 += 4;
        v17x0123 = _mm512_insertf32x4(v17x0123, _mm_loadu_ps(w21), 1);
        w21 += 4;
        v18x0123 = _mm512_insertf32x4(v18x0123, _mm_loadu_ps(w22), 1);
        w22 += 4;
        v19x0123 = _mm512_insertf32x4(v19x0123, _mm_loadu_ps(w23), 1);
        w23 += 4;
        v16x0123 = _mm512_insertf32x4(v16x0123, _mm_loadu_ps(w24), 2);
        w24 += 4;
        v17x0123 = _mm512_insertf32x4(v17x0123, _mm_loadu_ps(w25), 2);
        w25 += 4;
        v18x0123 = _mm512_insertf32x4(v18x0123, _mm_loadu_ps(w26), 2);
        w26 += 4;
        v19x0123 = _mm512_insertf32x4(v19x0123, _mm_loadu_ps(w27), 2);
        w27 += 4;
        v16x0123 = _mm512_insertf32x4(v16x0123, _mm_loadu_ps(w28), 3);
        w28 += 4;
        v17x0123 = _mm512_insertf32x4(v17x0123, _mm_loadu_ps(w29), 3);
        w29 += 4;
        v18x0123 = _mm512_insertf32x4(v18x0123, _mm_loadu_ps(w30), 3);
        w30 += 4;

        // Transpose 2x2
        const __m512 vtmp0x0123 = _mm512_unpacklo_ps(v0x0123, v1x0123);  // a e b f   from row 0, 1
        const __m512 vtmp1x0123 = _mm512_unpacklo_ps(v2x0123, v3x0123);  // i m j n   from row 2, 3
        const __m512 vtmp2x0123 = _mm512_unpackhi_ps(v0x0123, v1x0123);  // c g d h   from row 0, 1
        const __m512 vtmp3x0123 = _mm512_unpackhi_ps(v2x0123, v3x0123);  // k o l p   from row 2, 3
        const __m512 vtmp16x0123 = _mm512_unpacklo_ps(v16x0123, v17x0123);  // a e b f   from row 0, 1
        const __m512 vtmp17x0123 = _mm512_unpacklo_ps(v18x0123, v19x0123);  // i m j n   from row 2, 3
        const __m512 vtmp18x0123 = _mm512_unpackhi_ps(v16x0123, v17x0123);  // c g d h   from row 0, 1
        const __m512 vtmp19x0123 = _mm512_unpackhi_ps(v18x0123, v19x0123);  // k o l p   from row 2, 3
        // Transpose 4x4
        v0x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(vtmp0x0123), _mm512_castps_pd(vtmp1x0123)));  // a e i m   from row 0, 1
        v1x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(vtmp0x0123), _mm512_castps_pd(vtmp1x0123)));  // b f j n   from row 0, 1
        v2x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(vtmp2x0123), _mm512_castps_pd(vtmp3x0123)));  // c g k o   from row 2, 3
        v3x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(vtmp2x0123), _mm512_castps_pd(vtmp3x0123)));  // d h l p   from row 2, 3
        v16x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(vtmp16x0123), _mm512_castps_pd(vtmp17x0123)));  // a e i m   from row 0, 1
        v17x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(vtmp16x0123), _mm512_castps_pd(vtmp17x0123)));  // b f j n   from row 0, 1
        v18x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(vtmp18x0123), _mm512_castps_pd(vtmp19x0123)));  // c g k o   from row 2, 3
        v19x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(vtmp18x0123), _mm512_castps_pd(vtmp19x0123)));  // d h l p   from row 2, 3

        _mm512_store_ps(packed_w, v0x0123);
        _mm512_store_ps(packed_w + 16, v16x0123);
        _mm512_store_ps(packed_w + 32, v1x0123);
        _mm512_store_ps(packed_w + 48, v17x0123);
        _mm512_store_ps(packed_w + 64, v2x0123);
        _mm512_store_ps(packed_w + 80, v18x0123);
        _mm512_store_ps(packed_w + 96, v3x0123);
        _mm512_store_ps(packed_w + 112, v19x0123);
        packed_w += 128;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        if (k & 2) {
          // Read blocks of 4x2
          // a b
          // c d
          // e f
          // g h
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

          // Transpose 2x2
          const __m128 vtmp0 = _mm_unpacklo_ps(v0, v1);  // a c b d   from row 0, 1
          const __m128 vtmp1 = _mm_unpacklo_ps(v2, v3);  // e g f h   from row 2, 3
          // Transpose 4x4
          v0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp0), _mm_castps_pd(vtmp1)));  // a c e g   from row 0, 1
          v1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp0), _mm_castps_pd(vtmp1)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp4 = _mm_unpacklo_ps(v4, v5);  // a c b d   from row 0, 1
          const __m128 vtmp5 = _mm_unpacklo_ps(v6, v7);  // e g f h   from row 2, 3
          // Transpose 4x4
          v4 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp4), _mm_castps_pd(vtmp5)));  // a c e g   from row 0, 1
          v5 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp4), _mm_castps_pd(vtmp5)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp8 = _mm_unpacklo_ps(v8, v9);  // a c b d   from row 0, 1
          const __m128 vtmp9 = _mm_unpacklo_ps(v10, v11);  // e g f h   from row 2, 3
          // Transpose 4x4
          v8 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp8), _mm_castps_pd(vtmp9)));  // a c e g   from row 0, 1
          v9 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp8), _mm_castps_pd(vtmp9)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp12 = _mm_unpacklo_ps(v12, v13);  // a c b d   from row 0, 1
          const __m128 vtmp13 = _mm_unpacklo_ps(v14, v15);  // e g f h   from row 2, 3
          // Transpose 4x4
          v12 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp12), _mm_castps_pd(vtmp13)));  // a c e g   from row 0, 1
          v13 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp12), _mm_castps_pd(vtmp13)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp16 = _mm_unpacklo_ps(v16, v17);  // a c b d   from row 0, 1
          const __m128 vtmp17 = _mm_unpacklo_ps(v18, v19);  // e g f h   from row 2, 3
          // Transpose 4x4
          v16 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp16), _mm_castps_pd(vtmp17)));  // a c e g   from row 0, 1
          v17 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp16), _mm_castps_pd(vtmp17)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp20 = _mm_unpacklo_ps(v20, v21);  // a c b d   from row 0, 1
          const __m128 vtmp21 = _mm_unpacklo_ps(v22, v23);  // e g f h   from row 2, 3
          // Transpose 4x4
          v20 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp20), _mm_castps_pd(vtmp21)));  // a c e g   from row 0, 1
          v21 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp20), _mm_castps_pd(vtmp21)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp24 = _mm_unpacklo_ps(v24, v25);  // a c b d   from row 0, 1
          const __m128 vtmp25 = _mm_unpacklo_ps(v26, v27);  // e g f h   from row 2, 3
          // Transpose 4x4
          v24 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp24), _mm_castps_pd(vtmp25)));  // a c e g   from row 0, 1
          v25 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp24), _mm_castps_pd(vtmp25)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp28 = _mm_unpacklo_ps(v28, v29);  // a c b d   from row 0, 1
          const __m128 vtmp29 = _mm_unpacklo_ps(v30, v30);  // e g f h   from row 2, 3
          // Transpose 4x4
          v28 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp28), _mm_castps_pd(vtmp29)));  // a c e g   from row 0, 1
          v29 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp28), _mm_castps_pd(vtmp29)));  // b d f h   from row 0, 1

          _mm_store_ps(packed_w, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v8);
          _mm_store_ps(packed_w + 12, v12);
          _mm_store_ps(packed_w + 16, v16);
          _mm_store_ps(packed_w + 20, v20);
          _mm_store_ps(packed_w + 24, v24);
          _mm_store_ps(packed_w + 28, v28);
          _mm_store_ps(packed_w + 32, v1);
          _mm_store_ps(packed_w + 36, v5);
          _mm_store_ps(packed_w + 40, v9);
          _mm_store_ps(packed_w + 44, v13);
          _mm_store_ps(packed_w + 48, v17);
          _mm_store_ps(packed_w + 52, v21);
          _mm_store_ps(packed_w + 56, v25);
          _mm_store_ps(packed_w + 60, v29);
          packed_w += 64;
        }
        if (k & 1) {
          // Read blocks of 4x1
          // a
          // b
          // c
          // d
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

          // Transpose 2x2
          const __m128 vtmp0 = _mm_unpacklo_ps(v0, v1);  // a b  from row 0, 1
          const __m128 vtmp1 = _mm_unpacklo_ps(v2, v3);  // c d  from row 2, 3
          // Transpose 4x4
          v0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp0), _mm_castps_pd(vtmp1)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp4 = _mm_unpacklo_ps(v4, v5);  // a b  from row 0, 1
          const __m128 vtmp5 = _mm_unpacklo_ps(v6, v7);  // c d  from row 2, 3
          // Transpose 4x4
          v4 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp4), _mm_castps_pd(vtmp5)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp8 = _mm_unpacklo_ps(v8, v9);  // a b  from row 0, 1
          const __m128 vtmp9 = _mm_unpacklo_ps(v10, v11);  // c d  from row 2, 3
          // Transpose 4x4
          v8 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp8), _mm_castps_pd(vtmp9)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp12 = _mm_unpacklo_ps(v12, v13);  // a b  from row 0, 1
          const __m128 vtmp13 = _mm_unpacklo_ps(v14, v15);  // c d  from row 2, 3
          // Transpose 4x4
          v12 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp12), _mm_castps_pd(vtmp13)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp16 = _mm_unpacklo_ps(v16, v17);  // a b  from row 0, 1
          const __m128 vtmp17 = _mm_unpacklo_ps(v18, v19);  // c d  from row 2, 3
          // Transpose 4x4
          v16 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp16), _mm_castps_pd(vtmp17)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp20 = _mm_unpacklo_ps(v20, v21);  // a b  from row 0, 1
          const __m128 vtmp21 = _mm_unpacklo_ps(v22, v23);  // c d  from row 2, 3
          // Transpose 4x4
          v20 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp20), _mm_castps_pd(vtmp21)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp24 = _mm_unpacklo_ps(v24, v25);  // a b  from row 0, 1
          const __m128 vtmp25 = _mm_unpacklo_ps(v26, v27);  // c d  from row 2, 3
          // Transpose 4x4
          v24 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp24), _mm_castps_pd(vtmp25)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp28 = _mm_unpacklo_ps(v28, v29);  // a b  from row 0, 1
          const __m128 vtmp29 = _mm_unpacklo_ps(v30, v30);  // c d  from row 2, 3
          // Transpose 4x4
          v28 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp28), _mm_castps_pd(vtmp29)));  // a b c d   from row 0, 1

          _mm_store_ps(packed_w, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v8);
          _mm_store_ps(packed_w + 12, v12);
          _mm_store_ps(packed_w + 16, v16);
          _mm_store_ps(packed_w + 20, v20);
          _mm_store_ps(packed_w + 24, v24);
          _mm_store_ps(packed_w + 28, v28);
          packed_w += 32;
        }
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}
