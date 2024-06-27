// Auto-generated file. Do not edit!
//   Template: src/x32-packw/s4-sse2.c.in
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

#include "xnnpack/packw.h"


void xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_u4(
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
  assert(nr == 16);
  assert(kr == 1);
  assert(sr == 4);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const float* b = (const float*) bias;
  float* packed_w = (float*) packed_weights;
  do {
    // NC main loop multiple of 16
    const float* w0 = (const float*) weights;
    size_t n = nc;

    for (; n >= 16; n -= 16) {
      if XNN_LIKELY(b != NULL) {
        const __m128 vb0 = _mm_loadu_ps(b);
        const __m128 vb4 = _mm_loadu_ps(b + 4);
        const __m128 vb8 = _mm_loadu_ps(b + 8);
        const __m128 vb12 = _mm_loadu_ps(b + 12);
        _mm_store_ps(packed_w, vb0);
        _mm_store_ps(packed_w + 4, vb4);
        _mm_store_ps(packed_w + 8, vb8);
        _mm_store_ps(packed_w + 12, vb12);
        b += 16;
      } else {
        const __m128 vzero = _mm_setzero_ps();
        _mm_store_ps(packed_w, vzero);
        _mm_store_ps(packed_w + 4, vzero);
        _mm_store_ps(packed_w + 8, vzero);
        _mm_store_ps(packed_w + 12, vzero);
      }
      packed_w += 16;

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

      size_t k = kc;

      // KC multiple of 4
      for (; k >= 4; k -= 4) {
        // Read blocks of 4x4
        // a b c d
        // e f g h
        // i j k l
        // m n o p
        __m128 v0x0123 = _mm_loadu_ps(w0);
        w0 += 4;
        __m128 v1x0123 = _mm_loadu_ps(w1);
        w1 += 4;
        __m128 v2x0123 = _mm_loadu_ps(w2);
        w2 += 4;
        __m128 v3x0123 = _mm_loadu_ps(w3);
        w3 += 4;
        __m128 v4x0123 = _mm_loadu_ps(w4);
        w4 += 4;
        __m128 v5x0123 = _mm_loadu_ps(w5);
        w5 += 4;
        __m128 v6x0123 = _mm_loadu_ps(w6);
        w6 += 4;
        __m128 v7x0123 = _mm_loadu_ps(w7);
        w7 += 4;
        __m128 v8x0123 = _mm_loadu_ps(w8);
        w8 += 4;
        __m128 v9x0123 = _mm_loadu_ps(w9);
        w9 += 4;
        __m128 v10x0123 = _mm_loadu_ps(w10);
        w10 += 4;
        __m128 v11x0123 = _mm_loadu_ps(w11);
        w11 += 4;
        __m128 v12x0123 = _mm_loadu_ps(w12);
        w12 += 4;
        __m128 v13x0123 = _mm_loadu_ps(w13);
        w13 += 4;
        __m128 v14x0123 = _mm_loadu_ps(w14);
        w14 += 4;
        __m128 v15x0123 = _mm_loadu_ps(w15);
        w15 += 4;

        // Apply SR4 shuffle
        v1x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v1x0123), _MM_SHUFFLE(0, 3, 2, 1)));
        v2x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v2x0123), _MM_SHUFFLE(1, 0, 3, 2)));
        v3x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v3x0123), _MM_SHUFFLE(2, 1, 0, 3)));
        v5x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v5x0123), _MM_SHUFFLE(0, 3, 2, 1)));
        v6x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v6x0123), _MM_SHUFFLE(1, 0, 3, 2)));
        v7x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v7x0123), _MM_SHUFFLE(2, 1, 0, 3)));
        v9x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v9x0123), _MM_SHUFFLE(0, 3, 2, 1)));
        v10x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v10x0123), _MM_SHUFFLE(1, 0, 3, 2)));
        v11x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v11x0123), _MM_SHUFFLE(2, 1, 0, 3)));
        v13x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v13x0123), _MM_SHUFFLE(0, 3, 2, 1)));
        v14x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v14x0123), _MM_SHUFFLE(1, 0, 3, 2)));
        v15x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v15x0123), _MM_SHUFFLE(2, 1, 0, 3)));

        // Transpose 2x2
        const __m128 vtmp0x0123 = _mm_unpacklo_ps(v0x0123, v1x0123);  // a e b f   from row 0, 1
        const __m128 vtmp1x0123 = _mm_unpacklo_ps(v2x0123, v3x0123);  // i m j n   from row 2, 3
        const __m128 vtmp2x0123 = _mm_unpackhi_ps(v0x0123, v1x0123);  // c g d h   from row 0, 1
        const __m128 vtmp3x0123 = _mm_unpackhi_ps(v2x0123, v3x0123);  // k o l p   from row 2, 3
        const __m128 vtmp4x0123 = _mm_unpacklo_ps(v4x0123, v5x0123);  // a e b f   from row 0, 1
        const __m128 vtmp5x0123 = _mm_unpacklo_ps(v6x0123, v7x0123);  // i m j n   from row 2, 3
        const __m128 vtmp6x0123 = _mm_unpackhi_ps(v4x0123, v5x0123);  // c g d h   from row 0, 1
        const __m128 vtmp7x0123 = _mm_unpackhi_ps(v6x0123, v7x0123);  // k o l p   from row 2, 3
        const __m128 vtmp8x0123 = _mm_unpacklo_ps(v8x0123, v9x0123);  // a e b f   from row 0, 1
        const __m128 vtmp9x0123 = _mm_unpacklo_ps(v10x0123, v11x0123);  // i m j n   from row 2, 3
        const __m128 vtmp10x0123 = _mm_unpackhi_ps(v8x0123, v9x0123);  // c g d h   from row 0, 1
        const __m128 vtmp11x0123 = _mm_unpackhi_ps(v10x0123, v11x0123);  // k o l p   from row 2, 3
        const __m128 vtmp12x0123 = _mm_unpacklo_ps(v12x0123, v13x0123);  // a e b f   from row 0, 1
        const __m128 vtmp13x0123 = _mm_unpacklo_ps(v14x0123, v15x0123);  // i m j n   from row 2, 3
        const __m128 vtmp14x0123 = _mm_unpackhi_ps(v12x0123, v13x0123);  // c g d h   from row 0, 1
        const __m128 vtmp15x0123 = _mm_unpackhi_ps(v14x0123, v15x0123);  // k o l p   from row 2, 3
        // Transpose 4x4
        v0x0123 = _mm_movelh_ps(vtmp0x0123, vtmp1x0123);  // a e i m   from row 0, 1
        v1x0123 = _mm_movehl_ps(vtmp1x0123, vtmp0x0123);  // b f j n   from row 0, 1
        v2x0123 = _mm_movelh_ps(vtmp2x0123, vtmp3x0123);  // c g k o   from row 2, 3
        v3x0123 = _mm_movehl_ps(vtmp3x0123, vtmp2x0123);  // d h l p   from row 2, 3
        v4x0123 = _mm_movelh_ps(vtmp4x0123, vtmp5x0123);  // a e i m   from row 0, 1
        v5x0123 = _mm_movehl_ps(vtmp5x0123, vtmp4x0123);  // b f j n   from row 0, 1
        v6x0123 = _mm_movelh_ps(vtmp6x0123, vtmp7x0123);  // c g k o   from row 2, 3
        v7x0123 = _mm_movehl_ps(vtmp7x0123, vtmp6x0123);  // d h l p   from row 2, 3
        v8x0123 = _mm_movelh_ps(vtmp8x0123, vtmp9x0123);  // a e i m   from row 0, 1
        v9x0123 = _mm_movehl_ps(vtmp9x0123, vtmp8x0123);  // b f j n   from row 0, 1
        v10x0123 = _mm_movelh_ps(vtmp10x0123, vtmp11x0123);  // c g k o   from row 2, 3
        v11x0123 = _mm_movehl_ps(vtmp11x0123, vtmp10x0123);  // d h l p   from row 2, 3
        v12x0123 = _mm_movelh_ps(vtmp12x0123, vtmp13x0123);  // a e i m   from row 0, 1
        v13x0123 = _mm_movehl_ps(vtmp13x0123, vtmp12x0123);  // b f j n   from row 0, 1
        v14x0123 = _mm_movelh_ps(vtmp14x0123, vtmp15x0123);  // c g k o   from row 2, 3
        v15x0123 = _mm_movehl_ps(vtmp15x0123, vtmp14x0123);  // d h l p   from row 2, 3

        _mm_store_ps(packed_w, v0x0123);
        _mm_store_ps(packed_w + 4, v4x0123);
        _mm_store_ps(packed_w + 8, v8x0123);
        _mm_store_ps(packed_w + 12, v12x0123);
        _mm_store_ps(packed_w + 16, v1x0123);
        _mm_store_ps(packed_w + 20, v5x0123);
        _mm_store_ps(packed_w + 24, v9x0123);
        _mm_store_ps(packed_w + 28, v13x0123);
        _mm_store_ps(packed_w + 32, v2x0123);
        _mm_store_ps(packed_w + 36, v6x0123);
        _mm_store_ps(packed_w + 40, v10x0123);
        _mm_store_ps(packed_w + 44, v14x0123);
        _mm_store_ps(packed_w + 48, v3x0123);
        _mm_store_ps(packed_w + 52, v7x0123);
        _mm_store_ps(packed_w + 56, v11x0123);
        _mm_store_ps(packed_w + 60, v15x0123);
        packed_w += 64;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        __m128 v0 = _mm_undefined_ps();
        __m128 v1 = _mm_undefined_ps();
        __m128 v2 = _mm_undefined_ps();
        __m128 v3 = _mm_undefined_ps();
        __m128 v4 = _mm_undefined_ps();
        __m128 v5 = _mm_undefined_ps();
        __m128 v6 = _mm_undefined_ps();
        __m128 v7 = _mm_undefined_ps();
        __m128 v8 = _mm_undefined_ps();
        __m128 v9 = _mm_undefined_ps();
        __m128 v10 = _mm_undefined_ps();
        __m128 v11 = _mm_undefined_ps();
        __m128 v12 = _mm_undefined_ps();
        __m128 v13 = _mm_undefined_ps();
        __m128 v14 = _mm_undefined_ps();
        __m128 v15 = _mm_undefined_ps();

        switch (k) {
          case 1:
            // Read blocks of 4x1
            // a
            // e
            // i
            // m
            v0 = _mm_load_ss(w0);
            w0 += 1;
            v1 = _mm_load_ss(w1);
            w1 += 1;
            v2 = _mm_load_ss(w2);
            w2 += 1;
            v3 = _mm_load_ss(w3);
            w3 += 1;
            v4 = _mm_load_ss(w4);
            w4 += 1;
            v5 = _mm_load_ss(w5);
            w5 += 1;
            v6 = _mm_load_ss(w6);
            w6 += 1;
            v7 = _mm_load_ss(w7);
            w7 += 1;
            v8 = _mm_load_ss(w8);
            w8 += 1;
            v9 = _mm_load_ss(w9);
            w9 += 1;
            v10 = _mm_load_ss(w10);
            w10 += 1;
            v11 = _mm_load_ss(w11);
            w11 += 1;
            v12 = _mm_load_ss(w12);
            w12 += 1;
            v13 = _mm_load_ss(w13);
            w13 += 1;
            v14 = _mm_load_ss(w14);
            w14 += 1;
            v15 = _mm_load_ss(w15);
            w15 += 1;
            break;
          case 2:
            // Read blocks of 4x2
            // a b
            // e f
            // i j
            // m n
            v0 = _mm_castpd_ps(_mm_load_sd((const double*) w0));
            w0 += 2;
            v1 = _mm_castpd_ps(_mm_load_sd((const double*) w1));
            w1 += 2;
            v2 = _mm_castpd_ps(_mm_load_sd((const double*) w2));
            w2 += 2;
            v3 = _mm_castpd_ps(_mm_load_sd((const double*) w3));
            w3 += 2;
            v4 = _mm_castpd_ps(_mm_load_sd((const double*) w4));
            w4 += 2;
            v5 = _mm_castpd_ps(_mm_load_sd((const double*) w5));
            w5 += 2;
            v6 = _mm_castpd_ps(_mm_load_sd((const double*) w6));
            w6 += 2;
            v7 = _mm_castpd_ps(_mm_load_sd((const double*) w7));
            w7 += 2;
            v8 = _mm_castpd_ps(_mm_load_sd((const double*) w8));
            w8 += 2;
            v9 = _mm_castpd_ps(_mm_load_sd((const double*) w9));
            w9 += 2;
            v10 = _mm_castpd_ps(_mm_load_sd((const double*) w10));
            w10 += 2;
            v11 = _mm_castpd_ps(_mm_load_sd((const double*) w11));
            w11 += 2;
            v12 = _mm_castpd_ps(_mm_load_sd((const double*) w12));
            w12 += 2;
            v13 = _mm_castpd_ps(_mm_load_sd((const double*) w13));
            w13 += 2;
            v14 = _mm_castpd_ps(_mm_load_sd((const double*) w14));
            w14 += 2;
            v15 = _mm_castpd_ps(_mm_load_sd((const double*) w15));
            w15 += 2;
            break;
          case 3:
          {
            // Read blocks of 4x3
            // a b c
            // e f g
            // i j k
            // m n o
            const __m128 v0lo = _mm_castpd_ps(_mm_load_sd((const double*) w0));
            const __m128 v0hi = _mm_load_ss(w0 + 2);
            v0 = _mm_movelh_ps(v0lo, v0hi);
            w0 += 3;
            const __m128 v1lo = _mm_castpd_ps(_mm_load_sd((const double*) w1));
            const __m128 v1hi = _mm_load_ss(w1 + 2);
            v1 = _mm_movelh_ps(v1lo, v1hi);
            w1 += 3;
            const __m128 v2lo = _mm_castpd_ps(_mm_load_sd((const double*) w2));
            const __m128 v2hi = _mm_load_ss(w2 + 2);
            v2 = _mm_movelh_ps(v2lo, v2hi);
            w2 += 3;
            const __m128 v3lo = _mm_castpd_ps(_mm_load_sd((const double*) w3));
            const __m128 v3hi = _mm_load_ss(w3 + 2);
            v3 = _mm_movelh_ps(v3lo, v3hi);
            w3 += 3;
            const __m128 v4lo = _mm_castpd_ps(_mm_load_sd((const double*) w4));
            const __m128 v4hi = _mm_load_ss(w4 + 2);
            v4 = _mm_movelh_ps(v4lo, v4hi);
            w4 += 3;
            const __m128 v5lo = _mm_castpd_ps(_mm_load_sd((const double*) w5));
            const __m128 v5hi = _mm_load_ss(w5 + 2);
            v5 = _mm_movelh_ps(v5lo, v5hi);
            w5 += 3;
            const __m128 v6lo = _mm_castpd_ps(_mm_load_sd((const double*) w6));
            const __m128 v6hi = _mm_load_ss(w6 + 2);
            v6 = _mm_movelh_ps(v6lo, v6hi);
            w6 += 3;
            const __m128 v7lo = _mm_castpd_ps(_mm_load_sd((const double*) w7));
            const __m128 v7hi = _mm_load_ss(w7 + 2);
            v7 = _mm_movelh_ps(v7lo, v7hi);
            w7 += 3;
            const __m128 v8lo = _mm_castpd_ps(_mm_load_sd((const double*) w8));
            const __m128 v8hi = _mm_load_ss(w8 + 2);
            v8 = _mm_movelh_ps(v8lo, v8hi);
            w8 += 3;
            const __m128 v9lo = _mm_castpd_ps(_mm_load_sd((const double*) w9));
            const __m128 v9hi = _mm_load_ss(w9 + 2);
            v9 = _mm_movelh_ps(v9lo, v9hi);
            w9 += 3;
            const __m128 v10lo = _mm_castpd_ps(_mm_load_sd((const double*) w10));
            const __m128 v10hi = _mm_load_ss(w10 + 2);
            v10 = _mm_movelh_ps(v10lo, v10hi);
            w10 += 3;
            const __m128 v11lo = _mm_castpd_ps(_mm_load_sd((const double*) w11));
            const __m128 v11hi = _mm_load_ss(w11 + 2);
            v11 = _mm_movelh_ps(v11lo, v11hi);
            w11 += 3;
            const __m128 v12lo = _mm_castpd_ps(_mm_load_sd((const double*) w12));
            const __m128 v12hi = _mm_load_ss(w12 + 2);
            v12 = _mm_movelh_ps(v12lo, v12hi);
            w12 += 3;
            const __m128 v13lo = _mm_castpd_ps(_mm_load_sd((const double*) w13));
            const __m128 v13hi = _mm_load_ss(w13 + 2);
            v13 = _mm_movelh_ps(v13lo, v13hi);
            w13 += 3;
            const __m128 v14lo = _mm_castpd_ps(_mm_load_sd((const double*) w14));
            const __m128 v14hi = _mm_load_ss(w14 + 2);
            v14 = _mm_movelh_ps(v14lo, v14hi);
            w14 += 3;
            const __m128 v15lo = _mm_castpd_ps(_mm_load_sd((const double*) w15));
            const __m128 v15hi = _mm_load_ss(w15 + 2);
            v15 = _mm_movelh_ps(v15lo, v15hi);
            w15 += 3;
            break;
          }
          default:
            XNN_UNREACHABLE;
        }

        // Apply SR4 shuffle
        v1 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v1), _MM_SHUFFLE(0, 3, 2, 1)));
        v2 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v2), _MM_SHUFFLE(1, 0, 3, 2)));
        v3 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v3), _MM_SHUFFLE(2, 1, 0, 3)));
        v5 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v5), _MM_SHUFFLE(0, 3, 2, 1)));
        v6 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v6), _MM_SHUFFLE(1, 0, 3, 2)));
        v7 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v7), _MM_SHUFFLE(2, 1, 0, 3)));
        v9 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v9), _MM_SHUFFLE(0, 3, 2, 1)));
        v10 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v10), _MM_SHUFFLE(1, 0, 3, 2)));
        v11 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v11), _MM_SHUFFLE(2, 1, 0, 3)));
        v13 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v13), _MM_SHUFFLE(0, 3, 2, 1)));
        v14 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v14), _MM_SHUFFLE(1, 0, 3, 2)));
        v15 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v15), _MM_SHUFFLE(2, 1, 0, 3)));
        // Transpose 2x2
        const __m128 vtmp0 = _mm_unpacklo_ps(v0, v1);  // a e b f   from row 0, 1
        const __m128 vtmp1 = _mm_unpacklo_ps(v2, v3);  // i m j n   from row 2, 3
        const __m128 vtmp2 = _mm_unpackhi_ps(v0, v1);  // c g d h   from row 0, 1
        const __m128 vtmp3 = _mm_unpackhi_ps(v2, v3);  // k o l p   from row 2, 3
        const __m128 vtmp4 = _mm_unpacklo_ps(v4, v5);  // a e b f   from row 0, 1
        const __m128 vtmp5 = _mm_unpacklo_ps(v6, v7);  // i m j n   from row 2, 3
        const __m128 vtmp6 = _mm_unpackhi_ps(v4, v5);  // c g d h   from row 0, 1
        const __m128 vtmp7 = _mm_unpackhi_ps(v6, v7);  // k o l p   from row 2, 3
        const __m128 vtmp8 = _mm_unpacklo_ps(v8, v9);  // a e b f   from row 0, 1
        const __m128 vtmp9 = _mm_unpacklo_ps(v10, v11);  // i m j n   from row 2, 3
        const __m128 vtmp10 = _mm_unpackhi_ps(v8, v9);  // c g d h   from row 0, 1
        const __m128 vtmp11 = _mm_unpackhi_ps(v10, v11);  // k o l p   from row 2, 3
        const __m128 vtmp12 = _mm_unpacklo_ps(v12, v13);  // a e b f   from row 0, 1
        const __m128 vtmp13 = _mm_unpacklo_ps(v14, v15);  // i m j n   from row 2, 3
        const __m128 vtmp14 = _mm_unpackhi_ps(v12, v13);  // c g d h   from row 0, 1
        const __m128 vtmp15 = _mm_unpackhi_ps(v14, v15);  // k o l p   from row 2, 3
        // Transpose 4x4
        v0 = _mm_movelh_ps(vtmp0, vtmp1);  // a e i m   from row 0, 1
        v1 = _mm_movehl_ps(vtmp1, vtmp0);  // b f j n   from row 0, 1
        v2 = _mm_movelh_ps(vtmp2, vtmp3);  // c g k o   from row 2, 3
        v3 = _mm_movehl_ps(vtmp3, vtmp2);  // d h l p   from row 2, 3
        v4 = _mm_movelh_ps(vtmp4, vtmp5);  // a e i m   from row 0, 1
        v5 = _mm_movehl_ps(vtmp5, vtmp4);  // b f j n   from row 0, 1
        v6 = _mm_movelh_ps(vtmp6, vtmp7);  // c g k o   from row 2, 3
        v7 = _mm_movehl_ps(vtmp7, vtmp6);  // d h l p   from row 2, 3
        v8 = _mm_movelh_ps(vtmp8, vtmp9);  // a e i m   from row 0, 1
        v9 = _mm_movehl_ps(vtmp9, vtmp8);  // b f j n   from row 0, 1
        v10 = _mm_movelh_ps(vtmp10, vtmp11);  // c g k o   from row 2, 3
        v11 = _mm_movehl_ps(vtmp11, vtmp10);  // d h l p   from row 2, 3
        v12 = _mm_movelh_ps(vtmp12, vtmp13);  // a e i m   from row 0, 1
        v13 = _mm_movehl_ps(vtmp13, vtmp12);  // b f j n   from row 0, 1
        v14 = _mm_movelh_ps(vtmp14, vtmp15);  // c g k o   from row 2, 3
        v15 = _mm_movehl_ps(vtmp15, vtmp14);  // d h l p   from row 2, 3
        _mm_store_ps(packed_w, v0);
        _mm_store_ps(packed_w + 4, v4);
        _mm_store_ps(packed_w + 8, v8);
        _mm_store_ps(packed_w + 12, v12);
        _mm_store_ps(packed_w + 16, v1);
        _mm_store_ps(packed_w + 20, v5);
        _mm_store_ps(packed_w + 24, v9);
        _mm_store_ps(packed_w + 28, v13);
        _mm_store_ps(packed_w + 32, v2);
        _mm_store_ps(packed_w + 36, v6);
        _mm_store_ps(packed_w + 40, v10);
        _mm_store_ps(packed_w + 44, v14);
        _mm_store_ps(packed_w + 48, v3);
        _mm_store_ps(packed_w + 52, v7);
        _mm_store_ps(packed_w + 56, v11);
        _mm_store_ps(packed_w + 60, v15);
        packed_w += 64;
      }
      packed_w = (float*) ((uintptr_t) packed_w + extra_bytes);
      w0 = w15;
    }

    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 15);
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *packed_w++  = *b++;
        } while (--nb != 0);
        packed_w += (16 - n);
      } else {
        const __m128 vzero = _mm_setzero_ps();
        _mm_store_ps(packed_w, vzero);
        _mm_store_ps(packed_w + 4, vzero);
        _mm_store_ps(packed_w + 8, vzero);
        _mm_store_ps(packed_w + 12, vzero);
        packed_w += 16;
      }

      // NR remainder has less than 16 rows so last row is not loaded
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

      size_t k = kc;

      // KC multiple of 4
      for (; k >= 4; k -= 4) {
        // Read blocks of 4x4
        // a b c d
        // e f g h
        // i j k l
        // m n o p
        __m128 v0x0123 = _mm_loadu_ps(w0);
        w0 += 4;
        __m128 v1x0123 = _mm_loadu_ps(w1);
        w1 += 4;
        __m128 v2x0123 = _mm_loadu_ps(w2);
        w2 += 4;
        __m128 v3x0123 = _mm_loadu_ps(w3);
        w3 += 4;
        __m128 v4x0123 = _mm_loadu_ps(w4);
        w4 += 4;
        __m128 v5x0123 = _mm_loadu_ps(w5);
        w5 += 4;
        __m128 v6x0123 = _mm_loadu_ps(w6);
        w6 += 4;
        __m128 v7x0123 = _mm_loadu_ps(w7);
        w7 += 4;
        __m128 v8x0123 = _mm_loadu_ps(w8);
        w8 += 4;
        __m128 v9x0123 = _mm_loadu_ps(w9);
        w9 += 4;
        __m128 v10x0123 = _mm_loadu_ps(w10);
        w10 += 4;
        __m128 v11x0123 = _mm_loadu_ps(w11);
        w11 += 4;
        __m128 v12x0123 = _mm_loadu_ps(w12);
        w12 += 4;
        __m128 v13x0123 = _mm_loadu_ps(w13);
        w13 += 4;
        __m128 v14x0123 = _mm_loadu_ps(w14);
        w14 += 4;
        __m128 v15x0123  = _mm_undefined_ps();

        // Apply SR4 shuffle
        v1x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v1x0123), _MM_SHUFFLE(0, 3, 2, 1)));
        v2x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v2x0123), _MM_SHUFFLE(1, 0, 3, 2)));
        v3x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v3x0123), _MM_SHUFFLE(2, 1, 0, 3)));
        v5x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v5x0123), _MM_SHUFFLE(0, 3, 2, 1)));
        v6x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v6x0123), _MM_SHUFFLE(1, 0, 3, 2)));
        v7x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v7x0123), _MM_SHUFFLE(2, 1, 0, 3)));
        v9x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v9x0123), _MM_SHUFFLE(0, 3, 2, 1)));
        v10x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v10x0123), _MM_SHUFFLE(1, 0, 3, 2)));
        v11x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v11x0123), _MM_SHUFFLE(2, 1, 0, 3)));
        v13x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v13x0123), _MM_SHUFFLE(0, 3, 2, 1)));
        v14x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v14x0123), _MM_SHUFFLE(1, 0, 3, 2)));

        // Transpose 2x2
        const __m128 vtmp0x0123 = _mm_unpacklo_ps(v0x0123, v1x0123);  // a e b f   from row 0, 1
        const __m128 vtmp1x0123 = _mm_unpacklo_ps(v2x0123, v3x0123);  // i m j n   from row 2, 3
        const __m128 vtmp2x0123 = _mm_unpackhi_ps(v0x0123, v1x0123);  // c g d h   from row 0, 1
        const __m128 vtmp3x0123 = _mm_unpackhi_ps(v2x0123, v3x0123);  // k o l p   from row 2, 3
        const __m128 vtmp4x0123 = _mm_unpacklo_ps(v4x0123, v5x0123);  // a e b f   from row 0, 1
        const __m128 vtmp5x0123 = _mm_unpacklo_ps(v6x0123, v7x0123);  // i m j n   from row 2, 3
        const __m128 vtmp6x0123 = _mm_unpackhi_ps(v4x0123, v5x0123);  // c g d h   from row 0, 1
        const __m128 vtmp7x0123 = _mm_unpackhi_ps(v6x0123, v7x0123);  // k o l p   from row 2, 3
        const __m128 vtmp8x0123 = _mm_unpacklo_ps(v8x0123, v9x0123);  // a e b f   from row 0, 1
        const __m128 vtmp9x0123 = _mm_unpacklo_ps(v10x0123, v11x0123);  // i m j n   from row 2, 3
        const __m128 vtmp10x0123 = _mm_unpackhi_ps(v8x0123, v9x0123);  // c g d h   from row 0, 1
        const __m128 vtmp11x0123 = _mm_unpackhi_ps(v10x0123, v11x0123);  // k o l p   from row 2, 3
        const __m128 vtmp12x0123 = _mm_unpacklo_ps(v12x0123, v13x0123);  // a e b f   from row 0, 1
        const __m128 vtmp13x0123 = _mm_unpacklo_ps(v14x0123, v14x0123);  // i m j n   from row 2, 3
        const __m128 vtmp14x0123 = _mm_unpackhi_ps(v12x0123, v13x0123);  // c g d h   from row 0, 1
        const __m128 vtmp15x0123 = _mm_unpackhi_ps(v14x0123, v14x0123);  // k o l p   from row 2, 3
        // Transpose 4x4
        v0x0123 = _mm_movelh_ps(vtmp0x0123, vtmp1x0123);  // a e i m   from row 0, 1
        v1x0123 = _mm_movehl_ps(vtmp1x0123, vtmp0x0123);  // b f j n   from row 0, 1
        v2x0123 = _mm_movelh_ps(vtmp2x0123, vtmp3x0123);  // c g k o   from row 2, 3
        v3x0123 = _mm_movehl_ps(vtmp3x0123, vtmp2x0123);  // d h l p   from row 2, 3
        v4x0123 = _mm_movelh_ps(vtmp4x0123, vtmp5x0123);  // a e i m   from row 0, 1
        v5x0123 = _mm_movehl_ps(vtmp5x0123, vtmp4x0123);  // b f j n   from row 0, 1
        v6x0123 = _mm_movelh_ps(vtmp6x0123, vtmp7x0123);  // c g k o   from row 2, 3
        v7x0123 = _mm_movehl_ps(vtmp7x0123, vtmp6x0123);  // d h l p   from row 2, 3
        v8x0123 = _mm_movelh_ps(vtmp8x0123, vtmp9x0123);  // a e i m   from row 0, 1
        v9x0123 = _mm_movehl_ps(vtmp9x0123, vtmp8x0123);  // b f j n   from row 0, 1
        v10x0123 = _mm_movelh_ps(vtmp10x0123, vtmp11x0123);  // c g k o   from row 2, 3
        v11x0123 = _mm_movehl_ps(vtmp11x0123, vtmp10x0123);  // d h l p   from row 2, 3
        v12x0123 = _mm_movelh_ps(vtmp12x0123, vtmp13x0123);  // a e i m   from row 0, 1
        v13x0123 = _mm_movehl_ps(vtmp13x0123, vtmp12x0123);  // b f j n   from row 0, 1
        v14x0123 = _mm_movelh_ps(vtmp14x0123, vtmp15x0123);  // c g k o   from row 2, 3
        v15x0123 = _mm_movehl_ps(vtmp15x0123, vtmp14x0123);  // d h l p   from row 2, 3

        _mm_store_ps(packed_w, v0x0123);
        _mm_store_ps(packed_w + 4, v4x0123);
        _mm_store_ps(packed_w + 8, v8x0123);
        _mm_store_ps(packed_w + 12, v12x0123);
        _mm_store_ps(packed_w + 16, v1x0123);
        _mm_store_ps(packed_w + 20, v5x0123);
        _mm_store_ps(packed_w + 24, v9x0123);
        _mm_store_ps(packed_w + 28, v13x0123);
        _mm_store_ps(packed_w + 32, v2x0123);
        _mm_store_ps(packed_w + 36, v6x0123);
        _mm_store_ps(packed_w + 40, v10x0123);
        _mm_store_ps(packed_w + 44, v14x0123);
        _mm_store_ps(packed_w + 48, v3x0123);
        _mm_store_ps(packed_w + 52, v7x0123);
        _mm_store_ps(packed_w + 56, v11x0123);
        _mm_store_ps(packed_w + 60, v15x0123);
        packed_w += 64;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        __m128 v0 = _mm_undefined_ps();
        __m128 v1 = _mm_undefined_ps();
        __m128 v2 = _mm_undefined_ps();
        __m128 v3 = _mm_undefined_ps();
        __m128 v4 = _mm_undefined_ps();
        __m128 v5 = _mm_undefined_ps();
        __m128 v6 = _mm_undefined_ps();
        __m128 v7 = _mm_undefined_ps();
        __m128 v8 = _mm_undefined_ps();
        __m128 v9 = _mm_undefined_ps();
        __m128 v10 = _mm_undefined_ps();
        __m128 v11 = _mm_undefined_ps();
        __m128 v12 = _mm_undefined_ps();
        __m128 v13 = _mm_undefined_ps();
        __m128 v14 = _mm_undefined_ps();
        __m128 v15 = _mm_undefined_ps();

        switch (k) {
          case 1:
            // Read blocks of 4x1
            // a
            // e
            // i
            // m
            v0 = _mm_load_ss(w0);
            w0 += 1;
            v1 = _mm_load_ss(w1);
            w1 += 1;
            v2 = _mm_load_ss(w2);
            w2 += 1;
            v3 = _mm_load_ss(w3);
            w3 += 1;
            v4 = _mm_load_ss(w4);
            w4 += 1;
            v5 = _mm_load_ss(w5);
            w5 += 1;
            v6 = _mm_load_ss(w6);
            w6 += 1;
            v7 = _mm_load_ss(w7);
            w7 += 1;
            v8 = _mm_load_ss(w8);
            w8 += 1;
            v9 = _mm_load_ss(w9);
            w9 += 1;
            v10 = _mm_load_ss(w10);
            w10 += 1;
            v11 = _mm_load_ss(w11);
            w11 += 1;
            v12 = _mm_load_ss(w12);
            w12 += 1;
            v13 = _mm_load_ss(w13);
            w13 += 1;
            v14 = _mm_load_ss(w14);
            w14 += 1;
            break;
          case 2:
            // Read blocks of 4x2
            // a b
            // e f
            // i j
            // m n
            v0 = _mm_castpd_ps(_mm_load_sd((const double*) w0));
            w0 += 2;
            v1 = _mm_castpd_ps(_mm_load_sd((const double*) w1));
            w1 += 2;
            v2 = _mm_castpd_ps(_mm_load_sd((const double*) w2));
            w2 += 2;
            v3 = _mm_castpd_ps(_mm_load_sd((const double*) w3));
            w3 += 2;
            v4 = _mm_castpd_ps(_mm_load_sd((const double*) w4));
            w4 += 2;
            v5 = _mm_castpd_ps(_mm_load_sd((const double*) w5));
            w5 += 2;
            v6 = _mm_castpd_ps(_mm_load_sd((const double*) w6));
            w6 += 2;
            v7 = _mm_castpd_ps(_mm_load_sd((const double*) w7));
            w7 += 2;
            v8 = _mm_castpd_ps(_mm_load_sd((const double*) w8));
            w8 += 2;
            v9 = _mm_castpd_ps(_mm_load_sd((const double*) w9));
            w9 += 2;
            v10 = _mm_castpd_ps(_mm_load_sd((const double*) w10));
            w10 += 2;
            v11 = _mm_castpd_ps(_mm_load_sd((const double*) w11));
            w11 += 2;
            v12 = _mm_castpd_ps(_mm_load_sd((const double*) w12));
            w12 += 2;
            v13 = _mm_castpd_ps(_mm_load_sd((const double*) w13));
            w13 += 2;
            v14 = _mm_castpd_ps(_mm_load_sd((const double*) w14));
            w14 += 2;
            break;
          case 3:
          {
            // Read blocks of 4x3
            // a b c
            // e f g
            // i j k
            // m n o
            const __m128 v0lo = _mm_castpd_ps(_mm_load_sd((const double*) w0));
            const __m128 v0hi = _mm_load_ss(w0 + 2);
            v0 = _mm_movelh_ps(v0lo, v0hi);
            w0 += 3;
            const __m128 v1lo = _mm_castpd_ps(_mm_load_sd((const double*) w1));
            const __m128 v1hi = _mm_load_ss(w1 + 2);
            v1 = _mm_movelh_ps(v1lo, v1hi);
            w1 += 3;
            const __m128 v2lo = _mm_castpd_ps(_mm_load_sd((const double*) w2));
            const __m128 v2hi = _mm_load_ss(w2 + 2);
            v2 = _mm_movelh_ps(v2lo, v2hi);
            w2 += 3;
            const __m128 v3lo = _mm_castpd_ps(_mm_load_sd((const double*) w3));
            const __m128 v3hi = _mm_load_ss(w3 + 2);
            v3 = _mm_movelh_ps(v3lo, v3hi);
            w3 += 3;
            const __m128 v4lo = _mm_castpd_ps(_mm_load_sd((const double*) w4));
            const __m128 v4hi = _mm_load_ss(w4 + 2);
            v4 = _mm_movelh_ps(v4lo, v4hi);
            w4 += 3;
            const __m128 v5lo = _mm_castpd_ps(_mm_load_sd((const double*) w5));
            const __m128 v5hi = _mm_load_ss(w5 + 2);
            v5 = _mm_movelh_ps(v5lo, v5hi);
            w5 += 3;
            const __m128 v6lo = _mm_castpd_ps(_mm_load_sd((const double*) w6));
            const __m128 v6hi = _mm_load_ss(w6 + 2);
            v6 = _mm_movelh_ps(v6lo, v6hi);
            w6 += 3;
            const __m128 v7lo = _mm_castpd_ps(_mm_load_sd((const double*) w7));
            const __m128 v7hi = _mm_load_ss(w7 + 2);
            v7 = _mm_movelh_ps(v7lo, v7hi);
            w7 += 3;
            const __m128 v8lo = _mm_castpd_ps(_mm_load_sd((const double*) w8));
            const __m128 v8hi = _mm_load_ss(w8 + 2);
            v8 = _mm_movelh_ps(v8lo, v8hi);
            w8 += 3;
            const __m128 v9lo = _mm_castpd_ps(_mm_load_sd((const double*) w9));
            const __m128 v9hi = _mm_load_ss(w9 + 2);
            v9 = _mm_movelh_ps(v9lo, v9hi);
            w9 += 3;
            const __m128 v10lo = _mm_castpd_ps(_mm_load_sd((const double*) w10));
            const __m128 v10hi = _mm_load_ss(w10 + 2);
            v10 = _mm_movelh_ps(v10lo, v10hi);
            w10 += 3;
            const __m128 v11lo = _mm_castpd_ps(_mm_load_sd((const double*) w11));
            const __m128 v11hi = _mm_load_ss(w11 + 2);
            v11 = _mm_movelh_ps(v11lo, v11hi);
            w11 += 3;
            const __m128 v12lo = _mm_castpd_ps(_mm_load_sd((const double*) w12));
            const __m128 v12hi = _mm_load_ss(w12 + 2);
            v12 = _mm_movelh_ps(v12lo, v12hi);
            w12 += 3;
            const __m128 v13lo = _mm_castpd_ps(_mm_load_sd((const double*) w13));
            const __m128 v13hi = _mm_load_ss(w13 + 2);
            v13 = _mm_movelh_ps(v13lo, v13hi);
            w13 += 3;
            const __m128 v14lo = _mm_castpd_ps(_mm_load_sd((const double*) w14));
            const __m128 v14hi = _mm_load_ss(w14 + 2);
            v14 = _mm_movelh_ps(v14lo, v14hi);
            w14 += 3;
            break;
          }
          default:
            XNN_UNREACHABLE;
        }

        // Apply SR4 shuffle
        v1 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v1), _MM_SHUFFLE(0, 3, 2, 1)));
        v2 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v2), _MM_SHUFFLE(1, 0, 3, 2)));
        v3 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v3), _MM_SHUFFLE(2, 1, 0, 3)));
        v5 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v5), _MM_SHUFFLE(0, 3, 2, 1)));
        v6 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v6), _MM_SHUFFLE(1, 0, 3, 2)));
        v7 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v7), _MM_SHUFFLE(2, 1, 0, 3)));
        v9 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v9), _MM_SHUFFLE(0, 3, 2, 1)));
        v10 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v10), _MM_SHUFFLE(1, 0, 3, 2)));
        v11 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v11), _MM_SHUFFLE(2, 1, 0, 3)));
        v13 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v13), _MM_SHUFFLE(0, 3, 2, 1)));
        v14 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v14), _MM_SHUFFLE(1, 0, 3, 2)));
        // Transpose 2x2
        const __m128 vtmp0 = _mm_unpacklo_ps(v0, v1);  // a e b f   from row 0, 1
        const __m128 vtmp1 = _mm_unpacklo_ps(v2, v3);  // i m j n   from row 2, 3
        const __m128 vtmp2 = _mm_unpackhi_ps(v0, v1);  // c g d h   from row 0, 1
        const __m128 vtmp3 = _mm_unpackhi_ps(v2, v3);  // k o l p   from row 2, 3
        const __m128 vtmp4 = _mm_unpacklo_ps(v4, v5);  // a e b f   from row 0, 1
        const __m128 vtmp5 = _mm_unpacklo_ps(v6, v7);  // i m j n   from row 2, 3
        const __m128 vtmp6 = _mm_unpackhi_ps(v4, v5);  // c g d h   from row 0, 1
        const __m128 vtmp7 = _mm_unpackhi_ps(v6, v7);  // k o l p   from row 2, 3
        const __m128 vtmp8 = _mm_unpacklo_ps(v8, v9);  // a e b f   from row 0, 1
        const __m128 vtmp9 = _mm_unpacklo_ps(v10, v11);  // i m j n   from row 2, 3
        const __m128 vtmp10 = _mm_unpackhi_ps(v8, v9);  // c g d h   from row 0, 1
        const __m128 vtmp11 = _mm_unpackhi_ps(v10, v11);  // k o l p   from row 2, 3
        const __m128 vtmp12 = _mm_unpacklo_ps(v12, v13);  // a e b f   from row 0, 1
        const __m128 vtmp13 = _mm_unpacklo_ps(v14, v14);  // i m j n   from row 2, 3
        const __m128 vtmp14 = _mm_unpackhi_ps(v12, v13);  // c g d h   from row 0, 1
        const __m128 vtmp15 = _mm_unpackhi_ps(v14, v14);  // k o l p   from row 2, 3
        // Transpose 4x4
        v0 = _mm_movelh_ps(vtmp0, vtmp1);  // a e i m   from row 0, 1
        v1 = _mm_movehl_ps(vtmp1, vtmp0);  // b f j n   from row 0, 1
        v2 = _mm_movelh_ps(vtmp2, vtmp3);  // c g k o   from row 2, 3
        v3 = _mm_movehl_ps(vtmp3, vtmp2);  // d h l p   from row 2, 3
        v4 = _mm_movelh_ps(vtmp4, vtmp5);  // a e i m   from row 0, 1
        v5 = _mm_movehl_ps(vtmp5, vtmp4);  // b f j n   from row 0, 1
        v6 = _mm_movelh_ps(vtmp6, vtmp7);  // c g k o   from row 2, 3
        v7 = _mm_movehl_ps(vtmp7, vtmp6);  // d h l p   from row 2, 3
        v8 = _mm_movelh_ps(vtmp8, vtmp9);  // a e i m   from row 0, 1
        v9 = _mm_movehl_ps(vtmp9, vtmp8);  // b f j n   from row 0, 1
        v10 = _mm_movelh_ps(vtmp10, vtmp11);  // c g k o   from row 2, 3
        v11 = _mm_movehl_ps(vtmp11, vtmp10);  // d h l p   from row 2, 3
        v12 = _mm_movelh_ps(vtmp12, vtmp13);  // a e i m   from row 0, 1
        v13 = _mm_movehl_ps(vtmp13, vtmp12);  // b f j n   from row 0, 1
        v14 = _mm_movelh_ps(vtmp14, vtmp15);  // c g k o   from row 2, 3
        v15 = _mm_movehl_ps(vtmp15, vtmp14);  // d h l p   from row 2, 3
        _mm_store_ps(packed_w, v0);
        _mm_store_ps(packed_w + 4, v4);
        _mm_store_ps(packed_w + 8, v8);
        _mm_store_ps(packed_w + 12, v12);
        _mm_store_ps(packed_w + 16, v1);
        _mm_store_ps(packed_w + 20, v5);
        _mm_store_ps(packed_w + 24, v9);
        _mm_store_ps(packed_w + 28, v13);
        _mm_store_ps(packed_w + 32, v2);
        _mm_store_ps(packed_w + 36, v6);
        _mm_store_ps(packed_w + 40, v10);
        _mm_store_ps(packed_w + 44, v14);
        _mm_store_ps(packed_w + 48, v3);
        _mm_store_ps(packed_w + 52, v7);
        _mm_store_ps(packed_w + 56, v11);
        _mm_store_ps(packed_w + 60, v15);
        packed_w += 64;
      }
      packed_w = (float*) ((uintptr_t) packed_w + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
