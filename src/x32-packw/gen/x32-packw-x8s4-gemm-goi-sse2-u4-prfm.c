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
#include "xnnpack/prefetch.h"


void xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_u4_prfm(
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
  assert(nr == 8);
  assert(kr == 1);
  assert(sr == 4);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const float* b = (const float*) bias;
  float* packed_w = (float*) packed_weights;
  do {
    // NC main loop multiple of 8
    const float* w0 = (const float*) weights;
    size_t n = nc;

    for (; n >= 8; n -= 8) {
      if XNN_LIKELY(b != NULL) {
        const __m128 vb0 = _mm_loadu_ps(b);
        const __m128 vb4 = _mm_loadu_ps(b + 4);
        _mm_store_ps(packed_w, vb0);
        _mm_store_ps(packed_w + 4, vb4);
        b += 8;
      } else {
        const __m128 vzero = _mm_setzero_ps();
        _mm_store_ps(packed_w, vzero);
        _mm_store_ps(packed_w + 4, vzero);
      }
      packed_w += 8;

      const float* w1 = w0 + kc;
      const float* w2 = w1 + kc;
      const float* w3 = w2 + kc;
      const float* w4 = w3 + kc;
      const float* w5 = w4 + kc;
      const float* w6 = w5 + kc;
      const float* w7 = w6 + kc;
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

        // Apply SR4 shuffle
        v1x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v1x0123), _MM_SHUFFLE(0, 3, 2, 1)));
        v2x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v2x0123), _MM_SHUFFLE(1, 0, 3, 2)));
        v3x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v3x0123), _MM_SHUFFLE(2, 1, 0, 3)));
        v5x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v5x0123), _MM_SHUFFLE(0, 3, 2, 1)));
        v6x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v6x0123), _MM_SHUFFLE(1, 0, 3, 2)));
        v7x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v7x0123), _MM_SHUFFLE(2, 1, 0, 3)));

        // Transpose 2x2
        const __m128 vtmp0x0123 = _mm_unpacklo_ps(v0x0123, v1x0123);  // a e b f   from row 0, 1
        const __m128 vtmp1x0123 = _mm_unpacklo_ps(v2x0123, v3x0123);  // i m j n   from row 2, 3
        const __m128 vtmp2x0123 = _mm_unpackhi_ps(v0x0123, v1x0123);  // c g d h   from row 0, 1
        const __m128 vtmp3x0123 = _mm_unpackhi_ps(v2x0123, v3x0123);  // k o l p   from row 2, 3
        const __m128 vtmp4x0123 = _mm_unpacklo_ps(v4x0123, v5x0123);  // a e b f   from row 0, 1
        const __m128 vtmp5x0123 = _mm_unpacklo_ps(v6x0123, v7x0123);  // i m j n   from row 2, 3
        const __m128 vtmp6x0123 = _mm_unpackhi_ps(v4x0123, v5x0123);  // c g d h   from row 0, 1
        const __m128 vtmp7x0123 = _mm_unpackhi_ps(v6x0123, v7x0123);  // k o l p   from row 2, 3
        xnn_prefetch_to_l1((const int8_t*) w0 + 128);
        xnn_prefetch_to_l1((const int8_t*) w1 + 128);
        xnn_prefetch_to_l1((const int8_t*) w2 + 128);
        xnn_prefetch_to_l1((const int8_t*) w3 + 128);
        xnn_prefetch_to_l1((const int8_t*) w4 + 128);
        xnn_prefetch_to_l1((const int8_t*) w5 + 128);
        xnn_prefetch_to_l1((const int8_t*) w6 + 128);
        xnn_prefetch_to_l1((const int8_t*) w7 + 128);
        // Transpose 4x4
        v0x0123 = _mm_movelh_ps(vtmp0x0123, vtmp1x0123);  // a e i m   from row 0, 1
        v1x0123 = _mm_movehl_ps(vtmp1x0123, vtmp0x0123);  // b f j n   from row 0, 1
        v2x0123 = _mm_movelh_ps(vtmp2x0123, vtmp3x0123);  // c g k o   from row 2, 3
        v3x0123 = _mm_movehl_ps(vtmp3x0123, vtmp2x0123);  // d h l p   from row 2, 3
        v4x0123 = _mm_movelh_ps(vtmp4x0123, vtmp5x0123);  // a e i m   from row 0, 1
        v5x0123 = _mm_movehl_ps(vtmp5x0123, vtmp4x0123);  // b f j n   from row 0, 1
        v6x0123 = _mm_movelh_ps(vtmp6x0123, vtmp7x0123);  // c g k o   from row 2, 3
        v7x0123 = _mm_movehl_ps(vtmp7x0123, vtmp6x0123);  // d h l p   from row 2, 3

        _mm_store_ps(packed_w, v0x0123);
        _mm_store_ps(packed_w + 4, v4x0123);
        _mm_store_ps(packed_w + 8, v1x0123);
        _mm_store_ps(packed_w + 12, v5x0123);
        _mm_store_ps(packed_w + 16, v2x0123);
        _mm_store_ps(packed_w + 20, v6x0123);
        _mm_store_ps(packed_w + 24, v3x0123);
        _mm_store_ps(packed_w + 28, v7x0123);
        packed_w += 32;
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
        // Transpose 2x2
        const __m128 vtmp0 = _mm_unpacklo_ps(v0, v1);  // a e b f   from row 0, 1
        const __m128 vtmp1 = _mm_unpacklo_ps(v2, v3);  // i m j n   from row 2, 3
        const __m128 vtmp2 = _mm_unpackhi_ps(v0, v1);  // c g d h   from row 0, 1
        const __m128 vtmp3 = _mm_unpackhi_ps(v2, v3);  // k o l p   from row 2, 3
        const __m128 vtmp4 = _mm_unpacklo_ps(v4, v5);  // a e b f   from row 0, 1
        const __m128 vtmp5 = _mm_unpacklo_ps(v6, v7);  // i m j n   from row 2, 3
        const __m128 vtmp6 = _mm_unpackhi_ps(v4, v5);  // c g d h   from row 0, 1
        const __m128 vtmp7 = _mm_unpackhi_ps(v6, v7);  // k o l p   from row 2, 3
        // Transpose 4x4
        v0 = _mm_movelh_ps(vtmp0, vtmp1);  // a e i m   from row 0, 1
        v1 = _mm_movehl_ps(vtmp1, vtmp0);  // b f j n   from row 0, 1
        v2 = _mm_movelh_ps(vtmp2, vtmp3);  // c g k o   from row 2, 3
        v3 = _mm_movehl_ps(vtmp3, vtmp2);  // d h l p   from row 2, 3
        v4 = _mm_movelh_ps(vtmp4, vtmp5);  // a e i m   from row 0, 1
        v5 = _mm_movehl_ps(vtmp5, vtmp4);  // b f j n   from row 0, 1
        v6 = _mm_movelh_ps(vtmp6, vtmp7);  // c g k o   from row 2, 3
        v7 = _mm_movehl_ps(vtmp7, vtmp6);  // d h l p   from row 2, 3
        _mm_store_ps(packed_w, v0);
        _mm_store_ps(packed_w + 4, v4);
        _mm_store_ps(packed_w + 8, v1);
        _mm_store_ps(packed_w + 12, v5);
        _mm_store_ps(packed_w + 16, v2);
        _mm_store_ps(packed_w + 20, v6);
        _mm_store_ps(packed_w + 24, v3);
        _mm_store_ps(packed_w + 28, v7);
        packed_w += 32;
      }
      packed_w = (float*) ((uintptr_t) packed_w + extra_bytes);
      w0 = w7;
    }

    // NC remainder (1..7)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 7);
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *packed_w++  = *b++;
        } while (--nb != 0);
        packed_w += (8 - n);
      } else {
        const __m128 vzero = _mm_setzero_ps();
        _mm_store_ps(packed_w, vzero);
        _mm_store_ps(packed_w + 4, vzero);
        packed_w += 8;
      }

      // NR remainder has less than 8 rows so last row is not loaded
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
        __m128 v7x0123  = _mm_undefined_ps();

        // Apply SR4 shuffle
        v1x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v1x0123), _MM_SHUFFLE(0, 3, 2, 1)));
        v2x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v2x0123), _MM_SHUFFLE(1, 0, 3, 2)));
        v3x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v3x0123), _MM_SHUFFLE(2, 1, 0, 3)));
        v5x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v5x0123), _MM_SHUFFLE(0, 3, 2, 1)));
        v6x0123 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v6x0123), _MM_SHUFFLE(1, 0, 3, 2)));

        // Transpose 2x2
        const __m128 vtmp0x0123 = _mm_unpacklo_ps(v0x0123, v1x0123);  // a e b f   from row 0, 1
        const __m128 vtmp1x0123 = _mm_unpacklo_ps(v2x0123, v3x0123);  // i m j n   from row 2, 3
        const __m128 vtmp2x0123 = _mm_unpackhi_ps(v0x0123, v1x0123);  // c g d h   from row 0, 1
        const __m128 vtmp3x0123 = _mm_unpackhi_ps(v2x0123, v3x0123);  // k o l p   from row 2, 3
        const __m128 vtmp4x0123 = _mm_unpacklo_ps(v4x0123, v5x0123);  // a e b f   from row 0, 1
        const __m128 vtmp5x0123 = _mm_unpacklo_ps(v6x0123, v6x0123);  // i m j n   from row 2, 3
        const __m128 vtmp6x0123 = _mm_unpackhi_ps(v4x0123, v5x0123);  // c g d h   from row 0, 1
        const __m128 vtmp7x0123 = _mm_unpackhi_ps(v6x0123, v6x0123);  // k o l p   from row 2, 3
        xnn_prefetch_to_l1((const int8_t*) w0 + 128);
        xnn_prefetch_to_l1((const int8_t*) w1 + 128);
        xnn_prefetch_to_l1((const int8_t*) w2 + 128);
        xnn_prefetch_to_l1((const int8_t*) w3 + 128);
        xnn_prefetch_to_l1((const int8_t*) w4 + 128);
        xnn_prefetch_to_l1((const int8_t*) w5 + 128);
        xnn_prefetch_to_l1((const int8_t*) w6 + 128);
        // Transpose 4x4
        v0x0123 = _mm_movelh_ps(vtmp0x0123, vtmp1x0123);  // a e i m   from row 0, 1
        v1x0123 = _mm_movehl_ps(vtmp1x0123, vtmp0x0123);  // b f j n   from row 0, 1
        v2x0123 = _mm_movelh_ps(vtmp2x0123, vtmp3x0123);  // c g k o   from row 2, 3
        v3x0123 = _mm_movehl_ps(vtmp3x0123, vtmp2x0123);  // d h l p   from row 2, 3
        v4x0123 = _mm_movelh_ps(vtmp4x0123, vtmp5x0123);  // a e i m   from row 0, 1
        v5x0123 = _mm_movehl_ps(vtmp5x0123, vtmp4x0123);  // b f j n   from row 0, 1
        v6x0123 = _mm_movelh_ps(vtmp6x0123, vtmp7x0123);  // c g k o   from row 2, 3
        v7x0123 = _mm_movehl_ps(vtmp7x0123, vtmp6x0123);  // d h l p   from row 2, 3

        _mm_store_ps(packed_w, v0x0123);
        _mm_store_ps(packed_w + 4, v4x0123);
        _mm_store_ps(packed_w + 8, v1x0123);
        _mm_store_ps(packed_w + 12, v5x0123);
        _mm_store_ps(packed_w + 16, v2x0123);
        _mm_store_ps(packed_w + 20, v6x0123);
        _mm_store_ps(packed_w + 24, v3x0123);
        _mm_store_ps(packed_w + 28, v7x0123);
        packed_w += 32;
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
        // Transpose 2x2
        const __m128 vtmp0 = _mm_unpacklo_ps(v0, v1);  // a e b f   from row 0, 1
        const __m128 vtmp1 = _mm_unpacklo_ps(v2, v3);  // i m j n   from row 2, 3
        const __m128 vtmp2 = _mm_unpackhi_ps(v0, v1);  // c g d h   from row 0, 1
        const __m128 vtmp3 = _mm_unpackhi_ps(v2, v3);  // k o l p   from row 2, 3
        const __m128 vtmp4 = _mm_unpacklo_ps(v4, v5);  // a e b f   from row 0, 1
        const __m128 vtmp5 = _mm_unpacklo_ps(v6, v6);  // i m j n   from row 2, 3
        const __m128 vtmp6 = _mm_unpackhi_ps(v4, v5);  // c g d h   from row 0, 1
        const __m128 vtmp7 = _mm_unpackhi_ps(v6, v6);  // k o l p   from row 2, 3
        // Transpose 4x4
        v0 = _mm_movelh_ps(vtmp0, vtmp1);  // a e i m   from row 0, 1
        v1 = _mm_movehl_ps(vtmp1, vtmp0);  // b f j n   from row 0, 1
        v2 = _mm_movelh_ps(vtmp2, vtmp3);  // c g k o   from row 2, 3
        v3 = _mm_movehl_ps(vtmp3, vtmp2);  // d h l p   from row 2, 3
        v4 = _mm_movelh_ps(vtmp4, vtmp5);  // a e i m   from row 0, 1
        v5 = _mm_movehl_ps(vtmp5, vtmp4);  // b f j n   from row 0, 1
        v6 = _mm_movelh_ps(vtmp6, vtmp7);  // c g k o   from row 2, 3
        v7 = _mm_movehl_ps(vtmp7, vtmp6);  // d h l p   from row 2, 3
        _mm_store_ps(packed_w, v0);
        _mm_store_ps(packed_w + 4, v4);
        _mm_store_ps(packed_w + 8, v1);
        _mm_store_ps(packed_w + 12, v5);
        _mm_store_ps(packed_w + 16, v2);
        _mm_store_ps(packed_w + 20, v6);
        _mm_store_ps(packed_w + 24, v3);
        _mm_store_ps(packed_w + 28, v7);
        packed_w += 32;
      }
      packed_w = (float*) ((uintptr_t) packed_w + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
