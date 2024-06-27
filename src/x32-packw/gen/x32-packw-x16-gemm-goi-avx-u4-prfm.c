// Auto-generated file. Do not edit!
//   Template: src/x32-packw/avx.c.in
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

#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/packw.h"
#include "xnnpack/prefetch.h"


void xnn_x32_packw_gemm_goi_ukernel_x16__avx_u4_prfm(
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
  assert(nr == 16);   // This kernel is for NR=16
  assert(kr == 1);
  assert(sr == 1);
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
        const __m256 vb0 = _mm256_loadu_ps(b);
        const __m256 vb8 = _mm256_loadu_ps(b + 8);
        _mm256_store_ps(packed_w, vb0);
        _mm256_store_ps(packed_w + 8, vb8);
        b += 16;
      } else {
        const __m256 vzero = _mm256_setzero_ps();
        _mm256_store_ps(packed_w, vzero);
        _mm256_store_ps(packed_w + 8, vzero);
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

      // KC main loop multiple of 4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        // Read blocks of 4x4
        // a b c d
        // e f g h
        // i j k l
        // m n o p
        // Load first 4 rows of N into low part of each register
        __m256 v0x0123 = _mm256_castps128_ps256(_mm_loadu_ps(w0));
        w0 += 4;
        __m256 v1x0123 = _mm256_castps128_ps256(_mm_loadu_ps(w1));
        w1 += 4;
        __m256 v2x0123 = _mm256_castps128_ps256(_mm_loadu_ps(w2));
        w2 += 4;
        __m256 v3x0123 = _mm256_castps128_ps256(_mm_loadu_ps(w3));
        w3 += 4;
        __m256 v8x0123 = _mm256_castps128_ps256(_mm_loadu_ps(w8));
        w8 += 4;
        __m256 v9x0123 = _mm256_castps128_ps256(_mm_loadu_ps(w9));
        w9 += 4;
        __m256 v10x0123 = _mm256_castps128_ps256(_mm_loadu_ps(w10));
        w10 += 4;
        __m256 v11x0123 = _mm256_castps128_ps256(_mm_loadu_ps(w11));
        w11 += 4;
        // Load next 4 rows of N into the high part of each register
        v0x0123 = _mm256_insertf128_ps(v0x0123, _mm_loadu_ps(w4), 1);
        w4 += 4;
        v1x0123 = _mm256_insertf128_ps(v1x0123, _mm_loadu_ps(w5), 1);
        w5 += 4;
        v2x0123 = _mm256_insertf128_ps(v2x0123, _mm_loadu_ps(w6), 1);
        w6 += 4;
        v3x0123 = _mm256_insertf128_ps(v3x0123, _mm_loadu_ps(w7), 1);
        w7 += 4;
        v8x0123 = _mm256_insertf128_ps(v8x0123, _mm_loadu_ps(w12), 1);
        w12 += 4;
        v9x0123 = _mm256_insertf128_ps(v9x0123, _mm_loadu_ps(w13), 1);
        w13 += 4;
        v10x0123 = _mm256_insertf128_ps(v10x0123, _mm_loadu_ps(w14), 1);
        w14 += 4;
        v11x0123 = _mm256_insertf128_ps(v11x0123, _mm_loadu_ps(w15), 1);
        w15 += 4;

        // Transpose 2x2
        const __m256 vtmp0x0123 = _mm256_unpacklo_ps(v0x0123, v1x0123);  // a e b f   from row 0, 1
        const __m256 vtmp1x0123 = _mm256_unpacklo_ps(v2x0123, v3x0123);  // i m j n   from row 2, 3
        const __m256 vtmp2x0123 = _mm256_unpackhi_ps(v0x0123, v1x0123);  // c g d h   from row 0, 1
        const __m256 vtmp3x0123 = _mm256_unpackhi_ps(v2x0123, v3x0123);  // k o l p   from row 2, 3
        const __m256 vtmp8x0123 = _mm256_unpacklo_ps(v8x0123, v9x0123);  // a e b f   from row 0, 1
        const __m256 vtmp9x0123 = _mm256_unpacklo_ps(v10x0123, v11x0123);  // i m j n   from row 2, 3
        const __m256 vtmp10x0123 = _mm256_unpackhi_ps(v8x0123, v9x0123);  // c g d h   from row 0, 1
        const __m256 vtmp11x0123 = _mm256_unpackhi_ps(v10x0123, v11x0123);  // k o l p   from row 2, 3
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
        // Transpose 4x4
        v0x0123 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(vtmp0x0123), _mm256_castps_pd(vtmp1x0123)));  // a e i m   from row 0, 1
        v1x0123 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(vtmp0x0123), _mm256_castps_pd(vtmp1x0123)));  // b f j n   from row 0, 1
        v2x0123 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(vtmp2x0123), _mm256_castps_pd(vtmp3x0123)));  // c g k o   from row 2, 3
        v3x0123 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(vtmp2x0123), _mm256_castps_pd(vtmp3x0123)));  // d h l p   from row 2, 3
        v8x0123 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(vtmp8x0123), _mm256_castps_pd(vtmp9x0123)));  // a e i m   from row 0, 1
        v9x0123 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(vtmp8x0123), _mm256_castps_pd(vtmp9x0123)));  // b f j n   from row 0, 1
        v10x0123 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(vtmp10x0123), _mm256_castps_pd(vtmp11x0123)));  // c g k o   from row 2, 3
        v11x0123 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(vtmp10x0123), _mm256_castps_pd(vtmp11x0123)));  // d h l p   from row 2, 3

        _mm256_store_ps(packed_w, v0x0123);
        _mm256_store_ps(packed_w + 8, v8x0123);
        _mm256_store_ps(packed_w + 16, v1x0123);
        _mm256_store_ps(packed_w + 24, v9x0123);
        _mm256_store_ps(packed_w + 32, v2x0123);
        _mm256_store_ps(packed_w + 40, v10x0123);
        _mm256_store_ps(packed_w + 48, v3x0123);
        _mm256_store_ps(packed_w + 56, v11x0123);
        packed_w += 64;
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

          _mm_store_ps(packed_w, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v8);
          _mm_store_ps(packed_w + 12, v12);
          _mm_store_ps(packed_w + 16, v1);
          _mm_store_ps(packed_w + 20, v5);
          _mm_store_ps(packed_w + 24, v9);
          _mm_store_ps(packed_w + 28, v13);
          packed_w += 32;
        }
        if (k & 1) {
          // Read blocks of 4x1
          // a
          // b
          // c
          // d
          __m128 v0 = _mm_load_ss(w0);  w0 += 1;
          __m128 v1 = _mm_load_ss(w1);  w1 += 1;
          __m128 v2 = _mm_load_ss(w2);  w2 += 1;
          __m128 v3 = _mm_load_ss(w3);  w3 += 1;
          __m128 v4 = _mm_load_ss(w4);  w4 += 1;
          __m128 v5 = _mm_load_ss(w5);  w5 += 1;
          __m128 v6 = _mm_load_ss(w6);  w6 += 1;
          __m128 v7 = _mm_load_ss(w7);  w7 += 1;
          __m128 v8 = _mm_load_ss(w8);  w8 += 1;
          __m128 v9 = _mm_load_ss(w9);  w9 += 1;
          __m128 v10 = _mm_load_ss(w10);  w10 += 1;
          __m128 v11 = _mm_load_ss(w11);  w11 += 1;
          __m128 v12 = _mm_load_ss(w12);  w12 += 1;
          __m128 v13 = _mm_load_ss(w13);  w13 += 1;
          __m128 v14 = _mm_load_ss(w14);  w14 += 1;
          __m128 v15 = _mm_load_ss(w15);  w15 += 1;

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

          _mm_store_ps(packed_w, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v8);
          _mm_store_ps(packed_w + 12, v12);
          packed_w += 16;
        }
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
        const __m256 vzero = _mm256_setzero_ps();
        _mm256_store_ps(packed_w, vzero);
        _mm256_store_ps(packed_w + 8, vzero);
        packed_w += 16;
      }

      // NR remainder has less than 16 rows so last row is not loaded
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

      // KC main loop multiple of 4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        // Read blocks of 4x4
        // a b c d
        // e f g h
        // i j k l
        // m n o p
        // Load first 4 rows of N into low part of each register
        __m256 v0x0123 = _mm256_castps128_ps256(_mm_loadu_ps(w0));
        w0 += 4;
        __m256 v1x0123 = _mm256_castps128_ps256(_mm_loadu_ps(w1));
        w1 += 4;
        __m256 v2x0123 = _mm256_castps128_ps256(_mm_loadu_ps(w2));
        w2 += 4;
        __m256 v3x0123 = _mm256_castps128_ps256(_mm_loadu_ps(w3));
        w3 += 4;
        __m256 v8x0123 = _mm256_castps128_ps256(_mm_loadu_ps(w8));
        w8 += 4;
        __m256 v9x0123 = _mm256_castps128_ps256(_mm_loadu_ps(w9));
        w9 += 4;
        __m256 v10x0123 = _mm256_castps128_ps256(_mm_loadu_ps(w10));
        w10 += 4;
        // castps leaves upper 128 bits undefined, so zero them.
        __m256 v11x0123 = _mm256_zextps128_ps256(_mm_loadu_ps(w11));
        w11 += 4;
        // Load next 4 rows of N into the high part of each register
        v0x0123 = _mm256_insertf128_ps(v0x0123, _mm_loadu_ps(w4), 1);
        w4 += 4;
        v1x0123 = _mm256_insertf128_ps(v1x0123, _mm_loadu_ps(w5), 1);
        w5 += 4;
        v2x0123 = _mm256_insertf128_ps(v2x0123, _mm_loadu_ps(w6), 1);
        w6 += 4;
        v3x0123 = _mm256_insertf128_ps(v3x0123, _mm_loadu_ps(w7), 1);
        w7 += 4;
        v8x0123 = _mm256_insertf128_ps(v8x0123, _mm_loadu_ps(w12), 1);
        w12 += 4;
        v9x0123 = _mm256_insertf128_ps(v9x0123, _mm_loadu_ps(w13), 1);
        w13 += 4;
        v10x0123 = _mm256_insertf128_ps(v10x0123, _mm_loadu_ps(w14), 1);
        w14 += 4;

        // Transpose 2x2
        const __m256 vtmp0x0123 = _mm256_unpacklo_ps(v0x0123, v1x0123);  // a e b f   from row 0, 1
        const __m256 vtmp1x0123 = _mm256_unpacklo_ps(v2x0123, v3x0123);  // i m j n   from row 2, 3
        const __m256 vtmp2x0123 = _mm256_unpackhi_ps(v0x0123, v1x0123);  // c g d h   from row 0, 1
        const __m256 vtmp3x0123 = _mm256_unpackhi_ps(v2x0123, v3x0123);  // k o l p   from row 2, 3
        const __m256 vtmp8x0123 = _mm256_unpacklo_ps(v8x0123, v9x0123);  // a e b f   from row 0, 1
        const __m256 vtmp9x0123 = _mm256_unpacklo_ps(v10x0123, v11x0123);  // i m j n   from row 2, 3
        const __m256 vtmp10x0123 = _mm256_unpackhi_ps(v8x0123, v9x0123);  // c g d h   from row 0, 1
        const __m256 vtmp11x0123 = _mm256_unpackhi_ps(v10x0123, v11x0123);  // k o l p   from row 2, 3
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
        // Transpose 4x4
        v0x0123 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(vtmp0x0123), _mm256_castps_pd(vtmp1x0123)));  // a e i m   from row 0, 1
        v1x0123 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(vtmp0x0123), _mm256_castps_pd(vtmp1x0123)));  // b f j n   from row 0, 1
        v2x0123 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(vtmp2x0123), _mm256_castps_pd(vtmp3x0123)));  // c g k o   from row 2, 3
        v3x0123 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(vtmp2x0123), _mm256_castps_pd(vtmp3x0123)));  // d h l p   from row 2, 3
        v8x0123 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(vtmp8x0123), _mm256_castps_pd(vtmp9x0123)));  // a e i m   from row 0, 1
        v9x0123 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(vtmp8x0123), _mm256_castps_pd(vtmp9x0123)));  // b f j n   from row 0, 1
        v10x0123 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(vtmp10x0123), _mm256_castps_pd(vtmp11x0123)));  // c g k o   from row 2, 3
        v11x0123 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(vtmp10x0123), _mm256_castps_pd(vtmp11x0123)));  // d h l p   from row 2, 3

        _mm256_store_ps(packed_w, v0x0123);
        _mm256_store_ps(packed_w + 8, v8x0123);
        _mm256_store_ps(packed_w + 16, v1x0123);
        _mm256_store_ps(packed_w + 24, v9x0123);
        _mm256_store_ps(packed_w + 32, v2x0123);
        _mm256_store_ps(packed_w + 40, v10x0123);
        _mm256_store_ps(packed_w + 48, v3x0123);
        _mm256_store_ps(packed_w + 56, v11x0123);
        packed_w += 64;
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
          const __m128 vtmp13 = _mm_unpacklo_ps(v14, v14);  // e g f h   from row 2, 3
          // Transpose 4x4
          v12 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp12), _mm_castps_pd(vtmp13)));  // a c e g   from row 0, 1
          v13 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp12), _mm_castps_pd(vtmp13)));  // b d f h   from row 0, 1

          _mm_store_ps(packed_w, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v8);
          _mm_store_ps(packed_w + 12, v12);
          _mm_store_ps(packed_w + 16, v1);
          _mm_store_ps(packed_w + 20, v5);
          _mm_store_ps(packed_w + 24, v9);
          _mm_store_ps(packed_w + 28, v13);
          packed_w += 32;
        }
        if (k & 1) {
          // Read blocks of 4x1
          // a
          // b
          // c
          // d
          __m128 v0 = _mm_load_ss(w0);  w0 += 1;
          __m128 v1 = _mm_load_ss(w1);  w1 += 1;
          __m128 v2 = _mm_load_ss(w2);  w2 += 1;
          __m128 v3 = _mm_load_ss(w3);  w3 += 1;
          __m128 v4 = _mm_load_ss(w4);  w4 += 1;
          __m128 v5 = _mm_load_ss(w5);  w5 += 1;
          __m128 v6 = _mm_load_ss(w6);  w6 += 1;
          __m128 v7 = _mm_load_ss(w7);  w7 += 1;
          __m128 v8 = _mm_load_ss(w8);  w8 += 1;
          __m128 v9 = _mm_load_ss(w9);  w9 += 1;
          __m128 v10 = _mm_load_ss(w10);  w10 += 1;
          __m128 v11 = _mm_load_ss(w11);  w11 += 1;
          __m128 v12 = _mm_load_ss(w12);  w12 += 1;
          __m128 v13 = _mm_load_ss(w13);  w13 += 1;
          __m128 v14 = _mm_load_ss(w14);  w14 += 1;

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
          const __m128 vtmp13 = _mm_unpacklo_ps(v14, v14);  // c d  from row 2, 3
          // Transpose 4x4
          v12 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp12), _mm_castps_pd(vtmp13)));  // a b c d   from row 0, 1

          _mm_store_ps(packed_w, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v8);
          _mm_store_ps(packed_w + 12, v12);
          packed_w += 16;
        }
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}
