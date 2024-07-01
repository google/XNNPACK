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


void xnn_x32_packw_gemm_goi_ukernel_x8__avx_u4(
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
  assert(nr == 8);   // This kernel is for NR=8
  assert(kr == 1);
  assert(sr == 1);
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
        const __m256 vb0 = _mm256_loadu_ps(b);
        _mm256_store_ps(packed_w, vb0);
        b += 8;
      } else {
        const __m256 vzero = _mm256_setzero_ps();
        _mm256_store_ps(packed_w, vzero);
      }
      packed_w += 8;

      const float* w1 = w0 + kc;
      const float* w2 = w1 + kc;
      const float* w3 = w2 + kc;
      const float* w4 = w3 + kc;
      const float* w5 = w4 + kc;
      const float* w6 = w5 + kc;
      const float* w7 = w6 + kc;

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
        // Load next 4 rows of N into the high part of each register
        v0x0123 = _mm256_insertf128_ps(v0x0123, _mm_loadu_ps(w4), 1);
        w4 += 4;
        v1x0123 = _mm256_insertf128_ps(v1x0123, _mm_loadu_ps(w5), 1);
        w5 += 4;
        v2x0123 = _mm256_insertf128_ps(v2x0123, _mm_loadu_ps(w6), 1);
        w6 += 4;
        v3x0123 = _mm256_insertf128_ps(v3x0123, _mm_loadu_ps(w7), 1);
        w7 += 4;

        // Transpose 2x2
        const __m256 vtmp0x0123 = _mm256_unpacklo_ps(v0x0123, v1x0123);  // a e b f   from row 0, 1
        const __m256 vtmp1x0123 = _mm256_unpacklo_ps(v2x0123, v3x0123);  // i m j n   from row 2, 3
        const __m256 vtmp2x0123 = _mm256_unpackhi_ps(v0x0123, v1x0123);  // c g d h   from row 0, 1
        const __m256 vtmp3x0123 = _mm256_unpackhi_ps(v2x0123, v3x0123);  // k o l p   from row 2, 3
        // Transpose 4x4
        v0x0123 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(vtmp0x0123), _mm256_castps_pd(vtmp1x0123)));  // a e i m   from row 0, 1
        v1x0123 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(vtmp0x0123), _mm256_castps_pd(vtmp1x0123)));  // b f j n   from row 0, 1
        v2x0123 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(vtmp2x0123), _mm256_castps_pd(vtmp3x0123)));  // c g k o   from row 2, 3
        v3x0123 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(vtmp2x0123), _mm256_castps_pd(vtmp3x0123)));  // d h l p   from row 2, 3

        _mm256_store_ps(packed_w, v0x0123);
        _mm256_store_ps(packed_w + 8, v1x0123);
        _mm256_store_ps(packed_w + 16, v2x0123);
        _mm256_store_ps(packed_w + 24, v3x0123);
        packed_w += 32;
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

          _mm_store_ps(packed_w, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v1);
          _mm_store_ps(packed_w + 12, v5);
          packed_w += 16;
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

          _mm_store_ps(packed_w, v0);
          _mm_store_ps(packed_w + 4, v4);
          packed_w += 8;
        }
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
        const __m256 vzero = _mm256_setzero_ps();
        _mm256_store_ps(packed_w, vzero);
        packed_w += 8;
      }

      // NR remainder has less than 8 rows so last row is not loaded
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
        // castps leaves upper 128 bits undefined, so zero them.
        __m256 v3x0123 = _mm256_zextps128_ps256(_mm_loadu_ps(w3));
        w3 += 4;
        // Load next 4 rows of N into the high part of each register
        v0x0123 = _mm256_insertf128_ps(v0x0123, _mm_loadu_ps(w4), 1);
        w4 += 4;
        v1x0123 = _mm256_insertf128_ps(v1x0123, _mm_loadu_ps(w5), 1);
        w5 += 4;
        v2x0123 = _mm256_insertf128_ps(v2x0123, _mm_loadu_ps(w6), 1);
        w6 += 4;

        // Transpose 2x2
        const __m256 vtmp0x0123 = _mm256_unpacklo_ps(v0x0123, v1x0123);  // a e b f   from row 0, 1
        const __m256 vtmp1x0123 = _mm256_unpacklo_ps(v2x0123, v3x0123);  // i m j n   from row 2, 3
        const __m256 vtmp2x0123 = _mm256_unpackhi_ps(v0x0123, v1x0123);  // c g d h   from row 0, 1
        const __m256 vtmp3x0123 = _mm256_unpackhi_ps(v2x0123, v3x0123);  // k o l p   from row 2, 3
        // Transpose 4x4
        v0x0123 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(vtmp0x0123), _mm256_castps_pd(vtmp1x0123)));  // a e i m   from row 0, 1
        v1x0123 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(vtmp0x0123), _mm256_castps_pd(vtmp1x0123)));  // b f j n   from row 0, 1
        v2x0123 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(vtmp2x0123), _mm256_castps_pd(vtmp3x0123)));  // c g k o   from row 2, 3
        v3x0123 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(vtmp2x0123), _mm256_castps_pd(vtmp3x0123)));  // d h l p   from row 2, 3

        _mm256_store_ps(packed_w, v0x0123);
        _mm256_store_ps(packed_w + 8, v1x0123);
        _mm256_store_ps(packed_w + 16, v2x0123);
        _mm256_store_ps(packed_w + 24, v3x0123);
        packed_w += 32;
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

          // Transpose 2x2
          const __m128 vtmp0 = _mm_unpacklo_ps(v0, v1);  // a c b d   from row 0, 1
          const __m128 vtmp1 = _mm_unpacklo_ps(v2, v3);  // e g f h   from row 2, 3
          // Transpose 4x4
          v0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp0), _mm_castps_pd(vtmp1)));  // a c e g   from row 0, 1
          v1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp0), _mm_castps_pd(vtmp1)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp4 = _mm_unpacklo_ps(v4, v5);  // a c b d   from row 0, 1
          const __m128 vtmp5 = _mm_unpacklo_ps(v6, v6);  // e g f h   from row 2, 3
          // Transpose 4x4
          v4 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp4), _mm_castps_pd(vtmp5)));  // a c e g   from row 0, 1
          v5 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp4), _mm_castps_pd(vtmp5)));  // b d f h   from row 0, 1

          _mm_store_ps(packed_w, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v1);
          _mm_store_ps(packed_w + 12, v5);
          packed_w += 16;
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

          // Transpose 2x2
          const __m128 vtmp0 = _mm_unpacklo_ps(v0, v1);  // a b  from row 0, 1
          const __m128 vtmp1 = _mm_unpacklo_ps(v2, v3);  // c d  from row 2, 3
          // Transpose 4x4
          v0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp0), _mm_castps_pd(vtmp1)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp4 = _mm_unpacklo_ps(v4, v5);  // a b  from row 0, 1
          const __m128 vtmp5 = _mm_unpacklo_ps(v6, v6);  // c d  from row 2, 3
          // Transpose 4x4
          v4 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp4), _mm_castps_pd(vtmp5)));  // a b c d   from row 0, 1

          _mm_store_ps(packed_w, v0);
          _mm_store_ps(packed_w + 4, v4);
          packed_w += 8;
        }
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}
