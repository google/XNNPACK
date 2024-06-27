// Auto-generated file. Do not edit!
//   Template: src/x32-packw/c4-sse2.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <emmintrin.h>

#include "xnnpack/packw.h"
#include "xnnpack/prefetch.h"


void xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_u4_prfm(
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
  assert(nr == 2);
  assert(kr == 4);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);
  __m128 v0;
  __m128 v1;

  const float* b = (const float*) bias;
  float* packed_w = (float*) packed_weights;
  do {
    // NC main loop multiple of 2
    const float* w0 = (const float*) weights;
    size_t n = nc;

    for (; n >= 2; n -= 2) {
      if XNN_LIKELY(b != NULL) {
        packed_w[0] = b[0];
        packed_w[1] = b[1];
        b += 2;
      } else {
        packed_w[0] = 0.0f;
        packed_w[1] = 0.0f;
      }
      packed_w += 2;

      const float* w1 = w0 + kc;
      xnn_prefetch_to_l1((const int8_t*) w0);
      xnn_prefetch_to_l1((const int8_t*) w0 + 64);
      xnn_prefetch_to_l1((const int8_t*) w1);
      xnn_prefetch_to_l1((const int8_t*) w1 + 64);

      // KC main loop multiple of 2x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        // Read blocks of 2x4
        // a b c d
        // e f g h
        v0 = _mm_loadu_ps(w0);
        w0 += 4;
        v1 = _mm_loadu_ps(w1);
        w1 += 4;
        xnn_prefetch_to_l1((const int8_t*) w0 + 128);
        xnn_prefetch_to_l1((const int8_t*) w1 + 128);
        _mm_storeu_ps(packed_w, v0);
        _mm_storeu_ps(packed_w + 4, v1);
        packed_w += 8;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        switch (k) {
          case 1:
            // Read blocks of 2x1
            // a
            // e
            v0 = _mm_load_ss(w0);
            w0 += 1;
            v1 = _mm_load_ss(w1);
            w1 += 1;
            break;
          case 2:
            // Read blocks of 2x2
            // a b
            // e f
            v0 = _mm_castpd_ps(_mm_load_sd((const double*) w0));
            w0 += 2;
            v1 = _mm_castpd_ps(_mm_load_sd((const double*) w1));
            w1 += 2;
            break;
          case 3:
          {
            // Read blocks of 2x3
            // a b c
            // e f g
            const __m128 v0lo = _mm_castpd_ps(_mm_load_sd((const double*) w0));
            const __m128 v0hi = _mm_load_ss(w0 + 2);
            v0 = _mm_movelh_ps(v0lo, v0hi);
            w0 += 3;
            const __m128 v1lo = _mm_castpd_ps(_mm_load_sd((const double*) w1));
            const __m128 v1hi = _mm_load_ss(w1 + 2);
            v1 = _mm_movelh_ps(v1lo, v1hi);
            w1 += 3;
            break;
          }
          default:
            XNN_UNREACHABLE;
        }
        _mm_storeu_ps(packed_w, v0);
        _mm_storeu_ps(packed_w + 4, v1);
        packed_w += 8;
      }
      packed_w = (float*) ((uintptr_t) packed_w + extra_bytes);
      w0 = w1;
    }

    // NC remainder (1..1)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 1);
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *packed_w++  = *b++;
        } while (--nb != 0);
        packed_w += (2 - n);
      } else {
        packed_w[0] = 0.0f;
        packed_w[1] = 0.0f;
        packed_w += 2;
      }

      // NR remainder has less than 2 rows so last row is not loaded

      // KC main loop multiple of 2x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        // Read blocks of 2x4
        // a b c d
        // e f g h
        v0 = _mm_loadu_ps(w0);
        w0 += 4;
        xnn_prefetch_to_l1((const int8_t*) w0 + 128);
        _mm_storeu_ps(packed_w, v0);
        _mm_storeu_ps(packed_w + 4, v0);
        packed_w += 8;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        switch (k) {
          case 1:
            // Read blocks of 1x1
            // a
            v0 = _mm_load_ss(w0);
            w0 += 1;
            break;
          case 2:
            // Read blocks of 1x2
            // a b
            v0 = _mm_castpd_ps(_mm_load_sd((const double*) w0));
            w0 += 2;
            break;
          case 3:
          {
            // Read blocks of 1x3
            // a b c
            const __m128 v0lo = _mm_castpd_ps(_mm_load_sd((const double*) w0));
            const __m128 v0hi = _mm_load_ss(w0 + 2);
            v0 = _mm_movelh_ps(v0lo, v0hi);
            w0 += 3;
            break;
          }
          default:
            XNN_UNREACHABLE;
        }
        _mm_storeu_ps(packed_w, v0);
        _mm_storeu_ps(packed_w + 4, v0);
        packed_w += 8;
      }
      packed_w = (float*) ((uintptr_t) packed_w + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
