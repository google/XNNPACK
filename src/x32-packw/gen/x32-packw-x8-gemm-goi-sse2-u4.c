// Auto-generated file. Do not edit!
//   Template: src/x32-packw/sse2.c.in
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

void xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u4(
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
        const __m128 vb0123 = _mm_loadu_ps(b);
        const __m128 vb4567 = _mm_loadu_ps(b + 4);
        b += 8;

        _mm_store_ps(packed_w, vb0123);
        _mm_store_ps(packed_w + 4, vb4567);
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

      size_t k = kc;

      // KC multiple of 4
      for (; k >= 4; k -= 4) {
        const __m128 v0x0123 = _mm_loadu_ps(w0);
        w0 += 4;
        const __m128 v1x0123 = _mm_loadu_ps(w1);
        w1 += 4;
        const __m128 v2x0123 = _mm_loadu_ps(w2);
        w2 += 4;
        const __m128 v3x0123 = _mm_loadu_ps(w3);
        w3 += 4;
        const __m128 v4x0123 = _mm_loadu_ps(w4);
        w4 += 4;
        const __m128 v5x0123 = _mm_loadu_ps(w5);
        w5 += 4;
        const __m128 v6x0123 = _mm_loadu_ps(w6);
        w6 += 4;
        const __m128 v7x0123 = _mm_loadu_ps(w7);
        w7 += 4;

        const __m128 v01x0_01x1 = _mm_unpacklo_ps(v0x0123, v1x0123);
        const __m128 v23x0_23x1 = _mm_unpacklo_ps(v2x0123, v3x0123);
        const __m128 v01x2_01x3 = _mm_unpackhi_ps(v0x0123, v1x0123);
        const __m128 v23x2_23x3 = _mm_unpackhi_ps(v2x0123, v3x0123);
        const __m128 v45x0_45x1 = _mm_unpacklo_ps(v4x0123, v5x0123);
        const __m128 v67x0_67x1 = _mm_unpacklo_ps(v6x0123, v7x0123);
        const __m128 v45x2_45x3 = _mm_unpackhi_ps(v4x0123, v5x0123);
        const __m128 v67x2_67x3 = _mm_unpackhi_ps(v6x0123, v7x0123);
        const __m128 v0123x0 = _mm_movelh_ps(v01x0_01x1, v23x0_23x1);
        const __m128 v0123x1 = _mm_movehl_ps(v23x0_23x1, v01x0_01x1);
        const __m128 v0123x2 = _mm_movelh_ps(v01x2_01x3, v23x2_23x3);
        const __m128 v0123x3 = _mm_movehl_ps(v23x2_23x3, v01x2_01x3);
        const __m128 v4567x0 = _mm_movelh_ps(v45x0_45x1, v67x0_67x1);
        const __m128 v4567x1 = _mm_movehl_ps(v67x0_67x1, v45x0_45x1);
        const __m128 v4567x2 = _mm_movelh_ps(v45x2_45x3, v67x2_67x3);
        const __m128 v4567x3 = _mm_movehl_ps(v67x2_67x3, v45x2_45x3);

        _mm_store_ps(packed_w, v0123x0);
        _mm_store_ps(packed_w + 4, v4567x0);
        _mm_store_ps(packed_w + 8, v0123x1);
        _mm_store_ps(packed_w + 12, v4567x1);
        _mm_store_ps(packed_w + 16, v0123x2);
        _mm_store_ps(packed_w + 20, v4567x2);
        _mm_store_ps(packed_w + 24, v0123x3);
        _mm_store_ps(packed_w + 28, v4567x3);
        packed_w += 32;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        switch (k) {
          case 1:
          {
            const __m128 v0x0 = _mm_load_ss(w0);
            w0 += 1;
            const __m128 v1x0 = _mm_load_ss(w1);
            w1 += 1;
            const __m128 v2x0 = _mm_load_ss(w2);
            w2 += 1;
            const __m128 v3x0 = _mm_load_ss(w3);
            w3 += 1;
            const __m128 v4x0 = _mm_load_ss(w4);
            w4 += 1;
            const __m128 v5x0 = _mm_load_ss(w5);
            w5 += 1;
            const __m128 v6x0 = _mm_load_ss(w6);
            w6 += 1;
            const __m128 v7x0 = _mm_load_ss(w7);
            w7 += 1;

            const __m128 v01x0 = _mm_unpacklo_ps(v0x0, v1x0);
            const __m128 v23x0 = _mm_unpacklo_ps(v2x0, v3x0);
            const __m128 v45x0 = _mm_unpacklo_ps(v4x0, v5x0);
            const __m128 v67x0 = _mm_unpacklo_ps(v6x0, v7x0);

            const __m128 v0123x0 = _mm_movelh_ps(v01x0, v23x0);
            const __m128 v4567x0 = _mm_movelh_ps(v45x0, v67x0);

            _mm_store_ps(packed_w, v0123x0);
            _mm_store_ps(packed_w + 4, v4567x0);
            packed_w += 8;
            break;
          }
          case 2:
          {
            const __m128 v0x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w0));
            w0 += 2;
            const __m128 v1x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w1));
            w1 += 2;
            const __m128 v2x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w2));
            w2 += 2;
            const __m128 v3x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w3));
            w3 += 2;
            const __m128 v4x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w4));
            w4 += 2;
            const __m128 v5x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w5));
            w5 += 2;
            const __m128 v6x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w6));
            w6 += 2;
            const __m128 v7x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w7));
            w7 += 2;

            const __m128 v01x0_01x1 = _mm_unpacklo_ps(v0x01, v1x01);
            const __m128 v23x0_23x1 = _mm_unpacklo_ps(v2x01, v3x01);
            const __m128 v45x0_45x1 = _mm_unpacklo_ps(v4x01, v5x01);
            const __m128 v67x0_67x1 = _mm_unpacklo_ps(v6x01, v7x01);

            const __m128 v0123x0 = _mm_movelh_ps(v01x0_01x1, v23x0_23x1);
            const __m128 v0123x1 = _mm_movehl_ps(v23x0_23x1, v01x0_01x1);
            const __m128 v4567x0 = _mm_movelh_ps(v45x0_45x1, v67x0_67x1);
            const __m128 v4567x1 = _mm_movehl_ps(v67x0_67x1, v45x0_45x1);

            _mm_store_ps(packed_w, v0123x0);
            _mm_store_ps(packed_w + 4, v4567x0);
            _mm_store_ps(packed_w + 8, v0123x1);
            _mm_store_ps(packed_w + 12, v4567x1);
            packed_w += 16;
            break;
          }
          case 3:
          {
            __m128 v0x012 = _mm_load_ss(w0 + 2);
            __m128 v1x012 = _mm_load_ss(w1 + 2);
            __m128 v2x012 = _mm_load_ss(w2 + 2);
            __m128 v3x012 = _mm_load_ss(w3 + 2);
            __m128 v4x012 = _mm_load_ss(w4 + 2);
            __m128 v5x012 = _mm_load_ss(w5 + 2);
            __m128 v6x012 = _mm_load_ss(w6 + 2);
            __m128 v7x012 = _mm_load_ss(w7 + 2);

            v0x012 = _mm_movelh_ps(v0x012, v0x012);
            v1x012 = _mm_movelh_ps(v1x012, v1x012);
            v2x012 = _mm_movelh_ps(v2x012, v2x012);
            v3x012 = _mm_movelh_ps(v3x012, v3x012);
            v4x012 = _mm_movelh_ps(v4x012, v4x012);
            v5x012 = _mm_movelh_ps(v5x012, v5x012);
            v6x012 = _mm_movelh_ps(v6x012, v6x012);
            v7x012 = _mm_movelh_ps(v7x012, v7x012);

            v0x012 = _mm_loadl_pi(v0x012, (const __m64*) w0);
            w0 += 3;
            v1x012 = _mm_loadl_pi(v1x012, (const __m64*) w1);
            w1 += 3;
            v2x012 = _mm_loadl_pi(v2x012, (const __m64*) w2);
            w2 += 3;
            v3x012 = _mm_loadl_pi(v3x012, (const __m64*) w3);
            w3 += 3;
            v4x012 = _mm_loadl_pi(v4x012, (const __m64*) w4);
            w4 += 3;
            v5x012 = _mm_loadl_pi(v5x012, (const __m64*) w5);
            w5 += 3;
            v6x012 = _mm_loadl_pi(v6x012, (const __m64*) w6);
            w6 += 3;
            v7x012 = _mm_loadl_pi(v7x012, (const __m64*) w7);
            w7 += 3;

            const __m128 v01x0_01x1 = _mm_unpacklo_ps(v0x012, v1x012);
            const __m128 v23x0_23x1 = _mm_unpacklo_ps(v2x012, v3x012);
            const __m128 v01x2 = _mm_unpackhi_ps(v0x012, v1x012);
            const __m128 v23x2 = _mm_unpackhi_ps(v2x012, v3x012);
            const __m128 v45x0_45x1 = _mm_unpacklo_ps(v4x012, v5x012);
            const __m128 v67x0_67x1 = _mm_unpacklo_ps(v6x012, v7x012);
            const __m128 v45x2 = _mm_unpackhi_ps(v4x012, v5x012);
            const __m128 v67x2 = _mm_unpackhi_ps(v6x012, v7x012);

            const __m128 v0123x0 = _mm_movelh_ps(v01x0_01x1, v23x0_23x1);
            const __m128 v0123x1 = _mm_movehl_ps(v23x0_23x1, v01x0_01x1);
            const __m128 v0123x2 = _mm_movelh_ps(v01x2, v23x2);
            const __m128 v4567x0 = _mm_movelh_ps(v45x0_45x1, v67x0_67x1);
            const __m128 v4567x1 = _mm_movehl_ps(v67x0_67x1, v45x0_45x1);
            const __m128 v4567x2 = _mm_movelh_ps(v45x2, v67x2);

            _mm_store_ps(packed_w, v0123x0);
            _mm_store_ps(packed_w + 4, v4567x0);
            _mm_store_ps(packed_w + 8, v0123x1);
            _mm_store_ps(packed_w + 12, v4567x1);
            _mm_store_ps(packed_w + 16, v0123x2);
            _mm_store_ps(packed_w + 20, v4567x2);
            packed_w += 24;
            break;
          }
          default:
            XNN_UNREACHABLE;
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
        const __m128 vzero = _mm_setzero_ps();
        _mm_store_ps(packed_w, vzero);
        _mm_store_ps(packed_w + 4, vzero);
        packed_w += 8;
      }

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
        const __m128 v0x0123 = _mm_loadu_ps(w0);
        w0 += 4;
        const __m128 v1x0123 = _mm_loadu_ps(w1);
        w1 += 4;
        const __m128 v2x0123 = _mm_loadu_ps(w2);
        w2 += 4;
        const __m128 v3x0123 = _mm_loadu_ps(w3);
        w3 += 4;
        const __m128 v4x0123 = _mm_loadu_ps(w4);
        w4 += 4;
        const __m128 v5x0123 = _mm_loadu_ps(w5);
        w5 += 4;
        const __m128 v6x0123 = _mm_loadu_ps(w6);
        w6 += 4;

        const __m128 v01x0_01x1 = _mm_unpacklo_ps(v0x0123, v1x0123);
        const __m128 v23x0_23x1 = _mm_unpacklo_ps(v2x0123, v3x0123);
        const __m128 v01x2_01x3 = _mm_unpackhi_ps(v0x0123, v1x0123);
        const __m128 v23x2_23x3 = _mm_unpackhi_ps(v2x0123, v3x0123);
        const __m128 v45x0_45x1 = _mm_unpacklo_ps(v4x0123, v5x0123);
        const __m128 v67x0_67x1 = _mm_unpacklo_ps(v6x0123, v6x0123);
        const __m128 v45x2_45x3 = _mm_unpackhi_ps(v4x0123, v5x0123);
        const __m128 v67x2_67x3 = _mm_unpackhi_ps(v6x0123, v6x0123);
        const __m128 v0123x0 = _mm_movelh_ps(v01x0_01x1, v23x0_23x1);
        const __m128 v0123x1 = _mm_movehl_ps(v23x0_23x1, v01x0_01x1);
        const __m128 v0123x2 = _mm_movelh_ps(v01x2_01x3, v23x2_23x3);
        const __m128 v0123x3 = _mm_movehl_ps(v23x2_23x3, v01x2_01x3);
        const __m128 v4567x0 = _mm_movelh_ps(v45x0_45x1, v67x0_67x1);
        const __m128 v4567x1 = _mm_movehl_ps(v67x0_67x1, v45x0_45x1);
        const __m128 v4567x2 = _mm_movelh_ps(v45x2_45x3, v67x2_67x3);
        const __m128 v4567x3 = _mm_movehl_ps(v67x2_67x3, v45x2_45x3);

        _mm_store_ps(packed_w, v0123x0);
        _mm_store_ps(packed_w + 4, v4567x0);
        _mm_store_ps(packed_w + 8, v0123x1);
        _mm_store_ps(packed_w + 12, v4567x1);
        _mm_store_ps(packed_w + 16, v0123x2);
        _mm_store_ps(packed_w + 20, v4567x2);
        _mm_store_ps(packed_w + 24, v0123x3);
        _mm_store_ps(packed_w + 28, v4567x3);
        packed_w += 32;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        switch (k) {
          case 1:
          {
            const __m128 v0x0 = _mm_load_ss(w0);
            const __m128 v1x0 = _mm_load_ss(w1);
            const __m128 v2x0 = _mm_load_ss(w2);
            const __m128 v3x0 = _mm_load_ss(w3);
            const __m128 v4x0 = _mm_load_ss(w4);
            const __m128 v5x0 = _mm_load_ss(w5);
            const __m128 v6x0 = _mm_load_ss(w6);

            const __m128 v01x0 = _mm_unpacklo_ps(v0x0, v1x0);
            const __m128 v23x0 = _mm_unpacklo_ps(v2x0, v3x0);
            const __m128 v45x0 = _mm_unpacklo_ps(v4x0, v5x0);
            const __m128 v67x0 = _mm_unpacklo_ps(v6x0, v6x0);

            const __m128 v0123x0 = _mm_movelh_ps(v01x0, v23x0);
            const __m128 v4567x0 = _mm_movelh_ps(v45x0, v67x0);

            _mm_store_ps(packed_w, v0123x0);
            _mm_store_ps(packed_w + 4, v4567x0);
            packed_w += 8;
            break;
          }
          case 2:
          {
            const __m128 v0x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w0));
            const __m128 v1x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w1));
            const __m128 v2x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w2));
            const __m128 v3x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w3));
            const __m128 v4x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w4));
            const __m128 v5x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w5));
            const __m128 v6x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w6));

            const __m128 v01x0_01x1 = _mm_unpacklo_ps(v0x01, v1x01);
            const __m128 v23x0_23x1 = _mm_unpacklo_ps(v2x01, v3x01);
            const __m128 v45x0_45x1 = _mm_unpacklo_ps(v4x01, v5x01);
            const __m128 v67x0_67x1 = _mm_unpacklo_ps(v6x01, v6x01);

            const __m128 v0123x0 = _mm_movelh_ps(v01x0_01x1, v23x0_23x1);
            const __m128 v0123x1 = _mm_movehl_ps(v23x0_23x1, v01x0_01x1);
            const __m128 v4567x0 = _mm_movelh_ps(v45x0_45x1, v67x0_67x1);
            const __m128 v4567x1 = _mm_movehl_ps(v67x0_67x1, v45x0_45x1);

            _mm_store_ps(packed_w, v0123x0);
            _mm_store_ps(packed_w + 4, v4567x0);
            _mm_store_ps(packed_w + 8, v0123x1);
            _mm_store_ps(packed_w + 12, v4567x1);
            packed_w += 16;
            break;
          }
          case 3:
          {
            __m128 v0x012 = _mm_load_ss(w0 + 2);
            __m128 v1x012 = _mm_load_ss(w1 + 2);
            __m128 v2x012 = _mm_load_ss(w2 + 2);
            __m128 v3x012 = _mm_load_ss(w3 + 2);
            __m128 v4x012 = _mm_load_ss(w4 + 2);
            __m128 v5x012 = _mm_load_ss(w5 + 2);
            __m128 v6x012 = _mm_load_ss(w6 + 2);

            v0x012 = _mm_movelh_ps(v0x012, v0x012);
            v1x012 = _mm_movelh_ps(v1x012, v1x012);
            v2x012 = _mm_movelh_ps(v2x012, v2x012);
            v3x012 = _mm_movelh_ps(v3x012, v3x012);
            v4x012 = _mm_movelh_ps(v4x012, v4x012);
            v5x012 = _mm_movelh_ps(v5x012, v5x012);
            v6x012 = _mm_movelh_ps(v6x012, v6x012);

            v0x012 = _mm_loadl_pi(v0x012, (const __m64*) w0);
            v1x012 = _mm_loadl_pi(v1x012, (const __m64*) w1);
            v2x012 = _mm_loadl_pi(v2x012, (const __m64*) w2);
            v3x012 = _mm_loadl_pi(v3x012, (const __m64*) w3);
            v4x012 = _mm_loadl_pi(v4x012, (const __m64*) w4);
            v5x012 = _mm_loadl_pi(v5x012, (const __m64*) w5);
            v6x012 = _mm_loadl_pi(v6x012, (const __m64*) w6);

            const __m128 v01x0_01x1 = _mm_unpacklo_ps(v0x012, v1x012);
            const __m128 v23x0_23x1 = _mm_unpacklo_ps(v2x012, v3x012);
            const __m128 v01x2 = _mm_unpackhi_ps(v0x012, v1x012);
            const __m128 v23x2 = _mm_unpackhi_ps(v2x012, v3x012);
            const __m128 v45x0_45x1 = _mm_unpacklo_ps(v4x012, v5x012);
            const __m128 v67x0_67x1 = _mm_unpacklo_ps(v6x012, v6x012);
            const __m128 v45x2 = _mm_unpackhi_ps(v4x012, v5x012);
            const __m128 v67x2 = _mm_unpackhi_ps(v6x012, v6x012);

            const __m128 v0123x0 = _mm_movelh_ps(v01x0_01x1, v23x0_23x1);
            const __m128 v0123x1 = _mm_movehl_ps(v23x0_23x1, v01x0_01x1);
            const __m128 v0123x2 = _mm_movelh_ps(v01x2, v23x2);
            const __m128 v4567x0 = _mm_movelh_ps(v45x0_45x1, v67x0_67x1);
            const __m128 v4567x1 = _mm_movehl_ps(v67x0_67x1, v45x0_45x1);
            const __m128 v4567x2 = _mm_movelh_ps(v45x2, v67x2);

            _mm_store_ps(packed_w, v0123x0);
            _mm_store_ps(packed_w + 4, v4567x0);
            _mm_store_ps(packed_w + 8, v0123x1);
            _mm_store_ps(packed_w + 12, v4567x1);
            _mm_store_ps(packed_w + 16, v0123x2);
            _mm_store_ps(packed_w + 20, v4567x2);
            packed_w += 24;
            break;
          }
          default:
            XNN_UNREACHABLE;
        }
      }
      packed_w = (float*) ((uintptr_t) packed_w + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
