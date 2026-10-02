// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x8-packw/avx2.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/packw.h"
#include "src/xnnpack/unaligned.h"
#include "src/xnnpack/prefetch.h"

void xnn_x8_packw_gemm_goi_ukernel_x8__avx2_u16_prfm(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t n_stride,
  const int8_t* weights,
  const uint32_t* bias,
  const void* scale,
  int8_t* packed_weights,
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

  int8_t* out = (int8_t*) packed_weights;
  const uint32_t* b = (const uint32_t*) bias;

  do {
    const int8_t* wb = weights;
    size_t n = nc;
    // NC main loop multiple of 8
    for (; n >= 8; n -= 8) {
      const int8_t* w0 = wb;
      const int8_t* w1 = w0 + n_stride;
      const int8_t* w2 = w1 + n_stride;
      const int8_t* w3 = w2 + n_stride;
      const int8_t* w4 = w3 + n_stride;
      const int8_t* w5 = w4 + n_stride;
      const int8_t* w6 = w5 + n_stride;
      const int8_t* w7 = w6 + n_stride;

      uint32_t* packed_b = (uint32_t*) out;
      if XNN_LIKELY(b != NULL) {
        const __m256i vb0 = _mm256_loadu_si256((const __m256i*) (b + 0));
        _mm256_storeu_si256((__m256i*) (packed_b + 0), vb0);
        b += 8;
      } else {
        const __m256i vzero = _mm256_setzero_si256();
        _mm256_storeu_si256((__m256i*) (packed_b + 0), vzero);
      }
      out += 8 * sizeof(uint32_t);

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

      // KC main loop multiple of 16
      size_t k = kc;
      for (; k >= 16; k -= 16) {
        const __m128i v0 = _mm_loadu_si128((const __m128i*) w0);
        w0 += 16;
        const __m128i v1 = _mm_loadu_si128((const __m128i*) w1);
        w1 += 16;
        const __m128i v2 = _mm_loadu_si128((const __m128i*) w2);
        w2 += 16;
        const __m128i v3 = _mm_loadu_si128((const __m128i*) w3);
        w3 += 16;
        const __m128i v4 = _mm_loadu_si128((const __m128i*) w4);
        w4 += 16;
        const __m128i v5 = _mm_loadu_si128((const __m128i*) w5);
        w5 += 16;
        const __m128i v6 = _mm_loadu_si128((const __m128i*) w6);
        w6 += 16;
        const __m128i v7 = _mm_loadu_si128((const __m128i*) w7);
        w7 += 16;
        xnn_prefetch_to_l1((const int8_t*) w0 + 128);
        xnn_prefetch_to_l1((const int8_t*) w1 + 128);
        xnn_prefetch_to_l1((const int8_t*) w2 + 128);
        xnn_prefetch_to_l1((const int8_t*) w3 + 128);
        xnn_prefetch_to_l1((const int8_t*) w4 + 128);
        xnn_prefetch_to_l1((const int8_t*) w5 + 128);
        xnn_prefetch_to_l1((const int8_t*) w6 + 128);
        xnn_prefetch_to_l1((const int8_t*) w7 + 128);

        const __m128i t0 = _mm_unpacklo_epi8(v0, v1);
        const __m128i t1 = _mm_unpackhi_epi8(v0, v1);
        const __m128i t2 = _mm_unpacklo_epi8(v2, v3);
        const __m128i t3 = _mm_unpackhi_epi8(v2, v3);
        const __m128i t4 = _mm_unpacklo_epi8(v4, v5);
        const __m128i t5 = _mm_unpackhi_epi8(v4, v5);
        const __m128i t6 = _mm_unpacklo_epi8(v6, v7);
        const __m128i t7 = _mm_unpackhi_epi8(v6, v7);

        const __m128i u0 = _mm_unpacklo_epi16(t0, t2);
        const __m128i u1 = _mm_unpackhi_epi16(t0, t2);
        const __m128i u2 = _mm_unpacklo_epi16(t1, t3);
        const __m128i u3 = _mm_unpackhi_epi16(t1, t3);
        const __m128i u4 = _mm_unpacklo_epi16(t4, t6);
        const __m128i u5 = _mm_unpackhi_epi16(t4, t6);
        const __m128i u6 = _mm_unpacklo_epi16(t5, t7);
        const __m128i u7 = _mm_unpackhi_epi16(t5, t7);

        const __m128i s0 = _mm_unpacklo_epi32(u0, u4);
        const __m128i s1 = _mm_unpackhi_epi32(u0, u4);
        const __m128i s2 = _mm_unpacklo_epi32(u1, u5);
        const __m128i s3 = _mm_unpackhi_epi32(u1, u5);
        const __m128i s4 = _mm_unpacklo_epi32(u2, u6);
        const __m128i s5 = _mm_unpackhi_epi32(u2, u6);
        const __m128i s6 = _mm_unpacklo_epi32(u3, u7);
        const __m128i s7 = _mm_unpackhi_epi32(u3, u7);

        _mm_storeu_si128((__m128i*) (out + 0), s0);
        _mm_storeu_si128((__m128i*) (out + 16), s1);
        _mm_storeu_si128((__m128i*) (out + 32), s2);
        _mm_storeu_si128((__m128i*) (out + 48), s3);
        _mm_storeu_si128((__m128i*) (out + 64), s4);
        _mm_storeu_si128((__m128i*) (out + 80), s5);
        _mm_storeu_si128((__m128i*) (out + 96), s6);
        _mm_storeu_si128((__m128i*) (out + 112), s7);
        out += 128;
      }

      // KC remainder (1..15)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 15);

        if (k & 8) {
          const __m128i v0 = _mm_loadl_epi64((const __m128i*) w0);
          w0 += 8;
          const __m128i v1 = _mm_loadl_epi64((const __m128i*) w1);
          w1 += 8;
          const __m128i v2 = _mm_loadl_epi64((const __m128i*) w2);
          w2 += 8;
          const __m128i v3 = _mm_loadl_epi64((const __m128i*) w3);
          w3 += 8;
          const __m128i v4 = _mm_loadl_epi64((const __m128i*) w4);
          w4 += 8;
          const __m128i v5 = _mm_loadl_epi64((const __m128i*) w5);
          w5 += 8;
          const __m128i v6 = _mm_loadl_epi64((const __m128i*) w6);
          w6 += 8;
          const __m128i v7 = _mm_loadl_epi64((const __m128i*) w7);
          w7 += 8;

          const __m128i t0 = _mm_unpacklo_epi8(v0, v1);
          const __m128i t2 = _mm_unpacklo_epi8(v2, v3);
          const __m128i t4 = _mm_unpacklo_epi8(v4, v5);
          const __m128i t6 = _mm_unpacklo_epi8(v6, v7);

          const __m128i u0 = _mm_unpacklo_epi16(t0, t2);
          const __m128i u1 = _mm_unpackhi_epi16(t0, t2);
          const __m128i u4 = _mm_unpacklo_epi16(t4, t6);
          const __m128i u5 = _mm_unpackhi_epi16(t4, t6);

          const __m128i s0 = _mm_unpacklo_epi32(u0, u4);
          const __m128i s1 = _mm_unpackhi_epi32(u0, u4);
          const __m128i s2 = _mm_unpacklo_epi32(u1, u5);
          const __m128i s3 = _mm_unpackhi_epi32(u1, u5);

          _mm_storeu_si128((__m128i*) (out + 0), s0);
          _mm_storeu_si128((__m128i*) (out + 16), s1);
          _mm_storeu_si128((__m128i*) (out + 32), s2);
          _mm_storeu_si128((__m128i*) (out + 48), s3);
          out += 64;
        }

        if (k & 4) {
          const __m128i v0 = _mm_cvtsi32_si128((int) unaligned_load_u32(w0));
          w0 += 4;
          const __m128i v1 = _mm_cvtsi32_si128((int) unaligned_load_u32(w1));
          w1 += 4;
          const __m128i v2 = _mm_cvtsi32_si128((int) unaligned_load_u32(w2));
          w2 += 4;
          const __m128i v3 = _mm_cvtsi32_si128((int) unaligned_load_u32(w3));
          w3 += 4;
          const __m128i v4 = _mm_cvtsi32_si128((int) unaligned_load_u32(w4));
          w4 += 4;
          const __m128i v5 = _mm_cvtsi32_si128((int) unaligned_load_u32(w5));
          w5 += 4;
          const __m128i v6 = _mm_cvtsi32_si128((int) unaligned_load_u32(w6));
          w6 += 4;
          const __m128i v7 = _mm_cvtsi32_si128((int) unaligned_load_u32(w7));
          w7 += 4;

          const __m128i t0 = _mm_unpacklo_epi8(v0, v1);
          const __m128i t2 = _mm_unpacklo_epi8(v2, v3);
          const __m128i t4 = _mm_unpacklo_epi8(v4, v5);
          const __m128i t6 = _mm_unpacklo_epi8(v6, v7);

          const __m128i u0 = _mm_unpacklo_epi16(t0, t2);
          const __m128i u4 = _mm_unpacklo_epi16(t4, t6);

          const __m128i s0 = _mm_unpacklo_epi32(u0, u4);
          const __m128i s1 = _mm_unpackhi_epi32(u0, u4);

          _mm_storeu_si128((__m128i*) (out + 0), s0);
          _mm_storeu_si128((__m128i*) (out + 16), s1);
          out += 32;
        }

        if (k & 2) {
          const __m128i v0 = _mm_cvtsi32_si128((int) unaligned_load_u16(w0));
          w0 += 2;
          const __m128i v1 = _mm_cvtsi32_si128((int) unaligned_load_u16(w1));
          w1 += 2;
          const __m128i v2 = _mm_cvtsi32_si128((int) unaligned_load_u16(w2));
          w2 += 2;
          const __m128i v3 = _mm_cvtsi32_si128((int) unaligned_load_u16(w3));
          w3 += 2;
          const __m128i v4 = _mm_cvtsi32_si128((int) unaligned_load_u16(w4));
          w4 += 2;
          const __m128i v5 = _mm_cvtsi32_si128((int) unaligned_load_u16(w5));
          w5 += 2;
          const __m128i v6 = _mm_cvtsi32_si128((int) unaligned_load_u16(w6));
          w6 += 2;
          const __m128i v7 = _mm_cvtsi32_si128((int) unaligned_load_u16(w7));
          w7 += 2;

          const __m128i t0 = _mm_unpacklo_epi8(v0, v1);
          const __m128i t2 = _mm_unpacklo_epi8(v2, v3);
          const __m128i t4 = _mm_unpacklo_epi8(v4, v5);
          const __m128i t6 = _mm_unpacklo_epi8(v6, v7);

          const __m128i u0 = _mm_unpacklo_epi16(t0, t2);
          const __m128i u4 = _mm_unpacklo_epi16(t4, t6);

          const __m128i s0 = _mm_unpacklo_epi32(u0, u4);

          _mm_storeu_si128((__m128i*) out, s0);
          out += 16;
        }

        if (k & 1) {
          out[0] = *w0++;
          out[1] = *w1++;
          out[2] = *w2++;
          out[3] = *w3++;
          out[4] = *w4++;
          out[5] = *w5++;
          out[6] = *w6++;
          out[7] = *w7++;
          out += 8;
        }
      }

      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      wb += 8 * n_stride;
    }
    // NC remainder (1..7)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 7);
      const int8_t* w0 = wb;
      const int8_t* w1 = w0 + n_stride;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const int8_t* w2 = w1 + n_stride;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const int8_t* w3 = w2 + n_stride;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const int8_t* w4 = w3 + n_stride;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const int8_t* w5 = w4 + n_stride;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const int8_t* w6 = w5 + n_stride;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const int8_t* w7 = w6 + n_stride;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }

      uint32_t* packed_b = (uint32_t*) out;
      const __m256i vzero = _mm256_setzero_si256();
      _mm256_storeu_si256((__m256i*) (packed_b + 0), vzero);
      if XNN_LIKELY(b != NULL) {
        for (size_t nb = 0; nb < n; ++nb) {
          packed_b[nb] = b[nb];
        }
        b += n;
      }
      out += 8 * sizeof(uint32_t);

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

      // KC main loop multiple of 16
      size_t k = kc;
      for (; k >= 16; k -= 16) {
        const __m128i v0 = _mm_loadu_si128((const __m128i*) w0);
        w0 += 16;
        const __m128i v1 = _mm_loadu_si128((const __m128i*) w1);
        w1 += 16;
        const __m128i v2 = _mm_loadu_si128((const __m128i*) w2);
        w2 += 16;
        const __m128i v3 = _mm_loadu_si128((const __m128i*) w3);
        w3 += 16;
        const __m128i v4 = _mm_loadu_si128((const __m128i*) w4);
        w4 += 16;
        const __m128i v5 = _mm_loadu_si128((const __m128i*) w5);
        w5 += 16;
        const __m128i v6 = _mm_loadu_si128((const __m128i*) w6);
        w6 += 16;
        const __m128i v7 = _mm_loadu_si128((const __m128i*) w7);
        w7 += 16;
        xnn_prefetch_to_l1((const int8_t*) w0 + 128);
        xnn_prefetch_to_l1((const int8_t*) w1 + 128);
        xnn_prefetch_to_l1((const int8_t*) w2 + 128);
        xnn_prefetch_to_l1((const int8_t*) w3 + 128);
        xnn_prefetch_to_l1((const int8_t*) w4 + 128);
        xnn_prefetch_to_l1((const int8_t*) w5 + 128);
        xnn_prefetch_to_l1((const int8_t*) w6 + 128);
        xnn_prefetch_to_l1((const int8_t*) w7 + 128);

        const __m128i t0 = _mm_unpacklo_epi8(v0, v1);
        const __m128i t1 = _mm_unpackhi_epi8(v0, v1);
        const __m128i t2 = _mm_unpacklo_epi8(v2, v3);
        const __m128i t3 = _mm_unpackhi_epi8(v2, v3);
        const __m128i t4 = _mm_unpacklo_epi8(v4, v5);
        const __m128i t5 = _mm_unpackhi_epi8(v4, v5);
        const __m128i t6 = _mm_unpacklo_epi8(v6, v7);
        const __m128i t7 = _mm_unpackhi_epi8(v6, v7);

        const __m128i u0 = _mm_unpacklo_epi16(t0, t2);
        const __m128i u1 = _mm_unpackhi_epi16(t0, t2);
        const __m128i u2 = _mm_unpacklo_epi16(t1, t3);
        const __m128i u3 = _mm_unpackhi_epi16(t1, t3);
        const __m128i u4 = _mm_unpacklo_epi16(t4, t6);
        const __m128i u5 = _mm_unpackhi_epi16(t4, t6);
        const __m128i u6 = _mm_unpacklo_epi16(t5, t7);
        const __m128i u7 = _mm_unpackhi_epi16(t5, t7);

        const __m128i s0 = _mm_unpacklo_epi32(u0, u4);
        const __m128i s1 = _mm_unpackhi_epi32(u0, u4);
        const __m128i s2 = _mm_unpacklo_epi32(u1, u5);
        const __m128i s3 = _mm_unpackhi_epi32(u1, u5);
        const __m128i s4 = _mm_unpacklo_epi32(u2, u6);
        const __m128i s5 = _mm_unpackhi_epi32(u2, u6);
        const __m128i s6 = _mm_unpacklo_epi32(u3, u7);
        const __m128i s7 = _mm_unpackhi_epi32(u3, u7);

        _mm_storeu_si128((__m128i*) (out + 0), s0);
        _mm_storeu_si128((__m128i*) (out + 16), s1);
        _mm_storeu_si128((__m128i*) (out + 32), s2);
        _mm_storeu_si128((__m128i*) (out + 48), s3);
        _mm_storeu_si128((__m128i*) (out + 64), s4);
        _mm_storeu_si128((__m128i*) (out + 80), s5);
        _mm_storeu_si128((__m128i*) (out + 96), s6);
        _mm_storeu_si128((__m128i*) (out + 112), s7);
        out += 128;
      }

      // KC remainder (1..15)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 15);

        if (k & 8) {
          const __m128i v0 = _mm_loadl_epi64((const __m128i*) w0);
          w0 += 8;
          const __m128i v1 = _mm_loadl_epi64((const __m128i*) w1);
          w1 += 8;
          const __m128i v2 = _mm_loadl_epi64((const __m128i*) w2);
          w2 += 8;
          const __m128i v3 = _mm_loadl_epi64((const __m128i*) w3);
          w3 += 8;
          const __m128i v4 = _mm_loadl_epi64((const __m128i*) w4);
          w4 += 8;
          const __m128i v5 = _mm_loadl_epi64((const __m128i*) w5);
          w5 += 8;
          const __m128i v6 = _mm_loadl_epi64((const __m128i*) w6);
          w6 += 8;
          const __m128i v7 = _mm_loadl_epi64((const __m128i*) w7);
          w7 += 8;

          const __m128i t0 = _mm_unpacklo_epi8(v0, v1);
          const __m128i t2 = _mm_unpacklo_epi8(v2, v3);
          const __m128i t4 = _mm_unpacklo_epi8(v4, v5);
          const __m128i t6 = _mm_unpacklo_epi8(v6, v7);

          const __m128i u0 = _mm_unpacklo_epi16(t0, t2);
          const __m128i u1 = _mm_unpackhi_epi16(t0, t2);
          const __m128i u4 = _mm_unpacklo_epi16(t4, t6);
          const __m128i u5 = _mm_unpackhi_epi16(t4, t6);

          const __m128i s0 = _mm_unpacklo_epi32(u0, u4);
          const __m128i s1 = _mm_unpackhi_epi32(u0, u4);
          const __m128i s2 = _mm_unpacklo_epi32(u1, u5);
          const __m128i s3 = _mm_unpackhi_epi32(u1, u5);

          _mm_storeu_si128((__m128i*) (out + 0), s0);
          _mm_storeu_si128((__m128i*) (out + 16), s1);
          _mm_storeu_si128((__m128i*) (out + 32), s2);
          _mm_storeu_si128((__m128i*) (out + 48), s3);
          out += 64;
        }

        if (k & 4) {
          const __m128i v0 = _mm_cvtsi32_si128((int) unaligned_load_u32(w0));
          w0 += 4;
          const __m128i v1 = _mm_cvtsi32_si128((int) unaligned_load_u32(w1));
          w1 += 4;
          const __m128i v2 = _mm_cvtsi32_si128((int) unaligned_load_u32(w2));
          w2 += 4;
          const __m128i v3 = _mm_cvtsi32_si128((int) unaligned_load_u32(w3));
          w3 += 4;
          const __m128i v4 = _mm_cvtsi32_si128((int) unaligned_load_u32(w4));
          w4 += 4;
          const __m128i v5 = _mm_cvtsi32_si128((int) unaligned_load_u32(w5));
          w5 += 4;
          const __m128i v6 = _mm_cvtsi32_si128((int) unaligned_load_u32(w6));
          w6 += 4;
          const __m128i v7 = _mm_cvtsi32_si128((int) unaligned_load_u32(w7));
          w7 += 4;

          const __m128i t0 = _mm_unpacklo_epi8(v0, v1);
          const __m128i t2 = _mm_unpacklo_epi8(v2, v3);
          const __m128i t4 = _mm_unpacklo_epi8(v4, v5);
          const __m128i t6 = _mm_unpacklo_epi8(v6, v7);

          const __m128i u0 = _mm_unpacklo_epi16(t0, t2);
          const __m128i u4 = _mm_unpacklo_epi16(t4, t6);

          const __m128i s0 = _mm_unpacklo_epi32(u0, u4);
          const __m128i s1 = _mm_unpackhi_epi32(u0, u4);

          _mm_storeu_si128((__m128i*) (out + 0), s0);
          _mm_storeu_si128((__m128i*) (out + 16), s1);
          out += 32;
        }

        if (k & 2) {
          const __m128i v0 = _mm_cvtsi32_si128((int) unaligned_load_u16(w0));
          w0 += 2;
          const __m128i v1 = _mm_cvtsi32_si128((int) unaligned_load_u16(w1));
          w1 += 2;
          const __m128i v2 = _mm_cvtsi32_si128((int) unaligned_load_u16(w2));
          w2 += 2;
          const __m128i v3 = _mm_cvtsi32_si128((int) unaligned_load_u16(w3));
          w3 += 2;
          const __m128i v4 = _mm_cvtsi32_si128((int) unaligned_load_u16(w4));
          w4 += 2;
          const __m128i v5 = _mm_cvtsi32_si128((int) unaligned_load_u16(w5));
          w5 += 2;
          const __m128i v6 = _mm_cvtsi32_si128((int) unaligned_load_u16(w6));
          w6 += 2;
          const __m128i v7 = _mm_cvtsi32_si128((int) unaligned_load_u16(w7));
          w7 += 2;

          const __m128i t0 = _mm_unpacklo_epi8(v0, v1);
          const __m128i t2 = _mm_unpacklo_epi8(v2, v3);
          const __m128i t4 = _mm_unpacklo_epi8(v4, v5);
          const __m128i t6 = _mm_unpacklo_epi8(v6, v7);

          const __m128i u0 = _mm_unpacklo_epi16(t0, t2);
          const __m128i u4 = _mm_unpacklo_epi16(t4, t6);

          const __m128i s0 = _mm_unpacklo_epi32(u0, u4);

          _mm_storeu_si128((__m128i*) out, s0);
          out += 16;
        }

        if (k & 1) {
          out[0] = *w0++;
          out[1] = *w1++;
          out[2] = *w2++;
          out[3] = *w3++;
          out[4] = *w4++;
          out[5] = *w5++;
          out[6] = *w6++;
          out[7] = *w7++;
          out += 8;
        }
      }

      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }

    weights += nc * n_stride;
  } while (--g != 0);
}
