// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x16-packw/c2-avx512skx.c.in
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
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/packw.h"


void xnn_x16_packw_gemm_goi_ukernel_x16c2__avx512skx_u16(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t n_stride,
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
  assert(nr == 16);   // This kernel is for NR=16
  assert(kr == 2);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const uint16_t* b = bias;
  uint16_t* packed_w = packed_weights;
  do {
    // NC main loop multiple of 16
    const uint16_t* w_base = weights;
    size_t n = nc;

    for (; n >= 16; n -= 16) {
      if XNN_LIKELY(b != NULL) {
        const __m256i vb0 = _mm256_loadu_si256((const __m256i*) b);
        _mm256_storeu_si256((__m256i*) packed_w, vb0);
        b += 16;
      } else {
        const __m256i vzero = _mm256_setzero_si256();
        _mm256_storeu_si256((__m256i*) packed_w, vzero);
      }
      packed_w += 16;

      const ptrdiff_t stride1 = (ptrdiff_t) n_stride;
      const ptrdiff_t stride2 = stride1 * 2;
      const ptrdiff_t stride3 = stride1 * 3;
      const ptrdiff_t stride4 = stride1 * 4;
      const ptrdiff_t stride5 = stride1 * 5;
      const ptrdiff_t stride6 = stride1 * 6;
      const ptrdiff_t stride7 = stride1 * 7;

      const uint16_t* w0 = w_base;
      const uint16_t* w8 = w0 + 8 * stride1;

      // KC main loop multiple of 16
      size_t k = kc;
      for (; k >= 16; k -= 16) {
        const __m256i v0 = _mm256_loadu_si256((const __m256i*) w0);
        const __m256i v4 = _mm256_loadu_si256((const __m256i*) (w0 + stride4));
        const __m256i v8 = _mm256_loadu_si256((const __m256i*) w8);
        const __m256i v12 = _mm256_loadu_si256((const __m256i*) (w8 + stride4));
        const __m512i z0 = _mm512_inserti64x4(_mm512_castsi256_si512(v0), v4, 1);
        const __m512i z4 = _mm512_inserti64x4(_mm512_castsi256_si512(v8), v12, 1);
        const __m256i v1 = _mm256_loadu_si256((const __m256i*) (w0 + stride1));
        const __m256i v5 = _mm256_loadu_si256((const __m256i*) (w0 + stride5));
        const __m256i v9 = _mm256_loadu_si256((const __m256i*) (w8 + stride1));
        const __m256i v13 = _mm256_loadu_si256((const __m256i*) (w8 + stride5));
        const __m512i z1 = _mm512_inserti64x4(_mm512_castsi256_si512(v1), v5, 1);
        const __m512i z5 = _mm512_inserti64x4(_mm512_castsi256_si512(v9), v13, 1);
        const __m256i v2 = _mm256_loadu_si256((const __m256i*) (w0 + stride2));
        const __m256i v6 = _mm256_loadu_si256((const __m256i*) (w0 + stride6));
        const __m256i v10 = _mm256_loadu_si256((const __m256i*) (w8 + stride2));
        const __m256i v14 = _mm256_loadu_si256((const __m256i*) (w8 + stride6));
        const __m512i z2 = _mm512_inserti64x4(_mm512_castsi256_si512(v2), v6, 1);
        const __m512i z6 = _mm512_inserti64x4(_mm512_castsi256_si512(v10), v14, 1);
        const __m256i v3 = _mm256_loadu_si256((const __m256i*) (w0 + stride3));
        const __m256i v7 = _mm256_loadu_si256((const __m256i*) (w0 + stride7));
        const __m256i v11 = _mm256_loadu_si256((const __m256i*) (w8 + stride3));
        const __m256i v15 = _mm256_loadu_si256((const __m256i*) (w8 + stride7));
        const __m512i z3 = _mm512_inserti64x4(_mm512_castsi256_si512(v3), v7, 1);
        const __m512i z7 = _mm512_inserti64x4(_mm512_castsi256_si512(v11), v15, 1);
        w0 += 16;
        w8 += 16;

        const __m512i t0 = _mm512_unpacklo_epi32(z0, z1);
        const __m512i t1 = _mm512_unpackhi_epi32(z0, z1);
        const __m512i t2 = _mm512_unpacklo_epi32(z2, z3);
        const __m512i t3 = _mm512_unpackhi_epi32(z2, z3);
        const __m512i t4 = _mm512_unpacklo_epi32(z4, z5);
        const __m512i t5 = _mm512_unpackhi_epi32(z4, z5);
        const __m512i t6 = _mm512_unpacklo_epi32(z6, z7);
        const __m512i t7 = _mm512_unpackhi_epi32(z6, z7);

        const __m512i u0 = _mm512_unpacklo_epi64(t0, t2);
        const __m512i u1 = _mm512_unpackhi_epi64(t0, t2);
        const __m512i u2 = _mm512_unpacklo_epi64(t1, t3);
        const __m512i u3 = _mm512_unpackhi_epi64(t1, t3);
        const __m512i u4 = _mm512_unpacklo_epi64(t4, t6);
        const __m512i u5 = _mm512_unpackhi_epi64(t4, t6);
        const __m512i u6 = _mm512_unpacklo_epi64(t5, t7);
        const __m512i u7 = _mm512_unpackhi_epi64(t5, t7);

        const __m512i out0_0 = _mm512_shuffle_i32x4(u0, u4, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_4 = _mm512_shuffle_i32x4(u0, u4, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 0), out0_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 128), out0_4);
        const __m512i out0_1 = _mm512_shuffle_i32x4(u1, u5, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_5 = _mm512_shuffle_i32x4(u1, u5, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 32), out0_1);
        _mm512_storeu_si512((__m512i*) (packed_w + 160), out0_5);
        const __m512i out0_2 = _mm512_shuffle_i32x4(u2, u6, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_6 = _mm512_shuffle_i32x4(u2, u6, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 64), out0_2);
        _mm512_storeu_si512((__m512i*) (packed_w + 192), out0_6);
        const __m512i out0_3 = _mm512_shuffle_i32x4(u3, u7, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_7 = _mm512_shuffle_i32x4(u3, u7, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 96), out0_3);
        _mm512_storeu_si512((__m512i*) (packed_w + 224), out0_7);
        packed_w += 256;
      }

      // KC remainder (1..15) using AVX-512BW masking
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 15);
        const __mmask16 kmask = _cvtu32_mask16((1U << k) - 1U);

        const __m256i v0 = _mm256_maskz_loadu_epi16(kmask, w0);
        const __m256i v4 = _mm256_maskz_loadu_epi16(kmask, w0 + stride4);
        const __m256i v8 = _mm256_maskz_loadu_epi16(kmask, w8);
        const __m256i v12 = _mm256_maskz_loadu_epi16(kmask, w8 + stride4);
        const __m512i z0 = _mm512_inserti64x4(_mm512_castsi256_si512(v0), v4, 1);
        const __m512i z4 = _mm512_inserti64x4(_mm512_castsi256_si512(v8), v12, 1);
        const __m256i v1 = _mm256_maskz_loadu_epi16(kmask, w0 + stride1);
        const __m256i v5 = _mm256_maskz_loadu_epi16(kmask, w0 + stride5);
        const __m256i v9 = _mm256_maskz_loadu_epi16(kmask, w8 + stride1);
        const __m256i v13 = _mm256_maskz_loadu_epi16(kmask, w8 + stride5);
        const __m512i z1 = _mm512_inserti64x4(_mm512_castsi256_si512(v1), v5, 1);
        const __m512i z5 = _mm512_inserti64x4(_mm512_castsi256_si512(v9), v13, 1);
        const __m256i v2 = _mm256_maskz_loadu_epi16(kmask, w0 + stride2);
        const __m256i v6 = _mm256_maskz_loadu_epi16(kmask, w0 + stride6);
        const __m256i v10 = _mm256_maskz_loadu_epi16(kmask, w8 + stride2);
        const __m256i v14 = _mm256_maskz_loadu_epi16(kmask, w8 + stride6);
        const __m512i z2 = _mm512_inserti64x4(_mm512_castsi256_si512(v2), v6, 1);
        const __m512i z6 = _mm512_inserti64x4(_mm512_castsi256_si512(v10), v14, 1);
        const __m256i v3 = _mm256_maskz_loadu_epi16(kmask, w0 + stride3);
        const __m256i v7 = _mm256_maskz_loadu_epi16(kmask, w0 + stride7);
        const __m256i v11 = _mm256_maskz_loadu_epi16(kmask, w8 + stride3);
        const __m256i v15 = _mm256_maskz_loadu_epi16(kmask, w8 + stride7);
        const __m512i z3 = _mm512_inserti64x4(_mm512_castsi256_si512(v3), v7, 1);
        const __m512i z7 = _mm512_inserti64x4(_mm512_castsi256_si512(v11), v15, 1);

        const __m512i t0 = _mm512_unpacklo_epi32(z0, z1);
        const __m512i t1 = _mm512_unpackhi_epi32(z0, z1);
        const __m512i t2 = _mm512_unpacklo_epi32(z2, z3);
        const __m512i t3 = _mm512_unpackhi_epi32(z2, z3);
        const __m512i t4 = _mm512_unpacklo_epi32(z4, z5);
        const __m512i t5 = _mm512_unpackhi_epi32(z4, z5);
        const __m512i t6 = _mm512_unpacklo_epi32(z6, z7);
        const __m512i t7 = _mm512_unpackhi_epi32(z6, z7);

        const __m512i u0 = _mm512_unpacklo_epi64(t0, t2);
        const __m512i u1 = _mm512_unpackhi_epi64(t0, t2);
        const __m512i u2 = _mm512_unpacklo_epi64(t1, t3);
        const __m512i u3 = _mm512_unpackhi_epi64(t1, t3);
        const __m512i u4 = _mm512_unpacklo_epi64(t4, t6);
        const __m512i u5 = _mm512_unpackhi_epi64(t4, t6);
        const __m512i u6 = _mm512_unpacklo_epi64(t5, t7);
        const __m512i u7 = _mm512_unpackhi_epi64(t5, t7);

        const __m512i out0_0 = _mm512_shuffle_i32x4(u0, u4, _MM_SHUFFLE(2, 0, 2, 0));
        _mm512_storeu_si512((__m512i*) (packed_w + 0), out0_0);
        const __m512i out0_4 = _mm512_shuffle_i32x4(u0, u4, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 8) {
          _mm512_storeu_si512((__m512i*) (packed_w + 128), out0_4);
        }
        const __m512i out0_1 = _mm512_shuffle_i32x4(u1, u5, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 2) {
          _mm512_storeu_si512((__m512i*) (packed_w + 32), out0_1);
        }
        const __m512i out0_5 = _mm512_shuffle_i32x4(u1, u5, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 10) {
          _mm512_storeu_si512((__m512i*) (packed_w + 160), out0_5);
        }
        const __m512i out0_2 = _mm512_shuffle_i32x4(u2, u6, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 4) {
          _mm512_storeu_si512((__m512i*) (packed_w + 64), out0_2);
        }
        const __m512i out0_6 = _mm512_shuffle_i32x4(u2, u6, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 12) {
          _mm512_storeu_si512((__m512i*) (packed_w + 192), out0_6);
        }
        const __m512i out0_3 = _mm512_shuffle_i32x4(u3, u7, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 6) {
          _mm512_storeu_si512((__m512i*) (packed_w + 96), out0_3);
        }
        const __m512i out0_7 = _mm512_shuffle_i32x4(u3, u7, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 14) {
          _mm512_storeu_si512((__m512i*) (packed_w + 224), out0_7);
        }
        packed_w += ((k + 1) >> 1) * 32;
      }
      packed_w = (uint16_t*) ((uintptr_t) packed_w + extra_bytes);
      w_base += 16 * stride1;
    }

    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 15);
      const uint16_t* w0 = w_base;
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *packed_w++ = *b++;
        } while (--nb != 0);
        packed_w += (16 - n);
      } else {
        const __m256i vzero = _mm256_setzero_si256();
        _mm256_storeu_si256((__m256i*) packed_w, vzero);
        packed_w += 16;
      }

      // NR remainder has less than 16 rows so last row is not loaded
      const uint16_t* w1 = w0 + n_stride;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const uint16_t* w2 = w1 + n_stride;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const uint16_t* w3 = w2 + n_stride;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const uint16_t* w4 = w3 + n_stride;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const uint16_t* w5 = w4 + n_stride;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const uint16_t* w6 = w5 + n_stride;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const uint16_t* w7 = w6 + n_stride;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const uint16_t* w8 = w7 + n_stride;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const uint16_t* w9 = w8 + n_stride;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const uint16_t* w10 = w9 + n_stride;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const uint16_t* w11 = w10 + n_stride;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const uint16_t* w12 = w11 + n_stride;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const uint16_t* w13 = w12 + n_stride;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const uint16_t* w14 = w13 + n_stride;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }

      // KC main loop multiple of 16
      size_t k = kc;
      for (; k >= 16; k -= 16) {
        const __m256i v0 = _mm256_loadu_si256((const __m256i*) w0);
        w0 += 16;
        const __m256i v4 = _mm256_loadu_si256((const __m256i*) w4);
        w4 += 16;
        const __m256i v8 = _mm256_loadu_si256((const __m256i*) w8);
        w8 += 16;
        const __m256i v12 = _mm256_loadu_si256((const __m256i*) w12);
        w12 += 16;
        const __m512i z0 = _mm512_inserti64x4(_mm512_castsi256_si512(v0), v4, 1);
        const __m512i z4 = _mm512_inserti64x4(_mm512_castsi256_si512(v8), v12, 1);
        const __m256i v1 = _mm256_loadu_si256((const __m256i*) w1);
        w1 += 16;
        const __m256i v5 = _mm256_loadu_si256((const __m256i*) w5);
        w5 += 16;
        const __m256i v9 = _mm256_loadu_si256((const __m256i*) w9);
        w9 += 16;
        const __m256i v13 = _mm256_loadu_si256((const __m256i*) w13);
        w13 += 16;
        const __m512i z1 = _mm512_inserti64x4(_mm512_castsi256_si512(v1), v5, 1);
        const __m512i z5 = _mm512_inserti64x4(_mm512_castsi256_si512(v9), v13, 1);
        const __m256i v2 = _mm256_loadu_si256((const __m256i*) w2);
        w2 += 16;
        const __m256i v6 = _mm256_loadu_si256((const __m256i*) w6);
        w6 += 16;
        const __m256i v10 = _mm256_loadu_si256((const __m256i*) w10);
        w10 += 16;
        const __m256i v14 = _mm256_loadu_si256((const __m256i*) w14);
        w14 += 16;
        const __m512i z2 = _mm512_inserti64x4(_mm512_castsi256_si512(v2), v6, 1);
        const __m512i z6 = _mm512_inserti64x4(_mm512_castsi256_si512(v10), v14, 1);
        const __m256i v3 = _mm256_loadu_si256((const __m256i*) w3);
        w3 += 16;
        const __m256i v7 = _mm256_loadu_si256((const __m256i*) w7);
        w7 += 16;
        const __m256i v11 = _mm256_loadu_si256((const __m256i*) w11);
        w11 += 16;
        const __m256i v15 = _mm256_setzero_si256();
        const __m512i z3 = _mm512_inserti64x4(_mm512_castsi256_si512(v3), v7, 1);
        const __m512i z7 = _mm512_inserti64x4(_mm512_castsi256_si512(v11), v15, 1);

        const __m512i t0 = _mm512_unpacklo_epi32(z0, z1);
        const __m512i t1 = _mm512_unpackhi_epi32(z0, z1);
        const __m512i t2 = _mm512_unpacklo_epi32(z2, z3);
        const __m512i t3 = _mm512_unpackhi_epi32(z2, z3);
        const __m512i t4 = _mm512_unpacklo_epi32(z4, z5);
        const __m512i t5 = _mm512_unpackhi_epi32(z4, z5);
        const __m512i t6 = _mm512_unpacklo_epi32(z6, z7);
        const __m512i t7 = _mm512_unpackhi_epi32(z6, z7);

        const __m512i u0 = _mm512_unpacklo_epi64(t0, t2);
        const __m512i u1 = _mm512_unpackhi_epi64(t0, t2);
        const __m512i u2 = _mm512_unpacklo_epi64(t1, t3);
        const __m512i u3 = _mm512_unpackhi_epi64(t1, t3);
        const __m512i u4 = _mm512_unpacklo_epi64(t4, t6);
        const __m512i u5 = _mm512_unpackhi_epi64(t4, t6);
        const __m512i u6 = _mm512_unpacklo_epi64(t5, t7);
        const __m512i u7 = _mm512_unpackhi_epi64(t5, t7);

        const __m512i out0_0 = _mm512_shuffle_i32x4(u0, u4, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_4 = _mm512_shuffle_i32x4(u0, u4, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 0), out0_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 128), out0_4);
        const __m512i out0_1 = _mm512_shuffle_i32x4(u1, u5, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_5 = _mm512_shuffle_i32x4(u1, u5, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 32), out0_1);
        _mm512_storeu_si512((__m512i*) (packed_w + 160), out0_5);
        const __m512i out0_2 = _mm512_shuffle_i32x4(u2, u6, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_6 = _mm512_shuffle_i32x4(u2, u6, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 64), out0_2);
        _mm512_storeu_si512((__m512i*) (packed_w + 192), out0_6);
        const __m512i out0_3 = _mm512_shuffle_i32x4(u3, u7, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_7 = _mm512_shuffle_i32x4(u3, u7, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 96), out0_3);
        _mm512_storeu_si512((__m512i*) (packed_w + 224), out0_7);
        packed_w += 256;
      }

      // KC remainder (1..15) using AVX-512BW masking
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 15);
        const __mmask16 kmask = _cvtu32_mask16((1U << k) - 1U);

        const __m256i v0 = _mm256_maskz_loadu_epi16(kmask, w0);
        const __m256i v4 = _mm256_maskz_loadu_epi16(kmask, w4);
        const __m256i v8 = _mm256_maskz_loadu_epi16(kmask, w8);
        const __m256i v12 = _mm256_maskz_loadu_epi16(kmask, w12);
        const __m512i z0 = _mm512_inserti64x4(_mm512_castsi256_si512(v0), v4, 1);
        const __m512i z4 = _mm512_inserti64x4(_mm512_castsi256_si512(v8), v12, 1);
        const __m256i v1 = _mm256_maskz_loadu_epi16(kmask, w1);
        const __m256i v5 = _mm256_maskz_loadu_epi16(kmask, w5);
        const __m256i v9 = _mm256_maskz_loadu_epi16(kmask, w9);
        const __m256i v13 = _mm256_maskz_loadu_epi16(kmask, w13);
        const __m512i z1 = _mm512_inserti64x4(_mm512_castsi256_si512(v1), v5, 1);
        const __m512i z5 = _mm512_inserti64x4(_mm512_castsi256_si512(v9), v13, 1);
        const __m256i v2 = _mm256_maskz_loadu_epi16(kmask, w2);
        const __m256i v6 = _mm256_maskz_loadu_epi16(kmask, w6);
        const __m256i v10 = _mm256_maskz_loadu_epi16(kmask, w10);
        const __m256i v14 = _mm256_maskz_loadu_epi16(kmask, w14);
        const __m512i z2 = _mm512_inserti64x4(_mm512_castsi256_si512(v2), v6, 1);
        const __m512i z6 = _mm512_inserti64x4(_mm512_castsi256_si512(v10), v14, 1);
        const __m256i v3 = _mm256_maskz_loadu_epi16(kmask, w3);
        const __m256i v7 = _mm256_maskz_loadu_epi16(kmask, w7);
        const __m256i v11 = _mm256_maskz_loadu_epi16(kmask, w11);
        const __m256i v15 = _mm256_setzero_si256();
        const __m512i z3 = _mm512_inserti64x4(_mm512_castsi256_si512(v3), v7, 1);
        const __m512i z7 = _mm512_inserti64x4(_mm512_castsi256_si512(v11), v15, 1);

        const __m512i t0 = _mm512_unpacklo_epi32(z0, z1);
        const __m512i t1 = _mm512_unpackhi_epi32(z0, z1);
        const __m512i t2 = _mm512_unpacklo_epi32(z2, z3);
        const __m512i t3 = _mm512_unpackhi_epi32(z2, z3);
        const __m512i t4 = _mm512_unpacklo_epi32(z4, z5);
        const __m512i t5 = _mm512_unpackhi_epi32(z4, z5);
        const __m512i t6 = _mm512_unpacklo_epi32(z6, z7);
        const __m512i t7 = _mm512_unpackhi_epi32(z6, z7);

        const __m512i u0 = _mm512_unpacklo_epi64(t0, t2);
        const __m512i u1 = _mm512_unpackhi_epi64(t0, t2);
        const __m512i u2 = _mm512_unpacklo_epi64(t1, t3);
        const __m512i u3 = _mm512_unpackhi_epi64(t1, t3);
        const __m512i u4 = _mm512_unpacklo_epi64(t4, t6);
        const __m512i u5 = _mm512_unpackhi_epi64(t4, t6);
        const __m512i u6 = _mm512_unpacklo_epi64(t5, t7);
        const __m512i u7 = _mm512_unpackhi_epi64(t5, t7);

        const __m512i out0_0 = _mm512_shuffle_i32x4(u0, u4, _MM_SHUFFLE(2, 0, 2, 0));
        _mm512_storeu_si512((__m512i*) (packed_w + 0), out0_0);
        const __m512i out0_4 = _mm512_shuffle_i32x4(u0, u4, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 8) {
          _mm512_storeu_si512((__m512i*) (packed_w + 128), out0_4);
        }
        const __m512i out0_1 = _mm512_shuffle_i32x4(u1, u5, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 2) {
          _mm512_storeu_si512((__m512i*) (packed_w + 32), out0_1);
        }
        const __m512i out0_5 = _mm512_shuffle_i32x4(u1, u5, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 10) {
          _mm512_storeu_si512((__m512i*) (packed_w + 160), out0_5);
        }
        const __m512i out0_2 = _mm512_shuffle_i32x4(u2, u6, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 4) {
          _mm512_storeu_si512((__m512i*) (packed_w + 64), out0_2);
        }
        const __m512i out0_6 = _mm512_shuffle_i32x4(u2, u6, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 12) {
          _mm512_storeu_si512((__m512i*) (packed_w + 192), out0_6);
        }
        const __m512i out0_3 = _mm512_shuffle_i32x4(u3, u7, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 6) {
          _mm512_storeu_si512((__m512i*) (packed_w + 96), out0_3);
        }
        const __m512i out0_7 = _mm512_shuffle_i32x4(u3, u7, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 14) {
          _mm512_storeu_si512((__m512i*) (packed_w + 224), out0_7);
        }
        packed_w += ((k + 1) >> 1) * 32;
      }
      packed_w = (uint16_t*) ((uintptr_t) packed_w + extra_bytes);
    }
    weights += nc * n_stride;
  } while (--g != 0);
}
