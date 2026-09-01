// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x16-packw/avx512skx.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/packw.h"
#include "src/xnnpack/unaligned.h"
#include "src/xnnpack/prefetch.h"


void xnn_x16_packw_gemm_goi_ukernel_x32__avx512skx_u16_prfm(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
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
  assert(nr == 32);   // This kernel is for NR=32
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const uint16_t* b = bias;
  uint16_t* packed_w = packed_weights;
  do {
    // NC main loop multiple of 32
    const uint16_t* w_base = weights;
    size_t n = nc;

    for (; n >= 32; n -= 32) {
      if XNN_LIKELY(b != NULL) {
        const __m512i vb0 = _mm512_loadu_si512((const __m512i*) (b + 0));
        _mm512_storeu_si512((__m512i*) (packed_w + 0), vb0);
        b += 32;
      } else {
        const __m512i vzero = _mm512_setzero_si512();
        _mm512_storeu_si512((__m512i*) (packed_w + 0), vzero);
      }
      packed_w += 32;

      const ptrdiff_t stride1 = (ptrdiff_t) kc;
      const ptrdiff_t stride2 = stride1 * 2;
      const ptrdiff_t stride3 = stride1 * 3;
      const ptrdiff_t stride4 = stride1 * 4;
      const ptrdiff_t stride5 = stride1 * 5;
      const ptrdiff_t stride6 = stride1 * 6;
      const ptrdiff_t stride7 = stride1 * 7;

      const uint16_t* w0 = w_base;
      const uint16_t* w8 = w0 + 8 * stride1;
      const uint16_t* w16 = w0 + 16 * stride1;
      const uint16_t* w24 = w0 + 24 * stride1;
      xnn_prefetch_to_l1((const int8_t*) w0);
      xnn_prefetch_to_l1((const int8_t*) w0 + 64);
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride1));
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride1) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride2));
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride2) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride3));
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride3) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride4));
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride4) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride5));
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride5) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride6));
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride6) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride7));
      xnn_prefetch_to_l1((const int8_t*) (w0 + stride7) + 64);
      xnn_prefetch_to_l1((const int8_t*) w8);
      xnn_prefetch_to_l1((const int8_t*) w8 + 64);
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride1));
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride1) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride2));
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride2) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride3));
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride3) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride4));
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride4) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride5));
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride5) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride6));
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride6) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride7));
      xnn_prefetch_to_l1((const int8_t*) (w8 + stride7) + 64);
      xnn_prefetch_to_l1((const int8_t*) w16);
      xnn_prefetch_to_l1((const int8_t*) w16 + 64);
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride1));
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride1) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride2));
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride2) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride3));
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride3) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride4));
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride4) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride5));
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride5) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride6));
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride6) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride7));
      xnn_prefetch_to_l1((const int8_t*) (w16 + stride7) + 64);
      xnn_prefetch_to_l1((const int8_t*) w24);
      xnn_prefetch_to_l1((const int8_t*) w24 + 64);
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride1));
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride1) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride2));
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride2) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride3));
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride3) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride4));
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride4) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride5));
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride5) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride6));
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride6) + 64);
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride7));
      xnn_prefetch_to_l1((const int8_t*) (w24 + stride7) + 64);

      // KC main loop multiple of 16
      size_t k = kc;
      for (; k >= 16; k -= 16) {
        const __m256i v0 = _mm256_loadu_si256((const __m256i*) w0);
        const __m256i v8 = _mm256_loadu_si256((const __m256i*) w8);
        const __m256i v16 = _mm256_loadu_si256((const __m256i*) w16);
        const __m256i v24 = _mm256_loadu_si256((const __m256i*) w24);
        __m512i z0 = _mm512_inserti64x4(_mm512_castsi256_si512(v0), v8, 1);
        __m512i z8 = _mm512_inserti64x4(_mm512_castsi256_si512(v16), v24, 1);
        const __m256i v1 = _mm256_loadu_si256((const __m256i*) (w0 + stride1));
        const __m256i v9 = _mm256_loadu_si256((const __m256i*) (w8 + stride1));
        const __m256i v17 = _mm256_loadu_si256((const __m256i*) (w16 + stride1));
        const __m256i v25 = _mm256_loadu_si256((const __m256i*) (w24 + stride1));
        __m512i z1 = _mm512_inserti64x4(_mm512_castsi256_si512(v1), v9, 1);
        __m512i z9 = _mm512_inserti64x4(_mm512_castsi256_si512(v17), v25, 1);
        const __m256i v2 = _mm256_loadu_si256((const __m256i*) (w0 + stride2));
        const __m256i v10 = _mm256_loadu_si256((const __m256i*) (w8 + stride2));
        const __m256i v18 = _mm256_loadu_si256((const __m256i*) (w16 + stride2));
        const __m256i v26 = _mm256_loadu_si256((const __m256i*) (w24 + stride2));
        __m512i z2 = _mm512_inserti64x4(_mm512_castsi256_si512(v2), v10, 1);
        __m512i z10 = _mm512_inserti64x4(_mm512_castsi256_si512(v18), v26, 1);
        const __m256i v3 = _mm256_loadu_si256((const __m256i*) (w0 + stride3));
        const __m256i v11 = _mm256_loadu_si256((const __m256i*) (w8 + stride3));
        const __m256i v19 = _mm256_loadu_si256((const __m256i*) (w16 + stride3));
        const __m256i v27 = _mm256_loadu_si256((const __m256i*) (w24 + stride3));
        __m512i z3 = _mm512_inserti64x4(_mm512_castsi256_si512(v3), v11, 1);
        __m512i z11 = _mm512_inserti64x4(_mm512_castsi256_si512(v19), v27, 1);
        const __m256i v4 = _mm256_loadu_si256((const __m256i*) (w0 + stride4));
        const __m256i v12 = _mm256_loadu_si256((const __m256i*) (w8 + stride4));
        const __m256i v20 = _mm256_loadu_si256((const __m256i*) (w16 + stride4));
        const __m256i v28 = _mm256_loadu_si256((const __m256i*) (w24 + stride4));
        __m512i z4 = _mm512_inserti64x4(_mm512_castsi256_si512(v4), v12, 1);
        __m512i z12 = _mm512_inserti64x4(_mm512_castsi256_si512(v20), v28, 1);
        const __m256i v5 = _mm256_loadu_si256((const __m256i*) (w0 + stride5));
        const __m256i v13 = _mm256_loadu_si256((const __m256i*) (w8 + stride5));
        const __m256i v21 = _mm256_loadu_si256((const __m256i*) (w16 + stride5));
        const __m256i v29 = _mm256_loadu_si256((const __m256i*) (w24 + stride5));
        __m512i z5 = _mm512_inserti64x4(_mm512_castsi256_si512(v5), v13, 1);
        __m512i z13 = _mm512_inserti64x4(_mm512_castsi256_si512(v21), v29, 1);
        const __m256i v6 = _mm256_loadu_si256((const __m256i*) (w0 + stride6));
        const __m256i v14 = _mm256_loadu_si256((const __m256i*) (w8 + stride6));
        const __m256i v22 = _mm256_loadu_si256((const __m256i*) (w16 + stride6));
        const __m256i v30 = _mm256_loadu_si256((const __m256i*) (w24 + stride6));
        __m512i z6 = _mm512_inserti64x4(_mm512_castsi256_si512(v6), v14, 1);
        __m512i z14 = _mm512_inserti64x4(_mm512_castsi256_si512(v22), v30, 1);
        const __m256i v7 = _mm256_loadu_si256((const __m256i*) (w0 + stride7));
        const __m256i v15 = _mm256_loadu_si256((const __m256i*) (w8 + stride7));
        const __m256i v23 = _mm256_loadu_si256((const __m256i*) (w16 + stride7));
        const __m256i v31 = _mm256_loadu_si256((const __m256i*) (w24 + stride7));
        __m512i z7 = _mm512_inserti64x4(_mm512_castsi256_si512(v7), v15, 1);
        __m512i z15 = _mm512_inserti64x4(_mm512_castsi256_si512(v23), v31, 1);
        w0 += 16;
        w8 += 16;
        w16 += 16;
        w24 += 16;

        const __m512i t0 = _mm512_unpacklo_epi16(z0, z1);
        const __m512i t1 = _mm512_unpackhi_epi16(z0, z1);
        const __m512i t2 = _mm512_unpacklo_epi16(z2, z3);
        const __m512i t3 = _mm512_unpackhi_epi16(z2, z3);
        const __m512i t4 = _mm512_unpacklo_epi16(z4, z5);
        const __m512i t5 = _mm512_unpackhi_epi16(z4, z5);
        const __m512i t6 = _mm512_unpacklo_epi16(z6, z7);
        const __m512i t7 = _mm512_unpackhi_epi16(z6, z7);

        const __m512i u0 = _mm512_unpacklo_epi32(t0, t2);
        const __m512i u1 = _mm512_unpackhi_epi32(t0, t2);
        const __m512i u2 = _mm512_unpacklo_epi32(t1, t3);
        const __m512i u3 = _mm512_unpackhi_epi32(t1, t3);
        const __m512i u4 = _mm512_unpacklo_epi32(t4, t6);
        const __m512i u5 = _mm512_unpackhi_epi32(t4, t6);
        const __m512i u6 = _mm512_unpacklo_epi32(t5, t7);
        const __m512i u7 = _mm512_unpackhi_epi32(t5, t7);

        const __m512i w0 = _mm512_unpacklo_epi64(u0, u4);
        const __m512i w1 = _mm512_unpackhi_epi64(u0, u4);
        const __m512i w2 = _mm512_unpacklo_epi64(u1, u5);
        const __m512i w3 = _mm512_unpackhi_epi64(u1, u5);
        const __m512i w4 = _mm512_unpacklo_epi64(u2, u6);
        const __m512i w5 = _mm512_unpackhi_epi64(u2, u6);
        const __m512i w6 = _mm512_unpacklo_epi64(u3, u7);
        const __m512i w7 = _mm512_unpackhi_epi64(u3, u7);
        const __m512i t8 = _mm512_unpacklo_epi16(z8, z9);
        const __m512i t9 = _mm512_unpackhi_epi16(z8, z9);
        const __m512i t10 = _mm512_unpacklo_epi16(z10, z11);
        const __m512i t11 = _mm512_unpackhi_epi16(z10, z11);
        const __m512i t12 = _mm512_unpacklo_epi16(z12, z13);
        const __m512i t13 = _mm512_unpackhi_epi16(z12, z13);
        const __m512i t14 = _mm512_unpacklo_epi16(z14, z15);
        const __m512i t15 = _mm512_unpackhi_epi16(z14, z15);

        const __m512i u8 = _mm512_unpacklo_epi32(t8, t10);
        const __m512i u9 = _mm512_unpackhi_epi32(t8, t10);
        const __m512i u10 = _mm512_unpacklo_epi32(t9, t11);
        const __m512i u11 = _mm512_unpackhi_epi32(t9, t11);
        const __m512i u12 = _mm512_unpacklo_epi32(t12, t14);
        const __m512i u13 = _mm512_unpackhi_epi32(t12, t14);
        const __m512i u14 = _mm512_unpacklo_epi32(t13, t15);
        const __m512i u15 = _mm512_unpackhi_epi32(t13, t15);

        const __m512i w8 = _mm512_unpacklo_epi64(u8, u12);
        const __m512i w9 = _mm512_unpackhi_epi64(u8, u12);
        const __m512i w10 = _mm512_unpacklo_epi64(u9, u13);
        const __m512i w11 = _mm512_unpackhi_epi64(u9, u13);
        const __m512i w12 = _mm512_unpacklo_epi64(u10, u14);
        const __m512i w13 = _mm512_unpackhi_epi64(u10, u14);
        const __m512i w14 = _mm512_unpacklo_epi64(u11, u15);
        const __m512i w15 = _mm512_unpackhi_epi64(u11, u15);

        const __m512i out0_0 = _mm512_shuffle_i32x4(w0, w8, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_8 = _mm512_shuffle_i32x4(w0, w8, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 0), out0_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 256), out0_8);
        const __m512i out0_1 = _mm512_shuffle_i32x4(w1, w9, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_9 = _mm512_shuffle_i32x4(w1, w9, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 32), out0_1);
        _mm512_storeu_si512((__m512i*) (packed_w + 288), out0_9);
        const __m512i out0_2 = _mm512_shuffle_i32x4(w2, w10, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_10 = _mm512_shuffle_i32x4(w2, w10, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 64), out0_2);
        _mm512_storeu_si512((__m512i*) (packed_w + 320), out0_10);
        const __m512i out0_3 = _mm512_shuffle_i32x4(w3, w11, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_11 = _mm512_shuffle_i32x4(w3, w11, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 96), out0_3);
        _mm512_storeu_si512((__m512i*) (packed_w + 352), out0_11);
        const __m512i out0_4 = _mm512_shuffle_i32x4(w4, w12, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_12 = _mm512_shuffle_i32x4(w4, w12, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 128), out0_4);
        _mm512_storeu_si512((__m512i*) (packed_w + 384), out0_12);
        const __m512i out0_5 = _mm512_shuffle_i32x4(w5, w13, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_13 = _mm512_shuffle_i32x4(w5, w13, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 160), out0_5);
        _mm512_storeu_si512((__m512i*) (packed_w + 416), out0_13);
        const __m512i out0_6 = _mm512_shuffle_i32x4(w6, w14, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_14 = _mm512_shuffle_i32x4(w6, w14, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 192), out0_6);
        _mm512_storeu_si512((__m512i*) (packed_w + 448), out0_14);
        const __m512i out0_7 = _mm512_shuffle_i32x4(w7, w15, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_15 = _mm512_shuffle_i32x4(w7, w15, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 224), out0_7);
        _mm512_storeu_si512((__m512i*) (packed_w + 480), out0_15);
        packed_w += 512;
      }

      // KC remainder (1..15) using AVX-512BW masking
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 15);
        const __mmask16 kmask = _cvtu32_mask16((1U << k) - 1U);

        const __m256i v0 = _mm256_maskz_loadu_epi16(kmask, w0);
        const __m256i v8 = _mm256_maskz_loadu_epi16(kmask, w8);
        const __m256i v16 = _mm256_maskz_loadu_epi16(kmask, w16);
        const __m256i v24 = _mm256_maskz_loadu_epi16(kmask, w24);
        __m512i z0 = _mm512_inserti64x4(_mm512_castsi256_si512(v0), v8, 1);
        __m512i z8 = _mm512_inserti64x4(_mm512_castsi256_si512(v16), v24, 1);
        const __m256i v1 = _mm256_maskz_loadu_epi16(kmask, w0 + stride1);
        const __m256i v9 = _mm256_maskz_loadu_epi16(kmask, w8 + stride1);
        const __m256i v17 = _mm256_maskz_loadu_epi16(kmask, w16 + stride1);
        const __m256i v25 = _mm256_maskz_loadu_epi16(kmask, w24 + stride1);
        __m512i z1 = _mm512_inserti64x4(_mm512_castsi256_si512(v1), v9, 1);
        __m512i z9 = _mm512_inserti64x4(_mm512_castsi256_si512(v17), v25, 1);
        const __m256i v2 = _mm256_maskz_loadu_epi16(kmask, w0 + stride2);
        const __m256i v10 = _mm256_maskz_loadu_epi16(kmask, w8 + stride2);
        const __m256i v18 = _mm256_maskz_loadu_epi16(kmask, w16 + stride2);
        const __m256i v26 = _mm256_maskz_loadu_epi16(kmask, w24 + stride2);
        __m512i z2 = _mm512_inserti64x4(_mm512_castsi256_si512(v2), v10, 1);
        __m512i z10 = _mm512_inserti64x4(_mm512_castsi256_si512(v18), v26, 1);
        const __m256i v3 = _mm256_maskz_loadu_epi16(kmask, w0 + stride3);
        const __m256i v11 = _mm256_maskz_loadu_epi16(kmask, w8 + stride3);
        const __m256i v19 = _mm256_maskz_loadu_epi16(kmask, w16 + stride3);
        const __m256i v27 = _mm256_maskz_loadu_epi16(kmask, w24 + stride3);
        __m512i z3 = _mm512_inserti64x4(_mm512_castsi256_si512(v3), v11, 1);
        __m512i z11 = _mm512_inserti64x4(_mm512_castsi256_si512(v19), v27, 1);
        const __m256i v4 = _mm256_maskz_loadu_epi16(kmask, w0 + stride4);
        const __m256i v12 = _mm256_maskz_loadu_epi16(kmask, w8 + stride4);
        const __m256i v20 = _mm256_maskz_loadu_epi16(kmask, w16 + stride4);
        const __m256i v28 = _mm256_maskz_loadu_epi16(kmask, w24 + stride4);
        __m512i z4 = _mm512_inserti64x4(_mm512_castsi256_si512(v4), v12, 1);
        __m512i z12 = _mm512_inserti64x4(_mm512_castsi256_si512(v20), v28, 1);
        const __m256i v5 = _mm256_maskz_loadu_epi16(kmask, w0 + stride5);
        const __m256i v13 = _mm256_maskz_loadu_epi16(kmask, w8 + stride5);
        const __m256i v21 = _mm256_maskz_loadu_epi16(kmask, w16 + stride5);
        const __m256i v29 = _mm256_maskz_loadu_epi16(kmask, w24 + stride5);
        __m512i z5 = _mm512_inserti64x4(_mm512_castsi256_si512(v5), v13, 1);
        __m512i z13 = _mm512_inserti64x4(_mm512_castsi256_si512(v21), v29, 1);
        const __m256i v6 = _mm256_maskz_loadu_epi16(kmask, w0 + stride6);
        const __m256i v14 = _mm256_maskz_loadu_epi16(kmask, w8 + stride6);
        const __m256i v22 = _mm256_maskz_loadu_epi16(kmask, w16 + stride6);
        const __m256i v30 = _mm256_maskz_loadu_epi16(kmask, w24 + stride6);
        __m512i z6 = _mm512_inserti64x4(_mm512_castsi256_si512(v6), v14, 1);
        __m512i z14 = _mm512_inserti64x4(_mm512_castsi256_si512(v22), v30, 1);
        const __m256i v7 = _mm256_maskz_loadu_epi16(kmask, w0 + stride7);
        const __m256i v15 = _mm256_maskz_loadu_epi16(kmask, w8 + stride7);
        const __m256i v23 = _mm256_maskz_loadu_epi16(kmask, w16 + stride7);
        const __m256i v31 = _mm256_maskz_loadu_epi16(kmask, w24 + stride7);
        __m512i z7 = _mm512_inserti64x4(_mm512_castsi256_si512(v7), v15, 1);
        __m512i z15 = _mm512_inserti64x4(_mm512_castsi256_si512(v23), v31, 1);
        w0 += k;
        w8 += k;
        w16 += k;
        w24 += k;

        const __m512i t0 = _mm512_unpacklo_epi16(z0, z1);
        const __m512i t1 = _mm512_unpackhi_epi16(z0, z1);
        const __m512i t2 = _mm512_unpacklo_epi16(z2, z3);
        const __m512i t3 = _mm512_unpackhi_epi16(z2, z3);
        const __m512i t4 = _mm512_unpacklo_epi16(z4, z5);
        const __m512i t5 = _mm512_unpackhi_epi16(z4, z5);
        const __m512i t6 = _mm512_unpacklo_epi16(z6, z7);
        const __m512i t7 = _mm512_unpackhi_epi16(z6, z7);

        const __m512i u0 = _mm512_unpacklo_epi32(t0, t2);
        const __m512i u1 = _mm512_unpackhi_epi32(t0, t2);
        const __m512i u2 = _mm512_unpacklo_epi32(t1, t3);
        const __m512i u3 = _mm512_unpackhi_epi32(t1, t3);
        const __m512i u4 = _mm512_unpacklo_epi32(t4, t6);
        const __m512i u5 = _mm512_unpackhi_epi32(t4, t6);
        const __m512i u6 = _mm512_unpacklo_epi32(t5, t7);
        const __m512i u7 = _mm512_unpackhi_epi32(t5, t7);

        const __m512i w0 = _mm512_unpacklo_epi64(u0, u4);
        const __m512i w1 = _mm512_unpackhi_epi64(u0, u4);
        const __m512i w2 = _mm512_unpacklo_epi64(u1, u5);
        const __m512i w3 = _mm512_unpackhi_epi64(u1, u5);
        const __m512i w4 = _mm512_unpacklo_epi64(u2, u6);
        const __m512i w5 = _mm512_unpackhi_epi64(u2, u6);
        const __m512i w6 = _mm512_unpacklo_epi64(u3, u7);
        const __m512i w7 = _mm512_unpackhi_epi64(u3, u7);
        const __m512i t8 = _mm512_unpacklo_epi16(z8, z9);
        const __m512i t9 = _mm512_unpackhi_epi16(z8, z9);
        const __m512i t10 = _mm512_unpacklo_epi16(z10, z11);
        const __m512i t11 = _mm512_unpackhi_epi16(z10, z11);
        const __m512i t12 = _mm512_unpacklo_epi16(z12, z13);
        const __m512i t13 = _mm512_unpackhi_epi16(z12, z13);
        const __m512i t14 = _mm512_unpacklo_epi16(z14, z15);
        const __m512i t15 = _mm512_unpackhi_epi16(z14, z15);

        const __m512i u8 = _mm512_unpacklo_epi32(t8, t10);
        const __m512i u9 = _mm512_unpackhi_epi32(t8, t10);
        const __m512i u10 = _mm512_unpacklo_epi32(t9, t11);
        const __m512i u11 = _mm512_unpackhi_epi32(t9, t11);
        const __m512i u12 = _mm512_unpacklo_epi32(t12, t14);
        const __m512i u13 = _mm512_unpackhi_epi32(t12, t14);
        const __m512i u14 = _mm512_unpacklo_epi32(t13, t15);
        const __m512i u15 = _mm512_unpackhi_epi32(t13, t15);

        const __m512i w8 = _mm512_unpacklo_epi64(u8, u12);
        const __m512i w9 = _mm512_unpackhi_epi64(u8, u12);
        const __m512i w10 = _mm512_unpacklo_epi64(u9, u13);
        const __m512i w11 = _mm512_unpackhi_epi64(u9, u13);
        const __m512i w12 = _mm512_unpacklo_epi64(u10, u14);
        const __m512i w13 = _mm512_unpackhi_epi64(u10, u14);
        const __m512i w14 = _mm512_unpacklo_epi64(u11, u15);
        const __m512i w15 = _mm512_unpackhi_epi64(u11, u15);

        const __m512i out0_0 = _mm512_shuffle_i32x4(w0, w8, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 0) {
          _mm512_storeu_si512((__m512i*) (packed_w + 0), out0_0);
        }
        const __m512i out0_8 = _mm512_shuffle_i32x4(w0, w8, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 8) {
          _mm512_storeu_si512((__m512i*) (packed_w + 256), out0_8);
        }
        const __m512i out0_1 = _mm512_shuffle_i32x4(w1, w9, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 1) {
          _mm512_storeu_si512((__m512i*) (packed_w + 32), out0_1);
        }
        const __m512i out0_9 = _mm512_shuffle_i32x4(w1, w9, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 9) {
          _mm512_storeu_si512((__m512i*) (packed_w + 288), out0_9);
        }
        const __m512i out0_2 = _mm512_shuffle_i32x4(w2, w10, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 2) {
          _mm512_storeu_si512((__m512i*) (packed_w + 64), out0_2);
        }
        const __m512i out0_10 = _mm512_shuffle_i32x4(w2, w10, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 10) {
          _mm512_storeu_si512((__m512i*) (packed_w + 320), out0_10);
        }
        const __m512i out0_3 = _mm512_shuffle_i32x4(w3, w11, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 3) {
          _mm512_storeu_si512((__m512i*) (packed_w + 96), out0_3);
        }
        const __m512i out0_11 = _mm512_shuffle_i32x4(w3, w11, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 11) {
          _mm512_storeu_si512((__m512i*) (packed_w + 352), out0_11);
        }
        const __m512i out0_4 = _mm512_shuffle_i32x4(w4, w12, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 4) {
          _mm512_storeu_si512((__m512i*) (packed_w + 128), out0_4);
        }
        const __m512i out0_12 = _mm512_shuffle_i32x4(w4, w12, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 12) {
          _mm512_storeu_si512((__m512i*) (packed_w + 384), out0_12);
        }
        const __m512i out0_5 = _mm512_shuffle_i32x4(w5, w13, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 5) {
          _mm512_storeu_si512((__m512i*) (packed_w + 160), out0_5);
        }
        const __m512i out0_13 = _mm512_shuffle_i32x4(w5, w13, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 13) {
          _mm512_storeu_si512((__m512i*) (packed_w + 416), out0_13);
        }
        const __m512i out0_6 = _mm512_shuffle_i32x4(w6, w14, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 6) {
          _mm512_storeu_si512((__m512i*) (packed_w + 192), out0_6);
        }
        const __m512i out0_14 = _mm512_shuffle_i32x4(w6, w14, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 14) {
          _mm512_storeu_si512((__m512i*) (packed_w + 448), out0_14);
        }
        const __m512i out0_7 = _mm512_shuffle_i32x4(w7, w15, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 7) {
          _mm512_storeu_si512((__m512i*) (packed_w + 224), out0_7);
        }
        const __m512i out0_15 = _mm512_shuffle_i32x4(w7, w15, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 15) {
          _mm512_storeu_si512((__m512i*) (packed_w + 480), out0_15);
        }
        packed_w += 32 * k;
      }
      packed_w = (uint16_t*) ((uintptr_t) packed_w + extra_bytes);
      w_base += 32 * stride1;
    }

    // NC remainder (1..31)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 31);
      const uint16_t* w0 = w_base;
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *packed_w++ = *b++;
        } while (--nb != 0);
        packed_w += (32 - n);
      } else {
        const __m512i vzero = _mm512_setzero_si512();
        _mm512_storeu_si512((__m512i*) (packed_w + 0), vzero);
        packed_w += 32;
      }

      // NR remainder has less than 32 rows so last row is not loaded
      const uint16_t* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const uint16_t* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const uint16_t* w3 = w2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const uint16_t* w4 = w3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const uint16_t* w5 = w4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const uint16_t* w6 = w5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const uint16_t* w7 = w6 + kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const uint16_t* w8 = w7 + kc;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const uint16_t* w9 = w8 + kc;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const uint16_t* w10 = w9 + kc;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const uint16_t* w11 = w10 + kc;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const uint16_t* w12 = w11 + kc;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const uint16_t* w13 = w12 + kc;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const uint16_t* w14 = w13 + kc;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }
      const uint16_t* w15 = w14 + kc;
      if XNN_UNPREDICTABLE(n < 16) {
        w15 = w14;
      }
      const uint16_t* w16 = w15 + kc;
      if XNN_UNPREDICTABLE(n <= 16) {
        w16 = w15;
      }
      const uint16_t* w17 = w16 + kc;
      if XNN_UNPREDICTABLE(n < 18) {
        w17 = w16;
      }
      const uint16_t* w18 = w17 + kc;
      if XNN_UNPREDICTABLE(n <= 18) {
        w18 = w17;
      }
      const uint16_t* w19 = w18 + kc;
      if XNN_UNPREDICTABLE(n < 20) {
        w19 = w18;
      }
      const uint16_t* w20 = w19 + kc;
      if XNN_UNPREDICTABLE(n <= 20) {
        w20 = w19;
      }
      const uint16_t* w21 = w20 + kc;
      if XNN_UNPREDICTABLE(n < 22) {
        w21 = w20;
      }
      const uint16_t* w22 = w21 + kc;
      if XNN_UNPREDICTABLE(n <= 22) {
        w22 = w21;
      }
      const uint16_t* w23 = w22 + kc;
      if XNN_UNPREDICTABLE(n < 24) {
        w23 = w22;
      }
      const uint16_t* w24 = w23 + kc;
      if XNN_UNPREDICTABLE(n <= 24) {
        w24 = w23;
      }
      const uint16_t* w25 = w24 + kc;
      if XNN_UNPREDICTABLE(n < 26) {
        w25 = w24;
      }
      const uint16_t* w26 = w25 + kc;
      if XNN_UNPREDICTABLE(n <= 26) {
        w26 = w25;
      }
      const uint16_t* w27 = w26 + kc;
      if XNN_UNPREDICTABLE(n < 28) {
        w27 = w26;
      }
      const uint16_t* w28 = w27 + kc;
      if XNN_UNPREDICTABLE(n <= 28) {
        w28 = w27;
      }
      const uint16_t* w29 = w28 + kc;
      if XNN_UNPREDICTABLE(n < 30) {
        w29 = w28;
      }
      const uint16_t* w30 = w29 + kc;
      if XNN_UNPREDICTABLE(n <= 30) {
        w30 = w29;
      }

      // KC main loop multiple of 16
      size_t k = kc;
      for (; k >= 16; k -= 16) {
        const __m256i v0 = _mm256_loadu_si256((const __m256i*) w0);
        w0 += 16;
        const __m256i v8 = _mm256_loadu_si256((const __m256i*) w8);
        w8 += 16;
        const __m256i v16 = _mm256_loadu_si256((const __m256i*) w16);
        w16 += 16;
        const __m256i v24 = _mm256_loadu_si256((const __m256i*) w24);
        w24 += 16;
        __m512i z0 = _mm512_inserti64x4(_mm512_castsi256_si512(v0), v8, 1);
        __m512i z8 = _mm512_inserti64x4(_mm512_castsi256_si512(v16), v24, 1);
        const __m256i v1 = _mm256_loadu_si256((const __m256i*) w1);
        w1 += 16;
        const __m256i v9 = _mm256_loadu_si256((const __m256i*) w9);
        w9 += 16;
        const __m256i v17 = _mm256_loadu_si256((const __m256i*) w17);
        w17 += 16;
        const __m256i v25 = _mm256_loadu_si256((const __m256i*) w25);
        w25 += 16;
        __m512i z1 = _mm512_inserti64x4(_mm512_castsi256_si512(v1), v9, 1);
        __m512i z9 = _mm512_inserti64x4(_mm512_castsi256_si512(v17), v25, 1);
        const __m256i v2 = _mm256_loadu_si256((const __m256i*) w2);
        w2 += 16;
        const __m256i v10 = _mm256_loadu_si256((const __m256i*) w10);
        w10 += 16;
        const __m256i v18 = _mm256_loadu_si256((const __m256i*) w18);
        w18 += 16;
        const __m256i v26 = _mm256_loadu_si256((const __m256i*) w26);
        w26 += 16;
        __m512i z2 = _mm512_inserti64x4(_mm512_castsi256_si512(v2), v10, 1);
        __m512i z10 = _mm512_inserti64x4(_mm512_castsi256_si512(v18), v26, 1);
        const __m256i v3 = _mm256_loadu_si256((const __m256i*) w3);
        w3 += 16;
        const __m256i v11 = _mm256_loadu_si256((const __m256i*) w11);
        w11 += 16;
        const __m256i v19 = _mm256_loadu_si256((const __m256i*) w19);
        w19 += 16;
        const __m256i v27 = _mm256_loadu_si256((const __m256i*) w27);
        w27 += 16;
        __m512i z3 = _mm512_inserti64x4(_mm512_castsi256_si512(v3), v11, 1);
        __m512i z11 = _mm512_inserti64x4(_mm512_castsi256_si512(v19), v27, 1);
        const __m256i v4 = _mm256_loadu_si256((const __m256i*) w4);
        w4 += 16;
        const __m256i v12 = _mm256_loadu_si256((const __m256i*) w12);
        w12 += 16;
        const __m256i v20 = _mm256_loadu_si256((const __m256i*) w20);
        w20 += 16;
        const __m256i v28 = _mm256_loadu_si256((const __m256i*) w28);
        w28 += 16;
        __m512i z4 = _mm512_inserti64x4(_mm512_castsi256_si512(v4), v12, 1);
        __m512i z12 = _mm512_inserti64x4(_mm512_castsi256_si512(v20), v28, 1);
        const __m256i v5 = _mm256_loadu_si256((const __m256i*) w5);
        w5 += 16;
        const __m256i v13 = _mm256_loadu_si256((const __m256i*) w13);
        w13 += 16;
        const __m256i v21 = _mm256_loadu_si256((const __m256i*) w21);
        w21 += 16;
        const __m256i v29 = _mm256_loadu_si256((const __m256i*) w29);
        w29 += 16;
        __m512i z5 = _mm512_inserti64x4(_mm512_castsi256_si512(v5), v13, 1);
        __m512i z13 = _mm512_inserti64x4(_mm512_castsi256_si512(v21), v29, 1);
        const __m256i v6 = _mm256_loadu_si256((const __m256i*) w6);
        w6 += 16;
        const __m256i v14 = _mm256_loadu_si256((const __m256i*) w14);
        w14 += 16;
        const __m256i v22 = _mm256_loadu_si256((const __m256i*) w22);
        w22 += 16;
        const __m256i v30 = _mm256_loadu_si256((const __m256i*) w30);
        w30 += 16;
        __m512i z6 = _mm512_inserti64x4(_mm512_castsi256_si512(v6), v14, 1);
        __m512i z14 = _mm512_inserti64x4(_mm512_castsi256_si512(v22), v30, 1);
        const __m256i v7 = _mm256_loadu_si256((const __m256i*) w7);
        w7 += 16;
        const __m256i v15 = _mm256_loadu_si256((const __m256i*) w15);
        w15 += 16;
        const __m256i v23 = _mm256_loadu_si256((const __m256i*) w23);
        w23 += 16;
        const __m256i v31 = _mm256_setzero_si256();
        __m512i z7 = _mm512_inserti64x4(_mm512_castsi256_si512(v7), v15, 1);
        __m512i z15 = _mm512_inserti64x4(_mm512_castsi256_si512(v23), v31, 1);

        const __m512i t0 = _mm512_unpacklo_epi16(z0, z1);
        const __m512i t1 = _mm512_unpackhi_epi16(z0, z1);
        const __m512i t2 = _mm512_unpacklo_epi16(z2, z3);
        const __m512i t3 = _mm512_unpackhi_epi16(z2, z3);
        const __m512i t4 = _mm512_unpacklo_epi16(z4, z5);
        const __m512i t5 = _mm512_unpackhi_epi16(z4, z5);
        const __m512i t6 = _mm512_unpacklo_epi16(z6, z7);
        const __m512i t7 = _mm512_unpackhi_epi16(z6, z7);

        const __m512i u0 = _mm512_unpacklo_epi32(t0, t2);
        const __m512i u1 = _mm512_unpackhi_epi32(t0, t2);
        const __m512i u2 = _mm512_unpacklo_epi32(t1, t3);
        const __m512i u3 = _mm512_unpackhi_epi32(t1, t3);
        const __m512i u4 = _mm512_unpacklo_epi32(t4, t6);
        const __m512i u5 = _mm512_unpackhi_epi32(t4, t6);
        const __m512i u6 = _mm512_unpacklo_epi32(t5, t7);
        const __m512i u7 = _mm512_unpackhi_epi32(t5, t7);

        const __m512i w0 = _mm512_unpacklo_epi64(u0, u4);
        const __m512i w1 = _mm512_unpackhi_epi64(u0, u4);
        const __m512i w2 = _mm512_unpacklo_epi64(u1, u5);
        const __m512i w3 = _mm512_unpackhi_epi64(u1, u5);
        const __m512i w4 = _mm512_unpacklo_epi64(u2, u6);
        const __m512i w5 = _mm512_unpackhi_epi64(u2, u6);
        const __m512i w6 = _mm512_unpacklo_epi64(u3, u7);
        const __m512i w7 = _mm512_unpackhi_epi64(u3, u7);
        const __m512i t8 = _mm512_unpacklo_epi16(z8, z9);
        const __m512i t9 = _mm512_unpackhi_epi16(z8, z9);
        const __m512i t10 = _mm512_unpacklo_epi16(z10, z11);
        const __m512i t11 = _mm512_unpackhi_epi16(z10, z11);
        const __m512i t12 = _mm512_unpacklo_epi16(z12, z13);
        const __m512i t13 = _mm512_unpackhi_epi16(z12, z13);
        const __m512i t14 = _mm512_unpacklo_epi16(z14, z15);
        const __m512i t15 = _mm512_unpackhi_epi16(z14, z15);

        const __m512i u8 = _mm512_unpacklo_epi32(t8, t10);
        const __m512i u9 = _mm512_unpackhi_epi32(t8, t10);
        const __m512i u10 = _mm512_unpacklo_epi32(t9, t11);
        const __m512i u11 = _mm512_unpackhi_epi32(t9, t11);
        const __m512i u12 = _mm512_unpacklo_epi32(t12, t14);
        const __m512i u13 = _mm512_unpackhi_epi32(t12, t14);
        const __m512i u14 = _mm512_unpacklo_epi32(t13, t15);
        const __m512i u15 = _mm512_unpackhi_epi32(t13, t15);

        const __m512i w8 = _mm512_unpacklo_epi64(u8, u12);
        const __m512i w9 = _mm512_unpackhi_epi64(u8, u12);
        const __m512i w10 = _mm512_unpacklo_epi64(u9, u13);
        const __m512i w11 = _mm512_unpackhi_epi64(u9, u13);
        const __m512i w12 = _mm512_unpacklo_epi64(u10, u14);
        const __m512i w13 = _mm512_unpackhi_epi64(u10, u14);
        const __m512i w14 = _mm512_unpacklo_epi64(u11, u15);
        const __m512i w15 = _mm512_unpackhi_epi64(u11, u15);

        const __m512i out0_0 = _mm512_shuffle_i32x4(w0, w8, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_8 = _mm512_shuffle_i32x4(w0, w8, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 0), out0_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 256), out0_8);
        const __m512i out0_1 = _mm512_shuffle_i32x4(w1, w9, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_9 = _mm512_shuffle_i32x4(w1, w9, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 32), out0_1);
        _mm512_storeu_si512((__m512i*) (packed_w + 288), out0_9);
        const __m512i out0_2 = _mm512_shuffle_i32x4(w2, w10, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_10 = _mm512_shuffle_i32x4(w2, w10, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 64), out0_2);
        _mm512_storeu_si512((__m512i*) (packed_w + 320), out0_10);
        const __m512i out0_3 = _mm512_shuffle_i32x4(w3, w11, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_11 = _mm512_shuffle_i32x4(w3, w11, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 96), out0_3);
        _mm512_storeu_si512((__m512i*) (packed_w + 352), out0_11);
        const __m512i out0_4 = _mm512_shuffle_i32x4(w4, w12, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_12 = _mm512_shuffle_i32x4(w4, w12, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 128), out0_4);
        _mm512_storeu_si512((__m512i*) (packed_w + 384), out0_12);
        const __m512i out0_5 = _mm512_shuffle_i32x4(w5, w13, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_13 = _mm512_shuffle_i32x4(w5, w13, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 160), out0_5);
        _mm512_storeu_si512((__m512i*) (packed_w + 416), out0_13);
        const __m512i out0_6 = _mm512_shuffle_i32x4(w6, w14, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_14 = _mm512_shuffle_i32x4(w6, w14, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 192), out0_6);
        _mm512_storeu_si512((__m512i*) (packed_w + 448), out0_14);
        const __m512i out0_7 = _mm512_shuffle_i32x4(w7, w15, _MM_SHUFFLE(2, 0, 2, 0));
        const __m512i out0_15 = _mm512_shuffle_i32x4(w7, w15, _MM_SHUFFLE(3, 1, 3, 1));
        _mm512_storeu_si512((__m512i*) (packed_w + 224), out0_7);
        _mm512_storeu_si512((__m512i*) (packed_w + 480), out0_15);
        packed_w += 512;
      }

      // KC remainder (1..15) using AVX-512BW masking
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 15);
        const __mmask16 kmask = _cvtu32_mask16((1U << k) - 1U);

        const __m256i v0 = _mm256_maskz_loadu_epi16(kmask, w0);
        w0 += k;
        const __m256i v8 = _mm256_maskz_loadu_epi16(kmask, w8);
        w8 += k;
        const __m256i v16 = _mm256_maskz_loadu_epi16(kmask, w16);
        w16 += k;
        const __m256i v24 = _mm256_maskz_loadu_epi16(kmask, w24);
        w24 += k;
        __m512i z0 = _mm512_inserti64x4(_mm512_castsi256_si512(v0), v8, 1);
        __m512i z8 = _mm512_inserti64x4(_mm512_castsi256_si512(v16), v24, 1);
        const __m256i v1 = _mm256_maskz_loadu_epi16(kmask, w1);
        w1 += k;
        const __m256i v9 = _mm256_maskz_loadu_epi16(kmask, w9);
        w9 += k;
        const __m256i v17 = _mm256_maskz_loadu_epi16(kmask, w17);
        w17 += k;
        const __m256i v25 = _mm256_maskz_loadu_epi16(kmask, w25);
        w25 += k;
        __m512i z1 = _mm512_inserti64x4(_mm512_castsi256_si512(v1), v9, 1);
        __m512i z9 = _mm512_inserti64x4(_mm512_castsi256_si512(v17), v25, 1);
        const __m256i v2 = _mm256_maskz_loadu_epi16(kmask, w2);
        w2 += k;
        const __m256i v10 = _mm256_maskz_loadu_epi16(kmask, w10);
        w10 += k;
        const __m256i v18 = _mm256_maskz_loadu_epi16(kmask, w18);
        w18 += k;
        const __m256i v26 = _mm256_maskz_loadu_epi16(kmask, w26);
        w26 += k;
        __m512i z2 = _mm512_inserti64x4(_mm512_castsi256_si512(v2), v10, 1);
        __m512i z10 = _mm512_inserti64x4(_mm512_castsi256_si512(v18), v26, 1);
        const __m256i v3 = _mm256_maskz_loadu_epi16(kmask, w3);
        w3 += k;
        const __m256i v11 = _mm256_maskz_loadu_epi16(kmask, w11);
        w11 += k;
        const __m256i v19 = _mm256_maskz_loadu_epi16(kmask, w19);
        w19 += k;
        const __m256i v27 = _mm256_maskz_loadu_epi16(kmask, w27);
        w27 += k;
        __m512i z3 = _mm512_inserti64x4(_mm512_castsi256_si512(v3), v11, 1);
        __m512i z11 = _mm512_inserti64x4(_mm512_castsi256_si512(v19), v27, 1);
        const __m256i v4 = _mm256_maskz_loadu_epi16(kmask, w4);
        w4 += k;
        const __m256i v12 = _mm256_maskz_loadu_epi16(kmask, w12);
        w12 += k;
        const __m256i v20 = _mm256_maskz_loadu_epi16(kmask, w20);
        w20 += k;
        const __m256i v28 = _mm256_maskz_loadu_epi16(kmask, w28);
        w28 += k;
        __m512i z4 = _mm512_inserti64x4(_mm512_castsi256_si512(v4), v12, 1);
        __m512i z12 = _mm512_inserti64x4(_mm512_castsi256_si512(v20), v28, 1);
        const __m256i v5 = _mm256_maskz_loadu_epi16(kmask, w5);
        w5 += k;
        const __m256i v13 = _mm256_maskz_loadu_epi16(kmask, w13);
        w13 += k;
        const __m256i v21 = _mm256_maskz_loadu_epi16(kmask, w21);
        w21 += k;
        const __m256i v29 = _mm256_maskz_loadu_epi16(kmask, w29);
        w29 += k;
        __m512i z5 = _mm512_inserti64x4(_mm512_castsi256_si512(v5), v13, 1);
        __m512i z13 = _mm512_inserti64x4(_mm512_castsi256_si512(v21), v29, 1);
        const __m256i v6 = _mm256_maskz_loadu_epi16(kmask, w6);
        w6 += k;
        const __m256i v14 = _mm256_maskz_loadu_epi16(kmask, w14);
        w14 += k;
        const __m256i v22 = _mm256_maskz_loadu_epi16(kmask, w22);
        w22 += k;
        const __m256i v30 = _mm256_maskz_loadu_epi16(kmask, w30);
        w30 += k;
        __m512i z6 = _mm512_inserti64x4(_mm512_castsi256_si512(v6), v14, 1);
        __m512i z14 = _mm512_inserti64x4(_mm512_castsi256_si512(v22), v30, 1);
        const __m256i v7 = _mm256_maskz_loadu_epi16(kmask, w7);
        w7 += k;
        const __m256i v15 = _mm256_maskz_loadu_epi16(kmask, w15);
        w15 += k;
        const __m256i v23 = _mm256_maskz_loadu_epi16(kmask, w23);
        w23 += k;
        const __m256i v31 = _mm256_setzero_si256();
        __m512i z7 = _mm512_inserti64x4(_mm512_castsi256_si512(v7), v15, 1);
        __m512i z15 = _mm512_inserti64x4(_mm512_castsi256_si512(v23), v31, 1);

        const __m512i t0 = _mm512_unpacklo_epi16(z0, z1);
        const __m512i t1 = _mm512_unpackhi_epi16(z0, z1);
        const __m512i t2 = _mm512_unpacklo_epi16(z2, z3);
        const __m512i t3 = _mm512_unpackhi_epi16(z2, z3);
        const __m512i t4 = _mm512_unpacklo_epi16(z4, z5);
        const __m512i t5 = _mm512_unpackhi_epi16(z4, z5);
        const __m512i t6 = _mm512_unpacklo_epi16(z6, z7);
        const __m512i t7 = _mm512_unpackhi_epi16(z6, z7);

        const __m512i u0 = _mm512_unpacklo_epi32(t0, t2);
        const __m512i u1 = _mm512_unpackhi_epi32(t0, t2);
        const __m512i u2 = _mm512_unpacklo_epi32(t1, t3);
        const __m512i u3 = _mm512_unpackhi_epi32(t1, t3);
        const __m512i u4 = _mm512_unpacklo_epi32(t4, t6);
        const __m512i u5 = _mm512_unpackhi_epi32(t4, t6);
        const __m512i u6 = _mm512_unpacklo_epi32(t5, t7);
        const __m512i u7 = _mm512_unpackhi_epi32(t5, t7);

        const __m512i w0 = _mm512_unpacklo_epi64(u0, u4);
        const __m512i w1 = _mm512_unpackhi_epi64(u0, u4);
        const __m512i w2 = _mm512_unpacklo_epi64(u1, u5);
        const __m512i w3 = _mm512_unpackhi_epi64(u1, u5);
        const __m512i w4 = _mm512_unpacklo_epi64(u2, u6);
        const __m512i w5 = _mm512_unpackhi_epi64(u2, u6);
        const __m512i w6 = _mm512_unpacklo_epi64(u3, u7);
        const __m512i w7 = _mm512_unpackhi_epi64(u3, u7);
        const __m512i t8 = _mm512_unpacklo_epi16(z8, z9);
        const __m512i t9 = _mm512_unpackhi_epi16(z8, z9);
        const __m512i t10 = _mm512_unpacklo_epi16(z10, z11);
        const __m512i t11 = _mm512_unpackhi_epi16(z10, z11);
        const __m512i t12 = _mm512_unpacklo_epi16(z12, z13);
        const __m512i t13 = _mm512_unpackhi_epi16(z12, z13);
        const __m512i t14 = _mm512_unpacklo_epi16(z14, z15);
        const __m512i t15 = _mm512_unpackhi_epi16(z14, z15);

        const __m512i u8 = _mm512_unpacklo_epi32(t8, t10);
        const __m512i u9 = _mm512_unpackhi_epi32(t8, t10);
        const __m512i u10 = _mm512_unpacklo_epi32(t9, t11);
        const __m512i u11 = _mm512_unpackhi_epi32(t9, t11);
        const __m512i u12 = _mm512_unpacklo_epi32(t12, t14);
        const __m512i u13 = _mm512_unpackhi_epi32(t12, t14);
        const __m512i u14 = _mm512_unpacklo_epi32(t13, t15);
        const __m512i u15 = _mm512_unpackhi_epi32(t13, t15);

        const __m512i w8 = _mm512_unpacklo_epi64(u8, u12);
        const __m512i w9 = _mm512_unpackhi_epi64(u8, u12);
        const __m512i w10 = _mm512_unpacklo_epi64(u9, u13);
        const __m512i w11 = _mm512_unpackhi_epi64(u9, u13);
        const __m512i w12 = _mm512_unpacklo_epi64(u10, u14);
        const __m512i w13 = _mm512_unpackhi_epi64(u10, u14);
        const __m512i w14 = _mm512_unpacklo_epi64(u11, u15);
        const __m512i w15 = _mm512_unpackhi_epi64(u11, u15);

        const __m512i out0_0 = _mm512_shuffle_i32x4(w0, w8, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 0) {
          _mm512_storeu_si512((__m512i*) (packed_w + 0), out0_0);
        }
        const __m512i out0_8 = _mm512_shuffle_i32x4(w0, w8, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 8) {
          _mm512_storeu_si512((__m512i*) (packed_w + 256), out0_8);
        }
        const __m512i out0_1 = _mm512_shuffle_i32x4(w1, w9, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 1) {
          _mm512_storeu_si512((__m512i*) (packed_w + 32), out0_1);
        }
        const __m512i out0_9 = _mm512_shuffle_i32x4(w1, w9, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 9) {
          _mm512_storeu_si512((__m512i*) (packed_w + 288), out0_9);
        }
        const __m512i out0_2 = _mm512_shuffle_i32x4(w2, w10, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 2) {
          _mm512_storeu_si512((__m512i*) (packed_w + 64), out0_2);
        }
        const __m512i out0_10 = _mm512_shuffle_i32x4(w2, w10, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 10) {
          _mm512_storeu_si512((__m512i*) (packed_w + 320), out0_10);
        }
        const __m512i out0_3 = _mm512_shuffle_i32x4(w3, w11, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 3) {
          _mm512_storeu_si512((__m512i*) (packed_w + 96), out0_3);
        }
        const __m512i out0_11 = _mm512_shuffle_i32x4(w3, w11, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 11) {
          _mm512_storeu_si512((__m512i*) (packed_w + 352), out0_11);
        }
        const __m512i out0_4 = _mm512_shuffle_i32x4(w4, w12, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 4) {
          _mm512_storeu_si512((__m512i*) (packed_w + 128), out0_4);
        }
        const __m512i out0_12 = _mm512_shuffle_i32x4(w4, w12, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 12) {
          _mm512_storeu_si512((__m512i*) (packed_w + 384), out0_12);
        }
        const __m512i out0_5 = _mm512_shuffle_i32x4(w5, w13, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 5) {
          _mm512_storeu_si512((__m512i*) (packed_w + 160), out0_5);
        }
        const __m512i out0_13 = _mm512_shuffle_i32x4(w5, w13, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 13) {
          _mm512_storeu_si512((__m512i*) (packed_w + 416), out0_13);
        }
        const __m512i out0_6 = _mm512_shuffle_i32x4(w6, w14, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 6) {
          _mm512_storeu_si512((__m512i*) (packed_w + 192), out0_6);
        }
        const __m512i out0_14 = _mm512_shuffle_i32x4(w6, w14, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 14) {
          _mm512_storeu_si512((__m512i*) (packed_w + 448), out0_14);
        }
        const __m512i out0_7 = _mm512_shuffle_i32x4(w7, w15, _MM_SHUFFLE(2, 0, 2, 0));
        if (k > 7) {
          _mm512_storeu_si512((__m512i*) (packed_w + 224), out0_7);
        }
        const __m512i out0_15 = _mm512_shuffle_i32x4(w7, w15, _MM_SHUFFLE(3, 1, 3, 1));
        if (k > 15) {
          _mm512_storeu_si512((__m512i*) (packed_w + 480), out0_15);
        }
        packed_w += 32 * k;
      }
      packed_w = (uint16_t*) ((uintptr_t) packed_w + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
