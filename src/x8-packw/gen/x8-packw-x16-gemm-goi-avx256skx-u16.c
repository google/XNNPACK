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

void xnn_x8_packw_gemm_goi_ukernel_x16__avx256skx_u16(
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
  assert(nr == 16);
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  int8_t* out = (int8_t*) packed_weights;
  const uint32_t* b = (const uint32_t*) bias;

  do {
    const int8_t* wb = weights;
    size_t n = nc;
    // NC main loop multiple of 16
    for (; n >= 16; n -= 16) {
      const int8_t* w0 = wb;
      const int8_t* w1 = w0 + n_stride;
      const int8_t* w2 = w1 + n_stride;
      const int8_t* w3 = w2 + n_stride;
      const int8_t* w4 = w3 + n_stride;
      const int8_t* w5 = w4 + n_stride;
      const int8_t* w6 = w5 + n_stride;
      const int8_t* w7 = w6 + n_stride;
      const int8_t* w8 = w7 + n_stride;
      const int8_t* w9 = w8 + n_stride;
      const int8_t* w10 = w9 + n_stride;
      const int8_t* w11 = w10 + n_stride;
      const int8_t* w12 = w11 + n_stride;
      const int8_t* w13 = w12 + n_stride;
      const int8_t* w14 = w13 + n_stride;
      const int8_t* w15 = w14 + n_stride;

      uint32_t* packed_b = (uint32_t*) out;
      if XNN_LIKELY(b != NULL) {
        const __m256i vb0 = _mm256_loadu_si256((const __m256i*) (b + 0));
        const __m256i vb8 = _mm256_loadu_si256((const __m256i*) (b + 8));
        _mm256_storeu_si256((__m256i*) (packed_b + 0), vb0);
        _mm256_storeu_si256((__m256i*) (packed_b + 8), vb8);
        b += 16;
      } else {
        const __m256i vzero = _mm256_setzero_si256();
        _mm256_storeu_si256((__m256i*) (packed_b + 0), vzero);
        _mm256_storeu_si256((__m256i*) (packed_b + 8), vzero);
      }
      out += 16 * sizeof(uint32_t);


      // KC main loop multiple of 16
      size_t k = kc;
      for (; k >= 16; k -= 16) {
        const __m128i v0_0_lo = _mm_loadu_si128((const __m128i*) w0);
        const __m128i v0_0_hi = _mm_loadu_si128((const __m128i*) w8);
        const __m256i v0_0 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_0_lo), v0_0_hi, 1);
        w0 += 16;
        w8 += 16;
        const __m128i v0_1_lo = _mm_loadu_si128((const __m128i*) w1);
        const __m128i v0_1_hi = _mm_loadu_si128((const __m128i*) w9);
        const __m256i v0_1 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_1_lo), v0_1_hi, 1);
        w1 += 16;
        w9 += 16;
        const __m128i v0_2_lo = _mm_loadu_si128((const __m128i*) w2);
        const __m128i v0_2_hi = _mm_loadu_si128((const __m128i*) w10);
        const __m256i v0_2 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_2_lo), v0_2_hi, 1);
        w2 += 16;
        w10 += 16;
        const __m128i v0_3_lo = _mm_loadu_si128((const __m128i*) w3);
        const __m128i v0_3_hi = _mm_loadu_si128((const __m128i*) w11);
        const __m256i v0_3 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_3_lo), v0_3_hi, 1);
        w3 += 16;
        w11 += 16;
        const __m128i v0_4_lo = _mm_loadu_si128((const __m128i*) w4);
        const __m128i v0_4_hi = _mm_loadu_si128((const __m128i*) w12);
        const __m256i v0_4 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_4_lo), v0_4_hi, 1);
        w4 += 16;
        w12 += 16;
        const __m128i v0_5_lo = _mm_loadu_si128((const __m128i*) w5);
        const __m128i v0_5_hi = _mm_loadu_si128((const __m128i*) w13);
        const __m256i v0_5 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_5_lo), v0_5_hi, 1);
        w5 += 16;
        w13 += 16;
        const __m128i v0_6_lo = _mm_loadu_si128((const __m128i*) w6);
        const __m128i v0_6_hi = _mm_loadu_si128((const __m128i*) w14);
        const __m256i v0_6 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_6_lo), v0_6_hi, 1);
        w6 += 16;
        w14 += 16;
        const __m128i v0_7_lo = _mm_loadu_si128((const __m128i*) w7);
        const __m128i v0_7_hi = _mm_loadu_si128((const __m128i*) w15);
        const __m256i v0_7 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_7_lo), v0_7_hi, 1);
        w7 += 16;
        w15 += 16;

        const __m256i t0_0 = _mm256_unpacklo_epi8(v0_0, v0_1);
        const __m256i t0_1 = _mm256_unpackhi_epi8(v0_0, v0_1);
        const __m256i t0_2 = _mm256_unpacklo_epi8(v0_2, v0_3);
        const __m256i t0_3 = _mm256_unpackhi_epi8(v0_2, v0_3);
        const __m256i t0_4 = _mm256_unpacklo_epi8(v0_4, v0_5);
        const __m256i t0_5 = _mm256_unpackhi_epi8(v0_4, v0_5);
        const __m256i t0_6 = _mm256_unpacklo_epi8(v0_6, v0_7);
        const __m256i t0_7 = _mm256_unpackhi_epi8(v0_6, v0_7);

        const __m256i u0_0 = _mm256_unpacklo_epi16(t0_0, t0_2);
        const __m256i u0_1 = _mm256_unpackhi_epi16(t0_0, t0_2);
        const __m256i u0_2 = _mm256_unpacklo_epi16(t0_1, t0_3);
        const __m256i u0_3 = _mm256_unpackhi_epi16(t0_1, t0_3);
        const __m256i u0_4 = _mm256_unpacklo_epi16(t0_4, t0_6);
        const __m256i u0_5 = _mm256_unpackhi_epi16(t0_4, t0_6);
        const __m256i u0_6 = _mm256_unpacklo_epi16(t0_5, t0_7);
        const __m256i u0_7 = _mm256_unpackhi_epi16(t0_5, t0_7);

        const __m256i s0_0 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi32(u0_0, u0_4), 0xD8);
        const __m256i s0_1 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi32(u0_0, u0_4), 0xD8);
        const __m256i s0_2 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi32(u0_1, u0_5), 0xD8);
        const __m256i s0_3 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi32(u0_1, u0_5), 0xD8);
        const __m256i s0_4 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi32(u0_2, u0_6), 0xD8);
        const __m256i s0_5 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi32(u0_2, u0_6), 0xD8);
        const __m256i s0_6 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi32(u0_3, u0_7), 0xD8);
        const __m256i s0_7 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi32(u0_3, u0_7), 0xD8);

        _mm256_storeu_si256((__m256i*) (out + 0), s0_0);
        _mm256_storeu_si256((__m256i*) (out + 32), s0_1);
        _mm256_storeu_si256((__m256i*) (out + 64), s0_2);
        _mm256_storeu_si256((__m256i*) (out + 96), s0_3);
        _mm256_storeu_si256((__m256i*) (out + 128), s0_4);
        _mm256_storeu_si256((__m256i*) (out + 160), s0_5);
        _mm256_storeu_si256((__m256i*) (out + 192), s0_6);
        _mm256_storeu_si256((__m256i*) (out + 224), s0_7);
        out += 256;
      }

      // KC remainder (1..15)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 15);

        if (k & 8) {
          const __m128i v0_0_lo = _mm_loadl_epi64((const __m128i*) w0);
          const __m128i v0_0_hi = _mm_loadl_epi64((const __m128i*) w8);
          const __m256i v0_0 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_0_lo), v0_0_hi, 1);
          w0 += 8;
          w8 += 8;
          const __m128i v0_1_lo = _mm_loadl_epi64((const __m128i*) w1);
          const __m128i v0_1_hi = _mm_loadl_epi64((const __m128i*) w9);
          const __m256i v0_1 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_1_lo), v0_1_hi, 1);
          w1 += 8;
          w9 += 8;
          const __m128i v0_2_lo = _mm_loadl_epi64((const __m128i*) w2);
          const __m128i v0_2_hi = _mm_loadl_epi64((const __m128i*) w10);
          const __m256i v0_2 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_2_lo), v0_2_hi, 1);
          w2 += 8;
          w10 += 8;
          const __m128i v0_3_lo = _mm_loadl_epi64((const __m128i*) w3);
          const __m128i v0_3_hi = _mm_loadl_epi64((const __m128i*) w11);
          const __m256i v0_3 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_3_lo), v0_3_hi, 1);
          w3 += 8;
          w11 += 8;
          const __m128i v0_4_lo = _mm_loadl_epi64((const __m128i*) w4);
          const __m128i v0_4_hi = _mm_loadl_epi64((const __m128i*) w12);
          const __m256i v0_4 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_4_lo), v0_4_hi, 1);
          w4 += 8;
          w12 += 8;
          const __m128i v0_5_lo = _mm_loadl_epi64((const __m128i*) w5);
          const __m128i v0_5_hi = _mm_loadl_epi64((const __m128i*) w13);
          const __m256i v0_5 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_5_lo), v0_5_hi, 1);
          w5 += 8;
          w13 += 8;
          const __m128i v0_6_lo = _mm_loadl_epi64((const __m128i*) w6);
          const __m128i v0_6_hi = _mm_loadl_epi64((const __m128i*) w14);
          const __m256i v0_6 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_6_lo), v0_6_hi, 1);
          w6 += 8;
          w14 += 8;
          const __m128i v0_7_lo = _mm_loadl_epi64((const __m128i*) w7);
          const __m128i v0_7_hi = _mm_loadl_epi64((const __m128i*) w15);
          const __m256i v0_7 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_7_lo), v0_7_hi, 1);
          w7 += 8;
          w15 += 8;

          const __m256i t0_0 = _mm256_unpacklo_epi8(v0_0, v0_1);
          const __m256i t0_2 = _mm256_unpacklo_epi8(v0_2, v0_3);
          const __m256i t0_4 = _mm256_unpacklo_epi8(v0_4, v0_5);
          const __m256i t0_6 = _mm256_unpacklo_epi8(v0_6, v0_7);

          const __m256i u0_0 = _mm256_unpacklo_epi16(t0_0, t0_2);
          const __m256i u0_1 = _mm256_unpackhi_epi16(t0_0, t0_2);
          const __m256i u0_4 = _mm256_unpacklo_epi16(t0_4, t0_6);
          const __m256i u0_5 = _mm256_unpackhi_epi16(t0_4, t0_6);

          const __m256i s0_0 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi32(u0_0, u0_4), 0xD8);
          const __m256i s0_1 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi32(u0_0, u0_4), 0xD8);
          const __m256i s0_2 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi32(u0_1, u0_5), 0xD8);
          const __m256i s0_3 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi32(u0_1, u0_5), 0xD8);

          _mm256_storeu_si256((__m256i*) (out + 0), s0_0);
          _mm256_storeu_si256((__m256i*) (out + 32), s0_1);
          _mm256_storeu_si256((__m256i*) (out + 64), s0_2);
          _mm256_storeu_si256((__m256i*) (out + 96), s0_3);
          out += 128;
        }

        if (k & 4) {
          const __m128i v0_0_lo = _mm_cvtsi32_si128((int) unaligned_load_u32(w0));
          const __m128i v0_0_hi = _mm_cvtsi32_si128((int) unaligned_load_u32(w8));
          const __m256i v0_0 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_0_lo), v0_0_hi, 1);
          w0 += 4;
          w8 += 4;
          const __m128i v0_1_lo = _mm_cvtsi32_si128((int) unaligned_load_u32(w1));
          const __m128i v0_1_hi = _mm_cvtsi32_si128((int) unaligned_load_u32(w9));
          const __m256i v0_1 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_1_lo), v0_1_hi, 1);
          w1 += 4;
          w9 += 4;
          const __m128i v0_2_lo = _mm_cvtsi32_si128((int) unaligned_load_u32(w2));
          const __m128i v0_2_hi = _mm_cvtsi32_si128((int) unaligned_load_u32(w10));
          const __m256i v0_2 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_2_lo), v0_2_hi, 1);
          w2 += 4;
          w10 += 4;
          const __m128i v0_3_lo = _mm_cvtsi32_si128((int) unaligned_load_u32(w3));
          const __m128i v0_3_hi = _mm_cvtsi32_si128((int) unaligned_load_u32(w11));
          const __m256i v0_3 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_3_lo), v0_3_hi, 1);
          w3 += 4;
          w11 += 4;
          const __m128i v0_4_lo = _mm_cvtsi32_si128((int) unaligned_load_u32(w4));
          const __m128i v0_4_hi = _mm_cvtsi32_si128((int) unaligned_load_u32(w12));
          const __m256i v0_4 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_4_lo), v0_4_hi, 1);
          w4 += 4;
          w12 += 4;
          const __m128i v0_5_lo = _mm_cvtsi32_si128((int) unaligned_load_u32(w5));
          const __m128i v0_5_hi = _mm_cvtsi32_si128((int) unaligned_load_u32(w13));
          const __m256i v0_5 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_5_lo), v0_5_hi, 1);
          w5 += 4;
          w13 += 4;
          const __m128i v0_6_lo = _mm_cvtsi32_si128((int) unaligned_load_u32(w6));
          const __m128i v0_6_hi = _mm_cvtsi32_si128((int) unaligned_load_u32(w14));
          const __m256i v0_6 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_6_lo), v0_6_hi, 1);
          w6 += 4;
          w14 += 4;
          const __m128i v0_7_lo = _mm_cvtsi32_si128((int) unaligned_load_u32(w7));
          const __m128i v0_7_hi = _mm_cvtsi32_si128((int) unaligned_load_u32(w15));
          const __m256i v0_7 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_7_lo), v0_7_hi, 1);
          w7 += 4;
          w15 += 4;

          const __m256i t0_0 = _mm256_unpacklo_epi8(v0_0, v0_1);
          const __m256i t0_2 = _mm256_unpacklo_epi8(v0_2, v0_3);
          const __m256i t0_4 = _mm256_unpacklo_epi8(v0_4, v0_5);
          const __m256i t0_6 = _mm256_unpacklo_epi8(v0_6, v0_7);

          const __m256i u0_0 = _mm256_unpacklo_epi16(t0_0, t0_2);
          const __m256i u0_4 = _mm256_unpacklo_epi16(t0_4, t0_6);

          const __m256i s0_0 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi32(u0_0, u0_4), 0xD8);
          const __m256i s0_1 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi32(u0_0, u0_4), 0xD8);

          _mm256_storeu_si256((__m256i*) (out + 0), s0_0);
          _mm256_storeu_si256((__m256i*) (out + 32), s0_1);
          out += 64;
        }

        if (k & 2) {
          const __m128i v0_0_lo = _mm_cvtsi32_si128((int) unaligned_load_u16(w0));
          const __m128i v0_0_hi = _mm_cvtsi32_si128((int) unaligned_load_u16(w8));
          const __m256i v0_0 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_0_lo), v0_0_hi, 1);
          w0 += 2;
          w8 += 2;
          const __m128i v0_1_lo = _mm_cvtsi32_si128((int) unaligned_load_u16(w1));
          const __m128i v0_1_hi = _mm_cvtsi32_si128((int) unaligned_load_u16(w9));
          const __m256i v0_1 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_1_lo), v0_1_hi, 1);
          w1 += 2;
          w9 += 2;
          const __m128i v0_2_lo = _mm_cvtsi32_si128((int) unaligned_load_u16(w2));
          const __m128i v0_2_hi = _mm_cvtsi32_si128((int) unaligned_load_u16(w10));
          const __m256i v0_2 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_2_lo), v0_2_hi, 1);
          w2 += 2;
          w10 += 2;
          const __m128i v0_3_lo = _mm_cvtsi32_si128((int) unaligned_load_u16(w3));
          const __m128i v0_3_hi = _mm_cvtsi32_si128((int) unaligned_load_u16(w11));
          const __m256i v0_3 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_3_lo), v0_3_hi, 1);
          w3 += 2;
          w11 += 2;
          const __m128i v0_4_lo = _mm_cvtsi32_si128((int) unaligned_load_u16(w4));
          const __m128i v0_4_hi = _mm_cvtsi32_si128((int) unaligned_load_u16(w12));
          const __m256i v0_4 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_4_lo), v0_4_hi, 1);
          w4 += 2;
          w12 += 2;
          const __m128i v0_5_lo = _mm_cvtsi32_si128((int) unaligned_load_u16(w5));
          const __m128i v0_5_hi = _mm_cvtsi32_si128((int) unaligned_load_u16(w13));
          const __m256i v0_5 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_5_lo), v0_5_hi, 1);
          w5 += 2;
          w13 += 2;
          const __m128i v0_6_lo = _mm_cvtsi32_si128((int) unaligned_load_u16(w6));
          const __m128i v0_6_hi = _mm_cvtsi32_si128((int) unaligned_load_u16(w14));
          const __m256i v0_6 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_6_lo), v0_6_hi, 1);
          w6 += 2;
          w14 += 2;
          const __m128i v0_7_lo = _mm_cvtsi32_si128((int) unaligned_load_u16(w7));
          const __m128i v0_7_hi = _mm_cvtsi32_si128((int) unaligned_load_u16(w15));
          const __m256i v0_7 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_7_lo), v0_7_hi, 1);
          w7 += 2;
          w15 += 2;

          const __m256i t0_0 = _mm256_unpacklo_epi8(v0_0, v0_1);
          const __m256i t0_2 = _mm256_unpacklo_epi8(v0_2, v0_3);
          const __m256i t0_4 = _mm256_unpacklo_epi8(v0_4, v0_5);
          const __m256i t0_6 = _mm256_unpacklo_epi8(v0_6, v0_7);

          const __m256i u0_0 = _mm256_unpacklo_epi16(t0_0, t0_2);
          const __m256i u0_4 = _mm256_unpacklo_epi16(t0_4, t0_6);

          const __m256i s0_0 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi32(u0_0, u0_4), 0xD8);

          _mm256_storeu_si256((__m256i*) out, s0_0);
          out += 32;
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
          out[8] = *w8++;
          out[9] = *w9++;
          out[10] = *w10++;
          out[11] = *w11++;
          out[12] = *w12++;
          out[13] = *w13++;
          out[14] = *w14++;
          out[15] = *w15++;
          out += 16;
        }
      }

      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      wb += 16 * n_stride;
    }
    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 15);
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
      const int8_t* w8 = w7 + n_stride;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const int8_t* w9 = w8 + n_stride;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const int8_t* w10 = w9 + n_stride;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const int8_t* w11 = w10 + n_stride;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const int8_t* w12 = w11 + n_stride;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const int8_t* w13 = w12 + n_stride;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const int8_t* w14 = w13 + n_stride;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }
      const int8_t* w15 = w14 + n_stride;
      if XNN_UNPREDICTABLE(n < 16) {
        w15 = w14;
      }

      uint32_t* packed_b = (uint32_t*) out;
      const __m256i vzero = _mm256_setzero_si256();
      _mm256_storeu_si256((__m256i*) (packed_b + 0), vzero);
      _mm256_storeu_si256((__m256i*) (packed_b + 8), vzero);
      if XNN_LIKELY(b != NULL) {
        for (size_t nb = 0; nb < n; ++nb) {
          packed_b[nb] = b[nb];
        }
        b += n;
      }
      out += 16 * sizeof(uint32_t);


      // KC main loop multiple of 16
      size_t k = kc;
      for (; k >= 16; k -= 16) {
        const __m128i v0_0_lo = _mm_loadu_si128((const __m128i*) w0);
        const __m128i v0_0_hi = _mm_loadu_si128((const __m128i*) w8);
        const __m256i v0_0 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_0_lo), v0_0_hi, 1);
        w0 += 16;
        w8 += 16;
        const __m128i v0_1_lo = _mm_loadu_si128((const __m128i*) w1);
        const __m128i v0_1_hi = _mm_loadu_si128((const __m128i*) w9);
        const __m256i v0_1 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_1_lo), v0_1_hi, 1);
        w1 += 16;
        w9 += 16;
        const __m128i v0_2_lo = _mm_loadu_si128((const __m128i*) w2);
        const __m128i v0_2_hi = _mm_loadu_si128((const __m128i*) w10);
        const __m256i v0_2 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_2_lo), v0_2_hi, 1);
        w2 += 16;
        w10 += 16;
        const __m128i v0_3_lo = _mm_loadu_si128((const __m128i*) w3);
        const __m128i v0_3_hi = _mm_loadu_si128((const __m128i*) w11);
        const __m256i v0_3 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_3_lo), v0_3_hi, 1);
        w3 += 16;
        w11 += 16;
        const __m128i v0_4_lo = _mm_loadu_si128((const __m128i*) w4);
        const __m128i v0_4_hi = _mm_loadu_si128((const __m128i*) w12);
        const __m256i v0_4 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_4_lo), v0_4_hi, 1);
        w4 += 16;
        w12 += 16;
        const __m128i v0_5_lo = _mm_loadu_si128((const __m128i*) w5);
        const __m128i v0_5_hi = _mm_loadu_si128((const __m128i*) w13);
        const __m256i v0_5 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_5_lo), v0_5_hi, 1);
        w5 += 16;
        w13 += 16;
        const __m128i v0_6_lo = _mm_loadu_si128((const __m128i*) w6);
        const __m128i v0_6_hi = _mm_loadu_si128((const __m128i*) w14);
        const __m256i v0_6 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_6_lo), v0_6_hi, 1);
        w6 += 16;
        w14 += 16;
        const __m128i v0_7_lo = _mm_loadu_si128((const __m128i*) w7);
        const __m128i v0_7_hi = _mm_loadu_si128((const __m128i*) w15);
        const __m256i v0_7 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_7_lo), v0_7_hi, 1);
        w7 += 16;
        w15 += 16;

        const __m256i t0_0 = _mm256_unpacklo_epi8(v0_0, v0_1);
        const __m256i t0_1 = _mm256_unpackhi_epi8(v0_0, v0_1);
        const __m256i t0_2 = _mm256_unpacklo_epi8(v0_2, v0_3);
        const __m256i t0_3 = _mm256_unpackhi_epi8(v0_2, v0_3);
        const __m256i t0_4 = _mm256_unpacklo_epi8(v0_4, v0_5);
        const __m256i t0_5 = _mm256_unpackhi_epi8(v0_4, v0_5);
        const __m256i t0_6 = _mm256_unpacklo_epi8(v0_6, v0_7);
        const __m256i t0_7 = _mm256_unpackhi_epi8(v0_6, v0_7);

        const __m256i u0_0 = _mm256_unpacklo_epi16(t0_0, t0_2);
        const __m256i u0_1 = _mm256_unpackhi_epi16(t0_0, t0_2);
        const __m256i u0_2 = _mm256_unpacklo_epi16(t0_1, t0_3);
        const __m256i u0_3 = _mm256_unpackhi_epi16(t0_1, t0_3);
        const __m256i u0_4 = _mm256_unpacklo_epi16(t0_4, t0_6);
        const __m256i u0_5 = _mm256_unpackhi_epi16(t0_4, t0_6);
        const __m256i u0_6 = _mm256_unpacklo_epi16(t0_5, t0_7);
        const __m256i u0_7 = _mm256_unpackhi_epi16(t0_5, t0_7);

        const __m256i s0_0 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi32(u0_0, u0_4), 0xD8);
        const __m256i s0_1 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi32(u0_0, u0_4), 0xD8);
        const __m256i s0_2 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi32(u0_1, u0_5), 0xD8);
        const __m256i s0_3 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi32(u0_1, u0_5), 0xD8);
        const __m256i s0_4 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi32(u0_2, u0_6), 0xD8);
        const __m256i s0_5 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi32(u0_2, u0_6), 0xD8);
        const __m256i s0_6 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi32(u0_3, u0_7), 0xD8);
        const __m256i s0_7 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi32(u0_3, u0_7), 0xD8);

        _mm256_storeu_si256((__m256i*) (out + 0), s0_0);
        _mm256_storeu_si256((__m256i*) (out + 32), s0_1);
        _mm256_storeu_si256((__m256i*) (out + 64), s0_2);
        _mm256_storeu_si256((__m256i*) (out + 96), s0_3);
        _mm256_storeu_si256((__m256i*) (out + 128), s0_4);
        _mm256_storeu_si256((__m256i*) (out + 160), s0_5);
        _mm256_storeu_si256((__m256i*) (out + 192), s0_6);
        _mm256_storeu_si256((__m256i*) (out + 224), s0_7);
        out += 256;
      }

      // KC remainder (1..15)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 15);

        if (k & 8) {
          const __m128i v0_0_lo = _mm_loadl_epi64((const __m128i*) w0);
          const __m128i v0_0_hi = _mm_loadl_epi64((const __m128i*) w8);
          const __m256i v0_0 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_0_lo), v0_0_hi, 1);
          w0 += 8;
          w8 += 8;
          const __m128i v0_1_lo = _mm_loadl_epi64((const __m128i*) w1);
          const __m128i v0_1_hi = _mm_loadl_epi64((const __m128i*) w9);
          const __m256i v0_1 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_1_lo), v0_1_hi, 1);
          w1 += 8;
          w9 += 8;
          const __m128i v0_2_lo = _mm_loadl_epi64((const __m128i*) w2);
          const __m128i v0_2_hi = _mm_loadl_epi64((const __m128i*) w10);
          const __m256i v0_2 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_2_lo), v0_2_hi, 1);
          w2 += 8;
          w10 += 8;
          const __m128i v0_3_lo = _mm_loadl_epi64((const __m128i*) w3);
          const __m128i v0_3_hi = _mm_loadl_epi64((const __m128i*) w11);
          const __m256i v0_3 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_3_lo), v0_3_hi, 1);
          w3 += 8;
          w11 += 8;
          const __m128i v0_4_lo = _mm_loadl_epi64((const __m128i*) w4);
          const __m128i v0_4_hi = _mm_loadl_epi64((const __m128i*) w12);
          const __m256i v0_4 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_4_lo), v0_4_hi, 1);
          w4 += 8;
          w12 += 8;
          const __m128i v0_5_lo = _mm_loadl_epi64((const __m128i*) w5);
          const __m128i v0_5_hi = _mm_loadl_epi64((const __m128i*) w13);
          const __m256i v0_5 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_5_lo), v0_5_hi, 1);
          w5 += 8;
          w13 += 8;
          const __m128i v0_6_lo = _mm_loadl_epi64((const __m128i*) w6);
          const __m128i v0_6_hi = _mm_loadl_epi64((const __m128i*) w14);
          const __m256i v0_6 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_6_lo), v0_6_hi, 1);
          w6 += 8;
          w14 += 8;
          const __m128i v0_7_lo = _mm_loadl_epi64((const __m128i*) w7);
          const __m128i v0_7_hi = _mm_loadl_epi64((const __m128i*) w15);
          const __m256i v0_7 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_7_lo), v0_7_hi, 1);
          w7 += 8;
          w15 += 8;

          const __m256i t0_0 = _mm256_unpacklo_epi8(v0_0, v0_1);
          const __m256i t0_2 = _mm256_unpacklo_epi8(v0_2, v0_3);
          const __m256i t0_4 = _mm256_unpacklo_epi8(v0_4, v0_5);
          const __m256i t0_6 = _mm256_unpacklo_epi8(v0_6, v0_7);

          const __m256i u0_0 = _mm256_unpacklo_epi16(t0_0, t0_2);
          const __m256i u0_1 = _mm256_unpackhi_epi16(t0_0, t0_2);
          const __m256i u0_4 = _mm256_unpacklo_epi16(t0_4, t0_6);
          const __m256i u0_5 = _mm256_unpackhi_epi16(t0_4, t0_6);

          const __m256i s0_0 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi32(u0_0, u0_4), 0xD8);
          const __m256i s0_1 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi32(u0_0, u0_4), 0xD8);
          const __m256i s0_2 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi32(u0_1, u0_5), 0xD8);
          const __m256i s0_3 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi32(u0_1, u0_5), 0xD8);

          _mm256_storeu_si256((__m256i*) (out + 0), s0_0);
          _mm256_storeu_si256((__m256i*) (out + 32), s0_1);
          _mm256_storeu_si256((__m256i*) (out + 64), s0_2);
          _mm256_storeu_si256((__m256i*) (out + 96), s0_3);
          out += 128;
        }

        if (k & 4) {
          const __m128i v0_0_lo = _mm_cvtsi32_si128((int) unaligned_load_u32(w0));
          const __m128i v0_0_hi = _mm_cvtsi32_si128((int) unaligned_load_u32(w8));
          const __m256i v0_0 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_0_lo), v0_0_hi, 1);
          w0 += 4;
          w8 += 4;
          const __m128i v0_1_lo = _mm_cvtsi32_si128((int) unaligned_load_u32(w1));
          const __m128i v0_1_hi = _mm_cvtsi32_si128((int) unaligned_load_u32(w9));
          const __m256i v0_1 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_1_lo), v0_1_hi, 1);
          w1 += 4;
          w9 += 4;
          const __m128i v0_2_lo = _mm_cvtsi32_si128((int) unaligned_load_u32(w2));
          const __m128i v0_2_hi = _mm_cvtsi32_si128((int) unaligned_load_u32(w10));
          const __m256i v0_2 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_2_lo), v0_2_hi, 1);
          w2 += 4;
          w10 += 4;
          const __m128i v0_3_lo = _mm_cvtsi32_si128((int) unaligned_load_u32(w3));
          const __m128i v0_3_hi = _mm_cvtsi32_si128((int) unaligned_load_u32(w11));
          const __m256i v0_3 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_3_lo), v0_3_hi, 1);
          w3 += 4;
          w11 += 4;
          const __m128i v0_4_lo = _mm_cvtsi32_si128((int) unaligned_load_u32(w4));
          const __m128i v0_4_hi = _mm_cvtsi32_si128((int) unaligned_load_u32(w12));
          const __m256i v0_4 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_4_lo), v0_4_hi, 1);
          w4 += 4;
          w12 += 4;
          const __m128i v0_5_lo = _mm_cvtsi32_si128((int) unaligned_load_u32(w5));
          const __m128i v0_5_hi = _mm_cvtsi32_si128((int) unaligned_load_u32(w13));
          const __m256i v0_5 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_5_lo), v0_5_hi, 1);
          w5 += 4;
          w13 += 4;
          const __m128i v0_6_lo = _mm_cvtsi32_si128((int) unaligned_load_u32(w6));
          const __m128i v0_6_hi = _mm_cvtsi32_si128((int) unaligned_load_u32(w14));
          const __m256i v0_6 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_6_lo), v0_6_hi, 1);
          w6 += 4;
          w14 += 4;
          const __m128i v0_7_lo = _mm_cvtsi32_si128((int) unaligned_load_u32(w7));
          const __m128i v0_7_hi = _mm_cvtsi32_si128((int) unaligned_load_u32(w15));
          const __m256i v0_7 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_7_lo), v0_7_hi, 1);
          w7 += 4;
          w15 += 4;

          const __m256i t0_0 = _mm256_unpacklo_epi8(v0_0, v0_1);
          const __m256i t0_2 = _mm256_unpacklo_epi8(v0_2, v0_3);
          const __m256i t0_4 = _mm256_unpacklo_epi8(v0_4, v0_5);
          const __m256i t0_6 = _mm256_unpacklo_epi8(v0_6, v0_7);

          const __m256i u0_0 = _mm256_unpacklo_epi16(t0_0, t0_2);
          const __m256i u0_4 = _mm256_unpacklo_epi16(t0_4, t0_6);

          const __m256i s0_0 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi32(u0_0, u0_4), 0xD8);
          const __m256i s0_1 = _mm256_permute4x64_epi64(_mm256_unpackhi_epi32(u0_0, u0_4), 0xD8);

          _mm256_storeu_si256((__m256i*) (out + 0), s0_0);
          _mm256_storeu_si256((__m256i*) (out + 32), s0_1);
          out += 64;
        }

        if (k & 2) {
          const __m128i v0_0_lo = _mm_cvtsi32_si128((int) unaligned_load_u16(w0));
          const __m128i v0_0_hi = _mm_cvtsi32_si128((int) unaligned_load_u16(w8));
          const __m256i v0_0 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_0_lo), v0_0_hi, 1);
          w0 += 2;
          w8 += 2;
          const __m128i v0_1_lo = _mm_cvtsi32_si128((int) unaligned_load_u16(w1));
          const __m128i v0_1_hi = _mm_cvtsi32_si128((int) unaligned_load_u16(w9));
          const __m256i v0_1 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_1_lo), v0_1_hi, 1);
          w1 += 2;
          w9 += 2;
          const __m128i v0_2_lo = _mm_cvtsi32_si128((int) unaligned_load_u16(w2));
          const __m128i v0_2_hi = _mm_cvtsi32_si128((int) unaligned_load_u16(w10));
          const __m256i v0_2 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_2_lo), v0_2_hi, 1);
          w2 += 2;
          w10 += 2;
          const __m128i v0_3_lo = _mm_cvtsi32_si128((int) unaligned_load_u16(w3));
          const __m128i v0_3_hi = _mm_cvtsi32_si128((int) unaligned_load_u16(w11));
          const __m256i v0_3 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_3_lo), v0_3_hi, 1);
          w3 += 2;
          w11 += 2;
          const __m128i v0_4_lo = _mm_cvtsi32_si128((int) unaligned_load_u16(w4));
          const __m128i v0_4_hi = _mm_cvtsi32_si128((int) unaligned_load_u16(w12));
          const __m256i v0_4 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_4_lo), v0_4_hi, 1);
          w4 += 2;
          w12 += 2;
          const __m128i v0_5_lo = _mm_cvtsi32_si128((int) unaligned_load_u16(w5));
          const __m128i v0_5_hi = _mm_cvtsi32_si128((int) unaligned_load_u16(w13));
          const __m256i v0_5 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_5_lo), v0_5_hi, 1);
          w5 += 2;
          w13 += 2;
          const __m128i v0_6_lo = _mm_cvtsi32_si128((int) unaligned_load_u16(w6));
          const __m128i v0_6_hi = _mm_cvtsi32_si128((int) unaligned_load_u16(w14));
          const __m256i v0_6 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_6_lo), v0_6_hi, 1);
          w6 += 2;
          w14 += 2;
          const __m128i v0_7_lo = _mm_cvtsi32_si128((int) unaligned_load_u16(w7));
          const __m128i v0_7_hi = _mm_cvtsi32_si128((int) unaligned_load_u16(w15));
          const __m256i v0_7 = _mm256_inserti128_si256(_mm256_castsi128_si256(v0_7_lo), v0_7_hi, 1);
          w7 += 2;
          w15 += 2;

          const __m256i t0_0 = _mm256_unpacklo_epi8(v0_0, v0_1);
          const __m256i t0_2 = _mm256_unpacklo_epi8(v0_2, v0_3);
          const __m256i t0_4 = _mm256_unpacklo_epi8(v0_4, v0_5);
          const __m256i t0_6 = _mm256_unpacklo_epi8(v0_6, v0_7);

          const __m256i u0_0 = _mm256_unpacklo_epi16(t0_0, t0_2);
          const __m256i u0_4 = _mm256_unpacklo_epi16(t0_4, t0_6);

          const __m256i s0_0 = _mm256_permute4x64_epi64(_mm256_unpacklo_epi32(u0_0, u0_4), 0xD8);

          _mm256_storeu_si256((__m256i*) out, s0_0);
          out += 32;
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
          out[8] = *w8++;
          out[9] = *w9++;
          out[10] = *w10++;
          out[11] = *w11++;
          out[12] = *w12++;
          out[13] = *w13++;
          out[14] = *w14++;
          out[15] = *w15++;
          out += 16;
        }
      }

      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }

    weights += nc * n_stride;
  } while (--g != 0);
}
