// Auto-generated file. Do not edit!
//   Template: src/x32-transposec/sse2.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/transpose.h"
#include "xnnpack/unaligned.h"


void xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2(
    const uint16_t* input,
    uint16_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint16_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint16_t));

  const size_t tile_height = 8;
  const size_t tile_width = 8;
  const size_t tile_hbytes = tile_height * sizeof(uint16_t);
  const size_t tile_wbytes = tile_width * sizeof(uint16_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint16_t);

  const uint16_t* i0 = input;
  uint16_t* o0 = (uint16_t*) output;
  uint16_t* o1 = (uint16_t*) ((uintptr_t) o0 + output_stride);
  uint16_t* o2 = (uint16_t*) ((uintptr_t) o1 + output_stride);
  uint16_t* o3 = (uint16_t*) ((uintptr_t) o2 + output_stride);
  uint16_t* o4 = (uint16_t*) ((uintptr_t) o3 + output_stride);
  uint16_t* o5 = (uint16_t*) ((uintptr_t) o4 + output_stride);
  uint16_t* o6 = (uint16_t*) ((uintptr_t) o5 + output_stride);
  uint16_t* o7 = (uint16_t*) ((uintptr_t) o6 + output_stride);

  do {
    if XNN_UNPREDICTABLE(block_width < 2) {
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 2) {
      o2 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 4) {
      o3 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 4) {
      o4 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 6) {
      o5 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 6) {
      o6 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 8) {
      o7 = o0;
    }
    size_t bh = block_height;
    for (; bh >= 8; bh -= 8) {
      const __m128i v3_0 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v3_1 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v3_2 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v3_3 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v3_4 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v3_5 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v3_6 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v3_7 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);

      const __m128i v2_0 = _mm_unpacklo_epi16(v3_0, v3_1);
      const __m128i v2_1 = _mm_unpackhi_epi16(v3_0, v3_1);
      const __m128i v2_2 = _mm_unpacklo_epi16(v3_2, v3_3);
      const __m128i v2_3 = _mm_unpackhi_epi16(v3_2, v3_3);
      const __m128i v2_4 = _mm_unpacklo_epi16(v3_4, v3_5);
      const __m128i v2_5 = _mm_unpackhi_epi16(v3_4, v3_5);
      const __m128i v2_6 = _mm_unpacklo_epi16(v3_6, v3_7);
      const __m128i v2_7 = _mm_unpackhi_epi16(v3_6, v3_7);

      const __m128i v1_0 = _mm_unpacklo_epi32(v2_0, v2_2);
      const __m128i v1_1 = _mm_unpackhi_epi32(v2_0, v2_2);
      const __m128i v1_2 = _mm_unpacklo_epi32(v2_1, v2_3);
      const __m128i v1_3 = _mm_unpackhi_epi32(v2_1, v2_3);
      const __m128i v1_4 = _mm_unpacklo_epi32(v2_4, v2_6);
      const __m128i v1_5 = _mm_unpackhi_epi32(v2_4, v2_6);
      const __m128i v1_6 = _mm_unpacklo_epi32(v2_5, v2_7);
      const __m128i v1_7 = _mm_unpackhi_epi32(v2_5, v2_7);

      const __m128i v0_0 = _mm_unpacklo_epi64(v1_0, v1_4);
      const __m128i v0_1 = _mm_unpackhi_epi64(v1_0, v1_4);
      const __m128i v0_2 = _mm_unpacklo_epi64(v1_1, v1_5);
      const __m128i v0_3 = _mm_unpackhi_epi64(v1_1, v1_5);
      const __m128i v0_4 = _mm_unpacklo_epi64(v1_2, v1_6);
      const __m128i v0_5 = _mm_unpackhi_epi64(v1_2, v1_6);
      const __m128i v0_6 = _mm_unpacklo_epi64(v1_3, v1_7);
      const __m128i v0_7 = _mm_unpackhi_epi64(v1_3, v1_7);


      _mm_storeu_si128((__m128i*) o7, v0_7);
      o7 = (uint16_t*) ((uintptr_t) o7 + tile_hbytes);
      _mm_storeu_si128((__m128i*) o6, v0_6);
      o6 = (uint16_t*) ((uintptr_t) o6 + tile_hbytes);
      _mm_storeu_si128((__m128i*) o5, v0_5);
      o5 = (uint16_t*) ((uintptr_t) o5 + tile_hbytes);
      _mm_storeu_si128((__m128i*) o4, v0_4);
      o4 = (uint16_t*) ((uintptr_t) o4 + tile_hbytes);
      _mm_storeu_si128((__m128i*) o3, v0_3);
      o3 = (uint16_t*) ((uintptr_t) o3 + tile_hbytes);
      _mm_storeu_si128((__m128i*) o2, v0_2);
      o2 = (uint16_t*) ((uintptr_t) o2 + tile_hbytes);
      _mm_storeu_si128((__m128i*) o1, v0_1);
      o1 = (uint16_t*) ((uintptr_t) o1 + tile_hbytes);
      _mm_storeu_si128((__m128i*) o0, v0_0);
      o0 = (uint16_t*) ((uintptr_t) o0 + tile_hbytes);
    }
    if (bh != 0) {
      const __m128i v3_0 = _mm_loadu_si128((const __m128i*) i0);
      const uint16_t *i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const __m128i v3_1 = _mm_loadu_si128((const __m128i*) i1);
      const uint16_t *i2 = (const uint16_t*) ((uintptr_t) i1 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i1;
      }
      const __m128i v3_2 = _mm_loadu_si128((const __m128i*) i2);
      const uint16_t *i3 = (const uint16_t*) ((uintptr_t) i2 + input_stride);
      if XNN_UNPREDICTABLE(bh < 4) {
        i3 = i2;
      }
      const __m128i v3_3 = _mm_loadu_si128((const __m128i*) i3);
      const uint16_t *i4 = (const uint16_t*) ((uintptr_t) i3 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 4) {
        i4 = i3;
      }
      const __m128i v3_4 = _mm_loadu_si128((const __m128i*) i4);
      const uint16_t *i5 = (const uint16_t*) ((uintptr_t) i4 + input_stride);
      if XNN_UNPREDICTABLE(bh < 6) {
        i5 = i4;
      }
      const __m128i v3_5 = _mm_loadu_si128((const __m128i*) i5);
      const uint16_t *i6 = (const uint16_t*) ((uintptr_t) i5 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 6) {
        i6 = i5;
      }
      const __m128i v3_6 = _mm_loadu_si128((const __m128i*) i6);
      const __m128i v3_7 = _mm_undefined_si128();

      const __m128i v2_0 = _mm_unpacklo_epi16(v3_0, v3_1);
      const __m128i v2_1 = _mm_unpackhi_epi16(v3_0, v3_1);
      const __m128i v2_2 = _mm_unpacklo_epi16(v3_2, v3_3);
      const __m128i v2_3 = _mm_unpackhi_epi16(v3_2, v3_3);
      const __m128i v2_4 = _mm_unpacklo_epi16(v3_4, v3_5);
      const __m128i v2_5 = _mm_unpackhi_epi16(v3_4, v3_5);
      const __m128i v2_6 = _mm_unpacklo_epi16(v3_6, v3_7);
      const __m128i v2_7 = _mm_unpackhi_epi16(v3_6, v3_7);

      const __m128i v1_0 = _mm_unpacklo_epi32(v2_0, v2_2);
      const __m128i v1_1 = _mm_unpackhi_epi32(v2_0, v2_2);
      const __m128i v1_2 = _mm_unpacklo_epi32(v2_1, v2_3);
      const __m128i v1_3 = _mm_unpackhi_epi32(v2_1, v2_3);
      const __m128i v1_4 = _mm_unpacklo_epi32(v2_4, v2_6);
      const __m128i v1_5 = _mm_unpackhi_epi32(v2_4, v2_6);
      const __m128i v1_6 = _mm_unpacklo_epi32(v2_5, v2_7);
      const __m128i v1_7 = _mm_unpackhi_epi32(v2_5, v2_7);

      __m128i v0_0 = _mm_unpacklo_epi64(v1_0, v1_4);
      __m128i v0_1 = _mm_unpackhi_epi64(v1_0, v1_4);
      __m128i v0_2 = _mm_unpacklo_epi64(v1_1, v1_5);
      __m128i v0_3 = _mm_unpackhi_epi64(v1_1, v1_5);
      __m128i v0_4 = _mm_unpacklo_epi64(v1_2, v1_6);
      __m128i v0_5 = _mm_unpackhi_epi64(v1_2, v1_6);
      __m128i v0_6 = _mm_unpacklo_epi64(v1_3, v1_7);
      __m128i v0_7 = _mm_unpackhi_epi64(v1_3, v1_7);


      if (bh & 4) {
        _mm_storel_epi64((__m128i*) o7, v0_7);
        o7 += 4;
        _mm_storel_epi64((__m128i*) o6, v0_6);
        o6 += 4;
        _mm_storel_epi64((__m128i*) o5, v0_5);
        o5 += 4;
        _mm_storel_epi64((__m128i*) o4, v0_4);
        o4 += 4;
        _mm_storel_epi64((__m128i*) o3, v0_3);
        o3 += 4;
        _mm_storel_epi64((__m128i*) o2, v0_2);
        o2 += 4;
        _mm_storel_epi64((__m128i*) o1, v0_1);
        o1 += 4;
        _mm_storel_epi64((__m128i*) o0, v0_0);
        o0 += 4;
        v0_0 = _mm_unpackhi_epi64(v0_0, v0_0);
        v0_1 = _mm_unpackhi_epi64(v0_1, v0_1);
        v0_2 = _mm_unpackhi_epi64(v0_2, v0_2);
        v0_3 = _mm_unpackhi_epi64(v0_3, v0_3);
        v0_4 = _mm_unpackhi_epi64(v0_4, v0_4);
        v0_5 = _mm_unpackhi_epi64(v0_5, v0_5);
        v0_6 = _mm_unpackhi_epi64(v0_6, v0_6);
        v0_7 = _mm_unpackhi_epi64(v0_7, v0_7);
      }

      if (bh & 2) {
        unaligned_store_u32(o7, (uint32_t) _mm_cvtsi128_si32(v0_7));
        o7 += 2;
        unaligned_store_u32(o6, (uint32_t) _mm_cvtsi128_si32(v0_6));
        o6 += 2;
        unaligned_store_u32(o5, (uint32_t) _mm_cvtsi128_si32(v0_5));
        o5 += 2;
        unaligned_store_u32(o4, (uint32_t) _mm_cvtsi128_si32(v0_4));
        o4 += 2;
        unaligned_store_u32(o3, (uint32_t) _mm_cvtsi128_si32(v0_3));
        o3 += 2;
        unaligned_store_u32(o2, (uint32_t) _mm_cvtsi128_si32(v0_2));
        o2 += 2;
        unaligned_store_u32(o1, (uint32_t) _mm_cvtsi128_si32(v0_1));
        o1 += 2;
        unaligned_store_u32(o0, (uint32_t) _mm_cvtsi128_si32(v0_0));
        o0 += 2;
        v0_0 = _mm_srli_epi64(v0_0, 32);
        v0_1 = _mm_srli_epi64(v0_1, 32);
        v0_2 = _mm_srli_epi64(v0_2, 32);
        v0_3 = _mm_srli_epi64(v0_3, 32);
        v0_4 = _mm_srli_epi64(v0_4, 32);
        v0_5 = _mm_srli_epi64(v0_5, 32);
        v0_6 = _mm_srli_epi64(v0_6, 32);
        v0_7 = _mm_srli_epi64(v0_7, 32);
      }
      if (bh & 1) {
        unaligned_store_u16(o7, (uint16_t) _mm_cvtsi128_si32(v0_7));
        unaligned_store_u16(o6, (uint16_t) _mm_cvtsi128_si32(v0_6));
        unaligned_store_u16(o5, (uint16_t) _mm_cvtsi128_si32(v0_5));
        unaligned_store_u16(o4, (uint16_t) _mm_cvtsi128_si32(v0_4));
        unaligned_store_u16(o3, (uint16_t) _mm_cvtsi128_si32(v0_3));
        unaligned_store_u16(o2, (uint16_t) _mm_cvtsi128_si32(v0_2));
        unaligned_store_u16(o1, (uint16_t) _mm_cvtsi128_si32(v0_1));
        unaligned_store_u16(o0, (uint16_t) _mm_cvtsi128_si32(v0_0));
      }
    }

    i0 = (const uint16_t*) ((uintptr_t) i0 + input_reset);
    o0 = (uint16_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint16_t*) ((uintptr_t) o1 + output_reset);
    o2 = (uint16_t*) ((uintptr_t) o2 + output_reset);
    o3 = (uint16_t*) ((uintptr_t) o3 + output_reset);
    o4 = (uint16_t*) ((uintptr_t) o4 + output_reset);
    o5 = (uint16_t*) ((uintptr_t) o5 + output_reset);
    o6 = (uint16_t*) ((uintptr_t) o6 + output_reset);
    o7 = (uint16_t*) ((uintptr_t) o7 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
