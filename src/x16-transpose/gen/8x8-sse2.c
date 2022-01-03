// Auto-generated file. Do not edit!
//   Template: src/x32-transpose/sse2.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <emmintrin.h>

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/transpose.h>


void xnn_x16_transpose_ukernel__8x8_sse2(
    const uint16_t *input,
    uint16_t * output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height)
{
  assert(output_stride >= block_height * sizeof(uint16_t));
  assert(input_stride >= block_width * sizeof(uint16_t));

  const size_t tile_height = 8;
  const size_t tile_width = 8;
  const size_t tile_hbytes = tile_height * sizeof(uint16_t);
  const size_t tile_wbytes = tile_width * sizeof(uint16_t);
  const size_t input_reset = tile_wbytes - (block_height - ((block_height % tile_height) != 0)) * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint16_t);

  const uint16_t* i0 = input;

  uint16_t* o0 = (uint16_t*) output;

  do {
    size_t bh = block_height;
    for (; bh >= 8; bh -= 8) {
      const __m128i v0 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v1 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v2 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v3 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v4 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v5 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v6 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v7 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);

      __m128i v16_0 = _mm_unpacklo_epi16(v0, v1);
      __m128i v16_1 = _mm_unpackhi_epi16(v0, v1);
      __m128i v16_2 = _mm_unpacklo_epi16(v2, v3);
      __m128i v16_3 = _mm_unpackhi_epi16(v2, v3);
      __m128i v16_4 = _mm_unpacklo_epi16(v4, v5);
      __m128i v16_5 = _mm_unpackhi_epi16(v4, v5);
      __m128i v16_6 = _mm_unpacklo_epi16(v6, v7);
      __m128i v16_7 = _mm_unpackhi_epi16(v6, v7);

      const __m128i v32_0 = _mm_unpacklo_epi32(v16_0, v16_2);
      const __m128i v32_1 = _mm_unpackhi_epi32(v16_0, v16_2);
      const __m128i v32_2 = _mm_unpacklo_epi32(v16_1, v16_3);
      const __m128i v32_3 = _mm_unpackhi_epi32(v16_1, v16_3);
      const __m128i v32_4 = _mm_unpacklo_epi32(v16_4, v16_6);
      const __m128i v32_5 = _mm_unpackhi_epi32(v16_4, v16_6);
      const __m128i v32_6 = _mm_unpacklo_epi32(v16_5, v16_7);
      const __m128i v32_7 = _mm_unpackhi_epi32(v16_5, v16_7);
      const __m128i v64_0 = _mm_unpacklo_epi64(v32_0, v32_4);
      const __m128i v64_1 = _mm_unpackhi_epi64(v32_0, v32_4);
      const __m128i v64_2 = _mm_unpacklo_epi64(v32_1, v32_5);
      const __m128i v64_3 = _mm_unpackhi_epi64(v32_1, v32_5);
      const __m128i v64_4 = _mm_unpacklo_epi64(v32_2, v32_6);
      const __m128i v64_5 = _mm_unpackhi_epi64(v32_2, v32_6);
      const __m128i v64_6 = _mm_unpacklo_epi64(v32_3, v32_7);
      const __m128i v64_7 = _mm_unpackhi_epi64(v32_3, v32_7);

      size_t rem = min(block_width - 1, 7);
      uint16_t* o1 = (uint16_t*) ((uintptr_t) o0 + rem * output_stride);
      switch(rem){
      case (7):
        _mm_storeu_si128((__m128i*) o1, v64_7);
        o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
      case (6):
        _mm_storeu_si128((__m128i*) o1, v64_6);
        o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
      case (5):
        _mm_storeu_si128((__m128i*) o1, v64_5);
        o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
      case (4):
        _mm_storeu_si128((__m128i*) o1, v64_4);
        o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
      case (3):
        _mm_storeu_si128((__m128i*) o1, v64_3);
        o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
      case (2):
        _mm_storeu_si128((__m128i*) o1, v64_2);
        o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
      case (1):
        _mm_storeu_si128((__m128i*) o1, v64_1);
      }
      _mm_storeu_si128((__m128i*) o0, v64_0);
      o0 = (uint16_t*) ((uintptr_t) o0 + tile_hbytes);
    }

    if (bh != 0) {
      const __m128i v0 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 1) {
        i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      }
      const __m128i v1 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 2) {
        i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      }
      const __m128i v2 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 3) {
        i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      }
      const __m128i v3 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 4) {
        i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      }
      const __m128i v4 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 5) {
        i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      }
      const __m128i v5 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 6) {
        i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      }
      const __m128i v6 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 7) {
        i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      }
      __m128i v7 = _mm_setzero_si128();

      const __m128i v16_0 = _mm_unpacklo_epi16(v0, v1);
      const __m128i v16_1 = _mm_unpackhi_epi16(v0, v1);
      const __m128i v16_2 = _mm_unpacklo_epi16(v2, v3);
      const __m128i v16_3 = _mm_unpackhi_epi16(v2, v3);
      const __m128i v16_4 = _mm_unpacklo_epi16(v4, v5);
      const __m128i v16_5 = _mm_unpackhi_epi16(v4, v5);
      const __m128i v16_6 = _mm_unpacklo_epi16(v6, v7);
      const __m128i v16_7 = _mm_unpackhi_epi16(v6, v7);

      __m128i v32_0 = _mm_unpacklo_epi32(v16_0, v16_2);
      __m128i v32_1 = _mm_unpackhi_epi32(v16_0, v16_2);
      __m128i v32_2 = _mm_unpacklo_epi32(v16_1, v16_3);
      __m128i v32_3 = _mm_unpackhi_epi32(v16_1, v16_3);
      __m128i v32_4 = _mm_unpacklo_epi32(v16_4, v16_6);
      __m128i v32_5 = _mm_unpackhi_epi32(v16_4, v16_6);
      __m128i v32_6 = _mm_unpacklo_epi32(v16_5, v16_7);
      __m128i v32_7 = _mm_unpackhi_epi32(v16_5, v16_7);
      __m128i v64_0 = _mm_unpacklo_epi64(v32_0, v32_4);
      __m128i v64_1 = _mm_unpackhi_epi64(v32_0, v32_4);
      __m128i v64_2 = _mm_unpacklo_epi64(v32_1, v32_5);
      __m128i v64_3 = _mm_unpackhi_epi64(v32_1, v32_5);
      __m128i v64_4 = _mm_unpacklo_epi64(v32_2, v32_6);
      __m128i v64_5 = _mm_unpackhi_epi64(v32_2, v32_6);
      __m128i v64_6 = _mm_unpacklo_epi64(v32_3, v32_7);
      __m128i v64_7 = _mm_unpackhi_epi64(v32_3, v32_7);

      size_t rem = min(block_width - 1, 7);
      if (bh & 4) {
        uint16_t* o1 = (uint16_t*) ((uintptr_t) o0 + rem * output_stride);
        switch(rem){
        case (7):
          _mm_storel_epi64((__m128i*) o1, v64_7);
          o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
        case (6):
          _mm_storel_epi64((__m128i*) o1, v64_6);
          o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
        case (5):
          _mm_storel_epi64((__m128i*) o1, v64_5);
          o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
        case (4):
          _mm_storel_epi64((__m128i*) o1, v64_4);
          o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
        case (3):
          _mm_storel_epi64((__m128i*) o1, v64_3);
          o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
        case (2):
          _mm_storel_epi64((__m128i*) o1, v64_2);
          o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
        case (1):
          _mm_storel_epi64((__m128i*) o1, v64_1);
        }
        _mm_storel_epi64((__m128i*) o0, v64_0);
        o0 += 4;
        v64_0 = _mm_unpackhi_epi64(v64_0, v64_0);
        v64_1 = _mm_unpackhi_epi64(v64_1, v64_1);
        v64_2 = _mm_unpackhi_epi64(v64_2, v64_2);
        v64_3 = _mm_unpackhi_epi64(v64_3, v64_3);
        v64_4 = _mm_unpackhi_epi64(v64_4, v64_4);
        v64_5 = _mm_unpackhi_epi64(v64_5, v64_5);
        v64_6 = _mm_unpackhi_epi64(v64_6, v64_6);
        v64_7 = _mm_unpackhi_epi64(v64_7, v64_7);
      }

      if (bh & 2) {
        uint16_t* o1 = (uint16_t*) ((uintptr_t) o0 + rem * output_stride);
        switch(rem){
        case (7):
          *((int*) o1) = _mm_cvtsi128_si32(v64_7);
          o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
        case (6):
          *((int*) o1) = _mm_cvtsi128_si32(v64_6);
          o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
        case (5):
          *((int*) o1) = _mm_cvtsi128_si32(v64_5);
          o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
        case (4):
          *((int*) o1) = _mm_cvtsi128_si32(v64_4);
          o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
        case (3):
          *((int*) o1) = _mm_cvtsi128_si32(v64_3);
          o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
        case (2):
          *((int*) o1) = _mm_cvtsi128_si32(v64_2);
          o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
        case (1):
          *((int*) o1) = _mm_cvtsi128_si32(v64_1);
        }
        *((int*) o0) = _mm_cvtsi128_si32(v64_0);
        o0 += 2;
        v64_0 = _mm_srli_epi64(v64_0, 32);
        v64_1 = _mm_srli_epi64(v64_1, 32);
        v64_2 = _mm_srli_epi64(v64_2, 32);
        v64_3 = _mm_srli_epi64(v64_3, 32);
        v64_4 = _mm_srli_epi64(v64_4, 32);
        v64_5 = _mm_srli_epi64(v64_5, 32);
        v64_6 = _mm_srli_epi64(v64_6, 32);
        v64_7 = _mm_srli_epi64(v64_7, 32);
      }
      if (bh & 1) {
        uint16_t* o1 = (uint16_t*) ((uintptr_t) o0 + rem * output_stride);
        switch(rem){
        case (7):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_7);
          o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
        case (6):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_6);
          o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
        case (5):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_5);
          o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
        case (4):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_4);
          o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
        case (3):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_3);
          o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
        case (2):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_2);
          o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
        case (1):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_1);
        }
        *((uint16_t*) o0) = (uint16_t) _mm_cvtsi128_si32(v64_0);
      }
    }

    i0 = (const uint16_t*) ((uintptr_t) i0 + input_reset);
    o0 = (uint16_t*) ((uintptr_t) o0 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
