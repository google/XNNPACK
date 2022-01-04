// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdio.h>

#include <emmintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/transpose.h>

void xnn_x16_transpose_ukernel__8x8_sse2(
    const uint16_t* input,
    uint16_t* output,
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
      __m128i v0 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      __m128i v1 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      __m128i v2 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      __m128i v3 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      __m128i v4 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      __m128i v5 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      __m128i v6 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      __m128i v7 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);

      __m128i vtmp0 = _mm_unpacklo_epi16(v0, v1);
      __m128i vtmp1 = _mm_unpacklo_epi16(v2, v3);
      __m128i vtmp2 = _mm_unpacklo_epi16(v4, v5);
      __m128i vtmp3 = _mm_unpacklo_epi16(v6, v7);
      __m128i vtmp4 = _mm_unpackhi_epi16(v0, v1);
      __m128i vtmp5 = _mm_unpackhi_epi16(v2, v3);
      __m128i vtmp6 = _mm_unpackhi_epi16(v4, v5);
      __m128i vtmp7 = _mm_unpackhi_epi16(v6, v7);

      v0 = _mm_unpacklo_epi32(vtmp0, vtmp1);
      v1 = _mm_unpacklo_epi32(vtmp2, vtmp3);
      v2 = _mm_unpackhi_epi32(vtmp0, vtmp1);
      v3 = _mm_unpackhi_epi32(vtmp2, vtmp3);
      v4 = _mm_unpacklo_epi32(vtmp4, vtmp5);
      v5 = _mm_unpacklo_epi32(vtmp6, vtmp7);
      v6 = _mm_unpackhi_epi32(vtmp4, vtmp5);
      v7 = _mm_unpackhi_epi32(vtmp6, vtmp7);

      vtmp0 = _mm_unpacklo_epi64(v0, v1);
      vtmp1 = _mm_unpackhi_epi64(v0, v1);
      vtmp2 = _mm_unpacklo_epi64(v2, v3);
      vtmp3 = _mm_unpackhi_epi64(v2, v3);
      vtmp4 = _mm_unpacklo_epi64(v4, v5);
      vtmp5 = _mm_unpackhi_epi64(v4, v5);
      vtmp6 = _mm_unpacklo_epi64(v6, v7);
      vtmp7 = _mm_unpackhi_epi64(v6, v7);

      _mm_storeu_si128((__m128i*) o7, vtmp7);
      o7 = (uint16_t*) ((uintptr_t) o7 + tile_hbytes);
      _mm_storeu_si128((__m128i*) o6, vtmp6);
      o6 = (uint16_t*) ((uintptr_t) o6 + tile_hbytes);
      _mm_storeu_si128((__m128i*) o5, vtmp5);
      o5 = (uint16_t*) ((uintptr_t) o5 + tile_hbytes);
      _mm_storeu_si128((__m128i*) o4, vtmp4);
      o4 = (uint16_t*) ((uintptr_t) o4 + tile_hbytes);
      _mm_storeu_si128((__m128i*) o3, vtmp3);
      o3 = (uint16_t*) ((uintptr_t) o3 + tile_hbytes);
      _mm_storeu_si128((__m128i*) o2, vtmp2);
      o2 = (uint16_t*) ((uintptr_t) o2 + tile_hbytes);
      _mm_storeu_si128((__m128i*) o1, vtmp1);
      o1 = (uint16_t*) ((uintptr_t) o1 + tile_hbytes);
      _mm_storeu_si128((__m128i*) o0, vtmp0);
      o0 = (uint16_t*) ((uintptr_t) o0 + tile_hbytes);
    }

    if (bh != 0) {
      __m128i v0 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh >= 2) {
        i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      }
      __m128i v1 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 2) {
        i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      }
      __m128i v2 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh >= 4) {
        i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      }
      __m128i v3 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 4) {
        i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      }
      __m128i v4 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh >= 6) {
        i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      }
      __m128i v5 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 6) {
        i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      }
      __m128i v6 = _mm_loadu_si128((const __m128i*) i0);

      __m128i vtmp0 = _mm_unpacklo_epi16(v0, v1);
      __m128i vtmp1 = _mm_unpacklo_epi16(v2, v3);
      __m128i vtmp2 = _mm_unpacklo_epi16(v4, v5);
      __m128i vtmp3 = _mm_unpacklo_epi16(v6, v6);
      __m128i vtmp4 = _mm_unpackhi_epi16(v0, v1);
      __m128i vtmp5 = _mm_unpackhi_epi16(v2, v3);
      __m128i vtmp6 = _mm_unpackhi_epi16(v4, v5);
      __m128i vtmp7 = _mm_unpackhi_epi16(v6, v6);

      v0 = _mm_unpacklo_epi32(vtmp0, vtmp1);
      v1 = _mm_unpacklo_epi32(vtmp2, vtmp3);
      v2 = _mm_unpackhi_epi32(vtmp0, vtmp1);
      v3 = _mm_unpackhi_epi32(vtmp2, vtmp3);
      v4 = _mm_unpacklo_epi32(vtmp4, vtmp5);
      v5 = _mm_unpacklo_epi32(vtmp6, vtmp7);
      v6 = _mm_unpackhi_epi32(vtmp4, vtmp5);
      __m128i v7 = _mm_unpackhi_epi32(vtmp6, vtmp7);

      vtmp0 = _mm_unpacklo_epi64(v0, v1);
      vtmp1 = _mm_unpackhi_epi64(v0, v1);
      vtmp2 = _mm_unpacklo_epi64(v2, v3);
      vtmp3 = _mm_unpackhi_epi64(v2, v3);
      vtmp4 = _mm_unpacklo_epi64(v4, v5);
      vtmp5 = _mm_unpackhi_epi64(v4, v5);
      vtmp6 = _mm_unpacklo_epi64(v6, v7);
      vtmp7 = _mm_unpackhi_epi64(v6, v7);

      if (bh & 4) {
        _mm_storel_epi64((__m128i*) o7, vtmp7);
        o7 += 4;
        _mm_storel_epi64((__m128i*) o6, vtmp6);
        o6 += 4;
        _mm_storel_epi64((__m128i*) o5, vtmp5);
        o5 += 4;
        _mm_storel_epi64((__m128i*) o4, vtmp4);
        o4 += 4;
        _mm_storel_epi64((__m128i*) o3, vtmp3);
        o3 += 4;
        _mm_storel_epi64((__m128i*) o2, vtmp2);
        o2 += 4;
        _mm_storel_epi64((__m128i*) o1, vtmp1);
        o1 += 4;
        _mm_storel_epi64((__m128i*) o0, vtmp0);
        o0 += 4;
        vtmp0 = _mm_srli_si128(vtmp0, 8);
        vtmp1 = _mm_srli_si128(vtmp1, 8);
        vtmp2 = _mm_srli_si128(vtmp2, 8);
        vtmp3 = _mm_srli_si128(vtmp3, 8);
        vtmp4 = _mm_srli_si128(vtmp4, 8);
        vtmp5 = _mm_srli_si128(vtmp5, 8);
        vtmp6 = _mm_srli_si128(vtmp6, 8);
        vtmp7 = _mm_srli_si128(vtmp7, 8);
      }

      if (bh & 2) {
        *((int*) o7) = _mm_cvtsi128_si32(vtmp7);
        o7 += 2;
        *((int*) o6) = _mm_cvtsi128_si32(vtmp6);
        o6 += 2;
        *((int*) o5) = _mm_cvtsi128_si32(vtmp5);
        o5 += 2;
        *((int*) o4) = _mm_cvtsi128_si32(vtmp4);
        o4 += 2;
        *((int*) o3) = _mm_cvtsi128_si32(vtmp3);
        o3 += 2;
        *((int*) o2) = _mm_cvtsi128_si32(vtmp2);
        o2 += 2;
        *((int*) o1) = _mm_cvtsi128_si32(vtmp1);
        o1 += 2;
        *((int*) o0) = _mm_cvtsi128_si32(vtmp0);
        o0 += 2;
        vtmp0 =  _mm_srli_epi64(vtmp0, 32);
        vtmp1 =  _mm_srli_epi64(vtmp1, 32);
        vtmp2 =  _mm_srli_epi64(vtmp2, 32);
        vtmp3 =  _mm_srli_epi64(vtmp3, 32);
        vtmp4 =  _mm_srli_epi64(vtmp4, 32);
        vtmp5 =  _mm_srli_epi64(vtmp5, 32);
        vtmp6 =  _mm_srli_epi64(vtmp6, 32);
        vtmp7 =  _mm_srli_epi64(vtmp7, 32);
      }
      if (bh & 1) {
        *o7 = (uint16_t) _mm_cvtsi128_si32(vtmp7);
        *o6 = (uint16_t) _mm_cvtsi128_si32(vtmp6);
        *o5 = (uint16_t) _mm_cvtsi128_si32(vtmp5);
        *o4 = (uint16_t) _mm_cvtsi128_si32(vtmp4);
        *o3 = (uint16_t) _mm_cvtsi128_si32(vtmp3);
        *o2 = (uint16_t) _mm_cvtsi128_si32(vtmp2);
        *o1 = (uint16_t) _mm_cvtsi128_si32(vtmp1);
        *o0 = (uint16_t) _mm_cvtsi128_si32(vtmp0);
      }
    }
    i0 = (uint16_t*) ((uintptr_t) i0 + input_reset);
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
