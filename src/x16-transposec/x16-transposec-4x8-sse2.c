// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/transpose.h"
#include "xnnpack/unaligned.h"


void xnn_x16_transposec_ukernel__4x8_sse2(
    const uint16_t* input,
    uint16_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(output_stride >= block_height * sizeof(uint16_t));
  assert(input_stride >= block_width * sizeof(uint16_t));

  const size_t tile_height = 4;
  const size_t tile_width = 8;
  const size_t tile_hbytes = tile_height * sizeof(uint16_t);
  const size_t tile_wbytes = tile_width * sizeof(uint16_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint16_t);
  const size_t input_offset = tile_height * input_stride;

  const uint16_t* i0 = (const uint16_t*) input;
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
  const uint16_t* i2 = (const uint16_t*) ((uintptr_t) i1 + input_stride);
  const uint16_t* i3 = (const uint16_t*) ((uintptr_t) i2 + input_stride);

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
    for (; bh >= 4; bh -= 4) {
      __m128i v0 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      __m128i v1 = _mm_loadu_si128((const __m128i*) i1);
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      __m128i v2 = _mm_loadu_si128((const __m128i*) i2);
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      __m128i v3 = _mm_loadu_si128((const __m128i*) i3);
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);

      __m128i vtmp0 = _mm_unpacklo_epi16(v0, v2);
      __m128i vtmp1 = _mm_unpacklo_epi16(v1, v3);
      __m128i vtmp2 = _mm_unpackhi_epi16(v0, v2);
      __m128i vtmp3 = _mm_unpackhi_epi16(v1, v3);

      v0 = _mm_unpacklo_epi16(vtmp0, vtmp1);
      v1 = _mm_unpacklo_epi16(vtmp2, vtmp3);
      v2 = _mm_unpackhi_epi16(vtmp0, vtmp1);
      v3 = _mm_unpackhi_epi16(vtmp2, vtmp3);

      _mm_storeh_pi((__m64*) o7, _mm_castsi128_ps(v3));
      o7 = (uint16_t*) ((uintptr_t) o7 + tile_hbytes);
      _mm_storel_epi64((__m128i*) o6, v3);
      o6 = (uint16_t*) ((uintptr_t) o6 + tile_hbytes);
      _mm_storeh_pi((__m64*) o5, _mm_castsi128_ps(v1));
      o5 = (uint16_t*) ((uintptr_t) o5 + tile_hbytes);
      _mm_storel_epi64((__m128i*) o4, v1);
      o4 = (uint16_t*) ((uintptr_t) o4 + tile_hbytes);
      _mm_storeh_pi((__m64*) o3, _mm_castsi128_ps(v2));
      o3 = (uint16_t*) ((uintptr_t) o3 + tile_hbytes);
      _mm_storel_epi64((__m128i*) o2, v2);
      o2 = (uint16_t*) ((uintptr_t) o2 + tile_hbytes);
      _mm_storeh_pi((__m64*) o1, _mm_castsi128_ps(v0));
      o1 = (uint16_t*) ((uintptr_t) o1 + tile_hbytes);
      _mm_storel_epi64((__m128i*) o0, v0);
      o0 = (uint16_t*) ((uintptr_t) o0 + tile_hbytes);
    }

    if (bh != 0) {
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i0;
      }
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      __m128i v0 = _mm_loadu_si128((const __m128i*) i0);
      __m128i v1 = _mm_loadu_si128((const __m128i*) i1);
      __m128i v2 = _mm_loadu_si128((const __m128i*) i2);

      __m128i vtmp0 = _mm_unpacklo_epi16(v0, v2);
      __m128i vtmp1 = _mm_unpacklo_epi16(v1, v1);
      __m128i vtmp2 = _mm_unpackhi_epi16(v0, v2);
      __m128i vtmp3 = _mm_unpackhi_epi16(v1, v1);

      v0 = _mm_unpacklo_epi16(vtmp0, vtmp1);
      v1 = _mm_unpacklo_epi16(vtmp2, vtmp3);
      v2 = _mm_unpackhi_epi16(vtmp0, vtmp1);
      __m128i v3 = _mm_unpackhi_epi16(vtmp2, vtmp3);

      if (bh & 2) {
        unaligned_store_u32(o7, (uint32_t) _mm_cvtsi128_si32(_mm_shuffle_epi32(v3, 0xE)));
        o7 += 2;
        unaligned_store_u32(o6, (uint32_t) _mm_cvtsi128_si32(v3));
        o6 += 2;
        unaligned_store_u32(o5, (uint32_t) _mm_cvtsi128_si32(_mm_shuffle_epi32(v1, 0xE)));
        o5 += 2;
        unaligned_store_u32(o4, (uint32_t) _mm_cvtsi128_si32(v1));
        o4 += 2;
        unaligned_store_u32(o3, (uint32_t) _mm_cvtsi128_si32(_mm_shuffle_epi32(v2, 0xE)));
        o3 += 2;
        unaligned_store_u32(o2, (uint32_t) _mm_cvtsi128_si32(v2));
        o2 += 2;
        unaligned_store_u32(o1, (uint32_t) _mm_cvtsi128_si32(_mm_shuffle_epi32(v0, 0xE)));
        o1 += 2;
        unaligned_store_u32(o0, (uint32_t) _mm_cvtsi128_si32(v0));
        o0 += 2;
        v0 = _mm_srli_epi64(v0, 32);
        v1 = _mm_srli_epi64(v1, 32);
        v2 = _mm_srli_epi64(v2, 32);
        v3 = _mm_srli_epi64(v3, 32);
      }
      if (bh & 1) {
        *o7 = (uint16_t) _mm_extract_epi16(v3, 4);
        *o6 = (uint16_t) _mm_cvtsi128_si32(v3);
        *o5 = (uint16_t) _mm_extract_epi16(v1, 4);
        *o4 = (uint16_t) _mm_cvtsi128_si32(v1);
        *o3 = (uint16_t) _mm_extract_epi16(v2, 4);
        *o2 = (uint16_t) _mm_cvtsi128_si32(v2);
        *o1 = (uint16_t) _mm_extract_epi16(v0, 4);
        *o0 = (uint16_t) _mm_cvtsi128_si32(v0);
      }
    }
    i0 = (const uint16_t*) ((uintptr_t) i0 + input_reset);
    i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
    i2 = (const uint16_t*) ((uintptr_t) i1 + input_stride);
    i3 = (const uint16_t*) ((uintptr_t) i2 + input_stride);
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
