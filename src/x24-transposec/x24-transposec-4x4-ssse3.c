// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <tmmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/transpose.h"
#include "xnnpack/unaligned.h"

void xnn_x24_transposec_ukernel__4x4_ssse3(
    const void *input,
    void * output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{ 
  static const uint8_t pos0[16] = {0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11, -1, -1, -1, -1};
  static const uint8_t pos1[16] = {4, 8, 12, 6, 10, 14, 5, 9, 13, 7, 11, 15, -1, -1, -1, -1};
  static const uint8_t pos2[16] = {12, -1, -1, 14, -1, -1, 13, -1, -1, 15, -1, -1, -1, -1, -1, -1};
  static const uint8_t pos3[16] = {-1, 0, 4, -1, 2, 6, -1, 1, 5, -1, 3, 7, -1, -1, -1, -1};
  static const uint8_t pos4[16] = {8, 12, -1, 10, 14, -1, 9, 13, -1, 11, 15, -1, -1, -1, -1, -1};
  static const uint8_t pos5[16] = {-1, -1, 0, -1, -1, 2, -1, -1, 1, -1, -1, 3, -1, -1, -1, -1};

  assert(output_stride >= block_height * 3);
  assert(input_stride >= block_width * 3);

  assert(output_stride >= block_height * 3);
  assert(input_stride >= block_width * 3);

  const size_t tile_height = 4;
  const size_t tile_width = 4;
  const size_t tile_wbytes = tile_width * 3;
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_width * output_stride - block_height * 3;
  const size_t tile_stride = tile_height * input_stride;

  const uint8_t* i0 = (const uint8_t*) input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);

  uint8_t* o0 = (uint8_t*) output;
  uint8_t* o1 = (uint8_t*) ((uintptr_t) o0 + output_stride);
  uint8_t* o2 = (uint8_t*) ((uintptr_t) o1 + output_stride);
  uint8_t* o3 = (uint8_t*) ((uintptr_t) o2 + output_stride);

  const __m128i vperm0 = _mm_load_si128((const __m128i*) pos0);
  const __m128i vperm1 = _mm_load_si128((const __m128i*) pos1);
  const __m128i vperm2 = _mm_load_si128((const __m128i*) pos2);
  const __m128i vperm3 = _mm_load_si128((const __m128i*) pos3);
  const __m128i vperm4 = _mm_load_si128((const __m128i*) pos4);
  const __m128i vperm5 = _mm_load_si128((const __m128i*) pos5);
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
    size_t bh = block_height;
    for (; bh >= 4; bh -= 4) {
      const __m128i v0 = _mm_loadu_si128((const __m128i*) i0);
      const __m128i v1 = _mm_loadu_si128((const __m128i*) i1);
      const __m128i v2 = _mm_loadu_si128((const __m128i*) i2);
      const __m128i v3 = _mm_loadu_si128((const __m128i*) i3);
      i0 = (const uint8_t*) ((uintptr_t) i0 + tile_stride);
      i1 = (const uint8_t*) ((uintptr_t) i1 + tile_stride);
      i2 = (const uint8_t*) ((uintptr_t) i2 + tile_stride);
      i3 = (const uint8_t*) ((uintptr_t) i3 + tile_stride);

      const __m128i v1_0 = _mm_unpacklo_epi8(v0, v1);
      const __m128i v1_1 = _mm_unpackhi_epi8(v0, v1);
      const __m128i v1_2 = _mm_unpacklo_epi8(v2, v3);
      const __m128i v1_3 = _mm_unpackhi_epi8(v2, v3);

      const __m128i v3_0 = _mm_unpacklo_epi8(v1_0, v1_2);
      const __m128i v3_1 = _mm_unpackhi_epi8(v1_0, v1_2);
      const __m128i v3_2 = _mm_unpacklo_epi8(v1_1, v1_3);

      __m128i v4_0 = _mm_shuffle_epi8(v3_0, vperm0);
      __m128i v4_1 = _mm_or_si128(_mm_shuffle_epi8(v3_0, vperm2), _mm_shuffle_epi8(v3_1, vperm3));
      __m128i v4_2 = _mm_or_si128(_mm_shuffle_epi8(v3_1, vperm4), _mm_shuffle_epi8(v3_2, vperm5));
      __m128i v4_3 = _mm_shuffle_epi8(v3_2, vperm1);

      _mm_storel_epi64((__m128i*) o3, v4_3);
      _mm_storel_epi64((__m128i*) o2, v4_2);
      _mm_storel_epi64((__m128i*) o1, v4_1);
      _mm_storel_epi64((__m128i*) o0, v4_0);
      o3 += 8;
      o2 += 8;
      o1 += 8;
      o0 += 8;

      v4_3 = _mm_unpackhi_epi64(v4_3, v4_3);
      unaligned_store_u32(o3, (uint32_t) _mm_cvtsi128_si32(v4_3));
      v4_2 = _mm_unpackhi_epi64(v4_2, v4_2);
      unaligned_store_u32(o2, (uint32_t) _mm_cvtsi128_si32(v4_2));
      v4_1 = _mm_unpackhi_epi64(v4_1, v4_1);
      unaligned_store_u32(o1, (uint32_t) _mm_cvtsi128_si32(v4_1));
      v4_0 = _mm_unpackhi_epi64(v4_0, v4_0);
      unaligned_store_u32(o0, (uint32_t) _mm_cvtsi128_si32(v4_0));
      o3 += 4;
      o2 += 4;
      o1 += 4;
      o0 += 4;
    }

    if (bh != 0) {
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i0;
      }
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const __m128i v0 = _mm_loadu_si128((const __m128i*) i0);
      const __m128i v1 = _mm_loadu_si128((const __m128i*) i1);
      const __m128i v2 = _mm_loadu_si128((const __m128i*) i2);

      const __m128i v1_0 = _mm_unpacklo_epi8(v0, v1);
      const __m128i v1_1 = _mm_unpackhi_epi8(v0, v1);
      const __m128i v1_2 = _mm_unpacklo_epi8(v2, v2);
      const __m128i v1_3 = _mm_unpackhi_epi8(v2, v2);

      const __m128i v3_0 = _mm_unpacklo_epi8(v1_0, v1_2);
      const __m128i v3_1 = _mm_unpackhi_epi8(v1_0, v1_2);
      const __m128i v3_2 = _mm_unpacklo_epi8(v1_1, v1_3);

      __m128i v4_0 = _mm_shuffle_epi8(v3_0, vperm0);
      __m128i v4_1 = _mm_or_si128(_mm_shuffle_epi8(v3_0, vperm2), _mm_shuffle_epi8(v3_1, vperm3));
      __m128i v4_2 = _mm_or_si128(_mm_shuffle_epi8(v3_1, vperm4), _mm_shuffle_epi8(v3_2, vperm5));
      __m128i v4_3 = _mm_shuffle_epi8(v3_2, vperm1);

      if (bh & 2) {
        unaligned_store_u32(o3, (uint32_t) _mm_cvtsi128_si32(v4_3));
        unaligned_store_u32(o2, (uint32_t) _mm_cvtsi128_si32(v4_2));
        unaligned_store_u32(o1, (uint32_t) _mm_cvtsi128_si32(v4_1));
        unaligned_store_u32(o0, (uint32_t) _mm_cvtsi128_si32(v4_0));
        o3 += 4;
        o2 += 4;
        o1 += 4;
        o0 += 4;
        unaligned_store_u16(o3, (uint16_t) _mm_extract_epi16(v4_3, 2));
        unaligned_store_u16(o2, (uint16_t) _mm_extract_epi16(v4_2, 2));
        unaligned_store_u16(o1, (uint16_t) _mm_extract_epi16(v4_1, 2));
        unaligned_store_u16(o0, (uint16_t) _mm_extract_epi16(v4_0, 2));
        o3 += 2;
        o2 += 2;
        o1 += 2;
        o0 += 2;
        v4_3 = _mm_bsrli_si128(v4_3, 6);
        v4_2 = _mm_bsrli_si128(v4_2, 6);
        v4_1 = _mm_bsrli_si128(v4_1, 6);
        v4_0 = _mm_bsrli_si128(v4_0, 6);
      }
      if (bh & 1) {
        unaligned_store_u16(o3, (uint16_t) _mm_cvtsi128_si32(v4_3));
        unaligned_store_u16(o2, (uint16_t) _mm_cvtsi128_si32(v4_2));
        unaligned_store_u16(o1, (uint16_t) _mm_cvtsi128_si32(v4_1));
        unaligned_store_u16(o0, (uint16_t) _mm_cvtsi128_si32(v4_0));
        o3 += 2;
        o2 += 2;
        o1 += 2;
        o0 += 2;
        *((uint8_t*) o3) = (uint8_t) _mm_cvtsi128_si32(_mm_bsrli_si128(v4_3, 2));
        *((uint8_t*) o2) = (uint8_t) _mm_cvtsi128_si32(_mm_bsrli_si128(v4_2, 2));
        *((uint8_t*) o1) = (uint8_t) _mm_cvtsi128_si32(_mm_bsrli_si128(v4_1, 2));
        *((uint8_t*) o0) = (uint8_t) _mm_cvtsi128_si32(_mm_bsrli_si128(v4_0, 2));
        o3 += 1;
        o2 += 1;
        o1 += 1;
        o0 += 1;
      }
    }
    i0 = (const uint8_t*) ((uintptr_t) i0 + input_reset);
    i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
    i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
    i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
    o0 = (uint8_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint8_t*) ((uintptr_t) o1 + output_reset);
    o2 = (uint8_t*) ((uintptr_t) o2 + output_reset);
    o3 = (uint8_t*) ((uintptr_t) o3 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
