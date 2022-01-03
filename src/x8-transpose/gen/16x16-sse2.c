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


void xnn_x8_transpose_ukernel__16x16_sse2(
    const uint8_t *input,
    uint8_t * output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height)
{
  assert(output_stride >= block_height * sizeof(uint8_t));
  assert(input_stride >= block_width * sizeof(uint8_t));

  const size_t tile_height = 16;
  const size_t tile_width = 16;
  const size_t tile_hbytes = tile_height * sizeof(uint8_t);
  const size_t tile_wbytes = tile_width * sizeof(uint8_t);
  const size_t input_reset = tile_wbytes - (block_height - ((block_height % tile_height) != 0)) * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint8_t);

  const uint8_t* i0 = input;

  uint8_t* o0 = (uint8_t*) output;

  do {
    size_t bh = block_height;
    for (; bh >= 16; bh -= 16) {
      const __m128i v0 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v1 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v2 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v3 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v4 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v5 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v6 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v7 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v8 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v9 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v10 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v11 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v12 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v13 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v14 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m128i v15 = _mm_loadu_si128((const __m128i*) i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);

      __m128i v8_0 = _mm_unpacklo_epi8(v0, v1);
      __m128i v8_1 = _mm_unpackhi_epi8(v0, v1);
      __m128i v8_2 = _mm_unpacklo_epi8(v2, v3);
      __m128i v8_3 = _mm_unpackhi_epi8(v2, v3);
      __m128i v8_4 = _mm_unpacklo_epi8(v4, v5);
      __m128i v8_5 = _mm_unpackhi_epi8(v4, v5);
      __m128i v8_6 = _mm_unpacklo_epi8(v6, v7);
      __m128i v8_7 = _mm_unpackhi_epi8(v6, v7);
      __m128i v8_8 = _mm_unpacklo_epi8(v8, v9);
      __m128i v8_9 = _mm_unpackhi_epi8(v8, v9);
      __m128i v8_10 = _mm_unpacklo_epi8(v10, v11);
      __m128i v8_11 = _mm_unpackhi_epi8(v10, v11);
      __m128i v8_12 = _mm_unpacklo_epi8(v12, v13);
      __m128i v8_13 = _mm_unpackhi_epi8(v12, v13);
      __m128i v8_14 = _mm_unpacklo_epi8(v14, v15);
      __m128i v8_15 = _mm_unpackhi_epi8(v14, v15);

      const __m128i v16_0 = _mm_unpacklo_epi16(v8_0, v8_2);
      const __m128i v16_1 = _mm_unpackhi_epi16(v8_0, v8_2);
      const __m128i v16_2 = _mm_unpacklo_epi16(v8_1, v8_3);
      const __m128i v16_3 = _mm_unpackhi_epi16(v8_1, v8_3);
      const __m128i v16_4 = _mm_unpacklo_epi16(v8_4, v8_6);
      const __m128i v16_5 = _mm_unpackhi_epi16(v8_4, v8_6);
      const __m128i v16_6 = _mm_unpacklo_epi16(v8_5, v8_7);
      const __m128i v16_7 = _mm_unpackhi_epi16(v8_5, v8_7);
      const __m128i v16_8 = _mm_unpacklo_epi16(v8_8, v8_10);
      const __m128i v16_9 = _mm_unpackhi_epi16(v8_8, v8_10);
      const __m128i v16_10 = _mm_unpacklo_epi16(v8_9, v8_11);
      const __m128i v16_11 = _mm_unpackhi_epi16(v8_9, v8_11);
      const __m128i v16_12 = _mm_unpacklo_epi16(v8_12, v8_14);
      const __m128i v16_13 = _mm_unpackhi_epi16(v8_12, v8_14);
      const __m128i v16_14 = _mm_unpacklo_epi16(v8_13, v8_15);
      const __m128i v16_15 = _mm_unpackhi_epi16(v8_13, v8_15);
      const __m128i v32_0 = _mm_unpacklo_epi32(v16_0, v16_4);
      const __m128i v32_1 = _mm_unpackhi_epi32(v16_0, v16_4);
      const __m128i v32_2 = _mm_unpacklo_epi32(v16_1, v16_5);
      const __m128i v32_3 = _mm_unpackhi_epi32(v16_1, v16_5);
      const __m128i v32_4 = _mm_unpacklo_epi32(v16_2, v16_6);
      const __m128i v32_5 = _mm_unpackhi_epi32(v16_2, v16_6);
      const __m128i v32_6 = _mm_unpacklo_epi32(v16_3, v16_7);
      const __m128i v32_7 = _mm_unpackhi_epi32(v16_3, v16_7);
      const __m128i v32_8 = _mm_unpacklo_epi32(v16_8, v16_12);
      const __m128i v32_9 = _mm_unpackhi_epi32(v16_8, v16_12);
      const __m128i v32_10 = _mm_unpacklo_epi32(v16_9, v16_13);
      const __m128i v32_11 = _mm_unpackhi_epi32(v16_9, v16_13);
      const __m128i v32_12 = _mm_unpacklo_epi32(v16_10, v16_14);
      const __m128i v32_13 = _mm_unpackhi_epi32(v16_10, v16_14);
      const __m128i v32_14 = _mm_unpacklo_epi32(v16_11, v16_15);
      const __m128i v32_15 = _mm_unpackhi_epi32(v16_11, v16_15);
      const __m128i v64_0 = _mm_unpacklo_epi64(v32_0, v32_8);
      const __m128i v64_1 = _mm_unpackhi_epi64(v32_0, v32_8);
      const __m128i v64_2 = _mm_unpacklo_epi64(v32_1, v32_9);
      const __m128i v64_3 = _mm_unpackhi_epi64(v32_1, v32_9);
      const __m128i v64_4 = _mm_unpacklo_epi64(v32_2, v32_10);
      const __m128i v64_5 = _mm_unpackhi_epi64(v32_2, v32_10);
      const __m128i v64_6 = _mm_unpacklo_epi64(v32_3, v32_11);
      const __m128i v64_7 = _mm_unpackhi_epi64(v32_3, v32_11);
      const __m128i v64_8 = _mm_unpacklo_epi64(v32_4, v32_12);
      const __m128i v64_9 = _mm_unpackhi_epi64(v32_4, v32_12);
      const __m128i v64_10 = _mm_unpacklo_epi64(v32_5, v32_13);
      const __m128i v64_11 = _mm_unpackhi_epi64(v32_5, v32_13);
      const __m128i v64_12 = _mm_unpacklo_epi64(v32_6, v32_14);
      const __m128i v64_13 = _mm_unpackhi_epi64(v32_6, v32_14);
      const __m128i v64_14 = _mm_unpacklo_epi64(v32_7, v32_15);
      const __m128i v64_15 = _mm_unpackhi_epi64(v32_7, v32_15);

      size_t rem = min(block_width - 1, 15);
      uint16_t* o1 = (uint16_t*) ((uintptr_t) o0 + rem * output_stride);
      switch(rem){
      case (15):
        _mm_storeu_si128((__m128i*) o1, v64_15);
        o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
      case (14):
        _mm_storeu_si128((__m128i*) o1, v64_14);
        o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
      case (13):
        _mm_storeu_si128((__m128i*) o1, v64_13);
        o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
      case (12):
        _mm_storeu_si128((__m128i*) o1, v64_12);
        o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
      case (11):
        _mm_storeu_si128((__m128i*) o1, v64_11);
        o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
      case (10):
        _mm_storeu_si128((__m128i*) o1, v64_10);
        o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
      case (9):
        _mm_storeu_si128((__m128i*) o1, v64_9);
        o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
      case (8):
        _mm_storeu_si128((__m128i*) o1, v64_8);
        o1 = (uint16_t*) ((uintptr_t) o1 - output_stride);
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
      o0 = (uint8_t*) ((uintptr_t) o0 + tile_hbytes);
    }

    if (bh != 0) {
      const __m128i v0 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 1) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const __m128i v1 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 2) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const __m128i v2 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 3) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const __m128i v3 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 4) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const __m128i v4 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 5) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const __m128i v5 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 6) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const __m128i v6 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 7) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const __m128i v7 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 8) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const __m128i v8 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 9) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const __m128i v9 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 10) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const __m128i v10 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 11) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const __m128i v11 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 12) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const __m128i v12 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 13) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const __m128i v13 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 14) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const __m128i v14 = _mm_loadu_si128((const __m128i*) i0);
      if XNN_UNPREDICTABLE(bh > 15) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      __m128i v15 = _mm_setzero_si128();

      const __m128i v8_0 = _mm_unpacklo_epi8(v0, v1);
      const __m128i v8_1 = _mm_unpackhi_epi8(v0, v1);
      const __m128i v8_2 = _mm_unpacklo_epi8(v2, v3);
      const __m128i v8_3 = _mm_unpackhi_epi8(v2, v3);
      const __m128i v8_4 = _mm_unpacklo_epi8(v4, v5);
      const __m128i v8_5 = _mm_unpackhi_epi8(v4, v5);
      const __m128i v8_6 = _mm_unpacklo_epi8(v6, v7);
      const __m128i v8_7 = _mm_unpackhi_epi8(v6, v7);
      const __m128i v8_8 = _mm_unpacklo_epi8(v8, v9);
      const __m128i v8_9 = _mm_unpackhi_epi8(v8, v9);
      const __m128i v8_10 = _mm_unpacklo_epi8(v10, v11);
      const __m128i v8_11 = _mm_unpackhi_epi8(v10, v11);
      const __m128i v8_12 = _mm_unpacklo_epi8(v12, v13);
      const __m128i v8_13 = _mm_unpackhi_epi8(v12, v13);
      const __m128i v8_14 = _mm_unpacklo_epi8(v14, v15);
      const __m128i v8_15 = _mm_unpackhi_epi8(v14, v15);

      __m128i v16_0 = _mm_unpacklo_epi16(v8_0, v8_2);
      __m128i v16_1 = _mm_unpackhi_epi16(v8_0, v8_2);
      __m128i v16_2 = _mm_unpacklo_epi16(v8_1, v8_3);
      __m128i v16_3 = _mm_unpackhi_epi16(v8_1, v8_3);
      __m128i v16_4 = _mm_unpacklo_epi16(v8_4, v8_6);
      __m128i v16_5 = _mm_unpackhi_epi16(v8_4, v8_6);
      __m128i v16_6 = _mm_unpacklo_epi16(v8_5, v8_7);
      __m128i v16_7 = _mm_unpackhi_epi16(v8_5, v8_7);
      __m128i v16_8 = _mm_unpacklo_epi16(v8_8, v8_10);
      __m128i v16_9 = _mm_unpackhi_epi16(v8_8, v8_10);
      __m128i v16_10 = _mm_unpacklo_epi16(v8_9, v8_11);
      __m128i v16_11 = _mm_unpackhi_epi16(v8_9, v8_11);
      __m128i v16_12 = _mm_unpacklo_epi16(v8_12, v8_14);
      __m128i v16_13 = _mm_unpackhi_epi16(v8_12, v8_14);
      __m128i v16_14 = _mm_unpacklo_epi16(v8_13, v8_15);
      __m128i v16_15 = _mm_unpackhi_epi16(v8_13, v8_15);
      __m128i v32_0 = _mm_unpacklo_epi32(v16_0, v16_4);
      __m128i v32_1 = _mm_unpackhi_epi32(v16_0, v16_4);
      __m128i v32_2 = _mm_unpacklo_epi32(v16_1, v16_5);
      __m128i v32_3 = _mm_unpackhi_epi32(v16_1, v16_5);
      __m128i v32_4 = _mm_unpacklo_epi32(v16_2, v16_6);
      __m128i v32_5 = _mm_unpackhi_epi32(v16_2, v16_6);
      __m128i v32_6 = _mm_unpacklo_epi32(v16_3, v16_7);
      __m128i v32_7 = _mm_unpackhi_epi32(v16_3, v16_7);
      __m128i v32_8 = _mm_unpacklo_epi32(v16_8, v16_12);
      __m128i v32_9 = _mm_unpackhi_epi32(v16_8, v16_12);
      __m128i v32_10 = _mm_unpacklo_epi32(v16_9, v16_13);
      __m128i v32_11 = _mm_unpackhi_epi32(v16_9, v16_13);
      __m128i v32_12 = _mm_unpacklo_epi32(v16_10, v16_14);
      __m128i v32_13 = _mm_unpackhi_epi32(v16_10, v16_14);
      __m128i v32_14 = _mm_unpacklo_epi32(v16_11, v16_15);
      __m128i v32_15 = _mm_unpackhi_epi32(v16_11, v16_15);
      __m128i v64_0 = _mm_unpacklo_epi64(v32_0, v32_8);
      __m128i v64_1 = _mm_unpackhi_epi64(v32_0, v32_8);
      __m128i v64_2 = _mm_unpacklo_epi64(v32_1, v32_9);
      __m128i v64_3 = _mm_unpackhi_epi64(v32_1, v32_9);
      __m128i v64_4 = _mm_unpacklo_epi64(v32_2, v32_10);
      __m128i v64_5 = _mm_unpackhi_epi64(v32_2, v32_10);
      __m128i v64_6 = _mm_unpacklo_epi64(v32_3, v32_11);
      __m128i v64_7 = _mm_unpackhi_epi64(v32_3, v32_11);
      __m128i v64_8 = _mm_unpacklo_epi64(v32_4, v32_12);
      __m128i v64_9 = _mm_unpackhi_epi64(v32_4, v32_12);
      __m128i v64_10 = _mm_unpacklo_epi64(v32_5, v32_13);
      __m128i v64_11 = _mm_unpackhi_epi64(v32_5, v32_13);
      __m128i v64_12 = _mm_unpacklo_epi64(v32_6, v32_14);
      __m128i v64_13 = _mm_unpackhi_epi64(v32_6, v32_14);
      __m128i v64_14 = _mm_unpacklo_epi64(v32_7, v32_15);
      __m128i v64_15 = _mm_unpackhi_epi64(v32_7, v32_15);

      size_t rem = min(block_width - 1, 15);
      if (bh & 8) {
        uint8_t* o1 = (uint8_t*) ((uintptr_t) o0 + rem * output_stride);
        switch(rem){
        case (15):
          _mm_storel_epi64((__m128i*) o1, v64_15);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (14):
          _mm_storel_epi64((__m128i*) o1, v64_14);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (13):
          _mm_storel_epi64((__m128i*) o1, v64_13);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (12):
          _mm_storel_epi64((__m128i*) o1, v64_12);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (11):
          _mm_storel_epi64((__m128i*) o1, v64_11);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (10):
          _mm_storel_epi64((__m128i*) o1, v64_10);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (9):
          _mm_storel_epi64((__m128i*) o1, v64_9);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (8):
          _mm_storel_epi64((__m128i*) o1, v64_8);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (7):
          _mm_storel_epi64((__m128i*) o1, v64_7);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (6):
          _mm_storel_epi64((__m128i*) o1, v64_6);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (5):
          _mm_storel_epi64((__m128i*) o1, v64_5);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (4):
          _mm_storel_epi64((__m128i*) o1, v64_4);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (3):
          _mm_storel_epi64((__m128i*) o1, v64_3);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (2):
          _mm_storel_epi64((__m128i*) o1, v64_2);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (1):
          _mm_storel_epi64((__m128i*) o1, v64_1);
        }
        _mm_storel_epi64((__m128i*) o0, v64_0);
        o0 += 8;
        v64_0 = _mm_unpackhi_epi64(v64_0, v64_0);
        v64_1 = _mm_unpackhi_epi64(v64_1, v64_1);
        v64_2 = _mm_unpackhi_epi64(v64_2, v64_2);
        v64_3 = _mm_unpackhi_epi64(v64_3, v64_3);
        v64_4 = _mm_unpackhi_epi64(v64_4, v64_4);
        v64_5 = _mm_unpackhi_epi64(v64_5, v64_5);
        v64_6 = _mm_unpackhi_epi64(v64_6, v64_6);
        v64_7 = _mm_unpackhi_epi64(v64_7, v64_7);
        v64_8 = _mm_unpackhi_epi64(v64_8, v64_8);
        v64_9 = _mm_unpackhi_epi64(v64_9, v64_9);
        v64_10 = _mm_unpackhi_epi64(v64_10, v64_10);
        v64_11 = _mm_unpackhi_epi64(v64_11, v64_11);
        v64_12 = _mm_unpackhi_epi64(v64_12, v64_12);
        v64_13 = _mm_unpackhi_epi64(v64_13, v64_13);
        v64_14 = _mm_unpackhi_epi64(v64_14, v64_14);
        v64_15 = _mm_unpackhi_epi64(v64_15, v64_15);
      }

      if (bh & 4) {
        uint8_t* o1 = (uint8_t*) ((uintptr_t) o0 + rem * output_stride);
        switch(rem){
        case (15):
          *((int*) o1) = _mm_cvtsi128_si32(v64_15);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (14):
          *((int*) o1) = _mm_cvtsi128_si32(v64_14);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (13):
          *((int*) o1) = _mm_cvtsi128_si32(v64_13);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (12):
          *((int*) o1) = _mm_cvtsi128_si32(v64_12);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (11):
          *((int*) o1) = _mm_cvtsi128_si32(v64_11);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (10):
          *((int*) o1) = _mm_cvtsi128_si32(v64_10);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (9):
          *((int*) o1) = _mm_cvtsi128_si32(v64_9);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (8):
          *((int*) o1) = _mm_cvtsi128_si32(v64_8);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (7):
          *((int*) o1) = _mm_cvtsi128_si32(v64_7);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (6):
          *((int*) o1) = _mm_cvtsi128_si32(v64_6);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (5):
          *((int*) o1) = _mm_cvtsi128_si32(v64_5);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (4):
          *((int*) o1) = _mm_cvtsi128_si32(v64_4);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (3):
          *((int*) o1) = _mm_cvtsi128_si32(v64_3);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (2):
          *((int*) o1) = _mm_cvtsi128_si32(v64_2);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (1):
          *((int*) o1) = _mm_cvtsi128_si32(v64_1);
        }
        *((int*) o0) = _mm_cvtsi128_si32(v64_0);
        o0 += 4;
        v64_0 = _mm_srli_epi64(v64_0, 32);
        v64_1 = _mm_srli_epi64(v64_1, 32);
        v64_2 = _mm_srli_epi64(v64_2, 32);
        v64_3 = _mm_srli_epi64(v64_3, 32);
        v64_4 = _mm_srli_epi64(v64_4, 32);
        v64_5 = _mm_srli_epi64(v64_5, 32);
        v64_6 = _mm_srli_epi64(v64_6, 32);
        v64_7 = _mm_srli_epi64(v64_7, 32);
        v64_8 = _mm_srli_epi64(v64_8, 32);
        v64_9 = _mm_srli_epi64(v64_9, 32);
        v64_10 = _mm_srli_epi64(v64_10, 32);
        v64_11 = _mm_srli_epi64(v64_11, 32);
        v64_12 = _mm_srli_epi64(v64_12, 32);
        v64_13 = _mm_srli_epi64(v64_13, 32);
        v64_14 = _mm_srli_epi64(v64_14, 32);
        v64_15 = _mm_srli_epi64(v64_15, 32);
      }
      if (bh & 2) {
        uint8_t* o1 = (uint8_t*) ((uintptr_t) o0 + rem * output_stride);
        switch(rem){
        case (15):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_15);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (14):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_14);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (13):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_13);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (12):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_12);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (11):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_11);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (10):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_10);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (9):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_9);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (8):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_8);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (7):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_7);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (6):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_6);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (5):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_5);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (4):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_4);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (3):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_3);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (2):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_2);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (1):
          *((uint16_t*) o1) = (uint16_t) _mm_cvtsi128_si32(v64_1);
        }
        *((uint16_t*) o0) = (uint16_t) _mm_cvtsi128_si32(v64_0);
        o0 += 2;
        v64_0 = _mm_srli_epi32(v64_0, 16);
        v64_1 = _mm_srli_epi32(v64_1, 16);
        v64_2 = _mm_srli_epi32(v64_2, 16);
        v64_3 = _mm_srli_epi32(v64_3, 16);
        v64_4 = _mm_srli_epi32(v64_4, 16);
        v64_5 = _mm_srli_epi32(v64_5, 16);
        v64_6 = _mm_srli_epi32(v64_6, 16);
        v64_7 = _mm_srli_epi32(v64_7, 16);
        v64_8 = _mm_srli_epi32(v64_8, 16);
        v64_9 = _mm_srli_epi32(v64_9, 16);
        v64_10 = _mm_srli_epi32(v64_10, 16);
        v64_11 = _mm_srli_epi32(v64_11, 16);
        v64_12 = _mm_srli_epi32(v64_12, 16);
        v64_13 = _mm_srli_epi32(v64_13, 16);
        v64_14 = _mm_srli_epi32(v64_14, 16);
        v64_15 = _mm_srli_epi32(v64_15, 16);
      }
      if (bh & 1) {
        uint8_t* o1 = (uint8_t*) ((uintptr_t) o0 + rem * output_stride);
        switch(rem){
        case (15):
          *o1 = (uint8_t) _mm_cvtsi128_si32(v64_15);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (14):
          *o1 = (uint8_t) _mm_cvtsi128_si32(v64_14);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (13):
          *o1 = (uint8_t) _mm_cvtsi128_si32(v64_13);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (12):
          *o1 = (uint8_t) _mm_cvtsi128_si32(v64_12);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (11):
          *o1 = (uint8_t) _mm_cvtsi128_si32(v64_11);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (10):
          *o1 = (uint8_t) _mm_cvtsi128_si32(v64_10);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (9):
          *o1 = (uint8_t) _mm_cvtsi128_si32(v64_9);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (8):
          *o1 = (uint8_t) _mm_cvtsi128_si32(v64_8);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (7):
          *o1 = (uint8_t) _mm_cvtsi128_si32(v64_7);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (6):
          *o1 = (uint8_t) _mm_cvtsi128_si32(v64_6);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (5):
          *o1 = (uint8_t) _mm_cvtsi128_si32(v64_5);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (4):
          *o1 = (uint8_t) _mm_cvtsi128_si32(v64_4);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (3):
          *o1 = (uint8_t) _mm_cvtsi128_si32(v64_3);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (2):
          *o1 = (uint8_t) _mm_cvtsi128_si32(v64_2);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case (1):
          *o1 = (uint8_t) _mm_cvtsi128_si32(v64_1);
        }
        *o0 = (uint8_t) _mm_cvtsi128_si32(v64_0);
      }
    }

    i0 = (const uint8_t*) ((uintptr_t) i0 + input_reset);
    o0 = (uint8_t*) ((uintptr_t) o0 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
