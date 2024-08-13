// Auto-generated file. Do not edit!
//   Template: src/x32-transposec/avx2.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <immintrin.h>

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/transpose.h"
#include "xnnpack/unaligned.h"

void xnn_x16_transposec_ukernel__16x16_reuse_switch_avx2(
    const uint16_t* input,
    uint16_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint16_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint16_t));

  static const int32_t mask_table[15] = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  const size_t tile_height = 16;
  const size_t tile_width = 16;
  const size_t tile_hbytes = tile_height * sizeof(uint16_t);
  const size_t tile_wbytes = tile_width * sizeof(uint16_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  uint16_t* o = (uint16_t*) output;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint16_t);

  const uint16_t* i0 = (const uint16_t*) input;
  const size_t minus_output_stride = -output_stride;

  do {
    const size_t rem = min(block_width - 1, 15);
    const size_t oN_stride = rem * output_stride;

    __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7 ^ (rem>>1)]));

    size_t bh = block_height;
    for (; bh >= 16; bh -= 16) {
      const __m256i v4_0 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_1 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_2 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_3 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_4 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_5 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_6 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_7 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_8 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_9 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_10 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_11 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_12 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_13 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_14 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v4_15 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);

      const __m256i v3_0 = _mm256_unpacklo_epi16(v4_0, v4_4);
      const __m256i v3_1 = _mm256_unpackhi_epi16(v4_0, v4_4);
      const __m256i v3_2 = _mm256_unpacklo_epi16(v4_1, v4_5);
      const __m256i v3_3 = _mm256_unpackhi_epi16(v4_1, v4_5);
      const __m256i v3_4 = _mm256_unpacklo_epi16(v4_2, v4_6);
      const __m256i v3_5 = _mm256_unpackhi_epi16(v4_2, v4_6);
      const __m256i v3_6 = _mm256_unpacklo_epi16(v4_3, v4_7);
      const __m256i v3_7 = _mm256_unpackhi_epi16(v4_3, v4_7);
      const __m256i v3_8 = _mm256_unpacklo_epi16(v4_8, v4_12);
      const __m256i v3_9 = _mm256_unpackhi_epi16(v4_8, v4_12);
      const __m256i v3_10 = _mm256_unpacklo_epi16(v4_9, v4_13);
      const __m256i v3_11 = _mm256_unpackhi_epi16(v4_9, v4_13);
      const __m256i v3_12 = _mm256_unpacklo_epi16(v4_10, v4_14);
      const __m256i v3_13 = _mm256_unpackhi_epi16(v4_10, v4_14);
      const __m256i v3_14 = _mm256_unpacklo_epi16(v4_11, v4_15);
      const __m256i v3_15 = _mm256_unpackhi_epi16(v4_11, v4_15);
      const __m256i v2_0 = _mm256_unpacklo_epi16(v3_0, v3_4);
      const __m256i v2_1 = _mm256_unpackhi_epi16(v3_0, v3_4);
      const __m256i v2_2 = _mm256_unpacklo_epi16(v3_1, v3_5);
      const __m256i v2_3 = _mm256_unpackhi_epi16(v3_1, v3_5);
      const __m256i v2_4 = _mm256_unpacklo_epi16(v3_2, v3_6);
      const __m256i v2_5 = _mm256_unpackhi_epi16(v3_2, v3_6);
      const __m256i v2_6 = _mm256_unpacklo_epi16(v3_3, v3_7);
      const __m256i v2_7 = _mm256_unpackhi_epi16(v3_3, v3_7);
      const __m256i v2_8 = _mm256_unpacklo_epi16(v3_8, v3_12);
      const __m256i v2_9 = _mm256_unpackhi_epi16(v3_8, v3_12);
      const __m256i v2_10 = _mm256_unpacklo_epi16(v3_9, v3_13);
      const __m256i v2_11 = _mm256_unpackhi_epi16(v3_9, v3_13);
      const __m256i v2_12 = _mm256_unpacklo_epi16(v3_10, v3_14);
      const __m256i v2_13 = _mm256_unpackhi_epi16(v3_10, v3_14);
      const __m256i v2_14 = _mm256_unpacklo_epi16(v3_11, v3_15);
      const __m256i v2_15 = _mm256_unpackhi_epi16(v3_11, v3_15);
      const __m256i v1_0 = _mm256_unpacklo_epi16(v2_0, v2_4);
      const __m256i v1_1 = _mm256_unpackhi_epi16(v2_0, v2_4);
      const __m256i v1_2 = _mm256_unpacklo_epi16(v2_1, v2_5);
      const __m256i v1_3 = _mm256_unpackhi_epi16(v2_1, v2_5);
      const __m256i v1_4 = _mm256_unpacklo_epi16(v2_2, v2_6);
      const __m256i v1_5 = _mm256_unpackhi_epi16(v2_2, v2_6);
      const __m256i v1_6 = _mm256_unpacklo_epi16(v2_3, v2_7);
      const __m256i v1_7 = _mm256_unpackhi_epi16(v2_3, v2_7);
      const __m256i v1_8 = _mm256_unpacklo_epi16(v2_8, v2_12);
      const __m256i v1_9 = _mm256_unpackhi_epi16(v2_8, v2_12);
      const __m256i v1_10 = _mm256_unpacklo_epi16(v2_9, v2_13);
      const __m256i v1_11 = _mm256_unpackhi_epi16(v2_9, v2_13);
      const __m256i v1_12 = _mm256_unpacklo_epi16(v2_10, v2_14);
      const __m256i v1_13 = _mm256_unpackhi_epi16(v2_10, v2_14);
      const __m256i v1_14 = _mm256_unpacklo_epi16(v2_11, v2_15);
      const __m256i v1_15 = _mm256_unpackhi_epi16(v2_11, v2_15);


      uint16_t* oN = (uint16_t*) ((uintptr_t) o + oN_stride);
      switch (rem) {
        default:
          XNN_UNREACHABLE;
        case 15: {
          const __m256i v0_15 = _mm256_permute2f128_si256(v1_7, v1_15, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_15);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 14: {
          const __m256i v0_14 = _mm256_permute2f128_si256(v1_6, v1_14, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_14);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 13: {
          const __m256i v0_13 = _mm256_permute2f128_si256(v1_5, v1_13, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_13);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 12: {
          const __m256i v0_12 = _mm256_permute2f128_si256(v1_4, v1_12, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_12);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 11: {
          const __m256i v0_11 = _mm256_permute2f128_si256(v1_3, v1_11, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_11);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 10: {
          const __m256i v0_10 = _mm256_permute2f128_si256(v1_2, v1_10, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_10);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 9: {
          const __m256i v0_9 = _mm256_permute2f128_si256(v1_1, v1_9, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_9);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 8: {
          const __m256i v0_8 = _mm256_permute2f128_si256(v1_0, v1_8, 0x31);
          _mm256_storeu_si256((__m256i*) oN, v0_8);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 7: {
          const __m256i v0_7 = _mm256_insertf128_si256(v1_7, _mm256_castsi256_si128(v1_15), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_7);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 6: {
          const __m256i v0_6 = _mm256_insertf128_si256(v1_6, _mm256_castsi256_si128(v1_14), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_6);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 5: {
          const __m256i v0_5 = _mm256_insertf128_si256(v1_5, _mm256_castsi256_si128(v1_13), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_5);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 4: {
          const __m256i v0_4 = _mm256_insertf128_si256(v1_4, _mm256_castsi256_si128(v1_12), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_4);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 3: {
          const __m256i v0_3 = _mm256_insertf128_si256(v1_3, _mm256_castsi256_si128(v1_11), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_3);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 2: {
          const __m256i v0_2 = _mm256_insertf128_si256(v1_2, _mm256_castsi256_si128(v1_10), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_2);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
        }
        XNN_FALLTHROUGH
        case 1: {
          const __m256i v0_1 = _mm256_insertf128_si256(v1_1, _mm256_castsi256_si128(v1_9), 1);
          _mm256_storeu_si256((__m256i*) oN, v0_1);
        }
        XNN_FALLTHROUGH
        case 0: {
          const __m256i v0_0 = _mm256_insertf128_si256(v1_0, _mm256_castsi256_si128(v1_8), 1);
          _mm256_storeu_si256((__m256i*) o, v0_0);
          o = (uint16_t*) ((uintptr_t) o + tile_hbytes);
        }
      }
    }
    if (bh != 0) {
      const __m256i v4_0 = _mm256_maskload_epi32((const int*) i0, vmask);
      const uint16_t *i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const __m256i v4_1 = _mm256_maskload_epi32((const int*) i1, vmask);
      const uint16_t *i2 = (const uint16_t*) ((uintptr_t) i1 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i1;
      }
      const __m256i v4_2 = _mm256_maskload_epi32((const int*) i2, vmask);
      const uint16_t *i3 = (const uint16_t*) ((uintptr_t) i2 + input_stride);
      if XNN_UNPREDICTABLE(bh < 4) {
        i3 = i2;
      }
      const __m256i v4_3 = _mm256_maskload_epi32((const int*) i3, vmask);
      const uint16_t *i4 = (const uint16_t*) ((uintptr_t) i3 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 4) {
        i4 = i3;
      }
      const __m256i v4_4 = _mm256_maskload_epi32((const int*) i4, vmask);
      const uint16_t *i5 = (const uint16_t*) ((uintptr_t) i4 + input_stride);
      if XNN_UNPREDICTABLE(bh < 6) {
        i5 = i4;
      }
      const __m256i v4_5 = _mm256_maskload_epi32((const int*) i5, vmask);
      const uint16_t *i6 = (const uint16_t*) ((uintptr_t) i5 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 6) {
        i6 = i5;
      }
      const __m256i v4_6 = _mm256_maskload_epi32((const int*) i6, vmask);
      const uint16_t *i7 = (const uint16_t*) ((uintptr_t) i6 + input_stride);
      if XNN_UNPREDICTABLE(bh < 8) {
        i7 = i6;
      }
      const __m256i v4_7 = _mm256_maskload_epi32((const int*) i7, vmask);
      const uint16_t *i8 = (const uint16_t*) ((uintptr_t) i7 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 8) {
        i8 = i7;
      }
      const __m256i v4_8 = _mm256_maskload_epi32((const int*) i8, vmask);
      const uint16_t *i9 = (const uint16_t*) ((uintptr_t) i8 + input_stride);
      if XNN_UNPREDICTABLE(bh < 10) {
        i9 = i8;
      }
      const __m256i v4_9 = _mm256_maskload_epi32((const int*) i9, vmask);
      const uint16_t *i10 = (const uint16_t*) ((uintptr_t) i9 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 10) {
        i10 = i9;
      }
      const __m256i v4_10 = _mm256_maskload_epi32((const int*) i10, vmask);
      const uint16_t *i11 = (const uint16_t*) ((uintptr_t) i10 + input_stride);
      if XNN_UNPREDICTABLE(bh < 12) {
        i11 = i10;
      }
      const __m256i v4_11 = _mm256_maskload_epi32((const int*) i11, vmask);
      const uint16_t *i12 = (const uint16_t*) ((uintptr_t) i11 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 12) {
        i12 = i11;
      }
      const __m256i v4_12 = _mm256_maskload_epi32((const int*) i12, vmask);
      const uint16_t *i13 = (const uint16_t*) ((uintptr_t) i12 + input_stride);
      if XNN_UNPREDICTABLE(bh < 14) {
        i13 = i12;
      }
      const __m256i v4_13 = _mm256_maskload_epi32((const int*) i13, vmask);
      const uint16_t *i14 = (const uint16_t*) ((uintptr_t) i13 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 14) {
        i14 = i13;
      }
      const __m256i v4_14 = _mm256_maskload_epi32((const int*) i14, vmask);
      const __m256i v4_15 = _mm256_undefined_si256();

      const __m256i v3_0 = _mm256_unpacklo_epi16(v4_0, v4_4);
      const __m256i v3_1 = _mm256_unpackhi_epi16(v4_0, v4_4);
      const __m256i v3_2 = _mm256_unpacklo_epi16(v4_1, v4_5);
      const __m256i v3_3 = _mm256_unpackhi_epi16(v4_1, v4_5);
      const __m256i v3_4 = _mm256_unpacklo_epi16(v4_2, v4_6);
      const __m256i v3_5 = _mm256_unpackhi_epi16(v4_2, v4_6);
      const __m256i v3_6 = _mm256_unpacklo_epi16(v4_3, v4_7);
      const __m256i v3_7 = _mm256_unpackhi_epi16(v4_3, v4_7);
      const __m256i v3_8 = _mm256_unpacklo_epi16(v4_8, v4_12);
      const __m256i v3_9 = _mm256_unpackhi_epi16(v4_8, v4_12);
      const __m256i v3_10 = _mm256_unpacklo_epi16(v4_9, v4_13);
      const __m256i v3_11 = _mm256_unpackhi_epi16(v4_9, v4_13);
      const __m256i v3_12 = _mm256_unpacklo_epi16(v4_10, v4_14);
      const __m256i v3_13 = _mm256_unpackhi_epi16(v4_10, v4_14);
      const __m256i v3_14 = _mm256_unpacklo_epi16(v4_11, v4_15);
      const __m256i v3_15 = _mm256_unpackhi_epi16(v4_11, v4_15);
      const __m256i v2_0 = _mm256_unpacklo_epi16(v3_0, v3_4);
      const __m256i v2_1 = _mm256_unpackhi_epi16(v3_0, v3_4);
      const __m256i v2_2 = _mm256_unpacklo_epi16(v3_1, v3_5);
      const __m256i v2_3 = _mm256_unpackhi_epi16(v3_1, v3_5);
      const __m256i v2_4 = _mm256_unpacklo_epi16(v3_2, v3_6);
      const __m256i v2_5 = _mm256_unpackhi_epi16(v3_2, v3_6);
      const __m256i v2_6 = _mm256_unpacklo_epi16(v3_3, v3_7);
      const __m256i v2_7 = _mm256_unpackhi_epi16(v3_3, v3_7);
      const __m256i v2_8 = _mm256_unpacklo_epi16(v3_8, v3_12);
      const __m256i v2_9 = _mm256_unpackhi_epi16(v3_8, v3_12);
      const __m256i v2_10 = _mm256_unpacklo_epi16(v3_9, v3_13);
      const __m256i v2_11 = _mm256_unpackhi_epi16(v3_9, v3_13);
      const __m256i v2_12 = _mm256_unpacklo_epi16(v3_10, v3_14);
      const __m256i v2_13 = _mm256_unpackhi_epi16(v3_10, v3_14);
      const __m256i v2_14 = _mm256_unpacklo_epi16(v3_11, v3_15);
      const __m256i v2_15 = _mm256_unpackhi_epi16(v3_11, v3_15);
      const __m256i v1_0 = _mm256_unpacklo_epi16(v2_0, v2_4);
      const __m256i v1_1 = _mm256_unpackhi_epi16(v2_0, v2_4);
      const __m256i v1_2 = _mm256_unpacklo_epi16(v2_1, v2_5);
      const __m256i v1_3 = _mm256_unpackhi_epi16(v2_1, v2_5);
      const __m256i v1_4 = _mm256_unpacklo_epi16(v2_2, v2_6);
      const __m256i v1_5 = _mm256_unpackhi_epi16(v2_2, v2_6);
      const __m256i v1_6 = _mm256_unpacklo_epi16(v2_3, v2_7);
      const __m256i v1_7 = _mm256_unpackhi_epi16(v2_3, v2_7);
      const __m256i v1_8 = _mm256_unpacklo_epi16(v2_8, v2_12);
      const __m256i v1_9 = _mm256_unpackhi_epi16(v2_8, v2_12);
      const __m256i v1_10 = _mm256_unpacklo_epi16(v2_9, v2_13);
      const __m256i v1_11 = _mm256_unpackhi_epi16(v2_9, v2_13);
      const __m256i v1_12 = _mm256_unpacklo_epi16(v2_10, v2_14);
      const __m256i v1_13 = _mm256_unpackhi_epi16(v2_10, v2_14);
      const __m256i v1_14 = _mm256_unpacklo_epi16(v2_11, v2_15);
      const __m256i v1_15 = _mm256_unpackhi_epi16(v2_11, v2_15);

      __m128i v0_0_lo = _mm256_castsi256_si128(v1_0);
      __m128i v0_1_lo = _mm256_castsi256_si128(v1_1);
      __m128i v0_2_lo = _mm256_castsi256_si128(v1_2);
      __m128i v0_3_lo = _mm256_castsi256_si128(v1_3);
      __m128i v0_4_lo = _mm256_castsi256_si128(v1_4);
      __m128i v0_5_lo = _mm256_castsi256_si128(v1_5);
      __m128i v0_6_lo = _mm256_castsi256_si128(v1_6);
      __m128i v0_7_lo = _mm256_castsi256_si128(v1_7);
      __m128i v0_8_lo = _mm256_extractf128_si256(v1_0, 0x1);
      __m128i v0_9_lo = _mm256_extractf128_si256(v1_1, 0x1);
      __m128i v0_10_lo = _mm256_extractf128_si256(v1_2, 0x1);
      __m128i v0_11_lo = _mm256_extractf128_si256(v1_3, 0x1);
      __m128i v0_12_lo = _mm256_extractf128_si256(v1_4, 0x1);
      __m128i v0_13_lo = _mm256_extractf128_si256(v1_5, 0x1);
      __m128i v0_14_lo = _mm256_extractf128_si256(v1_6, 0x1);
      __m128i v0_15_lo = _mm256_extractf128_si256(v1_7, 0x1);

      if (bh & 8) {
        uint16_t* oN = (uint16_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 15:
            _mm_storeu_si128((__m128i*) oN, v0_15_lo);
             v0_15_lo = _mm256_extractf128_si256(v1_15, 0x1);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 14:
            _mm_storeu_si128((__m128i*) oN, v0_14_lo);
             v0_14_lo = _mm256_extractf128_si256(v1_14, 0x1);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 13:
            _mm_storeu_si128((__m128i*) oN, v0_13_lo);
             v0_13_lo = _mm256_extractf128_si256(v1_13, 0x1);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 12:
            _mm_storeu_si128((__m128i*) oN, v0_12_lo);
             v0_12_lo = _mm256_extractf128_si256(v1_12, 0x1);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 11:
            _mm_storeu_si128((__m128i*) oN, v0_11_lo);
             v0_11_lo = _mm256_extractf128_si256(v1_11, 0x1);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 10:
            _mm_storeu_si128((__m128i*) oN, v0_10_lo);
             v0_10_lo = _mm256_extractf128_si256(v1_10, 0x1);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 9:
            _mm_storeu_si128((__m128i*) oN, v0_9_lo);
             v0_9_lo = _mm256_extractf128_si256(v1_9, 0x1);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 8:
            _mm_storeu_si128((__m128i*) oN, v0_8_lo);
             v0_8_lo = _mm256_extractf128_si256(v1_8, 0x1);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 7:
            _mm_storeu_si128((__m128i*) oN, v0_7_lo);
             v0_7_lo = _mm256_castsi256_si128(v1_15);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 6:
            _mm_storeu_si128((__m128i*) oN, v0_6_lo);
             v0_6_lo = _mm256_castsi256_si128(v1_14);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 5:
            _mm_storeu_si128((__m128i*) oN, v0_5_lo);
             v0_5_lo = _mm256_castsi256_si128(v1_13);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 4:
            _mm_storeu_si128((__m128i*) oN, v0_4_lo);
             v0_4_lo = _mm256_castsi256_si128(v1_12);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 3:
            _mm_storeu_si128((__m128i*) oN, v0_3_lo);
             v0_3_lo = _mm256_castsi256_si128(v1_11);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 2:
            _mm_storeu_si128((__m128i*) oN, v0_2_lo);
             v0_2_lo = _mm256_castsi256_si128(v1_10);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 1:
            _mm_storeu_si128((__m128i*) oN, v0_1_lo);
            v0_1_lo = _mm256_castsi256_si128(v1_9);
            XNN_FALLTHROUGH
          case 0:
            _mm_storeu_si128((__m128i*) o, v0_0_lo);
            v0_0_lo = _mm256_castsi256_si128(v1_8);
            break;
          default:
            XNN_UNREACHABLE;
        }
        o += 8;
      }

      if (bh & 4) {
        uint16_t* oN = (uint16_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 15:
            _mm_storel_epi64((__m128i*) oN, v0_15_lo);
            v0_15_lo = _mm_unpackhi_epi64(v0_15_lo, v0_15_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 14:
            _mm_storel_epi64((__m128i*) oN, v0_14_lo);
            v0_14_lo = _mm_unpackhi_epi64(v0_14_lo, v0_14_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 13:
            _mm_storel_epi64((__m128i*) oN, v0_13_lo);
            v0_13_lo = _mm_unpackhi_epi64(v0_13_lo, v0_13_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 12:
            _mm_storel_epi64((__m128i*) oN, v0_12_lo);
            v0_12_lo = _mm_unpackhi_epi64(v0_12_lo, v0_12_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 11:
            _mm_storel_epi64((__m128i*) oN, v0_11_lo);
            v0_11_lo = _mm_unpackhi_epi64(v0_11_lo, v0_11_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 10:
            _mm_storel_epi64((__m128i*) oN, v0_10_lo);
            v0_10_lo = _mm_unpackhi_epi64(v0_10_lo, v0_10_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 9:
            _mm_storel_epi64((__m128i*) oN, v0_9_lo);
            v0_9_lo = _mm_unpackhi_epi64(v0_9_lo, v0_9_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 8:
            _mm_storel_epi64((__m128i*) oN, v0_8_lo);
            v0_8_lo = _mm_unpackhi_epi64(v0_8_lo, v0_8_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 7:
            _mm_storel_epi64((__m128i*) oN, v0_7_lo);
            v0_7_lo = _mm_unpackhi_epi64(v0_7_lo, v0_7_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 6:
            _mm_storel_epi64((__m128i*) oN, v0_6_lo);
            v0_6_lo = _mm_unpackhi_epi64(v0_6_lo, v0_6_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 5:
            _mm_storel_epi64((__m128i*) oN, v0_5_lo);
            v0_5_lo = _mm_unpackhi_epi64(v0_5_lo, v0_5_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 4:
            _mm_storel_epi64((__m128i*) oN, v0_4_lo);
            v0_4_lo = _mm_unpackhi_epi64(v0_4_lo, v0_4_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 3:
            _mm_storel_epi64((__m128i*) oN, v0_3_lo);
            v0_3_lo = _mm_unpackhi_epi64(v0_3_lo, v0_3_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 2:
            _mm_storel_epi64((__m128i*) oN, v0_2_lo);
            v0_2_lo = _mm_unpackhi_epi64(v0_2_lo, v0_2_lo);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 1:
            _mm_storel_epi64((__m128i*) oN, v0_1_lo);
            v0_1_lo = _mm_unpackhi_epi64(v0_1_lo, v0_1_lo);
            XNN_FALLTHROUGH
          case 0:
            _mm_storel_epi64((__m128i*) o, v0_0_lo);
            v0_0_lo = _mm_unpackhi_epi64(v0_0_lo, v0_0_lo);
            break;
          default:
            XNN_UNREACHABLE;
        }
        o += 4;
      }
      if (bh & 2) {
        uint16_t* oN = (uint16_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 15:
            _mm_storeu_si32(oN, v0_15_lo);
            v0_15_lo = _mm_srli_epi64(v0_15_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 14:
            _mm_storeu_si32(oN, v0_14_lo);
            v0_14_lo = _mm_srli_epi64(v0_14_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 13:
            _mm_storeu_si32(oN, v0_13_lo);
            v0_13_lo = _mm_srli_epi64(v0_13_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 12:
            _mm_storeu_si32(oN, v0_12_lo);
            v0_12_lo = _mm_srli_epi64(v0_12_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 11:
            _mm_storeu_si32(oN, v0_11_lo);
            v0_11_lo = _mm_srli_epi64(v0_11_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 10:
            _mm_storeu_si32(oN, v0_10_lo);
            v0_10_lo = _mm_srli_epi64(v0_10_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 9:
            _mm_storeu_si32(oN, v0_9_lo);
            v0_9_lo = _mm_srli_epi64(v0_9_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 8:
            _mm_storeu_si32(oN, v0_8_lo);
            v0_8_lo = _mm_srli_epi64(v0_8_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 7:
            _mm_storeu_si32(oN, v0_7_lo);
            v0_7_lo = _mm_srli_epi64(v0_7_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 6:
            _mm_storeu_si32(oN, v0_6_lo);
            v0_6_lo = _mm_srli_epi64(v0_6_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 5:
            _mm_storeu_si32(oN, v0_5_lo);
            v0_5_lo = _mm_srli_epi64(v0_5_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 4:
            _mm_storeu_si32(oN, v0_4_lo);
            v0_4_lo = _mm_srli_epi64(v0_4_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 3:
            _mm_storeu_si32(oN, v0_3_lo);
            v0_3_lo = _mm_srli_epi64(v0_3_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 2:
            _mm_storeu_si32(oN, v0_2_lo);
            v0_2_lo = _mm_srli_epi64(v0_2_lo, 32);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 1:
            _mm_storeu_si32(oN, v0_1_lo);
            v0_1_lo = _mm_srli_epi64(v0_1_lo, 32);
            XNN_FALLTHROUGH
          case 0:
            _mm_storeu_si32(o, v0_0_lo);
            v0_0_lo = _mm_srli_epi64(v0_0_lo, 32);
            break;
          default:
            XNN_UNREACHABLE;
        }
        o += 2;
      }
      if (bh & 1) {
        uint16_t* oN = (uint16_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 15:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_15_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 14:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_14_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 13:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_13_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 12:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_12_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 11:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_11_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 10:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_10_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 9:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_9_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 8:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_8_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 7:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_7_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 6:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_6_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 5:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_5_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 4:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_4_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 3:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_3_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 2:
            unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_2_lo));
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 1:
             unaligned_store_u16(oN, (uint16_t) _mm_cvtsi128_si32(v0_1_lo));
             XNN_FALLTHROUGH
          case 0:
             unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_0_lo));
            break;
          default:
            XNN_UNREACHABLE;
        }
      }
    }

    i0 = (const uint16_t*) ((uintptr_t) i0 + input_reset);
    o = (uint16_t*) ((uintptr_t) o + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
