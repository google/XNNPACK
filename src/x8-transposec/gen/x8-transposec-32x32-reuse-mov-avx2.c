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

void xnn_x8_transposec_ukernel__32x32_reuse_mov_avx2(
    const uint8_t* input,
    uint8_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint8_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint8_t));

  static const int32_t mask_table[15] = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  const size_t tile_height = 32;
  const size_t tile_width = 32;
  const size_t tile_hbytes = tile_height * sizeof(uint8_t);
  const size_t tile_wbytes = tile_width * sizeof(uint8_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  uint8_t* o = (uint8_t*) ((uintptr_t) output - tile_hbytes);
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint8_t) - tile_hbytes;

  const uint8_t* i0 = (const uint8_t*) input;
  const size_t minus_output_stride = -output_stride;

  do {
    const size_t rem = min(block_width - 1, 31);
    const size_t oN_stride = rem * output_stride;
    const size_t oN_offset = oN_stride + tile_hbytes;

    __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7 ^ (rem>>2)]));

    size_t bh = block_height;
    for (; bh >= 32; bh -= 32) {
      const __m256i v5_0 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_1 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_2 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_3 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_4 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_5 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_6 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_7 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_8 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_9 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_10 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_11 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_12 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_13 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_14 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_15 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_16 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_17 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_18 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_19 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_20 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_21 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_22 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_23 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_24 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_25 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_26 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_27 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_28 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_29 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_30 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const __m256i v5_31 = _mm256_maskload_epi32((const int*) i0, vmask);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);

      const __m256i v4_0 = _mm256_unpacklo_epi8(v5_0, v5_8);
      const __m256i v4_1 = _mm256_unpackhi_epi8(v5_0, v5_8);
      const __m256i v4_2 = _mm256_unpacklo_epi8(v5_1, v5_9);
      const __m256i v4_3 = _mm256_unpackhi_epi8(v5_1, v5_9);
      const __m256i v4_4 = _mm256_unpacklo_epi8(v5_2, v5_10);
      const __m256i v4_5 = _mm256_unpackhi_epi8(v5_2, v5_10);
      const __m256i v4_6 = _mm256_unpacklo_epi8(v5_3, v5_11);
      const __m256i v4_7 = _mm256_unpackhi_epi8(v5_3, v5_11);
      const __m256i v4_8 = _mm256_unpacklo_epi8(v5_4, v5_12);
      const __m256i v4_9 = _mm256_unpackhi_epi8(v5_4, v5_12);
      const __m256i v4_10 = _mm256_unpacklo_epi8(v5_5, v5_13);
      const __m256i v4_11 = _mm256_unpackhi_epi8(v5_5, v5_13);
      const __m256i v4_12 = _mm256_unpacklo_epi8(v5_6, v5_14);
      const __m256i v4_13 = _mm256_unpackhi_epi8(v5_6, v5_14);
      const __m256i v4_14 = _mm256_unpacklo_epi8(v5_7, v5_15);
      const __m256i v4_15 = _mm256_unpackhi_epi8(v5_7, v5_15);
      const __m256i v4_16 = _mm256_unpacklo_epi8(v5_16, v5_24);
      const __m256i v4_17 = _mm256_unpackhi_epi8(v5_16, v5_24);
      const __m256i v4_18 = _mm256_unpacklo_epi8(v5_17, v5_25);
      const __m256i v4_19 = _mm256_unpackhi_epi8(v5_17, v5_25);
      const __m256i v4_20 = _mm256_unpacklo_epi8(v5_18, v5_26);
      const __m256i v4_21 = _mm256_unpackhi_epi8(v5_18, v5_26);
      const __m256i v4_22 = _mm256_unpacklo_epi8(v5_19, v5_27);
      const __m256i v4_23 = _mm256_unpackhi_epi8(v5_19, v5_27);
      const __m256i v4_24 = _mm256_unpacklo_epi8(v5_20, v5_28);
      const __m256i v4_25 = _mm256_unpackhi_epi8(v5_20, v5_28);
      const __m256i v4_26 = _mm256_unpacklo_epi8(v5_21, v5_29);
      const __m256i v4_27 = _mm256_unpackhi_epi8(v5_21, v5_29);
      const __m256i v4_28 = _mm256_unpacklo_epi8(v5_22, v5_30);
      const __m256i v4_29 = _mm256_unpackhi_epi8(v5_22, v5_30);
      const __m256i v4_30 = _mm256_unpacklo_epi8(v5_23, v5_31);
      const __m256i v4_31 = _mm256_unpackhi_epi8(v5_23, v5_31);
      const __m256i v3_0 = _mm256_unpacklo_epi8(v4_0, v4_8);
      const __m256i v3_1 = _mm256_unpackhi_epi8(v4_0, v4_8);
      const __m256i v3_2 = _mm256_unpacklo_epi8(v4_1, v4_9);
      const __m256i v3_3 = _mm256_unpackhi_epi8(v4_1, v4_9);
      const __m256i v3_4 = _mm256_unpacklo_epi8(v4_2, v4_10);
      const __m256i v3_5 = _mm256_unpackhi_epi8(v4_2, v4_10);
      const __m256i v3_6 = _mm256_unpacklo_epi8(v4_3, v4_11);
      const __m256i v3_7 = _mm256_unpackhi_epi8(v4_3, v4_11);
      const __m256i v3_8 = _mm256_unpacklo_epi8(v4_4, v4_12);
      const __m256i v3_9 = _mm256_unpackhi_epi8(v4_4, v4_12);
      const __m256i v3_10 = _mm256_unpacklo_epi8(v4_5, v4_13);
      const __m256i v3_11 = _mm256_unpackhi_epi8(v4_5, v4_13);
      const __m256i v3_12 = _mm256_unpacklo_epi8(v4_6, v4_14);
      const __m256i v3_13 = _mm256_unpackhi_epi8(v4_6, v4_14);
      const __m256i v3_14 = _mm256_unpacklo_epi8(v4_7, v4_15);
      const __m256i v3_15 = _mm256_unpackhi_epi8(v4_7, v4_15);
      const __m256i v3_16 = _mm256_unpacklo_epi8(v4_16, v4_24);
      const __m256i v3_17 = _mm256_unpackhi_epi8(v4_16, v4_24);
      const __m256i v3_18 = _mm256_unpacklo_epi8(v4_17, v4_25);
      const __m256i v3_19 = _mm256_unpackhi_epi8(v4_17, v4_25);
      const __m256i v3_20 = _mm256_unpacklo_epi8(v4_18, v4_26);
      const __m256i v3_21 = _mm256_unpackhi_epi8(v4_18, v4_26);
      const __m256i v3_22 = _mm256_unpacklo_epi8(v4_19, v4_27);
      const __m256i v3_23 = _mm256_unpackhi_epi8(v4_19, v4_27);
      const __m256i v3_24 = _mm256_unpacklo_epi8(v4_20, v4_28);
      const __m256i v3_25 = _mm256_unpackhi_epi8(v4_20, v4_28);
      const __m256i v3_26 = _mm256_unpacklo_epi8(v4_21, v4_29);
      const __m256i v3_27 = _mm256_unpackhi_epi8(v4_21, v4_29);
      const __m256i v3_28 = _mm256_unpacklo_epi8(v4_22, v4_30);
      const __m256i v3_29 = _mm256_unpackhi_epi8(v4_22, v4_30);
      const __m256i v3_30 = _mm256_unpacklo_epi8(v4_23, v4_31);
      const __m256i v3_31 = _mm256_unpackhi_epi8(v4_23, v4_31);
      const __m256i v2_0 = _mm256_unpacklo_epi8(v3_0, v3_8);
      const __m256i v2_1 = _mm256_unpackhi_epi8(v3_0, v3_8);
      const __m256i v2_2 = _mm256_unpacklo_epi8(v3_1, v3_9);
      const __m256i v2_3 = _mm256_unpackhi_epi8(v3_1, v3_9);
      const __m256i v2_4 = _mm256_unpacklo_epi8(v3_2, v3_10);
      const __m256i v2_5 = _mm256_unpackhi_epi8(v3_2, v3_10);
      const __m256i v2_6 = _mm256_unpacklo_epi8(v3_3, v3_11);
      const __m256i v2_7 = _mm256_unpackhi_epi8(v3_3, v3_11);
      const __m256i v2_8 = _mm256_unpacklo_epi8(v3_4, v3_12);
      const __m256i v2_9 = _mm256_unpackhi_epi8(v3_4, v3_12);
      const __m256i v2_10 = _mm256_unpacklo_epi8(v3_5, v3_13);
      const __m256i v2_11 = _mm256_unpackhi_epi8(v3_5, v3_13);
      const __m256i v2_12 = _mm256_unpacklo_epi8(v3_6, v3_14);
      const __m256i v2_13 = _mm256_unpackhi_epi8(v3_6, v3_14);
      const __m256i v2_14 = _mm256_unpacklo_epi8(v3_7, v3_15);
      const __m256i v2_15 = _mm256_unpackhi_epi8(v3_7, v3_15);
      const __m256i v2_16 = _mm256_unpacklo_epi8(v3_16, v3_24);
      const __m256i v2_17 = _mm256_unpackhi_epi8(v3_16, v3_24);
      const __m256i v2_18 = _mm256_unpacklo_epi8(v3_17, v3_25);
      const __m256i v2_19 = _mm256_unpackhi_epi8(v3_17, v3_25);
      const __m256i v2_20 = _mm256_unpacklo_epi8(v3_18, v3_26);
      const __m256i v2_21 = _mm256_unpackhi_epi8(v3_18, v3_26);
      const __m256i v2_22 = _mm256_unpacklo_epi8(v3_19, v3_27);
      const __m256i v2_23 = _mm256_unpackhi_epi8(v3_19, v3_27);
      const __m256i v2_24 = _mm256_unpacklo_epi8(v3_20, v3_28);
      const __m256i v2_25 = _mm256_unpackhi_epi8(v3_20, v3_28);
      const __m256i v2_26 = _mm256_unpacklo_epi8(v3_21, v3_29);
      const __m256i v2_27 = _mm256_unpackhi_epi8(v3_21, v3_29);
      const __m256i v2_28 = _mm256_unpacklo_epi8(v3_22, v3_30);
      const __m256i v2_29 = _mm256_unpackhi_epi8(v3_22, v3_30);
      const __m256i v2_30 = _mm256_unpacklo_epi8(v3_23, v3_31);
      const __m256i v2_31 = _mm256_unpackhi_epi8(v3_23, v3_31);
      const __m256i v1_0 = _mm256_unpacklo_epi8(v2_0, v2_8);
      const __m256i v1_1 = _mm256_unpackhi_epi8(v2_0, v2_8);
      const __m256i v1_2 = _mm256_unpacklo_epi8(v2_1, v2_9);
      const __m256i v1_3 = _mm256_unpackhi_epi8(v2_1, v2_9);
      const __m256i v1_4 = _mm256_unpacklo_epi8(v2_2, v2_10);
      const __m256i v1_5 = _mm256_unpackhi_epi8(v2_2, v2_10);
      const __m256i v1_6 = _mm256_unpacklo_epi8(v2_3, v2_11);
      const __m256i v1_7 = _mm256_unpackhi_epi8(v2_3, v2_11);
      const __m256i v1_8 = _mm256_unpacklo_epi8(v2_4, v2_12);
      const __m256i v1_9 = _mm256_unpackhi_epi8(v2_4, v2_12);
      const __m256i v1_10 = _mm256_unpacklo_epi8(v2_5, v2_13);
      const __m256i v1_11 = _mm256_unpackhi_epi8(v2_5, v2_13);
      const __m256i v1_12 = _mm256_unpacklo_epi8(v2_6, v2_14);
      const __m256i v1_13 = _mm256_unpackhi_epi8(v2_6, v2_14);
      const __m256i v1_14 = _mm256_unpacklo_epi8(v2_7, v2_15);
      const __m256i v1_15 = _mm256_unpackhi_epi8(v2_7, v2_15);
      const __m256i v1_16 = _mm256_unpacklo_epi8(v2_16, v2_24);
      const __m256i v1_17 = _mm256_unpackhi_epi8(v2_16, v2_24);
      const __m256i v1_18 = _mm256_unpacklo_epi8(v2_17, v2_25);
      const __m256i v1_19 = _mm256_unpackhi_epi8(v2_17, v2_25);
      const __m256i v1_20 = _mm256_unpacklo_epi8(v2_18, v2_26);
      const __m256i v1_21 = _mm256_unpackhi_epi8(v2_18, v2_26);
      const __m256i v1_22 = _mm256_unpacklo_epi8(v2_19, v2_27);
      const __m256i v1_23 = _mm256_unpackhi_epi8(v2_19, v2_27);
      const __m256i v1_24 = _mm256_unpacklo_epi8(v2_20, v2_28);
      const __m256i v1_25 = _mm256_unpackhi_epi8(v2_20, v2_28);
      const __m256i v1_26 = _mm256_unpacklo_epi8(v2_21, v2_29);
      const __m256i v1_27 = _mm256_unpackhi_epi8(v2_21, v2_29);
      const __m256i v1_28 = _mm256_unpacklo_epi8(v2_22, v2_30);
      const __m256i v1_29 = _mm256_unpackhi_epi8(v2_22, v2_30);
      const __m256i v1_30 = _mm256_unpacklo_epi8(v2_23, v2_31);
      const __m256i v1_31 = _mm256_unpackhi_epi8(v2_23, v2_31);

      const __m256i v0_0 = _mm256_insertf128_si256(v1_0, _mm256_castsi256_si128(v1_16), 1);
      const __m256i v0_16 = _mm256_permute2f128_si256(v1_0, v1_16, 0x31);
      const __m256i v0_1 = _mm256_insertf128_si256(v1_1, _mm256_castsi256_si128(v1_17), 1);
      const __m256i v0_17 = _mm256_permute2f128_si256(v1_1, v1_17, 0x31);
      const __m256i v0_2 = _mm256_insertf128_si256(v1_2, _mm256_castsi256_si128(v1_18), 1);
      const __m256i v0_18 = _mm256_permute2f128_si256(v1_2, v1_18, 0x31);
      const __m256i v0_3 = _mm256_insertf128_si256(v1_3, _mm256_castsi256_si128(v1_19), 1);
      const __m256i v0_19 = _mm256_permute2f128_si256(v1_3, v1_19, 0x31);
      const __m256i v0_4 = _mm256_insertf128_si256(v1_4, _mm256_castsi256_si128(v1_20), 1);
      const __m256i v0_20 = _mm256_permute2f128_si256(v1_4, v1_20, 0x31);
      const __m256i v0_5 = _mm256_insertf128_si256(v1_5, _mm256_castsi256_si128(v1_21), 1);
      const __m256i v0_21 = _mm256_permute2f128_si256(v1_5, v1_21, 0x31);
      const __m256i v0_6 = _mm256_insertf128_si256(v1_6, _mm256_castsi256_si128(v1_22), 1);
      const __m256i v0_22 = _mm256_permute2f128_si256(v1_6, v1_22, 0x31);
      const __m256i v0_7 = _mm256_insertf128_si256(v1_7, _mm256_castsi256_si128(v1_23), 1);
      const __m256i v0_23 = _mm256_permute2f128_si256(v1_7, v1_23, 0x31);
      const __m256i v0_8 = _mm256_insertf128_si256(v1_8, _mm256_castsi256_si128(v1_24), 1);
      const __m256i v0_24 = _mm256_permute2f128_si256(v1_8, v1_24, 0x31);
      const __m256i v0_9 = _mm256_insertf128_si256(v1_9, _mm256_castsi256_si128(v1_25), 1);
      const __m256i v0_25 = _mm256_permute2f128_si256(v1_9, v1_25, 0x31);
      const __m256i v0_10 = _mm256_insertf128_si256(v1_10, _mm256_castsi256_si128(v1_26), 1);
      const __m256i v0_26 = _mm256_permute2f128_si256(v1_10, v1_26, 0x31);
      const __m256i v0_11 = _mm256_insertf128_si256(v1_11, _mm256_castsi256_si128(v1_27), 1);
      const __m256i v0_27 = _mm256_permute2f128_si256(v1_11, v1_27, 0x31);
      const __m256i v0_12 = _mm256_insertf128_si256(v1_12, _mm256_castsi256_si128(v1_28), 1);
      const __m256i v0_28 = _mm256_permute2f128_si256(v1_12, v1_28, 0x31);
      const __m256i v0_13 = _mm256_insertf128_si256(v1_13, _mm256_castsi256_si128(v1_29), 1);
      const __m256i v0_29 = _mm256_permute2f128_si256(v1_13, v1_29, 0x31);
      const __m256i v0_14 = _mm256_insertf128_si256(v1_14, _mm256_castsi256_si128(v1_30), 1);
      const __m256i v0_30 = _mm256_permute2f128_si256(v1_14, v1_30, 0x31);
      const __m256i v0_15 = _mm256_insertf128_si256(v1_15, _mm256_castsi256_si128(v1_31), 1);
      const __m256i v0_31 = _mm256_permute2f128_si256(v1_15, v1_31, 0x31);

      o = (uint8_t*) ((uintptr_t) o + oN_offset);
      _mm256_storeu_si256((__m256i*) o, v0_31);
      uint8_t *oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 31) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_30);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 31) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_29);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 29) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_28);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 29) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_27);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 27) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_26);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 27) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_25);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 25) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_24);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 25) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_23);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 23) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_22);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 23) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_21);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 21) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_20);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 21) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_19);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 19) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_18);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 19) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_17);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 17) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_16);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 17) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_15);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 15) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_14);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 15) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_13);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 13) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_12);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 13) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_11);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 11) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_10);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 11) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_9);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 9) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_8);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 9) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_7);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 7) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_6);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 7) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_5);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 5) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_4);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 5) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_3);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 3) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_2);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 3) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_1);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 1) {
        o = oN;
      }
      _mm256_storeu_si256((__m256i*) o, v0_0);
    }
    o = (uint8_t*) ((uintptr_t) o + tile_hbytes);
    if (bh != 0) {
      const __m256i v5_0 = _mm256_maskload_epi32((const int*) i0, vmask);
      const uint8_t *i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const __m256i v5_1 = _mm256_maskload_epi32((const int*) i1, vmask);
      const uint8_t *i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i1;
      }
      const __m256i v5_2 = _mm256_maskload_epi32((const int*) i2, vmask);
      const uint8_t *i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
      if XNN_UNPREDICTABLE(bh < 4) {
        i3 = i2;
      }
      const __m256i v5_3 = _mm256_maskload_epi32((const int*) i3, vmask);
      const uint8_t *i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 4) {
        i4 = i3;
      }
      const __m256i v5_4 = _mm256_maskload_epi32((const int*) i4, vmask);
      const uint8_t *i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
      if XNN_UNPREDICTABLE(bh < 6) {
        i5 = i4;
      }
      const __m256i v5_5 = _mm256_maskload_epi32((const int*) i5, vmask);
      const uint8_t *i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 6) {
        i6 = i5;
      }
      const __m256i v5_6 = _mm256_maskload_epi32((const int*) i6, vmask);
      const uint8_t *i7 = (const uint8_t*) ((uintptr_t) i6 + input_stride);
      if XNN_UNPREDICTABLE(bh < 8) {
        i7 = i6;
      }
      const __m256i v5_7 = _mm256_maskload_epi32((const int*) i7, vmask);
      const uint8_t *i8 = (const uint8_t*) ((uintptr_t) i7 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 8) {
        i8 = i7;
      }
      const __m256i v5_8 = _mm256_maskload_epi32((const int*) i8, vmask);
      const uint8_t *i9 = (const uint8_t*) ((uintptr_t) i8 + input_stride);
      if XNN_UNPREDICTABLE(bh < 10) {
        i9 = i8;
      }
      const __m256i v5_9 = _mm256_maskload_epi32((const int*) i9, vmask);
      const uint8_t *i10 = (const uint8_t*) ((uintptr_t) i9 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 10) {
        i10 = i9;
      }
      const __m256i v5_10 = _mm256_maskload_epi32((const int*) i10, vmask);
      const uint8_t *i11 = (const uint8_t*) ((uintptr_t) i10 + input_stride);
      if XNN_UNPREDICTABLE(bh < 12) {
        i11 = i10;
      }
      const __m256i v5_11 = _mm256_maskload_epi32((const int*) i11, vmask);
      const uint8_t *i12 = (const uint8_t*) ((uintptr_t) i11 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 12) {
        i12 = i11;
      }
      const __m256i v5_12 = _mm256_maskload_epi32((const int*) i12, vmask);
      const uint8_t *i13 = (const uint8_t*) ((uintptr_t) i12 + input_stride);
      if XNN_UNPREDICTABLE(bh < 14) {
        i13 = i12;
      }
      const __m256i v5_13 = _mm256_maskload_epi32((const int*) i13, vmask);
      const uint8_t *i14 = (const uint8_t*) ((uintptr_t) i13 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 14) {
        i14 = i13;
      }
      const __m256i v5_14 = _mm256_maskload_epi32((const int*) i14, vmask);
      const uint8_t *i15 = (const uint8_t*) ((uintptr_t) i14 + input_stride);
      if XNN_UNPREDICTABLE(bh < 16) {
        i15 = i14;
      }
      const __m256i v5_15 = _mm256_maskload_epi32((const int*) i15, vmask);
      const uint8_t *i16 = (const uint8_t*) ((uintptr_t) i15 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 16) {
        i16 = i15;
      }
      const __m256i v5_16 = _mm256_maskload_epi32((const int*) i16, vmask);
      const uint8_t *i17 = (const uint8_t*) ((uintptr_t) i16 + input_stride);
      if XNN_UNPREDICTABLE(bh < 18) {
        i17 = i16;
      }
      const __m256i v5_17 = _mm256_maskload_epi32((const int*) i17, vmask);
      const uint8_t *i18 = (const uint8_t*) ((uintptr_t) i17 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 18) {
        i18 = i17;
      }
      const __m256i v5_18 = _mm256_maskload_epi32((const int*) i18, vmask);
      const uint8_t *i19 = (const uint8_t*) ((uintptr_t) i18 + input_stride);
      if XNN_UNPREDICTABLE(bh < 20) {
        i19 = i18;
      }
      const __m256i v5_19 = _mm256_maskload_epi32((const int*) i19, vmask);
      const uint8_t *i20 = (const uint8_t*) ((uintptr_t) i19 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 20) {
        i20 = i19;
      }
      const __m256i v5_20 = _mm256_maskload_epi32((const int*) i20, vmask);
      const uint8_t *i21 = (const uint8_t*) ((uintptr_t) i20 + input_stride);
      if XNN_UNPREDICTABLE(bh < 22) {
        i21 = i20;
      }
      const __m256i v5_21 = _mm256_maskload_epi32((const int*) i21, vmask);
      const uint8_t *i22 = (const uint8_t*) ((uintptr_t) i21 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 22) {
        i22 = i21;
      }
      const __m256i v5_22 = _mm256_maskload_epi32((const int*) i22, vmask);
      const uint8_t *i23 = (const uint8_t*) ((uintptr_t) i22 + input_stride);
      if XNN_UNPREDICTABLE(bh < 24) {
        i23 = i22;
      }
      const __m256i v5_23 = _mm256_maskload_epi32((const int*) i23, vmask);
      const uint8_t *i24 = (const uint8_t*) ((uintptr_t) i23 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 24) {
        i24 = i23;
      }
      const __m256i v5_24 = _mm256_maskload_epi32((const int*) i24, vmask);
      const uint8_t *i25 = (const uint8_t*) ((uintptr_t) i24 + input_stride);
      if XNN_UNPREDICTABLE(bh < 26) {
        i25 = i24;
      }
      const __m256i v5_25 = _mm256_maskload_epi32((const int*) i25, vmask);
      const uint8_t *i26 = (const uint8_t*) ((uintptr_t) i25 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 26) {
        i26 = i25;
      }
      const __m256i v5_26 = _mm256_maskload_epi32((const int*) i26, vmask);
      const uint8_t *i27 = (const uint8_t*) ((uintptr_t) i26 + input_stride);
      if XNN_UNPREDICTABLE(bh < 28) {
        i27 = i26;
      }
      const __m256i v5_27 = _mm256_maskload_epi32((const int*) i27, vmask);
      const uint8_t *i28 = (const uint8_t*) ((uintptr_t) i27 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 28) {
        i28 = i27;
      }
      const __m256i v5_28 = _mm256_maskload_epi32((const int*) i28, vmask);
      const uint8_t *i29 = (const uint8_t*) ((uintptr_t) i28 + input_stride);
      if XNN_UNPREDICTABLE(bh < 30) {
        i29 = i28;
      }
      const __m256i v5_29 = _mm256_maskload_epi32((const int*) i29, vmask);
      const uint8_t *i30 = (const uint8_t*) ((uintptr_t) i29 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 30) {
        i30 = i29;
      }
      const __m256i v5_30 = _mm256_maskload_epi32((const int*) i30, vmask);
      const __m256i v5_31 = _mm256_undefined_si256();

      const __m256i v4_0 = _mm256_unpacklo_epi8(v5_0, v5_8);
      const __m256i v4_1 = _mm256_unpackhi_epi8(v5_0, v5_8);
      const __m256i v4_2 = _mm256_unpacklo_epi8(v5_1, v5_9);
      const __m256i v4_3 = _mm256_unpackhi_epi8(v5_1, v5_9);
      const __m256i v4_4 = _mm256_unpacklo_epi8(v5_2, v5_10);
      const __m256i v4_5 = _mm256_unpackhi_epi8(v5_2, v5_10);
      const __m256i v4_6 = _mm256_unpacklo_epi8(v5_3, v5_11);
      const __m256i v4_7 = _mm256_unpackhi_epi8(v5_3, v5_11);
      const __m256i v4_8 = _mm256_unpacklo_epi8(v5_4, v5_12);
      const __m256i v4_9 = _mm256_unpackhi_epi8(v5_4, v5_12);
      const __m256i v4_10 = _mm256_unpacklo_epi8(v5_5, v5_13);
      const __m256i v4_11 = _mm256_unpackhi_epi8(v5_5, v5_13);
      const __m256i v4_12 = _mm256_unpacklo_epi8(v5_6, v5_14);
      const __m256i v4_13 = _mm256_unpackhi_epi8(v5_6, v5_14);
      const __m256i v4_14 = _mm256_unpacklo_epi8(v5_7, v5_15);
      const __m256i v4_15 = _mm256_unpackhi_epi8(v5_7, v5_15);
      const __m256i v4_16 = _mm256_unpacklo_epi8(v5_16, v5_24);
      const __m256i v4_17 = _mm256_unpackhi_epi8(v5_16, v5_24);
      const __m256i v4_18 = _mm256_unpacklo_epi8(v5_17, v5_25);
      const __m256i v4_19 = _mm256_unpackhi_epi8(v5_17, v5_25);
      const __m256i v4_20 = _mm256_unpacklo_epi8(v5_18, v5_26);
      const __m256i v4_21 = _mm256_unpackhi_epi8(v5_18, v5_26);
      const __m256i v4_22 = _mm256_unpacklo_epi8(v5_19, v5_27);
      const __m256i v4_23 = _mm256_unpackhi_epi8(v5_19, v5_27);
      const __m256i v4_24 = _mm256_unpacklo_epi8(v5_20, v5_28);
      const __m256i v4_25 = _mm256_unpackhi_epi8(v5_20, v5_28);
      const __m256i v4_26 = _mm256_unpacklo_epi8(v5_21, v5_29);
      const __m256i v4_27 = _mm256_unpackhi_epi8(v5_21, v5_29);
      const __m256i v4_28 = _mm256_unpacklo_epi8(v5_22, v5_30);
      const __m256i v4_29 = _mm256_unpackhi_epi8(v5_22, v5_30);
      const __m256i v4_30 = _mm256_unpacklo_epi8(v5_23, v5_31);
      const __m256i v4_31 = _mm256_unpackhi_epi8(v5_23, v5_31);
      const __m256i v3_0 = _mm256_unpacklo_epi8(v4_0, v4_8);
      const __m256i v3_1 = _mm256_unpackhi_epi8(v4_0, v4_8);
      const __m256i v3_2 = _mm256_unpacklo_epi8(v4_1, v4_9);
      const __m256i v3_3 = _mm256_unpackhi_epi8(v4_1, v4_9);
      const __m256i v3_4 = _mm256_unpacklo_epi8(v4_2, v4_10);
      const __m256i v3_5 = _mm256_unpackhi_epi8(v4_2, v4_10);
      const __m256i v3_6 = _mm256_unpacklo_epi8(v4_3, v4_11);
      const __m256i v3_7 = _mm256_unpackhi_epi8(v4_3, v4_11);
      const __m256i v3_8 = _mm256_unpacklo_epi8(v4_4, v4_12);
      const __m256i v3_9 = _mm256_unpackhi_epi8(v4_4, v4_12);
      const __m256i v3_10 = _mm256_unpacklo_epi8(v4_5, v4_13);
      const __m256i v3_11 = _mm256_unpackhi_epi8(v4_5, v4_13);
      const __m256i v3_12 = _mm256_unpacklo_epi8(v4_6, v4_14);
      const __m256i v3_13 = _mm256_unpackhi_epi8(v4_6, v4_14);
      const __m256i v3_14 = _mm256_unpacklo_epi8(v4_7, v4_15);
      const __m256i v3_15 = _mm256_unpackhi_epi8(v4_7, v4_15);
      const __m256i v3_16 = _mm256_unpacklo_epi8(v4_16, v4_24);
      const __m256i v3_17 = _mm256_unpackhi_epi8(v4_16, v4_24);
      const __m256i v3_18 = _mm256_unpacklo_epi8(v4_17, v4_25);
      const __m256i v3_19 = _mm256_unpackhi_epi8(v4_17, v4_25);
      const __m256i v3_20 = _mm256_unpacklo_epi8(v4_18, v4_26);
      const __m256i v3_21 = _mm256_unpackhi_epi8(v4_18, v4_26);
      const __m256i v3_22 = _mm256_unpacklo_epi8(v4_19, v4_27);
      const __m256i v3_23 = _mm256_unpackhi_epi8(v4_19, v4_27);
      const __m256i v3_24 = _mm256_unpacklo_epi8(v4_20, v4_28);
      const __m256i v3_25 = _mm256_unpackhi_epi8(v4_20, v4_28);
      const __m256i v3_26 = _mm256_unpacklo_epi8(v4_21, v4_29);
      const __m256i v3_27 = _mm256_unpackhi_epi8(v4_21, v4_29);
      const __m256i v3_28 = _mm256_unpacklo_epi8(v4_22, v4_30);
      const __m256i v3_29 = _mm256_unpackhi_epi8(v4_22, v4_30);
      const __m256i v3_30 = _mm256_unpacklo_epi8(v4_23, v4_31);
      const __m256i v3_31 = _mm256_unpackhi_epi8(v4_23, v4_31);
      const __m256i v2_0 = _mm256_unpacklo_epi8(v3_0, v3_8);
      const __m256i v2_1 = _mm256_unpackhi_epi8(v3_0, v3_8);
      const __m256i v2_2 = _mm256_unpacklo_epi8(v3_1, v3_9);
      const __m256i v2_3 = _mm256_unpackhi_epi8(v3_1, v3_9);
      const __m256i v2_4 = _mm256_unpacklo_epi8(v3_2, v3_10);
      const __m256i v2_5 = _mm256_unpackhi_epi8(v3_2, v3_10);
      const __m256i v2_6 = _mm256_unpacklo_epi8(v3_3, v3_11);
      const __m256i v2_7 = _mm256_unpackhi_epi8(v3_3, v3_11);
      const __m256i v2_8 = _mm256_unpacklo_epi8(v3_4, v3_12);
      const __m256i v2_9 = _mm256_unpackhi_epi8(v3_4, v3_12);
      const __m256i v2_10 = _mm256_unpacklo_epi8(v3_5, v3_13);
      const __m256i v2_11 = _mm256_unpackhi_epi8(v3_5, v3_13);
      const __m256i v2_12 = _mm256_unpacklo_epi8(v3_6, v3_14);
      const __m256i v2_13 = _mm256_unpackhi_epi8(v3_6, v3_14);
      const __m256i v2_14 = _mm256_unpacklo_epi8(v3_7, v3_15);
      const __m256i v2_15 = _mm256_unpackhi_epi8(v3_7, v3_15);
      const __m256i v2_16 = _mm256_unpacklo_epi8(v3_16, v3_24);
      const __m256i v2_17 = _mm256_unpackhi_epi8(v3_16, v3_24);
      const __m256i v2_18 = _mm256_unpacklo_epi8(v3_17, v3_25);
      const __m256i v2_19 = _mm256_unpackhi_epi8(v3_17, v3_25);
      const __m256i v2_20 = _mm256_unpacklo_epi8(v3_18, v3_26);
      const __m256i v2_21 = _mm256_unpackhi_epi8(v3_18, v3_26);
      const __m256i v2_22 = _mm256_unpacklo_epi8(v3_19, v3_27);
      const __m256i v2_23 = _mm256_unpackhi_epi8(v3_19, v3_27);
      const __m256i v2_24 = _mm256_unpacklo_epi8(v3_20, v3_28);
      const __m256i v2_25 = _mm256_unpackhi_epi8(v3_20, v3_28);
      const __m256i v2_26 = _mm256_unpacklo_epi8(v3_21, v3_29);
      const __m256i v2_27 = _mm256_unpackhi_epi8(v3_21, v3_29);
      const __m256i v2_28 = _mm256_unpacklo_epi8(v3_22, v3_30);
      const __m256i v2_29 = _mm256_unpackhi_epi8(v3_22, v3_30);
      const __m256i v2_30 = _mm256_unpacklo_epi8(v3_23, v3_31);
      const __m256i v2_31 = _mm256_unpackhi_epi8(v3_23, v3_31);
      const __m256i v1_0 = _mm256_unpacklo_epi8(v2_0, v2_8);
      const __m256i v1_1 = _mm256_unpackhi_epi8(v2_0, v2_8);
      const __m256i v1_2 = _mm256_unpacklo_epi8(v2_1, v2_9);
      const __m256i v1_3 = _mm256_unpackhi_epi8(v2_1, v2_9);
      const __m256i v1_4 = _mm256_unpacklo_epi8(v2_2, v2_10);
      const __m256i v1_5 = _mm256_unpackhi_epi8(v2_2, v2_10);
      const __m256i v1_6 = _mm256_unpacklo_epi8(v2_3, v2_11);
      const __m256i v1_7 = _mm256_unpackhi_epi8(v2_3, v2_11);
      const __m256i v1_8 = _mm256_unpacklo_epi8(v2_4, v2_12);
      const __m256i v1_9 = _mm256_unpackhi_epi8(v2_4, v2_12);
      const __m256i v1_10 = _mm256_unpacklo_epi8(v2_5, v2_13);
      const __m256i v1_11 = _mm256_unpackhi_epi8(v2_5, v2_13);
      const __m256i v1_12 = _mm256_unpacklo_epi8(v2_6, v2_14);
      const __m256i v1_13 = _mm256_unpackhi_epi8(v2_6, v2_14);
      const __m256i v1_14 = _mm256_unpacklo_epi8(v2_7, v2_15);
      const __m256i v1_15 = _mm256_unpackhi_epi8(v2_7, v2_15);
      const __m256i v1_16 = _mm256_unpacklo_epi8(v2_16, v2_24);
      const __m256i v1_17 = _mm256_unpackhi_epi8(v2_16, v2_24);
      const __m256i v1_18 = _mm256_unpacklo_epi8(v2_17, v2_25);
      const __m256i v1_19 = _mm256_unpackhi_epi8(v2_17, v2_25);
      const __m256i v1_20 = _mm256_unpacklo_epi8(v2_18, v2_26);
      const __m256i v1_21 = _mm256_unpackhi_epi8(v2_18, v2_26);
      const __m256i v1_22 = _mm256_unpacklo_epi8(v2_19, v2_27);
      const __m256i v1_23 = _mm256_unpackhi_epi8(v2_19, v2_27);
      const __m256i v1_24 = _mm256_unpacklo_epi8(v2_20, v2_28);
      const __m256i v1_25 = _mm256_unpackhi_epi8(v2_20, v2_28);
      const __m256i v1_26 = _mm256_unpacklo_epi8(v2_21, v2_29);
      const __m256i v1_27 = _mm256_unpackhi_epi8(v2_21, v2_29);
      const __m256i v1_28 = _mm256_unpacklo_epi8(v2_22, v2_30);
      const __m256i v1_29 = _mm256_unpackhi_epi8(v2_22, v2_30);
      const __m256i v1_30 = _mm256_unpacklo_epi8(v2_23, v2_31);
      const __m256i v1_31 = _mm256_unpackhi_epi8(v2_23, v2_31);

      __m128i v0_0_lo = _mm256_castsi256_si128(v1_0);
      __m128i v0_1_lo = _mm256_castsi256_si128(v1_1);
      __m128i v0_2_lo = _mm256_castsi256_si128(v1_2);
      __m128i v0_3_lo = _mm256_castsi256_si128(v1_3);
      __m128i v0_4_lo = _mm256_castsi256_si128(v1_4);
      __m128i v0_5_lo = _mm256_castsi256_si128(v1_5);
      __m128i v0_6_lo = _mm256_castsi256_si128(v1_6);
      __m128i v0_7_lo = _mm256_castsi256_si128(v1_7);
      __m128i v0_8_lo = _mm256_castsi256_si128(v1_8);
      __m128i v0_9_lo = _mm256_castsi256_si128(v1_9);
      __m128i v0_10_lo = _mm256_castsi256_si128(v1_10);
      __m128i v0_11_lo = _mm256_castsi256_si128(v1_11);
      __m128i v0_12_lo = _mm256_castsi256_si128(v1_12);
      __m128i v0_13_lo = _mm256_castsi256_si128(v1_13);
      __m128i v0_14_lo = _mm256_castsi256_si128(v1_14);
      __m128i v0_15_lo = _mm256_castsi256_si128(v1_15);
      __m128i v0_16_lo = _mm256_extractf128_si256(v1_0, 0x1);
      __m128i v0_17_lo = _mm256_extractf128_si256(v1_1, 0x1);
      __m128i v0_18_lo = _mm256_extractf128_si256(v1_2, 0x1);
      __m128i v0_19_lo = _mm256_extractf128_si256(v1_3, 0x1);
      __m128i v0_20_lo = _mm256_extractf128_si256(v1_4, 0x1);
      __m128i v0_21_lo = _mm256_extractf128_si256(v1_5, 0x1);
      __m128i v0_22_lo = _mm256_extractf128_si256(v1_6, 0x1);
      __m128i v0_23_lo = _mm256_extractf128_si256(v1_7, 0x1);
      __m128i v0_24_lo = _mm256_extractf128_si256(v1_8, 0x1);
      __m128i v0_25_lo = _mm256_extractf128_si256(v1_9, 0x1);
      __m128i v0_26_lo = _mm256_extractf128_si256(v1_10, 0x1);
      __m128i v0_27_lo = _mm256_extractf128_si256(v1_11, 0x1);
      __m128i v0_28_lo = _mm256_extractf128_si256(v1_12, 0x1);
      __m128i v0_29_lo = _mm256_extractf128_si256(v1_13, 0x1);
      __m128i v0_30_lo = _mm256_extractf128_si256(v1_14, 0x1);
      __m128i v0_31_lo = _mm256_extractf128_si256(v1_15, 0x1);

      if (bh & 16) {
        o = (uint8_t*) ((uintptr_t) o + oN_stride);
        _mm_storeu_si128((__m128i*) o, v0_31_lo);
        v0_31_lo = _mm256_extractf128_si256(v1_31, 0x1);
        uint8_t *oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 31) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_30_lo);
        v0_30_lo = _mm256_extractf128_si256(v1_30, 0x1);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 31) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_29_lo);
        v0_29_lo = _mm256_extractf128_si256(v1_29, 0x1);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 29) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_28_lo);
        v0_28_lo = _mm256_extractf128_si256(v1_28, 0x1);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 29) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_27_lo);
        v0_27_lo = _mm256_extractf128_si256(v1_27, 0x1);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 27) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_26_lo);
        v0_26_lo = _mm256_extractf128_si256(v1_26, 0x1);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 27) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_25_lo);
        v0_25_lo = _mm256_extractf128_si256(v1_25, 0x1);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 25) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_24_lo);
        v0_24_lo = _mm256_extractf128_si256(v1_24, 0x1);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 25) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_23_lo);
        v0_23_lo = _mm256_extractf128_si256(v1_23, 0x1);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 23) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_22_lo);
        v0_22_lo = _mm256_extractf128_si256(v1_22, 0x1);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 23) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_21_lo);
        v0_21_lo = _mm256_extractf128_si256(v1_21, 0x1);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 21) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_20_lo);
        v0_20_lo = _mm256_extractf128_si256(v1_20, 0x1);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 21) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_19_lo);
        v0_19_lo = _mm256_extractf128_si256(v1_19, 0x1);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 19) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_18_lo);
        v0_18_lo = _mm256_extractf128_si256(v1_18, 0x1);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 19) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_17_lo);
        v0_17_lo = _mm256_extractf128_si256(v1_17, 0x1);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 17) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_16_lo);
        v0_16_lo = _mm256_extractf128_si256(v1_16, 0x1);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 17) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_15_lo);
        v0_15_lo = _mm256_castsi256_si128(v1_31);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_14_lo);
        v0_14_lo = _mm256_castsi256_si128(v1_30);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_13_lo);
        v0_13_lo = _mm256_castsi256_si128(v1_29);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_12_lo);
        v0_12_lo = _mm256_castsi256_si128(v1_28);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_11_lo);
        v0_11_lo = _mm256_castsi256_si128(v1_27);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_10_lo);
        v0_10_lo = _mm256_castsi256_si128(v1_26);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_9_lo);
        v0_9_lo = _mm256_castsi256_si128(v1_25);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_8_lo);
        v0_8_lo = _mm256_castsi256_si128(v1_24);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_7_lo);
        v0_7_lo = _mm256_castsi256_si128(v1_23);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_6_lo);
        v0_6_lo = _mm256_castsi256_si128(v1_22);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_5_lo);
        v0_5_lo = _mm256_castsi256_si128(v1_21);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_4_lo);
        v0_4_lo = _mm256_castsi256_si128(v1_20);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_3_lo);
        v0_3_lo = _mm256_castsi256_si128(v1_19);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_2_lo);
        v0_2_lo = _mm256_castsi256_si128(v1_18);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_1_lo);
        v0_1_lo = _mm256_castsi256_si128(v1_17);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        _mm_storeu_si128((__m128i*) o, v0_0_lo);
        v0_0_lo = _mm256_castsi256_si128(v1_16);
        o += 16;
      }

      if (bh & 8) {
        o = (uint8_t*) ((uintptr_t) o + oN_stride);
        _mm_storel_epi64((__m128i*) o, v0_31_lo);
        v0_31_lo = _mm_unpackhi_epi64(v0_31_lo, v0_31_lo);
        uint8_t *oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 31) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_30_lo);
        v0_30_lo = _mm_unpackhi_epi64(v0_30_lo, v0_30_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 31) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_29_lo);
        v0_29_lo = _mm_unpackhi_epi64(v0_29_lo, v0_29_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 29) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_28_lo);
        v0_28_lo = _mm_unpackhi_epi64(v0_28_lo, v0_28_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 29) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_27_lo);
        v0_27_lo = _mm_unpackhi_epi64(v0_27_lo, v0_27_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 27) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_26_lo);
        v0_26_lo = _mm_unpackhi_epi64(v0_26_lo, v0_26_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 27) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_25_lo);
        v0_25_lo = _mm_unpackhi_epi64(v0_25_lo, v0_25_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 25) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_24_lo);
        v0_24_lo = _mm_unpackhi_epi64(v0_24_lo, v0_24_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 25) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_23_lo);
        v0_23_lo = _mm_unpackhi_epi64(v0_23_lo, v0_23_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 23) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_22_lo);
        v0_22_lo = _mm_unpackhi_epi64(v0_22_lo, v0_22_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 23) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_21_lo);
        v0_21_lo = _mm_unpackhi_epi64(v0_21_lo, v0_21_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 21) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_20_lo);
        v0_20_lo = _mm_unpackhi_epi64(v0_20_lo, v0_20_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 21) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_19_lo);
        v0_19_lo = _mm_unpackhi_epi64(v0_19_lo, v0_19_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 19) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_18_lo);
        v0_18_lo = _mm_unpackhi_epi64(v0_18_lo, v0_18_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 19) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_17_lo);
        v0_17_lo = _mm_unpackhi_epi64(v0_17_lo, v0_17_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 17) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_16_lo);
        v0_16_lo = _mm_unpackhi_epi64(v0_16_lo, v0_16_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 17) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_15_lo);
        v0_15_lo = _mm_unpackhi_epi64(v0_15_lo, v0_15_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_14_lo);
        v0_14_lo = _mm_unpackhi_epi64(v0_14_lo, v0_14_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_13_lo);
        v0_13_lo = _mm_unpackhi_epi64(v0_13_lo, v0_13_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_12_lo);
        v0_12_lo = _mm_unpackhi_epi64(v0_12_lo, v0_12_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_11_lo);
        v0_11_lo = _mm_unpackhi_epi64(v0_11_lo, v0_11_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_10_lo);
        v0_10_lo = _mm_unpackhi_epi64(v0_10_lo, v0_10_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_9_lo);
        v0_9_lo = _mm_unpackhi_epi64(v0_9_lo, v0_9_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_8_lo);
        v0_8_lo = _mm_unpackhi_epi64(v0_8_lo, v0_8_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_7_lo);
        v0_7_lo = _mm_unpackhi_epi64(v0_7_lo, v0_7_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_6_lo);
        v0_6_lo = _mm_unpackhi_epi64(v0_6_lo, v0_6_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_5_lo);
        v0_5_lo = _mm_unpackhi_epi64(v0_5_lo, v0_5_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_4_lo);
        v0_4_lo = _mm_unpackhi_epi64(v0_4_lo, v0_4_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_3_lo);
        v0_3_lo = _mm_unpackhi_epi64(v0_3_lo, v0_3_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_2_lo);
        v0_2_lo = _mm_unpackhi_epi64(v0_2_lo, v0_2_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_1_lo);
        v0_1_lo = _mm_unpackhi_epi64(v0_1_lo, v0_1_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        _mm_storel_epi64((__m128i*) o, v0_0_lo);
        v0_0_lo = _mm_unpackhi_epi64(v0_0_lo, v0_0_lo);
        o += 8;
      }
      if (bh & 4) {
        o = (uint8_t*) ((uintptr_t) o + oN_stride);
        _mm_storeu_si32(o, v0_31_lo);
        v0_31_lo = _mm_srli_epi64(v0_31_lo, 32);
        uint8_t* oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 31) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_30_lo);
        v0_30_lo = _mm_srli_epi64(v0_30_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 31) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_29_lo);
        v0_29_lo = _mm_srli_epi64(v0_29_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 29) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_28_lo);
        v0_28_lo = _mm_srli_epi64(v0_28_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 29) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_27_lo);
        v0_27_lo = _mm_srli_epi64(v0_27_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 27) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_26_lo);
        v0_26_lo = _mm_srli_epi64(v0_26_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 27) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_25_lo);
        v0_25_lo = _mm_srli_epi64(v0_25_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 25) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_24_lo);
        v0_24_lo = _mm_srli_epi64(v0_24_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 25) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_23_lo);
        v0_23_lo = _mm_srli_epi64(v0_23_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 23) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_22_lo);
        v0_22_lo = _mm_srli_epi64(v0_22_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 23) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_21_lo);
        v0_21_lo = _mm_srli_epi64(v0_21_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 21) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_20_lo);
        v0_20_lo = _mm_srli_epi64(v0_20_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 21) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_19_lo);
        v0_19_lo = _mm_srli_epi64(v0_19_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 19) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_18_lo);
        v0_18_lo = _mm_srli_epi64(v0_18_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 19) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_17_lo);
        v0_17_lo = _mm_srli_epi64(v0_17_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 17) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_16_lo);
        v0_16_lo = _mm_srli_epi64(v0_16_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 17) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_15_lo);
        v0_15_lo = _mm_srli_epi64(v0_15_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_14_lo);
        v0_14_lo = _mm_srli_epi64(v0_14_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_13_lo);
        v0_13_lo = _mm_srli_epi64(v0_13_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_12_lo);
        v0_12_lo = _mm_srli_epi64(v0_12_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_11_lo);
        v0_11_lo = _mm_srli_epi64(v0_11_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_10_lo);
        v0_10_lo = _mm_srli_epi64(v0_10_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_9_lo);
        v0_9_lo = _mm_srli_epi64(v0_9_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_8_lo);
        v0_8_lo = _mm_srli_epi64(v0_8_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_7_lo);
        v0_7_lo = _mm_srli_epi64(v0_7_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_6_lo);
        v0_6_lo = _mm_srli_epi64(v0_6_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_5_lo);
        v0_5_lo = _mm_srli_epi64(v0_5_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_4_lo);
        v0_4_lo = _mm_srli_epi64(v0_4_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_3_lo);
        v0_3_lo = _mm_srli_epi64(v0_3_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_2_lo);
        v0_2_lo = _mm_srli_epi64(v0_2_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_1_lo);
        v0_1_lo = _mm_srli_epi64(v0_1_lo, 32);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        _mm_storeu_si32(o, v0_0_lo);
        v0_0_lo = _mm_srli_epi64(v0_0_lo, 32);
        o += 4;
      }
      if (bh & 2) {
        o = (uint8_t*) ((uintptr_t) o + oN_stride);
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_31_lo));
        v0_31_lo = _mm_srli_epi32(v0_31_lo, 16);
        uint8_t* oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 31) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_30_lo));
        v0_30_lo = _mm_srli_epi32(v0_30_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 31) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_29_lo));
        v0_29_lo = _mm_srli_epi32(v0_29_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 29) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_28_lo));
        v0_28_lo = _mm_srli_epi32(v0_28_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 29) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_27_lo));
        v0_27_lo = _mm_srli_epi32(v0_27_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 27) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_26_lo));
        v0_26_lo = _mm_srli_epi32(v0_26_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 27) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_25_lo));
        v0_25_lo = _mm_srli_epi32(v0_25_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 25) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_24_lo));
        v0_24_lo = _mm_srli_epi32(v0_24_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 25) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_23_lo));
        v0_23_lo = _mm_srli_epi32(v0_23_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 23) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_22_lo));
        v0_22_lo = _mm_srli_epi32(v0_22_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 23) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_21_lo));
        v0_21_lo = _mm_srli_epi32(v0_21_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 21) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_20_lo));
        v0_20_lo = _mm_srli_epi32(v0_20_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 21) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_19_lo));
        v0_19_lo = _mm_srli_epi32(v0_19_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 19) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_18_lo));
        v0_18_lo = _mm_srli_epi32(v0_18_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 19) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_17_lo));
        v0_17_lo = _mm_srli_epi32(v0_17_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 17) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_16_lo));
        v0_16_lo = _mm_srli_epi32(v0_16_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 17) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_15_lo));
        v0_15_lo = _mm_srli_epi32(v0_15_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_14_lo));
        v0_14_lo = _mm_srli_epi32(v0_14_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_13_lo));
        v0_13_lo = _mm_srli_epi32(v0_13_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_12_lo));
        v0_12_lo = _mm_srli_epi32(v0_12_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_11_lo));
        v0_11_lo = _mm_srli_epi32(v0_11_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_10_lo));
        v0_10_lo = _mm_srli_epi32(v0_10_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_9_lo));
        v0_9_lo = _mm_srli_epi32(v0_9_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_8_lo));
        v0_8_lo = _mm_srli_epi32(v0_8_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_7_lo));
        v0_7_lo = _mm_srli_epi32(v0_7_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_6_lo));
        v0_6_lo = _mm_srli_epi32(v0_6_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_5_lo));
        v0_5_lo = _mm_srli_epi32(v0_5_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_4_lo));
        v0_4_lo = _mm_srli_epi32(v0_4_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_3_lo));
        v0_3_lo = _mm_srli_epi32(v0_3_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_2_lo));
        v0_2_lo = _mm_srli_epi32(v0_2_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_1_lo));
        v0_1_lo = _mm_srli_epi32(v0_1_lo, 16);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        unaligned_store_u16(o, (uint16_t) _mm_cvtsi128_si32(v0_0_lo));
        v0_0_lo = _mm_srli_epi32(v0_0_lo, 16);
        o += 2;
      }
      if (bh & 1) {
        o = (uint8_t*) ((uintptr_t) o + oN_stride);
        *o = (uint8_t) _mm_cvtsi128_si32(v0_31_lo);
        uint8_t* oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 31) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_30_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 31) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_29_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 29) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_28_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 29) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_27_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 27) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_26_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 27) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_25_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 25) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_24_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 25) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_23_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 23) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_22_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 23) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_21_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 21) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_20_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 21) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_19_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 19) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_18_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 19) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_17_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 17) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_16_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 17) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_15_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_14_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_13_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_12_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_11_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_10_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_9_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_8_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_7_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_6_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_5_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_4_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_3_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_2_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_1_lo);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        *o = (uint8_t) _mm_cvtsi128_si32(v0_0_lo);
      }
    }

    i0 = (const uint8_t*) ((uintptr_t) i0 + input_reset);
    o = (uint8_t*) ((uintptr_t) o + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
