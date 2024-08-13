// Auto-generated file. Do not edit!
//   Template: src/x32-transposec/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <wasm_simd128.h>

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/transpose.h"

void xnn_x8_transposec_ukernel__16x16_reuse_switch_wasmsimd(
    const uint8_t* input,
    uint8_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint8_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint8_t));

  const size_t tile_height = 16;
  const size_t tile_width = 16;
  const size_t tile_hbytes = tile_height * sizeof(uint8_t);
  const size_t tile_wbytes = tile_width * sizeof(uint8_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint8_t);

  const uint8_t* i0 = input;
  uint8_t* o = (uint8_t*) output;
  const size_t minus_output_stride = -output_stride;

  do {
    const size_t rem = min(block_width - 1, 15);
    const size_t oN_stride = rem * output_stride;
    size_t bh = block_height;
    for (; bh >= 16; bh -= 16) {
      const v128_t v4_0 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_1 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_2 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_3 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_4 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_5 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_6 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_7 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_8 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_9 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_10 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_11 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_12 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_13 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_14 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_15 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);

      const v128_t v3_0 = wasm_v8x16_shuffle(v4_0, v4_8, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_1 = wasm_v8x16_shuffle(v4_0, v4_8, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v3_2 = wasm_v8x16_shuffle(v4_1, v4_9, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_3 = wasm_v8x16_shuffle(v4_1, v4_9, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v3_4 = wasm_v8x16_shuffle(v4_2, v4_10, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_5 = wasm_v8x16_shuffle(v4_2, v4_10, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v3_6 = wasm_v8x16_shuffle(v4_3, v4_11, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_7 = wasm_v8x16_shuffle(v4_3, v4_11, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v3_8 = wasm_v8x16_shuffle(v4_4, v4_12, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_9 = wasm_v8x16_shuffle(v4_4, v4_12, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v3_10 = wasm_v8x16_shuffle(v4_5, v4_13, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_11 = wasm_v8x16_shuffle(v4_5, v4_13, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v3_12 = wasm_v8x16_shuffle(v4_6, v4_14, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_13 = wasm_v8x16_shuffle(v4_6, v4_14, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v3_14 = wasm_v8x16_shuffle(v4_7, v4_15, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_15 = wasm_v8x16_shuffle(v4_7, v4_15, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_0 = wasm_v8x16_shuffle(v3_0, v3_8, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_1 = wasm_v8x16_shuffle(v3_0, v3_8, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_2 = wasm_v8x16_shuffle(v3_1, v3_9, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_3 = wasm_v8x16_shuffle(v3_1, v3_9, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_4 = wasm_v8x16_shuffle(v3_2, v3_10, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_5 = wasm_v8x16_shuffle(v3_2, v3_10, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_6 = wasm_v8x16_shuffle(v3_3, v3_11, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_7 = wasm_v8x16_shuffle(v3_3, v3_11, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_8 = wasm_v8x16_shuffle(v3_4, v3_12, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_9 = wasm_v8x16_shuffle(v3_4, v3_12, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_10 = wasm_v8x16_shuffle(v3_5, v3_13, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_11 = wasm_v8x16_shuffle(v3_5, v3_13, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_12 = wasm_v8x16_shuffle(v3_6, v3_14, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_13 = wasm_v8x16_shuffle(v3_6, v3_14, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_14 = wasm_v8x16_shuffle(v3_7, v3_15, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_15 = wasm_v8x16_shuffle(v3_7, v3_15, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_0 = wasm_v8x16_shuffle(v2_0, v2_8, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_1 = wasm_v8x16_shuffle(v2_0, v2_8, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_2 = wasm_v8x16_shuffle(v2_1, v2_9, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_3 = wasm_v8x16_shuffle(v2_1, v2_9, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_4 = wasm_v8x16_shuffle(v2_2, v2_10, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_5 = wasm_v8x16_shuffle(v2_2, v2_10, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_6 = wasm_v8x16_shuffle(v2_3, v2_11, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_7 = wasm_v8x16_shuffle(v2_3, v2_11, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_8 = wasm_v8x16_shuffle(v2_4, v2_12, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_9 = wasm_v8x16_shuffle(v2_4, v2_12, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_10 = wasm_v8x16_shuffle(v2_5, v2_13, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_11 = wasm_v8x16_shuffle(v2_5, v2_13, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_12 = wasm_v8x16_shuffle(v2_6, v2_14, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_13 = wasm_v8x16_shuffle(v2_6, v2_14, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_14 = wasm_v8x16_shuffle(v2_7, v2_15, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_15 = wasm_v8x16_shuffle(v2_7, v2_15, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v0_0 = wasm_v8x16_shuffle(v1_0, v1_8, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v0_1 = wasm_v8x16_shuffle(v1_0, v1_8, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v0_2 = wasm_v8x16_shuffle(v1_1, v1_9, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v0_3 = wasm_v8x16_shuffle(v1_1, v1_9, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v0_4 = wasm_v8x16_shuffle(v1_2, v1_10, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v0_5 = wasm_v8x16_shuffle(v1_2, v1_10, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v0_6 = wasm_v8x16_shuffle(v1_3, v1_11, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v0_7 = wasm_v8x16_shuffle(v1_3, v1_11, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v0_8 = wasm_v8x16_shuffle(v1_4, v1_12, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v0_9 = wasm_v8x16_shuffle(v1_4, v1_12, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v0_10 = wasm_v8x16_shuffle(v1_5, v1_13, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v0_11 = wasm_v8x16_shuffle(v1_5, v1_13, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v0_12 = wasm_v8x16_shuffle(v1_6, v1_14, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v0_13 = wasm_v8x16_shuffle(v1_6, v1_14, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v0_14 = wasm_v8x16_shuffle(v1_7, v1_15, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v0_15 = wasm_v8x16_shuffle(v1_7, v1_15, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);

      uint8_t *oN = (uint8_t*) ((uintptr_t) o + oN_stride);
      switch (rem) {
        case 15:
          wasm_v128_store(oN, v0_15);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 14:
          wasm_v128_store(oN, v0_14);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 13:
          wasm_v128_store(oN, v0_13);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 12:
          wasm_v128_store(oN, v0_12);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 11:
          wasm_v128_store(oN, v0_11);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 10:
          wasm_v128_store(oN, v0_10);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 9:
          wasm_v128_store(oN, v0_9);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 8:
          wasm_v128_store(oN, v0_8);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 7:
          wasm_v128_store(oN, v0_7);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 6:
          wasm_v128_store(oN, v0_6);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 5:
          wasm_v128_store(oN, v0_5);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 4:
          wasm_v128_store(oN, v0_4);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 3:
          wasm_v128_store(oN, v0_3);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 2:
          wasm_v128_store(oN, v0_2);
          oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 1:
          wasm_v128_store(oN, v0_1);
          XNN_FALLTHROUGH
        case 0:
          wasm_v128_store(o, v0_0);
          o = (uint8_t*) ((uintptr_t) o + tile_hbytes);
          break;
        default:
          XNN_UNREACHABLE;
      }
    }

    if (bh != 0) {
      const v128_t v4_0 = wasm_v128_load(i0);
      const uint8_t *i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const v128_t v4_1 = wasm_v128_load(i1);
      const uint8_t *i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i1;
      }
      const v128_t v4_2 = wasm_v128_load(i2);
      const uint8_t *i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
      if XNN_UNPREDICTABLE(bh < 4) {
        i3 = i2;
      }
      const v128_t v4_3 = wasm_v128_load(i3);
      const uint8_t *i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 4) {
        i4 = i3;
      }
      const v128_t v4_4 = wasm_v128_load(i4);
      const uint8_t *i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
      if XNN_UNPREDICTABLE(bh < 6) {
        i5 = i4;
      }
      const v128_t v4_5 = wasm_v128_load(i5);
      const uint8_t *i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 6) {
        i6 = i5;
      }
      const v128_t v4_6 = wasm_v128_load(i6);
      const uint8_t *i7 = (const uint8_t*) ((uintptr_t) i6 + input_stride);
      if XNN_UNPREDICTABLE(bh < 8) {
        i7 = i6;
      }
      const v128_t v4_7 = wasm_v128_load(i7);
      const uint8_t *i8 = (const uint8_t*) ((uintptr_t) i7 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 8) {
        i8 = i7;
      }
      const v128_t v4_8 = wasm_v128_load(i8);
      const uint8_t *i9 = (const uint8_t*) ((uintptr_t) i8 + input_stride);
      if XNN_UNPREDICTABLE(bh < 10) {
        i9 = i8;
      }
      const v128_t v4_9 = wasm_v128_load(i9);
      const uint8_t *i10 = (const uint8_t*) ((uintptr_t) i9 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 10) {
        i10 = i9;
      }
      const v128_t v4_10 = wasm_v128_load(i10);
      const uint8_t *i11 = (const uint8_t*) ((uintptr_t) i10 + input_stride);
      if XNN_UNPREDICTABLE(bh < 12) {
        i11 = i10;
      }
      const v128_t v4_11 = wasm_v128_load(i11);
      const uint8_t *i12 = (const uint8_t*) ((uintptr_t) i11 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 12) {
        i12 = i11;
      }
      const v128_t v4_12 = wasm_v128_load(i12);
      const uint8_t *i13 = (const uint8_t*) ((uintptr_t) i12 + input_stride);
      if XNN_UNPREDICTABLE(bh < 14) {
        i13 = i12;
      }
      const v128_t v4_13 = wasm_v128_load(i13);
      const uint8_t *i14 = (const uint8_t*) ((uintptr_t) i13 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 14) {
        i14 = i13;
      }
      const v128_t v4_14 = wasm_v128_load(i14);
      const v128_t v4_15 = wasm_v128_xor(v4_0, v4_0);

      const v128_t v3_0 = wasm_v8x16_shuffle(v4_0, v4_8, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_1 = wasm_v8x16_shuffle(v4_0, v4_8, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v3_2 = wasm_v8x16_shuffle(v4_1, v4_9, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_3 = wasm_v8x16_shuffle(v4_1, v4_9, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v3_4 = wasm_v8x16_shuffle(v4_2, v4_10, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_5 = wasm_v8x16_shuffle(v4_2, v4_10, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v3_6 = wasm_v8x16_shuffle(v4_3, v4_11, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_7 = wasm_v8x16_shuffle(v4_3, v4_11, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v3_8 = wasm_v8x16_shuffle(v4_4, v4_12, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_9 = wasm_v8x16_shuffle(v4_4, v4_12, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v3_10 = wasm_v8x16_shuffle(v4_5, v4_13, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_11 = wasm_v8x16_shuffle(v4_5, v4_13, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v3_12 = wasm_v8x16_shuffle(v4_6, v4_14, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_13 = wasm_v8x16_shuffle(v4_6, v4_14, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v3_14 = wasm_v8x16_shuffle(v4_7, v4_15, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_15 = wasm_v8x16_shuffle(v4_7, v4_15, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_0 = wasm_v8x16_shuffle(v3_0, v3_8, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_1 = wasm_v8x16_shuffle(v3_0, v3_8, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_2 = wasm_v8x16_shuffle(v3_1, v3_9, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_3 = wasm_v8x16_shuffle(v3_1, v3_9, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_4 = wasm_v8x16_shuffle(v3_2, v3_10, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_5 = wasm_v8x16_shuffle(v3_2, v3_10, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_6 = wasm_v8x16_shuffle(v3_3, v3_11, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_7 = wasm_v8x16_shuffle(v3_3, v3_11, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_8 = wasm_v8x16_shuffle(v3_4, v3_12, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_9 = wasm_v8x16_shuffle(v3_4, v3_12, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_10 = wasm_v8x16_shuffle(v3_5, v3_13, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_11 = wasm_v8x16_shuffle(v3_5, v3_13, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_12 = wasm_v8x16_shuffle(v3_6, v3_14, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_13 = wasm_v8x16_shuffle(v3_6, v3_14, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_14 = wasm_v8x16_shuffle(v3_7, v3_15, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_15 = wasm_v8x16_shuffle(v3_7, v3_15, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_0 = wasm_v8x16_shuffle(v2_0, v2_8, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_1 = wasm_v8x16_shuffle(v2_0, v2_8, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_2 = wasm_v8x16_shuffle(v2_1, v2_9, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_3 = wasm_v8x16_shuffle(v2_1, v2_9, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_4 = wasm_v8x16_shuffle(v2_2, v2_10, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_5 = wasm_v8x16_shuffle(v2_2, v2_10, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_6 = wasm_v8x16_shuffle(v2_3, v2_11, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_7 = wasm_v8x16_shuffle(v2_3, v2_11, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_8 = wasm_v8x16_shuffle(v2_4, v2_12, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_9 = wasm_v8x16_shuffle(v2_4, v2_12, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_10 = wasm_v8x16_shuffle(v2_5, v2_13, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_11 = wasm_v8x16_shuffle(v2_5, v2_13, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_12 = wasm_v8x16_shuffle(v2_6, v2_14, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_13 = wasm_v8x16_shuffle(v2_6, v2_14, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_14 = wasm_v8x16_shuffle(v2_7, v2_15, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_15 = wasm_v8x16_shuffle(v2_7, v2_15, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);

      v128_t v0_0 = wasm_v8x16_shuffle(v1_0, v1_8, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v0_1 = wasm_v8x16_shuffle(v1_0, v1_8, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v0_2 = wasm_v8x16_shuffle(v1_1, v1_9, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v0_3 = wasm_v8x16_shuffle(v1_1, v1_9, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v0_4 = wasm_v8x16_shuffle(v1_2, v1_10, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v0_5 = wasm_v8x16_shuffle(v1_2, v1_10, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v0_6 = wasm_v8x16_shuffle(v1_3, v1_11, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v0_7 = wasm_v8x16_shuffle(v1_3, v1_11, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v0_8 = wasm_v8x16_shuffle(v1_4, v1_12, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v0_9 = wasm_v8x16_shuffle(v1_4, v1_12, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v0_10 = wasm_v8x16_shuffle(v1_5, v1_13, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v0_11 = wasm_v8x16_shuffle(v1_5, v1_13, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v0_12 = wasm_v8x16_shuffle(v1_6, v1_14, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v0_13 = wasm_v8x16_shuffle(v1_6, v1_14, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v0_14 = wasm_v8x16_shuffle(v1_7, v1_15, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v0_15 = wasm_v8x16_shuffle(v1_7, v1_15, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);

      if (bh & 8) {
        uint8_t* oN = (uint8_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 15:
            wasm_v128_store64_lane(oN, v0_15, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 14:
            wasm_v128_store64_lane(oN, v0_14, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 13:
            wasm_v128_store64_lane(oN, v0_13, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 12:
            wasm_v128_store64_lane(oN, v0_12, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 11:
            wasm_v128_store64_lane(oN, v0_11, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 10:
            wasm_v128_store64_lane(oN, v0_10, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 9:
            wasm_v128_store64_lane(oN, v0_9, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 8:
            wasm_v128_store64_lane(oN, v0_8, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 7:
            wasm_v128_store64_lane(oN, v0_7, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 6:
            wasm_v128_store64_lane(oN, v0_6, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 5:
            wasm_v128_store64_lane(oN, v0_5, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 4:
            wasm_v128_store64_lane(oN, v0_4, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 3:
            wasm_v128_store64_lane(oN, v0_3, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 2:
            wasm_v128_store64_lane(oN, v0_2, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 1:
            wasm_v128_store64_lane(oN, v0_1, 0);
            XNN_FALLTHROUGH
          case 0:
            wasm_v128_store64_lane(o, v0_0, 0);
            o += 8;
            break;
          default:
            XNN_UNREACHABLE;
        }
        v0_0 = wasm_v64x2_shuffle(v0_0, v0_0, 1, 1);
        v0_1 = wasm_v64x2_shuffle(v0_1, v0_1, 1, 1);
        v0_2 = wasm_v64x2_shuffle(v0_2, v0_2, 1, 1);
        v0_3 = wasm_v64x2_shuffle(v0_3, v0_3, 1, 1);
        v0_4 = wasm_v64x2_shuffle(v0_4, v0_4, 1, 1);
        v0_5 = wasm_v64x2_shuffle(v0_5, v0_5, 1, 1);
        v0_6 = wasm_v64x2_shuffle(v0_6, v0_6, 1, 1);
        v0_7 = wasm_v64x2_shuffle(v0_7, v0_7, 1, 1);
        v0_8 = wasm_v64x2_shuffle(v0_8, v0_8, 1, 1);
        v0_9 = wasm_v64x2_shuffle(v0_9, v0_9, 1, 1);
        v0_10 = wasm_v64x2_shuffle(v0_10, v0_10, 1, 1);
        v0_11 = wasm_v64x2_shuffle(v0_11, v0_11, 1, 1);
        v0_12 = wasm_v64x2_shuffle(v0_12, v0_12, 1, 1);
        v0_13 = wasm_v64x2_shuffle(v0_13, v0_13, 1, 1);
        v0_14 = wasm_v64x2_shuffle(v0_14, v0_14, 1, 1);
        v0_15 = wasm_v64x2_shuffle(v0_15, v0_15, 1, 1);
      }

      if (bh & 4) {
        uint8_t* oN = (uint8_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 15:
            wasm_v128_store32_lane(oN, v0_15, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 14:
            wasm_v128_store32_lane(oN, v0_14, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 13:
            wasm_v128_store32_lane(oN, v0_13, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 12:
            wasm_v128_store32_lane(oN, v0_12, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 11:
            wasm_v128_store32_lane(oN, v0_11, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 10:
            wasm_v128_store32_lane(oN, v0_10, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 9:
            wasm_v128_store32_lane(oN, v0_9, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 8:
            wasm_v128_store32_lane(oN, v0_8, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 7:
            wasm_v128_store32_lane(oN, v0_7, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 6:
            wasm_v128_store32_lane(oN, v0_6, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 5:
            wasm_v128_store32_lane(oN, v0_5, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 4:
            wasm_v128_store32_lane(oN, v0_4, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 3:
            wasm_v128_store32_lane(oN, v0_3, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 2:
            wasm_v128_store32_lane(oN, v0_2, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 1:
            wasm_v128_store32_lane(oN, v0_1, 0);
            XNN_FALLTHROUGH
          case 0:
            wasm_v128_store32_lane(o, v0_0, 0);
            o += 4;
            break;
          default:
            XNN_UNREACHABLE;
        }
        v0_0 = wasm_u64x2_shr(v0_0, 32);
        v0_1 = wasm_u64x2_shr(v0_1, 32);
        v0_2 = wasm_u64x2_shr(v0_2, 32);
        v0_3 = wasm_u64x2_shr(v0_3, 32);
        v0_4 = wasm_u64x2_shr(v0_4, 32);
        v0_5 = wasm_u64x2_shr(v0_5, 32);
        v0_6 = wasm_u64x2_shr(v0_6, 32);
        v0_7 = wasm_u64x2_shr(v0_7, 32);
        v0_8 = wasm_u64x2_shr(v0_8, 32);
        v0_9 = wasm_u64x2_shr(v0_9, 32);
        v0_10 = wasm_u64x2_shr(v0_10, 32);
        v0_11 = wasm_u64x2_shr(v0_11, 32);
        v0_12 = wasm_u64x2_shr(v0_12, 32);
        v0_13 = wasm_u64x2_shr(v0_13, 32);
        v0_14 = wasm_u64x2_shr(v0_14, 32);
        v0_15 = wasm_u64x2_shr(v0_15, 32);
      }
      if (bh & 2) {
        uint8_t* oN = (uint8_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 15:
            wasm_v128_store16_lane(oN, v0_15, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 14:
            wasm_v128_store16_lane(oN, v0_14, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 13:
            wasm_v128_store16_lane(oN, v0_13, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 12:
            wasm_v128_store16_lane(oN, v0_12, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 11:
            wasm_v128_store16_lane(oN, v0_11, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 10:
            wasm_v128_store16_lane(oN, v0_10, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 9:
            wasm_v128_store16_lane(oN, v0_9, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 8:
            wasm_v128_store16_lane(oN, v0_8, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 7:
            wasm_v128_store16_lane(oN, v0_7, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 6:
            wasm_v128_store16_lane(oN, v0_6, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 5:
            wasm_v128_store16_lane(oN, v0_5, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 4:
            wasm_v128_store16_lane(oN, v0_4, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 3:
            wasm_v128_store16_lane(oN, v0_3, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 2:
            wasm_v128_store16_lane(oN, v0_2, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 1:
            wasm_v128_store16_lane(oN, v0_1, 0);
          case 0:
            wasm_v128_store16_lane(o, v0_0, 0);
            o += 2;
            break;
          default:
            XNN_UNREACHABLE;
        }
        v0_0 = wasm_u32x4_shr(v0_0, 16);
        v0_1 = wasm_u32x4_shr(v0_1, 16);
        v0_2 = wasm_u32x4_shr(v0_2, 16);
        v0_3 = wasm_u32x4_shr(v0_3, 16);
        v0_4 = wasm_u32x4_shr(v0_4, 16);
        v0_5 = wasm_u32x4_shr(v0_5, 16);
        v0_6 = wasm_u32x4_shr(v0_6, 16);
        v0_7 = wasm_u32x4_shr(v0_7, 16);
        v0_8 = wasm_u32x4_shr(v0_8, 16);
        v0_9 = wasm_u32x4_shr(v0_9, 16);
        v0_10 = wasm_u32x4_shr(v0_10, 16);
        v0_11 = wasm_u32x4_shr(v0_11, 16);
        v0_12 = wasm_u32x4_shr(v0_12, 16);
        v0_13 = wasm_u32x4_shr(v0_13, 16);
        v0_14 = wasm_u32x4_shr(v0_14, 16);
        v0_15 = wasm_u32x4_shr(v0_15, 16);
      }
      if (bh & 1) {
        uint8_t* oN = (uint8_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 15:
            wasm_v128_store8_lane(oN, v0_15, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 14:
            wasm_v128_store8_lane(oN, v0_14, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 13:
            wasm_v128_store8_lane(oN, v0_13, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 12:
            wasm_v128_store8_lane(oN, v0_12, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 11:
            wasm_v128_store8_lane(oN, v0_11, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 10:
            wasm_v128_store8_lane(oN, v0_10, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 9:
            wasm_v128_store8_lane(oN, v0_9, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 8:
            wasm_v128_store8_lane(oN, v0_8, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 7:
            wasm_v128_store8_lane(oN, v0_7, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 6:
            wasm_v128_store8_lane(oN, v0_6, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 5:
            wasm_v128_store8_lane(oN, v0_5, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 4:
            wasm_v128_store8_lane(oN, v0_4, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 3:
            wasm_v128_store8_lane(oN, v0_3, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 2:
            wasm_v128_store8_lane(oN, v0_2, 0);
            oN = (uint8_t*) ((uintptr_t) oN + minus_output_stride);
          case 1:
            wasm_v128_store8_lane(oN, v0_1, 0);
          case 0:
            wasm_v128_store8_lane(o, v0_0, 0);
            break;
          default:
            XNN_UNREACHABLE;
        }
      }
    }

    i0 = (const uint8_t*) ((uintptr_t) i0 + input_reset);
    o = (uint8_t*) ((uintptr_t) o + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
