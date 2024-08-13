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

void xnn_x16_transposec_ukernel__8x8_reuse_switch_wasmsimd(
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
  uint16_t* o = (uint16_t*) output;
  const size_t minus_output_stride = -output_stride;

  do {
    const size_t rem = min(block_width - 1, 7);
    const size_t oN_stride = rem * output_stride;
    size_t bh = block_height;
    for (; bh >= 8; bh -= 8) {
      const v128_t v3_0 = wasm_v128_load(i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v3_1 = wasm_v128_load(i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v3_2 = wasm_v128_load(i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v3_3 = wasm_v128_load(i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v3_4 = wasm_v128_load(i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v3_5 = wasm_v128_load(i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v3_6 = wasm_v128_load(i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v3_7 = wasm_v128_load(i0);
      i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);

      const v128_t v2_0 = wasm_v16x8_shuffle(v3_0, v3_4, 0, 8, 1, 9, 2, 10, 3, 11);
      const v128_t v2_1 = wasm_v16x8_shuffle(v3_0, v3_4, 4, 12, 5, 13, 6, 14, 7, 15);
      const v128_t v2_2 = wasm_v16x8_shuffle(v3_1, v3_5, 0, 8, 1, 9, 2, 10, 3, 11);
      const v128_t v2_3 = wasm_v16x8_shuffle(v3_1, v3_5, 4, 12, 5, 13, 6, 14, 7, 15);
      const v128_t v2_4 = wasm_v16x8_shuffle(v3_2, v3_6, 0, 8, 1, 9, 2, 10, 3, 11);
      const v128_t v2_5 = wasm_v16x8_shuffle(v3_2, v3_6, 4, 12, 5, 13, 6, 14, 7, 15);
      const v128_t v2_6 = wasm_v16x8_shuffle(v3_3, v3_7, 0, 8, 1, 9, 2, 10, 3, 11);
      const v128_t v2_7 = wasm_v16x8_shuffle(v3_3, v3_7, 4, 12, 5, 13, 6, 14, 7, 15);
      const v128_t v1_0 = wasm_v16x8_shuffle(v2_0, v2_4, 0, 8, 1, 9, 2, 10, 3, 11);
      const v128_t v1_1 = wasm_v16x8_shuffle(v2_0, v2_4, 4, 12, 5, 13, 6, 14, 7, 15);
      const v128_t v1_2 = wasm_v16x8_shuffle(v2_1, v2_5, 0, 8, 1, 9, 2, 10, 3, 11);
      const v128_t v1_3 = wasm_v16x8_shuffle(v2_1, v2_5, 4, 12, 5, 13, 6, 14, 7, 15);
      const v128_t v1_4 = wasm_v16x8_shuffle(v2_2, v2_6, 0, 8, 1, 9, 2, 10, 3, 11);
      const v128_t v1_5 = wasm_v16x8_shuffle(v2_2, v2_6, 4, 12, 5, 13, 6, 14, 7, 15);
      const v128_t v1_6 = wasm_v16x8_shuffle(v2_3, v2_7, 0, 8, 1, 9, 2, 10, 3, 11);
      const v128_t v1_7 = wasm_v16x8_shuffle(v2_3, v2_7, 4, 12, 5, 13, 6, 14, 7, 15);
      const v128_t v0_0 = wasm_v16x8_shuffle(v1_0, v1_4, 0, 8, 1, 9, 2, 10, 3, 11);
      const v128_t v0_1 = wasm_v16x8_shuffle(v1_0, v1_4, 4, 12, 5, 13, 6, 14, 7, 15);
      const v128_t v0_2 = wasm_v16x8_shuffle(v1_1, v1_5, 0, 8, 1, 9, 2, 10, 3, 11);
      const v128_t v0_3 = wasm_v16x8_shuffle(v1_1, v1_5, 4, 12, 5, 13, 6, 14, 7, 15);
      const v128_t v0_4 = wasm_v16x8_shuffle(v1_2, v1_6, 0, 8, 1, 9, 2, 10, 3, 11);
      const v128_t v0_5 = wasm_v16x8_shuffle(v1_2, v1_6, 4, 12, 5, 13, 6, 14, 7, 15);
      const v128_t v0_6 = wasm_v16x8_shuffle(v1_3, v1_7, 0, 8, 1, 9, 2, 10, 3, 11);
      const v128_t v0_7 = wasm_v16x8_shuffle(v1_3, v1_7, 4, 12, 5, 13, 6, 14, 7, 15);

      uint16_t *oN = (uint16_t*) ((uintptr_t) o + oN_stride);
      switch (rem) {
        case 7:
          wasm_v128_store(oN, v0_7);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 6:
          wasm_v128_store(oN, v0_6);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 5:
          wasm_v128_store(oN, v0_5);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 4:
          wasm_v128_store(oN, v0_4);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 3:
          wasm_v128_store(oN, v0_3);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 2:
          wasm_v128_store(oN, v0_2);
          oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 1:
          wasm_v128_store(oN, v0_1);
          XNN_FALLTHROUGH
        case 0:
          wasm_v128_store(o, v0_0);
          o = (uint16_t*) ((uintptr_t) o + tile_hbytes);
          break;
        default:
          XNN_UNREACHABLE;
      }
    }

    if (bh != 0) {
      const v128_t v3_0 = wasm_v128_load(i0);
      const uint16_t *i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const v128_t v3_1 = wasm_v128_load(i1);
      const uint16_t *i2 = (const uint16_t*) ((uintptr_t) i1 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i1;
      }
      const v128_t v3_2 = wasm_v128_load(i2);
      const uint16_t *i3 = (const uint16_t*) ((uintptr_t) i2 + input_stride);
      if XNN_UNPREDICTABLE(bh < 4) {
        i3 = i2;
      }
      const v128_t v3_3 = wasm_v128_load(i3);
      const uint16_t *i4 = (const uint16_t*) ((uintptr_t) i3 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 4) {
        i4 = i3;
      }
      const v128_t v3_4 = wasm_v128_load(i4);
      const uint16_t *i5 = (const uint16_t*) ((uintptr_t) i4 + input_stride);
      if XNN_UNPREDICTABLE(bh < 6) {
        i5 = i4;
      }
      const v128_t v3_5 = wasm_v128_load(i5);
      const uint16_t *i6 = (const uint16_t*) ((uintptr_t) i5 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 6) {
        i6 = i5;
      }
      const v128_t v3_6 = wasm_v128_load(i6);
      const v128_t v3_7 = wasm_v128_xor(v3_0, v3_0);

      const v128_t v2_0 = wasm_v16x8_shuffle(v3_0, v3_4, 0, 8, 1, 9, 2, 10, 3, 11);
      const v128_t v2_1 = wasm_v16x8_shuffle(v3_0, v3_4, 4, 12, 5, 13, 6, 14, 7, 15);
      const v128_t v2_2 = wasm_v16x8_shuffle(v3_1, v3_5, 0, 8, 1, 9, 2, 10, 3, 11);
      const v128_t v2_3 = wasm_v16x8_shuffle(v3_1, v3_5, 4, 12, 5, 13, 6, 14, 7, 15);
      const v128_t v2_4 = wasm_v16x8_shuffle(v3_2, v3_6, 0, 8, 1, 9, 2, 10, 3, 11);
      const v128_t v2_5 = wasm_v16x8_shuffle(v3_2, v3_6, 4, 12, 5, 13, 6, 14, 7, 15);
      const v128_t v2_6 = wasm_v16x8_shuffle(v3_3, v3_7, 0, 8, 1, 9, 2, 10, 3, 11);
      const v128_t v2_7 = wasm_v16x8_shuffle(v3_3, v3_7, 4, 12, 5, 13, 6, 14, 7, 15);
      const v128_t v1_0 = wasm_v16x8_shuffle(v2_0, v2_4, 0, 8, 1, 9, 2, 10, 3, 11);
      const v128_t v1_1 = wasm_v16x8_shuffle(v2_0, v2_4, 4, 12, 5, 13, 6, 14, 7, 15);
      const v128_t v1_2 = wasm_v16x8_shuffle(v2_1, v2_5, 0, 8, 1, 9, 2, 10, 3, 11);
      const v128_t v1_3 = wasm_v16x8_shuffle(v2_1, v2_5, 4, 12, 5, 13, 6, 14, 7, 15);
      const v128_t v1_4 = wasm_v16x8_shuffle(v2_2, v2_6, 0, 8, 1, 9, 2, 10, 3, 11);
      const v128_t v1_5 = wasm_v16x8_shuffle(v2_2, v2_6, 4, 12, 5, 13, 6, 14, 7, 15);
      const v128_t v1_6 = wasm_v16x8_shuffle(v2_3, v2_7, 0, 8, 1, 9, 2, 10, 3, 11);
      const v128_t v1_7 = wasm_v16x8_shuffle(v2_3, v2_7, 4, 12, 5, 13, 6, 14, 7, 15);

      v128_t v0_0 = wasm_v16x8_shuffle(v1_0, v1_4, 0, 8, 1, 9, 2, 10, 3, 11);
      v128_t v0_1 = wasm_v16x8_shuffle(v1_0, v1_4, 4, 12, 5, 13, 6, 14, 7, 15);
      v128_t v0_2 = wasm_v16x8_shuffle(v1_1, v1_5, 0, 8, 1, 9, 2, 10, 3, 11);
      v128_t v0_3 = wasm_v16x8_shuffle(v1_1, v1_5, 4, 12, 5, 13, 6, 14, 7, 15);
      v128_t v0_4 = wasm_v16x8_shuffle(v1_2, v1_6, 0, 8, 1, 9, 2, 10, 3, 11);
      v128_t v0_5 = wasm_v16x8_shuffle(v1_2, v1_6, 4, 12, 5, 13, 6, 14, 7, 15);
      v128_t v0_6 = wasm_v16x8_shuffle(v1_3, v1_7, 0, 8, 1, 9, 2, 10, 3, 11);
      v128_t v0_7 = wasm_v16x8_shuffle(v1_3, v1_7, 4, 12, 5, 13, 6, 14, 7, 15);

      if (bh & 4) {
        uint16_t* oN = (uint16_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 7:
            wasm_v128_store64_lane(oN, v0_7, 0);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 6:
            wasm_v128_store64_lane(oN, v0_6, 0);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 5:
            wasm_v128_store64_lane(oN, v0_5, 0);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 4:
            wasm_v128_store64_lane(oN, v0_4, 0);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 3:
            wasm_v128_store64_lane(oN, v0_3, 0);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 2:
            wasm_v128_store64_lane(oN, v0_2, 0);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 1:
            wasm_v128_store64_lane(oN, v0_1, 0);
            XNN_FALLTHROUGH
          case 0:
            wasm_v128_store64_lane(o, v0_0, 0);
            o += 4;
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
      }

      if (bh & 2) {
        uint16_t* oN = (uint16_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 7:
            wasm_v128_store32_lane(oN, v0_7, 0);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 6:
            wasm_v128_store32_lane(oN, v0_6, 0);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 5:
            wasm_v128_store32_lane(oN, v0_5, 0);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 4:
            wasm_v128_store32_lane(oN, v0_4, 0);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 3:
            wasm_v128_store32_lane(oN, v0_3, 0);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 2:
            wasm_v128_store32_lane(oN, v0_2, 0);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 1:
            wasm_v128_store32_lane(oN, v0_1, 0);
            XNN_FALLTHROUGH
          case 0:
            wasm_v128_store32_lane(o, v0_0, 0);
            o += 2;
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
      }
      if (bh & 1) {
        uint16_t* oN = (uint16_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 7:
            wasm_v128_store16_lane(oN, v0_7, 0);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 6:
            wasm_v128_store16_lane(oN, v0_6, 0);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 5:
            wasm_v128_store16_lane(oN, v0_5, 0);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 4:
            wasm_v128_store16_lane(oN, v0_4, 0);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 3:
            wasm_v128_store16_lane(oN, v0_3, 0);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 2:
            wasm_v128_store16_lane(oN, v0_2, 0);
            oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          case 1:
            wasm_v128_store16_lane(oN, v0_1, 0);
          case 0:
            wasm_v128_store16_lane(o, v0_0, 0);
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
