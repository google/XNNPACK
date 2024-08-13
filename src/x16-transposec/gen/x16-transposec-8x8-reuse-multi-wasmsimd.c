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

void xnn_x16_transposec_ukernel__8x8_reuse_multi_wasmsimd(
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

      wasm_v128_store(o7, v0_7);
      o7 = (uint16_t*) ((uintptr_t) o7 + tile_hbytes);
      wasm_v128_store(o6, v0_6);
      o6 = (uint16_t*) ((uintptr_t) o6 + tile_hbytes);
      wasm_v128_store(o5, v0_5);
      o5 = (uint16_t*) ((uintptr_t) o5 + tile_hbytes);
      wasm_v128_store(o4, v0_4);
      o4 = (uint16_t*) ((uintptr_t) o4 + tile_hbytes);
      wasm_v128_store(o3, v0_3);
      o3 = (uint16_t*) ((uintptr_t) o3 + tile_hbytes);
      wasm_v128_store(o2, v0_2);
      o2 = (uint16_t*) ((uintptr_t) o2 + tile_hbytes);
      wasm_v128_store(o1, v0_1);
      o1 = (uint16_t*) ((uintptr_t) o1 + tile_hbytes);
      wasm_v128_store(o0, v0_0);
      o0 = (uint16_t*) ((uintptr_t) o0 + tile_hbytes);
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
        wasm_v128_store64_lane(o7, v0_7, 0);
        o7 += 4;
        wasm_v128_store64_lane(o6, v0_6, 0);
        o6 += 4;
        wasm_v128_store64_lane(o5, v0_5, 0);
        o5 += 4;
        wasm_v128_store64_lane(o4, v0_4, 0);
        o4 += 4;
        wasm_v128_store64_lane(o3, v0_3, 0);
        o3 += 4;
        wasm_v128_store64_lane(o2, v0_2, 0);
        o2 += 4;
        wasm_v128_store64_lane(o1, v0_1, 0);
        o1 += 4;
        wasm_v128_store64_lane(o0, v0_0, 0);
        o0 += 4;
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
        wasm_v128_store32_lane(o7, v0_7, 0);
        o7 += 2;
        wasm_v128_store32_lane(o6, v0_6, 0);
        o6 += 2;
        wasm_v128_store32_lane(o5, v0_5, 0);
        o5 += 2;
        wasm_v128_store32_lane(o4, v0_4, 0);
        o4 += 2;
        wasm_v128_store32_lane(o3, v0_3, 0);
        o3 += 2;
        wasm_v128_store32_lane(o2, v0_2, 0);
        o2 += 2;
        wasm_v128_store32_lane(o1, v0_1, 0);
        o1 += 2;
        wasm_v128_store32_lane(o0, v0_0, 0);
        o0 += 2;
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
        wasm_v128_store16_lane(o7, v0_7, 0);
        wasm_v128_store16_lane(o6, v0_6, 0);
        wasm_v128_store16_lane(o5, v0_5, 0);
        wasm_v128_store16_lane(o4, v0_4, 0);
        wasm_v128_store16_lane(o3, v0_3, 0);
        wasm_v128_store16_lane(o2, v0_2, 0);
        wasm_v128_store16_lane(o1, v0_1, 0);
        wasm_v128_store16_lane(o0, v0_0, 0);
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
