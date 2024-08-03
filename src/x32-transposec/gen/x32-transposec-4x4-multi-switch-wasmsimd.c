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

void xnn_x32_transposec_ukernel__4x4_multi_switch_wasmsimd(
    const uint32_t* input,
    uint32_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint32_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint32_t));

  const size_t tile_height = 4;
  const size_t tile_width = 4;
  const size_t tile_hbytes = tile_height * sizeof(uint32_t);
  const size_t tile_wbytes = tile_width * sizeof(uint32_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t input_offset = tile_height * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint32_t);

  const uint32_t* i0 = input;
  const uint32_t* i1 = (const uint32_t*) ((uintptr_t) i0 + input_stride);
  const uint32_t* i2 = (const uint32_t*) ((uintptr_t) i1 + input_stride);
  const uint32_t* i3 = (const uint32_t*) ((uintptr_t) i2 + input_stride);
  uint32_t* o = (uint32_t*) output;
  const size_t minus_output_stride = -output_stride;

  do {
    const size_t rem = min(block_width - 1, 3);
    const size_t oN_stride = rem * output_stride;
    size_t bh = block_height;
    for (; bh >= 4; bh -= 4) {
      const v128_t v2_0 = wasm_v128_load(i0);
      i0 = (uint32_t*) ((uintptr_t) i0 + input_offset);
      const v128_t v2_1 = wasm_v128_load(i1);
      i1 = (uint32_t*) ((uintptr_t) i1 + input_offset);
      const v128_t v2_2 = wasm_v128_load(i2);
      i2 = (uint32_t*) ((uintptr_t) i2 + input_offset);
      const v128_t v2_3 = wasm_v128_load(i3);
      i3 = (uint32_t*) ((uintptr_t) i3 + input_offset);

      const v128_t v1_0 = wasm_v32x4_shuffle(v2_0, v2_2, 0, 4, 1, 5);
      const v128_t v1_1 = wasm_v32x4_shuffle(v2_0, v2_2, 2, 6, 3, 7);
      const v128_t v1_2 = wasm_v32x4_shuffle(v2_1, v2_3, 0, 4, 1, 5);
      const v128_t v1_3 = wasm_v32x4_shuffle(v2_1, v2_3, 2, 6, 3, 7);
      const v128_t v0_0 = wasm_v32x4_shuffle(v1_0, v1_2, 0, 4, 1, 5);
      const v128_t v0_1 = wasm_v32x4_shuffle(v1_0, v1_2, 2, 6, 3, 7);
      const v128_t v0_2 = wasm_v32x4_shuffle(v1_1, v1_3, 0, 4, 1, 5);
      const v128_t v0_3 = wasm_v32x4_shuffle(v1_1, v1_3, 2, 6, 3, 7);

      uint32_t *oN = (uint32_t*) ((uintptr_t) o + oN_stride);
      switch (rem) {
        case 3:
          wasm_v128_store(oN, v0_3);
          oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 2:
          wasm_v128_store(oN, v0_2);
          oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 1:
          wasm_v128_store(oN, v0_1);
          XNN_FALLTHROUGH
        case 0:
          wasm_v128_store(o, v0_0);
          o = (uint32_t*) ((uintptr_t) o + tile_hbytes);
          break;
        default:
          XNN_UNREACHABLE;
      }
    }

    if (bh != 0) {
      const v128_t v2_0 = wasm_v128_load(i0);
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const v128_t v2_1 = wasm_v128_load(i1);
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i0;
      }
      const v128_t v2_2 = wasm_v128_load(i2);
      const v128_t v2_3 = wasm_v128_xor(v2_0, v2_0);

      const v128_t v1_0 = wasm_v32x4_shuffle(v2_0, v2_2, 0, 4, 1, 5);
      const v128_t v1_1 = wasm_v32x4_shuffle(v2_0, v2_2, 2, 6, 3, 7);
      const v128_t v1_2 = wasm_v32x4_shuffle(v2_1, v2_3, 0, 4, 1, 5);
      const v128_t v1_3 = wasm_v32x4_shuffle(v2_1, v2_3, 2, 6, 3, 7);

      v128_t v0_0 = wasm_v32x4_shuffle(v1_0, v1_2, 0, 4, 1, 5);
      v128_t v0_1 = wasm_v32x4_shuffle(v1_0, v1_2, 2, 6, 3, 7);
      v128_t v0_2 = wasm_v32x4_shuffle(v1_1, v1_3, 0, 4, 1, 5);
      v128_t v0_3 = wasm_v32x4_shuffle(v1_1, v1_3, 2, 6, 3, 7);

      if (bh & 2) {
        uint32_t* oN = (uint32_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 3:
            wasm_v128_store64_lane(oN, v0_3, 0);
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 2:
            wasm_v128_store64_lane(oN, v0_2, 0);
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 1:
            wasm_v128_store64_lane(oN, v0_1, 0);
            XNN_FALLTHROUGH
          case 0:
            wasm_v128_store64_lane(o, v0_0, 0);
            o += 2;
            break;
          default:
            XNN_UNREACHABLE;
        }
        v0_0 = wasm_v64x2_shuffle(v0_0, v0_0, 1, 1);
        v0_1 = wasm_v64x2_shuffle(v0_1, v0_1, 1, 1);
        v0_2 = wasm_v64x2_shuffle(v0_2, v0_2, 1, 1);
        v0_3 = wasm_v64x2_shuffle(v0_3, v0_3, 1, 1);
      }

      if (bh & 1) {
        uint32_t* oN = (uint32_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 3:
            wasm_v128_store32_lane(oN, v0_3, 0);
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 2:
            wasm_v128_store32_lane(oN, v0_2, 0);
            oN = (uint32_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 1:
            wasm_v128_store32_lane(oN, v0_1, 0);
            XNN_FALLTHROUGH
          case 0:
            wasm_v128_store32_lane(o, v0_0, 0);
            break;
          default:
            XNN_UNREACHABLE;
        }
      }
    }

    i0 = (const uint32_t*) ((uintptr_t) i0 + input_reset);
    i1 = (const uint32_t*) ((uintptr_t) i0 + input_stride);
    i2 = (const uint32_t*) ((uintptr_t) i1 + input_stride);
    i3 = (const uint32_t*) ((uintptr_t) i2 + input_stride);
    o = (uint32_t*) ((uintptr_t) o + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
