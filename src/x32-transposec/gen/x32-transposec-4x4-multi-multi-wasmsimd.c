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

void xnn_x32_transposec_ukernel__4x4_multi_multi_wasmsimd(
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
  uint32_t* o0 = (uint32_t*) output;
  uint32_t* o1 = (uint32_t*) ((uintptr_t) o0 + output_stride);
  uint32_t* o2 = (uint32_t*) ((uintptr_t) o1 + output_stride);
  uint32_t* o3 = (uint32_t*) ((uintptr_t) o2 + output_stride);

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

      wasm_v128_store(o3, v0_3);
      o3 = (uint32_t*) ((uintptr_t) o3 + tile_hbytes);
      wasm_v128_store(o2, v0_2);
      o2 = (uint32_t*) ((uintptr_t) o2 + tile_hbytes);
      wasm_v128_store(o1, v0_1);
      o1 = (uint32_t*) ((uintptr_t) o1 + tile_hbytes);
      wasm_v128_store(o0, v0_0);
      o0 = (uint32_t*) ((uintptr_t) o0 + tile_hbytes);
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
        wasm_v128_store64_lane(o3, v0_3, 0);
        o3 += 2;
        wasm_v128_store64_lane(o2, v0_2, 0);
        o2 += 2;
        wasm_v128_store64_lane(o1, v0_1, 0);
        o1 += 2;
        wasm_v128_store64_lane(o0, v0_0, 0);
        o0 += 2;
        v0_0 = wasm_v64x2_shuffle(v0_0, v0_0, 1, 1);
        v0_1 = wasm_v64x2_shuffle(v0_1, v0_1, 1, 1);
        v0_2 = wasm_v64x2_shuffle(v0_2, v0_2, 1, 1);
        v0_3 = wasm_v64x2_shuffle(v0_3, v0_3, 1, 1);
      }

      if (bh & 1) {
        wasm_v128_store32_lane(o3, v0_3, 0);
        wasm_v128_store32_lane(o2, v0_2, 0);
        wasm_v128_store32_lane(o1, v0_1, 0);
        wasm_v128_store32_lane(o0, v0_0, 0);
      }
    }

    i0 = (const uint32_t*) ((uintptr_t) i0 + input_reset);
    i1 = (const uint32_t*) ((uintptr_t) i0 + input_stride);
    i2 = (const uint32_t*) ((uintptr_t) i1 + input_stride);
    i3 = (const uint32_t*) ((uintptr_t) i2 + input_stride);
    o0 = (uint32_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint32_t*) ((uintptr_t) o1 + output_reset);
    o2 = (uint32_t*) ((uintptr_t) o2 + output_reset);
    o3 = (uint32_t*) ((uintptr_t) o3 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
