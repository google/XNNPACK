// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/transpose.h>

void xnn_x32_transpose_ukernel__4x4_wasmsimd(
    const uint32_t* input,
    uint32_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height)
{
  assert(output_stride >= block_height * sizeof(float));
  assert(input_stride >= block_width * sizeof(float));

  const size_t tile_height = 4;
  const size_t tile_width = 4;
  const size_t tile_wbytes = tile_width * sizeof(float);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_height * output_stride - round_down_po2(block_height, 2) * sizeof(uint32_t);
  const size_t input_offset = tile_height * input_stride;

  const float* i0 = (const float*) input;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_stride);

  float* o0 = (float*) output;
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);
  float* o2 = (float*) ((uintptr_t) o1 + output_stride);
  float* o3 = (float*) ((uintptr_t) o2 + output_stride);

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
      v128_t v0 = wasm_v128_load(i0);
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      v128_t v1 = wasm_v128_load(i1);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      v128_t v2 = wasm_v128_load(i2);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      v128_t v3 = wasm_v128_load(i3);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);

      const v128_t vtmp0 = wasm_v32x4_shuffle(v0, v1, 0, 4, 1, 5);
      const v128_t vtmp1 = wasm_v32x4_shuffle(v2, v3, 0, 4, 1, 5);
      const v128_t vtmp2 = wasm_v32x4_shuffle(v0, v1, 2, 6, 3, 7);
      const v128_t vtmp3 = wasm_v32x4_shuffle(v2, v3, 2, 6, 3, 7);

      v0 = wasm_v64x2_shuffle(vtmp0, vtmp1, 0, 2);
      v1 = wasm_v64x2_shuffle(vtmp0, vtmp1, 1, 3);
      v2 = wasm_v64x2_shuffle(vtmp2, vtmp3, 0, 2);
      v3 = wasm_v64x2_shuffle(vtmp2, vtmp3, 1, 3);

      wasm_v128_store(o3, v3);
      o3 = (float*) ((uintptr_t) o3 + tile_wbytes);
      wasm_v128_store(o2, v2);
      o2 = (float*) ((uintptr_t) o2 + tile_wbytes);
      wasm_v128_store(o1, v1);
      o1 = (float*) ((uintptr_t) o1 + tile_wbytes);
      wasm_v128_store(o0, v0);
      o0 = (float*) ((uintptr_t) o0 + tile_wbytes);
    }

    if (bh != 0) {
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i0;
      }
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }

      v128_t v0 = wasm_v128_load(i0);
      v128_t v1 = wasm_v128_load(i1);
      v128_t v2 = wasm_v128_load(i2);

      const v128_t vtmp0 = wasm_v32x4_shuffle(v0, v1, 0, 4, 1, 5);
      const v128_t vtmp1 = wasm_v32x4_shuffle(v2, v1, 0, 4, 1, 5);
      const v128_t vtmp2 = wasm_v32x4_shuffle(v0, v1, 2, 6, 3, 7);
      const v128_t vtmp3 = wasm_v32x4_shuffle(v2, v1, 2, 6, 3, 7);

      v0 = wasm_v64x2_shuffle(vtmp0, vtmp1, 0, 2);
      v1 = wasm_v64x2_shuffle(vtmp0, vtmp1, 1, 3);
      v2 = wasm_v64x2_shuffle(vtmp2, vtmp3, 0, 2);
      v128_t v3 = wasm_v64x2_shuffle(vtmp2, vtmp3, 1, 3);

      if (bh & 2) {
        *((double*) o3) = wasm_f64x2_extract_lane(v3, 0);
        o3 += 2;
        *((double*) o2) = wasm_f64x2_extract_lane(v2, 0);
        o2 += 2;
        *((double*) o1) = wasm_f64x2_extract_lane(v1, 0);
        o1 += 2;
        *((double*) o0) = wasm_f64x2_extract_lane(v0, 0);
        o0 += 2;

        v0 = wasm_v64x2_shuffle(v0, v0, 1, 1);
        v1 = wasm_v64x2_shuffle(v1, v1, 1, 1);
        v2 = wasm_v64x2_shuffle(v2, v2, 1, 1);
        v3 = wasm_v64x2_shuffle(v3, v3, 1, 1);
      }
      if (bh & 1) {
        *o3 = wasm_f32x4_extract_lane(v3, 0);
        *o2 = wasm_f32x4_extract_lane(v2, 0);
        *o1 = wasm_f32x4_extract_lane(v1, 0);
        *o0 = wasm_f32x4_extract_lane(v0, 0);
      }
    }
    i0 = (const float*) ((uintptr_t) i0 + input_reset);
    i1 = (const float*) ((uintptr_t) i0 + input_stride);
    i2 = (const float*) ((uintptr_t) i1 + input_stride);
    i3 = (const float*) ((uintptr_t) i2 + input_stride);
    o0 = (float*) ((uintptr_t) o0 + output_reset);
    o1 = (float*) ((uintptr_t) o1 + output_reset);
    o2 = (float*) ((uintptr_t) o2 + output_reset);
    o3 = (float*) ((uintptr_t) o3 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
