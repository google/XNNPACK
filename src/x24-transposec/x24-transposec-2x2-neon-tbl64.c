// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/transpose.h>

void xnn_x24_transposec_ukernel__2x2_neon_tbl64(
    const void* input,
    void* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height,
    const union xnn_x24_transpose_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_stride >= block_height * 3);
  assert(input_stride >= block_width * 3);

  const size_t tile_height = 2;
  const size_t tile_width = 2;
  const size_t tile_wbytes = tile_width * 3;
  const size_t tile_wbytes_minus_4 = tile_wbytes - 4;
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_height * output_stride - block_height * 3;

  const size_t tile_stride = tile_height * input_stride;

  const uint8_t* i0 = (const uint8_t*) input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);

  uint8_t* o0 = (uint8_t*) output;
  uint8_t* o1 = (uint8_t*) ((uintptr_t) o0 + output_stride);

  const uint8x8_t vperm0 = vld1_u8(params->neon_tbl64.pos0);
  const uint8x8_t vperm1 = vld1_u8(params->neon_tbl64.pos1);
  do {
    if XNN_UNPREDICTABLE(block_width < 2) {
      o1 = o0;
    }
    size_t bh = block_height;
    for (; bh >= 2; bh -= 2) {
      uint8x8x2_t v;
      v.val[0] = vld1_u8(i0); i0 = (const uint8_t*) ((uintptr_t) i0 + tile_stride);
      v.val[1] = vld1_u8(i1); i1 = (const uint8_t*) ((uintptr_t) i1 + tile_stride);

      const uint8x8_t vres0 = vtbl2_u8(v, vperm0);
      const uint8x8_t vres1 = vtbl2_u8(v, vperm1);

      vst1_lane_u32((void*) o1, vreinterpret_u32_u8(vres1), 0); o1 = (uint8_t*) ((uintptr_t) o1 + 4);
      vst1_lane_u32((void*) o0, vreinterpret_u32_u8(vres0), 0); o0 = (uint8_t*) ((uintptr_t) o0 + 4);
      vst1_lane_u16((void*) o1, vreinterpret_u16_u8(vres1), 2); o1 = (uint8_t*) ((uintptr_t) o1 + tile_wbytes_minus_4);
      vst1_lane_u16((void*) o0, vreinterpret_u16_u8(vres0), 2); o0 = (uint8_t*) ((uintptr_t) o0 + tile_wbytes_minus_4);
    }

    if (bh != 0) {
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      uint8x8_t v = vld1_u8(i0);

      const uint8x8_t vres0 = vtbl1_u8(v, vperm0);
      const uint8x8_t vres1 = vtbl1_u8(v, vperm1);

      if (bh & 1) {
        vst1_lane_u16((void*) o1, vreinterpret_u16_u8(vres1), 0); o1 += 2;
        vst1_lane_u16((void*) o0, vreinterpret_u16_u8(vres0), 0); o0 += 2;
        vst1_lane_u8(o1, vres1, 2); o1 += 1;
        vst1_lane_u8(o0, vres0, 2); o0 += 1;
      }
    }
    i0 = (const uint8_t*) ((uintptr_t) i0 + input_reset);
    i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
    o0 = (uint8_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint8_t*) ((uintptr_t) o1 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
