// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/transpose.h"

void xnn_x32_transposec_ukernel__4x4_aarch64_neon_tbl128(
    const uint32_t* input,
    uint32_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{  
  static const uint8_t pos0[16] = {0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51};
  static const uint8_t pos1[16] = {4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55};
  static const uint8_t pos2[16] = {8, 9, 10, 11, 24, 25, 26, 27, 40, 41, 42, 43, 56, 57, 58, 59};
  static const uint8_t pos3[16] = {12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63};

  assert(block_width == 1 || output_stride >= block_height * sizeof(uint32_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint32_t));

  const size_t tile_height = 4;
  const size_t tile_width = 4;
  const size_t tile_wbytes = tile_width * sizeof(uint32_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_height * output_stride - round_down_po2(block_height, 2) * sizeof(uint32_t);
  const size_t tile_stride = tile_height * input_stride;

  const uint8_t* i0 = (const uint8_t*) input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);

  uint8_t* o0 = (uint8_t*) output;
  uint8_t* o1 = (uint8_t*) ((uintptr_t) o0 + output_stride);
  uint8_t* o2 = (uint8_t*) ((uintptr_t) o1 + output_stride);
  uint8_t* o3 = (uint8_t*) ((uintptr_t) o2 + output_stride);

  const uint8x16_t vperm0 = vld1q_u8(pos0);
  const uint8x16_t vperm1 = vld1q_u8(pos1);
  const uint8x16_t vperm2 = vld1q_u8(pos2);
  const uint8x16_t vperm3 = vld1q_u8(pos3);
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
      uint8x16x4_t v;
      v.val[0] = vld1q_u8(i0); i0 = (const uint8_t*) ((uintptr_t) i0 + tile_stride);
      v.val[1] = vld1q_u8(i1); i1 = (const uint8_t*) ((uintptr_t) i1 + tile_stride);
      v.val[2] = vld1q_u8(i2); i2 = (const uint8_t*) ((uintptr_t) i2 + tile_stride);
      v.val[3] = vld1q_u8(i3); i3 = (const uint8_t*) ((uintptr_t) i3 + tile_stride);

      uint8x16_t vres0 = vqtbl4q_u8(v, vperm0);
      uint8x16_t vres1 = vqtbl4q_u8(v, vperm1);
      uint8x16_t vres2 = vqtbl4q_u8(v, vperm2);
      uint8x16_t vres3 = vqtbl4q_u8(v, vperm3);

      vst1q_u8(o3, vres3); o3 = (uint8_t*) ((uintptr_t) o3 + tile_wbytes);
      vst1q_u8(o2, vres2); o2 = (uint8_t*) ((uintptr_t) o2 + tile_wbytes);
      vst1q_u8(o1, vres1); o1 = (uint8_t*) ((uintptr_t) o1 + tile_wbytes);
      vst1q_u8(o0, vres0); o0 = (uint8_t*) ((uintptr_t) o0 + tile_wbytes);
    }

    if (bh != 0) {
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i0;
      }
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      uint8x16x4_t v;
      v.val[0] = vld1q_u8(i0);
      v.val[1] = vld1q_u8(i1);
      v.val[2] = vld1q_u8(i2);

      uint8x16_t vres0 = vqtbl4q_u8(v, vperm0);
      uint8x16_t vres1 = vqtbl4q_u8(v, vperm1);
      uint8x16_t vres2 = vqtbl4q_u8(v, vperm2);
      uint8x16_t vres3 = vqtbl4q_u8(v, vperm3);

      uint8x8_t vres0_low = vget_low_u8(vres0);
      uint8x8_t vres1_low = vget_low_u8(vres1);
      uint8x8_t vres2_low = vget_low_u8(vres2);
      uint8x8_t vres3_low = vget_low_u8(vres3);

      if (bh & 2) {
        vst1_u8(o3, vres3_low); o3 += 8;
        vst1_u8(o2, vres2_low); o2 += 8;
        vst1_u8(o1, vres1_low); o1 += 8;
        vst1_u8(o0, vres0_low); o0 += 8;
        vres0_low = vget_high_u8(vres0);
        vres1_low = vget_high_u8(vres1);
        vres2_low = vget_high_u8(vres2);
        vres3_low = vget_high_u8(vres3);
      }
      if (bh & 1) {
        vst1_lane_u32((void*) o3, vreinterpret_u32_u8(vres3_low), 0);
        vst1_lane_u32((void*) o2, vreinterpret_u32_u8(vres2_low), 0);
        vst1_lane_u32((void*) o1, vreinterpret_u32_u8(vres1_low), 0);
        vst1_lane_u32((void*) o0, vreinterpret_u32_u8(vres0_low), 0);
      }
    }
    i0 = (const uint8_t*) ((uintptr_t) i0 + input_reset);
    i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
    i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
    i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
    o0 = (uint8_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint8_t*) ((uintptr_t) o1 + output_reset);
    o2 = (uint8_t*) ((uintptr_t) o2 + output_reset);
    o3 = (uint8_t*) ((uintptr_t) o3 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
