// Auto-generated file. Do not edit!
//   Template: src/x32-transposec/neon-zip.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/transpose.h"

void xnn_x16_transposec_ukernel__8x8_reuse_multi_zip_neon(
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
      const uint16x8_t v3_0 = vld1q_u16(i0); i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const uint16x8_t v3_1 = vld1q_u16(i0); i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const uint16x8_t v3_2 = vld1q_u16(i0); i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const uint16x8_t v3_3 = vld1q_u16(i0); i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const uint16x8_t v3_4 = vld1q_u16(i0); i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const uint16x8_t v3_5 = vld1q_u16(i0); i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const uint16x8_t v3_6 = vld1q_u16(i0); i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);
      const uint16x8_t v3_7 = vld1q_u16(i0); i0 = (uint16_t*) ((uintptr_t) i0 + input_stride);

      const uint16x8x2_t v2_0 = vzipq_u16(v3_0, v3_4);
      const uint16x8x2_t v2_1 = vzipq_u16(v3_1, v3_5);
      const uint16x8x2_t v2_2 = vzipq_u16(v3_2, v3_6);
      const uint16x8x2_t v2_3 = vzipq_u16(v3_3, v3_7);

      const uint16x8x2_t v1_0 = vzipq_u16(v2_0.val[0], v2_2.val[0]);
      const uint16x8x2_t v1_1 = vzipq_u16(v2_0.val[1], v2_2.val[1]);
      const uint16x8x2_t v1_2 = vzipq_u16(v2_1.val[0], v2_3.val[0]);
      const uint16x8x2_t v1_3 = vzipq_u16(v2_1.val[1], v2_3.val[1]);
      const uint16x8x2_t v0_0 = vzipq_u16(v1_0.val[0], v1_2.val[0]);
      const uint16x8x2_t v0_1 = vzipq_u16(v1_0.val[1], v1_2.val[1]);
      const uint16x8x2_t v0_2 = vzipq_u16(v1_1.val[0], v1_3.val[0]);
      const uint16x8x2_t v0_3 = vzipq_u16(v1_1.val[1], v1_3.val[1]);

      vst1q_u16(o7, v0_3.val[1]); o7 = (uint16_t*) ((uintptr_t) o7 + tile_hbytes);
      vst1q_u16(o6, v0_3.val[0]); o6 = (uint16_t*) ((uintptr_t) o6 + tile_hbytes);
      vst1q_u16(o5, v0_2.val[1]); o5 = (uint16_t*) ((uintptr_t) o5 + tile_hbytes);
      vst1q_u16(o4, v0_2.val[0]); o4 = (uint16_t*) ((uintptr_t) o4 + tile_hbytes);
      vst1q_u16(o3, v0_1.val[1]); o3 = (uint16_t*) ((uintptr_t) o3 + tile_hbytes);
      vst1q_u16(o2, v0_1.val[0]); o2 = (uint16_t*) ((uintptr_t) o2 + tile_hbytes);
      vst1q_u16(o1, v0_0.val[1]); o1 = (uint16_t*) ((uintptr_t) o1 + tile_hbytes);
      vst1q_u16(o0, v0_0.val[0]); o0 = (uint16_t*) ((uintptr_t) o0 + tile_hbytes);
    }

    if (bh != 0) {
      const uint16x8_t v3_0 = vld1q_u16(i0);
      const uint16_t *i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const uint16x8_t v3_1 = vld1q_u16(i1);
      const uint16_t *i2 = (const uint16_t*) ((uintptr_t) i1 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i1;
      }
      const uint16x8_t v3_2 = vld1q_u16(i2);
      const uint16_t *i3 = (const uint16_t*) ((uintptr_t) i2 + input_stride);
      if XNN_UNPREDICTABLE(bh < 4) {
        i3 = i2;
      }
      const uint16x8_t v3_3 = vld1q_u16(i3);
      const uint16_t *i4 = (const uint16_t*) ((uintptr_t) i3 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 4) {
        i4 = i3;
      }
      const uint16x8_t v3_4 = vld1q_u16(i4);
      const uint16_t *i5 = (const uint16_t*) ((uintptr_t) i4 + input_stride);
      if XNN_UNPREDICTABLE(bh < 6) {
        i5 = i4;
      }
      const uint16x8_t v3_5 = vld1q_u16(i5);
      const uint16_t *i6 = (const uint16_t*) ((uintptr_t) i5 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 6) {
        i6 = i5;
      }
      const uint16x8_t v3_6 = vld1q_u16(i6);
      const uint16x8_t v3_7 = vmovq_n_u16(0);

      const uint16x8x2_t v2_0 = vzipq_u16(v3_0, v3_4);
      const uint16x8x2_t v2_1 = vzipq_u16(v3_1, v3_5);
      const uint16x8x2_t v2_2 = vzipq_u16(v3_2, v3_6);
      const uint16x8x2_t v2_3 = vzipq_u16(v3_3, v3_7);

      const uint16x8x2_t v1_0 = vzipq_u16(v2_0.val[0], v2_2.val[0]);
      const uint16x8x2_t v1_1 = vzipq_u16(v2_0.val[1], v2_2.val[1]);
      const uint16x8x2_t v1_2 = vzipq_u16(v2_1.val[0], v2_3.val[0]);
      const uint16x8x2_t v1_3 = vzipq_u16(v2_1.val[1], v2_3.val[1]);
      const uint16x8x2_t v0_0 = vzipq_u16(v1_0.val[0], v1_2.val[0]);
      const uint16x8x2_t v0_1 = vzipq_u16(v1_0.val[1], v1_2.val[1]);
      const uint16x8x2_t v0_2 = vzipq_u16(v1_1.val[0], v1_3.val[0]);
      const uint16x8x2_t v0_3 = vzipq_u16(v1_1.val[1], v1_3.val[1]);

      uint16x4_t v0_low = vget_low_u16(v0_0.val[0]);
      uint16x4_t v1_low = vget_low_u16(v0_0.val[1]);
      uint16x4_t v2_low = vget_low_u16(v0_1.val[0]);
      uint16x4_t v3_low = vget_low_u16(v0_1.val[1]);
      uint16x4_t v4_low = vget_low_u16(v0_2.val[0]);
      uint16x4_t v5_low = vget_low_u16(v0_2.val[1]);
      uint16x4_t v6_low = vget_low_u16(v0_3.val[0]);
      uint16x4_t v7_low = vget_low_u16(v0_3.val[1]);

      if (bh & 4) {
        vst1_u16(o7, v7_low); o7 += 4;
        vst1_u16(o6, v6_low); o6 += 4;
        vst1_u16(o5, v5_low); o5 += 4;
        vst1_u16(o4, v4_low); o4 += 4;
        vst1_u16(o3, v3_low); o3 += 4;
        vst1_u16(o2, v2_low); o2 += 4;
        vst1_u16(o1, v1_low); o1 += 4;
        vst1_u16(o0, v0_low); o0 += 4;
        v0_low = vget_high_u16(v0_0.val[0]);
        v1_low = vget_high_u16(v0_0.val[1]);
        v2_low = vget_high_u16(v0_1.val[0]);
        v3_low = vget_high_u16(v0_1.val[1]);
        v4_low = vget_high_u16(v0_2.val[0]);
        v5_low = vget_high_u16(v0_2.val[1]);
        v6_low = vget_high_u16(v0_3.val[0]);
        v7_low = vget_high_u16(v0_3.val[1]);
      }

      if (bh & 2) {
        vst1_lane_u32((void*) o7, vreinterpret_u32_u16(v7_low), 0); o7 += 2;
        vst1_lane_u32((void*) o6, vreinterpret_u32_u16(v6_low), 0); o6 += 2;
        vst1_lane_u32((void*) o5, vreinterpret_u32_u16(v5_low), 0); o5 += 2;
        vst1_lane_u32((void*) o4, vreinterpret_u32_u16(v4_low), 0); o4 += 2;
        vst1_lane_u32((void*) o3, vreinterpret_u32_u16(v3_low), 0); o3 += 2;
        vst1_lane_u32((void*) o2, vreinterpret_u32_u16(v2_low), 0); o2 += 2;
        vst1_lane_u32((void*) o1, vreinterpret_u32_u16(v1_low), 0); o1 += 2;
        vst1_lane_u32((void*) o0, vreinterpret_u32_u16(v0_low), 0); o0 += 2;
        v0_low = vext_u16(v0_low, v0_low, 2);
        v1_low = vext_u16(v1_low, v1_low, 2);
        v2_low = vext_u16(v2_low, v2_low, 2);
        v3_low = vext_u16(v3_low, v3_low, 2);
        v4_low = vext_u16(v4_low, v4_low, 2);
        v5_low = vext_u16(v5_low, v5_low, 2);
        v6_low = vext_u16(v6_low, v6_low, 2);
        v7_low = vext_u16(v7_low, v7_low, 2);
      }
      if (bh & 1) {
        vst1_lane_u16(o7, v7_low, 0);
        vst1_lane_u16(o6, v6_low, 0);
        vst1_lane_u16(o5, v5_low, 0);
        vst1_lane_u16(o4, v4_low, 0);
        vst1_lane_u16(o3, v3_low, 0);
        vst1_lane_u16(o2, v2_low, 0);
        vst1_lane_u16(o1, v1_low, 0);
        vst1_lane_u16(o0, v0_low, 0);
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
