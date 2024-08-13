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

void xnn_x16_transposec_ukernel__4x4_multi_dec_zip_neon(
    const uint16_t* input,
    uint16_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint16_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint16_t));

  const size_t tile_height = 4;
  const size_t tile_width = 4;
  const size_t tile_hbytes = tile_height * sizeof(uint16_t);
  const size_t tile_wbytes = tile_width * sizeof(uint16_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t input_offset = tile_height * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint16_t) - tile_hbytes;

  const uint16_t* i0 = input;
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
  const uint16_t* i2 = (const uint16_t*) ((uintptr_t) i1 + input_stride);
  const uint16_t* i3 = (const uint16_t*) ((uintptr_t) i2 + input_stride);
  uint16_t* o = (uint16_t*) ((uintptr_t) output - tile_hbytes);
  const size_t minus_output_stride = -output_stride;

  do {
    const size_t rem = min(block_width - 1, 3);
    const size_t oN_stride = rem * output_stride;
    const size_t oN_offset = oN_stride + tile_hbytes;
    size_t bh = block_height;
    for (; bh >= 4; bh -= 4) {
      const uint16x4_t v2_0 = vld1_u16(i0); i0 = (uint16_t*) ((uintptr_t) i0 + input_offset);
      const uint16x4_t v2_1 = vld1_u16(i1); i1 = (uint16_t*) ((uintptr_t) i1 + input_offset);
      const uint16x4_t v2_2 = vld1_u16(i2); i2 = (uint16_t*) ((uintptr_t) i2 + input_offset);
      const uint16x4_t v2_3 = vld1_u16(i3); i3 = (uint16_t*) ((uintptr_t) i3 + input_offset);

      const uint16x4x2_t v1_0 = vzip_u16(v2_0, v2_2);
      const uint16x4x2_t v1_1 = vzip_u16(v2_1, v2_3);

      const uint16x4x2_t v0_0 = vzip_u16(v1_0.val[0], v1_1.val[0]);
      const uint16x4x2_t v0_1 = vzip_u16(v1_0.val[1], v1_1.val[1]);

      o = (uint16_t*) ((uintptr_t) o + oN_offset);
      vst1_u16(o, v0_1.val[1]);
      if XNN_UNPREDICTABLE(block_width > 3) {
        o = (uint16_t*) ((uintptr_t) o + minus_output_stride);
      }
      vst1_u16(o, v0_1.val[0]);
      if XNN_UNPREDICTABLE(block_width >= 3) {
        o = (uint16_t*) ((uintptr_t) o + minus_output_stride);
      }
      vst1_u16(o, v0_0.val[1]);
      if XNN_UNPREDICTABLE(block_width > 1) {
        o = (uint16_t*) ((uintptr_t) o + minus_output_stride);
      }
      vst1_u16(o, v0_0.val[0]);
    }
    o = (uint16_t*) ((uintptr_t) o + tile_hbytes);

    if (bh != 0) {
      const uint16x4_t v2_0 = vld1_u16(i0);
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const uint16x4_t v2_1 = vld1_u16(i1);
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i0;
      }
      const uint16x4_t v2_2 = vld1_u16(i2);
      const uint16x4_t v2_3 = vmov_n_u16(0);

      const uint16x4x2_t v1_0 = vzip_u16(v2_0, v2_2);
      const uint16x4x2_t v1_1 = vzip_u16(v2_1, v2_3);

      const uint16x4x2_t v0_0 = vzip_u16(v1_0.val[0], v1_1.val[0]);
      const uint16x4x2_t v0_1 = vzip_u16(v1_0.val[1], v1_1.val[1]);

      uint16x4_t v0_low = v0_0.val[0];
      uint16x4_t v1_low = v0_0.val[1];
      uint16x4_t v2_low = v0_1.val[0];
      uint16x4_t v3_low = v0_1.val[1];

      if (bh & 2) {
        o = (uint16_t*) ((uintptr_t) o + oN_stride);
        vst1_lane_u32((void*) o, vreinterpret_u32_u16(v3_low), 0);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        }
        vst1_lane_u32((void*) o, vreinterpret_u32_u16(v2_low), 0);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        }
        vst1_lane_u32((void*) o, vreinterpret_u32_u16(v1_low), 0);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        }
        vst1_lane_u32((void*) o, vreinterpret_u32_u16(v0_low), 0); o += 2;
        v0_low = vext_u16(v0_low, v0_low, 2);
        v1_low = vext_u16(v1_low, v1_low, 2);
        v2_low = vext_u16(v2_low, v2_low, 2);
        v3_low = vext_u16(v3_low, v3_low, 2);
      }
      if (bh & 1) {
        o = (uint16_t*) ((uintptr_t) o + oN_stride);
        vst1_lane_u16(o, v3_low, 0);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        }
        vst1_lane_u16(o, v2_low, 0);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        }
        vst1_lane_u16(o, v1_low, 0);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = (uint16_t*) ((uintptr_t) o + minus_output_stride);
        }
        vst1_lane_u16(o, v0_low, 0);
      }
    }

    i0 = (const uint16_t*) ((uintptr_t) i0 + input_reset);
    i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
    i2 = (const uint16_t*) ((uintptr_t) i1 + input_stride);
    i3 = (const uint16_t*) ((uintptr_t) i2 + input_stride);
    o = (uint16_t*) ((uintptr_t) o + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
