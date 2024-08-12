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

void xnn_x16_transposec_ukernel__8x8_multi_switch_zip_neon(
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
  const size_t input_offset = tile_height * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint16_t);

  const uint16_t* i0 = input;
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
  const uint16_t* i2 = (const uint16_t*) ((uintptr_t) i1 + input_stride);
  const uint16_t* i3 = (const uint16_t*) ((uintptr_t) i2 + input_stride);
  const uint16_t* i4 = (const uint16_t*) ((uintptr_t) i3 + input_stride);
  const uint16_t* i5 = (const uint16_t*) ((uintptr_t) i4 + input_stride);
  const uint16_t* i6 = (const uint16_t*) ((uintptr_t) i5 + input_stride);
  const uint16_t* i7 = (const uint16_t*) ((uintptr_t) i6 + input_stride);
  uint16_t* o = (uint16_t*) output;
  const size_t minus_output_stride = -output_stride;

  do {
    const size_t rem = min(block_width - 1, 7);
    const size_t oN_stride = rem * output_stride;
    size_t bh = block_height;
    for (; bh >= 8; bh -= 8) {
      const uint16x8_t v3_0 = vld1q_u16(i0); i0 = (uint16_t*) ((uintptr_t) i0 + input_offset);
      const uint16x8_t v3_1 = vld1q_u16(i1); i1 = (uint16_t*) ((uintptr_t) i1 + input_offset);
      const uint16x8_t v3_2 = vld1q_u16(i2); i2 = (uint16_t*) ((uintptr_t) i2 + input_offset);
      const uint16x8_t v3_3 = vld1q_u16(i3); i3 = (uint16_t*) ((uintptr_t) i3 + input_offset);
      const uint16x8_t v3_4 = vld1q_u16(i4); i4 = (uint16_t*) ((uintptr_t) i4 + input_offset);
      const uint16x8_t v3_5 = vld1q_u16(i5); i5 = (uint16_t*) ((uintptr_t) i5 + input_offset);
      const uint16x8_t v3_6 = vld1q_u16(i6); i6 = (uint16_t*) ((uintptr_t) i6 + input_offset);
      const uint16x8_t v3_7 = vld1q_u16(i7); i7 = (uint16_t*) ((uintptr_t) i7 + input_offset);

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

      uint16_t *oN = (uint16_t*) ((uintptr_t) o + oN_stride);
      switch (rem) {
        case 7:
          vst1q_u16(oN, v0_3.val[1]); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 6:
          vst1q_u16(oN, v0_3.val[0]); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 5:
          vst1q_u16(oN, v0_2.val[1]); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 4:
          vst1q_u16(oN, v0_2.val[0]); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 3:
          vst1q_u16(oN, v0_1.val[1]); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 2:
          vst1q_u16(oN, v0_1.val[0]); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
        case 1:
          vst1q_u16(oN, v0_0.val[1]);
          XNN_FALLTHROUGH
        case 0:
          vst1q_u16(o, v0_0.val[0]); o = (uint16_t*) ((uintptr_t) o + tile_hbytes);
          break;
        default:
          XNN_UNREACHABLE;
      }
    }

    if (bh != 0) {
      const uint16x8_t v3_0 = vld1q_u16(i0);
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const uint16x8_t v3_1 = vld1q_u16(i1);
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i0;
      }
      const uint16x8_t v3_2 = vld1q_u16(i2);
      if XNN_UNPREDICTABLE(bh < 4) {
        i3 = i0;
      }
      const uint16x8_t v3_3 = vld1q_u16(i3);
      if XNN_UNPREDICTABLE(bh <= 4) {
        i4 = i0;
      }
      const uint16x8_t v3_4 = vld1q_u16(i4);
      if XNN_UNPREDICTABLE(bh < 6) {
        i5 = i0;
      }
      const uint16x8_t v3_5 = vld1q_u16(i5);
      if XNN_UNPREDICTABLE(bh <= 6) {
        i6 = i0;
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
        uint16_t* oN = (uint16_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 7:
            vst1_u16(oN, v7_low); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 6:
            vst1_u16(oN, v6_low); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 5:
            vst1_u16(oN, v5_low); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 4:
            vst1_u16(oN, v4_low); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 3:
            vst1_u16(oN, v3_low); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 2:
            vst1_u16(oN, v2_low); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 1:
            vst1_u16(oN, v1_low);
            XNN_FALLTHROUGH
          case 0:
            vst1_u16(o, v0_low); o += 4;
            break;
          default:
            XNN_UNREACHABLE;
        }
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
        uint16_t* oN = (uint16_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 7:
            vst1_lane_u32((void*) oN, vreinterpret_u32_u16(v7_low), 0); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
          case 6:
            vst1_lane_u32((void*) oN, vreinterpret_u32_u16(v6_low), 0); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
          case 5:
            vst1_lane_u32((void*) oN, vreinterpret_u32_u16(v5_low), 0); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
          case 4:
            vst1_lane_u32((void*) oN, vreinterpret_u32_u16(v4_low), 0); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
          case 3:
            vst1_lane_u32((void*) oN, vreinterpret_u32_u16(v3_low), 0); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
          case 2:
            vst1_lane_u32((void*) oN, vreinterpret_u32_u16(v2_low), 0); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
          XNN_FALLTHROUGH
          case 1:
            vst1_lane_u32((void*) oN, vreinterpret_u32_u16(v1_low), 0);
            XNN_FALLTHROUGH
          case 0:
            vst1_lane_u32((void*) o, vreinterpret_u32_u16(v0_low), 0); o += 2;
            break;
          default:
            XNN_UNREACHABLE;
        }
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
        uint16_t* oN = (uint16_t*) ((uintptr_t) o + oN_stride);
        switch (rem) {
          case 7:
            vst1_lane_u16(oN, v7_low, 0); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 6:
            vst1_lane_u16(oN, v6_low, 0); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 5:
            vst1_lane_u16(oN, v5_low, 0); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 4:
            vst1_lane_u16(oN, v4_low, 0); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 3:
            vst1_lane_u16(oN, v3_low, 0); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 2:
            vst1_lane_u16(oN, v2_low, 0); oN = (uint16_t*) ((uintptr_t) oN + minus_output_stride);
            XNN_FALLTHROUGH
          case 1:
            vst1_lane_u16(oN, v1_low, 0);
            XNN_FALLTHROUGH
          case 0:
            vst1_lane_u16(o, v0_low, 0);
            break;
          default:
            XNN_UNREACHABLE;
        }
      }
    }

    i0 = (const uint16_t*) ((uintptr_t) i0 + input_reset);
    i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
    i2 = (const uint16_t*) ((uintptr_t) i1 + input_stride);
    i3 = (const uint16_t*) ((uintptr_t) i2 + input_stride);
    i4 = (const uint16_t*) ((uintptr_t) i3 + input_stride);
    i5 = (const uint16_t*) ((uintptr_t) i4 + input_stride);
    i6 = (const uint16_t*) ((uintptr_t) i5 + input_stride);
    i7 = (const uint16_t*) ((uintptr_t) i6 + input_stride);
    o = (uint16_t*) ((uintptr_t) o + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
