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

void xnn_x8_transposec_ukernel__16x16_reuse_mov_zip_neon(
    const uint8_t* input,
    uint8_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint8_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint8_t));

  const size_t tile_height = 16;
  const size_t tile_width = 16;
  const size_t tile_hbytes = tile_height * sizeof(uint8_t);
  const size_t tile_wbytes = tile_width * sizeof(uint8_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint8_t) - tile_hbytes;

  const uint8_t* i0 = input;
  uint8_t* o = (uint8_t*) ((uintptr_t) output - tile_hbytes);
  const size_t minus_output_stride = -output_stride;

  do {
    const size_t rem = min(block_width - 1, 15);
    const size_t oN_stride = rem * output_stride;
    const size_t oN_offset = oN_stride + tile_hbytes;
    size_t bh = block_height;
    for (; bh >= 16; bh -= 16) {
      const uint8x16_t v4_0 = vld1q_u8(i0); i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const uint8x16_t v4_1 = vld1q_u8(i0); i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const uint8x16_t v4_2 = vld1q_u8(i0); i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const uint8x16_t v4_3 = vld1q_u8(i0); i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const uint8x16_t v4_4 = vld1q_u8(i0); i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const uint8x16_t v4_5 = vld1q_u8(i0); i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const uint8x16_t v4_6 = vld1q_u8(i0); i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const uint8x16_t v4_7 = vld1q_u8(i0); i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const uint8x16_t v4_8 = vld1q_u8(i0); i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const uint8x16_t v4_9 = vld1q_u8(i0); i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const uint8x16_t v4_10 = vld1q_u8(i0); i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const uint8x16_t v4_11 = vld1q_u8(i0); i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const uint8x16_t v4_12 = vld1q_u8(i0); i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const uint8x16_t v4_13 = vld1q_u8(i0); i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const uint8x16_t v4_14 = vld1q_u8(i0); i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const uint8x16_t v4_15 = vld1q_u8(i0); i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);

      const uint8x16x2_t v3_0 = vzipq_u8(v4_0, v4_8);
      const uint8x16x2_t v3_1 = vzipq_u8(v4_1, v4_9);
      const uint8x16x2_t v3_2 = vzipq_u8(v4_2, v4_10);
      const uint8x16x2_t v3_3 = vzipq_u8(v4_3, v4_11);
      const uint8x16x2_t v3_4 = vzipq_u8(v4_4, v4_12);
      const uint8x16x2_t v3_5 = vzipq_u8(v4_5, v4_13);
      const uint8x16x2_t v3_6 = vzipq_u8(v4_6, v4_14);
      const uint8x16x2_t v3_7 = vzipq_u8(v4_7, v4_15);

      const uint8x16x2_t v2_0 = vzipq_u8(v3_0.val[0], v3_4.val[0]);
      const uint8x16x2_t v2_1 = vzipq_u8(v3_0.val[1], v3_4.val[1]);
      const uint8x16x2_t v2_2 = vzipq_u8(v3_1.val[0], v3_5.val[0]);
      const uint8x16x2_t v2_3 = vzipq_u8(v3_1.val[1], v3_5.val[1]);
      const uint8x16x2_t v2_4 = vzipq_u8(v3_2.val[0], v3_6.val[0]);
      const uint8x16x2_t v2_5 = vzipq_u8(v3_2.val[1], v3_6.val[1]);
      const uint8x16x2_t v2_6 = vzipq_u8(v3_3.val[0], v3_7.val[0]);
      const uint8x16x2_t v2_7 = vzipq_u8(v3_3.val[1], v3_7.val[1]);
      const uint8x16x2_t v1_0 = vzipq_u8(v2_0.val[0], v2_4.val[0]);
      const uint8x16x2_t v1_1 = vzipq_u8(v2_0.val[1], v2_4.val[1]);
      const uint8x16x2_t v1_2 = vzipq_u8(v2_1.val[0], v2_5.val[0]);
      const uint8x16x2_t v1_3 = vzipq_u8(v2_1.val[1], v2_5.val[1]);
      const uint8x16x2_t v1_4 = vzipq_u8(v2_2.val[0], v2_6.val[0]);
      const uint8x16x2_t v1_5 = vzipq_u8(v2_2.val[1], v2_6.val[1]);
      const uint8x16x2_t v1_6 = vzipq_u8(v2_3.val[0], v2_7.val[0]);
      const uint8x16x2_t v1_7 = vzipq_u8(v2_3.val[1], v2_7.val[1]);
      const uint8x16x2_t v0_0 = vzipq_u8(v1_0.val[0], v1_4.val[0]);
      const uint8x16x2_t v0_1 = vzipq_u8(v1_0.val[1], v1_4.val[1]);
      const uint8x16x2_t v0_2 = vzipq_u8(v1_1.val[0], v1_5.val[0]);
      const uint8x16x2_t v0_3 = vzipq_u8(v1_1.val[1], v1_5.val[1]);
      const uint8x16x2_t v0_4 = vzipq_u8(v1_2.val[0], v1_6.val[0]);
      const uint8x16x2_t v0_5 = vzipq_u8(v1_2.val[1], v1_6.val[1]);
      const uint8x16x2_t v0_6 = vzipq_u8(v1_3.val[0], v1_7.val[0]);
      const uint8x16x2_t v0_7 = vzipq_u8(v1_3.val[1], v1_7.val[1]);

      o = (uint8_t*) ((uintptr_t) o + oN_offset);
      vst1q_u8(o, v0_7.val[1]);
      uint8_t *oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 15) {
        o = oN;
      }
      vst1q_u8(o, v0_7.val[0]);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 15) {
        o = oN;
      }
      vst1q_u8(o, v0_6.val[1]);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 13) {
        o = oN;
      }
      vst1q_u8(o, v0_6.val[0]);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 13) {
        o = oN;
      }
      vst1q_u8(o, v0_5.val[1]);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 11) {
        o = oN;
      }
      vst1q_u8(o, v0_5.val[0]);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 11) {
        o = oN;
      }
      vst1q_u8(o, v0_4.val[1]);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 9) {
        o = oN;
      }
      vst1q_u8(o, v0_4.val[0]);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 9) {
        o = oN;
      }
      vst1q_u8(o, v0_3.val[1]);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 7) {
        o = oN;
      }
      vst1q_u8(o, v0_3.val[0]);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 7) {
        o = oN;
      }
      vst1q_u8(o, v0_2.val[1]);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 5) {
        o = oN;
      }
      vst1q_u8(o, v0_2.val[0]);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 5) {
        o = oN;
      }
      vst1q_u8(o, v0_1.val[1]);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 3) {
        o = oN;
      }
      vst1q_u8(o, v0_1.val[0]);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 3) {
        o = oN;
      }
      vst1q_u8(o, v0_0.val[1]);
      oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 1) {
        o = oN;
      }
      vst1q_u8(o, v0_0.val[0]);
    }
    o = (uint8_t*) ((uintptr_t) o + tile_hbytes);

    if (bh != 0) {
      const uint8x16_t v4_0 = vld1q_u8(i0);
      const uint8_t *i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const uint8x16_t v4_1 = vld1q_u8(i1);
      const uint8_t *i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i1;
      }
      const uint8x16_t v4_2 = vld1q_u8(i2);
      const uint8_t *i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
      if XNN_UNPREDICTABLE(bh < 4) {
        i3 = i2;
      }
      const uint8x16_t v4_3 = vld1q_u8(i3);
      const uint8_t *i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 4) {
        i4 = i3;
      }
      const uint8x16_t v4_4 = vld1q_u8(i4);
      const uint8_t *i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
      if XNN_UNPREDICTABLE(bh < 6) {
        i5 = i4;
      }
      const uint8x16_t v4_5 = vld1q_u8(i5);
      const uint8_t *i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 6) {
        i6 = i5;
      }
      const uint8x16_t v4_6 = vld1q_u8(i6);
      const uint8_t *i7 = (const uint8_t*) ((uintptr_t) i6 + input_stride);
      if XNN_UNPREDICTABLE(bh < 8) {
        i7 = i6;
      }
      const uint8x16_t v4_7 = vld1q_u8(i7);
      const uint8_t *i8 = (const uint8_t*) ((uintptr_t) i7 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 8) {
        i8 = i7;
      }
      const uint8x16_t v4_8 = vld1q_u8(i8);
      const uint8_t *i9 = (const uint8_t*) ((uintptr_t) i8 + input_stride);
      if XNN_UNPREDICTABLE(bh < 10) {
        i9 = i8;
      }
      const uint8x16_t v4_9 = vld1q_u8(i9);
      const uint8_t *i10 = (const uint8_t*) ((uintptr_t) i9 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 10) {
        i10 = i9;
      }
      const uint8x16_t v4_10 = vld1q_u8(i10);
      const uint8_t *i11 = (const uint8_t*) ((uintptr_t) i10 + input_stride);
      if XNN_UNPREDICTABLE(bh < 12) {
        i11 = i10;
      }
      const uint8x16_t v4_11 = vld1q_u8(i11);
      const uint8_t *i12 = (const uint8_t*) ((uintptr_t) i11 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 12) {
        i12 = i11;
      }
      const uint8x16_t v4_12 = vld1q_u8(i12);
      const uint8_t *i13 = (const uint8_t*) ((uintptr_t) i12 + input_stride);
      if XNN_UNPREDICTABLE(bh < 14) {
        i13 = i12;
      }
      const uint8x16_t v4_13 = vld1q_u8(i13);
      const uint8_t *i14 = (const uint8_t*) ((uintptr_t) i13 + input_stride);
      if XNN_UNPREDICTABLE(bh <= 14) {
        i14 = i13;
      }
      const uint8x16_t v4_14 = vld1q_u8(i14);
      const uint8x16_t v4_15 = vmovq_n_u8(0);

      const uint8x16x2_t v3_0 = vzipq_u8(v4_0, v4_8);
      const uint8x16x2_t v3_1 = vzipq_u8(v4_1, v4_9);
      const uint8x16x2_t v3_2 = vzipq_u8(v4_2, v4_10);
      const uint8x16x2_t v3_3 = vzipq_u8(v4_3, v4_11);
      const uint8x16x2_t v3_4 = vzipq_u8(v4_4, v4_12);
      const uint8x16x2_t v3_5 = vzipq_u8(v4_5, v4_13);
      const uint8x16x2_t v3_6 = vzipq_u8(v4_6, v4_14);
      const uint8x16x2_t v3_7 = vzipq_u8(v4_7, v4_15);

      const uint8x16x2_t v2_0 = vzipq_u8(v3_0.val[0], v3_4.val[0]);
      const uint8x16x2_t v2_1 = vzipq_u8(v3_0.val[1], v3_4.val[1]);
      const uint8x16x2_t v2_2 = vzipq_u8(v3_1.val[0], v3_5.val[0]);
      const uint8x16x2_t v2_3 = vzipq_u8(v3_1.val[1], v3_5.val[1]);
      const uint8x16x2_t v2_4 = vzipq_u8(v3_2.val[0], v3_6.val[0]);
      const uint8x16x2_t v2_5 = vzipq_u8(v3_2.val[1], v3_6.val[1]);
      const uint8x16x2_t v2_6 = vzipq_u8(v3_3.val[0], v3_7.val[0]);
      const uint8x16x2_t v2_7 = vzipq_u8(v3_3.val[1], v3_7.val[1]);
      const uint8x16x2_t v1_0 = vzipq_u8(v2_0.val[0], v2_4.val[0]);
      const uint8x16x2_t v1_1 = vzipq_u8(v2_0.val[1], v2_4.val[1]);
      const uint8x16x2_t v1_2 = vzipq_u8(v2_1.val[0], v2_5.val[0]);
      const uint8x16x2_t v1_3 = vzipq_u8(v2_1.val[1], v2_5.val[1]);
      const uint8x16x2_t v1_4 = vzipq_u8(v2_2.val[0], v2_6.val[0]);
      const uint8x16x2_t v1_5 = vzipq_u8(v2_2.val[1], v2_6.val[1]);
      const uint8x16x2_t v1_6 = vzipq_u8(v2_3.val[0], v2_7.val[0]);
      const uint8x16x2_t v1_7 = vzipq_u8(v2_3.val[1], v2_7.val[1]);
      const uint8x16x2_t v0_0 = vzipq_u8(v1_0.val[0], v1_4.val[0]);
      const uint8x16x2_t v0_1 = vzipq_u8(v1_0.val[1], v1_4.val[1]);
      const uint8x16x2_t v0_2 = vzipq_u8(v1_1.val[0], v1_5.val[0]);
      const uint8x16x2_t v0_3 = vzipq_u8(v1_1.val[1], v1_5.val[1]);
      const uint8x16x2_t v0_4 = vzipq_u8(v1_2.val[0], v1_6.val[0]);
      const uint8x16x2_t v0_5 = vzipq_u8(v1_2.val[1], v1_6.val[1]);
      const uint8x16x2_t v0_6 = vzipq_u8(v1_3.val[0], v1_7.val[0]);
      const uint8x16x2_t v0_7 = vzipq_u8(v1_3.val[1], v1_7.val[1]);

      uint8x8_t v0_low = vget_low_u8(v0_0.val[0]);
      uint8x8_t v1_low = vget_low_u8(v0_0.val[1]);
      uint8x8_t v2_low = vget_low_u8(v0_1.val[0]);
      uint8x8_t v3_low = vget_low_u8(v0_1.val[1]);
      uint8x8_t v4_low = vget_low_u8(v0_2.val[0]);
      uint8x8_t v5_low = vget_low_u8(v0_2.val[1]);
      uint8x8_t v6_low = vget_low_u8(v0_3.val[0]);
      uint8x8_t v7_low = vget_low_u8(v0_3.val[1]);
      uint8x8_t v8_low = vget_low_u8(v0_4.val[0]);
      uint8x8_t v9_low = vget_low_u8(v0_4.val[1]);
      uint8x8_t v10_low = vget_low_u8(v0_5.val[0]);
      uint8x8_t v11_low = vget_low_u8(v0_5.val[1]);
      uint8x8_t v12_low = vget_low_u8(v0_6.val[0]);
      uint8x8_t v13_low = vget_low_u8(v0_6.val[1]);
      uint8x8_t v14_low = vget_low_u8(v0_7.val[0]);
      uint8x8_t v15_low = vget_low_u8(v0_7.val[1]);

      if (bh & 8) {
        o = (uint8_t*) ((uintptr_t) o + oN_stride);
        vst1_u8(o, v15_low);
        uint8_t *oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        vst1_u8(o, v14_low);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        vst1_u8(o, v13_low);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        vst1_u8(o, v12_low);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        vst1_u8(o, v11_low);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        vst1_u8(o, v10_low);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        vst1_u8(o, v9_low);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        vst1_u8(o, v8_low);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        vst1_u8(o, v7_low);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        vst1_u8(o, v6_low);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        vst1_u8(o, v5_low);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        vst1_u8(o, v4_low);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        vst1_u8(o, v3_low);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        vst1_u8(o, v2_low);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        vst1_u8(o, v1_low);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        vst1_u8(o, v0_low); o += 8;
        v0_low = vget_high_u8(v0_0.val[0]);
        v1_low = vget_high_u8(v0_0.val[1]);
        v2_low = vget_high_u8(v0_1.val[0]);
        v3_low = vget_high_u8(v0_1.val[1]);
        v4_low = vget_high_u8(v0_2.val[0]);
        v5_low = vget_high_u8(v0_2.val[1]);
        v6_low = vget_high_u8(v0_3.val[0]);
        v7_low = vget_high_u8(v0_3.val[1]);
        v8_low = vget_high_u8(v0_4.val[0]);
        v9_low = vget_high_u8(v0_4.val[1]);
        v10_low = vget_high_u8(v0_5.val[0]);
        v11_low = vget_high_u8(v0_5.val[1]);
        v12_low = vget_high_u8(v0_6.val[0]);
        v13_low = vget_high_u8(v0_6.val[1]);
        v14_low = vget_high_u8(v0_7.val[0]);
        v15_low = vget_high_u8(v0_7.val[1]);
      }

      if (bh & 4) {
        o = (uint8_t*) ((uintptr_t) o + oN_stride);
        vst1_lane_u32((void*) o, vreinterpret_u32_u8(v15_low), 0);
        uint8_t *oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        vst1_lane_u32((void*) o, vreinterpret_u32_u8(v14_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        vst1_lane_u32((void*) o, vreinterpret_u32_u8(v13_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        vst1_lane_u32((void*) o, vreinterpret_u32_u8(v12_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        vst1_lane_u32((void*) o, vreinterpret_u32_u8(v11_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        vst1_lane_u32((void*) o, vreinterpret_u32_u8(v10_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        vst1_lane_u32((void*) o, vreinterpret_u32_u8(v9_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        vst1_lane_u32((void*) o, vreinterpret_u32_u8(v8_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        vst1_lane_u32((void*) o, vreinterpret_u32_u8(v7_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        vst1_lane_u32((void*) o, vreinterpret_u32_u8(v6_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        vst1_lane_u32((void*) o, vreinterpret_u32_u8(v5_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        vst1_lane_u32((void*) o, vreinterpret_u32_u8(v4_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        vst1_lane_u32((void*) o, vreinterpret_u32_u8(v3_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        vst1_lane_u32((void*) o, vreinterpret_u32_u8(v2_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        vst1_lane_u32((void*) o, vreinterpret_u32_u8(v1_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        vst1_lane_u32((void*) o, vreinterpret_u32_u8(v0_low), 0); o += 4;
        v0_low = vext_u8(v0_low, v0_low, 4);
        v1_low = vext_u8(v1_low, v1_low, 4);
        v2_low = vext_u8(v2_low, v2_low, 4);
        v3_low = vext_u8(v3_low, v3_low, 4);
        v4_low = vext_u8(v4_low, v4_low, 4);
        v5_low = vext_u8(v5_low, v5_low, 4);
        v6_low = vext_u8(v6_low, v6_low, 4);
        v7_low = vext_u8(v7_low, v7_low, 4);
        v8_low = vext_u8(v8_low, v8_low, 4);
        v9_low = vext_u8(v9_low, v9_low, 4);
        v10_low = vext_u8(v10_low, v10_low, 4);
        v11_low = vext_u8(v11_low, v11_low, 4);
        v12_low = vext_u8(v12_low, v12_low, 4);
        v13_low = vext_u8(v13_low, v13_low, 4);
        v14_low = vext_u8(v14_low, v14_low, 4);
        v15_low = vext_u8(v15_low, v15_low, 4);
      }
      if (bh & 2) {
        o = (uint8_t*) ((uintptr_t) o + oN_stride);
        vst1_lane_u16((void*) o, vreinterpret_u16_u8(v15_low), 0);
        uint8_t *oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        vst1_lane_u16((void*) o, vreinterpret_u16_u8(v14_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        vst1_lane_u16((void*) o, vreinterpret_u16_u8(v13_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        vst1_lane_u16((void*) o, vreinterpret_u16_u8(v12_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        vst1_lane_u16((void*) o, vreinterpret_u16_u8(v11_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        vst1_lane_u16((void*) o, vreinterpret_u16_u8(v10_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        vst1_lane_u16((void*) o, vreinterpret_u16_u8(v9_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        vst1_lane_u16((void*) o, vreinterpret_u16_u8(v8_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        vst1_lane_u16((void*) o, vreinterpret_u16_u8(v7_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        vst1_lane_u16((void*) o, vreinterpret_u16_u8(v6_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        vst1_lane_u16((void*) o, vreinterpret_u16_u8(v5_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        vst1_lane_u16((void*) o, vreinterpret_u16_u8(v4_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        vst1_lane_u16((void*) o, vreinterpret_u16_u8(v3_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        vst1_lane_u16((void*) o, vreinterpret_u16_u8(v2_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        vst1_lane_u16((void*) o, vreinterpret_u16_u8(v1_low), 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        vst1_lane_u16((void*) o, vreinterpret_u16_u8(v0_low), 0); o += 2;
        v0_low = vext_u8(v0_low, v0_low, 2);
        v1_low = vext_u8(v1_low, v1_low, 2);
        v2_low = vext_u8(v2_low, v2_low, 2);
        v3_low = vext_u8(v3_low, v3_low, 2);
        v4_low = vext_u8(v4_low, v4_low, 2);
        v5_low = vext_u8(v5_low, v5_low, 2);
        v6_low = vext_u8(v6_low, v6_low, 2);
        v7_low = vext_u8(v7_low, v7_low, 2);
        v8_low = vext_u8(v8_low, v8_low, 2);
        v9_low = vext_u8(v9_low, v9_low, 2);
        v10_low = vext_u8(v10_low, v10_low, 2);
        v11_low = vext_u8(v11_low, v11_low, 2);
        v12_low = vext_u8(v12_low, v12_low, 2);
        v13_low = vext_u8(v13_low, v13_low, 2);
        v14_low = vext_u8(v14_low, v14_low, 2);
        v15_low = vext_u8(v15_low, v15_low, 2);
      }
      if (bh & 1) {
        o = (uint8_t*) ((uintptr_t) o + oN_stride);
        vst1_lane_u8(o, v15_low, 0);
        uint8_t *oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 15) {
          o = oN;
        }
        vst1_lane_u8(o, v14_low, 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 15) {
          o = oN;
        }
        vst1_lane_u8(o, v13_low, 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 13) {
          o = oN;
        }
        vst1_lane_u8(o, v12_low, 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 13) {
          o = oN;
        }
        vst1_lane_u8(o, v11_low, 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 11) {
          o = oN;
        }
        vst1_lane_u8(o, v10_low, 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 11) {
          o = oN;
        }
        vst1_lane_u8(o, v9_low, 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 9) {
          o = oN;
        }
        vst1_lane_u8(o, v8_low, 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 9) {
          o = oN;
        }
        vst1_lane_u8(o, v7_low, 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 7) {
          o = oN;
        }
        vst1_lane_u8(o, v6_low, 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 7) {
          o = oN;
        }
        vst1_lane_u8(o, v5_low, 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 5) {
          o = oN;
        }
        vst1_lane_u8(o, v4_low, 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 5) {
          o = oN;
        }
        vst1_lane_u8(o, v3_low, 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 3) {
          o = oN;
        }
        vst1_lane_u8(o, v2_low, 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width >= 3) {
          o = oN;
        }
        vst1_lane_u8(o, v1_low, 0);
        oN = (uint8_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        vst1_lane_u8(o, v0_low, 0);
      }
    }

    i0 = (const uint8_t*) ((uintptr_t) i0 + input_reset);
    o = (uint8_t*) ((uintptr_t) o + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
