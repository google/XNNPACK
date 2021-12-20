// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/transpose.h>

void xnn_x32_transpose_ukernel__4x4_neon_zip(
    const uint32_t *input,
    uint32_t * output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height)
{
  assert(output_stride >= block_height * sizeof(uint32_t));
  assert(input_stride >= block_width * sizeof(uint32_t));
  const size_t tile_height = 4;
  const size_t tile_width = 4;
  const size_t tile_wbytes = tile_width * sizeof(uint32_t);
  const size_t input_reset = tile_wbytes - block_height * input_stride;
  const size_t output_reset = tile_height * output_stride - block_height * sizeof(uint32_t);
  size_t bw = block_width;
  size_t bh = block_height;

  const uint32_t *i0 = input;
  const uint32_t *i1 = (uint32_t*) ((uintptr_t) i0 + input_stride);
  const uint32_t *i2 = (uint32_t*) ((uintptr_t) i1 + input_stride);
  const uint32_t *i3 = (uint32_t*) ((uintptr_t) i2 + input_stride);

  uint32_t *o0 = (uint32_t*) output;
  uint32_t *o1 = (uint32_t*) ((uintptr_t) o0 + output_stride);
  uint32_t *o2 = (uint32_t*) ((uintptr_t) o1 + output_stride);
  uint32_t *o3 = (uint32_t*) ((uintptr_t) o2 + output_stride);

  do{
    if XNN_UNPREDICTABLE(bw < 2) {
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(bw <= 2) {
      o2 = o0;
    }
    if XNN_UNPREDICTABLE(bw < 4) {
      o3 = o0;
    }
    bh = block_height;
    for (; bh >= 4; bh -= 4){
      uint32x4_t v0 = vld1q_u32(i0);
      i0 = (uint32_t*) ((uintptr_t) i0 + tile_height * input_stride);
      uint32x4_t v1 = vld1q_u32(i1);
      i1 = (uint32_t*) ((uintptr_t) i1 + tile_height * input_stride);
      uint32x4_t v2 = vld1q_u32(i2);
      i2 = (uint32_t*) ((uintptr_t) i2 + tile_height * input_stride);
      uint32x4_t v3 = vld1q_u32(i3);
      i3 = (uint32_t*) ((uintptr_t) i3 + tile_height * input_stride);

      uint32x4x2_t vzip_02 =  vzipq_u32(v0, v2);
      uint32x4x2_t vzip_13 =  vzipq_u32(v1, v3);
      uint32x4x2_t vres_01 =  vzipq_u32(vzip_02.val[0], vzip_13.val[0]);
      uint32x4x2_t vres_23 =  vzipq_u32(vzip_02.val[1], vzip_13.val[1]);

      vst1q_u32(o3, vres_23.val[1]);
      o3 = (uint32_t*) ((uintptr_t) o3 + tile_wbytes);
      vst1q_u32(o2, vres_23.val[0]);
      o2 = (uint32_t*) ((uintptr_t) o2 + tile_wbytes);
      vst1q_u32(o1, vres_01.val[1]);
      o1 = (uint32_t*) ((uintptr_t) o1 + tile_wbytes);
      vst1q_u32(o0, vres_01.val[0]);
      o0 = (uint32_t*) ((uintptr_t) o0 + tile_wbytes);
    }

    if (bh != 0){
      if XNN_UNPREDICTABLE(bh < 4) {
        i3 = i0;
      }
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i0;
      }
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }

      uint32x4_t v0 = vld1q_u32(i0);
      uint32x4_t v1 = vld1q_u32(i1);
      uint32x4_t v2 = vld1q_u32(i2);
      uint32x4_t v3 = vld1q_u32(i3);

      uint32x4x2_t vzip_02 =  vzipq_u32(v0, v2);
      uint32x4x2_t vzip_13 =  vzipq_u32(v1, v3);
      uint32x4x2_t vres_01 =  vzipq_u32(vzip_02.val[0], vzip_13.val[0]);
      uint32x4x2_t vres_23 =  vzipq_u32(vzip_02.val[1], vzip_13.val[1]);

      uint32x2_t v0_low = vget_low_u32(vres_01.val[0]);
      uint32x2_t v1_low = vget_low_u32(vres_01.val[1]);
      uint32x2_t v2_low = vget_low_u32(vres_23.val[0]);
      uint32x2_t v3_low = vget_low_u32(vres_23.val[1]);
      if (bh & 2){
        vst1_u32(o3, v3_low);
        o3 = (uint32_t*) ((uintptr_t) o3 + 2 * sizeof(uint32_t));
        vst1_u32(o2, v2_low);
        o2 = (uint32_t*) ((uintptr_t) o2 + 2 * sizeof(uint32_t));
        vst1_u32(o1, v1_low);
        o1 = (uint32_t*) ((uintptr_t) o1 + 2 * sizeof(uint32_t));
        vst1_u32(o0, v0_low);
        o0 = (uint32_t*) ((uintptr_t) o0 + 2 * sizeof(uint32_t));
        v0_low = vget_high_u32(vres_01.val[0]);
        v1_low = vget_high_u32(vres_01.val[1]);
        v2_low = vget_high_u32(vres_23.val[0]);
        v3_low = vget_high_u32(vres_23.val[1]);
        i0 = (uint32_t*) ((uintptr_t) i0 + 2 * input_stride);
      }
      if (bh & 1){
        vst1_lane_u32(o3, v3_low, 0);
        vst1_lane_u32(o2, v2_low, 0);
        vst1_lane_u32(o1, v1_low, 0);
        vst1_lane_u32(o0, v0_low, 0);
        o0 = (uint32_t*) ((uintptr_t) o0 + sizeof(uint32_t));
        i0 = (uint32_t*) ((uintptr_t) i0 + input_stride);
      }
    }
    i0 = (uint32_t*) ((uintptr_t) i0 + input_reset);
    i1 = (uint32_t*) ((uintptr_t) i0 + input_stride);
    i2 = (uint32_t*) ((uintptr_t) i1 + input_stride);
    i3 = (uint32_t*) ((uintptr_t) i2 + input_stride);
    o0 = (uint32_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint32_t*) ((uintptr_t) o0 + output_stride);
    o2 = (uint32_t*) ((uintptr_t) o1 + output_stride);
    o3 = (uint32_t*) ((uintptr_t) o2 + output_stride);
    bw = doz(bw, tile_width);
  } while (bw != 0);
}
