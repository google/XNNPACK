// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/transpose.h>

static const uint8_t pos0[8] = {0, 1, 2, 3, 8, 9, 10, 11};
static const uint8_t pos1[8] = {16, 17, 18, 19, 24, 25, 26, 27};
static const uint8_t pos2[8] = {4, 5, 6, 7, 12, 13, 14, 15};
static const uint8_t pos3[8] = {20, 21, 22, 23, 28, 29, 30, 31};

void xnn_x32_transpose_ukernel__4x4_neon_tbl(
    const uint32_t* input,
    uint32_t* output,
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
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_height * output_stride - round_down_po2(block_height, 2) * sizeof(uint32_t);
  const size_t load_bytes = sizeof(uint8x8_t);
  const size_t tile_stride = tile_height * input_stride - load_bytes;

  const uint8_t* i0 = (const uint8_t*) input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);

  uint8_t* o0 = (uint8_t*) output;
  uint8_t* o1 = (uint8_t*) ((uintptr_t) o0 + output_stride);
  uint8_t* o2 = (uint8_t*) ((uintptr_t) o1 + output_stride);
  uint8_t* o3 = (uint8_t*) ((uintptr_t) o2 + output_stride);

  const uint8x8_t vperm0 = vld1_u8(&pos0[0]);
  const uint8x8_t vperm1 = vld1_u8(&pos1[0]);
  const uint8x8_t vperm2 = vld1_u8(&pos2[0]);
  const uint8x8_t vperm3 = vld1_u8(&pos3[0]);

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
      uint8x8x4_t v0, v1;
      v0.val[0] = vld1_u8(i0);
      i0 = (const uint8_t*) ((uintptr_t) i0 + load_bytes);
      v0.val[1] = vld1_u8(i1);
      i1 = (const uint8_t*) ((uintptr_t) i1 + load_bytes);
      v0.val[2] = vld1_u8(i2);
      i2 = (const uint8_t*) ((uintptr_t) i2 + load_bytes);
      v0.val[3] = vld1_u8(i3);
      i3 = (const uint8_t*) ((uintptr_t) i3 + load_bytes);
      v1.val[0] = vld1_u8(i0);
      i0 = (const uint8_t*) ((uintptr_t) i0 + tile_stride);
      v1.val[1] = vld1_u8(i1);
      i1 = (const uint8_t*) ((uintptr_t) i1 + tile_stride);
      v1.val[2] = vld1_u8(i2);
      i2 = (const uint8_t*) ((uintptr_t) i2 + tile_stride);
      v1.val[3] = vld1_u8(i3);
      i3 = (const uint8_t*) ((uintptr_t) i3 + tile_stride);

      uint8x8_t vres0 = vtbl4_u8(v0, vperm0);
      uint8x8_t vres1 = vtbl4_u8(v0, vperm1);
      uint8x8_t vres2 = vtbl4_u8(v0, vperm2);
      uint8x8_t vres3 = vtbl4_u8(v0, vperm3);
      uint8x8_t vres4 = vtbl4_u8(v1, vperm0);
      uint8x8_t vres5 = vtbl4_u8(v1, vperm1);
      uint8x8_t vres6 = vtbl4_u8(v1, vperm2);
      uint8x8_t vres7 = vtbl4_u8(v1, vperm3);

      vst1_u8(o3, vres6);
      vst1_u8(o3 + 8, vres7);
      o3 = (uint8_t*) ((uintptr_t) o3 + tile_wbytes);
      vst1_u8(o2, vres4);
      vst1_u8(o2 + 8, vres5);
      o2 = (uint8_t*) ((uintptr_t) o2 + tile_wbytes);
      vst1_u8(o1, vres2);
      vst1_u8(o1 + 8, vres3);
      o1 = (uint8_t*) ((uintptr_t) o1 + tile_wbytes);
      vst1_u8(o0, vres0);
      vst1_u8(o0 + 8, vres1);
      o0 = (uint8_t*) ((uintptr_t) o0 + tile_wbytes);
    }

    if (bh != 0) {
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i0;
      }
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      uint8x8x4_t v0, v1;
      v0.val[0] = vld1_u8(i0);
      v0.val[1] = vld1_u8(i1);
      v0.val[2] = vld1_u8(i2);
      v1.val[0] = vld1_u8(i0 + 8);
      v1.val[1] = vld1_u8(i1 + 8);
      v1.val[2] = vld1_u8(i2 + 8);

      uint8x8_t vres0 = vtbl4_u8(v0, vperm0);
      uint8x8_t vres1 = vtbl4_u8(v0, vperm1);
      uint8x8_t vres2 = vtbl4_u8(v0, vperm2);
      uint8x8_t vres3 = vtbl4_u8(v0, vperm3);
      uint8x8_t vres4 = vtbl4_u8(v1, vperm0);
      uint8x8_t vres5 = vtbl4_u8(v1, vperm1);
      uint8x8_t vres6 = vtbl4_u8(v1, vperm2);
      uint8x8_t vres7 = vtbl4_u8(v1, vperm3);

      if (bh & 2) {
        vst1_u8(o3, vres6);
        o3 += 8;
        vst1_u8(o2, vres4);
        o2 += 8;
        vst1_u8(o1, vres2);
        o1 += 8;
        vst1_u8(o0, vres0);
        o0 += 8;
        vres0 = vres1;
        vres2 = vres3;
        vres4 = vres5;
        vres6 = vres7;
      }
      if (bh & 1) {
        vst1_lane_u32(o3, vreinterpret_u32_u8(vres6), 0);
        vst1_lane_u32(o2, vreinterpret_u32_u8(vres4), 0);
        vst1_lane_u32(o1, vreinterpret_u32_u8(vres2), 0);
        vst1_lane_u32(o0, vreinterpret_u32_u8(vres0), 0);
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
