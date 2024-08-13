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

void xnn_x32_transposec_ukernel__2x2_multi_multi_zip_neon(
    const uint32_t* input,
    uint32_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint32_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint32_t));

  const size_t tile_height = 2;
  const size_t tile_width = 2;
  const size_t tile_hbytes = tile_height * sizeof(uint32_t);
  const size_t tile_wbytes = tile_width * sizeof(uint32_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t input_offset = tile_height * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint32_t);

  const uint32_t* i0 = input;
  const uint32_t* i1 = (const uint32_t*) ((uintptr_t) i0 + input_stride);
  uint32_t* o0 = (uint32_t*) output;
  uint32_t* o1 = (uint32_t*) ((uintptr_t) o0 + output_stride);

  do {
    if XNN_UNPREDICTABLE(block_width < 2) {
      o1 = o0;
    }
    size_t bh = block_height;
    for (; bh >= 2; bh -= 2) {
      const uint32x2_t v1_0 = vld1_u32(i0); i0 = (uint32_t*) ((uintptr_t) i0 + input_offset);
      const uint32x2_t v1_1 = vld1_u32(i1); i1 = (uint32_t*) ((uintptr_t) i1 + input_offset);

      const uint32x2x2_t v0_0 = vzip_u32(v1_0, v1_1);


      vst1_u32(o1, v0_0.val[1]); o1 = (uint32_t*) ((uintptr_t) o1 + tile_hbytes);
      vst1_u32(o0, v0_0.val[0]); o0 = (uint32_t*) ((uintptr_t) o0 + tile_hbytes);
    }

    if (bh != 0) {
      const uint32x2_t v1_0 = vld1_u32(i0);
      const uint32x2_t v1_1 = vmov_n_u32(0);

      const uint32x2x2_t v0_0 = vzip_u32(v1_0, v1_1);


      uint32x2_t v0_low = v0_0.val[0];
      uint32x2_t v1_low = v0_0.val[1];

      if (bh & 1) {
        vst1_lane_u32(o1, v1_low, 0);
        vst1_lane_u32(o0, v0_low, 0);
      }
    }

    i0 = (const uint32_t*) ((uintptr_t) i0 + input_reset);
    i1 = (const uint32_t*) ((uintptr_t) i0 + input_stride);
    o0 = (uint32_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint32_t*) ((uintptr_t) o1 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
