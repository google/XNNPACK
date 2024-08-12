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

void xnn_x64_transposec_ukernel__2x2_reuse_mov_zip_neon(
    const uint64_t* input,
    uint64_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint64_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint64_t));

  const size_t tile_height = 2;
  const size_t tile_width = 2;
  const size_t tile_hbytes = tile_height * sizeof(uint64_t);
  const size_t tile_wbytes = tile_width * sizeof(uint64_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint64_t) - tile_hbytes;

  const uint64_t* i0 = input;
  uint64_t* o = (uint64_t*) ((uintptr_t) output - tile_hbytes);
  const size_t minus_output_stride = -output_stride;

  do {
    const size_t rem = min(block_width - 1, 1);
    const size_t oN_stride = rem * output_stride;
    const size_t oN_offset = oN_stride + tile_hbytes;
    size_t bh = block_height;
    for (; bh >= 2; bh -= 2) {
      const uint64x2_t v1_0 = vld1q_u64(i0); i0 = (uint64_t*) ((uintptr_t) i0 + input_stride);
      const uint64x2_t v1_1 = vld1q_u64(i0); i0 = (uint64_t*) ((uintptr_t) i0 + input_stride);

      uint64x2x2_t v0_0;
      #if XNN_ARCH_ARM64
        v0_0.val[0] = vzip1q_u64(v1_0, v1_1);
        v0_0.val[1] = vzip2q_u64(v1_0, v1_1);
      #else
        v0_0.val[0] = vcombine_u64(vget_low_u64(v1_0), vget_low_u64(v1_1));
        v0_0.val[1] = vcombine_u64(vget_high_u64(v1_0), vget_high_u64(v1_1));
      #endif


      o = (uint64_t*) ((uintptr_t) o + oN_offset);
      vst1q_u64(o, v0_0.val[1]);
      uint64_t *oN = (uint64_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 1) {
        o = oN;
      }
      vst1q_u64(o, v0_0.val[0]);
    }
    o = (uint64_t*) ((uintptr_t) o + tile_hbytes);

    if (bh != 0) {
      const uint64x2_t v1_0 = vld1q_u64(i0);
      const uint64x2_t v1_1 = vmovq_n_u64(0);

      uint64x2x2_t v0_0;
      #if XNN_ARCH_ARM64
        v0_0.val[0] = vzip1q_u64(v1_0, v1_1);
        v0_0.val[1] = vzip2q_u64(v1_0, v1_1);
      #else
        v0_0.val[0] = vcombine_u64(vget_low_u64(v1_0), vget_low_u64(v1_1));
        v0_0.val[1] = vcombine_u64(vget_high_u64(v1_0), vget_high_u64(v1_1));
      #endif


      uint64x1_t v0_low = vget_low_u64(v0_0.val[0]);
      uint64x1_t v1_low = vget_low_u64(v0_0.val[1]);

      if (bh & 1) {
        o = (uint64_t*) ((uintptr_t) o + oN_stride);
        vst1_u64(o, v1_low);
        uint64_t *oN = (uint64_t*) ((uintptr_t) o + minus_output_stride);
        if XNN_UNPREDICTABLE(block_width > 1) {
          o = oN;
        }
        vst1_u64(o, v0_low);
      }

    }

    i0 = (const uint64_t*) ((uintptr_t) i0 + input_reset);
    o = (uint64_t*) ((uintptr_t) o + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
