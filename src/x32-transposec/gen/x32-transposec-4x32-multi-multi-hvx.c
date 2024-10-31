// Auto-generated file. Do not edit!
//   Template: src/x32-transposec/hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/simd/f32-hvx.h>

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/transpose.h"

void xnn_x32_transposec_ukernel__4x32_multi_multi_hvx(
    const uint32_t* input,
    uint32_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint32_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint32_t));

  const size_t tile_height = 4;
  const size_t tile_width = 32;
  const size_t tile_hbytes = tile_height * sizeof(uint32_t);
  const size_t tile_wbytes = tile_width * sizeof(uint32_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t input_offset = tile_height * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint32_t);

  const uint32_t* i0 = input;
  const uint32_t* i1 = (const uint32_t*) ((uintptr_t) i0 + input_stride);
  const uint32_t* i2 = (const uint32_t*) ((uintptr_t) i1 + input_stride);
  const uint32_t* i3 = (const uint32_t*) ((uintptr_t) i2 + input_stride);
  uint32_t* o0 = (uint32_t*) output;
  uint32_t* o1 = (uint32_t*) ((uintptr_t) o0 + output_stride);
  uint32_t* o2 = (uint32_t*) ((uintptr_t) o1 + output_stride);
  uint32_t* o3 = (uint32_t*) ((uintptr_t) o2 + output_stride);
  const size_t minus_output_stride = -output_stride;

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
      const HVX_Vector v2_0 = *((HVX_UVector *) i0); i0 = (uint32_t*) ((uintptr_t) i0 + input_offset);
      const HVX_Vector v2_1 = *((HVX_UVector *) i1); i1 = (uint32_t*) ((uintptr_t) i1 + input_offset);
      const HVX_Vector v2_2 = *((HVX_UVector *) i2); i2 = (uint32_t*) ((uintptr_t) i2 + input_offset);
      const HVX_Vector v2_3 = *((HVX_UVector *) i3); i3 = (uint32_t*) ((uintptr_t) i3 + input_offset);

      int rt = -4;
      const HVX_VectorPair v1_0 = Q6_W_vshuff_VVR(v2_1, v2_0, rt);
      const HVX_VectorPair v1_1 = Q6_W_vshuff_VVR(v2_3, v2_2, rt);

      rt = rt << 1;
      HVX_VectorPair v0_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_1), Q6_V_lo_W(v1_0), rt);
      HVX_VectorPair v0_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_1), Q6_V_hi_W(v1_0), rt);
      
      xnn_storeu_f32(o3, Q6_V_hi_W(v0_1)); o3 = (uint32_t*) ((uintptr_t) o3 + tile_hbytes);
      xnn_storeu_f32(o2, Q6_V_lo_W(v0_1)); o2 = (uint32_t*) ((uintptr_t) o2 + tile_hbytes);
      xnn_storeu_f32(o1, Q6_V_hi_W(v0_0)); o1 = (uint32_t*) ((uintptr_t) o1 + tile_hbytes);
      xnn_storeu_f32(o0, Q6_V_lo_W(v0_0)); o0 = (uint32_t*) ((uintptr_t) o0 + tile_hbytes);
    }


    i0 = (const uint32_t*) ((uintptr_t) i0 + input_reset);
    i1 = (const uint32_t*) ((uintptr_t) i0 + input_stride);
    i2 = (const uint32_t*) ((uintptr_t) i1 + input_stride);
    i3 = (const uint32_t*) ((uintptr_t) i2 + input_stride);
    o0 = (uint32_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint32_t*) ((uintptr_t) o1 + output_reset);
    o2 = (uint32_t*) ((uintptr_t) o2 + output_reset);
    o3 = (uint32_t*) ((uintptr_t) o3 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}


