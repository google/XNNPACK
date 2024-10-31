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

void xnn_x32_transposec_ukernel__16x32_multi_multi_hvx(
    const uint32_t* input,
    uint32_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint32_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint32_t));

  const size_t tile_height = 16;
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
  const uint32_t* i4 = (const uint32_t*) ((uintptr_t) i3 + input_stride);
  const uint32_t* i5 = (const uint32_t*) ((uintptr_t) i4 + input_stride);
  const uint32_t* i6 = (const uint32_t*) ((uintptr_t) i5 + input_stride);
  const uint32_t* i7 = (const uint32_t*) ((uintptr_t) i6 + input_stride);
  const uint32_t* i8 = (const uint32_t*) ((uintptr_t) i7 + input_stride);
  const uint32_t* i9 = (const uint32_t*) ((uintptr_t) i8 + input_stride);
  const uint32_t* i10 = (const uint32_t*) ((uintptr_t) i9 + input_stride);
  const uint32_t* i11 = (const uint32_t*) ((uintptr_t) i10 + input_stride);
  const uint32_t* i12 = (const uint32_t*) ((uintptr_t) i11 + input_stride);
  const uint32_t* i13 = (const uint32_t*) ((uintptr_t) i12 + input_stride);
  const uint32_t* i14 = (const uint32_t*) ((uintptr_t) i13 + input_stride);
  const uint32_t* i15 = (const uint32_t*) ((uintptr_t) i14 + input_stride);
  uint32_t* o0 = (uint32_t*) output;
  uint32_t* o1 = (uint32_t*) ((uintptr_t) o0 + output_stride);
  uint32_t* o2 = (uint32_t*) ((uintptr_t) o1 + output_stride);
  uint32_t* o3 = (uint32_t*) ((uintptr_t) o2 + output_stride);
  uint32_t* o4 = (uint32_t*) ((uintptr_t) o3 + output_stride);
  uint32_t* o5 = (uint32_t*) ((uintptr_t) o4 + output_stride);
  uint32_t* o6 = (uint32_t*) ((uintptr_t) o5 + output_stride);
  uint32_t* o7 = (uint32_t*) ((uintptr_t) o6 + output_stride);
  uint32_t* o8 = (uint32_t*) ((uintptr_t) o7 + output_stride);
  uint32_t* o9 = (uint32_t*) ((uintptr_t) o8 + output_stride);
  uint32_t* o10 = (uint32_t*) ((uintptr_t) o9 + output_stride);
  uint32_t* o11 = (uint32_t*) ((uintptr_t) o10 + output_stride);
  uint32_t* o12 = (uint32_t*) ((uintptr_t) o11 + output_stride);
  uint32_t* o13 = (uint32_t*) ((uintptr_t) o12 + output_stride);
  uint32_t* o14 = (uint32_t*) ((uintptr_t) o13 + output_stride);
  uint32_t* o15 = (uint32_t*) ((uintptr_t) o14 + output_stride);
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
    if XNN_UNPREDICTABLE(block_width <= 8) {
      o8 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 10) {
      o9 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 10) {
      o10 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 12) {
      o11 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 12) {
      o12 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 14) {
      o13 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 14) {
      o14 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 16) {
      o15 = o0;
    }
    size_t bh = block_height;
    for (; bh >= 16; bh -= 16) {
      const HVX_Vector v4_0 = *((HVX_UVector *) i0); i0 = (uint32_t*) ((uintptr_t) i0 + input_offset);
      const HVX_Vector v4_1 = *((HVX_UVector *) i1); i1 = (uint32_t*) ((uintptr_t) i1 + input_offset);
      const HVX_Vector v4_2 = *((HVX_UVector *) i2); i2 = (uint32_t*) ((uintptr_t) i2 + input_offset);
      const HVX_Vector v4_3 = *((HVX_UVector *) i3); i3 = (uint32_t*) ((uintptr_t) i3 + input_offset);
      const HVX_Vector v4_4 = *((HVX_UVector *) i4); i4 = (uint32_t*) ((uintptr_t) i4 + input_offset);
      const HVX_Vector v4_5 = *((HVX_UVector *) i5); i5 = (uint32_t*) ((uintptr_t) i5 + input_offset);
      const HVX_Vector v4_6 = *((HVX_UVector *) i6); i6 = (uint32_t*) ((uintptr_t) i6 + input_offset);
      const HVX_Vector v4_7 = *((HVX_UVector *) i7); i7 = (uint32_t*) ((uintptr_t) i7 + input_offset);
      const HVX_Vector v4_8 = *((HVX_UVector *) i8); i8 = (uint32_t*) ((uintptr_t) i8 + input_offset);
      const HVX_Vector v4_9 = *((HVX_UVector *) i9); i9 = (uint32_t*) ((uintptr_t) i9 + input_offset);
      const HVX_Vector v4_10 = *((HVX_UVector *) i10); i10 = (uint32_t*) ((uintptr_t) i10 + input_offset);
      const HVX_Vector v4_11 = *((HVX_UVector *) i11); i11 = (uint32_t*) ((uintptr_t) i11 + input_offset);
      const HVX_Vector v4_12 = *((HVX_UVector *) i12); i12 = (uint32_t*) ((uintptr_t) i12 + input_offset);
      const HVX_Vector v4_13 = *((HVX_UVector *) i13); i13 = (uint32_t*) ((uintptr_t) i13 + input_offset);
      const HVX_Vector v4_14 = *((HVX_UVector *) i14); i14 = (uint32_t*) ((uintptr_t) i14 + input_offset);
      const HVX_Vector v4_15 = *((HVX_UVector *) i15); i15 = (uint32_t*) ((uintptr_t) i15 + input_offset);

      int rt = -4;
      const HVX_VectorPair v3_0 = Q6_W_vshuff_VVR(v4_1, v4_0, rt);
      const HVX_VectorPair v3_1 = Q6_W_vshuff_VVR(v4_3, v4_2, rt);
      const HVX_VectorPair v3_2 = Q6_W_vshuff_VVR(v4_5, v4_4, rt);
      const HVX_VectorPair v3_3 = Q6_W_vshuff_VVR(v4_7, v4_6, rt);
      const HVX_VectorPair v3_4 = Q6_W_vshuff_VVR(v4_9, v4_8, rt);
      const HVX_VectorPair v3_5 = Q6_W_vshuff_VVR(v4_11, v4_10, rt);
      const HVX_VectorPair v3_6 = Q6_W_vshuff_VVR(v4_13, v4_12, rt);
      const HVX_VectorPair v3_7 = Q6_W_vshuff_VVR(v4_15, v4_14, rt);

      rt = rt << 1;
      HVX_VectorPair v2_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_1), Q6_V_lo_W(v3_0), rt);
      HVX_VectorPair v2_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_1), Q6_V_hi_W(v3_0), rt);
      
      HVX_VectorPair v2_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_3), Q6_V_lo_W(v3_2), rt);
      HVX_VectorPair v2_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_3), Q6_V_hi_W(v3_2), rt);
      
      HVX_VectorPair v2_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_5), Q6_V_lo_W(v3_4), rt);
      HVX_VectorPair v2_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_5), Q6_V_hi_W(v3_4), rt);
      
      HVX_VectorPair v2_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_7), Q6_V_lo_W(v3_6), rt);
      HVX_VectorPair v2_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_7), Q6_V_hi_W(v3_6), rt);
      
      rt = rt << 1;
      HVX_VectorPair v1_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_2), Q6_V_lo_W(v2_0), rt);
      HVX_VectorPair v1_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_2), Q6_V_hi_W(v2_0), rt);
      
      HVX_VectorPair v1_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_3), Q6_V_lo_W(v2_1), rt);
      HVX_VectorPair v1_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_3), Q6_V_hi_W(v2_1), rt);
      
      HVX_VectorPair v1_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_6), Q6_V_lo_W(v2_4), rt);
      HVX_VectorPair v1_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_6), Q6_V_hi_W(v2_4), rt);
      
      HVX_VectorPair v1_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_7), Q6_V_lo_W(v2_5), rt);
      HVX_VectorPair v1_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_7), Q6_V_hi_W(v2_5), rt);
      
      rt = rt << 1;
      HVX_VectorPair v0_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_4), Q6_V_lo_W(v1_0), rt);
      HVX_VectorPair v0_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_4), Q6_V_hi_W(v1_0), rt);
      
      HVX_VectorPair v0_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_5), Q6_V_lo_W(v1_1), rt);
      HVX_VectorPair v0_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_5), Q6_V_hi_W(v1_1), rt);
      
      HVX_VectorPair v0_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_6), Q6_V_lo_W(v1_2), rt);
      HVX_VectorPair v0_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_6), Q6_V_hi_W(v1_2), rt);
      
      HVX_VectorPair v0_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_7), Q6_V_lo_W(v1_3), rt);
      HVX_VectorPair v0_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_7), Q6_V_hi_W(v1_3), rt);
      
      xnn_storeu_f32(o15, Q6_V_hi_W(v0_7)); o15 = (uint32_t*) ((uintptr_t) o15 + tile_hbytes);
      xnn_storeu_f32(o14, Q6_V_lo_W(v0_7)); o14 = (uint32_t*) ((uintptr_t) o14 + tile_hbytes);
      xnn_storeu_f32(o13, Q6_V_hi_W(v0_6)); o13 = (uint32_t*) ((uintptr_t) o13 + tile_hbytes);
      xnn_storeu_f32(o12, Q6_V_lo_W(v0_6)); o12 = (uint32_t*) ((uintptr_t) o12 + tile_hbytes);
      xnn_storeu_f32(o11, Q6_V_hi_W(v0_5)); o11 = (uint32_t*) ((uintptr_t) o11 + tile_hbytes);
      xnn_storeu_f32(o10, Q6_V_lo_W(v0_5)); o10 = (uint32_t*) ((uintptr_t) o10 + tile_hbytes);
      xnn_storeu_f32(o9, Q6_V_hi_W(v0_4)); o9 = (uint32_t*) ((uintptr_t) o9 + tile_hbytes);
      xnn_storeu_f32(o8, Q6_V_lo_W(v0_4)); o8 = (uint32_t*) ((uintptr_t) o8 + tile_hbytes);
      xnn_storeu_f32(o7, Q6_V_hi_W(v0_3)); o7 = (uint32_t*) ((uintptr_t) o7 + tile_hbytes);
      xnn_storeu_f32(o6, Q6_V_lo_W(v0_3)); o6 = (uint32_t*) ((uintptr_t) o6 + tile_hbytes);
      xnn_storeu_f32(o5, Q6_V_hi_W(v0_2)); o5 = (uint32_t*) ((uintptr_t) o5 + tile_hbytes);
      xnn_storeu_f32(o4, Q6_V_lo_W(v0_2)); o4 = (uint32_t*) ((uintptr_t) o4 + tile_hbytes);
      xnn_storeu_f32(o3, Q6_V_hi_W(v0_1)); o3 = (uint32_t*) ((uintptr_t) o3 + tile_hbytes);
      xnn_storeu_f32(o2, Q6_V_lo_W(v0_1)); o2 = (uint32_t*) ((uintptr_t) o2 + tile_hbytes);
      xnn_storeu_f32(o1, Q6_V_hi_W(v0_0)); o1 = (uint32_t*) ((uintptr_t) o1 + tile_hbytes);
      xnn_storeu_f32(o0, Q6_V_lo_W(v0_0)); o0 = (uint32_t*) ((uintptr_t) o0 + tile_hbytes);
    }


    i0 = (const uint32_t*) ((uintptr_t) i0 + input_reset);
    i1 = (const uint32_t*) ((uintptr_t) i0 + input_stride);
    i2 = (const uint32_t*) ((uintptr_t) i1 + input_stride);
    i3 = (const uint32_t*) ((uintptr_t) i2 + input_stride);
    i4 = (const uint32_t*) ((uintptr_t) i3 + input_stride);
    i5 = (const uint32_t*) ((uintptr_t) i4 + input_stride);
    i6 = (const uint32_t*) ((uintptr_t) i5 + input_stride);
    i7 = (const uint32_t*) ((uintptr_t) i6 + input_stride);
    i8 = (const uint32_t*) ((uintptr_t) i7 + input_stride);
    i9 = (const uint32_t*) ((uintptr_t) i8 + input_stride);
    i10 = (const uint32_t*) ((uintptr_t) i9 + input_stride);
    i11 = (const uint32_t*) ((uintptr_t) i10 + input_stride);
    i12 = (const uint32_t*) ((uintptr_t) i11 + input_stride);
    i13 = (const uint32_t*) ((uintptr_t) i12 + input_stride);
    i14 = (const uint32_t*) ((uintptr_t) i13 + input_stride);
    i15 = (const uint32_t*) ((uintptr_t) i14 + input_stride);
    o0 = (uint32_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint32_t*) ((uintptr_t) o1 + output_reset);
    o2 = (uint32_t*) ((uintptr_t) o2 + output_reset);
    o3 = (uint32_t*) ((uintptr_t) o3 + output_reset);
    o4 = (uint32_t*) ((uintptr_t) o4 + output_reset);
    o5 = (uint32_t*) ((uintptr_t) o5 + output_reset);
    o6 = (uint32_t*) ((uintptr_t) o6 + output_reset);
    o7 = (uint32_t*) ((uintptr_t) o7 + output_reset);
    o8 = (uint32_t*) ((uintptr_t) o8 + output_reset);
    o9 = (uint32_t*) ((uintptr_t) o9 + output_reset);
    o10 = (uint32_t*) ((uintptr_t) o10 + output_reset);
    o11 = (uint32_t*) ((uintptr_t) o11 + output_reset);
    o12 = (uint32_t*) ((uintptr_t) o12 + output_reset);
    o13 = (uint32_t*) ((uintptr_t) o13 + output_reset);
    o14 = (uint32_t*) ((uintptr_t) o14 + output_reset);
    o15 = (uint32_t*) ((uintptr_t) o15 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}


