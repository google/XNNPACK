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

void xnn_x32_transposec_ukernel__32x32_multi_mov_hvx(
    const uint32_t* input,
    uint32_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint32_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint32_t));

  const size_t tile_height = 32;
  const size_t tile_width = 32;
  const size_t tile_hbytes = tile_height * sizeof(uint32_t);
  const size_t tile_wbytes = tile_width * sizeof(uint32_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t input_offset = tile_height * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint32_t) - tile_hbytes;

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
  const uint32_t* i16 = (const uint32_t*) ((uintptr_t) i15 + input_stride);
  const uint32_t* i17 = (const uint32_t*) ((uintptr_t) i16 + input_stride);
  const uint32_t* i18 = (const uint32_t*) ((uintptr_t) i17 + input_stride);
  const uint32_t* i19 = (const uint32_t*) ((uintptr_t) i18 + input_stride);
  const uint32_t* i20 = (const uint32_t*) ((uintptr_t) i19 + input_stride);
  const uint32_t* i21 = (const uint32_t*) ((uintptr_t) i20 + input_stride);
  const uint32_t* i22 = (const uint32_t*) ((uintptr_t) i21 + input_stride);
  const uint32_t* i23 = (const uint32_t*) ((uintptr_t) i22 + input_stride);
  const uint32_t* i24 = (const uint32_t*) ((uintptr_t) i23 + input_stride);
  const uint32_t* i25 = (const uint32_t*) ((uintptr_t) i24 + input_stride);
  const uint32_t* i26 = (const uint32_t*) ((uintptr_t) i25 + input_stride);
  const uint32_t* i27 = (const uint32_t*) ((uintptr_t) i26 + input_stride);
  const uint32_t* i28 = (const uint32_t*) ((uintptr_t) i27 + input_stride);
  const uint32_t* i29 = (const uint32_t*) ((uintptr_t) i28 + input_stride);
  const uint32_t* i30 = (const uint32_t*) ((uintptr_t) i29 + input_stride);
  const uint32_t* i31 = (const uint32_t*) ((uintptr_t) i30 + input_stride);
  uint32_t* o = (uint32_t*) ((uintptr_t) output - tile_hbytes);
  const size_t minus_output_stride = -output_stride;

  do {
    const size_t rem = min(block_width - 1, 31);
    const size_t oN_stride = rem * output_stride;
    const size_t oN_offset = oN_stride + tile_hbytes;
    size_t bh = block_height;
    for (; bh >= 32; bh -= 32) {
      const HVX_Vector v5_0 = *((HVX_UVector *) i0); i0 = (uint32_t*) ((uintptr_t) i0 + input_offset);
      const HVX_Vector v5_1 = *((HVX_UVector *) i1); i1 = (uint32_t*) ((uintptr_t) i1 + input_offset);
      const HVX_Vector v5_2 = *((HVX_UVector *) i2); i2 = (uint32_t*) ((uintptr_t) i2 + input_offset);
      const HVX_Vector v5_3 = *((HVX_UVector *) i3); i3 = (uint32_t*) ((uintptr_t) i3 + input_offset);
      const HVX_Vector v5_4 = *((HVX_UVector *) i4); i4 = (uint32_t*) ((uintptr_t) i4 + input_offset);
      const HVX_Vector v5_5 = *((HVX_UVector *) i5); i5 = (uint32_t*) ((uintptr_t) i5 + input_offset);
      const HVX_Vector v5_6 = *((HVX_UVector *) i6); i6 = (uint32_t*) ((uintptr_t) i6 + input_offset);
      const HVX_Vector v5_7 = *((HVX_UVector *) i7); i7 = (uint32_t*) ((uintptr_t) i7 + input_offset);
      const HVX_Vector v5_8 = *((HVX_UVector *) i8); i8 = (uint32_t*) ((uintptr_t) i8 + input_offset);
      const HVX_Vector v5_9 = *((HVX_UVector *) i9); i9 = (uint32_t*) ((uintptr_t) i9 + input_offset);
      const HVX_Vector v5_10 = *((HVX_UVector *) i10); i10 = (uint32_t*) ((uintptr_t) i10 + input_offset);
      const HVX_Vector v5_11 = *((HVX_UVector *) i11); i11 = (uint32_t*) ((uintptr_t) i11 + input_offset);
      const HVX_Vector v5_12 = *((HVX_UVector *) i12); i12 = (uint32_t*) ((uintptr_t) i12 + input_offset);
      const HVX_Vector v5_13 = *((HVX_UVector *) i13); i13 = (uint32_t*) ((uintptr_t) i13 + input_offset);
      const HVX_Vector v5_14 = *((HVX_UVector *) i14); i14 = (uint32_t*) ((uintptr_t) i14 + input_offset);
      const HVX_Vector v5_15 = *((HVX_UVector *) i15); i15 = (uint32_t*) ((uintptr_t) i15 + input_offset);
      const HVX_Vector v5_16 = *((HVX_UVector *) i16); i16 = (uint32_t*) ((uintptr_t) i16 + input_offset);
      const HVX_Vector v5_17 = *((HVX_UVector *) i17); i17 = (uint32_t*) ((uintptr_t) i17 + input_offset);
      const HVX_Vector v5_18 = *((HVX_UVector *) i18); i18 = (uint32_t*) ((uintptr_t) i18 + input_offset);
      const HVX_Vector v5_19 = *((HVX_UVector *) i19); i19 = (uint32_t*) ((uintptr_t) i19 + input_offset);
      const HVX_Vector v5_20 = *((HVX_UVector *) i20); i20 = (uint32_t*) ((uintptr_t) i20 + input_offset);
      const HVX_Vector v5_21 = *((HVX_UVector *) i21); i21 = (uint32_t*) ((uintptr_t) i21 + input_offset);
      const HVX_Vector v5_22 = *((HVX_UVector *) i22); i22 = (uint32_t*) ((uintptr_t) i22 + input_offset);
      const HVX_Vector v5_23 = *((HVX_UVector *) i23); i23 = (uint32_t*) ((uintptr_t) i23 + input_offset);
      const HVX_Vector v5_24 = *((HVX_UVector *) i24); i24 = (uint32_t*) ((uintptr_t) i24 + input_offset);
      const HVX_Vector v5_25 = *((HVX_UVector *) i25); i25 = (uint32_t*) ((uintptr_t) i25 + input_offset);
      const HVX_Vector v5_26 = *((HVX_UVector *) i26); i26 = (uint32_t*) ((uintptr_t) i26 + input_offset);
      const HVX_Vector v5_27 = *((HVX_UVector *) i27); i27 = (uint32_t*) ((uintptr_t) i27 + input_offset);
      const HVX_Vector v5_28 = *((HVX_UVector *) i28); i28 = (uint32_t*) ((uintptr_t) i28 + input_offset);
      const HVX_Vector v5_29 = *((HVX_UVector *) i29); i29 = (uint32_t*) ((uintptr_t) i29 + input_offset);
      const HVX_Vector v5_30 = *((HVX_UVector *) i30); i30 = (uint32_t*) ((uintptr_t) i30 + input_offset);
      const HVX_Vector v5_31 = *((HVX_UVector *) i31); i31 = (uint32_t*) ((uintptr_t) i31 + input_offset);

      int rt = -4;
      const HVX_VectorPair v4_0 = Q6_W_vshuff_VVR(v5_1, v5_0, rt);
      const HVX_VectorPair v4_1 = Q6_W_vshuff_VVR(v5_3, v5_2, rt);
      const HVX_VectorPair v4_2 = Q6_W_vshuff_VVR(v5_5, v5_4, rt);
      const HVX_VectorPair v4_3 = Q6_W_vshuff_VVR(v5_7, v5_6, rt);
      const HVX_VectorPair v4_4 = Q6_W_vshuff_VVR(v5_9, v5_8, rt);
      const HVX_VectorPair v4_5 = Q6_W_vshuff_VVR(v5_11, v5_10, rt);
      const HVX_VectorPair v4_6 = Q6_W_vshuff_VVR(v5_13, v5_12, rt);
      const HVX_VectorPair v4_7 = Q6_W_vshuff_VVR(v5_15, v5_14, rt);
      const HVX_VectorPair v4_8 = Q6_W_vshuff_VVR(v5_17, v5_16, rt);
      const HVX_VectorPair v4_9 = Q6_W_vshuff_VVR(v5_19, v5_18, rt);
      const HVX_VectorPair v4_10 = Q6_W_vshuff_VVR(v5_21, v5_20, rt);
      const HVX_VectorPair v4_11 = Q6_W_vshuff_VVR(v5_23, v5_22, rt);
      const HVX_VectorPair v4_12 = Q6_W_vshuff_VVR(v5_25, v5_24, rt);
      const HVX_VectorPair v4_13 = Q6_W_vshuff_VVR(v5_27, v5_26, rt);
      const HVX_VectorPair v4_14 = Q6_W_vshuff_VVR(v5_29, v5_28, rt);
      const HVX_VectorPair v4_15 = Q6_W_vshuff_VVR(v5_31, v5_30, rt);

      rt = rt << 1;
      HVX_VectorPair v3_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_1), Q6_V_lo_W(v4_0), rt);
      HVX_VectorPair v3_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_1), Q6_V_hi_W(v4_0), rt);
      
      HVX_VectorPair v3_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_3), Q6_V_lo_W(v4_2), rt);
      HVX_VectorPair v3_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_3), Q6_V_hi_W(v4_2), rt);
      
      HVX_VectorPair v3_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_5), Q6_V_lo_W(v4_4), rt);
      HVX_VectorPair v3_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_5), Q6_V_hi_W(v4_4), rt);
      
      HVX_VectorPair v3_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_7), Q6_V_lo_W(v4_6), rt);
      HVX_VectorPair v3_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_7), Q6_V_hi_W(v4_6), rt);
      
      HVX_VectorPair v3_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_9), Q6_V_lo_W(v4_8), rt);
      HVX_VectorPair v3_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_9), Q6_V_hi_W(v4_8), rt);
      
      HVX_VectorPair v3_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_11), Q6_V_lo_W(v4_10), rt);
      HVX_VectorPair v3_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_11), Q6_V_hi_W(v4_10), rt);
      
      HVX_VectorPair v3_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_13), Q6_V_lo_W(v4_12), rt);
      HVX_VectorPair v3_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_13), Q6_V_hi_W(v4_12), rt);
      
      HVX_VectorPair v3_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_15), Q6_V_lo_W(v4_14), rt);
      HVX_VectorPair v3_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_15), Q6_V_hi_W(v4_14), rt);
      
      rt = rt << 1;
      HVX_VectorPair v2_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_2), Q6_V_lo_W(v3_0), rt);
      HVX_VectorPair v2_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_2), Q6_V_hi_W(v3_0), rt);
      
      HVX_VectorPair v2_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_3), Q6_V_lo_W(v3_1), rt);
      HVX_VectorPair v2_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_3), Q6_V_hi_W(v3_1), rt);
      
      HVX_VectorPair v2_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_6), Q6_V_lo_W(v3_4), rt);
      HVX_VectorPair v2_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_6), Q6_V_hi_W(v3_4), rt);
      
      HVX_VectorPair v2_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_7), Q6_V_lo_W(v3_5), rt);
      HVX_VectorPair v2_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_7), Q6_V_hi_W(v3_5), rt);
      
      HVX_VectorPair v2_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_10), Q6_V_lo_W(v3_8), rt);
      HVX_VectorPair v2_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_10), Q6_V_hi_W(v3_8), rt);
      
      HVX_VectorPair v2_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_11), Q6_V_lo_W(v3_9), rt);
      HVX_VectorPair v2_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_11), Q6_V_hi_W(v3_9), rt);
      
      HVX_VectorPair v2_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_14), Q6_V_lo_W(v3_12), rt);
      HVX_VectorPair v2_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_14), Q6_V_hi_W(v3_12), rt);
      
      HVX_VectorPair v2_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_15), Q6_V_lo_W(v3_13), rt);
      HVX_VectorPair v2_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_15), Q6_V_hi_W(v3_13), rt);
      
      rt = rt << 1;
      HVX_VectorPair v1_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_4), Q6_V_lo_W(v2_0), rt);
      HVX_VectorPair v1_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_4), Q6_V_hi_W(v2_0), rt);
      
      HVX_VectorPair v1_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_5), Q6_V_lo_W(v2_1), rt);
      HVX_VectorPair v1_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_5), Q6_V_hi_W(v2_1), rt);
      
      HVX_VectorPair v1_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_6), Q6_V_lo_W(v2_2), rt);
      HVX_VectorPair v1_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_6), Q6_V_hi_W(v2_2), rt);
      
      HVX_VectorPair v1_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_7), Q6_V_lo_W(v2_3), rt);
      HVX_VectorPair v1_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_7), Q6_V_hi_W(v2_3), rt);
      
      HVX_VectorPair v1_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_12), Q6_V_lo_W(v2_8), rt);
      HVX_VectorPair v1_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_12), Q6_V_hi_W(v2_8), rt);
      
      HVX_VectorPair v1_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_13), Q6_V_lo_W(v2_9), rt);
      HVX_VectorPair v1_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_13), Q6_V_hi_W(v2_9), rt);
      
      HVX_VectorPair v1_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_14), Q6_V_lo_W(v2_10), rt);
      HVX_VectorPair v1_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_14), Q6_V_hi_W(v2_10), rt);
      
      HVX_VectorPair v1_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_15), Q6_V_lo_W(v2_11), rt);
      HVX_VectorPair v1_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_15), Q6_V_hi_W(v2_11), rt);
      
      rt = rt << 1;
      HVX_VectorPair v0_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_8), Q6_V_lo_W(v1_0), rt);
      HVX_VectorPair v0_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_8), Q6_V_hi_W(v1_0), rt);
      
      HVX_VectorPair v0_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_9), Q6_V_lo_W(v1_1), rt);
      HVX_VectorPair v0_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_9), Q6_V_hi_W(v1_1), rt);
      
      HVX_VectorPair v0_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_10), Q6_V_lo_W(v1_2), rt);
      HVX_VectorPair v0_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_10), Q6_V_hi_W(v1_2), rt);
      
      HVX_VectorPair v0_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_11), Q6_V_lo_W(v1_3), rt);
      HVX_VectorPair v0_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_11), Q6_V_hi_W(v1_3), rt);
      
      HVX_VectorPair v0_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_12), Q6_V_lo_W(v1_4), rt);
      HVX_VectorPair v0_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_12), Q6_V_hi_W(v1_4), rt);
      
      HVX_VectorPair v0_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_13), Q6_V_lo_W(v1_5), rt);
      HVX_VectorPair v0_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_13), Q6_V_hi_W(v1_5), rt);
      
      HVX_VectorPair v0_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_14), Q6_V_lo_W(v1_6), rt);
      HVX_VectorPair v0_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_14), Q6_V_hi_W(v1_6), rt);
      
      HVX_VectorPair v0_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_15), Q6_V_lo_W(v1_7), rt);
      HVX_VectorPair v0_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_15), Q6_V_hi_W(v1_7), rt);
      
      o = (uint32_t*) ((uintptr_t) o + oN_offset);
      xnn_storeu_f32(o, Q6_V_hi_W(v0_15));
      uint32_t *oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 31) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_lo_W(v0_15));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 30) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_hi_W(v0_14));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 29) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_lo_W(v0_14));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 28) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_hi_W(v0_13));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 27) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_lo_W(v0_13));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 26) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_hi_W(v0_12));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 25) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_lo_W(v0_12));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 24) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_hi_W(v0_11));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 23) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_lo_W(v0_11));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 22) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_hi_W(v0_10));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 21) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_lo_W(v0_10));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 20) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_hi_W(v0_9));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 19) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_lo_W(v0_9));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 18) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_hi_W(v0_8));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 17) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_lo_W(v0_8));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 16) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_hi_W(v0_7));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 15) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_lo_W(v0_7));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 14) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_hi_W(v0_6));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 13) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_lo_W(v0_6));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 12) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_hi_W(v0_5));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 11) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_lo_W(v0_5));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 10) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_hi_W(v0_4));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 9) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_lo_W(v0_4));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 8) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_hi_W(v0_3));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 7) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_lo_W(v0_3));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 6) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_hi_W(v0_2));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 5) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_lo_W(v0_2));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 4) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_hi_W(v0_1));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 3) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_lo_W(v0_1));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width >= 2) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_hi_W(v0_0));
      oN = (uint32_t*) ((uintptr_t) o + minus_output_stride);
      if XNN_UNPREDICTABLE(block_width > 1) {
        o = oN;
      }
      xnn_storeu_f32(o, Q6_V_lo_W(v0_0));
    }
    o = (uint32_t*) ((uintptr_t) o + tile_hbytes);


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
    i16 = (const uint32_t*) ((uintptr_t) i15 + input_stride);
    i17 = (const uint32_t*) ((uintptr_t) i16 + input_stride);
    i18 = (const uint32_t*) ((uintptr_t) i17 + input_stride);
    i19 = (const uint32_t*) ((uintptr_t) i18 + input_stride);
    i20 = (const uint32_t*) ((uintptr_t) i19 + input_stride);
    i21 = (const uint32_t*) ((uintptr_t) i20 + input_stride);
    i22 = (const uint32_t*) ((uintptr_t) i21 + input_stride);
    i23 = (const uint32_t*) ((uintptr_t) i22 + input_stride);
    i24 = (const uint32_t*) ((uintptr_t) i23 + input_stride);
    i25 = (const uint32_t*) ((uintptr_t) i24 + input_stride);
    i26 = (const uint32_t*) ((uintptr_t) i25 + input_stride);
    i27 = (const uint32_t*) ((uintptr_t) i26 + input_stride);
    i28 = (const uint32_t*) ((uintptr_t) i27 + input_stride);
    i29 = (const uint32_t*) ((uintptr_t) i28 + input_stride);
    i30 = (const uint32_t*) ((uintptr_t) i29 + input_stride);
    i31 = (const uint32_t*) ((uintptr_t) i30 + input_stride);
    o = (uint32_t*) ((uintptr_t) o + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}


