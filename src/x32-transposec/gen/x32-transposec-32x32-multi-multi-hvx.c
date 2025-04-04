// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x32-transposec/hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/simd/f32-hvx.h"

#include <assert.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/transpose.h"

void xnn_x32_transposec_ukernel__32x32_multi_multi_hvx(
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
  uint32_t* o16 = (uint32_t*) ((uintptr_t) o15 + output_stride);
  uint32_t* o17 = (uint32_t*) ((uintptr_t) o16 + output_stride);
  uint32_t* o18 = (uint32_t*) ((uintptr_t) o17 + output_stride);
  uint32_t* o19 = (uint32_t*) ((uintptr_t) o18 + output_stride);
  uint32_t* o20 = (uint32_t*) ((uintptr_t) o19 + output_stride);
  uint32_t* o21 = (uint32_t*) ((uintptr_t) o20 + output_stride);
  uint32_t* o22 = (uint32_t*) ((uintptr_t) o21 + output_stride);
  uint32_t* o23 = (uint32_t*) ((uintptr_t) o22 + output_stride);
  uint32_t* o24 = (uint32_t*) ((uintptr_t) o23 + output_stride);
  uint32_t* o25 = (uint32_t*) ((uintptr_t) o24 + output_stride);
  uint32_t* o26 = (uint32_t*) ((uintptr_t) o25 + output_stride);
  uint32_t* o27 = (uint32_t*) ((uintptr_t) o26 + output_stride);
  uint32_t* o28 = (uint32_t*) ((uintptr_t) o27 + output_stride);
  uint32_t* o29 = (uint32_t*) ((uintptr_t) o28 + output_stride);
  uint32_t* o30 = (uint32_t*) ((uintptr_t) o29 + output_stride);
  uint32_t* o31 = (uint32_t*) ((uintptr_t) o30 + output_stride);

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
    if XNN_UNPREDICTABLE(block_width <= 16) {
      o16 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 18) {
      o17 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 18) {
      o18 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 20) {
      o19 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 20) {
      o20 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 22) {
      o21 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 22) {
      o22 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 24) {
      o23 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 24) {
      o24 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 26) {
      o25 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 26) {
      o26 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 28) {
      o27 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 28) {
      o28 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 30) {
      o29 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 30) {
      o30 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 32) {
      o31 = o0;
    }
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
      
      *((HVX_UVector*)o31) = Q6_V_hi_W(v0_15); o31 = (uint32_t*) ((uintptr_t) o31 + tile_hbytes);
      *((HVX_UVector*)o30) = Q6_V_lo_W(v0_15); o30 = (uint32_t*) ((uintptr_t) o30 + tile_hbytes);
      *((HVX_UVector*)o29) = Q6_V_hi_W(v0_14); o29 = (uint32_t*) ((uintptr_t) o29 + tile_hbytes);
      *((HVX_UVector*)o28) = Q6_V_lo_W(v0_14); o28 = (uint32_t*) ((uintptr_t) o28 + tile_hbytes);
      *((HVX_UVector*)o27) = Q6_V_hi_W(v0_13); o27 = (uint32_t*) ((uintptr_t) o27 + tile_hbytes);
      *((HVX_UVector*)o26) = Q6_V_lo_W(v0_13); o26 = (uint32_t*) ((uintptr_t) o26 + tile_hbytes);
      *((HVX_UVector*)o25) = Q6_V_hi_W(v0_12); o25 = (uint32_t*) ((uintptr_t) o25 + tile_hbytes);
      *((HVX_UVector*)o24) = Q6_V_lo_W(v0_12); o24 = (uint32_t*) ((uintptr_t) o24 + tile_hbytes);
      *((HVX_UVector*)o23) = Q6_V_hi_W(v0_11); o23 = (uint32_t*) ((uintptr_t) o23 + tile_hbytes);
      *((HVX_UVector*)o22) = Q6_V_lo_W(v0_11); o22 = (uint32_t*) ((uintptr_t) o22 + tile_hbytes);
      *((HVX_UVector*)o21) = Q6_V_hi_W(v0_10); o21 = (uint32_t*) ((uintptr_t) o21 + tile_hbytes);
      *((HVX_UVector*)o20) = Q6_V_lo_W(v0_10); o20 = (uint32_t*) ((uintptr_t) o20 + tile_hbytes);
      *((HVX_UVector*)o19) = Q6_V_hi_W(v0_9); o19 = (uint32_t*) ((uintptr_t) o19 + tile_hbytes);
      *((HVX_UVector*)o18) = Q6_V_lo_W(v0_9); o18 = (uint32_t*) ((uintptr_t) o18 + tile_hbytes);
      *((HVX_UVector*)o17) = Q6_V_hi_W(v0_8); o17 = (uint32_t*) ((uintptr_t) o17 + tile_hbytes);
      *((HVX_UVector*)o16) = Q6_V_lo_W(v0_8); o16 = (uint32_t*) ((uintptr_t) o16 + tile_hbytes);
      *((HVX_UVector*)o15) = Q6_V_hi_W(v0_7); o15 = (uint32_t*) ((uintptr_t) o15 + tile_hbytes);
      *((HVX_UVector*)o14) = Q6_V_lo_W(v0_7); o14 = (uint32_t*) ((uintptr_t) o14 + tile_hbytes);
      *((HVX_UVector*)o13) = Q6_V_hi_W(v0_6); o13 = (uint32_t*) ((uintptr_t) o13 + tile_hbytes);
      *((HVX_UVector*)o12) = Q6_V_lo_W(v0_6); o12 = (uint32_t*) ((uintptr_t) o12 + tile_hbytes);
      *((HVX_UVector*)o11) = Q6_V_hi_W(v0_5); o11 = (uint32_t*) ((uintptr_t) o11 + tile_hbytes);
      *((HVX_UVector*)o10) = Q6_V_lo_W(v0_5); o10 = (uint32_t*) ((uintptr_t) o10 + tile_hbytes);
      *((HVX_UVector*)o9) = Q6_V_hi_W(v0_4); o9 = (uint32_t*) ((uintptr_t) o9 + tile_hbytes);
      *((HVX_UVector*)o8) = Q6_V_lo_W(v0_4); o8 = (uint32_t*) ((uintptr_t) o8 + tile_hbytes);
      *((HVX_UVector*)o7) = Q6_V_hi_W(v0_3); o7 = (uint32_t*) ((uintptr_t) o7 + tile_hbytes);
      *((HVX_UVector*)o6) = Q6_V_lo_W(v0_3); o6 = (uint32_t*) ((uintptr_t) o6 + tile_hbytes);
      *((HVX_UVector*)o5) = Q6_V_hi_W(v0_2); o5 = (uint32_t*) ((uintptr_t) o5 + tile_hbytes);
      *((HVX_UVector*)o4) = Q6_V_lo_W(v0_2); o4 = (uint32_t*) ((uintptr_t) o4 + tile_hbytes);
      *((HVX_UVector*)o3) = Q6_V_hi_W(v0_1); o3 = (uint32_t*) ((uintptr_t) o3 + tile_hbytes);
      *((HVX_UVector*)o2) = Q6_V_lo_W(v0_1); o2 = (uint32_t*) ((uintptr_t) o2 + tile_hbytes);
      *((HVX_UVector*)o1) = Q6_V_hi_W(v0_0); o1 = (uint32_t*) ((uintptr_t) o1 + tile_hbytes);
      *((HVX_UVector*)o0) = Q6_V_lo_W(v0_0); o0 = (uint32_t*) ((uintptr_t) o0 + tile_hbytes);
    }


    if (bh != 0){
      // This is a scalar implementation. This tail code is for the case where TILE_SIZE==32.
      uint32_t* i = (uint32_t *)i0;
      uint32_t* o = (uint32_t *) o0;
      size_t tail_bw = min(block_width, tile_width);
      if (bh & 16){
        for(size_t bw = 0; bw < tail_bw; bw++){
          const size_t oN_offset = output_stride * bw;
          *((uint32_t *) ((uintptr_t) o + oN_offset)) = *(i + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 1) = *((uint32_t *) ((uintptr_t) i + input_stride) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 2) = *((uint32_t *) ((uintptr_t) i + input_stride * 2) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 3) = *((uint32_t *) ((uintptr_t) i + input_stride * 3) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 4) = *((uint32_t *) ((uintptr_t) i + input_stride * 4) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 5) = *((uint32_t *) ((uintptr_t) i + input_stride * 5) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 6) = *((uint32_t *) ((uintptr_t) i + input_stride * 6) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 7) = *((uint32_t *) ((uintptr_t) i + input_stride * 7) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 8) = *((uint32_t *) ((uintptr_t) i + input_stride * 8) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 9) = *((uint32_t *) ((uintptr_t) i + input_stride * 9) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 10) = *((uint32_t *) ((uintptr_t) i + input_stride * 10) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 11) = *((uint32_t *) ((uintptr_t) i + input_stride * 11) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 12) = *((uint32_t *) ((uintptr_t) i + input_stride * 12) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 13) = *((uint32_t *) ((uintptr_t) i + input_stride * 13) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 14) = *((uint32_t *) ((uintptr_t) i + input_stride * 14) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 15) = *((uint32_t *) ((uintptr_t) i + input_stride * 15) + bw);
        }
        o += 16;
        i = (uint32_t *)((uintptr_t) i + input_stride * 16);
      }
      if (bh & 8){
        for(size_t bw = 0; bw < tail_bw; bw++){
          const size_t oN_offset = output_stride * bw;
          *((uint32_t *) ((uintptr_t) o + oN_offset)) = *(i + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 1) = *((uint32_t *) ((uintptr_t) i + input_stride) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 2) = *((uint32_t *) ((uintptr_t) i + input_stride * 2) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 3) = *((uint32_t *) ((uintptr_t) i + input_stride * 3) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 4) = *((uint32_t *) ((uintptr_t) i + input_stride * 4) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 5) = *((uint32_t *) ((uintptr_t) i + input_stride * 5) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 6) = *((uint32_t *) ((uintptr_t) i + input_stride * 6) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 7) = *((uint32_t *) ((uintptr_t) i + input_stride * 7) + bw);
        }
        o += 8;
        i = (uint32_t *)((uintptr_t) i + input_stride * 8);
      }
      if (bh & 4){
        for(size_t bw = 0; bw < tail_bw; bw++){
          const size_t oN_offset = output_stride * bw;
          *((uint32_t *) ((uintptr_t) o + oN_offset)) = *(i + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 1) = *((uint32_t *) ((uintptr_t) i + input_stride) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 2) = *((uint32_t *) ((uintptr_t) i + input_stride * 2) + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 3) = *((uint32_t *) ((uintptr_t) i + input_stride * 3) + bw);
        }
        o += 4;
        i = (uint32_t *)((uintptr_t) i + input_stride * 4);
      }
      if (bh & 2){
        for(size_t bw = 0; bw < tail_bw; bw++){
          const size_t oN_offset = output_stride * bw;
          *((uint32_t *) ((uintptr_t) o + oN_offset)) = *(i + bw);
          *((uint32_t *) ((uintptr_t) o + oN_offset) + 1) = *((uint32_t *) ((uintptr_t) i + input_stride) + bw);
        }
        o += 2;
        i = (uint32_t *)((uintptr_t) i + input_stride * 2);
      }
      if (bh & 1){
        for(size_t bw = 0; bw < block_width; bw++){
          *((uint32_t *) ((uintptr_t) o + output_stride * bw)) = *(i + bw);
        }
      }
      o0 = o;
    }

    if (block_width > tile_width){
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
      o0 = (uint32_t*) ((uintptr_t) o0 + output_reset);
      o1 = (uint32_t*) ((uintptr_t) o0 + output_stride);
      o2 = (uint32_t*) ((uintptr_t) o1 + output_stride);
      o3 = (uint32_t*) ((uintptr_t) o2 + output_stride);
      o4 = (uint32_t*) ((uintptr_t) o3 + output_stride);
      o5 = (uint32_t*) ((uintptr_t) o4 + output_stride);
      o6 = (uint32_t*) ((uintptr_t) o5 + output_stride);
      o7 = (uint32_t*) ((uintptr_t) o6 + output_stride);
      o8 = (uint32_t*) ((uintptr_t) o7 + output_stride);
      o9 = (uint32_t*) ((uintptr_t) o8 + output_stride);
      o10 = (uint32_t*) ((uintptr_t) o9 + output_stride);
      o11 = (uint32_t*) ((uintptr_t) o10 + output_stride);
      o12 = (uint32_t*) ((uintptr_t) o11 + output_stride);
      o13 = (uint32_t*) ((uintptr_t) o12 + output_stride);
      o14 = (uint32_t*) ((uintptr_t) o13 + output_stride);
      o15 = (uint32_t*) ((uintptr_t) o14 + output_stride);
      o16 = (uint32_t*) ((uintptr_t) o15 + output_stride);
      o17 = (uint32_t*) ((uintptr_t) o16 + output_stride);
      o18 = (uint32_t*) ((uintptr_t) o17 + output_stride);
      o19 = (uint32_t*) ((uintptr_t) o18 + output_stride);
      o20 = (uint32_t*) ((uintptr_t) o19 + output_stride);
      o21 = (uint32_t*) ((uintptr_t) o20 + output_stride);
      o22 = (uint32_t*) ((uintptr_t) o21 + output_stride);
      o23 = (uint32_t*) ((uintptr_t) o22 + output_stride);
      o24 = (uint32_t*) ((uintptr_t) o23 + output_stride);
      o25 = (uint32_t*) ((uintptr_t) o24 + output_stride);
      o26 = (uint32_t*) ((uintptr_t) o25 + output_stride);
      o27 = (uint32_t*) ((uintptr_t) o26 + output_stride);
      o28 = (uint32_t*) ((uintptr_t) o27 + output_stride);
      o29 = (uint32_t*) ((uintptr_t) o28 + output_stride);
      o30 = (uint32_t*) ((uintptr_t) o29 + output_stride);
      o31 = (uint32_t*) ((uintptr_t) o30 + output_stride);
    }
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
