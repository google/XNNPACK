// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Generator: tools/update-microkernels.py -a

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#endif  // XNN_ENABLE_KLEIDIAI

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/lut.h"
#include "xnnpack/math.h"
#include "xnnpack/microparams.h"
#include "xnnpack/packq.h"
#include "xnnpack/transpose.h"
#include "xnnpack/vbinary.h"
#include "xnnpack/vunary.h"


void xnn_f32_vdiv_minmax_ukernel__aarch64_neon_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
  const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float32x4_t va0 = vld1q_f32(input_a); input_a += 4;
    const float32x4_t vb0 = vld1q_f32(input_b); input_b += 4;
    const float32x4_t va1 = vld1q_f32(input_a); input_a += 4;
    const float32x4_t vb1 = vld1q_f32(input_b); input_b += 4;

    float32x4_t vacc0 = vdivq_f32(va0, vb0);
    float32x4_t vacc1 = vdivq_f32(va1, vb1);


    vacc0 = vmaxq_f32(vacc0, voutput_min);
    vacc1 = vmaxq_f32(vacc1, voutput_min);

    vacc0 = vminq_f32(vacc0, voutput_max);
    vacc1 = vminq_f32(vacc1, voutput_max);

    vst1q_f32(output, vacc0); output += 4;
    vst1q_f32(output, vacc1); output += 4;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t va = vld1q_f32(input_a); input_a += 4;
    const float32x4_t vb = vld1q_f32(input_b); input_b += 4;

    float32x4_t vacc = vdivq_f32(va, vb);
    vacc = vmaxq_f32(vacc, voutput_min);
    vacc = vminq_f32(vacc, voutput_max);

    vst1q_f32(output, vacc); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t va = vld1q_f32(input_a);
    const float32x4_t vb = vld1q_f32(input_b);

    float32x4_t vacc = vdivq_f32(va, vb);
    vacc = vmaxq_f32(vacc, voutput_min);
    vacc = vminq_f32(vacc, voutput_max);

    float32x2_t vacc_lo = vget_low_f32(vacc);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vacc_lo); output += 2;
      vacc_lo = vget_high_f32(vacc);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vacc_lo, 0);
    }
  }
}

void xnn_f32_vdivc_minmax_ukernel__aarch64_neon_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
  const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
  const float32x4_t vb = vld1q_dup_f32(input_b);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    float32x4_t vacc_ = vld1q_f32(input_a); input_a += 4;
    float32x4_t vaccl = vld1q_f32(input_a); input_a += 4;

    vacc_ = vdivq_f32(vacc_, vb);
    vaccl = vdivq_f32(vaccl, vb);


    vacc_ = vmaxq_f32(vacc_, voutput_min);
    vaccl = vmaxq_f32(vaccl, voutput_min);

    vacc_ = vminq_f32(vacc_, voutput_max);
    vaccl = vminq_f32(vaccl, voutput_max);

    vst1q_f32(output, vacc_); output += 4;
    vst1q_f32(output, vaccl); output += 4;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t va = vld1q_f32(input_a); input_a += 4;

    float32x4_t vacc = vdivq_f32(va, vb);
    vacc = vmaxq_f32(vacc, voutput_min);
    vacc = vminq_f32(vacc, voutput_max);

    vst1q_f32(output, vacc); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t va = vld1q_f32(input_a);

    float32x4_t vacc = vdivq_f32(va, vb);
    vacc = vmaxq_f32(vacc, voutput_min);
    vacc = vminq_f32(vacc, voutput_max);

    float32x2_t vacc_lo = vget_low_f32(vacc);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vacc_lo); output += 2;
      vacc_lo = vget_high_f32(vacc);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vacc_lo, 0);
    }
  }
}

void xnn_f32_vrdivc_minmax_ukernel__aarch64_neon_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
  const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
  const float32x4_t vb = vld1q_dup_f32(input_b);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    float32x4_t vacc_ = vld1q_f32(input_a); input_a += 4;
    float32x4_t vaccl = vld1q_f32(input_a); input_a += 4;

    vacc_ = vdivq_f32(vb, vacc_);
    vaccl = vdivq_f32(vb, vaccl);


    vacc_ = vmaxq_f32(vacc_, voutput_min);
    vaccl = vmaxq_f32(vaccl, voutput_min);

    vacc_ = vminq_f32(vacc_, voutput_max);
    vaccl = vminq_f32(vaccl, voutput_max);

    vst1q_f32(output, vacc_); output += 4;
    vst1q_f32(output, vaccl); output += 4;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t va = vld1q_f32(input_a); input_a += 4;

    float32x4_t vacc = vdivq_f32(vb, va);
    vacc = vmaxq_f32(vacc, voutput_min);
    vacc = vminq_f32(vacc, voutput_max);

    vst1q_f32(output, vacc); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t va = vld1q_f32(input_a);

    float32x4_t vacc = vdivq_f32(vb, va);
    vacc = vmaxq_f32(vacc, voutput_min);
    vacc = vminq_f32(vacc, voutput_max);

    float32x2_t vacc_lo = vget_low_f32(vacc);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vacc_lo); output += 2;
      vacc_lo = vget_high_f32(vacc);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vacc_lo, 0);
    }
  }
}

void xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;
    const float32x4_t vy = vsqrtq_f32(vx);
    vst1q_f32(output, vy); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = vld1q_f32(input);

    float32x2_t vy_lo = vsqrt_f32(vget_low_f32(vx));
    const float32x2_t vx_hi = vget_high_f32(vx);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vy_lo); output += 2;
      vy_lo = vsqrt_f32(vx_hi);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vy_lo, 0);
    }
  }
}

void xnn_x24_transposec_ukernel__4x4_aarch64_neon_tbl128(
    const void* input,
    void* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  static const uint8_t pos0[16] = {0, 1, 2, 16, 17, 18, 32, 33, 34, 48, 49, 50, 0, 0, 0, 0};
  static const uint8_t pos1[16] = {3, 4, 5, 19, 20, 21, 35, 36, 37, 51, 52, 53, 0, 0, 0, 0};
  static const uint8_t pos2[16] = {6, 7, 8, 22, 23, 24, 38, 39, 40, 54, 55, 56, 0, 0, 0, 0};
  static const uint8_t pos3[16] = {9, 10, 11, 25, 26, 27, 41, 42, 43, 57, 58, 59, 0, 0, 0, 0};

  assert(output_stride >= block_height * 3);
  assert(input_stride >= block_width * 3);

  const size_t tile_height = 4;
  const size_t tile_width = 4;
  const size_t tile_wbytes = tile_width * 3;
  const size_t tile_wbytes_minus_8 = tile_wbytes - 8;
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_width * output_stride - block_height * 3;
  const size_t tile_stride = tile_height * input_stride;

  const uint8_t* i0 = (const uint8_t*) input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);

  uint8_t* o0 = (uint8_t*) output;
  uint8_t* o1 = (uint8_t*) ((uintptr_t) o0 + output_stride);
  uint8_t* o2 = (uint8_t*) ((uintptr_t) o1 + output_stride);
  uint8_t* o3 = (uint8_t*) ((uintptr_t) o2 + output_stride);

  const uint8x16_t vperm0 = vld1q_u8(pos0);
  const uint8x16_t vperm1 = vld1q_u8(pos1);
  const uint8x16_t vperm2 = vld1q_u8(pos2);
  const uint8x16_t vperm3 = vld1q_u8(pos3);
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
      uint8x16x4_t v;
      v.val[0] = vld1q_u8(i0); i0 = (const uint8_t*) ((uintptr_t) i0 + tile_stride);
      v.val[1] = vld1q_u8(i1); i1 = (const uint8_t*) ((uintptr_t) i1 + tile_stride);
      v.val[2] = vld1q_u8(i2); i2 = (const uint8_t*) ((uintptr_t) i2 + tile_stride);
      v.val[3] = vld1q_u8(i3); i3 = (const uint8_t*) ((uintptr_t) i3 + tile_stride);

      const uint8x16_t vres0 = vqtbl4q_u8(v, vperm0);
      const uint8x16_t vres1 = vqtbl4q_u8(v, vperm1);
      const uint8x16_t vres2 = vqtbl4q_u8(v, vperm2);
      const uint8x16_t vres3 = vqtbl4q_u8(v, vperm3);

      vst1_u8(o3, vget_low_u8(vres3)); o3 += 8;
      vst1_u8(o2, vget_low_u8(vres2)); o2 += 8;
      vst1_u8(o1, vget_low_u8(vres1)); o1 += 8;
      vst1_u8(o0, vget_low_u8(vres0)); o0 += 8;
      vst1q_lane_u32((void*) o3, vreinterpretq_u32_u8(vres3), 2); o3 = (uint8_t*) ((uintptr_t) o3 + tile_wbytes_minus_8);
      vst1q_lane_u32((void*) o2, vreinterpretq_u32_u8(vres2), 2); o2 = (uint8_t*) ((uintptr_t) o2 + tile_wbytes_minus_8);
      vst1q_lane_u32((void*) o1, vreinterpretq_u32_u8(vres1), 2); o1 = (uint8_t*) ((uintptr_t) o1 + tile_wbytes_minus_8);
      vst1q_lane_u32((void*) o0, vreinterpretq_u32_u8(vres0), 2); o0 = (uint8_t*) ((uintptr_t) o0 + tile_wbytes_minus_8);
    }

    if (bh != 0) {
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i0;
      }
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      uint8x16x3_t v;
      v.val[0] = vld1q_u8(i0);
      v.val[1] = vld1q_u8(i1);
      v.val[2] = vld1q_u8(i2);

      uint8x16_t vres0 = vqtbl3q_u8(v, vperm0);
      uint8x16_t vres1 = vqtbl3q_u8(v, vperm1);
      uint8x16_t vres2 = vqtbl3q_u8(v, vperm2);
      uint8x16_t vres3 = vqtbl3q_u8(v, vperm3);

      uint8x8_t vres0_lo = vget_low_u8(vres0);
      uint8x8_t vres1_lo = vget_low_u8(vres1);
      uint8x8_t vres2_lo = vget_low_u8(vres2);
      uint8x8_t vres3_lo = vget_low_u8(vres3);

      if (bh & 2) {
        vst1_lane_u32((void*) o3, vreinterpret_u32_u8(vres3_lo), 0); o3 += 4;
        vst1_lane_u32((void*) o2, vreinterpret_u32_u8(vres2_lo), 0); o2 += 4;
        vst1_lane_u32((void*) o1, vreinterpret_u32_u8(vres1_lo), 0); o1 += 4;
        vst1_lane_u32((void*) o0, vreinterpret_u32_u8(vres0_lo), 0); o0 += 4;
        vst1_lane_u16((void*) o3, vreinterpret_u16_u8(vres3_lo), 2); o3 += 2;
        vst1_lane_u16((void*) o2, vreinterpret_u16_u8(vres2_lo), 2); o2 += 2;
        vst1_lane_u16((void*) o1, vreinterpret_u16_u8(vres1_lo), 2); o1 += 2;
        vst1_lane_u16((void*) o0, vreinterpret_u16_u8(vres0_lo), 2); o0 += 2;
        vres0_lo = vget_low_u8(vextq_u8(vres0, vres0, 6));
        vres1_lo = vget_low_u8(vextq_u8(vres1, vres1, 6));
        vres2_lo = vget_low_u8(vextq_u8(vres2, vres2, 6));
        vres3_lo = vget_low_u8(vextq_u8(vres3, vres3, 6));
      }
      if (bh & 1) {
        vst1_lane_u16((void*) o3, vreinterpret_u16_u8(vres3_lo), 0); o3 += 2;
        vst1_lane_u16((void*) o2, vreinterpret_u16_u8(vres2_lo), 0); o2 += 2;
        vst1_lane_u16((void*) o1, vreinterpret_u16_u8(vres1_lo), 0); o1 += 2;
        vst1_lane_u16((void*) o0, vreinterpret_u16_u8(vres0_lo), 0); o0 += 2;
        vst1_lane_u8(o3, vres3_lo, 2); o3 += 1;
        vst1_lane_u8(o2, vres2_lo, 2); o2 += 1;
        vst1_lane_u8(o1, vres1_lo, 2); o1 += 1;
        vst1_lane_u8(o0, vres0_lo, 2); o0 += 1;
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

void xnn_x32_transposec_ukernel__4x4_aarch64_neon_tbl128(
    const uint32_t* input,
    uint32_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{  
  static const uint8_t pos0[16] = {0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51};
  static const uint8_t pos1[16] = {4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55};
  static const uint8_t pos2[16] = {8, 9, 10, 11, 24, 25, 26, 27, 40, 41, 42, 43, 56, 57, 58, 59};
  static const uint8_t pos3[16] = {12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63};

  assert(block_width == 1 || output_stride >= block_height * sizeof(uint32_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint32_t));

  const size_t tile_height = 4;
  const size_t tile_width = 4;
  const size_t tile_wbytes = tile_width * sizeof(uint32_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_height * output_stride - round_down_po2(block_height, 2) * sizeof(uint32_t);
  const size_t tile_stride = tile_height * input_stride;

  const uint8_t* i0 = (const uint8_t*) input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);

  uint8_t* o0 = (uint8_t*) output;
  uint8_t* o1 = (uint8_t*) ((uintptr_t) o0 + output_stride);
  uint8_t* o2 = (uint8_t*) ((uintptr_t) o1 + output_stride);
  uint8_t* o3 = (uint8_t*) ((uintptr_t) o2 + output_stride);

  const uint8x16_t vperm0 = vld1q_u8(pos0);
  const uint8x16_t vperm1 = vld1q_u8(pos1);
  const uint8x16_t vperm2 = vld1q_u8(pos2);
  const uint8x16_t vperm3 = vld1q_u8(pos3);
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
      uint8x16x4_t v;
      v.val[0] = vld1q_u8(i0); i0 = (const uint8_t*) ((uintptr_t) i0 + tile_stride);
      v.val[1] = vld1q_u8(i1); i1 = (const uint8_t*) ((uintptr_t) i1 + tile_stride);
      v.val[2] = vld1q_u8(i2); i2 = (const uint8_t*) ((uintptr_t) i2 + tile_stride);
      v.val[3] = vld1q_u8(i3); i3 = (const uint8_t*) ((uintptr_t) i3 + tile_stride);

      uint8x16_t vres0 = vqtbl4q_u8(v, vperm0);
      uint8x16_t vres1 = vqtbl4q_u8(v, vperm1);
      uint8x16_t vres2 = vqtbl4q_u8(v, vperm2);
      uint8x16_t vres3 = vqtbl4q_u8(v, vperm3);

      vst1q_u8(o3, vres3); o3 = (uint8_t*) ((uintptr_t) o3 + tile_wbytes);
      vst1q_u8(o2, vres2); o2 = (uint8_t*) ((uintptr_t) o2 + tile_wbytes);
      vst1q_u8(o1, vres1); o1 = (uint8_t*) ((uintptr_t) o1 + tile_wbytes);
      vst1q_u8(o0, vres0); o0 = (uint8_t*) ((uintptr_t) o0 + tile_wbytes);
    }

    if (bh != 0) {
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i0;
      }
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      uint8x16x4_t v;
      v.val[0] = vld1q_u8(i0);
      v.val[1] = vld1q_u8(i1);
      v.val[2] = vld1q_u8(i2);

      uint8x16_t vres0 = vqtbl4q_u8(v, vperm0);
      uint8x16_t vres1 = vqtbl4q_u8(v, vperm1);
      uint8x16_t vres2 = vqtbl4q_u8(v, vperm2);
      uint8x16_t vres3 = vqtbl4q_u8(v, vperm3);

      uint8x8_t vres0_low = vget_low_u8(vres0);
      uint8x8_t vres1_low = vget_low_u8(vres1);
      uint8x8_t vres2_low = vget_low_u8(vres2);
      uint8x8_t vres3_low = vget_low_u8(vres3);

      if (bh & 2) {
        vst1_u8(o3, vres3_low); o3 += 8;
        vst1_u8(o2, vres2_low); o2 += 8;
        vst1_u8(o1, vres1_low); o1 += 8;
        vst1_u8(o0, vres0_low); o0 += 8;
        vres0_low = vget_high_u8(vres0);
        vres1_low = vget_high_u8(vres1);
        vres2_low = vget_high_u8(vres2);
        vres3_low = vget_high_u8(vres3);
      }
      if (bh & 1) {
        vst1_lane_u32((void*) o3, vreinterpret_u32_u8(vres3_low), 0);
        vst1_lane_u32((void*) o2, vreinterpret_u32_u8(vres2_low), 0);
        vst1_lane_u32((void*) o1, vreinterpret_u32_u8(vres1_low), 0);
        vst1_lane_u32((void*) o0, vreinterpret_u32_u8(vres0_low), 0);
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

void xnn_x8_lut_ukernel__aarch64_neon_tbx128x4_u64(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const uint8_t table[restrict XNN_MIN_ELEMENTS(256)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint8x16x4_t vtable0123 = vld1q_u8_x4(table);
  const uint8x16x4_t vtable4567 = vld1q_u8_x4(table + 64);
  const uint8x16x4_t vtable89AB = vld1q_u8_x4(table + 128);
  const uint8x16x4_t vtableCDEF = vld1q_u8_x4(table + 192);
  const uint8x16_t voffset = vmovq_n_u8(64);
  for (; batch >= 64 * sizeof(uint8_t); batch -= 64 * sizeof(uint8_t)) {
    uint8x16_t vx0 = vld1q_u8(input); input += 16;
    uint8x16_t vx1 = vld1q_u8(input); input += 16;
    uint8x16_t vx2 = vld1q_u8(input); input += 16;
    uint8x16_t vx3 = vld1q_u8(input); input += 16;

    uint8x16_t vy0 = vqtbl4q_u8(vtable0123, vx0);
    vx0 = vsubq_u8(vx0, voffset);
    uint8x16_t vy1 = vqtbl4q_u8(vtable0123, vx1);
    vx1 = vsubq_u8(vx1, voffset);
    uint8x16_t vy2 = vqtbl4q_u8(vtable0123, vx2);
    vx2 = vsubq_u8(vx2, voffset);
    uint8x16_t vy3 = vqtbl4q_u8(vtable0123, vx3);
    vx3 = vsubq_u8(vx3, voffset);

    vy0 = vqtbx4q_u8(vy0, vtable4567, vx0);
    vx0 = vsubq_u8(vx0, voffset);
    vy1 = vqtbx4q_u8(vy1, vtable4567, vx1);
    vx1 = vsubq_u8(vx1, voffset);
    vy2 = vqtbx4q_u8(vy2, vtable4567, vx2);
    vx2 = vsubq_u8(vx2, voffset);
    vy3 = vqtbx4q_u8(vy3, vtable4567, vx3);
    vx3 = vsubq_u8(vx3, voffset);

    vy0 = vqtbx4q_u8(vy0, vtable89AB, vx0);
    vx0 = vsubq_u8(vx0, voffset);
    vy1 = vqtbx4q_u8(vy1, vtable89AB, vx1);
    vx1 = vsubq_u8(vx1, voffset);
    vy2 = vqtbx4q_u8(vy2, vtable89AB, vx2);
    vx2 = vsubq_u8(vx2, voffset);
    vy3 = vqtbx4q_u8(vy3, vtable89AB, vx3);
    vx3 = vsubq_u8(vx3, voffset);

    vy0 = vqtbx4q_u8(vy0, vtableCDEF, vx0);
    vy1 = vqtbx4q_u8(vy1, vtableCDEF, vx1);
    vy2 = vqtbx4q_u8(vy2, vtableCDEF, vx2);
    vy3 = vqtbx4q_u8(vy3, vtableCDEF, vx3);

    vst1q_u8(output, vy0); output += 16;
    vst1q_u8(output, vy1); output += 16;
    vst1q_u8(output, vy2); output += 16;
    vst1q_u8(output, vy3); output += 16;
  }
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    uint8x16_t vx = vld1q_u8(input); input += 16;

    uint8x16_t vy = vqtbl4q_u8(vtable0123, vx);

    vx = vsubq_u8(vx, voffset);
    vy = vqtbx4q_u8(vy, vtable4567, vx);

    vx = vsubq_u8(vx, voffset);
    vy = vqtbx4q_u8(vy, vtable89AB, vx);

    vx = vsubq_u8(vx, voffset);
    vy = vqtbx4q_u8(vy, vtableCDEF, vx);

    vst1q_u8(output, vy); output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    uint8x16_t vx = vld1q_u8(input);

    uint8x16_t vy = vqtbl4q_u8(vtable0123, vx);

    vx = vsubq_u8(vx, voffset);
    vy = vqtbx4q_u8(vy, vtable4567, vx);

    vx = vsubq_u8(vx, voffset);
    vy = vqtbx4q_u8(vy, vtable89AB, vx);

    vx = vsubq_u8(vx, voffset);
    vy = vqtbx4q_u8(vy, vtableCDEF, vx);

    uint8x8_t vy_lo = vget_low_u8(vy);
    if (batch & (8 * sizeof(uint8_t))) {
      vst1_u8(output, vy_lo); output += 8;
      vy_lo = vget_high_u8(vy);
    }
    if (batch & (4 * sizeof(uint8_t))) {
      vst1_lane_u32((void*) output, vreinterpret_u32_u8(vy_lo), 0); output += 4;
      vy_lo = vext_u8(vy_lo, vy_lo, 4);
    }
    if (batch & (2 * sizeof(uint8_t))) {
      vst1_lane_u16((void*) output, vreinterpret_u16_u8(vy_lo), 0); output += 2;
      vy_lo = vext_u8(vy_lo, vy_lo, 2);
    }
    if (batch & (1 * sizeof(uint8_t))) {
      vst1_lane_u8(output, vy_lo, 0);
    }
  }
}

void xnn_x8_packq_f32qp8_ukernel__aarch64_neon_u2(size_t m, size_t k, size_t mr,
                                          size_t kr, size_t sr,
                                          size_t m_idx_start,
                                          const float* XNN_RESTRICT lhs,
                                          size_t lhs_stride,
                                          void* XNN_RESTRICT lhs_packed) {
#if XNN_ENABLE_KLEIDIAI
  kai_run_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr, m_idx_start, lhs,
                                     lhs_stride, lhs_packed);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
#endif  // XNN_ENABLE_KLEIDIAI
}
