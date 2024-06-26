// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv2d-chw/5x5p2-neon.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_3x4_acc2(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top == 2);

  const uint32x4_t vmask = vld1q_u32(params->neon_stride1.mask);
  const float32x4_t vmax = vld1q_dup_f32(&params->neon_stride1.max);
  const float32x4_t vmin = vld1q_dup_f32(&params->neon_stride1.min);

  const float32x4_t vw0123 = vld1q_f32(weights);
  const float32x4_t vw4567 = vld1q_f32(weights + 4);
  const float32x4_t vw89AB = vld1q_f32(weights + 8);
  const float32x4_t vwCDEF = vld1q_f32(weights + 12);
  const float32x4_t vwGHIJ = vld1q_f32(weights + 16);
  const float32x4_t vwKLMN = vld1q_f32(weights + 20);
  const float32x2_t vwOP = vld1_f32(weights + 24);

  const size_t input_decrement = round_up_po2(input_width, 4 * sizeof(float));

  const float* i0 = zero;
  const float* i1 = zero;
  const float* i2 = input;
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_width);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + input_width);
  float* o2 = (float*) ((uintptr_t) o1 + input_width);

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i3 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(output_height < 3) {
      i4 = zero;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(output_height < 4) {
      i5 = zero;
    }
    if XNN_UNPREDICTABLE(output_height < 5) {
      i6 = zero;
    }

    float32x4_t vi0x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi1x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi2x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi3x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi4x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi5x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi6x0123 = vmovq_n_f32(0.0f);

    float32x4_t vi0x4567 = vld1q_f32(i0); i0 += 4;
    float32x4_t vi1x4567 = vld1q_f32(i1); i1 += 4;
    float32x4_t vi2x4567 = vld1q_f32(i2); i2 += 4;
    float32x4_t vi3x4567 = vld1q_f32(i3); i3 += 4;
    float32x4_t vi4x4567 = vld1q_f32(i4); i4 += 4;
    float32x4_t vi5x4567 = vld1q_f32(i5); i5 += 4;
    float32x4_t vi6x4567 = vld1q_f32(i6); i6 += 4;

    size_t w = input_width;
    for (; w > 8 * sizeof(float); w -= 4 * sizeof(float)) {
      float32x4_t vo0p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo1p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo2p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);

      const float32x4_t vi0x89AB = vld1q_f32(i0); i0 += 4;
      const float32x4_t vi1x89AB = vld1q_f32(i1); i1 += 4;
      const float32x4_t vi2x89AB = vld1q_f32(i2); i2 += 4;
      const float32x4_t vi3x89AB = vld1q_f32(i3); i3 += 4;
      const float32x4_t vi4x89AB = vld1q_f32(i4); i4 += 4;
      const float32x4_t vi5x89AB = vld1q_f32(i5); i5 += 4;
      const float32x4_t vi6x89AB = vld1q_f32(i6); i6 += 4;

      float32x4_t vo0p1 = vmulq_lane_f32(vi0x4567, vget_high_f32(vw0123), 1);
      float32x4_t vo1p1 = vmulq_lane_f32(vi1x4567, vget_high_f32(vw0123), 1);
      float32x4_t vo2p1 = vmulq_lane_f32(vi2x4567, vget_high_f32(vw0123), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x4567, vget_low_f32(vw89AB), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x4567, vget_low_f32(vw89AB), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi3x4567, vget_low_f32(vw89AB), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x4567, vget_low_f32(vwCDEF), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x4567, vget_low_f32(vwCDEF), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x4567, vget_low_f32(vwCDEF), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi3x4567, vget_high_f32(vwGHIJ), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi4x4567, vget_high_f32(vwGHIJ), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi5x4567, vget_high_f32(vwGHIJ), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi4x4567, vget_high_f32(vwKLMN), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi5x4567, vget_high_f32(vwKLMN), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi6x4567, vget_high_f32(vwKLMN), 1);

      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);
      const float32x4_t vi3x3456 = vextq_f32(vi3x0123, vi3x4567, 3);
      const float32x4_t vi4x3456 = vextq_f32(vi4x0123, vi4x4567, 3);
      const float32x4_t vi5x3456 = vextq_f32(vi5x0123, vi5x4567, 3);
      const float32x4_t vi6x3456 = vextq_f32(vi6x0123, vi6x4567, 3);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi0x3456, vget_high_f32(vw0123), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi1x3456, vget_high_f32(vw0123), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi2x3456, vget_high_f32(vw0123), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x3456, vget_high_f32(vw4567), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x3456, vget_high_f32(vw4567), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi3x3456, vget_high_f32(vw4567), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi2x3456, vget_low_f32(vwCDEF), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi3x3456, vget_low_f32(vwCDEF), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi4x3456, vget_low_f32(vwCDEF), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi3x3456, vget_low_f32(vwGHIJ), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi4x3456, vget_low_f32(vwGHIJ), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi5x3456, vget_low_f32(vwGHIJ), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi4x3456, vget_high_f32(vwKLMN), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi5x3456, vget_high_f32(vwKLMN), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi6x3456, vget_high_f32(vwKLMN), 0);

      const float32x4_t vi0x2345 = vextq_f32(vi0x0123, vi0x4567, 2);
      vi0x0123 = vi0x4567;
      const float32x4_t vi1x2345 = vextq_f32(vi1x0123, vi1x4567, 2);
      vi1x0123 = vi1x4567;
      const float32x4_t vi2x2345 = vextq_f32(vi2x0123, vi2x4567, 2);
      vi2x0123 = vi2x4567;
      const float32x4_t vi3x2345 = vextq_f32(vi3x0123, vi3x4567, 2);
      vi3x0123 = vi3x4567;
      const float32x4_t vi4x2345 = vextq_f32(vi4x0123, vi4x4567, 2);
      vi4x0123 = vi4x4567;
      const float32x4_t vi5x2345 = vextq_f32(vi5x0123, vi5x4567, 2);
      vi5x0123 = vi5x4567;
      const float32x4_t vi6x2345 = vextq_f32(vi6x0123, vi6x4567, 2);
      vi6x0123 = vi6x4567;

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x2345, vget_low_f32(vw0123), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi1x2345, vget_low_f32(vw0123), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi2x2345, vget_low_f32(vw0123), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi1x2345, vget_high_f32(vw4567), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi2x2345, vget_high_f32(vw4567), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi3x2345, vget_high_f32(vw4567), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x2345, vget_high_f32(vw89AB), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x2345, vget_high_f32(vw89AB), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x2345, vget_high_f32(vw89AB), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi3x2345, vget_low_f32(vwGHIJ), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi4x2345, vget_low_f32(vwGHIJ), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi5x2345, vget_low_f32(vwGHIJ), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi4x2345, vget_low_f32(vwKLMN), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi5x2345, vget_low_f32(vwKLMN), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi6x2345, vget_low_f32(vwKLMN), 1);

      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vi0x89AB, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vi1x89AB, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vi2x89AB, 1);
      const float32x4_t vi3x5678 = vextq_f32(vi3x4567, vi3x89AB, 1);
      const float32x4_t vi4x5678 = vextq_f32(vi4x4567, vi4x89AB, 1);
      const float32x4_t vi5x5678 = vextq_f32(vi5x4567, vi5x89AB, 1);
      const float32x4_t vi6x5678 = vextq_f32(vi6x4567, vi6x89AB, 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi0x5678, vget_low_f32(vw4567), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi1x5678, vget_low_f32(vw4567), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi2x5678, vget_low_f32(vw4567), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x5678, vget_low_f32(vw89AB), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x5678, vget_low_f32(vw89AB), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi3x5678, vget_low_f32(vw89AB), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi2x5678, vget_high_f32(vwCDEF), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi3x5678, vget_high_f32(vwCDEF), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi4x5678, vget_high_f32(vwCDEF), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi3x5678, vget_high_f32(vwGHIJ), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi4x5678, vget_high_f32(vwGHIJ), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi5x5678, vget_high_f32(vwGHIJ), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi4x5678, vwOP, 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi5x5678, vwOP, 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi6x5678, vwOP, 0);

      const float32x4_t vi0x6789 = vextq_f32(vi0x4567, vi0x89AB, 2);
      vi0x4567 = vi0x89AB;
      const float32x4_t vi1x6789 = vextq_f32(vi1x4567, vi1x89AB, 2);
      vi1x4567 = vi1x89AB;
      const float32x4_t vi2x6789 = vextq_f32(vi2x4567, vi2x89AB, 2);
      vi2x4567 = vi2x89AB;
      const float32x4_t vi3x6789 = vextq_f32(vi3x4567, vi3x89AB, 2);
      vi3x4567 = vi3x89AB;
      const float32x4_t vi4x6789 = vextq_f32(vi4x4567, vi4x89AB, 2);
      vi4x4567 = vi4x89AB;
      const float32x4_t vi5x6789 = vextq_f32(vi5x4567, vi5x89AB, 2);
      vi5x4567 = vi5x89AB;
      const float32x4_t vi6x6789 = vextq_f32(vi6x4567, vi6x89AB, 2);
      vi6x4567 = vi6x89AB;

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x6789, vget_low_f32(vw4567), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi1x6789, vget_low_f32(vw4567), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi2x6789, vget_low_f32(vw4567), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi1x6789, vget_high_f32(vw89AB), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi2x6789, vget_high_f32(vw89AB), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi3x6789, vget_high_f32(vw89AB), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x6789, vget_high_f32(vwCDEF), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x6789, vget_high_f32(vwCDEF), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x6789, vget_high_f32(vwCDEF), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi3x6789, vget_low_f32(vwKLMN), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi4x6789, vget_low_f32(vwKLMN), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi5x6789, vget_low_f32(vwKLMN), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi4x6789, vwOP, 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi5x6789, vwOP, 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi6x6789, vwOP, 1);

      vo0p0 = vaddq_f32(vo0p0, vo0p1);
      vo1p0 = vaddq_f32(vo1p0, vo1p1);
      vo2p0 = vaddq_f32(vo2p0, vo2p1);

      float32x4_t vo0 = vmaxq_f32(vo0p0, vmin);
      float32x4_t vo1 = vmaxq_f32(vo1p0, vmin);
      float32x4_t vo2 = vmaxq_f32(vo2p0, vmin);

      vo0 = vminq_f32(vo0, vmax);
      vo1 = vminq_f32(vo1, vmax);
      vo2 = vminq_f32(vo2, vmax);

      vst1q_f32(o2, vo2); o2 += 4;
      vst1q_f32(o1, vo1); o1 += 4;
      vst1q_f32(o0, vo0); o0 += 4;
    }
    // Always process the last block of 5..8 pixels.
    if XNN_LIKELY(w > 4 * sizeof(float)) {
      float32x4_t vo0p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo1p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo2p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);

      float32x4_t vi0x89AB = vld1q_f32(i0); i0 += 4;
      float32x4_t vi1x89AB = vld1q_f32(i1); i1 += 4;
      float32x4_t vi2x89AB = vld1q_f32(i2); i2 += 4;
      float32x4_t vi3x89AB = vld1q_f32(i3); i3 += 4;
      float32x4_t vi4x89AB = vld1q_f32(i4); i4 += 4;
      float32x4_t vi5x89AB = vld1q_f32(i5); i5 += 4;
      float32x4_t vi6x89AB = vld1q_f32(i6); i6 += 4;

      vi0x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi0x89AB)));
      vi1x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi1x89AB)));
      vi2x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi2x89AB)));
      vi3x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi3x89AB)));
      vi4x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi4x89AB)));
      vi5x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi5x89AB)));
      vi6x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi6x89AB)));

      float32x4_t vo0p1 = vmulq_lane_f32(vi0x4567, vget_high_f32(vw0123), 1);
      float32x4_t vo1p1 = vmulq_lane_f32(vi1x4567, vget_high_f32(vw0123), 1);
      float32x4_t vo2p1 = vmulq_lane_f32(vi2x4567, vget_high_f32(vw0123), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x4567, vget_low_f32(vw89AB), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x4567, vget_low_f32(vw89AB), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi3x4567, vget_low_f32(vw89AB), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x4567, vget_low_f32(vwCDEF), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x4567, vget_low_f32(vwCDEF), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x4567, vget_low_f32(vwCDEF), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi3x4567, vget_high_f32(vwGHIJ), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi4x4567, vget_high_f32(vwGHIJ), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi5x4567, vget_high_f32(vwGHIJ), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi4x4567, vget_high_f32(vwKLMN), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi5x4567, vget_high_f32(vwKLMN), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi6x4567, vget_high_f32(vwKLMN), 1);

      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);
      const float32x4_t vi3x3456 = vextq_f32(vi3x0123, vi3x4567, 3);
      const float32x4_t vi4x3456 = vextq_f32(vi4x0123, vi4x4567, 3);
      const float32x4_t vi5x3456 = vextq_f32(vi5x0123, vi5x4567, 3);
      const float32x4_t vi6x3456 = vextq_f32(vi6x0123, vi6x4567, 3);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi0x3456, vget_high_f32(vw0123), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi1x3456, vget_high_f32(vw0123), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi2x3456, vget_high_f32(vw0123), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x3456, vget_high_f32(vw4567), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x3456, vget_high_f32(vw4567), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi3x3456, vget_high_f32(vw4567), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi2x3456, vget_low_f32(vwCDEF), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi3x3456, vget_low_f32(vwCDEF), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi4x3456, vget_low_f32(vwCDEF), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi3x3456, vget_low_f32(vwGHIJ), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi4x3456, vget_low_f32(vwGHIJ), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi5x3456, vget_low_f32(vwGHIJ), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi4x3456, vget_high_f32(vwKLMN), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi5x3456, vget_high_f32(vwKLMN), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi6x3456, vget_high_f32(vwKLMN), 0);

      const float32x4_t vi0x2345 = vextq_f32(vi0x0123, vi0x4567, 2);
      vi0x0123 = vi0x4567;
      const float32x4_t vi1x2345 = vextq_f32(vi1x0123, vi1x4567, 2);
      vi1x0123 = vi1x4567;
      const float32x4_t vi2x2345 = vextq_f32(vi2x0123, vi2x4567, 2);
      vi2x0123 = vi2x4567;
      const float32x4_t vi3x2345 = vextq_f32(vi3x0123, vi3x4567, 2);
      vi3x0123 = vi3x4567;
      const float32x4_t vi4x2345 = vextq_f32(vi4x0123, vi4x4567, 2);
      vi4x0123 = vi4x4567;
      const float32x4_t vi5x2345 = vextq_f32(vi5x0123, vi5x4567, 2);
      vi5x0123 = vi5x4567;
      const float32x4_t vi6x2345 = vextq_f32(vi6x0123, vi6x4567, 2);
      vi6x0123 = vi6x4567;

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x2345, vget_low_f32(vw0123), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi1x2345, vget_low_f32(vw0123), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi2x2345, vget_low_f32(vw0123), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi1x2345, vget_high_f32(vw4567), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi2x2345, vget_high_f32(vw4567), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi3x2345, vget_high_f32(vw4567), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x2345, vget_high_f32(vw89AB), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x2345, vget_high_f32(vw89AB), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x2345, vget_high_f32(vw89AB), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi3x2345, vget_low_f32(vwGHIJ), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi4x2345, vget_low_f32(vwGHIJ), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi5x2345, vget_low_f32(vwGHIJ), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi4x2345, vget_low_f32(vwKLMN), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi5x2345, vget_low_f32(vwKLMN), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi6x2345, vget_low_f32(vwKLMN), 1);

      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vi0x89AB, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vi1x89AB, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vi2x89AB, 1);
      const float32x4_t vi3x5678 = vextq_f32(vi3x4567, vi3x89AB, 1);
      const float32x4_t vi4x5678 = vextq_f32(vi4x4567, vi4x89AB, 1);
      const float32x4_t vi5x5678 = vextq_f32(vi5x4567, vi5x89AB, 1);
      const float32x4_t vi6x5678 = vextq_f32(vi6x4567, vi6x89AB, 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi0x5678, vget_low_f32(vw4567), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi1x5678, vget_low_f32(vw4567), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi2x5678, vget_low_f32(vw4567), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x5678, vget_low_f32(vw89AB), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x5678, vget_low_f32(vw89AB), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi3x5678, vget_low_f32(vw89AB), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi2x5678, vget_high_f32(vwCDEF), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi3x5678, vget_high_f32(vwCDEF), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi4x5678, vget_high_f32(vwCDEF), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi3x5678, vget_high_f32(vwGHIJ), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi4x5678, vget_high_f32(vwGHIJ), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi5x5678, vget_high_f32(vwGHIJ), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi4x5678, vwOP, 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi5x5678, vwOP, 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi6x5678, vwOP, 0);

      const float32x4_t vi0x6789 = vextq_f32(vi0x4567, vi0x89AB, 2);
      vi0x4567 = vi0x89AB;
      const float32x4_t vi1x6789 = vextq_f32(vi1x4567, vi1x89AB, 2);
      vi1x4567 = vi1x89AB;
      const float32x4_t vi2x6789 = vextq_f32(vi2x4567, vi2x89AB, 2);
      vi2x4567 = vi2x89AB;
      const float32x4_t vi3x6789 = vextq_f32(vi3x4567, vi3x89AB, 2);
      vi3x4567 = vi3x89AB;
      const float32x4_t vi4x6789 = vextq_f32(vi4x4567, vi4x89AB, 2);
      vi4x4567 = vi4x89AB;
      const float32x4_t vi5x6789 = vextq_f32(vi5x4567, vi5x89AB, 2);
      vi5x4567 = vi5x89AB;
      const float32x4_t vi6x6789 = vextq_f32(vi6x4567, vi6x89AB, 2);
      vi6x4567 = vi6x89AB;

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x6789, vget_low_f32(vw4567), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi1x6789, vget_low_f32(vw4567), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi2x6789, vget_low_f32(vw4567), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi1x6789, vget_high_f32(vw89AB), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi2x6789, vget_high_f32(vw89AB), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi3x6789, vget_high_f32(vw89AB), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x6789, vget_high_f32(vwCDEF), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x6789, vget_high_f32(vwCDEF), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x6789, vget_high_f32(vwCDEF), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi3x6789, vget_low_f32(vwKLMN), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi4x6789, vget_low_f32(vwKLMN), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi5x6789, vget_low_f32(vwKLMN), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi4x6789, vwOP, 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi5x6789, vwOP, 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi6x6789, vwOP, 1);

      vo0p0 = vaddq_f32(vo0p0, vo0p1);
      vo1p0 = vaddq_f32(vo1p0, vo1p1);
      vo2p0 = vaddq_f32(vo2p0, vo2p1);

      float32x4_t vo0 = vmaxq_f32(vo0p0, vmin);
      float32x4_t vo1 = vmaxq_f32(vo1p0, vmin);
      float32x4_t vo2 = vmaxq_f32(vo2p0, vmin);

      vo0 = vminq_f32(vo0, vmax);
      vo1 = vminq_f32(vo1, vmax);
      vo2 = vminq_f32(vo2, vmax);

      vst1q_f32(o2, vo2); o2 += 4;
      vst1q_f32(o1, vo1); o1 += 4;
      vst1q_f32(o0, vo0); o0 += 4;

      w -= 4 * sizeof(float);
    }
    assert(w >= 1 * sizeof(float));
    assert(w <= 4 * sizeof(float));
    {
      float32x4_t vo0p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo1p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo2p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);

      vi0x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi0x4567)));
      vi1x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi1x4567)));
      vi2x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi2x4567)));
      vi3x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi3x4567)));
      vi4x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi4x4567)));
      vi5x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi5x4567)));
      vi6x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi6x4567)));

      float32x4_t vo0p1 = vmulq_lane_f32(vi0x4567, vget_high_f32(vw0123), 1);
      float32x4_t vo1p1 = vmulq_lane_f32(vi1x4567, vget_high_f32(vw0123), 1);
      float32x4_t vo2p1 = vmulq_lane_f32(vi2x4567, vget_high_f32(vw0123), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x4567, vget_low_f32(vw89AB), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x4567, vget_low_f32(vw89AB), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi3x4567, vget_low_f32(vw89AB), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x4567, vget_low_f32(vwCDEF), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x4567, vget_low_f32(vwCDEF), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x4567, vget_low_f32(vwCDEF), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi3x4567, vget_high_f32(vwGHIJ), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi4x4567, vget_high_f32(vwGHIJ), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi5x4567, vget_high_f32(vwGHIJ), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi4x4567, vget_high_f32(vwKLMN), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi5x4567, vget_high_f32(vwKLMN), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi6x4567, vget_high_f32(vwKLMN), 1);

      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);
      const float32x4_t vi3x3456 = vextq_f32(vi3x0123, vi3x4567, 3);
      const float32x4_t vi4x3456 = vextq_f32(vi4x0123, vi4x4567, 3);
      const float32x4_t vi5x3456 = vextq_f32(vi5x0123, vi5x4567, 3);
      const float32x4_t vi6x3456 = vextq_f32(vi6x0123, vi6x4567, 3);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi0x3456, vget_high_f32(vw0123), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi1x3456, vget_high_f32(vw0123), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi2x3456, vget_high_f32(vw0123), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x3456, vget_high_f32(vw4567), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x3456, vget_high_f32(vw4567), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi3x3456, vget_high_f32(vw4567), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi2x3456, vget_low_f32(vwCDEF), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi3x3456, vget_low_f32(vwCDEF), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi4x3456, vget_low_f32(vwCDEF), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi3x3456, vget_low_f32(vwGHIJ), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi4x3456, vget_low_f32(vwGHIJ), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi5x3456, vget_low_f32(vwGHIJ), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi4x3456, vget_high_f32(vwKLMN), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi5x3456, vget_high_f32(vwKLMN), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi6x3456, vget_high_f32(vwKLMN), 0);

      const float32x4_t vi0x2345 = vextq_f32(vi0x0123, vi0x4567, 2);
      const float32x4_t vi1x2345 = vextq_f32(vi1x0123, vi1x4567, 2);
      const float32x4_t vi2x2345 = vextq_f32(vi2x0123, vi2x4567, 2);
      const float32x4_t vi3x2345 = vextq_f32(vi3x0123, vi3x4567, 2);
      const float32x4_t vi4x2345 = vextq_f32(vi4x0123, vi4x4567, 2);
      const float32x4_t vi5x2345 = vextq_f32(vi5x0123, vi5x4567, 2);
      const float32x4_t vi6x2345 = vextq_f32(vi6x0123, vi6x4567, 2);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x2345, vget_low_f32(vw0123), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi1x2345, vget_low_f32(vw0123), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi2x2345, vget_low_f32(vw0123), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi1x2345, vget_high_f32(vw4567), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi2x2345, vget_high_f32(vw4567), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi3x2345, vget_high_f32(vw4567), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x2345, vget_high_f32(vw89AB), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x2345, vget_high_f32(vw89AB), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x2345, vget_high_f32(vw89AB), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi3x2345, vget_low_f32(vwGHIJ), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi4x2345, vget_low_f32(vwGHIJ), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi5x2345, vget_low_f32(vwGHIJ), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi4x2345, vget_low_f32(vwKLMN), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi5x2345, vget_low_f32(vwKLMN), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi6x2345, vget_low_f32(vwKLMN), 1);

      const float32x4_t vzero = vmovq_n_f32(0.0f);
      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vzero, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vzero, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vzero, 1);
      const float32x4_t vi3x5678 = vextq_f32(vi3x4567, vzero, 1);
      const float32x4_t vi4x5678 = vextq_f32(vi4x4567, vzero, 1);
      const float32x4_t vi5x5678 = vextq_f32(vi5x4567, vzero, 1);
      const float32x4_t vi6x5678 = vextq_f32(vi6x4567, vzero, 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi0x5678, vget_low_f32(vw4567), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi1x5678, vget_low_f32(vw4567), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi2x5678, vget_low_f32(vw4567), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x5678, vget_low_f32(vw89AB), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x5678, vget_low_f32(vw89AB), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi3x5678, vget_low_f32(vw89AB), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi2x5678, vget_high_f32(vwCDEF), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi3x5678, vget_high_f32(vwCDEF), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi4x5678, vget_high_f32(vwCDEF), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi3x5678, vget_high_f32(vwGHIJ), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi4x5678, vget_high_f32(vwGHIJ), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi5x5678, vget_high_f32(vwGHIJ), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi4x5678, vwOP, 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi5x5678, vwOP, 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi6x5678, vwOP, 0);

      const float32x4_t vi0x6789 = vextq_f32(vi0x5678, vzero, 1);
      const float32x4_t vi1x6789 = vextq_f32(vi1x5678, vzero, 1);
      const float32x4_t vi2x6789 = vextq_f32(vi2x5678, vzero, 1);
      const float32x4_t vi3x6789 = vextq_f32(vi3x5678, vzero, 1);
      const float32x4_t vi4x6789 = vextq_f32(vi4x5678, vzero, 1);
      const float32x4_t vi5x6789 = vextq_f32(vi5x5678, vzero, 1);
      const float32x4_t vi6x6789 = vextq_f32(vi6x5678, vzero, 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x6789, vget_low_f32(vw4567), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi1x6789, vget_low_f32(vw4567), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi2x6789, vget_low_f32(vw4567), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi1x6789, vget_high_f32(vw89AB), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi2x6789, vget_high_f32(vw89AB), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi3x6789, vget_high_f32(vw89AB), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x6789, vget_high_f32(vwCDEF), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x6789, vget_high_f32(vwCDEF), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x6789, vget_high_f32(vwCDEF), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi3x6789, vget_low_f32(vwKLMN), 0);
      vo1p1 = vmlaq_lane_f32(vo1p1, vi4x6789, vget_low_f32(vwKLMN), 0);
      vo2p1 = vmlaq_lane_f32(vo2p1, vi5x6789, vget_low_f32(vwKLMN), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi4x6789, vwOP, 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi5x6789, vwOP, 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi6x6789, vwOP, 1);

      vo0p0 = vaddq_f32(vo0p0, vo0p1);
      vo1p0 = vaddq_f32(vo1p0, vo1p1);
      vo2p0 = vaddq_f32(vo2p0, vo2p1);

      float32x4_t vo0 = vmaxq_f32(vo0p0, vmin);
      float32x4_t vo1 = vmaxq_f32(vo1p0, vmin);
      float32x4_t vo2 = vmaxq_f32(vo2p0, vmin);

      vo0 = vminq_f32(vo0, vmax);
      vo1 = vminq_f32(vo1, vmax);
      vo2 = vminq_f32(vo2, vmax);

      if XNN_LIKELY(w & (4 * sizeof(float))) {
        vst1q_f32(o2, vo2); o2 += 4;
        vst1q_f32(o1, vo1); o1 += 4;
        vst1q_f32(o0, vo0); o0 += 4;
      } else {
        float32x2_t vo0_lo = vget_low_f32(vo0);
        float32x2_t vo1_lo = vget_low_f32(vo1);
        float32x2_t vo2_lo = vget_low_f32(vo2);
        if (w & (2 * sizeof(float))) {
          vst1_f32(o2, vo2_lo); o2 += 2;
          vst1_f32(o1, vo1_lo); o1 += 2;
          vst1_f32(o0, vo0_lo); o0 += 2;

          vo0_lo = vget_high_f32(vo0);
          vo1_lo = vget_high_f32(vo1);
          vo2_lo = vget_high_f32(vo2);
        }
        if (w & (1 * sizeof(float))) {
          vst1_lane_f32(o2, vo2_lo, 0); o2 += 1;
          vst1_lane_f32(o1, vo1_lo, 0); o1 += 1;
          vst1_lane_f32(o0, vo0_lo, 0); o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i3 - input_decrement);
    i1 = (const float*) ((uintptr_t) i4 - input_decrement);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);
    i5 = (const float*) ((uintptr_t) i4 + input_width);
    i6 = (const float*) ((uintptr_t) i5 + input_width);

    o0 = o2;
    o1 = (float*) ((uintptr_t) o0 + input_width);
    o2 = (float*) ((uintptr_t) o1 + input_width);

    output_height = doz(output_height, 3);
  } while (output_height != 0);
}
