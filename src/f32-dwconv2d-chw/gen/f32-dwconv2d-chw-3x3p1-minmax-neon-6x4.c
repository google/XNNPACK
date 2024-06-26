// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv2d-chw/3x3p1-neon.c.in
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


void xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_6x4(
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
  assert(padding_top == 1);

  const uint32x4_t vmask = vld1q_u32(params->neon_stride1.mask);
  const float32x4_t vmax = vld1q_dup_f32(&params->neon_stride1.max);
  const float32x4_t vmin = vld1q_dup_f32(&params->neon_stride1.min);

  const float32x4_t vw0123 = vld1q_f32(weights);
  const float32x4_t vw4567 = vld1q_f32(weights + 4);
  const float32x2_t vw89 = vld1_f32(weights + 8);

  const size_t input_decrement = round_up_po2(input_width, 4 * sizeof(float));

  const float* i0 = zero;
  const float* i1 = input;
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_width);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_width);
  const float* i7 = (const float*) ((uintptr_t) i6 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + input_width);
  float* o2 = (float*) ((uintptr_t) o1 + input_width);
  float* o3 = (float*) ((uintptr_t) o2 + input_width);
  float* o4 = (float*) ((uintptr_t) o3 + input_width);
  float* o5 = (float*) ((uintptr_t) o4 + input_width);

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i2 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(output_height < 3) {
      i3 = zero;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(output_height < 4) {
      i4 = zero;
      o3 = o2;
    }
    if XNN_UNPREDICTABLE(output_height < 5) {
      i5 = zero;
      o4 = o3;
    }
    if XNN_UNPREDICTABLE(output_height < 6) {
      i6 = zero;
      o5 = o4;
    }
    if XNN_UNPREDICTABLE(output_height < 7) {
      i7 = zero;
    }

    float32x4_t vi0x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi1x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi2x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi3x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi4x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi5x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi6x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi7x0123 = vmovq_n_f32(0.0f);

    float32x4_t vi0x4567 = vld1q_f32(i0); i0 += 4;
    float32x4_t vi1x4567 = vld1q_f32(i1); i1 += 4;
    float32x4_t vi2x4567 = vld1q_f32(i2); i2 += 4;
    float32x4_t vi3x4567 = vld1q_f32(i3); i3 += 4;
    float32x4_t vi4x4567 = vld1q_f32(i4); i4 += 4;
    float32x4_t vi5x4567 = vld1q_f32(i5); i5 += 4;
    float32x4_t vi6x4567 = vld1q_f32(i6); i6 += 4;
    float32x4_t vi7x4567 = vld1q_f32(i7); i7 += 4;

    size_t w = input_width;
    for (; w > 4 * sizeof(float); w -= 4 * sizeof(float)) {
      float32x4_t vo0p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo1p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo2p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo3p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo4p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo5p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);

      const float32x4_t vi0x89AB = vld1q_f32(i0); i0 += 4;
      const float32x4_t vi1x89AB = vld1q_f32(i1); i1 += 4;
      const float32x4_t vi2x89AB = vld1q_f32(i2); i2 += 4;
      const float32x4_t vi3x89AB = vld1q_f32(i3); i3 += 4;
      const float32x4_t vi4x89AB = vld1q_f32(i4); i4 += 4;
      const float32x4_t vi5x89AB = vld1q_f32(i5); i5 += 4;
      const float32x4_t vi6x89AB = vld1q_f32(i6); i6 += 4;
      const float32x4_t vi7x89AB = vld1q_f32(i7); i7 += 4;

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x4567, vget_high_f32(vw0123), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi1x4567, vget_high_f32(vw0123), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi2x4567, vget_high_f32(vw0123), 0);
      vo3p0 = vmlaq_lane_f32(vo3p0, vi3x4567, vget_high_f32(vw0123), 0);
      vo4p0 = vmlaq_lane_f32(vo4p0, vi4x4567, vget_high_f32(vw0123), 0);
      vo5p0 = vmlaq_lane_f32(vo5p0, vi5x4567, vget_high_f32(vw0123), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x4567, vget_low_f32(vw4567), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x4567, vget_low_f32(vw4567), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi3x4567, vget_low_f32(vw4567), 1);
      vo3p0 = vmlaq_lane_f32(vo3p0, vi4x4567, vget_low_f32(vw4567), 1);
      vo4p0 = vmlaq_lane_f32(vo4p0, vi5x4567, vget_low_f32(vw4567), 1);
      vo5p0 = vmlaq_lane_f32(vo5p0, vi6x4567, vget_low_f32(vw4567), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x4567, vw89, 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x4567, vw89, 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x4567, vw89, 0);
      vo3p0 = vmlaq_lane_f32(vo3p0, vi5x4567, vw89, 0);
      vo4p0 = vmlaq_lane_f32(vo4p0, vi6x4567, vw89, 0);
      vo5p0 = vmlaq_lane_f32(vo5p0, vi7x4567, vw89, 0);

      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);
      const float32x4_t vi3x3456 = vextq_f32(vi3x0123, vi3x4567, 3);
      const float32x4_t vi4x3456 = vextq_f32(vi4x0123, vi4x4567, 3);
      const float32x4_t vi5x3456 = vextq_f32(vi5x0123, vi5x4567, 3);
      const float32x4_t vi6x3456 = vextq_f32(vi6x0123, vi6x4567, 3);
      const float32x4_t vi7x3456 = vextq_f32(vi7x0123, vi7x4567, 3);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x3456, vget_low_f32(vw0123), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi1x3456, vget_low_f32(vw0123), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi2x3456, vget_low_f32(vw0123), 1);
      vo3p0 = vmlaq_lane_f32(vo3p0, vi3x3456, vget_low_f32(vw0123), 1);
      vo4p0 = vmlaq_lane_f32(vo4p0, vi4x3456, vget_low_f32(vw0123), 1);
      vo5p0 = vmlaq_lane_f32(vo5p0, vi5x3456, vget_low_f32(vw0123), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x3456, vget_low_f32(vw4567), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x3456, vget_low_f32(vw4567), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi3x3456, vget_low_f32(vw4567), 0);
      vo3p0 = vmlaq_lane_f32(vo3p0, vi4x3456, vget_low_f32(vw4567), 0);
      vo4p0 = vmlaq_lane_f32(vo4p0, vi5x3456, vget_low_f32(vw4567), 0);
      vo5p0 = vmlaq_lane_f32(vo5p0, vi6x3456, vget_low_f32(vw4567), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x3456, vget_high_f32(vw4567), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x3456, vget_high_f32(vw4567), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x3456, vget_high_f32(vw4567), 1);
      vo3p0 = vmlaq_lane_f32(vo3p0, vi5x3456, vget_high_f32(vw4567), 1);
      vo4p0 = vmlaq_lane_f32(vo4p0, vi6x3456, vget_high_f32(vw4567), 1);
      vo5p0 = vmlaq_lane_f32(vo5p0, vi7x3456, vget_high_f32(vw4567), 1);

      vi0x0123 = vi0x4567;
      vi1x0123 = vi1x4567;
      vi2x0123 = vi2x4567;
      vi3x0123 = vi3x4567;
      vi4x0123 = vi4x4567;
      vi5x0123 = vi5x4567;
      vi6x0123 = vi6x4567;
      vi7x0123 = vi7x4567;

      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vi0x89AB, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vi1x89AB, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vi2x89AB, 1);
      const float32x4_t vi3x5678 = vextq_f32(vi3x4567, vi3x89AB, 1);
      const float32x4_t vi4x5678 = vextq_f32(vi4x4567, vi4x89AB, 1);
      const float32x4_t vi5x5678 = vextq_f32(vi5x4567, vi5x89AB, 1);
      const float32x4_t vi6x5678 = vextq_f32(vi6x4567, vi6x89AB, 1);
      const float32x4_t vi7x5678 = vextq_f32(vi7x4567, vi7x89AB, 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x5678, vget_high_f32(vw0123), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi1x5678, vget_high_f32(vw0123), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi2x5678, vget_high_f32(vw0123), 1);
      vo3p0 = vmlaq_lane_f32(vo3p0, vi3x5678, vget_high_f32(vw0123), 1);
      vo4p0 = vmlaq_lane_f32(vo4p0, vi4x5678, vget_high_f32(vw0123), 1);
      vo5p0 = vmlaq_lane_f32(vo5p0, vi5x5678, vget_high_f32(vw0123), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x5678, vget_high_f32(vw4567), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x5678, vget_high_f32(vw4567), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi3x5678, vget_high_f32(vw4567), 0);
      vo3p0 = vmlaq_lane_f32(vo3p0, vi4x5678, vget_high_f32(vw4567), 0);
      vo4p0 = vmlaq_lane_f32(vo4p0, vi5x5678, vget_high_f32(vw4567), 0);
      vo5p0 = vmlaq_lane_f32(vo5p0, vi6x5678, vget_high_f32(vw4567), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x5678, vw89, 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x5678, vw89, 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x5678, vw89, 1);
      vo3p0 = vmlaq_lane_f32(vo3p0, vi5x5678, vw89, 1);
      vo4p0 = vmlaq_lane_f32(vo4p0, vi6x5678, vw89, 1);
      vo5p0 = vmlaq_lane_f32(vo5p0, vi7x5678, vw89, 1);

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;
      vi4x4567 = vi4x89AB;
      vi5x4567 = vi5x89AB;
      vi6x4567 = vi6x89AB;
      vi7x4567 = vi7x89AB;


      float32x4_t vo0 = vmaxq_f32(vo0p0, vmin);
      float32x4_t vo1 = vmaxq_f32(vo1p0, vmin);
      float32x4_t vo2 = vmaxq_f32(vo2p0, vmin);
      float32x4_t vo3 = vmaxq_f32(vo3p0, vmin);
      float32x4_t vo4 = vmaxq_f32(vo4p0, vmin);
      float32x4_t vo5 = vmaxq_f32(vo5p0, vmin);

      vo0 = vminq_f32(vo0, vmax);
      vo1 = vminq_f32(vo1, vmax);
      vo2 = vminq_f32(vo2, vmax);
      vo3 = vminq_f32(vo3, vmax);
      vo4 = vminq_f32(vo4, vmax);
      vo5 = vminq_f32(vo5, vmax);

      vst1q_f32(o5, vo5); o5 += 4;
      vst1q_f32(o4, vo4); o4 += 4;
      vst1q_f32(o3, vo3); o3 += 4;
      vst1q_f32(o2, vo2); o2 += 4;
      vst1q_f32(o1, vo1); o1 += 4;
      vst1q_f32(o0, vo0); o0 += 4;
    }
    // Always process the last block of 1..4 pixels.
    assert(w >= 1 * sizeof(float));
    assert(w <= 4 * sizeof(float));
    {
      float32x4_t vo0p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo1p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo2p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo3p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo4p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo5p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);

      vi0x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi0x4567)));
      vi1x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi1x4567)));
      vi2x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi2x4567)));
      vi3x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi3x4567)));
      vi4x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi4x4567)));
      vi5x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi5x4567)));
      vi6x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi6x4567)));
      vi7x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi7x4567)));

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x4567, vget_high_f32(vw0123), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi1x4567, vget_high_f32(vw0123), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi2x4567, vget_high_f32(vw0123), 0);
      vo3p0 = vmlaq_lane_f32(vo3p0, vi3x4567, vget_high_f32(vw0123), 0);
      vo4p0 = vmlaq_lane_f32(vo4p0, vi4x4567, vget_high_f32(vw0123), 0);
      vo5p0 = vmlaq_lane_f32(vo5p0, vi5x4567, vget_high_f32(vw0123), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x4567, vget_low_f32(vw4567), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x4567, vget_low_f32(vw4567), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi3x4567, vget_low_f32(vw4567), 1);
      vo3p0 = vmlaq_lane_f32(vo3p0, vi4x4567, vget_low_f32(vw4567), 1);
      vo4p0 = vmlaq_lane_f32(vo4p0, vi5x4567, vget_low_f32(vw4567), 1);
      vo5p0 = vmlaq_lane_f32(vo5p0, vi6x4567, vget_low_f32(vw4567), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x4567, vw89, 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x4567, vw89, 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x4567, vw89, 0);
      vo3p0 = vmlaq_lane_f32(vo3p0, vi5x4567, vw89, 0);
      vo4p0 = vmlaq_lane_f32(vo4p0, vi6x4567, vw89, 0);
      vo5p0 = vmlaq_lane_f32(vo5p0, vi7x4567, vw89, 0);

      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);
      const float32x4_t vi3x3456 = vextq_f32(vi3x0123, vi3x4567, 3);
      const float32x4_t vi4x3456 = vextq_f32(vi4x0123, vi4x4567, 3);
      const float32x4_t vi5x3456 = vextq_f32(vi5x0123, vi5x4567, 3);
      const float32x4_t vi6x3456 = vextq_f32(vi6x0123, vi6x4567, 3);
      const float32x4_t vi7x3456 = vextq_f32(vi7x0123, vi7x4567, 3);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x3456, vget_low_f32(vw0123), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi1x3456, vget_low_f32(vw0123), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi2x3456, vget_low_f32(vw0123), 1);
      vo3p0 = vmlaq_lane_f32(vo3p0, vi3x3456, vget_low_f32(vw0123), 1);
      vo4p0 = vmlaq_lane_f32(vo4p0, vi4x3456, vget_low_f32(vw0123), 1);
      vo5p0 = vmlaq_lane_f32(vo5p0, vi5x3456, vget_low_f32(vw0123), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x3456, vget_low_f32(vw4567), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x3456, vget_low_f32(vw4567), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi3x3456, vget_low_f32(vw4567), 0);
      vo3p0 = vmlaq_lane_f32(vo3p0, vi4x3456, vget_low_f32(vw4567), 0);
      vo4p0 = vmlaq_lane_f32(vo4p0, vi5x3456, vget_low_f32(vw4567), 0);
      vo5p0 = vmlaq_lane_f32(vo5p0, vi6x3456, vget_low_f32(vw4567), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x3456, vget_high_f32(vw4567), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x3456, vget_high_f32(vw4567), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x3456, vget_high_f32(vw4567), 1);
      vo3p0 = vmlaq_lane_f32(vo3p0, vi5x3456, vget_high_f32(vw4567), 1);
      vo4p0 = vmlaq_lane_f32(vo4p0, vi6x3456, vget_high_f32(vw4567), 1);
      vo5p0 = vmlaq_lane_f32(vo5p0, vi7x3456, vget_high_f32(vw4567), 1);

      const float32x4_t vzero = vmovq_n_f32(0.0f);
      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vzero, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vzero, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vzero, 1);
      const float32x4_t vi3x5678 = vextq_f32(vi3x4567, vzero, 1);
      const float32x4_t vi4x5678 = vextq_f32(vi4x4567, vzero, 1);
      const float32x4_t vi5x5678 = vextq_f32(vi5x4567, vzero, 1);
      const float32x4_t vi6x5678 = vextq_f32(vi6x4567, vzero, 1);
      const float32x4_t vi7x5678 = vextq_f32(vi7x4567, vzero, 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x5678, vget_high_f32(vw0123), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi1x5678, vget_high_f32(vw0123), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi2x5678, vget_high_f32(vw0123), 1);
      vo3p0 = vmlaq_lane_f32(vo3p0, vi3x5678, vget_high_f32(vw0123), 1);
      vo4p0 = vmlaq_lane_f32(vo4p0, vi4x5678, vget_high_f32(vw0123), 1);
      vo5p0 = vmlaq_lane_f32(vo5p0, vi5x5678, vget_high_f32(vw0123), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x5678, vget_high_f32(vw4567), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x5678, vget_high_f32(vw4567), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi3x5678, vget_high_f32(vw4567), 0);
      vo3p0 = vmlaq_lane_f32(vo3p0, vi4x5678, vget_high_f32(vw4567), 0);
      vo4p0 = vmlaq_lane_f32(vo4p0, vi5x5678, vget_high_f32(vw4567), 0);
      vo5p0 = vmlaq_lane_f32(vo5p0, vi6x5678, vget_high_f32(vw4567), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x5678, vw89, 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x5678, vw89, 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x5678, vw89, 1);
      vo3p0 = vmlaq_lane_f32(vo3p0, vi5x5678, vw89, 1);
      vo4p0 = vmlaq_lane_f32(vo4p0, vi6x5678, vw89, 1);
      vo5p0 = vmlaq_lane_f32(vo5p0, vi7x5678, vw89, 1);


      float32x4_t vo0 = vmaxq_f32(vo0p0, vmin);
      float32x4_t vo1 = vmaxq_f32(vo1p0, vmin);
      float32x4_t vo2 = vmaxq_f32(vo2p0, vmin);
      float32x4_t vo3 = vmaxq_f32(vo3p0, vmin);
      float32x4_t vo4 = vmaxq_f32(vo4p0, vmin);
      float32x4_t vo5 = vmaxq_f32(vo5p0, vmin);

      vo0 = vminq_f32(vo0, vmax);
      vo1 = vminq_f32(vo1, vmax);
      vo2 = vminq_f32(vo2, vmax);
      vo3 = vminq_f32(vo3, vmax);
      vo4 = vminq_f32(vo4, vmax);
      vo5 = vminq_f32(vo5, vmax);

      if XNN_LIKELY(w == 4 * sizeof(float)) {
        vst1q_f32(o5, vo5); o5 += 4;
        vst1q_f32(o4, vo4); o4 += 4;
        vst1q_f32(o3, vo3); o3 += 4;
        vst1q_f32(o2, vo2); o2 += 4;
        vst1q_f32(o1, vo1); o1 += 4;
        vst1q_f32(o0, vo0); o0 += 4;
      } else {
        float32x2_t vo0_lo = vget_low_f32(vo0);
        float32x2_t vo1_lo = vget_low_f32(vo1);
        float32x2_t vo2_lo = vget_low_f32(vo2);
        float32x2_t vo3_lo = vget_low_f32(vo3);
        float32x2_t vo4_lo = vget_low_f32(vo4);
        float32x2_t vo5_lo = vget_low_f32(vo5);
        if (w & (2 * sizeof(float))) {
          vst1_f32(o5, vo5_lo); o5 += 2;
          vst1_f32(o4, vo4_lo); o4 += 2;
          vst1_f32(o3, vo3_lo); o3 += 2;
          vst1_f32(o2, vo2_lo); o2 += 2;
          vst1_f32(o1, vo1_lo); o1 += 2;
          vst1_f32(o0, vo0_lo); o0 += 2;

          vo0_lo = vget_high_f32(vo0);
          vo1_lo = vget_high_f32(vo1);
          vo2_lo = vget_high_f32(vo2);
          vo3_lo = vget_high_f32(vo3);
          vo4_lo = vget_high_f32(vo4);
          vo5_lo = vget_high_f32(vo5);
        }
        if (w & (1 * sizeof(float))) {
          vst1_lane_f32(o5, vo5_lo, 0); o5 += 1;
          vst1_lane_f32(o4, vo4_lo, 0); o4 += 1;
          vst1_lane_f32(o3, vo3_lo, 0); o3 += 1;
          vst1_lane_f32(o2, vo2_lo, 0); o2 += 1;
          vst1_lane_f32(o1, vo1_lo, 0); o1 += 1;
          vst1_lane_f32(o0, vo0_lo, 0); o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i6 - input_decrement);
    i1 = (const float*) ((uintptr_t) i7 - input_decrement);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);
    i5 = (const float*) ((uintptr_t) i4 + input_width);
    i6 = (const float*) ((uintptr_t) i5 + input_width);
    i7 = (const float*) ((uintptr_t) i6 + input_width);

    o0 = o5;
    o1 = (float*) ((uintptr_t) o0 + input_width);
    o2 = (float*) ((uintptr_t) o1 + input_width);
    o3 = (float*) ((uintptr_t) o2 + input_width);
    o4 = (float*) ((uintptr_t) o3 + input_width);
    o5 = (float*) ((uintptr_t) o4 + input_width);

    output_height = doz(output_height, 6);
  } while (output_height != 0);
}
