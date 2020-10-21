// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>


void xnn_f32_dwconv_chw_ukernel_3x3p1__neonfma(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_width != 0);
  assert(input_height != 0);
  assert(padding_top == 1);

  const size_t input_tuple_stride = 4 * sizeof(float);
  const size_t output_tuple_stride = 4 * sizeof(float);
  const size_t input_width_stride = input_width * sizeof(float);
  const size_t output_width = input_width;
  const size_t output_width_stride = output_width * sizeof(float);

  const size_t padded_input_height = input_height + padding_top + 1 /* padding_bottom */;
  size_t output_height = padded_input_height - 3 + 1;

  const uint32x4_t vmask = vld1q_u32(params->neon.mask);
  const float32x4_t vmax = vld1q_dup_f32(&params->neon.max);
  const float32x4_t vmin = vld1q_dup_f32(&params->neon.min);

  const size_t input_width_increment = 3 * input_width_stride - round_up_po2(input_width, 4) / 4 * input_tuple_stride;
  const size_t output_width_increment = 3 * output_width_stride - (input_width - 1) / 4 * output_tuple_stride;
  const size_t input_width_increment_single = input_width_stride - round_up_po2(input_width, 4) / 4 * input_tuple_stride;
  const size_t output_width_increment_single = output_width_stride - (input_width - 1) / 4 * output_tuple_stride;
  const size_t input_width_increment_single_backward = round_up_po2(input_width, 4) / 4 * input_tuple_stride;

  const float* i0 = zero;
  const float* i1 = input;
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width_stride);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width_stride);
  if (input_height <= 3) {
    i4 = zero;
  }
  if (input_height <= 2) {
    i3 = zero;
  }
  if (input_height == 1) {
    i2 = zero;
  }

  float* o0 = output;
  float* o1 = (float *) ((uintptr_t) o0 + output_width_stride);
  float* o2 = (float *) ((uintptr_t) o1 + output_width_stride);

  const float32x4_t vw0123 = vld1q_f32(weights);
  const float32x4_t vw4567 = vld1q_f32(weights + 4);
  const float32x2_t vw89 = vld1_f32(weights + 8);

  while (output_height >= 3) {
    float32x4_t vi0x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi1x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi2x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi3x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi4x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi0x4567 = vld1q_f32(i0); i0 += 4;
    float32x4_t vi1x4567 = vld1q_f32(i1); i1 += 4;
    float32x4_t vi2x4567 = vld1q_f32(i2); i2 += 4;
    float32x4_t vi3x4567 = vld1q_f32(i3); i3 += 4;
    float32x4_t vi4x4567 = vld1q_f32(i4); i4 += 4;

    size_t w = input_width;
    for (; w > 4; w -= 4) {
      float32x4_t vo4567p00 = vdupq_laneq_f32(vw0123, 0);
      float32x4_t vo4567p01 = vdupq_laneq_f32(vw0123, 0);
      float32x4_t vo4567p02 = vdupq_laneq_f32(vw0123, 0);

      const float32x4_t vi0x89AB = vld1q_f32(i0); i0 += 4;
      const float32x4_t vi1x89AB = vld1q_f32(i1); i1 += 4;
      const float32x4_t vi2x89AB = vld1q_f32(i2); i2 += 4;
      const float32x4_t vi3x89AB = vld1q_f32(i3); i3 += 4;
      const float32x4_t vi4x89AB = vld1q_f32(i4); i4 += 4;

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x4567, vw0123, 2);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x4567, vw4567, 1);
      vo4567p00 = vfmaq_lane_f32(vo4567p00, vi2x4567, vw89, 0);

      vo4567p01 = vfmaq_laneq_f32(vo4567p01, vi1x4567, vw0123, 2);
      vo4567p01 = vfmaq_laneq_f32(vo4567p01, vi2x4567, vw4567, 1);
      vo4567p01 = vfmaq_lane_f32(vo4567p01, vi3x4567, vw89, 0);

      vo4567p02 = vfmaq_laneq_f32(vo4567p02, vi2x4567, vw0123, 2);
      vo4567p02 = vfmaq_laneq_f32(vo4567p02, vi3x4567, vw4567, 1);
      vo4567p02 = vfmaq_lane_f32(vo4567p02, vi4x4567, vw89, 0);

      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);
      const float32x4_t vi3x3456 = vextq_f32(vi3x0123, vi3x4567, 3);
      const float32x4_t vi4x3456 = vextq_f32(vi4x0123, vi4x4567, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x3456, vw0123, 1);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x3456, vw4567, 0);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x3456, vw4567, 3);

      vo4567p01 = vfmaq_laneq_f32(vo4567p01, vi1x3456, vw0123, 1);
      vo4567p01 = vfmaq_laneq_f32(vo4567p01, vi2x3456, vw4567, 0);
      vo4567p01 = vfmaq_laneq_f32(vo4567p01, vi3x3456, vw4567, 3);

      vo4567p02 = vfmaq_laneq_f32(vo4567p02, vi2x3456, vw0123, 1);
      vo4567p02 = vfmaq_laneq_f32(vo4567p02, vi3x3456, vw4567, 0);
      vo4567p02 = vfmaq_laneq_f32(vo4567p02, vi4x3456, vw4567, 3);

      vi0x0123 = vi0x4567;
      vi1x0123 = vi1x4567;
      vi2x0123 = vi2x4567;
      vi3x0123 = vi3x4567;
      vi4x0123 = vi4x4567;

      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vi0x89AB, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vi1x89AB, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vi2x89AB, 1);
      const float32x4_t vi3x5678 = vextq_f32(vi3x4567, vi3x89AB, 1);
      const float32x4_t vi4x5678 = vextq_f32(vi4x4567, vi4x89AB, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x5678, vw0123, 3);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x5678, vw4567, 2);
      vo4567p00 = vfmaq_lane_f32(vo4567p00, vi2x5678, vw89, 1);

      vo4567p01 = vfmaq_laneq_f32(vo4567p01, vi1x5678, vw0123, 3);
      vo4567p01 = vfmaq_laneq_f32(vo4567p01, vi2x5678, vw4567, 2);
      vo4567p01 = vfmaq_lane_f32(vo4567p01, vi3x5678, vw89, 1);

      vo4567p02 = vfmaq_laneq_f32(vo4567p02, vi2x5678, vw0123, 3);
      vo4567p02 = vfmaq_laneq_f32(vo4567p02, vi3x5678, vw4567, 2);
      vo4567p02 = vfmaq_lane_f32(vo4567p02, vi4x5678, vw89, 1);

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;
      vi4x4567 = vi4x89AB;

      float32x4_t vo0 = vo4567p00;
      float32x4_t vo1 = vo4567p01;
      float32x4_t vo2 = vo4567p02;

      vo0 = vmaxq_f32(vo0, vmin);
      vo0 = vminq_f32(vo0, vmax);
      vo1 = vmaxq_f32(vo1, vmin);
      vo1 = vminq_f32(vo1, vmax);
      vo2 = vmaxq_f32(vo2, vmin);
      vo2 = vminq_f32(vo2, vmax);

      vst1q_f32(o0, vo0); o0 += 4;
      vst1q_f32(o1, vo1); o1 += 4;
      vst1q_f32(o2, vo2); o2 += 4;
    }
    // Always process the last block of 1..4 pixels.
    assert(w >= 1);
    assert(w <= 4);
    {
      float32x4_t vo4567p00 = vdupq_laneq_f32(vw0123, 0);
      float32x4_t vo4567p01 = vdupq_laneq_f32(vw0123, 0);
      float32x4_t vo4567p02 = vdupq_laneq_f32(vw0123, 0);

      vi0x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi0x4567)));
      vi1x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi1x4567)));
      vi2x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi2x4567)));
      vi3x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi3x4567)));
      vi4x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi4x4567)));

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x4567, vw0123, 2);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x4567, vw4567, 1);
      vo4567p00 = vfmaq_lane_f32(vo4567p00, vi2x4567, vw89, 0);

      vo4567p01 = vfmaq_laneq_f32(vo4567p01, vi1x4567, vw0123, 2);
      vo4567p01 = vfmaq_laneq_f32(vo4567p01, vi2x4567, vw4567, 1);
      vo4567p01 = vfmaq_lane_f32(vo4567p01, vi3x4567, vw89, 0);

      vo4567p02 = vfmaq_laneq_f32(vo4567p02, vi2x4567, vw0123, 2);
      vo4567p02 = vfmaq_laneq_f32(vo4567p02, vi3x4567, vw4567, 1);
      vo4567p02 = vfmaq_lane_f32(vo4567p02, vi4x4567, vw89, 0);

      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);
      const float32x4_t vi3x3456 = vextq_f32(vi3x0123, vi3x4567, 3);
      const float32x4_t vi4x3456 = vextq_f32(vi4x0123, vi4x4567, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x3456, vw0123, 1);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x3456, vw4567, 0);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x3456, vw4567, 3);

      vo4567p01 = vfmaq_laneq_f32(vo4567p01, vi1x3456, vw0123, 1);
      vo4567p01 = vfmaq_laneq_f32(vo4567p01, vi2x3456, vw4567, 0);
      vo4567p01 = vfmaq_laneq_f32(vo4567p01, vi3x3456, vw4567, 3);

      vo4567p02 = vfmaq_laneq_f32(vo4567p02, vi2x3456, vw0123, 1);
      vo4567p02 = vfmaq_laneq_f32(vo4567p02, vi3x3456, vw4567, 0);
      vo4567p02 = vfmaq_laneq_f32(vo4567p02, vi4x3456, vw4567, 3);

      const float32x4_t vzero = vmovq_n_f32(0.0f);
      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vzero, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vzero, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vzero, 1);
      const float32x4_t vi3x5678 = vextq_f32(vi3x4567, vzero, 1);
      const float32x4_t vi4x5678 = vextq_f32(vi4x4567, vzero, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x5678, vw0123, 3);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x5678, vw4567, 2);
      vo4567p00 = vfmaq_lane_f32(vo4567p00, vi2x5678, vw89, 1);

      vo4567p01 = vfmaq_laneq_f32(vo4567p01, vi1x5678, vw0123, 3);
      vo4567p01 = vfmaq_laneq_f32(vo4567p01, vi2x5678, vw4567, 2);
      vo4567p01 = vfmaq_lane_f32(vo4567p01, vi3x5678, vw89, 1);

      vo4567p02 = vfmaq_laneq_f32(vo4567p02, vi2x5678, vw0123, 3);
      vo4567p02 = vfmaq_laneq_f32(vo4567p02, vi3x5678, vw4567, 2);
      vo4567p02 = vfmaq_lane_f32(vo4567p02, vi4x5678, vw89, 1);

      float32x4_t vo0 = vo4567p00;
      float32x4_t vo1 = vo4567p01;
      float32x4_t vo2 = vo4567p02;

      vo0 = vmaxq_f32(vo0, vmin);
      vo1 = vmaxq_f32(vo1, vmin);
      vo2 = vmaxq_f32(vo2, vmin);

      vo0 = vminq_f32(vo0, vmax);
      vo1 = vminq_f32(vo1, vmax);
      vo2 = vminq_f32(vo2, vmax);

      if XNN_LIKELY(w & 4) {
        vst1q_f32(o0, vo0);
        vst1q_f32(o1, vo1);
        vst1q_f32(o2, vo2);
      } else {
        float* o0_tmp = o0;
        float* o1_tmp = o1;
        float* o2_tmp = o2;
        float32x2_t vo0_lo = vget_low_f32(vo0);
        float32x2_t vo1_lo = vget_low_f32(vo1);
        float32x2_t vo2_lo = vget_low_f32(vo2);
        if (w & 2) {
          vst1_f32(o0_tmp, vo0_lo); o0_tmp += 2;
          vst1_f32(o1_tmp, vo1_lo); o1_tmp += 2;
          vst1_f32(o2_tmp, vo2_lo); o2_tmp += 2;
          vo0_lo = vget_high_f32(vo0);
          vo1_lo = vget_high_f32(vo1);
          vo2_lo = vget_high_f32(vo2);
        }
        if (w & 1) {
          vst1_lane_f32(o0_tmp, vo0_lo, 0);
          vst1_lane_f32(o1_tmp, vo1_lo, 0);
          vst1_lane_f32(o2_tmp, vo2_lo, 0);
        }
      }
    }

    output_height -= 3;
    i0 = (const float*) ((uintptr_t) i2 + input_width_increment_single);
    i1 = (const float*) ((uintptr_t) i1 + input_width_increment);
    i2 = (const float*) ((uintptr_t) i2 + input_width_increment);
    i3 = (const float*) ((uintptr_t) i3 + input_width_increment);
    i4 = (const float*) ((uintptr_t) i4 + input_width_increment);
    o0 = (float*) ((uintptr_t) o0 + output_width_increment);
    o1 = (float*) ((uintptr_t) o1 + output_width_increment);
    o2 = (float*) ((uintptr_t) o2 + output_width_increment);
    if (output_height == 3) {
      i4 = zero;
    }
  }

  if (output_height == 1) {
    i2 = zero;
  }

  while (output_height != 0) {
    float32x4_t vi0x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi1x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi2x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi0x4567 = vld1q_f32(i0); i0 += 4;
    float32x4_t vi1x4567 = vld1q_f32(i1); i1 += 4;
    float32x4_t vi2x4567 = vld1q_f32(i2); i2 += 4;

    size_t w = input_width;
    for (; w > 4; w -= 4) {
      float32x4_t vo4567p0 = vdupq_laneq_f32(vw0123, 0);

      const float32x4_t vi0x89AB = vld1q_f32(i0); i0 += 4;
      const float32x4_t vi1x89AB = vld1q_f32(i1); i1 += 4;
      const float32x4_t vi2x89AB = vld1q_f32(i2); i2 += 4;

      vo4567p0 = vfmaq_laneq_f32(vo4567p0, vi0x4567, vw0123, 2);
      float32x4_t vo4567p1 = vmulq_laneq_f32(vi1x4567, vw4567, 1);
      float32x4_t vo4567p2 = vmulq_lane_f32(vi2x4567, vw89, 0);

      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);

      vo4567p0 = vfmaq_laneq_f32(vo4567p0, vi0x3456, vw0123, 1);
      vo4567p1 = vfmaq_laneq_f32(vo4567p1, vi1x3456, vw4567, 0);
      vo4567p2 = vfmaq_laneq_f32(vo4567p2, vi2x3456, vw4567, 3);

      vi0x0123 = vi0x4567;
      vi1x0123 = vi1x4567;
      vi2x0123 = vi2x4567;

      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vi0x89AB, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vi1x89AB, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vi2x89AB, 1);

      vo4567p0 = vfmaq_laneq_f32(vo4567p0, vi0x5678, vw0123, 3);
      vo4567p1 = vfmaq_laneq_f32(vo4567p1, vi1x5678, vw4567, 2);
      vo4567p2 = vfmaq_lane_f32(vo4567p2, vi2x5678, vw89, 1);

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;

      float32x4_t vo = vaddq_f32(vo4567p0, vo4567p1);
      vo = vaddq_f32(vo, vo4567p2);

      vo = vmaxq_f32(vo, vmin);
      vo = vminq_f32(vo, vmax);

      vst1q_f32(o0, vo); o0 += 4;
    }
    // Always process the last block of 1..4 pixels.
    assert(w >= 1);
    assert(w <= 4);
    {
      float32x4_t vo4567p0 = vdupq_laneq_f32(vw0123, 0);

      vi0x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi0x4567)));
      vi1x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi1x4567)));
      vi2x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi2x4567)));

      vo4567p0 = vfmaq_laneq_f32(vo4567p0, vi0x4567, vw0123, 2);
      float32x4_t vo4567p1 = vmulq_laneq_f32(vi1x4567, vw4567, 1);
      float32x4_t vo4567p2 = vmulq_lane_f32(vi2x4567, vw89, 0);

      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);

      vo4567p0 = vfmaq_laneq_f32(vo4567p0, vi0x3456, vw0123, 1);
      vo4567p1 = vfmaq_laneq_f32(vo4567p1, vi1x3456, vw4567, 0);
      vo4567p2 = vfmaq_laneq_f32(vo4567p2, vi2x3456, vw4567, 3);

      const float32x4_t vzero = vmovq_n_f32(0.0f);
      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vzero, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vzero, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vzero, 1);

      vo4567p0 = vfmaq_laneq_f32(vo4567p0, vi0x5678, vw0123, 3);
      vo4567p1 = vfmaq_laneq_f32(vo4567p1, vi1x5678, vw4567, 2);
      vo4567p2 = vfmaq_lane_f32(vo4567p2, vi2x5678, vw89, 1);

      float32x4_t vo = vaddq_f32(vo4567p0, vo4567p1);
      vo = vaddq_f32(vo, vo4567p2);

      vo = vmaxq_f32(vo, vmin);
      vo = vminq_f32(vo, vmax);

      if XNN_LIKELY(w & 4) {
        vst1q_f32(o0, vo);
      } else {
        float* o0_tmp = o0;
        float32x2_t vo_lo = vget_low_f32(vo);
        if (w & 2) {
          vst1_f32(o0_tmp, vo_lo); o0_tmp += 2;
          vo_lo = vget_high_f32(vo);
        }
        if (w & 1) {
          vst1_lane_f32(o0_tmp, vo_lo, 0);
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i1 - input_width_increment_single_backward);
    i1 = (const float*) ((uintptr_t) i1 + input_width_increment_single);
    i2 = zero;
    o0 = (float*) ((uintptr_t) o0 + output_width_increment_single);
    output_height -= 1;
  }
}
