// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/conv.h>
#include <xnnpack/math.h>


void xnn_f32_conv_hwc_ukernel_3x3s2p1c3x4__neonfma_2x2(
    size_t input_height,
    size_t input_width,
    size_t output_y_start,
    size_t output_y_end,
    const float* input,
    const float* zero,
    const float* weights,
    float* output,
    size_t input_padding_top,
    size_t output_channels,
    size_t output_height_stride,
    size_t output_width_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_width != 0);
  assert(output_y_end > output_y_start);
  assert(input_padding_top <= 1);
  assert(output_channels != 0);

  const size_t input_height_stride = input_width * 3 /* channels */ * sizeof(float);
  const size_t input_width_increment = round_down_po2(input_width, 4) * 3 /* channels */ * sizeof(float);
  const size_t output_width = (input_width + 1) / 2;
  const size_t output_channel_increment = 4 * sizeof(float) - output_width * output_width_stride;

  // Adjustment for padding processed below
  const float* i0 = (const float*) ((uintptr_t) input + input_height_stride * (output_y_start * 2 - input_padding_top));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_height_stride);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_height_stride);
  float* output0 = (float*) ((uintptr_t) output + output_height_stride * output_y_start);
  float* output1 = (float*) ((uintptr_t) output0 + output_height_stride);

  if XNN_UNPREDICTABLE(output_y_start < input_padding_top) {
    i0 = zero;
  }

  const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
  const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);

  for (size_t output_y = output_y_start; output_y < output_y_end; output_y += 2) {
    const size_t input_y2 = output_y * 2 + 2 - input_padding_top;
    const size_t input_y4 = input_y2 + 2;
    if XNN_UNPREDICTABLE(input_y2 >= input_height) {
      i2 = zero;
    }
    if XNN_UNPREDICTABLE(input_y4 > input_height) {
      i3 = zero;
    }
    if XNN_UNPREDICTABLE(input_y4 >= input_height) {
      i4 = zero;
    }
    if XNN_UNPREDICTABLE(output_y + 2 > output_y_end) {
      output1 = output0;
    }

    const float* w = weights;
    size_t c = output_channels;
    float* o0 = output0;
    float* o1 = output1;
    do {
      // viMx0 = ( iM0c2, iM0c1, iM0c0, --- )
      float32x4_t vi0x0 = vmovq_n_f32(0.0f);
      float32x4_t vi1x0 = vmovq_n_f32(0.0f);
      float32x4_t vi2x0 = vmovq_n_f32(0.0f);
      float32x4_t vi3x0 = vmovq_n_f32(0.0f);
      float32x4_t vi4x0 = vmovq_n_f32(0.0f);

      size_t iw = input_width;
      for (; iw >= 4; iw -= 4) {
        float32x4_t vo0x0 = vld1q_f32(w);
        float32x4_t vo1x0 = vo0x0;
        float32x4_t vo0x1 = vo0x0;
        float32x4_t vo1x1 = vo0x0;

        const float32x4_t vk00c0 = vld1q_f32(w + 4);

        // viMx1 = ( iM2c0, iM1c2, iM1c1, iM1c0 )
        const float32x4_t vi0x1 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi1x1 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi2x1 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi3x1 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi4x1 = vld1q_f32(i4); i4 += 4;

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk00c0, vi0x0, 1);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk00c0, vi2x0, 1);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk00c0, vi0x1, 3);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk00c0, vi2x1, 3);

        const float32x4_t vk10c0 = vld1q_f32(w + 8);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk10c0, vi1x0, 1);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk10c0, vi3x0, 1);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk10c0, vi1x1, 3);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk10c0, vi3x1, 3);

        const float32x4_t vk20c0 = vld1q_f32(w + 12);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk20c0, vi2x0, 1);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk20c0, vi4x0, 1);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk20c0, vi2x1, 3);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk20c0, vi4x1, 3);

        const float32x4_t vk00c1 = vld1q_f32(w + 16);

        // viMx2 = ( iM3c1, iM3c0, iM2c2, iM2c1 )
        const float32x4_t vi0x2 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi1x2 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi2x2 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi3x2 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi4x2 = vld1q_f32(i4); i4 += 4;

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk00c1, vi0x0, 2);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk00c1, vi2x0, 2);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk00c1, vi0x2, 0);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk00c1, vi2x2, 0);

        const float32x4_t vk10c1 = vld1q_f32(w + 20);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk10c1, vi1x0, 2);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk10c1, vi3x0, 2);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk10c1, vi1x2, 0);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk10c1, vi3x2, 0);

        const float32x4_t vk20c1 = vld1q_f32(w + 24);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk20c1, vi2x0, 2);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk20c1, vi4x0, 2);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk20c1, vi2x2, 0);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk20c1, vi4x2, 0);

        const float32x4_t vk00c2 = vld1q_f32(w + 28);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk00c2, vi0x0, 3);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk00c2, vi2x0, 3);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk00c2, vi0x2, 1);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk00c2, vi2x2, 1);

        const float32x4_t vk10c2 = vld1q_f32(w + 32);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk10c2, vi1x0, 3);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk10c2, vi3x0, 3);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk10c2, vi1x2, 1);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk10c2, vi3x2, 1);

        const float32x4_t vk20c2 = vld1q_f32(w + 36);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk20c2, vi2x0, 3);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk20c2, vi4x0, 3);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk20c2, vi2x2, 1);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk20c2, vi4x2, 1);

        const float32x4_t vk01c0 = vld1q_f32(w + 40);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk01c0, vi0x1, 0);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk01c0, vi2x1, 0);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk01c0, vi0x2, 2);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk01c0, vi2x2, 2);

        const float32x4_t vk11c0 = vld1q_f32(w + 44);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk11c0, vi1x1, 0);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk11c0, vi3x1, 0);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk11c0, vi1x2, 2);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk11c0, vi3x2, 2);

        const float32x4_t vk21c0 = vld1q_f32(w + 48);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk21c0, vi2x1, 0);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk21c0, vi4x1, 0);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk21c0, vi2x2, 2);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk21c0, vi4x2, 2);

        const float32x4_t vk01c1 = vld1q_f32(w + 52);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk01c1, vi0x1, 1);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk01c1, vi2x1, 1);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk01c1, vi0x2, 3);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk01c1, vi2x2, 3);

        const float32x4_t vk11c1 = vld1q_f32(w + 56);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk11c1, vi1x1, 1);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk11c1, vi3x1, 1);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk11c1, vi1x2, 3);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk11c1, vi3x2, 3);

        const float32x4_t vk21c1 = vld1q_f32(w + 60);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk21c1, vi2x1, 1);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk21c1, vi4x1, 1);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk21c1, vi2x2, 3);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk21c1, vi4x2, 3);

        const float32x4_t vk01c2 = vld1q_f32(w + 64);

        // viMx3 = ( iM4c2, iM4c1, iM4c0, iM3c2 )
        const float32x4_t vi0x3 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi1x3 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi2x3 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi3x3 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi4x3 = vld1q_f32(i4); i4 += 4;

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk01c2, vi0x1, 2);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk01c2, vi2x1, 2);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk01c2, vi0x3, 0);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk01c2, vi2x3, 0);

        const float32x4_t vk11c2 = vld1q_f32(w + 68);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk11c2, vi1x1, 2);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk11c2, vi3x1, 2);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk11c2, vi1x3, 0);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk11c2, vi3x3, 0);

        const float32x4_t vk21c2 = vld1q_f32(w + 72);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk21c2, vi2x1, 2);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk21c2, vi4x1, 2);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk21c2, vi2x3, 0);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk21c2, vi4x3, 0);

        const float32x4_t vk02c0 = vld1q_f32(w + 76);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk02c0, vi0x1, 3);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk02c0, vi2x1, 3);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk02c0, vi0x3, 1);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk02c0, vi2x3, 1);

        const float32x4_t vk12c0 = vld1q_f32(w + 80);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk12c0, vi1x1, 3);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk12c0, vi3x1, 3);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk12c0, vi1x3, 1);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk12c0, vi3x3, 1);

        const float32x4_t vk22c0 = vld1q_f32(w + 84);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk22c0, vi2x1, 3);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk22c0, vi4x1, 3);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk22c0, vi2x3, 1);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk22c0, vi4x3, 1);

        const float32x4_t vk02c1 = vld1q_f32(w + 88);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk02c1, vi0x2, 0);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk02c1, vi2x2, 0);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk02c1, vi0x3, 2);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk02c1, vi2x3, 2);

        const float32x4_t vk12c1 = vld1q_f32(w + 92);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk12c1, vi1x2, 0);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk12c1, vi3x2, 0);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk12c1, vi1x3, 2);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk12c1, vi3x3, 2);

        const float32x4_t vk22c1 = vld1q_f32(w + 96);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk22c1, vi2x2, 0);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk22c1, vi4x2, 0);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk22c1, vi2x3, 2);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk22c1, vi4x3, 2);

        const float32x4_t vk02c2 = vld1q_f32(w + 100);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk02c2, vi0x2, 1);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk02c2, vi2x2, 1);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk02c2, vi0x3, 3);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk02c2, vi2x3, 3);

        const float32x4_t vk12c2 = vld1q_f32(w + 104);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk12c2, vi1x2, 1);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk12c2, vi3x2, 1);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk12c2, vi1x3, 3);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk12c2, vi3x3, 3);

        const float32x4_t vk22c2 = vld1q_f32(w + 108);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk22c2, vi2x2, 1);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk22c2, vi4x2, 1);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk22c2, vi2x3, 3);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk22c2, vi4x3, 3);

        vi0x0 = vi0x3;
        vi1x0 = vi1x3;
        vi2x0 = vi2x3;
        vi3x0 = vi3x3;
        vi4x0 = vi4x3;

        vo0x0 = vmaxq_f32(vo0x0, vmin);
        vo1x0 = vmaxq_f32(vo1x0, vmin);
        vo0x1 = vmaxq_f32(vo0x1, vmin);
        vo1x1 = vmaxq_f32(vo1x1, vmin);

        vo0x0 = vminq_f32(vo0x0, vmax);
        vo1x0 = vminq_f32(vo1x0, vmax);
        vo0x1 = vminq_f32(vo0x1, vmax);
        vo1x1 = vminq_f32(vo1x1, vmax);

        if XNN_LIKELY(c >= 4) {
          vst1q_f32(o1, vo1x0);
          o1 = (float*) ((uintptr_t) o1 + output_width_stride);
          vst1q_f32(o0, vo0x0);
          o0 = (float*) ((uintptr_t) o0 + output_width_stride);

          vst1q_f32(o1, vo1x1);
          o1 = (float*) ((uintptr_t) o1 + output_width_stride);
          vst1q_f32(o0, vo0x1);
          o0 = (float*) ((uintptr_t) o0 + output_width_stride);
        } else {
          float32x2_t vo0x0_lo = vget_low_f32(vo0x0);
          float32x2_t vo1x0_lo = vget_low_f32(vo1x0);
          float32x2_t vo0x1_lo = vget_low_f32(vo0x1);
          float32x2_t vo1x1_lo = vget_low_f32(vo1x1);
          float* o0_tmp = o0;
          float* o1_tmp = o1;
          if (c & 2) {
            vst1_f32((float*) ((uintptr_t) o1_tmp + output_width_stride), vo1x1_lo);
            vo1x1_lo = vget_high_f32(vo1x1);
            vst1_f32((float*) ((uintptr_t) o0_tmp + output_width_stride), vo0x1_lo);
            vo0x1_lo = vget_high_f32(vo0x1);

            vst1_f32(o1_tmp, vo1x0_lo); o1_tmp += 2;
            vo1x0_lo = vget_high_f32(vo1x0);
            vst1_f32(o0_tmp, vo0x0_lo); o0_tmp += 2;
            vo0x0_lo = vget_high_f32(vo0x0);
          }
          if (c & 1) {
            vst1_lane_f32(o1_tmp, vo1x0_lo, 0);
            vst1_lane_f32(o0_tmp, vo0x0_lo, 0);

            vst1_lane_f32((float*) ((uintptr_t) o1_tmp + output_width_stride), vo1x1_lo, 0);
            vst1_lane_f32((float*) ((uintptr_t) o0_tmp + output_width_stride), vo0x1_lo, 0);
          }
          o0 = (float*) ((uintptr_t) o0 + output_width_stride * 2);
          o1 = (float*) ((uintptr_t) o1 + output_width_stride * 2);
        }
      }
      assert(iw < 4);
      if XNN_UNLIKELY(iw != 0) {
        float32x4_t vo0x0 = vld1q_f32(w);
        float32x4_t vo1x0 = vo0x0;
        float32x4_t vo0x1 = vo0x0;
        float32x4_t vo1x1 = vo0x0;

        const float32x4_t vk00c0 = vld1q_f32(w + 4);

        // viMx1 = ( iM2c0, iM1c2, iM1c1, iM1c0 )
        float32x4_t vi0x1 = vld1q_f32(i0);
        float32x4_t vi1x1 = vld1q_f32(i1);
        float32x4_t vi2x1 = vld1q_f32(i2);
        float32x4_t vi3x1 = vld1q_f32(i3);
        float32x4_t vi4x1 = vld1q_f32(i4);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk00c0, vi0x0, 1);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk00c0, vi2x0, 1);
        if (iw > 2) {
          vo0x1 = vfmaq_laneq_f32(vo0x1, vk00c0, vi0x1, 3);
          vo1x1 = vfmaq_laneq_f32(vo1x1, vk00c0, vi2x1, 3);
        }

        const float32x4_t vk10c0 = vld1q_f32(w + 8);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk10c0, vi1x0, 1);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk10c0, vi3x0, 1);
        if (iw > 2) {
          vo0x1 = vfmaq_laneq_f32(vo0x1, vk10c0, vi1x1, 3);
          vo1x1 = vfmaq_laneq_f32(vo1x1, vk10c0, vi3x1, 3);
        }

        const float32x4_t vk20c0 = vld1q_f32(w + 12);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk20c0, vi2x0, 1);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk20c0, vi4x0, 1);
        if (iw > 2) {
          vo0x1 = vfmaq_laneq_f32(vo0x1, vk20c0, vi2x1, 3);
          vo1x1 = vfmaq_laneq_f32(vo1x1, vk20c0, vi4x1, 3);
        }

        const float32x4_t vk00c1 = vld1q_f32(w + 16);

        float32x4_t vi0x2 = vmovq_n_f32(0.0f);
        float32x4_t vi1x2 = vmovq_n_f32(0.0f);
        float32x4_t vi2x2 = vmovq_n_f32(0.0f);
        float32x4_t vi3x2 = vmovq_n_f32(0.0f);
        float32x4_t vi4x2 = vmovq_n_f32(0.0f);
        if (iw >= 2) {
          // viMx2 = ( iM3c1, iM3c0, iM2c2, iM2c1 )
          vi0x2 = vld1q_f32(i0 + 4);
          vi1x2 = vld1q_f32(i1 + 4);
          vi2x2 = vld1q_f32(i2 + 4);
          vi3x2 = vld1q_f32(i3 + 4);
          vi4x2 = vld1q_f32(i4 + 4);
        }

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk00c1, vi0x0, 2);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk00c1, vi2x0, 2);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk00c1, vi0x2, 0);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk00c1, vi2x2, 0);

        const float32x4_t vk10c1 = vld1q_f32(w + 20);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk10c1, vi1x0, 2);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk10c1, vi3x0, 2);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk10c1, vi1x2, 0);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk10c1, vi3x2, 0);

        const float32x4_t vk20c1 = vld1q_f32(w + 24);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk20c1, vi2x0, 2);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk20c1, vi4x0, 2);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk20c1, vi2x2, 0);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk20c1, vi4x2, 0);

        const float32x4_t vk00c2 = vld1q_f32(w + 28);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk00c2, vi0x0, 3);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk00c2, vi2x0, 3);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk00c2, vi0x2, 1);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk00c2, vi2x2, 1);

        const float32x4_t vk10c2 = vld1q_f32(w + 32);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk10c2, vi1x0, 3);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk10c2, vi3x0, 3);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk10c2, vi1x2, 1);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk10c2, vi3x2, 1);

        const float32x4_t vk20c2 = vld1q_f32(w + 36);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk20c2, vi2x0, 3);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk20c2, vi4x0, 3);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk20c2, vi2x2, 1);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk20c2, vi4x2, 1);

        const float32x4_t vk01c0 = vld1q_f32(w + 40);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk01c0, vi0x1, 0);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk01c0, vi2x1, 0);
        if (iw > 2) {
          vo0x1 = vfmaq_laneq_f32(vo0x1, vk01c0, vi0x2, 2);
          vo1x1 = vfmaq_laneq_f32(vo1x1, vk01c0, vi2x2, 2);
        }

        const float32x4_t vk11c0 = vld1q_f32(w + 44);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk11c0, vi1x1, 0);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk11c0, vi3x1, 0);
        if (iw > 2) {
          vo0x1 = vfmaq_laneq_f32(vo0x1, vk11c0, vi1x2, 2);
          vo1x1 = vfmaq_laneq_f32(vo1x1, vk11c0, vi3x2, 2);
        }

        const float32x4_t vk21c0 = vld1q_f32(w + 48);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk21c0, vi2x1, 0);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk21c0, vi4x1, 0);
        if (iw > 2) {
          vo0x1 = vfmaq_laneq_f32(vo0x1, vk21c0, vi2x2, 2);
          vo1x1 = vfmaq_laneq_f32(vo1x1, vk21c0, vi4x2, 2);
        }

        const float32x4_t vk01c1 = vld1q_f32(w + 52);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk01c1, vi0x1, 1);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk01c1, vi2x1, 1);
        if (iw > 2) {
          vo0x1 = vfmaq_laneq_f32(vo0x1, vk01c1, vi0x2, 3);
          vo1x1 = vfmaq_laneq_f32(vo1x1, vk01c1, vi2x2, 3);
        }

        const float32x4_t vk11c1 = vld1q_f32(w + 56);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk11c1, vi1x1, 1);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk11c1, vi3x1, 1);
        if (iw > 2) {
          vo0x1 = vfmaq_laneq_f32(vo0x1, vk11c1, vi1x2, 3);
          vo1x1 = vfmaq_laneq_f32(vo1x1, vk11c1, vi3x2, 3);
        }

        const float32x4_t vk21c1 = vld1q_f32(w + 60);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk21c1, vi2x1, 1);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk21c1, vi4x1, 1);
        if (iw > 2) {
          vo0x1 = vfmaq_laneq_f32(vo0x1, vk21c1, vi2x2, 3);
          vo1x1 = vfmaq_laneq_f32(vo1x1, vk21c1, vi4x2, 3);
        }

        const float32x4_t vk01c2 = vld1q_f32(w + 64);

        float32x4_t vi0x3 = vmovq_n_f32(0.0f);
        float32x4_t vi1x3 = vmovq_n_f32(0.0f);
        float32x4_t vi2x3 = vmovq_n_f32(0.0f);
        float32x4_t vi3x3 = vmovq_n_f32(0.0f);
        float32x4_t vi4x3 = vmovq_n_f32(0.0f);
        if (iw > 2) {
          // viMx3 = ( 0.0, 0.0, 0.0, iM3c2 )
          vi0x3 = vld1q_lane_f32(i0 + 8, vi0x3, 0);
          vi1x3 = vld1q_lane_f32(i1 + 8, vi1x3, 0);
          vi2x3 = vld1q_lane_f32(i2 + 8, vi2x3, 0);
          vi3x3 = vld1q_lane_f32(i3 + 8, vi3x3, 0);
          vi4x3 = vld1q_lane_f32(i4 + 8, vi4x3, 0);
        }

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk01c2, vi0x1, 2);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk01c2, vi2x1, 2);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk01c2, vi0x3, 0);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk01c2, vi2x3, 0);

        const float32x4_t vk11c2 = vld1q_f32(w + 68);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk11c2, vi1x1, 2);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk11c2, vi3x1, 2);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk11c2, vi1x3, 0);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk11c2, vi3x3, 0);

        const float32x4_t vk21c2 = vld1q_f32(w + 72);

        vo0x0 = vfmaq_laneq_f32(vo0x0, vk21c2, vi2x1, 2);
        vo1x0 = vfmaq_laneq_f32(vo1x0, vk21c2, vi4x1, 2);
        vo0x1 = vfmaq_laneq_f32(vo0x1, vk21c2, vi2x3, 0);
        vo1x1 = vfmaq_laneq_f32(vo1x1, vk21c2, vi4x3, 0);

        if (iw >= 2) {
          const float32x4_t vk02c0 = vld1q_f32(w + 76);

          vo0x0 = vfmaq_laneq_f32(vo0x0, vk02c0, vi0x1, 3);
          vo1x0 = vfmaq_laneq_f32(vo1x0, vk02c0, vi2x1, 3);

          const float32x4_t vk12c0 = vld1q_f32(w + 80);

          vo0x0 = vfmaq_laneq_f32(vo0x0, vk12c0, vi1x1, 3);
          vo1x0 = vfmaq_laneq_f32(vo1x0, vk12c0, vi3x1, 3);

          const float32x4_t vk22c0 = vld1q_f32(w + 84);

          vo0x0 = vfmaq_laneq_f32(vo0x0, vk22c0, vi2x1, 3);
          vo1x0 = vfmaq_laneq_f32(vo1x0, vk22c0, vi4x1, 3);

          const float32x4_t vk02c1 = vld1q_f32(w + 88);

          vo0x0 = vfmaq_laneq_f32(vo0x0, vk02c1, vi0x2, 0);
          vo1x0 = vfmaq_laneq_f32(vo1x0, vk02c1, vi2x2, 0);

          const float32x4_t vk12c1 = vld1q_f32(w + 92);

          vo0x0 = vfmaq_laneq_f32(vo0x0, vk12c1, vi1x2, 0);
          vo1x0 = vfmaq_laneq_f32(vo1x0, vk12c1, vi3x2, 0);

          const float32x4_t vk22c1 = vld1q_f32(w + 96);

          vo0x0 = vfmaq_laneq_f32(vo0x0, vk22c1, vi2x2, 0);
          vo1x0 = vfmaq_laneq_f32(vo1x0, vk22c1, vi4x2, 0);

          const float32x4_t vk02c2 = vld1q_f32(w + 100);

          vo0x0 = vfmaq_laneq_f32(vo0x0, vk02c2, vi0x2, 1);
          vo1x0 = vfmaq_laneq_f32(vo1x0, vk02c2, vi2x2, 1);

          const float32x4_t vk12c2 = vld1q_f32(w + 104);

          vo0x0 = vfmaq_laneq_f32(vo0x0, vk12c2, vi1x2, 1);
          vo1x0 = vfmaq_laneq_f32(vo1x0, vk12c2, vi3x2, 1);

          const float32x4_t vk22c2 = vld1q_f32(w + 108);

          vo0x0 = vfmaq_laneq_f32(vo0x0, vk22c2, vi2x2, 1);
          vo1x0 = vfmaq_laneq_f32(vo1x0, vk22c2, vi4x2, 1);
        }

        vo0x0 = vmaxq_f32(vo0x0, vmin);
        vo1x0 = vmaxq_f32(vo1x0, vmin);
        vo0x1 = vmaxq_f32(vo0x1, vmin);
        vo1x1 = vmaxq_f32(vo1x1, vmin);

        vo0x0 = vminq_f32(vo0x0, vmax);
        vo1x0 = vminq_f32(vo1x0, vmax);
        vo0x1 = vminq_f32(vo0x1, vmax);
        vo1x1 = vminq_f32(vo1x1, vmax);

        iw += 1;
        if XNN_LIKELY(c >= 4) {
          vst1q_f32(o1, vo1x0);
          o1 = (float*) ((uintptr_t) o1 + output_width_stride);
          vst1q_f32(o0, vo0x0);
          o0 = (float*) ((uintptr_t) o0 + output_width_stride);

          if (iw & 4) {
            vst1q_f32(o1, vo1x1);
            o1 = (float*) ((uintptr_t) o1 + output_width_stride);
            vst1q_f32(o0, vo0x1);
            o0 = (float*) ((uintptr_t) o0 + output_width_stride);
          }
        } else {
          float* o0_tmp = o0;
          float* o1_tmp = o1;
          float32x2_t vo0x0_lo = vget_low_f32(vo0x0);
          float32x2_t vo1x0_lo = vget_low_f32(vo1x0);
          float32x2_t vo0x1_lo = vget_low_f32(vo0x1);
          float32x2_t vo1x1_lo = vget_low_f32(vo1x1);
          if (c & 2) {
            if (iw & 4) {
              vst1_f32((float*) ((uintptr_t) o1_tmp + output_width_stride), vo1x1_lo);
              vo1x1_lo = vget_high_f32(vo1x1);
              vst1_f32((float*) ((uintptr_t) o0_tmp + output_width_stride), vo0x1_lo);
              vo0x1_lo = vget_high_f32(vo0x1);
            }

            vst1_f32(o1_tmp, vo1x0_lo); o1_tmp += 2;
            vo1x0_lo = vget_high_f32(vo1x0);
            vst1_f32(o0_tmp, vo0x0_lo); o0_tmp += 2;
            vo0x0_lo = vget_high_f32(vo0x0);
          }
          if (c & 1) {
            vst1_lane_f32(o1_tmp, vo1x0_lo, 0);
            vst1_lane_f32(o0_tmp, vo0x0_lo, 0);

            if (iw & 4) {
              vst1_lane_f32((float*) ((uintptr_t) o1_tmp + output_width_stride), vo1x1_lo, 0);
              vst1_lane_f32((float*) ((uintptr_t) o0_tmp + output_width_stride), vo0x1_lo, 0);
            }
          }
          o0 = (float*) ((uintptr_t) o0 + output_width_stride * 2);
          o1 = (float*) ((uintptr_t) o1 + output_width_stride * 2);
        }
      }
      // Move output pointers back to the position of the first pixel in a row,
      // and forward to the next block of output channels
      o0 = (float*) ((uintptr_t) o0 + output_channel_increment);
      o1 = (float*) ((uintptr_t) o1 + output_channel_increment);
      // Revert input pointers to the position of the first pixel in a row
      i0 = (const float*) ((uintptr_t) i0 - input_width_increment);
      i1 = (const float*) ((uintptr_t) i1 - input_width_increment);
      i2 = (const float*) ((uintptr_t) i2 - input_width_increment);
      i3 = (const float*) ((uintptr_t) i3 - input_width_increment);
      i4 = (const float*) ((uintptr_t) i4 - input_width_increment);
      // Move to the block of weights for the next 4 output channels
      w += 112;
      c = doz(c, 4);
    } while (c != 0);
    // Move output pointers forward to the next two rows
    output0 = (float*) ((uintptr_t) output1 + output_height_stride);
    output1 = (float*) ((uintptr_t) output0 + output_height_stride);
    // Move input pointers forward to the next four rows
    i0 = i4;
    i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
    i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
    i3 = (const float*) ((uintptr_t) i2 + input_height_stride);
    i4 = (const float*) ((uintptr_t) i3 + input_height_stride);
  }
}
