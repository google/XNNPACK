// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/conv.h>
#include <xnnpack/dwconv.h>
#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/lut.h>
#include <xnnpack/math.h>
#include <xnnpack/microparams.h>
#include <xnnpack/prefetch.h>
#include <xnnpack/spmm.h>
#include <xnnpack/transpose.h>
#include <xnnpack/vbinary.h>
#include <xnnpack/vunary.h>


void xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2(
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
    size_t output_channel_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_width != 0);
  assert(output_y_end > output_y_start);
  assert(input_padding_top <= 1);
  assert(output_channels != 0);

  const size_t input_height_stride = input_width * 3 /* channels */ * sizeof(float);
  const size_t input_width_increment = round_down_po2(input_width, 4) * 3 /* channels */ * sizeof(float);
  const size_t output_width = (input_width + 1) / 2;
  const size_t output_channel_increment = output_channel_stride * 4 - output_width * sizeof(float);

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
    float* o0c0 = output0;
    float* o1c0 = output1;
    float* o0c1 = (float*) ((uintptr_t) o0c0 + output_channel_stride);
    float* o1c1 = (float*) ((uintptr_t) o1c0 + output_channel_stride);
    float* o0c2 = (float*) ((uintptr_t) o0c1 + output_channel_stride);
    float* o1c2 = (float*) ((uintptr_t) o1c1 + output_channel_stride);
    float* o0c3 = (float*) ((uintptr_t) o0c2 + output_channel_stride);
    float* o1c3 = (float*) ((uintptr_t) o1c2 + output_channel_stride);
    do {
      if XNN_UNPREDICTABLE(c < 2) {
        o0c1 = o0c0;
        o1c1 = o1c0;
      }
      if XNN_UNPREDICTABLE(c <= 2) {
        o0c2 = o0c1;
        o1c2 = o1c1;
      }
      if XNN_UNPREDICTABLE(c < 4) {
        o0c3 = o0c2;
        o1c3 = o1c2;
      }

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

        const float32x4_t vo0c01 = vzip1q_f32(vo0x0, vo0x1);
        const float32x4_t vo0c23 = vzip2q_f32(vo0x0, vo0x1);
        const float32x4_t vo1c01 = vzip1q_f32(vo1x0, vo1x1);
        const float32x4_t vo1c23 = vzip2q_f32(vo1x0, vo1x1);

        // Always 2+ output width elements remaining
        vst1_f32(o1c0, vget_low_f32(vo1c01)); o1c0 += 2;
        vst1_f32(o1c1, vget_high_f32(vo1c01)); o1c1 += 2;
        vst1_f32(o1c2, vget_low_f32(vo1c23)); o1c2 += 2;
        vst1_f32(o1c3, vget_high_f32(vo1c23)); o1c3 += 2;

        vst1_f32(o0c0, vget_low_f32(vo0c01)); o0c0 += 2;
        vst1_f32(o0c1, vget_high_f32(vo0c01)); o0c1 += 2;
        vst1_f32(o0c2, vget_low_f32(vo0c23)); o0c2 += 2;
        vst1_f32(o0c3, vget_high_f32(vo0c23)); o0c3 += 2;
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

        if (iw == 3) {
          // Exactly 2 output width elements remaining
          const float32x4_t vo0c01 = vzip1q_f32(vo0x0, vo0x1);
          const float32x4_t vo0c23 = vzip2q_f32(vo0x0, vo0x1);
          const float32x4_t vo1c01 = vzip1q_f32(vo1x0, vo1x1);
          const float32x4_t vo1c23 = vzip2q_f32(vo1x0, vo1x1);

          vst1_f32(o1c0, vget_low_f32(vo1c01)); o1c0 += 2;
          vst1_f32(o1c1, vget_high_f32(vo1c01)); o1c1 += 2;
          vst1_f32(o1c2, vget_low_f32(vo1c23)); o1c2 += 2;
          vst1_f32(o1c3, vget_high_f32(vo1c23)); o1c3 += 2;

          vst1_f32(o0c0, vget_low_f32(vo0c01)); o0c0 += 2;
          vst1_f32(o0c1, vget_high_f32(vo0c01)); o0c1 += 2;
          vst1_f32(o0c2, vget_low_f32(vo0c23)); o0c2 += 2;
          vst1_f32(o0c3, vget_high_f32(vo0c23)); o0c3 += 2;
        } else {
          // Exactly 1 output width element remaining

          vst1q_lane_f32(o1c0, vo1x0, 0); o1c0 += 1;
          vst1q_lane_f32(o1c1, vo1x0, 1); o1c1 += 1;
          vst1q_lane_f32(o1c2, vo1x0, 2); o1c2 += 1;
          vst1q_lane_f32(o1c3, vo1x0, 3); o1c3 += 1;

          vst1q_lane_f32(o0c0, vo0x0, 0); o0c0 += 1;
          vst1q_lane_f32(o0c1, vo0x0, 1); o0c1 += 1;
          vst1q_lane_f32(o0c2, vo0x0, 2); o0c2 += 1;
          vst1q_lane_f32(o0c3, vo0x0, 3); o0c3 += 1;
        }
      }
      // Move output pointers back to the position of the first pixel in a row,
      // and forward to the next block of output channels.
      o0c0 = (float*) ((uintptr_t) o0c0 + output_channel_increment);
      o0c1 = (float*) ((uintptr_t) o0c1 + output_channel_increment);
      o0c2 = (float*) ((uintptr_t) o0c2 + output_channel_increment);
      o0c3 = (float*) ((uintptr_t) o0c3 + output_channel_increment);
      o1c0 = (float*) ((uintptr_t) o1c0 + output_channel_increment);
      o1c1 = (float*) ((uintptr_t) o1c1 + output_channel_increment);
      o1c2 = (float*) ((uintptr_t) o1c2 + output_channel_increment);
      o1c3 = (float*) ((uintptr_t) o1c3 + output_channel_increment);
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

void xnn_f32_dwconv2d_chw_ukernel_3x3p1__aarch64_neonfma_3x4(
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

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + input_width);
  float* o2 = (float*) ((uintptr_t) o1 + input_width);

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
    }

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
    for (; w > 4 * sizeof(float); w -= 4 * sizeof(float)) {
      float32x4_t vo0p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo1p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo2p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);

      const float32x4_t vi0x89AB = vld1q_f32(i0); i0 += 4;
      const float32x4_t vi1x89AB = vld1q_f32(i1); i1 += 4;
      const float32x4_t vi2x89AB = vld1q_f32(i2); i2 += 4;
      const float32x4_t vi3x89AB = vld1q_f32(i3); i3 += 4;
      const float32x4_t vi4x89AB = vld1q_f32(i4); i4 += 4;

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x4567, vget_high_f32(vw0123), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi1x4567, vget_high_f32(vw0123), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi2x4567, vget_high_f32(vw0123), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x4567, vget_low_f32(vw4567), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x4567, vget_low_f32(vw4567), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi3x4567, vget_low_f32(vw4567), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x4567, vw89, 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x4567, vw89, 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi4x4567, vw89, 0);

      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);
      const float32x4_t vi3x3456 = vextq_f32(vi3x0123, vi3x4567, 3);
      const float32x4_t vi4x3456 = vextq_f32(vi4x0123, vi4x4567, 3);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x3456, vget_low_f32(vw0123), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi1x3456, vget_low_f32(vw0123), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi2x3456, vget_low_f32(vw0123), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x3456, vget_low_f32(vw4567), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x3456, vget_low_f32(vw4567), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi3x3456, vget_low_f32(vw4567), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x3456, vget_high_f32(vw4567), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x3456, vget_high_f32(vw4567), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi4x3456, vget_high_f32(vw4567), 1);

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

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x5678, vget_high_f32(vw0123), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi1x5678, vget_high_f32(vw0123), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi2x5678, vget_high_f32(vw0123), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x5678, vget_high_f32(vw4567), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x5678, vget_high_f32(vw4567), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi3x5678, vget_high_f32(vw4567), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x5678, vw89, 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x5678, vw89, 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi4x5678, vw89, 1);

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;
      vi4x4567 = vi4x89AB;


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
    // Always process the last block of 1..4 pixels.
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

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x4567, vget_high_f32(vw0123), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi1x4567, vget_high_f32(vw0123), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi2x4567, vget_high_f32(vw0123), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x4567, vget_low_f32(vw4567), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x4567, vget_low_f32(vw4567), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi3x4567, vget_low_f32(vw4567), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x4567, vw89, 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x4567, vw89, 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi4x4567, vw89, 0);

      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);
      const float32x4_t vi3x3456 = vextq_f32(vi3x0123, vi3x4567, 3);
      const float32x4_t vi4x3456 = vextq_f32(vi4x0123, vi4x4567, 3);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x3456, vget_low_f32(vw0123), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi1x3456, vget_low_f32(vw0123), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi2x3456, vget_low_f32(vw0123), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x3456, vget_low_f32(vw4567), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x3456, vget_low_f32(vw4567), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi3x3456, vget_low_f32(vw4567), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x3456, vget_high_f32(vw4567), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x3456, vget_high_f32(vw4567), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi4x3456, vget_high_f32(vw4567), 1);

      const float32x4_t vzero = vmovq_n_f32(0.0f);
      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vzero, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vzero, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vzero, 1);
      const float32x4_t vi3x5678 = vextq_f32(vi3x4567, vzero, 1);
      const float32x4_t vi4x5678 = vextq_f32(vi4x4567, vzero, 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x5678, vget_high_f32(vw0123), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi1x5678, vget_high_f32(vw0123), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi2x5678, vget_high_f32(vw0123), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x5678, vget_high_f32(vw4567), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x5678, vget_high_f32(vw4567), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi3x5678, vget_high_f32(vw4567), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x5678, vw89, 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x5678, vw89, 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi4x5678, vw89, 1);


      float32x4_t vo0 = vmaxq_f32(vo0p0, vmin);
      float32x4_t vo1 = vmaxq_f32(vo1p0, vmin);
      float32x4_t vo2 = vmaxq_f32(vo2p0, vmin);

      vo0 = vminq_f32(vo0, vmax);
      vo1 = vminq_f32(vo1, vmax);
      vo2 = vminq_f32(vo2, vmax);

      if XNN_LIKELY(w == 4 * sizeof(float)) {
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

    o0 = o2;
    o1 = (float*) ((uintptr_t) o0 + input_width);
    o2 = (float*) ((uintptr_t) o1 + input_width);

    output_height = doz(output_height, 3);
  } while (output_height != 0);
}

void xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__aarch64_neonfma_2x4_acc2(
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
  assert(padding_top >= 0);
  assert(padding_top <= 1);

  const uint32x4_t vmask_even = vld1q_u32(params->neon_stride2.mask_even);
  const uint32x4_t vmask_odd  = vld1q_u32(params->neon_stride2.mask_odd);
  const float32x4_t vmax = vld1q_dup_f32(&params->neon_stride2.max);
  const float32x4_t vmin = vld1q_dup_f32(&params->neon_stride2.min);

  const float32x4_t vw0123 = vld1q_f32(weights);
  const float32x4_t vw4567 = vld1q_f32(weights + 4);
  const float32x2_t vw89 = vld1_f32(weights + 8);

  const size_t input_decrement = round_down_po2(input_width, 4 /* SIMD output width */ * 2 /* subsampling */ * sizeof(float));
  const size_t output_width = round_down_po2((input_width + (2 /* padding */ - 3 /* kernel size */ + 2 /* subsampling */) * sizeof(float)) / 2, sizeof(float));

  const float* i0 = (const float*) ((uintptr_t) input - ((-padding_top) & input_width));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_width);
  if XNN_UNPREDICTABLE(padding_top != 0) {
    i0 = zero;
  }
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + output_width);

  size_t padded_input_height = input_height + padding_top + 1 /* padding bottom */;
  size_t output_height = (padded_input_height - 3 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 4) {
      i2 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 5) {
      i3 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 6) {
      i4 = zero;
    }

    float32x4_t vi0x1357 = vmovq_n_f32(0.0f);
    float32x4_t vi1x1357 = vmovq_n_f32(0.0f);
    float32x4_t vi2x1357 = vmovq_n_f32(0.0f);
    float32x4_t vi3x1357 = vmovq_n_f32(0.0f);
    float32x4_t vi4x1357 = vmovq_n_f32(0.0f);

    size_t w = input_width;
    for (; w >= 8 * sizeof(float); w -= 8 * sizeof(float)) {
      float32x4_t vo0p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo1p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);

      const float32x4x2_t vi0x8ACE9BDF = vld2q_f32(i0); i0 += 8;
      const float32x4x2_t vi1x8ACE9BDF = vld2q_f32(i1); i1 += 8;
      const float32x4x2_t vi2x8ACE9BDF = vld2q_f32(i2); i2 += 8;
      const float32x4x2_t vi3x8ACE9BDF = vld2q_f32(i3); i3 += 8;
      const float32x4x2_t vi4x8ACE9BDF = vld2q_f32(i4); i4 += 8;

      float32x4_t vo0p1 = vmulq_lane_f32(vi0x8ACE9BDF.val[0], vget_high_f32(vw0123), 0);
      float32x4_t vo1p1 = vmulq_lane_f32(vi2x8ACE9BDF.val[0], vget_high_f32(vw0123), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x8ACE9BDF.val[0], vget_low_f32(vw4567), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x8ACE9BDF.val[0], vget_low_f32(vw4567), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x8ACE9BDF.val[0], vw89, 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi4x8ACE9BDF.val[0], vw89, 0);

      const float32x4_t vi0x79BD = vextq_f32(vi0x1357, vi0x8ACE9BDF.val[1], 3);
      vi0x1357 = vi0x8ACE9BDF.val[1];
      const float32x4_t vi1x79BD = vextq_f32(vi1x1357, vi1x8ACE9BDF.val[1], 3);
      vi1x1357 = vi1x8ACE9BDF.val[1];
      const float32x4_t vi2x79BD = vextq_f32(vi2x1357, vi2x8ACE9BDF.val[1], 3);
      vi2x1357 = vi2x8ACE9BDF.val[1];
      const float32x4_t vi3x79BD = vextq_f32(vi3x1357, vi3x8ACE9BDF.val[1], 3);
      vi3x1357 = vi3x8ACE9BDF.val[1];
      const float32x4_t vi4x79BD = vextq_f32(vi4x1357, vi4x8ACE9BDF.val[1], 3);
      vi4x1357 = vi4x8ACE9BDF.val[1];

      vo0p1 = vfmaq_lane_f32(vo0p1, vi0x79BD, vget_low_f32(vw0123), 1);
      vo1p1 = vfmaq_lane_f32(vo1p1, vi2x79BD, vget_low_f32(vw0123), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x79BD, vget_low_f32(vw4567), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x79BD, vget_low_f32(vw4567), 0);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi2x79BD, vget_high_f32(vw4567), 1);
      vo1p1 = vfmaq_lane_f32(vo1p1, vi4x79BD, vget_high_f32(vw4567), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x8ACE9BDF.val[1], vget_high_f32(vw0123), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x8ACE9BDF.val[1], vget_high_f32(vw0123), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi1x8ACE9BDF.val[1], vget_high_f32(vw4567), 0);
      vo1p1 = vfmaq_lane_f32(vo1p1, vi3x8ACE9BDF.val[1], vget_high_f32(vw4567), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x8ACE9BDF.val[1], vw89, 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi4x8ACE9BDF.val[1], vw89, 1);

      vo0p0 = vaddq_f32(vo0p0, vo0p1);
      vo1p0 = vaddq_f32(vo1p0, vo1p1);

      float32x4_t vo0 = vmaxq_f32(vo0p0, vmin);
      float32x4_t vo1 = vmaxq_f32(vo1p0, vmin);

      vo0 = vminq_f32(vo0, vmax);
      vo1 = vminq_f32(vo1, vmax);

      vst1q_f32(o1, vo1); o1 += 4;
      vst1q_f32(o0, vo0); o0 += 4;
    }
    // Last block has 0-7 pixels to process.
    assert(w < 8 * sizeof(float));
    if XNN_LIKELY(w != 0) {
      float32x4_t vo0p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo1p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);

      const float32x4x2_t vi0x8ACE9BDF = vld2q_f32(i0);
      const float32x4x2_t vi1x8ACE9BDF = vld2q_f32(i1);
      const float32x4x2_t vi2x8ACE9BDF = vld2q_f32(i2);
      const float32x4x2_t vi3x8ACE9BDF = vld2q_f32(i3);
      const float32x4x2_t vi4x8ACE9BDF = vld2q_f32(i4);

      const float32x4_t vi0x8ACE = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi0x8ACE9BDF.val[0])));
      const float32x4_t vi0x9BDF = vreinterpretq_f32_u32(vandq_u32(vmask_odd,  vreinterpretq_u32_f32(vi0x8ACE9BDF.val[1])));
      const float32x4_t vi1x8ACE = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi1x8ACE9BDF.val[0])));
      const float32x4_t vi1x9BDF = vreinterpretq_f32_u32(vandq_u32(vmask_odd,  vreinterpretq_u32_f32(vi1x8ACE9BDF.val[1])));
      const float32x4_t vi2x8ACE = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi2x8ACE9BDF.val[0])));
      const float32x4_t vi2x9BDF = vreinterpretq_f32_u32(vandq_u32(vmask_odd,  vreinterpretq_u32_f32(vi2x8ACE9BDF.val[1])));
      const float32x4_t vi3x8ACE = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi3x8ACE9BDF.val[0])));
      const float32x4_t vi3x9BDF = vreinterpretq_f32_u32(vandq_u32(vmask_odd,  vreinterpretq_u32_f32(vi3x8ACE9BDF.val[1])));
      const float32x4_t vi4x8ACE = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi4x8ACE9BDF.val[0])));
      const float32x4_t vi4x9BDF = vreinterpretq_f32_u32(vandq_u32(vmask_odd,  vreinterpretq_u32_f32(vi4x8ACE9BDF.val[1])));

      float32x4_t vo0p1 = vmulq_lane_f32(vi0x8ACE, vget_high_f32(vw0123), 0);
      float32x4_t vo1p1 = vmulq_lane_f32(vi2x8ACE, vget_high_f32(vw0123), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x8ACE, vget_low_f32(vw4567), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x8ACE, vget_low_f32(vw4567), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x8ACE, vw89, 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi4x8ACE, vw89, 0);

      const float32x4_t vi0x79BD = vextq_f32(vi0x1357, vi0x9BDF, 3);
      const float32x4_t vi1x79BD = vextq_f32(vi1x1357, vi1x9BDF, 3);
      const float32x4_t vi2x79BD = vextq_f32(vi2x1357, vi2x9BDF, 3);
      const float32x4_t vi3x79BD = vextq_f32(vi3x1357, vi3x9BDF, 3);
      const float32x4_t vi4x79BD = vextq_f32(vi4x1357, vi4x9BDF, 3);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi0x79BD, vget_low_f32(vw0123), 1);
      vo1p1 = vfmaq_lane_f32(vo1p1, vi2x79BD, vget_low_f32(vw0123), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x79BD, vget_low_f32(vw4567), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x79BD, vget_low_f32(vw4567), 0);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi2x79BD, vget_high_f32(vw4567), 1);
      vo1p1 = vfmaq_lane_f32(vo1p1, vi4x79BD, vget_high_f32(vw4567), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x9BDF, vget_high_f32(vw0123), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x9BDF, vget_high_f32(vw0123), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi1x9BDF, vget_high_f32(vw4567), 0);
      vo1p1 = vfmaq_lane_f32(vo1p1, vi3x9BDF, vget_high_f32(vw4567), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x9BDF, vw89, 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi4x9BDF, vw89, 1);

      vo0p0 = vaddq_f32(vo0p0, vo0p1);
      vo1p0 = vaddq_f32(vo1p0, vo1p1);

      float32x4_t vo0 = vmaxq_f32(vo0p0, vmin);
      float32x4_t vo1 = vmaxq_f32(vo1p0, vmin);

      vo0 = vminq_f32(vo0, vmax);
      vo1 = vminq_f32(vo1, vmax);

      w += 1 * sizeof(float);
      if (w & (8 * sizeof(float))) {
        vst1q_f32(o1, vo1); o1 += 4;
        vst1q_f32(o0, vo0); o0 += 4;
      } else {
        float32x2_t vo0_lo = vget_low_f32(vo0);
        float32x2_t vo1_lo = vget_low_f32(vo1);
        if (w & (4 * sizeof(float))) {
          vst1_f32(o1, vo1_lo); o1 += 2;
          vst1_f32(o0, vo0_lo); o0 += 2;

          vo0_lo = vget_high_f32(vo0);
          vo1_lo = vget_high_f32(vo1);
        }
        if (w & (2 * sizeof(float))) {
          vst1_lane_f32(o1, vo1_lo, 0); o1 += 1;
          vst1_lane_f32(o0, vo0_lo, 0); o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i4 - input_decrement);
    i1 = (const float*) ((uintptr_t) i0 + input_width);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);

    o0 = o1;
    o1 = (float*) ((uintptr_t) o0 + output_width);

    output_height = doz(output_height, 2);
    padded_input_height = doz(padded_input_height, 4);
  } while (output_height != 0);
}

void xnn_f32_dwconv2d_chw_ukernel_5x5p2__aarch64_neonfma_4x4(
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
  const float* i7 = (const float*) ((uintptr_t) i6 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + input_width);
  float* o2 = (float*) ((uintptr_t) o1 + input_width);
  float* o3 = (float*) ((uintptr_t) o2 + input_width);

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
      o3 = o2;
    }
    if XNN_UNPREDICTABLE(output_height < 5) {
      i6 = zero;
    }
    if XNN_UNPREDICTABLE(output_height < 6) {
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
    for (; w > 8 * sizeof(float); w -= 4 * sizeof(float)) {
      float32x4_t vo0p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo1p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo2p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo3p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);

      const float32x4_t vi0x89AB = vld1q_f32(i0); i0 += 4;
      const float32x4_t vi1x89AB = vld1q_f32(i1); i1 += 4;
      const float32x4_t vi2x89AB = vld1q_f32(i2); i2 += 4;
      const float32x4_t vi3x89AB = vld1q_f32(i3); i3 += 4;
      const float32x4_t vi4x89AB = vld1q_f32(i4); i4 += 4;
      const float32x4_t vi5x89AB = vld1q_f32(i5); i5 += 4;
      const float32x4_t vi6x89AB = vld1q_f32(i6); i6 += 4;
      const float32x4_t vi7x89AB = vld1q_f32(i7); i7 += 4;

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x4567, vget_high_f32(vw0123), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi1x4567, vget_high_f32(vw0123), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi2x4567, vget_high_f32(vw0123), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi3x4567, vget_high_f32(vw0123), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x4567, vget_low_f32(vw89AB), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x4567, vget_low_f32(vw89AB), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi3x4567, vget_low_f32(vw89AB), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi4x4567, vget_low_f32(vw89AB), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x4567, vget_low_f32(vwCDEF), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x4567, vget_low_f32(vwCDEF), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi4x4567, vget_low_f32(vwCDEF), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi5x4567, vget_low_f32(vwCDEF), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi3x4567, vget_high_f32(vwGHIJ), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi4x4567, vget_high_f32(vwGHIJ), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi5x4567, vget_high_f32(vwGHIJ), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi6x4567, vget_high_f32(vwGHIJ), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi4x4567, vget_high_f32(vwKLMN), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi5x4567, vget_high_f32(vwKLMN), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi6x4567, vget_high_f32(vwKLMN), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi7x4567, vget_high_f32(vwKLMN), 1);

      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);
      const float32x4_t vi3x3456 = vextq_f32(vi3x0123, vi3x4567, 3);
      const float32x4_t vi4x3456 = vextq_f32(vi4x0123, vi4x4567, 3);
      const float32x4_t vi5x3456 = vextq_f32(vi5x0123, vi5x4567, 3);
      const float32x4_t vi6x3456 = vextq_f32(vi6x0123, vi6x4567, 3);
      const float32x4_t vi7x3456 = vextq_f32(vi7x0123, vi7x4567, 3);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x3456, vget_high_f32(vw0123), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi1x3456, vget_high_f32(vw0123), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi2x3456, vget_high_f32(vw0123), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi3x3456, vget_high_f32(vw0123), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x3456, vget_high_f32(vw4567), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x3456, vget_high_f32(vw4567), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi3x3456, vget_high_f32(vw4567), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi4x3456, vget_high_f32(vw4567), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x3456, vget_low_f32(vwCDEF), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x3456, vget_low_f32(vwCDEF), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi4x3456, vget_low_f32(vwCDEF), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi5x3456, vget_low_f32(vwCDEF), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi3x3456, vget_low_f32(vwGHIJ), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi4x3456, vget_low_f32(vwGHIJ), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi5x3456, vget_low_f32(vwGHIJ), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi6x3456, vget_low_f32(vwGHIJ), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi4x3456, vget_high_f32(vwKLMN), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi5x3456, vget_high_f32(vwKLMN), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi6x3456, vget_high_f32(vwKLMN), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi7x3456, vget_high_f32(vwKLMN), 0);

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
      const float32x4_t vi7x2345 = vextq_f32(vi7x0123, vi7x4567, 2);
      vi7x0123 = vi7x4567;

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x2345, vget_low_f32(vw0123), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi1x2345, vget_low_f32(vw0123), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi2x2345, vget_low_f32(vw0123), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi3x2345, vget_low_f32(vw0123), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x2345, vget_high_f32(vw4567), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x2345, vget_high_f32(vw4567), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi3x2345, vget_high_f32(vw4567), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi4x2345, vget_high_f32(vw4567), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x2345, vget_high_f32(vw89AB), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x2345, vget_high_f32(vw89AB), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi4x2345, vget_high_f32(vw89AB), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi5x2345, vget_high_f32(vw89AB), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi3x2345, vget_low_f32(vwGHIJ), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi4x2345, vget_low_f32(vwGHIJ), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi5x2345, vget_low_f32(vwGHIJ), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi6x2345, vget_low_f32(vwGHIJ), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi4x2345, vget_low_f32(vwKLMN), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi5x2345, vget_low_f32(vwKLMN), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi6x2345, vget_low_f32(vwKLMN), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi7x2345, vget_low_f32(vwKLMN), 1);

      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vi0x89AB, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vi1x89AB, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vi2x89AB, 1);
      const float32x4_t vi3x5678 = vextq_f32(vi3x4567, vi3x89AB, 1);
      const float32x4_t vi4x5678 = vextq_f32(vi4x4567, vi4x89AB, 1);
      const float32x4_t vi5x5678 = vextq_f32(vi5x4567, vi5x89AB, 1);
      const float32x4_t vi6x5678 = vextq_f32(vi6x4567, vi6x89AB, 1);
      const float32x4_t vi7x5678 = vextq_f32(vi7x4567, vi7x89AB, 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x5678, vget_low_f32(vw4567), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi1x5678, vget_low_f32(vw4567), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi2x5678, vget_low_f32(vw4567), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi3x5678, vget_low_f32(vw4567), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x5678, vget_low_f32(vw89AB), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x5678, vget_low_f32(vw89AB), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi3x5678, vget_low_f32(vw89AB), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi4x5678, vget_low_f32(vw89AB), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x5678, vget_high_f32(vwCDEF), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x5678, vget_high_f32(vwCDEF), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi4x5678, vget_high_f32(vwCDEF), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi5x5678, vget_high_f32(vwCDEF), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi3x5678, vget_high_f32(vwGHIJ), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi4x5678, vget_high_f32(vwGHIJ), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi5x5678, vget_high_f32(vwGHIJ), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi6x5678, vget_high_f32(vwGHIJ), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi4x5678, vwOP, 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi5x5678, vwOP, 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi6x5678, vwOP, 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi7x5678, vwOP, 0);

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
      const float32x4_t vi7x6789 = vextq_f32(vi7x4567, vi7x89AB, 2);
      vi7x4567 = vi7x89AB;

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x6789, vget_low_f32(vw4567), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi1x6789, vget_low_f32(vw4567), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi2x6789, vget_low_f32(vw4567), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi3x6789, vget_low_f32(vw4567), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x6789, vget_high_f32(vw89AB), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x6789, vget_high_f32(vw89AB), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi3x6789, vget_high_f32(vw89AB), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi4x6789, vget_high_f32(vw89AB), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x6789, vget_high_f32(vwCDEF), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x6789, vget_high_f32(vwCDEF), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi4x6789, vget_high_f32(vwCDEF), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi5x6789, vget_high_f32(vwCDEF), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi3x6789, vget_low_f32(vwKLMN), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi4x6789, vget_low_f32(vwKLMN), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi5x6789, vget_low_f32(vwKLMN), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi6x6789, vget_low_f32(vwKLMN), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi4x6789, vwOP, 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi5x6789, vwOP, 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi6x6789, vwOP, 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi7x6789, vwOP, 1);


      float32x4_t vo0 = vmaxq_f32(vo0p0, vmin);
      float32x4_t vo1 = vmaxq_f32(vo1p0, vmin);
      float32x4_t vo2 = vmaxq_f32(vo2p0, vmin);
      float32x4_t vo3 = vmaxq_f32(vo3p0, vmin);

      vo0 = vminq_f32(vo0, vmax);
      vo1 = vminq_f32(vo1, vmax);
      vo2 = vminq_f32(vo2, vmax);
      vo3 = vminq_f32(vo3, vmax);

      vst1q_f32(o3, vo3); o3 += 4;
      vst1q_f32(o2, vo2); o2 += 4;
      vst1q_f32(o1, vo1); o1 += 4;
      vst1q_f32(o0, vo0); o0 += 4;
    }
    // Always process the last block of 5..8 pixels.
    if XNN_LIKELY(w > 4 * sizeof(float)) {
      float32x4_t vo0p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo1p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo2p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo3p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);

      float32x4_t vi0x89AB = vld1q_f32(i0); i0 += 4;
      float32x4_t vi1x89AB = vld1q_f32(i1); i1 += 4;
      float32x4_t vi2x89AB = vld1q_f32(i2); i2 += 4;
      float32x4_t vi3x89AB = vld1q_f32(i3); i3 += 4;
      float32x4_t vi4x89AB = vld1q_f32(i4); i4 += 4;
      float32x4_t vi5x89AB = vld1q_f32(i5); i5 += 4;
      float32x4_t vi6x89AB = vld1q_f32(i6); i6 += 4;
      float32x4_t vi7x89AB = vld1q_f32(i7); i7 += 4;

      vi0x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi0x89AB)));
      vi1x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi1x89AB)));
      vi2x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi2x89AB)));
      vi3x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi3x89AB)));
      vi4x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi4x89AB)));
      vi5x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi5x89AB)));
      vi6x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi6x89AB)));
      vi7x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi7x89AB)));

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x4567, vget_high_f32(vw0123), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi1x4567, vget_high_f32(vw0123), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi2x4567, vget_high_f32(vw0123), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi3x4567, vget_high_f32(vw0123), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x4567, vget_low_f32(vw89AB), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x4567, vget_low_f32(vw89AB), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi3x4567, vget_low_f32(vw89AB), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi4x4567, vget_low_f32(vw89AB), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x4567, vget_low_f32(vwCDEF), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x4567, vget_low_f32(vwCDEF), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi4x4567, vget_low_f32(vwCDEF), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi5x4567, vget_low_f32(vwCDEF), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi3x4567, vget_high_f32(vwGHIJ), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi4x4567, vget_high_f32(vwGHIJ), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi5x4567, vget_high_f32(vwGHIJ), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi6x4567, vget_high_f32(vwGHIJ), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi4x4567, vget_high_f32(vwKLMN), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi5x4567, vget_high_f32(vwKLMN), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi6x4567, vget_high_f32(vwKLMN), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi7x4567, vget_high_f32(vwKLMN), 1);

      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);
      const float32x4_t vi3x3456 = vextq_f32(vi3x0123, vi3x4567, 3);
      const float32x4_t vi4x3456 = vextq_f32(vi4x0123, vi4x4567, 3);
      const float32x4_t vi5x3456 = vextq_f32(vi5x0123, vi5x4567, 3);
      const float32x4_t vi6x3456 = vextq_f32(vi6x0123, vi6x4567, 3);
      const float32x4_t vi7x3456 = vextq_f32(vi7x0123, vi7x4567, 3);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x3456, vget_high_f32(vw0123), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi1x3456, vget_high_f32(vw0123), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi2x3456, vget_high_f32(vw0123), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi3x3456, vget_high_f32(vw0123), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x3456, vget_high_f32(vw4567), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x3456, vget_high_f32(vw4567), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi3x3456, vget_high_f32(vw4567), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi4x3456, vget_high_f32(vw4567), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x3456, vget_low_f32(vwCDEF), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x3456, vget_low_f32(vwCDEF), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi4x3456, vget_low_f32(vwCDEF), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi5x3456, vget_low_f32(vwCDEF), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi3x3456, vget_low_f32(vwGHIJ), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi4x3456, vget_low_f32(vwGHIJ), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi5x3456, vget_low_f32(vwGHIJ), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi6x3456, vget_low_f32(vwGHIJ), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi4x3456, vget_high_f32(vwKLMN), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi5x3456, vget_high_f32(vwKLMN), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi6x3456, vget_high_f32(vwKLMN), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi7x3456, vget_high_f32(vwKLMN), 0);

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
      const float32x4_t vi7x2345 = vextq_f32(vi7x0123, vi7x4567, 2);
      vi7x0123 = vi7x4567;

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x2345, vget_low_f32(vw0123), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi1x2345, vget_low_f32(vw0123), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi2x2345, vget_low_f32(vw0123), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi3x2345, vget_low_f32(vw0123), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x2345, vget_high_f32(vw4567), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x2345, vget_high_f32(vw4567), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi3x2345, vget_high_f32(vw4567), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi4x2345, vget_high_f32(vw4567), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x2345, vget_high_f32(vw89AB), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x2345, vget_high_f32(vw89AB), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi4x2345, vget_high_f32(vw89AB), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi5x2345, vget_high_f32(vw89AB), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi3x2345, vget_low_f32(vwGHIJ), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi4x2345, vget_low_f32(vwGHIJ), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi5x2345, vget_low_f32(vwGHIJ), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi6x2345, vget_low_f32(vwGHIJ), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi4x2345, vget_low_f32(vwKLMN), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi5x2345, vget_low_f32(vwKLMN), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi6x2345, vget_low_f32(vwKLMN), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi7x2345, vget_low_f32(vwKLMN), 1);

      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vi0x89AB, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vi1x89AB, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vi2x89AB, 1);
      const float32x4_t vi3x5678 = vextq_f32(vi3x4567, vi3x89AB, 1);
      const float32x4_t vi4x5678 = vextq_f32(vi4x4567, vi4x89AB, 1);
      const float32x4_t vi5x5678 = vextq_f32(vi5x4567, vi5x89AB, 1);
      const float32x4_t vi6x5678 = vextq_f32(vi6x4567, vi6x89AB, 1);
      const float32x4_t vi7x5678 = vextq_f32(vi7x4567, vi7x89AB, 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x5678, vget_low_f32(vw4567), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi1x5678, vget_low_f32(vw4567), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi2x5678, vget_low_f32(vw4567), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi3x5678, vget_low_f32(vw4567), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x5678, vget_low_f32(vw89AB), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x5678, vget_low_f32(vw89AB), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi3x5678, vget_low_f32(vw89AB), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi4x5678, vget_low_f32(vw89AB), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x5678, vget_high_f32(vwCDEF), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x5678, vget_high_f32(vwCDEF), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi4x5678, vget_high_f32(vwCDEF), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi5x5678, vget_high_f32(vwCDEF), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi3x5678, vget_high_f32(vwGHIJ), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi4x5678, vget_high_f32(vwGHIJ), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi5x5678, vget_high_f32(vwGHIJ), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi6x5678, vget_high_f32(vwGHIJ), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi4x5678, vwOP, 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi5x5678, vwOP, 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi6x5678, vwOP, 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi7x5678, vwOP, 0);

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
      const float32x4_t vi7x6789 = vextq_f32(vi7x4567, vi7x89AB, 2);
      vi7x4567 = vi7x89AB;

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x6789, vget_low_f32(vw4567), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi1x6789, vget_low_f32(vw4567), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi2x6789, vget_low_f32(vw4567), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi3x6789, vget_low_f32(vw4567), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x6789, vget_high_f32(vw89AB), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x6789, vget_high_f32(vw89AB), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi3x6789, vget_high_f32(vw89AB), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi4x6789, vget_high_f32(vw89AB), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x6789, vget_high_f32(vwCDEF), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x6789, vget_high_f32(vwCDEF), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi4x6789, vget_high_f32(vwCDEF), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi5x6789, vget_high_f32(vwCDEF), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi3x6789, vget_low_f32(vwKLMN), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi4x6789, vget_low_f32(vwKLMN), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi5x6789, vget_low_f32(vwKLMN), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi6x6789, vget_low_f32(vwKLMN), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi4x6789, vwOP, 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi5x6789, vwOP, 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi6x6789, vwOP, 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi7x6789, vwOP, 1);


      float32x4_t vo0 = vmaxq_f32(vo0p0, vmin);
      float32x4_t vo1 = vmaxq_f32(vo1p0, vmin);
      float32x4_t vo2 = vmaxq_f32(vo2p0, vmin);
      float32x4_t vo3 = vmaxq_f32(vo3p0, vmin);

      vo0 = vminq_f32(vo0, vmax);
      vo1 = vminq_f32(vo1, vmax);
      vo2 = vminq_f32(vo2, vmax);
      vo3 = vminq_f32(vo3, vmax);

      vst1q_f32(o3, vo3); o3 += 4;
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
      float32x4_t vo3p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);

      vi0x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi0x4567)));
      vi1x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi1x4567)));
      vi2x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi2x4567)));
      vi3x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi3x4567)));
      vi4x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi4x4567)));
      vi5x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi5x4567)));
      vi6x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi6x4567)));
      vi7x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi7x4567)));

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x4567, vget_high_f32(vw0123), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi1x4567, vget_high_f32(vw0123), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi2x4567, vget_high_f32(vw0123), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi3x4567, vget_high_f32(vw0123), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x4567, vget_low_f32(vw89AB), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x4567, vget_low_f32(vw89AB), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi3x4567, vget_low_f32(vw89AB), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi4x4567, vget_low_f32(vw89AB), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x4567, vget_low_f32(vwCDEF), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x4567, vget_low_f32(vwCDEF), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi4x4567, vget_low_f32(vwCDEF), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi5x4567, vget_low_f32(vwCDEF), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi3x4567, vget_high_f32(vwGHIJ), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi4x4567, vget_high_f32(vwGHIJ), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi5x4567, vget_high_f32(vwGHIJ), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi6x4567, vget_high_f32(vwGHIJ), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi4x4567, vget_high_f32(vwKLMN), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi5x4567, vget_high_f32(vwKLMN), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi6x4567, vget_high_f32(vwKLMN), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi7x4567, vget_high_f32(vwKLMN), 1);

      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);
      const float32x4_t vi3x3456 = vextq_f32(vi3x0123, vi3x4567, 3);
      const float32x4_t vi4x3456 = vextq_f32(vi4x0123, vi4x4567, 3);
      const float32x4_t vi5x3456 = vextq_f32(vi5x0123, vi5x4567, 3);
      const float32x4_t vi6x3456 = vextq_f32(vi6x0123, vi6x4567, 3);
      const float32x4_t vi7x3456 = vextq_f32(vi7x0123, vi7x4567, 3);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x3456, vget_high_f32(vw0123), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi1x3456, vget_high_f32(vw0123), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi2x3456, vget_high_f32(vw0123), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi3x3456, vget_high_f32(vw0123), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x3456, vget_high_f32(vw4567), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x3456, vget_high_f32(vw4567), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi3x3456, vget_high_f32(vw4567), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi4x3456, vget_high_f32(vw4567), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x3456, vget_low_f32(vwCDEF), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x3456, vget_low_f32(vwCDEF), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi4x3456, vget_low_f32(vwCDEF), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi5x3456, vget_low_f32(vwCDEF), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi3x3456, vget_low_f32(vwGHIJ), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi4x3456, vget_low_f32(vwGHIJ), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi5x3456, vget_low_f32(vwGHIJ), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi6x3456, vget_low_f32(vwGHIJ), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi4x3456, vget_high_f32(vwKLMN), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi5x3456, vget_high_f32(vwKLMN), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi6x3456, vget_high_f32(vwKLMN), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi7x3456, vget_high_f32(vwKLMN), 0);

      const float32x4_t vi0x2345 = vextq_f32(vi0x0123, vi0x4567, 2);
      const float32x4_t vi1x2345 = vextq_f32(vi1x0123, vi1x4567, 2);
      const float32x4_t vi2x2345 = vextq_f32(vi2x0123, vi2x4567, 2);
      const float32x4_t vi3x2345 = vextq_f32(vi3x0123, vi3x4567, 2);
      const float32x4_t vi4x2345 = vextq_f32(vi4x0123, vi4x4567, 2);
      const float32x4_t vi5x2345 = vextq_f32(vi5x0123, vi5x4567, 2);
      const float32x4_t vi6x2345 = vextq_f32(vi6x0123, vi6x4567, 2);
      const float32x4_t vi7x2345 = vextq_f32(vi7x0123, vi7x4567, 2);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x2345, vget_low_f32(vw0123), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi1x2345, vget_low_f32(vw0123), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi2x2345, vget_low_f32(vw0123), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi3x2345, vget_low_f32(vw0123), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x2345, vget_high_f32(vw4567), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x2345, vget_high_f32(vw4567), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi3x2345, vget_high_f32(vw4567), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi4x2345, vget_high_f32(vw4567), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x2345, vget_high_f32(vw89AB), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x2345, vget_high_f32(vw89AB), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi4x2345, vget_high_f32(vw89AB), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi5x2345, vget_high_f32(vw89AB), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi3x2345, vget_low_f32(vwGHIJ), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi4x2345, vget_low_f32(vwGHIJ), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi5x2345, vget_low_f32(vwGHIJ), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi6x2345, vget_low_f32(vwGHIJ), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi4x2345, vget_low_f32(vwKLMN), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi5x2345, vget_low_f32(vwKLMN), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi6x2345, vget_low_f32(vwKLMN), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi7x2345, vget_low_f32(vwKLMN), 1);

      const float32x4_t vzero = vmovq_n_f32(0.0f);
      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vzero, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vzero, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vzero, 1);
      const float32x4_t vi3x5678 = vextq_f32(vi3x4567, vzero, 1);
      const float32x4_t vi4x5678 = vextq_f32(vi4x4567, vzero, 1);
      const float32x4_t vi5x5678 = vextq_f32(vi5x4567, vzero, 1);
      const float32x4_t vi6x5678 = vextq_f32(vi6x4567, vzero, 1);
      const float32x4_t vi7x5678 = vextq_f32(vi7x4567, vzero, 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x5678, vget_low_f32(vw4567), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi1x5678, vget_low_f32(vw4567), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi2x5678, vget_low_f32(vw4567), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi3x5678, vget_low_f32(vw4567), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x5678, vget_low_f32(vw89AB), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x5678, vget_low_f32(vw89AB), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi3x5678, vget_low_f32(vw89AB), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi4x5678, vget_low_f32(vw89AB), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x5678, vget_high_f32(vwCDEF), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x5678, vget_high_f32(vwCDEF), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi4x5678, vget_high_f32(vwCDEF), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi5x5678, vget_high_f32(vwCDEF), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi3x5678, vget_high_f32(vwGHIJ), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi4x5678, vget_high_f32(vwGHIJ), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi5x5678, vget_high_f32(vwGHIJ), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi6x5678, vget_high_f32(vwGHIJ), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi4x5678, vwOP, 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi5x5678, vwOP, 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi6x5678, vwOP, 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi7x5678, vwOP, 0);

      const float32x4_t vi0x6789 = vextq_f32(vi0x5678, vzero, 1);
      const float32x4_t vi1x6789 = vextq_f32(vi1x5678, vzero, 1);
      const float32x4_t vi2x6789 = vextq_f32(vi2x5678, vzero, 1);
      const float32x4_t vi3x6789 = vextq_f32(vi3x5678, vzero, 1);
      const float32x4_t vi4x6789 = vextq_f32(vi4x5678, vzero, 1);
      const float32x4_t vi5x6789 = vextq_f32(vi5x5678, vzero, 1);
      const float32x4_t vi6x6789 = vextq_f32(vi6x5678, vzero, 1);
      const float32x4_t vi7x6789 = vextq_f32(vi7x5678, vzero, 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x6789, vget_low_f32(vw4567), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi1x6789, vget_low_f32(vw4567), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi2x6789, vget_low_f32(vw4567), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi3x6789, vget_low_f32(vw4567), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x6789, vget_high_f32(vw89AB), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi2x6789, vget_high_f32(vw89AB), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi3x6789, vget_high_f32(vw89AB), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi4x6789, vget_high_f32(vw89AB), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x6789, vget_high_f32(vwCDEF), 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi3x6789, vget_high_f32(vwCDEF), 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi4x6789, vget_high_f32(vwCDEF), 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi5x6789, vget_high_f32(vwCDEF), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi3x6789, vget_low_f32(vwKLMN), 0);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi4x6789, vget_low_f32(vwKLMN), 0);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi5x6789, vget_low_f32(vwKLMN), 0);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi6x6789, vget_low_f32(vwKLMN), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi4x6789, vwOP, 1);
      vo1p0 = vfmaq_lane_f32(vo1p0, vi5x6789, vwOP, 1);
      vo2p0 = vfmaq_lane_f32(vo2p0, vi6x6789, vwOP, 1);
      vo3p0 = vfmaq_lane_f32(vo3p0, vi7x6789, vwOP, 1);


      float32x4_t vo0 = vmaxq_f32(vo0p0, vmin);
      float32x4_t vo1 = vmaxq_f32(vo1p0, vmin);
      float32x4_t vo2 = vmaxq_f32(vo2p0, vmin);
      float32x4_t vo3 = vmaxq_f32(vo3p0, vmin);

      vo0 = vminq_f32(vo0, vmax);
      vo1 = vminq_f32(vo1, vmax);
      vo2 = vminq_f32(vo2, vmax);
      vo3 = vminq_f32(vo3, vmax);

      if XNN_LIKELY(w & (4 * sizeof(float))) {
        vst1q_f32(o3, vo3); o3 += 4;
        vst1q_f32(o2, vo2); o2 += 4;
        vst1q_f32(o1, vo1); o1 += 4;
        vst1q_f32(o0, vo0); o0 += 4;
      } else {
        float32x2_t vo0_lo = vget_low_f32(vo0);
        float32x2_t vo1_lo = vget_low_f32(vo1);
        float32x2_t vo2_lo = vget_low_f32(vo2);
        float32x2_t vo3_lo = vget_low_f32(vo3);
        if (w & (2 * sizeof(float))) {
          vst1_f32(o3, vo3_lo); o3 += 2;
          vst1_f32(o2, vo2_lo); o2 += 2;
          vst1_f32(o1, vo1_lo); o1 += 2;
          vst1_f32(o0, vo0_lo); o0 += 2;

          vo0_lo = vget_high_f32(vo0);
          vo1_lo = vget_high_f32(vo1);
          vo2_lo = vget_high_f32(vo2);
          vo3_lo = vget_high_f32(vo3);
        }
        if (w & (1 * sizeof(float))) {
          vst1_lane_f32(o3, vo3_lo, 0); o3 += 1;
          vst1_lane_f32(o2, vo2_lo, 0); o2 += 1;
          vst1_lane_f32(o1, vo1_lo, 0); o1 += 1;
          vst1_lane_f32(o0, vo0_lo, 0); o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i4 - input_decrement);
    i1 = (const float*) ((uintptr_t) i5 - input_decrement);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);
    i5 = (const float*) ((uintptr_t) i4 + input_width);
    i6 = (const float*) ((uintptr_t) i5 + input_width);
    i7 = (const float*) ((uintptr_t) i6 + input_width);

    o0 = o3;
    o1 = (float*) ((uintptr_t) o0 + input_width);
    o2 = (float*) ((uintptr_t) o1 + input_width);
    o3 = (float*) ((uintptr_t) o2 + input_width);

    output_height = doz(output_height, 4);
  } while (output_height != 0);
}

void xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__aarch64_neonfma_1x4_acc2(
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
  assert(padding_top >= 1);
  assert(padding_top <= 2);

  const uint32x4_t vmask_even = vld1q_u32(params->neon_stride2.mask_even);
  const uint32x4_t vmask_odd = vld1q_u32(params->neon_stride2.mask_odd);
  const float32x4_t vmax = vld1q_dup_f32(&params->neon_stride2.max);
  const float32x4_t vmin = vld1q_dup_f32(&params->neon_stride2.min);

  const float32x4_t vw0123 = vld1q_f32(weights);
  const float32x4_t vw4567 = vld1q_f32(weights + 4);
  const float32x4_t vw89AB = vld1q_f32(weights + 8);
  const float32x4_t vwCDEF = vld1q_f32(weights + 12);
  const float32x4_t vwGHIJ = vld1q_f32(weights + 16);
  const float32x4_t vwKLMN = vld1q_f32(weights + 20);
  const float32x2_t vwOP   = vld1_f32(weights + 24);

  const uint32_t padding_top_less_1 = padding_top - 1;
  const size_t input_decrement = round_up_po2(input_width, 8 * sizeof(float));

  const float* i0 = zero;
  const float* i1 = (const float*) ((uintptr_t) input - ((-padding_top_less_1) & input_width));
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  if XNN_UNPREDICTABLE(padding_top_less_1 != 0) {
    i1 = zero;
  }
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);


  float* o0 = output;

  size_t padded_input_height = input_height + (padding_top_less_1 + 1) + 2 /* padding bottom */;
  size_t output_height = (padded_input_height - 5 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 6) {
      i3 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 7) {
      i4 = zero;
    }

    float32x4_t vi0x0246 = vmovq_n_f32(0.0f);
    float32x4_t vi1x0246 = vmovq_n_f32(0.0f);
    float32x4_t vi2x0246 = vmovq_n_f32(0.0f);
    float32x4_t vi3x0246 = vmovq_n_f32(0.0f);
    float32x4_t vi4x0246 = vmovq_n_f32(0.0f);

    float32x4_t vi0x1357 = vmovq_n_f32(0.0f);
    float32x4_t vi1x1357 = vmovq_n_f32(0.0f);
    float32x4_t vi2x1357 = vmovq_n_f32(0.0f);
    float32x4_t vi3x1357 = vmovq_n_f32(0.0f);
    float32x4_t vi4x1357 = vmovq_n_f32(0.0f);

    float32x4x2_t vi0x8ACE9BDF = vld2q_f32(i0); i0 += 8;
    float32x4x2_t vi1x8ACE9BDF = vld2q_f32(i1); i1 += 8;
    float32x4x2_t vi2x8ACE9BDF = vld2q_f32(i2); i2 += 8;
    float32x4x2_t vi3x8ACE9BDF = vld2q_f32(i3); i3 += 8;
    float32x4x2_t vi4x8ACE9BDF = vld2q_f32(i4); i4 += 8;

    size_t w = input_width;
    for (; w > 8 * sizeof(float); w -= 8 * sizeof(float)) {
      float32x4_t vo0p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);

      float32x4_t vo0p1 = vmulq_lane_f32(vi0x8ACE9BDF.val[0], vget_high_f32(vw0123), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x8ACE9BDF.val[0], vget_low_f32(vw89AB), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x8ACE9BDF.val[0], vget_low_f32(vwCDEF), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi3x8ACE9BDF.val[0], vget_high_f32(vwGHIJ), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi4x8ACE9BDF.val[0], vget_high_f32(vwKLMN), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi0x8ACE9BDF.val[1], vget_low_f32(vw4567), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x8ACE9BDF.val[1], vget_low_f32(vw89AB), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi2x8ACE9BDF.val[1], vget_high_f32(vwCDEF), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi3x8ACE9BDF.val[1], vget_high_f32(vwGHIJ), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi4x8ACE9BDF.val[1], vwOP, 0);

      const float32x4_t vi0x68AC = vextq_f32(vi0x0246, vi0x8ACE9BDF.val[0], 3);
      vi0x0246 = vi0x8ACE9BDF.val[0];
      const float32x4_t vi1x68AC = vextq_f32(vi1x0246, vi1x8ACE9BDF.val[0], 3);
      vi1x0246 = vi1x8ACE9BDF.val[0];
      const float32x4_t vi2x68AC = vextq_f32(vi2x0246, vi2x8ACE9BDF.val[0], 3);
      vi2x0246 = vi2x8ACE9BDF.val[0];
      const float32x4_t vi3x68AC = vextq_f32(vi3x0246, vi3x8ACE9BDF.val[0], 3);
      vi3x0246 = vi3x8ACE9BDF.val[0];
      const float32x4_t vi4x68AC = vextq_f32(vi4x0246, vi4x8ACE9BDF.val[0], 3);
      vi4x0246 = vi4x8ACE9BDF.val[0];

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x68AC, vget_low_f32(vw0123), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi1x68AC, vget_high_f32(vw4567), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x68AC, vget_high_f32(vw89AB), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi3x68AC, vget_low_f32(vwGHIJ), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi4x68AC, vget_low_f32(vwKLMN), 1);

      const float32x4_t vi0x79BD = vextq_f32(vi0x1357, vi0x8ACE9BDF.val[1], 3);
      vi0x1357 = vi0x8ACE9BDF.val[1];
      const float32x4_t vi1x79BD = vextq_f32(vi1x1357, vi1x8ACE9BDF.val[1], 3);
      vi1x1357 = vi1x8ACE9BDF.val[1];
      const float32x4_t vi2x79BD = vextq_f32(vi2x1357, vi2x8ACE9BDF.val[1], 3);
      vi2x1357 = vi2x8ACE9BDF.val[1];
      const float32x4_t vi3x79BD = vextq_f32(vi3x1357, vi3x8ACE9BDF.val[1], 3);
      vi3x1357 = vi3x8ACE9BDF.val[1];
      const float32x4_t vi4x79BD = vextq_f32(vi4x1357, vi4x8ACE9BDF.val[1], 3);
      vi4x1357 = vi4x8ACE9BDF.val[1];

      const float32x4x2_t vi0xGIKMHJLN = vld2q_f32(i0); i0 += 8;
      const float32x4x2_t vi1xGIKMHJLN = vld2q_f32(i1); i1 += 8;
      const float32x4x2_t vi2xGIKMHJLN = vld2q_f32(i2); i2 += 8;
      const float32x4x2_t vi3xGIKMHJLN = vld2q_f32(i3); i3 += 8;
      const float32x4x2_t vi4xGIKMHJLN = vld2q_f32(i4); i4 += 8;

      vo0p1 = vfmaq_lane_f32(vo0p1, vi0x79BD, vget_high_f32(vw0123), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x79BD, vget_high_f32(vw4567), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi2x79BD, vget_low_f32(vwCDEF), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi3x79BD, vget_low_f32(vwGHIJ), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi4x79BD, vget_high_f32(vwKLMN), 0);

      const float32x4_t vi0xACEG = vextq_f32(vi0x8ACE9BDF.val[0], vi0xGIKMHJLN.val[0], 1);
      vi0x8ACE9BDF = vi0xGIKMHJLN;
      const float32x4_t vi1xACEG = vextq_f32(vi1x8ACE9BDF.val[0], vi1xGIKMHJLN.val[0], 1);
      vi1x8ACE9BDF = vi1xGIKMHJLN;
      const float32x4_t vi2xACEG = vextq_f32(vi2x8ACE9BDF.val[0], vi2xGIKMHJLN.val[0], 1);
      vi2x8ACE9BDF = vi2xGIKMHJLN;
      const float32x4_t vi3xACEG = vextq_f32(vi3x8ACE9BDF.val[0], vi3xGIKMHJLN.val[0], 1);
      vi3x8ACE9BDF = vi3xGIKMHJLN;
      const float32x4_t vi4xACEG = vextq_f32(vi4x8ACE9BDF.val[0], vi4xGIKMHJLN.val[0], 1);
      vi4x8ACE9BDF = vi4xGIKMHJLN;

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0xACEG, vget_low_f32(vw4567), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi1xACEG, vget_high_f32(vw89AB), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2xACEG, vget_high_f32(vwCDEF), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi3xACEG, vget_low_f32(vwKLMN), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi4xACEG, vwOP, 1);

      vo0p0 = vaddq_f32(vo0p0, vo0p1);

      float32x4_t vo0 = vmaxq_f32(vo0p0, vmin);

      vo0 = vminq_f32(vo0, vmax);

      vst1q_f32(o0, vo0); o0 += 4;
    }
    // Last block has 1-8 pixels to process.
    assert(w <= 8 * sizeof(float));
    assert(w >= 1 * sizeof(float));
    {
      float32x4_t vo0p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);

      const float32x4_t vi0x8ACE = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi0x8ACE9BDF.val[0])));
      const float32x4_t vi1x8ACE = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi1x8ACE9BDF.val[0])));
      const float32x4_t vi2x8ACE = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi2x8ACE9BDF.val[0])));
      const float32x4_t vi3x8ACE = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi3x8ACE9BDF.val[0])));
      const float32x4_t vi4x8ACE = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi4x8ACE9BDF.val[0])));

      const float32x4_t vi0x9BDF = vreinterpretq_f32_u32(vandq_u32(vmask_odd, vreinterpretq_u32_f32(vi0x8ACE9BDF.val[1])));
      const float32x4_t vi1x9BDF = vreinterpretq_f32_u32(vandq_u32(vmask_odd, vreinterpretq_u32_f32(vi1x8ACE9BDF.val[1])));
      const float32x4_t vi2x9BDF = vreinterpretq_f32_u32(vandq_u32(vmask_odd, vreinterpretq_u32_f32(vi2x8ACE9BDF.val[1])));
      const float32x4_t vi3x9BDF = vreinterpretq_f32_u32(vandq_u32(vmask_odd, vreinterpretq_u32_f32(vi3x8ACE9BDF.val[1])));
      const float32x4_t vi4x9BDF = vreinterpretq_f32_u32(vandq_u32(vmask_odd, vreinterpretq_u32_f32(vi4x8ACE9BDF.val[1])));

      float32x4_t vo0p1 = vmulq_lane_f32(vi0x8ACE, vget_high_f32(vw0123), 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x8ACE, vget_low_f32(vw89AB), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x8ACE, vget_low_f32(vwCDEF), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi3x8ACE, vget_high_f32(vwGHIJ), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi4x8ACE, vget_high_f32(vwKLMN), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi0x9BDF, vget_low_f32(vw4567), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x9BDF, vget_low_f32(vw89AB), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi2x9BDF, vget_high_f32(vwCDEF), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi3x9BDF, vget_high_f32(vwGHIJ), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi4x9BDF, vwOP, 0);

      const float32x4_t vi0x68AC = vextq_f32(vi0x0246, vi0x8ACE, 3);
      const float32x4_t vi1x68AC = vextq_f32(vi1x0246, vi1x8ACE, 3);
      const float32x4_t vi2x68AC = vextq_f32(vi2x0246, vi2x8ACE, 3);
      const float32x4_t vi3x68AC = vextq_f32(vi3x0246, vi3x8ACE, 3);
      const float32x4_t vi4x68AC = vextq_f32(vi4x0246, vi4x8ACE, 3);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0x68AC, vget_low_f32(vw0123), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi1x68AC, vget_high_f32(vw4567), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2x68AC, vget_high_f32(vw89AB), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi3x68AC, vget_low_f32(vwGHIJ), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi4x68AC, vget_low_f32(vwKLMN), 1);

      const float32x4_t vi0x79BD = vextq_f32(vi0x1357, vi0x9BDF, 3);
      const float32x4_t vi1x79BD = vextq_f32(vi1x1357, vi1x9BDF, 3);
      const float32x4_t vi2x79BD = vextq_f32(vi2x1357, vi2x9BDF, 3);
      const float32x4_t vi3x79BD = vextq_f32(vi3x1357, vi3x9BDF, 3);
      const float32x4_t vi4x79BD = vextq_f32(vi4x1357, vi4x9BDF, 3);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi0x79BD, vget_high_f32(vw0123), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi1x79BD, vget_high_f32(vw4567), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi2x79BD, vget_low_f32(vwCDEF), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi3x79BD, vget_low_f32(vwGHIJ), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi4x79BD, vget_high_f32(vwKLMN), 0);

      const float32x4_t vzero = vmovq_n_f32(0.0f);
      const float32x4_t vi0xACEG = vextq_f32(vi0x8ACE, vzero, 1);
      const float32x4_t vi1xACEG = vextq_f32(vi1x8ACE, vzero, 1);
      const float32x4_t vi2xACEG = vextq_f32(vi2x8ACE, vzero, 1);
      const float32x4_t vi3xACEG = vextq_f32(vi3x8ACE, vzero, 1);
      const float32x4_t vi4xACEG = vextq_f32(vi4x8ACE, vzero, 1);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi0xACEG, vget_low_f32(vw4567), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi1xACEG, vget_high_f32(vw89AB), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi2xACEG, vget_high_f32(vwCDEF), 1);

      vo0p1 = vfmaq_lane_f32(vo0p1, vi3xACEG, vget_low_f32(vwKLMN), 0);

      vo0p0 = vfmaq_lane_f32(vo0p0, vi4xACEG, vwOP, 1);

      vo0p0 = vaddq_f32(vo0p0, vo0p1);

      float32x4_t vo0 = vmaxq_f32(vo0p0, vmin);

      vo0 = vminq_f32(vo0, vmax);

      size_t w_tmp = (w + 1 * sizeof(float)) / (2 * sizeof(float));
      if XNN_LIKELY(w_tmp >= 4) {
        vst1q_f32(o0, vo0); o0 += 4;
      } else {
        float32x2_t vo0_lo = vget_low_f32(vo0);
        if (w_tmp & 2) {
          vst1_f32(o0, vo0_lo); o0 += 2;

          vo0_lo = vget_high_f32(vo0);
        }
        if (w_tmp & 1) {
          vst1_lane_f32(o0, vo0_lo, 0); o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i2 - input_decrement);
    i1 = (const float*) ((uintptr_t) i3 - input_decrement);
    i2 = (const float*) ((uintptr_t) i4 - input_decrement);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);


    output_height -= 1;
    padded_input_height -= 2;
  } while (output_height != 0);
}

void xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  do {
    float32x4_t vacc0x0123 = vld1q_f32(w); w += 4;
    float32x4_t vacc0x4567 = vld1q_f32(w); w += 4;

    size_t k = kc;
    for (; k >= 2 * sizeof(float); k -= 2 * sizeof(float)) {
      const float32x2_t va0 = vld1_f32(a0); a0 += 2;

      const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

      vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c0, va0, 0);
      vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c0, va0, 0);
      const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

      vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c1, va0, 1);
      vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c1, va0, 1);
    }
    if XNN_UNLIKELY(k != 0) {
      const float32x4_t va0 = vld1q_dup_f32(a0); a0 += 1;

      const float32x4_t vb0123 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567 = vld1q_f32(w); w += 4;

      vacc0x0123 = vfmaq_f32(vacc0x0123, va0, vb0123);
      vacc0x4567 = vfmaq_f32(vacc0x4567, va0, vb4567);
    }
    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0123 = vminq_f32(vacc0x0123, vmax);
    vacc0x4567 = vminq_f32(vacc0x4567, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0123 = vmaxq_f32(vacc0x0123, vmin);
    vacc0x4567 = vmaxq_f32(vacc0x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vacc0x0123);
      vst1q_f32(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;

    } else {
      if (nc & 4) {
        vst1q_f32(c0, vacc0x0123); c0 += 4;

        vacc0x0123 = vacc0x4567;
      }
      float32x2_t vacc0x01 = vget_low_f32(vacc0x0123);
      if (nc & 2) {
        vst1_f32(c0, vacc0x01); c0 += 2;

        vacc0x01 = vget_high_f32(vacc0x0123);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vacc0x01, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    float32x2_t vacc0x01 = vld1_f32(w); w += 2;
    float32x2_t vacc1x01 = vacc0x01;
    float32x2_t vacc2x01 = vacc0x01;
    float32x2_t vacc3x01 = vacc0x01;

    size_t k = kc;
    for (; k >= 2 * sizeof(float); k -= 2 * sizeof(float)) {
      const float32x2_t va0 = vld1_f32(a0); a0 += 2;
      const float32x2_t va1 = vld1_f32(a1); a1 += 2;
      const float32x2_t va2 = vld1_f32(a2); a2 += 2;
      const float32x2_t va3 = vld1_f32(a3); a3 += 2;

      const float32x2_t vb01c0 = vld1_f32(w); w += 2;

      #if XNN_ARCH_ARM64
        vacc0x01 = vfma_lane_f32(vacc0x01, vb01c0, va0, 0);
        vacc1x01 = vfma_lane_f32(vacc1x01, vb01c0, va1, 0);
        vacc2x01 = vfma_lane_f32(vacc2x01, vb01c0, va2, 0);
        vacc3x01 = vfma_lane_f32(vacc3x01, vb01c0, va3, 0);
      #else
        const float32x2_t va0c0 = vdup_lane_f32(va0, 0);
        const float32x2_t va1c0 = vdup_lane_f32(va1, 0);
        const float32x2_t va2c0 = vdup_lane_f32(va2, 0);
        const float32x2_t va3c0 = vdup_lane_f32(va3, 0);
        vacc0x01 = vfma_f32(vacc0x01, va0c0, vb01c0);
        vacc1x01 = vfma_f32(vacc1x01, va1c0, vb01c0);
        vacc2x01 = vfma_f32(vacc2x01, va2c0, vb01c0);
        vacc3x01 = vfma_f32(vacc3x01, va3c0, vb01c0);
      #endif
      const float32x2_t vb01c1 = vld1_f32(w); w += 2;

      #if XNN_ARCH_ARM64
        vacc0x01 = vfma_lane_f32(vacc0x01, vb01c1, va0, 1);
        vacc1x01 = vfma_lane_f32(vacc1x01, vb01c1, va1, 1);
        vacc2x01 = vfma_lane_f32(vacc2x01, vb01c1, va2, 1);
        vacc3x01 = vfma_lane_f32(vacc3x01, vb01c1, va3, 1);
      #else
        const float32x2_t va0c1 = vdup_lane_f32(va0, 1);
        const float32x2_t va1c1 = vdup_lane_f32(va1, 1);
        const float32x2_t va2c1 = vdup_lane_f32(va2, 1);
        const float32x2_t va3c1 = vdup_lane_f32(va3, 1);
        vacc0x01 = vfma_f32(vacc0x01, va0c1, vb01c1);
        vacc1x01 = vfma_f32(vacc1x01, va1c1, vb01c1);
        vacc2x01 = vfma_f32(vacc2x01, va2c1, vb01c1);
        vacc3x01 = vfma_f32(vacc3x01, va3c1, vb01c1);
      #endif
    }
    if XNN_UNLIKELY(k != 0) {
      const float32x2_t va0 = vld1_dup_f32(a0); a0 += 1;
      const float32x2_t va1 = vld1_dup_f32(a1); a1 += 1;
      const float32x2_t va2 = vld1_dup_f32(a2); a2 += 1;
      const float32x2_t va3 = vld1_dup_f32(a3); a3 += 1;

      const float32x2_t vb01 = vld1_f32(w); w += 2;

      vacc0x01 = vfma_f32(vacc0x01, va0, vb01);
      vacc1x01 = vfma_f32(vacc1x01, va1, vb01);
      vacc2x01 = vfma_f32(vacc2x01, va2, vb01);
      vacc3x01 = vfma_f32(vacc3x01, va3, vb01);
    }

    const float32x2_t vmax = vld1_dup_f32(&params->scalar.max);
    vacc0x01 = vmin_f32(vacc0x01, vmax);
    vacc1x01 = vmin_f32(vacc1x01, vmax);
    vacc2x01 = vmin_f32(vacc2x01, vmax);
    vacc3x01 = vmin_f32(vacc3x01, vmax);

    const float32x2_t vmin = vld1_dup_f32(&params->scalar.min);
    vacc0x01 = vmax_f32(vacc0x01, vmin);
    vacc1x01 = vmax_f32(vacc1x01, vmin);
    vacc2x01 = vmax_f32(vacc2x01, vmin);
    vacc3x01 = vmax_f32(vacc3x01, vmin);

    if XNN_LIKELY(nc >= 2) {
      vst1_f32(c0, vacc0x01);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      vst1_f32(c1, vacc1x01);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1_f32(c2, vacc2x01);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1_f32(c3, vacc3x01);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);

      nc -= 2;
    } else {
      assert(nc == 1);
      vst1_lane_f32(c0, vacc0x01, 0);
      vst1_lane_f32(c1, vacc1x01, 0);
      vst1_lane_f32(c2, vacc2x01, 0);
      vst1_lane_f32(c3, vacc3x01, 0);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  do {
    float32x2_t vacc0x01 = vld1_f32(w); w += 2;
    float32x2_t vacc1x01 = vacc0x01;
    float32x2_t vacc2x01 = vacc0x01;
    float32x2_t vacc3x01 = vacc0x01;
    float32x2_t vacc4x01 = vacc0x01;
    float32x2_t vacc5x01 = vacc0x01;

    size_t k = kc;
    for (; k >= 2 * sizeof(float); k -= 2 * sizeof(float)) {
      const float32x2_t va0 = vld1_f32(a0); a0 += 2;
      const float32x2_t va1 = vld1_f32(a1); a1 += 2;
      const float32x2_t va2 = vld1_f32(a2); a2 += 2;
      const float32x2_t va3 = vld1_f32(a3); a3 += 2;
      const float32x2_t va4 = vld1_f32(a4); a4 += 2;
      const float32x2_t va5 = vld1_f32(a5); a5 += 2;

      const float32x2_t vb01c0 = vld1_f32(w); w += 2;

      #if XNN_ARCH_ARM64
        vacc0x01 = vfma_lane_f32(vacc0x01, vb01c0, va0, 0);
        vacc1x01 = vfma_lane_f32(vacc1x01, vb01c0, va1, 0);
        vacc2x01 = vfma_lane_f32(vacc2x01, vb01c0, va2, 0);
        vacc3x01 = vfma_lane_f32(vacc3x01, vb01c0, va3, 0);
        vacc4x01 = vfma_lane_f32(vacc4x01, vb01c0, va4, 0);
        vacc5x01 = vfma_lane_f32(vacc5x01, vb01c0, va5, 0);
      #else
        const float32x2_t va0c0 = vdup_lane_f32(va0, 0);
        const float32x2_t va1c0 = vdup_lane_f32(va1, 0);
        const float32x2_t va2c0 = vdup_lane_f32(va2, 0);
        const float32x2_t va3c0 = vdup_lane_f32(va3, 0);
        const float32x2_t va4c0 = vdup_lane_f32(va4, 0);
        const float32x2_t va5c0 = vdup_lane_f32(va5, 0);
        vacc0x01 = vfma_f32(vacc0x01, va0c0, vb01c0);
        vacc1x01 = vfma_f32(vacc1x01, va1c0, vb01c0);
        vacc2x01 = vfma_f32(vacc2x01, va2c0, vb01c0);
        vacc3x01 = vfma_f32(vacc3x01, va3c0, vb01c0);
        vacc4x01 = vfma_f32(vacc4x01, va4c0, vb01c0);
        vacc5x01 = vfma_f32(vacc5x01, va5c0, vb01c0);
      #endif
      const float32x2_t vb01c1 = vld1_f32(w); w += 2;

      #if XNN_ARCH_ARM64
        vacc0x01 = vfma_lane_f32(vacc0x01, vb01c1, va0, 1);
        vacc1x01 = vfma_lane_f32(vacc1x01, vb01c1, va1, 1);
        vacc2x01 = vfma_lane_f32(vacc2x01, vb01c1, va2, 1);
        vacc3x01 = vfma_lane_f32(vacc3x01, vb01c1, va3, 1);
        vacc4x01 = vfma_lane_f32(vacc4x01, vb01c1, va4, 1);
        vacc5x01 = vfma_lane_f32(vacc5x01, vb01c1, va5, 1);
      #else
        const float32x2_t va0c1 = vdup_lane_f32(va0, 1);
        const float32x2_t va1c1 = vdup_lane_f32(va1, 1);
        const float32x2_t va2c1 = vdup_lane_f32(va2, 1);
        const float32x2_t va3c1 = vdup_lane_f32(va3, 1);
        const float32x2_t va4c1 = vdup_lane_f32(va4, 1);
        const float32x2_t va5c1 = vdup_lane_f32(va5, 1);
        vacc0x01 = vfma_f32(vacc0x01, va0c1, vb01c1);
        vacc1x01 = vfma_f32(vacc1x01, va1c1, vb01c1);
        vacc2x01 = vfma_f32(vacc2x01, va2c1, vb01c1);
        vacc3x01 = vfma_f32(vacc3x01, va3c1, vb01c1);
        vacc4x01 = vfma_f32(vacc4x01, va4c1, vb01c1);
        vacc5x01 = vfma_f32(vacc5x01, va5c1, vb01c1);
      #endif
    }
    if XNN_UNLIKELY(k != 0) {
      const float32x2_t va0 = vld1_dup_f32(a0); a0 += 1;
      const float32x2_t va1 = vld1_dup_f32(a1); a1 += 1;
      const float32x2_t va2 = vld1_dup_f32(a2); a2 += 1;
      const float32x2_t va3 = vld1_dup_f32(a3); a3 += 1;
      const float32x2_t va4 = vld1_dup_f32(a4); a4 += 1;
      const float32x2_t va5 = vld1_dup_f32(a5); a5 += 1;

      const float32x2_t vb01 = vld1_f32(w); w += 2;

      vacc0x01 = vfma_f32(vacc0x01, va0, vb01);
      vacc1x01 = vfma_f32(vacc1x01, va1, vb01);
      vacc2x01 = vfma_f32(vacc2x01, va2, vb01);
      vacc3x01 = vfma_f32(vacc3x01, va3, vb01);
      vacc4x01 = vfma_f32(vacc4x01, va4, vb01);
      vacc5x01 = vfma_f32(vacc5x01, va5, vb01);
    }

    const float32x2_t vmax = vld1_dup_f32(&params->scalar.max);
    vacc0x01 = vmin_f32(vacc0x01, vmax);
    vacc1x01 = vmin_f32(vacc1x01, vmax);
    vacc2x01 = vmin_f32(vacc2x01, vmax);
    vacc3x01 = vmin_f32(vacc3x01, vmax);
    vacc4x01 = vmin_f32(vacc4x01, vmax);
    vacc5x01 = vmin_f32(vacc5x01, vmax);

    const float32x2_t vmin = vld1_dup_f32(&params->scalar.min);
    vacc0x01 = vmax_f32(vacc0x01, vmin);
    vacc1x01 = vmax_f32(vacc1x01, vmin);
    vacc2x01 = vmax_f32(vacc2x01, vmin);
    vacc3x01 = vmax_f32(vacc3x01, vmin);
    vacc4x01 = vmax_f32(vacc4x01, vmin);
    vacc5x01 = vmax_f32(vacc5x01, vmin);

    if XNN_LIKELY(nc >= 2) {
      vst1_f32(c0, vacc0x01);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      vst1_f32(c1, vacc1x01);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1_f32(c2, vacc2x01);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1_f32(c3, vacc3x01);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      vst1_f32(c4, vacc4x01);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      vst1_f32(c5, vacc5x01);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);

      nc -= 2;
    } else {
      assert(nc == 1);
      vst1_lane_f32(c0, vacc0x01, 0);
      vst1_lane_f32(c1, vacc1x01, 0);
      vst1_lane_f32(c2, vacc2x01, 0);
      vst1_lane_f32(c3, vacc3x01, 0);
      vst1_lane_f32(c4, vacc4x01, 0);
      vst1_lane_f32(c5, vacc5x01, 0);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  do {
    float32x4_t vacc0x0123 = vld1q_f32(w); w += 4;
    float32x4_t vacc0x4567 = vld1q_f32(w); w += 4;
    float32x4_t vacc1x0123 = vacc0x0123;
    float32x4_t vacc1x4567 = vacc0x4567;
    float32x4_t vacc2x0123 = vacc0x0123;
    float32x4_t vacc2x4567 = vacc0x4567;
    float32x4_t vacc3x0123 = vacc0x0123;
    float32x4_t vacc3x4567 = vacc0x4567;
    float32x4_t vacc4x0123 = vacc0x0123;
    float32x4_t vacc4x4567 = vacc0x4567;
    float32x4_t vacc5x0123 = vacc0x0123;
    float32x4_t vacc5x4567 = vacc0x4567;

    size_t k = kc;
    for (; k >= 2 * sizeof(float); k -= 2 * sizeof(float)) {
      const float32x2_t va0 = vld1_f32(a0); a0 += 2;
      const float32x2_t va1 = vld1_f32(a1); a1 += 2;
      const float32x2_t va2 = vld1_f32(a2); a2 += 2;
      const float32x2_t va3 = vld1_f32(a3); a3 += 2;
      const float32x2_t va4 = vld1_f32(a4); a4 += 2;
      const float32x2_t va5 = vld1_f32(a5); a5 += 2;

      const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

      vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c0, va0, 0);
      vacc1x0123 = vfmaq_lane_f32(vacc1x0123, vb0123c0, va1, 0);
      vacc2x0123 = vfmaq_lane_f32(vacc2x0123, vb0123c0, va2, 0);
      vacc3x0123 = vfmaq_lane_f32(vacc3x0123, vb0123c0, va3, 0);
      vacc4x0123 = vfmaq_lane_f32(vacc4x0123, vb0123c0, va4, 0);
      vacc5x0123 = vfmaq_lane_f32(vacc5x0123, vb0123c0, va5, 0);
      vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c0, va0, 0);
      vacc1x4567 = vfmaq_lane_f32(vacc1x4567, vb4567c0, va1, 0);
      vacc2x4567 = vfmaq_lane_f32(vacc2x4567, vb4567c0, va2, 0);
      vacc3x4567 = vfmaq_lane_f32(vacc3x4567, vb4567c0, va3, 0);
      vacc4x4567 = vfmaq_lane_f32(vacc4x4567, vb4567c0, va4, 0);
      vacc5x4567 = vfmaq_lane_f32(vacc5x4567, vb4567c0, va5, 0);
      const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

      vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c1, va0, 1);
      vacc1x0123 = vfmaq_lane_f32(vacc1x0123, vb0123c1, va1, 1);
      vacc2x0123 = vfmaq_lane_f32(vacc2x0123, vb0123c1, va2, 1);
      vacc3x0123 = vfmaq_lane_f32(vacc3x0123, vb0123c1, va3, 1);
      vacc4x0123 = vfmaq_lane_f32(vacc4x0123, vb0123c1, va4, 1);
      vacc5x0123 = vfmaq_lane_f32(vacc5x0123, vb0123c1, va5, 1);
      vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c1, va0, 1);
      vacc1x4567 = vfmaq_lane_f32(vacc1x4567, vb4567c1, va1, 1);
      vacc2x4567 = vfmaq_lane_f32(vacc2x4567, vb4567c1, va2, 1);
      vacc3x4567 = vfmaq_lane_f32(vacc3x4567, vb4567c1, va3, 1);
      vacc4x4567 = vfmaq_lane_f32(vacc4x4567, vb4567c1, va4, 1);
      vacc5x4567 = vfmaq_lane_f32(vacc5x4567, vb4567c1, va5, 1);
    }
    if XNN_UNLIKELY(k != 0) {
      const float32x4_t va0 = vld1q_dup_f32(a0); a0 += 1;
      const float32x4_t va1 = vld1q_dup_f32(a1); a1 += 1;
      const float32x4_t va2 = vld1q_dup_f32(a2); a2 += 1;
      const float32x4_t va3 = vld1q_dup_f32(a3); a3 += 1;
      const float32x4_t va4 = vld1q_dup_f32(a4); a4 += 1;
      const float32x4_t va5 = vld1q_dup_f32(a5); a5 += 1;

      const float32x4_t vb0123 = vld1q_f32(w); w += 4;
      const float32x4_t vb4567 = vld1q_f32(w); w += 4;

      vacc0x0123 = vfmaq_f32(vacc0x0123, va0, vb0123);
      vacc1x0123 = vfmaq_f32(vacc1x0123, va1, vb0123);
      vacc2x0123 = vfmaq_f32(vacc2x0123, va2, vb0123);
      vacc3x0123 = vfmaq_f32(vacc3x0123, va3, vb0123);
      vacc4x0123 = vfmaq_f32(vacc4x0123, va4, vb0123);
      vacc5x0123 = vfmaq_f32(vacc5x0123, va5, vb0123);
      vacc0x4567 = vfmaq_f32(vacc0x4567, va0, vb4567);
      vacc1x4567 = vfmaq_f32(vacc1x4567, va1, vb4567);
      vacc2x4567 = vfmaq_f32(vacc2x4567, va2, vb4567);
      vacc3x4567 = vfmaq_f32(vacc3x4567, va3, vb4567);
      vacc4x4567 = vfmaq_f32(vacc4x4567, va4, vb4567);
      vacc5x4567 = vfmaq_f32(vacc5x4567, va5, vb4567);
    }
    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0123 = vminq_f32(vacc0x0123, vmax);
    vacc1x0123 = vminq_f32(vacc1x0123, vmax);
    vacc2x0123 = vminq_f32(vacc2x0123, vmax);
    vacc3x0123 = vminq_f32(vacc3x0123, vmax);
    vacc4x0123 = vminq_f32(vacc4x0123, vmax);
    vacc5x0123 = vminq_f32(vacc5x0123, vmax);
    vacc0x4567 = vminq_f32(vacc0x4567, vmax);
    vacc1x4567 = vminq_f32(vacc1x4567, vmax);
    vacc2x4567 = vminq_f32(vacc2x4567, vmax);
    vacc3x4567 = vminq_f32(vacc3x4567, vmax);
    vacc4x4567 = vminq_f32(vacc4x4567, vmax);
    vacc5x4567 = vminq_f32(vacc5x4567, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0123 = vmaxq_f32(vacc0x0123, vmin);
    vacc1x0123 = vmaxq_f32(vacc1x0123, vmin);
    vacc2x0123 = vmaxq_f32(vacc2x0123, vmin);
    vacc3x0123 = vmaxq_f32(vacc3x0123, vmin);
    vacc4x0123 = vmaxq_f32(vacc4x0123, vmin);
    vacc5x0123 = vmaxq_f32(vacc5x0123, vmin);
    vacc0x4567 = vmaxq_f32(vacc0x4567, vmin);
    vacc1x4567 = vmaxq_f32(vacc1x4567, vmin);
    vacc2x4567 = vmaxq_f32(vacc2x4567, vmin);
    vacc3x4567 = vmaxq_f32(vacc3x4567, vmin);
    vacc4x4567 = vmaxq_f32(vacc4x4567, vmin);
    vacc5x4567 = vmaxq_f32(vacc5x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c5, vacc5x0123);
      vst1q_f32(c5 + 4, vacc5x4567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      vst1q_f32(c4, vacc4x0123);
      vst1q_f32(c4 + 4, vacc4x4567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      vst1q_f32(c3, vacc3x0123);
      vst1q_f32(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      vst1q_f32(c2, vacc2x0123);
      vst1q_f32(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1q_f32(c1, vacc1x0123);
      vst1q_f32(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1q_f32(c0, vacc0x0123);
      vst1q_f32(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a5 = (const float*) ((uintptr_t) a5 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;

    } else {
      if (nc & 4) {
        vst1q_f32(c5, vacc5x0123); c5 += 4;
        vst1q_f32(c4, vacc4x0123); c4 += 4;
        vst1q_f32(c3, vacc3x0123); c3 += 4;
        vst1q_f32(c2, vacc2x0123); c2 += 4;
        vst1q_f32(c1, vacc1x0123); c1 += 4;
        vst1q_f32(c0, vacc0x0123); c0 += 4;

        vacc5x0123 = vacc5x4567;
        vacc4x0123 = vacc4x4567;
        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;
      }
      float32x2_t vacc5x01 = vget_low_f32(vacc5x0123);
      float32x2_t vacc4x01 = vget_low_f32(vacc4x0123);
      float32x2_t vacc3x01 = vget_low_f32(vacc3x0123);
      float32x2_t vacc2x01 = vget_low_f32(vacc2x0123);
      float32x2_t vacc1x01 = vget_low_f32(vacc1x0123);
      float32x2_t vacc0x01 = vget_low_f32(vacc0x0123);
      if (nc & 2) {
        vst1_f32(c5, vacc5x01); c5 += 2;
        vst1_f32(c4, vacc4x01); c4 += 2;
        vst1_f32(c3, vacc3x01); c3 += 2;
        vst1_f32(c2, vacc2x01); c2 += 2;
        vst1_f32(c1, vacc1x01); c1 += 2;
        vst1_f32(c0, vacc0x01); c0 += 2;

        vacc5x01 = vget_high_f32(vacc5x0123);
        vacc4x01 = vget_high_f32(vacc4x0123);
        vacc3x01 = vget_high_f32(vacc3x0123);
        vacc2x01 = vget_high_f32(vacc2x0123);
        vacc1x01 = vget_high_f32(vacc1x0123);
        vacc0x01 = vget_high_f32(vacc0x0123);
      }
      if (nc & 1) {
        vst1_lane_f32(c5, vacc5x01, 0);
        vst1_lane_f32(c4, vacc4x01, 0);
        vst1_lane_f32(c3, vacc3x01, 0);
        vst1_lane_f32(c2, vacc2x01, 0);
        vst1_lane_f32(c1, vacc1x01, 0);
        vst1_lane_f32(c0, vacc0x01, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  do {
    float32x4_t vacc0x0123 = vld1q_f32(w); w += 4;
    float32x4_t vacc0x4567 = vld1q_f32(w); w += 4;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      for (; k >= 2 * sizeof(float); k -= 2 * sizeof(float)) {
        const float32x2_t va0 = vld1_f32(a0); a0 += 2;

        const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

        vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c0, va0, 0);
        vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c0, va0, 0);
        const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

        vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c1, va0, 1);
        vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c1, va0, 1);

      }
      if XNN_UNLIKELY(k != 0) {
        const float32x4_t va0 = vld1q_dup_f32(a0);

        const float32x4_t vb0123 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567 = vld1q_f32(w); w += 4;

        vacc0x0123 = vfmaq_f32(vacc0x0123, va0, vb0123);
        vacc0x4567 = vfmaq_f32(vacc0x4567, va0, vb4567);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0123 = vminq_f32(vacc0x0123, vmax);
    vacc0x4567 = vminq_f32(vacc0x4567, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0123 = vmaxq_f32(vacc0x0123, vmin);
    vacc0x4567 = vmaxq_f32(vacc0x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vacc0x0123);
      vst1q_f32(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        vst1q_f32(c0, vacc0x0123); c0 += 4;

        vacc0x0123 = vacc0x4567;
      }
      float32x2_t vacc0x01 = vget_low_f32(vacc0x0123);
      if (nc & 2) {
        vst1_f32(c0, vacc0x01); c0 += 2;

        vacc0x01 = vget_high_f32(vacc0x0123);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vacc0x01, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    float32x2_t vacc0x01 = vld1_f32(w); w += 2;
    float32x2_t vacc1x01 = vacc0x01;
    float32x2_t vacc2x01 = vacc0x01;
    float32x2_t vacc3x01 = vacc0x01;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      for (; k >= 2 * sizeof(float); k -= 2 * sizeof(float)) {
        const float32x2_t va0 = vld1_f32(a0); a0 += 2;
        const float32x2_t va1 = vld1_f32(a1); a1 += 2;
        const float32x2_t va2 = vld1_f32(a2); a2 += 2;
        const float32x2_t va3 = vld1_f32(a3); a3 += 2;

        const float32x2_t vb01c0 = vld1_f32(w); w += 2;

        #if XNN_ARCH_ARM64
          vacc0x01 = vfma_lane_f32(vacc0x01, vb01c0, va0, 0);
          vacc1x01 = vfma_lane_f32(vacc1x01, vb01c0, va1, 0);
          vacc2x01 = vfma_lane_f32(vacc2x01, vb01c0, va2, 0);
          vacc3x01 = vfma_lane_f32(vacc3x01, vb01c0, va3, 0);
        #else
          const float32x2_t va0c0 = vdup_lane_f32(va0, 0);
          const float32x2_t va1c0 = vdup_lane_f32(va1, 0);
          const float32x2_t va2c0 = vdup_lane_f32(va2, 0);
          const float32x2_t va3c0 = vdup_lane_f32(va3, 0);
          vacc0x01 = vfma_f32(vacc0x01, va0c0, vb01c0);
          vacc1x01 = vfma_f32(vacc1x01, va1c0, vb01c0);
          vacc2x01 = vfma_f32(vacc2x01, va2c0, vb01c0);
          vacc3x01 = vfma_f32(vacc3x01, va3c0, vb01c0);
        #endif
        const float32x2_t vb01c1 = vld1_f32(w); w += 2;

        #if XNN_ARCH_ARM64
          vacc0x01 = vfma_lane_f32(vacc0x01, vb01c1, va0, 1);
          vacc1x01 = vfma_lane_f32(vacc1x01, vb01c1, va1, 1);
          vacc2x01 = vfma_lane_f32(vacc2x01, vb01c1, va2, 1);
          vacc3x01 = vfma_lane_f32(vacc3x01, vb01c1, va3, 1);
        #else
          const float32x2_t va0c1 = vdup_lane_f32(va0, 1);
          const float32x2_t va1c1 = vdup_lane_f32(va1, 1);
          const float32x2_t va2c1 = vdup_lane_f32(va2, 1);
          const float32x2_t va3c1 = vdup_lane_f32(va3, 1);
          vacc0x01 = vfma_f32(vacc0x01, va0c1, vb01c1);
          vacc1x01 = vfma_f32(vacc1x01, va1c1, vb01c1);
          vacc2x01 = vfma_f32(vacc2x01, va2c1, vb01c1);
          vacc3x01 = vfma_f32(vacc3x01, va3c1, vb01c1);
        #endif
      }
      if XNN_UNLIKELY(k != 0) {
        const float32x2_t va0 = vld1_dup_f32(a0);
        const float32x2_t va1 = vld1_dup_f32(a1);
        const float32x2_t va2 = vld1_dup_f32(a2);
        const float32x2_t va3 = vld1_dup_f32(a3);

        const float32x2_t vb01 = vld1_f32(w); w += 2;

        vacc0x01 = vfma_f32(vacc0x01, va0, vb01);
        vacc1x01 = vfma_f32(vacc1x01, va1, vb01);
        vacc2x01 = vfma_f32(vacc2x01, va2, vb01);
        vacc3x01 = vfma_f32(vacc3x01, va3, vb01);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const float32x2_t vmax = vld1_dup_f32(&params->scalar.max);
    vacc0x01 = vmin_f32(vacc0x01, vmax);
    vacc1x01 = vmin_f32(vacc1x01, vmax);
    vacc2x01 = vmin_f32(vacc2x01, vmax);
    vacc3x01 = vmin_f32(vacc3x01, vmax);

    const float32x2_t vmin = vld1_dup_f32(&params->scalar.min);
    vacc0x01 = vmax_f32(vacc0x01, vmin);
    vacc1x01 = vmax_f32(vacc1x01, vmin);
    vacc2x01 = vmax_f32(vacc2x01, vmin);
    vacc3x01 = vmax_f32(vacc3x01, vmin);

    if XNN_LIKELY(nc >= 2) {
      vst1_f32(c3, vacc3x01);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      vst1_f32(c2, vacc2x01);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1_f32(c1, vacc1x01);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1_f32(c0, vacc0x01);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      assert(nc == 1);
      vst1_lane_f32(c3, vacc3x01, 0);
      vst1_lane_f32(c2, vacc2x01, 0);
      vst1_lane_f32(c1, vacc1x01, 0);
      vst1_lane_f32(c0, vacc0x01, 0);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (6 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    c5 = c4;
  }

  do {
    float32x2_t vacc0x01 = vld1_f32(w); w += 2;
    float32x2_t vacc1x01 = vacc0x01;
    float32x2_t vacc2x01 = vacc0x01;
    float32x2_t vacc3x01 = vacc0x01;
    float32x2_t vacc4x01 = vacc0x01;
    float32x2_t vacc5x01 = vacc0x01;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      const float* restrict a4 = a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const float*) ((uintptr_t) a4 + a_offset);
      }
      const float* restrict a5 = a[5];
      assert(a5 != NULL);
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const float*) ((uintptr_t) a5 + a_offset);
      }
      a += 6;

      size_t k = kc;
      for (; k >= 2 * sizeof(float); k -= 2 * sizeof(float)) {
        const float32x2_t va0 = vld1_f32(a0); a0 += 2;
        const float32x2_t va1 = vld1_f32(a1); a1 += 2;
        const float32x2_t va2 = vld1_f32(a2); a2 += 2;
        const float32x2_t va3 = vld1_f32(a3); a3 += 2;
        const float32x2_t va4 = vld1_f32(a4); a4 += 2;
        const float32x2_t va5 = vld1_f32(a5); a5 += 2;

        const float32x2_t vb01c0 = vld1_f32(w); w += 2;

        #if XNN_ARCH_ARM64
          vacc0x01 = vfma_lane_f32(vacc0x01, vb01c0, va0, 0);
          vacc1x01 = vfma_lane_f32(vacc1x01, vb01c0, va1, 0);
          vacc2x01 = vfma_lane_f32(vacc2x01, vb01c0, va2, 0);
          vacc3x01 = vfma_lane_f32(vacc3x01, vb01c0, va3, 0);
          vacc4x01 = vfma_lane_f32(vacc4x01, vb01c0, va4, 0);
          vacc5x01 = vfma_lane_f32(vacc5x01, vb01c0, va5, 0);
        #else
          const float32x2_t va0c0 = vdup_lane_f32(va0, 0);
          const float32x2_t va1c0 = vdup_lane_f32(va1, 0);
          const float32x2_t va2c0 = vdup_lane_f32(va2, 0);
          const float32x2_t va3c0 = vdup_lane_f32(va3, 0);
          const float32x2_t va4c0 = vdup_lane_f32(va4, 0);
          const float32x2_t va5c0 = vdup_lane_f32(va5, 0);
          vacc0x01 = vfma_f32(vacc0x01, va0c0, vb01c0);
          vacc1x01 = vfma_f32(vacc1x01, va1c0, vb01c0);
          vacc2x01 = vfma_f32(vacc2x01, va2c0, vb01c0);
          vacc3x01 = vfma_f32(vacc3x01, va3c0, vb01c0);
          vacc4x01 = vfma_f32(vacc4x01, va4c0, vb01c0);
          vacc5x01 = vfma_f32(vacc5x01, va5c0, vb01c0);
        #endif
        const float32x2_t vb01c1 = vld1_f32(w); w += 2;

        #if XNN_ARCH_ARM64
          vacc0x01 = vfma_lane_f32(vacc0x01, vb01c1, va0, 1);
          vacc1x01 = vfma_lane_f32(vacc1x01, vb01c1, va1, 1);
          vacc2x01 = vfma_lane_f32(vacc2x01, vb01c1, va2, 1);
          vacc3x01 = vfma_lane_f32(vacc3x01, vb01c1, va3, 1);
          vacc4x01 = vfma_lane_f32(vacc4x01, vb01c1, va4, 1);
          vacc5x01 = vfma_lane_f32(vacc5x01, vb01c1, va5, 1);
        #else
          const float32x2_t va0c1 = vdup_lane_f32(va0, 1);
          const float32x2_t va1c1 = vdup_lane_f32(va1, 1);
          const float32x2_t va2c1 = vdup_lane_f32(va2, 1);
          const float32x2_t va3c1 = vdup_lane_f32(va3, 1);
          const float32x2_t va4c1 = vdup_lane_f32(va4, 1);
          const float32x2_t va5c1 = vdup_lane_f32(va5, 1);
          vacc0x01 = vfma_f32(vacc0x01, va0c1, vb01c1);
          vacc1x01 = vfma_f32(vacc1x01, va1c1, vb01c1);
          vacc2x01 = vfma_f32(vacc2x01, va2c1, vb01c1);
          vacc3x01 = vfma_f32(vacc3x01, va3c1, vb01c1);
          vacc4x01 = vfma_f32(vacc4x01, va4c1, vb01c1);
          vacc5x01 = vfma_f32(vacc5x01, va5c1, vb01c1);
        #endif
      }
      if XNN_UNLIKELY(k != 0) {
        const float32x2_t va0 = vld1_dup_f32(a0);
        const float32x2_t va1 = vld1_dup_f32(a1);
        const float32x2_t va2 = vld1_dup_f32(a2);
        const float32x2_t va3 = vld1_dup_f32(a3);
        const float32x2_t va4 = vld1_dup_f32(a4);
        const float32x2_t va5 = vld1_dup_f32(a5);

        const float32x2_t vb01 = vld1_f32(w); w += 2;

        vacc0x01 = vfma_f32(vacc0x01, va0, vb01);
        vacc1x01 = vfma_f32(vacc1x01, va1, vb01);
        vacc2x01 = vfma_f32(vacc2x01, va2, vb01);
        vacc3x01 = vfma_f32(vacc3x01, va3, vb01);
        vacc4x01 = vfma_f32(vacc4x01, va4, vb01);
        vacc5x01 = vfma_f32(vacc5x01, va5, vb01);
      }
      p -= 6 * sizeof(void*);
    } while (p != 0);

    const float32x2_t vmax = vld1_dup_f32(&params->scalar.max);
    vacc0x01 = vmin_f32(vacc0x01, vmax);
    vacc1x01 = vmin_f32(vacc1x01, vmax);
    vacc2x01 = vmin_f32(vacc2x01, vmax);
    vacc3x01 = vmin_f32(vacc3x01, vmax);
    vacc4x01 = vmin_f32(vacc4x01, vmax);
    vacc5x01 = vmin_f32(vacc5x01, vmax);

    const float32x2_t vmin = vld1_dup_f32(&params->scalar.min);
    vacc0x01 = vmax_f32(vacc0x01, vmin);
    vacc1x01 = vmax_f32(vacc1x01, vmin);
    vacc2x01 = vmax_f32(vacc2x01, vmin);
    vacc3x01 = vmax_f32(vacc3x01, vmin);
    vacc4x01 = vmax_f32(vacc4x01, vmin);
    vacc5x01 = vmax_f32(vacc5x01, vmin);

    if XNN_LIKELY(nc >= 2) {
      vst1_f32(c5, vacc5x01);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      vst1_f32(c4, vacc4x01);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      vst1_f32(c3, vacc3x01);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      vst1_f32(c2, vacc2x01);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1_f32(c1, vacc1x01);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1_f32(c0, vacc0x01);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      assert(nc == 1);
      vst1_lane_f32(c5, vacc5x01, 0);
      vst1_lane_f32(c4, vacc4x01, 0);
      vst1_lane_f32(c3, vacc3x01, 0);
      vst1_lane_f32(c2, vacc2x01, 0);
      vst1_lane_f32(c1, vacc1x01, 0);
      vst1_lane_f32(c0, vacc0x01, 0);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (6 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    c5 = c4;
  }

  do {
    float32x4_t vacc0x0123 = vld1q_f32(w); w += 4;
    float32x4_t vacc0x4567 = vld1q_f32(w); w += 4;
    float32x4_t vacc1x0123 = vacc0x0123;
    float32x4_t vacc1x4567 = vacc0x4567;
    float32x4_t vacc2x0123 = vacc0x0123;
    float32x4_t vacc2x4567 = vacc0x4567;
    float32x4_t vacc3x0123 = vacc0x0123;
    float32x4_t vacc3x4567 = vacc0x4567;
    float32x4_t vacc4x0123 = vacc0x0123;
    float32x4_t vacc4x4567 = vacc0x4567;
    float32x4_t vacc5x0123 = vacc0x0123;
    float32x4_t vacc5x4567 = vacc0x4567;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      const float* restrict a4 = a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const float*) ((uintptr_t) a4 + a_offset);
      }
      const float* restrict a5 = a[5];
      assert(a5 != NULL);
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const float*) ((uintptr_t) a5 + a_offset);
      }
      a += 6;

      size_t k = kc;
      for (; k >= 2 * sizeof(float); k -= 2 * sizeof(float)) {
        const float32x2_t va0 = vld1_f32(a0); a0 += 2;
        const float32x2_t va1 = vld1_f32(a1); a1 += 2;
        const float32x2_t va2 = vld1_f32(a2); a2 += 2;
        const float32x2_t va3 = vld1_f32(a3); a3 += 2;
        const float32x2_t va4 = vld1_f32(a4); a4 += 2;
        const float32x2_t va5 = vld1_f32(a5); a5 += 2;

        const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

        vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c0, va0, 0);
        vacc1x0123 = vfmaq_lane_f32(vacc1x0123, vb0123c0, va1, 0);
        vacc2x0123 = vfmaq_lane_f32(vacc2x0123, vb0123c0, va2, 0);
        vacc3x0123 = vfmaq_lane_f32(vacc3x0123, vb0123c0, va3, 0);
        vacc4x0123 = vfmaq_lane_f32(vacc4x0123, vb0123c0, va4, 0);
        vacc5x0123 = vfmaq_lane_f32(vacc5x0123, vb0123c0, va5, 0);
        vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c0, va0, 0);
        vacc1x4567 = vfmaq_lane_f32(vacc1x4567, vb4567c0, va1, 0);
        vacc2x4567 = vfmaq_lane_f32(vacc2x4567, vb4567c0, va2, 0);
        vacc3x4567 = vfmaq_lane_f32(vacc3x4567, vb4567c0, va3, 0);
        vacc4x4567 = vfmaq_lane_f32(vacc4x4567, vb4567c0, va4, 0);
        vacc5x4567 = vfmaq_lane_f32(vacc5x4567, vb4567c0, va5, 0);
        const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

        vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123c1, va0, 1);
        vacc1x0123 = vfmaq_lane_f32(vacc1x0123, vb0123c1, va1, 1);
        vacc2x0123 = vfmaq_lane_f32(vacc2x0123, vb0123c1, va2, 1);
        vacc3x0123 = vfmaq_lane_f32(vacc3x0123, vb0123c1, va3, 1);
        vacc4x0123 = vfmaq_lane_f32(vacc4x0123, vb0123c1, va4, 1);
        vacc5x0123 = vfmaq_lane_f32(vacc5x0123, vb0123c1, va5, 1);
        vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567c1, va0, 1);
        vacc1x4567 = vfmaq_lane_f32(vacc1x4567, vb4567c1, va1, 1);
        vacc2x4567 = vfmaq_lane_f32(vacc2x4567, vb4567c1, va2, 1);
        vacc3x4567 = vfmaq_lane_f32(vacc3x4567, vb4567c1, va3, 1);
        vacc4x4567 = vfmaq_lane_f32(vacc4x4567, vb4567c1, va4, 1);
        vacc5x4567 = vfmaq_lane_f32(vacc5x4567, vb4567c1, va5, 1);

      }
      if XNN_UNLIKELY(k != 0) {
        const float32x4_t va0 = vld1q_dup_f32(a0);
        const float32x4_t va1 = vld1q_dup_f32(a1);
        const float32x4_t va2 = vld1q_dup_f32(a2);
        const float32x4_t va3 = vld1q_dup_f32(a3);
        const float32x4_t va4 = vld1q_dup_f32(a4);
        const float32x4_t va5 = vld1q_dup_f32(a5);

        const float32x4_t vb0123 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567 = vld1q_f32(w); w += 4;

        vacc0x0123 = vfmaq_f32(vacc0x0123, va0, vb0123);
        vacc1x0123 = vfmaq_f32(vacc1x0123, va1, vb0123);
        vacc2x0123 = vfmaq_f32(vacc2x0123, va2, vb0123);
        vacc3x0123 = vfmaq_f32(vacc3x0123, va3, vb0123);
        vacc4x0123 = vfmaq_f32(vacc4x0123, va4, vb0123);
        vacc5x0123 = vfmaq_f32(vacc5x0123, va5, vb0123);
        vacc0x4567 = vfmaq_f32(vacc0x4567, va0, vb4567);
        vacc1x4567 = vfmaq_f32(vacc1x4567, va1, vb4567);
        vacc2x4567 = vfmaq_f32(vacc2x4567, va2, vb4567);
        vacc3x4567 = vfmaq_f32(vacc3x4567, va3, vb4567);
        vacc4x4567 = vfmaq_f32(vacc4x4567, va4, vb4567);
        vacc5x4567 = vfmaq_f32(vacc5x4567, va5, vb4567);
      }
      p -= 6 * sizeof(void*);
    } while (p != 0);

    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0123 = vminq_f32(vacc0x0123, vmax);
    vacc1x0123 = vminq_f32(vacc1x0123, vmax);
    vacc2x0123 = vminq_f32(vacc2x0123, vmax);
    vacc3x0123 = vminq_f32(vacc3x0123, vmax);
    vacc4x0123 = vminq_f32(vacc4x0123, vmax);
    vacc5x0123 = vminq_f32(vacc5x0123, vmax);
    vacc0x4567 = vminq_f32(vacc0x4567, vmax);
    vacc1x4567 = vminq_f32(vacc1x4567, vmax);
    vacc2x4567 = vminq_f32(vacc2x4567, vmax);
    vacc3x4567 = vminq_f32(vacc3x4567, vmax);
    vacc4x4567 = vminq_f32(vacc4x4567, vmax);
    vacc5x4567 = vminq_f32(vacc5x4567, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0123 = vmaxq_f32(vacc0x0123, vmin);
    vacc1x0123 = vmaxq_f32(vacc1x0123, vmin);
    vacc2x0123 = vmaxq_f32(vacc2x0123, vmin);
    vacc3x0123 = vmaxq_f32(vacc3x0123, vmin);
    vacc4x0123 = vmaxq_f32(vacc4x0123, vmin);
    vacc5x0123 = vmaxq_f32(vacc5x0123, vmin);
    vacc0x4567 = vmaxq_f32(vacc0x4567, vmin);
    vacc1x4567 = vmaxq_f32(vacc1x4567, vmin);
    vacc2x4567 = vmaxq_f32(vacc2x4567, vmin);
    vacc3x4567 = vmaxq_f32(vacc3x4567, vmin);
    vacc4x4567 = vmaxq_f32(vacc4x4567, vmin);
    vacc5x4567 = vmaxq_f32(vacc5x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c5, vacc5x0123);
      vst1q_f32(c5 + 4, vacc5x4567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      vst1q_f32(c4, vacc4x0123);
      vst1q_f32(c4 + 4, vacc4x4567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      vst1q_f32(c3, vacc3x0123);
      vst1q_f32(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      vst1q_f32(c2, vacc2x0123);
      vst1q_f32(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1q_f32(c1, vacc1x0123);
      vst1q_f32(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1q_f32(c0, vacc0x0123);
      vst1q_f32(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        vst1q_f32(c5, vacc5x0123); c5 += 4;
        vst1q_f32(c4, vacc4x0123); c4 += 4;
        vst1q_f32(c3, vacc3x0123); c3 += 4;
        vst1q_f32(c2, vacc2x0123); c2 += 4;
        vst1q_f32(c1, vacc1x0123); c1 += 4;
        vst1q_f32(c0, vacc0x0123); c0 += 4;

        vacc5x0123 = vacc5x4567;
        vacc4x0123 = vacc4x4567;
        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;
      }
      float32x2_t vacc5x01 = vget_low_f32(vacc5x0123);
      float32x2_t vacc4x01 = vget_low_f32(vacc4x0123);
      float32x2_t vacc3x01 = vget_low_f32(vacc3x0123);
      float32x2_t vacc2x01 = vget_low_f32(vacc2x0123);
      float32x2_t vacc1x01 = vget_low_f32(vacc1x0123);
      float32x2_t vacc0x01 = vget_low_f32(vacc0x0123);
      if (nc & 2) {
        vst1_f32(c5, vacc5x01); c5 += 2;
        vst1_f32(c4, vacc4x01); c4 += 2;
        vst1_f32(c3, vacc3x01); c3 += 2;
        vst1_f32(c2, vacc2x01); c2 += 2;
        vst1_f32(c1, vacc1x01); c1 += 2;
        vst1_f32(c0, vacc0x01); c0 += 2;

        vacc5x01 = vget_high_f32(vacc5x0123);
        vacc4x01 = vget_high_f32(vacc4x0123);
        vacc3x01 = vget_high_f32(vacc3x0123);
        vacc2x01 = vget_high_f32(vacc2x0123);
        vacc1x01 = vget_high_f32(vacc1x0123);
        vacc0x01 = vget_high_f32(vacc0x0123);
      }
      if (nc & 1) {
        vst1_lane_f32(c5, vacc5x01, 0);
        vst1_lane_f32(c4, vacc4x01, 0);
        vst1_lane_f32(c3, vacc3x01, 0);
        vst1_lane_f32(c2, vacc2x01, 0);
        vst1_lane_f32(c1, vacc1x01, 0);
        vst1_lane_f32(c0, vacc0x01, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_spmm_minmax_ukernel_32x2__aarch64_neonfma(
    size_t mc,
    size_t nc,
    const float* input,
    const float* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    float* output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mc != 0);
  assert(mc % sizeof(float) == 0);
  assert(nc != 0);

  #if XNN_ARCH_ARM64
    const float32x4x2_t vminmax = vld2q_dup_f32(&params->scalar.min);
    const float32x4_t vmin = vminmax.val[0];
    const float32x4_t vmax = vminmax.val[1];
  #else
    const float32x2x2_t vminmax = vld2_dup_f32(&params->scalar.min);
    const float32x4_t vmin = vcombine_f32(vminmax.val[0],vminmax.val[0]);
    const float32x4_t vmax = vcombine_f32(vminmax.val[1],vminmax.val[1]);
  #endif

  size_t output_decrement = output_stride * nc - 32 * sizeof(float);
  while XNN_LIKELY(mc >= 32 * sizeof(float)) {
    const float* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    while (n >= 2) {
      uint32_t nnz = *nnzmap++;
      float32x4_t vacc0123n0 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567n0 = vacc0123n0;
      float32x4_t vacc89ABn0 = vacc0123n0;
      float32x4_t vaccCDEFn0 = vacc0123n0;
      float32x4_t vaccGHIJn0 = vacc0123n0;
      float32x4_t vaccKLMNn0 = vacc0123n0;
      float32x4_t vaccOPQRn0 = vacc0123n0;
      float32x4_t vaccSTUVn0 = vacc0123n0;
      float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567n1 = vacc0123n1;
      float32x4_t vacc89ABn1 = vacc0123n1;
      float32x4_t vaccCDEFn1 = vacc0123n1;
      float32x4_t vaccGHIJn1 = vacc0123n1;
      float32x4_t vaccKLMNn1 = vacc0123n1;
      float32x4_t vaccOPQRn1 = vacc0123n1;
      float32x4_t vaccSTUVn1 = vacc0123n1;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const float32x4_t vi0123 = vld1q_f32(input);
          const float32x4_t vi4567 = vld1q_f32(input + 4);
          const float32x4_t vi89AB = vld1q_f32(input + 8);
          const float32x4_t viCDEF = vld1q_f32(input + 12);
          const float32x4_t viGHIJ = vld1q_f32(input + 16);
          const float32x4_t viKLMN = vld1q_f32(input + 20);
          const float32x4_t viOPQR = vld1q_f32(input + 24);
          const float32x4_t viSTUV = vld1q_f32(input + 28);
          input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
          xnn_prefetch_to_l1(input + 16);
          xnn_prefetch_to_l1(input + 32);
          const float32x2_t vw = vld1_f32(w); w += 2;
          xnn_prefetch_to_l1(w + 32);
          vacc0123n0 = vfmaq_lane_f32(vacc0123n0, vi0123, vw, 0);
          vacc4567n0 = vfmaq_lane_f32(vacc4567n0, vi4567, vw, 0);
          vacc89ABn0 = vfmaq_lane_f32(vacc89ABn0, vi89AB, vw, 0);
          vaccCDEFn0 = vfmaq_lane_f32(vaccCDEFn0, viCDEF, vw, 0);
          vaccGHIJn0 = vfmaq_lane_f32(vaccGHIJn0, viGHIJ, vw, 0);
          vaccKLMNn0 = vfmaq_lane_f32(vaccKLMNn0, viKLMN, vw, 0);
          vaccOPQRn0 = vfmaq_lane_f32(vaccOPQRn0, viOPQR, vw, 0);
          vaccSTUVn0 = vfmaq_lane_f32(vaccSTUVn0, viSTUV, vw, 0);
          vacc0123n1 = vfmaq_lane_f32(vacc0123n1, vi0123, vw, 1);
          vacc4567n1 = vfmaq_lane_f32(vacc4567n1, vi4567, vw, 1);
          vacc89ABn1 = vfmaq_lane_f32(vacc89ABn1, vi89AB, vw, 1);
          vaccCDEFn1 = vfmaq_lane_f32(vaccCDEFn1, viCDEF, vw, 1);
          vaccGHIJn1 = vfmaq_lane_f32(vaccGHIJn1, viGHIJ, vw, 1);
          vaccKLMNn1 = vfmaq_lane_f32(vaccKLMNn1, viKLMN, vw, 1);
          vaccOPQRn1 = vfmaq_lane_f32(vaccOPQRn1, viOPQR, vw, 1);
          vaccSTUVn1 = vfmaq_lane_f32(vaccSTUVn1, viSTUV, vw, 1);
        } while (--nnz != 0);
      }
      float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
      float32x4_t vout4567n0 = vminq_f32(vacc4567n0, vmax);
      float32x4_t vout89ABn0 = vminq_f32(vacc89ABn0, vmax);
      float32x4_t voutCDEFn0 = vminq_f32(vaccCDEFn0, vmax);
      float32x4_t voutGHIJn0 = vminq_f32(vaccGHIJn0, vmax);
      float32x4_t voutKLMNn0 = vminq_f32(vaccKLMNn0, vmax);
      float32x4_t voutOPQRn0 = vminq_f32(vaccOPQRn0, vmax);
      float32x4_t voutSTUVn0 = vminq_f32(vaccSTUVn0, vmax);
      float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);
      float32x4_t vout4567n1 = vminq_f32(vacc4567n1, vmax);
      float32x4_t vout89ABn1 = vminq_f32(vacc89ABn1, vmax);
      float32x4_t voutCDEFn1 = vminq_f32(vaccCDEFn1, vmax);
      float32x4_t voutGHIJn1 = vminq_f32(vaccGHIJn1, vmax);
      float32x4_t voutKLMNn1 = vminq_f32(vaccKLMNn1, vmax);
      float32x4_t voutOPQRn1 = vminq_f32(vaccOPQRn1, vmax);
      float32x4_t voutSTUVn1 = vminq_f32(vaccSTUVn1, vmax);

      vout0123n0 = vmaxq_f32(vout0123n0, vmin);
      vout4567n0 = vmaxq_f32(vout4567n0, vmin);
      vout89ABn0 = vmaxq_f32(vout89ABn0, vmin);
      voutCDEFn0 = vmaxq_f32(voutCDEFn0, vmin);
      voutGHIJn0 = vmaxq_f32(voutGHIJn0, vmin);
      voutKLMNn0 = vmaxq_f32(voutKLMNn0, vmin);
      voutOPQRn0 = vmaxq_f32(voutOPQRn0, vmin);
      voutSTUVn0 = vmaxq_f32(voutSTUVn0, vmin);
      vout0123n1 = vmaxq_f32(vout0123n1, vmin);
      vout4567n1 = vmaxq_f32(vout4567n1, vmin);
      vout89ABn1 = vmaxq_f32(vout89ABn1, vmin);
      voutCDEFn1 = vmaxq_f32(voutCDEFn1, vmin);
      voutGHIJn1 = vmaxq_f32(voutGHIJn1, vmin);
      voutKLMNn1 = vmaxq_f32(voutKLMNn1, vmin);
      voutOPQRn1 = vmaxq_f32(voutOPQRn1, vmin);
      voutSTUVn1 = vmaxq_f32(voutSTUVn1, vmin);

      vst1q_f32(output + 0, vout0123n0);
      vst1q_f32(output + 4, vout4567n0);
      vst1q_f32(output + 8, vout89ABn0);
      vst1q_f32(output + 12, voutCDEFn0);
      vst1q_f32(output + 16, voutGHIJn0);
      vst1q_f32(output + 20, voutKLMNn0);
      vst1q_f32(output + 24, voutOPQRn0);
      vst1q_f32(output + 28, voutSTUVn0);
      output = (float*) ((uintptr_t) output + output_stride);
      vst1q_f32(output + 0, vout0123n1);
      vst1q_f32(output + 4, vout4567n1);
      vst1q_f32(output + 8, vout89ABn1);
      vst1q_f32(output + 12, voutCDEFn1);
      vst1q_f32(output + 16, voutGHIJn1);
      vst1q_f32(output + 20, voutKLMNn1);
      vst1q_f32(output + 24, voutOPQRn1);
      vst1q_f32(output + 28, voutSTUVn1);
      output = (float*) ((uintptr_t) output + output_stride);
      n -= 2;
    }

    // clean up loop, fall back to nr=1
    if XNN_UNLIKELY(n != 0) {
      do {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567 = vacc0123;
        float32x4_t vacc89AB = vacc0123;
        float32x4_t vaccCDEF = vacc0123;
        float32x4_t vaccGHIJ = vacc0123;
        float32x4_t vaccKLMN = vacc0123;
        float32x4_t vaccOPQR = vacc0123;
        float32x4_t vaccSTUV = vacc0123;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            const float32x4_t vi89AB = vld1q_f32(input + 8);
            const float32x4_t viCDEF = vld1q_f32(input + 12);
            const float32x4_t viGHIJ = vld1q_f32(input + 16);
            const float32x4_t viKLMN = vld1q_f32(input + 20);
            const float32x4_t viOPQR = vld1q_f32(input + 24);
            const float32x4_t viSTUV = vld1q_f32(input + 28);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            xnn_prefetch_to_l1(input + 16);
            xnn_prefetch_to_l1(input + 32);
            const float32x4_t vw = vld1q_dup_f32(w); w += 1;
            xnn_prefetch_to_l1(w + 32);
            vacc0123 = vfmaq_f32(vacc0123, vi0123, vw);
            vacc4567 = vfmaq_f32(vacc4567, vi4567, vw);
            vacc89AB = vfmaq_f32(vacc89AB, vi89AB, vw);
            vaccCDEF = vfmaq_f32(vaccCDEF, viCDEF, vw);
            vaccGHIJ = vfmaq_f32(vaccGHIJ, viGHIJ, vw);
            vaccKLMN = vfmaq_f32(vaccKLMN, viKLMN, vw);
            vaccOPQR = vfmaq_f32(vaccOPQR, viOPQR, vw);
            vaccSTUV = vfmaq_f32(vaccSTUV, viSTUV, vw);
          } while (--nnz != 0);
        }
        float32x4_t vout0123 = vminq_f32(vacc0123, vmax);
        float32x4_t vout4567 = vminq_f32(vacc4567, vmax);
        float32x4_t vout89AB = vminq_f32(vacc89AB, vmax);
        float32x4_t voutCDEF = vminq_f32(vaccCDEF, vmax);
        float32x4_t voutGHIJ = vminq_f32(vaccGHIJ, vmax);
        float32x4_t voutKLMN = vminq_f32(vaccKLMN, vmax);
        float32x4_t voutOPQR = vminq_f32(vaccOPQR, vmax);
        float32x4_t voutSTUV = vminq_f32(vaccSTUV, vmax);

        vout0123 = vmaxq_f32(vout0123, vmin);
        vout4567 = vmaxq_f32(vout4567, vmin);
        vout89AB = vmaxq_f32(vout89AB, vmin);
        voutCDEF = vmaxq_f32(voutCDEF, vmin);
        voutGHIJ = vmaxq_f32(voutGHIJ, vmin);
        voutKLMN = vmaxq_f32(voutKLMN, vmin);
        voutOPQR = vmaxq_f32(voutOPQR, vmin);
        voutSTUV = vmaxq_f32(voutSTUV, vmin);

        vst1q_f32(output + 0, vout0123);
        vst1q_f32(output + 4, vout4567);
        vst1q_f32(output + 8, vout89AB);
        vst1q_f32(output + 12, voutCDEF);
        vst1q_f32(output + 16, voutGHIJ);
        vst1q_f32(output + 20, voutKLMN);
        vst1q_f32(output + 24, voutOPQR);
        vst1q_f32(output + 28, voutSTUV);
        output = (float*) ((uintptr_t) output + output_stride);
        n -= 1;
      } while (n != 0);
    }
    output = (float*) ((uintptr_t) output - output_decrement);
    input += 32;
    mc -= 32 * sizeof(float);
  }
  if XNN_UNLIKELY(mc != 0) {
    output_decrement += 16 * sizeof(float);
    if (mc & (16 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 2) {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123n0 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n0 = vacc0123n0;
        float32x4_t vacc89ABn0 = vacc0123n0;
        float32x4_t vaccCDEFn0 = vacc0123n0;
        float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n1 = vacc0123n1;
        float32x4_t vacc89ABn1 = vacc0123n1;
        float32x4_t vaccCDEFn1 = vacc0123n1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            const float32x4_t vi89AB = vld1q_f32(input + 8);
            const float32x4_t viCDEF = vld1q_f32(input + 12);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw = vld1_f32(w); w += 2;

            vacc0123n0 = vfmaq_lane_f32(vacc0123n0, vi0123, vw, 0);
            vacc4567n0 = vfmaq_lane_f32(vacc4567n0, vi4567, vw, 0);
            vacc89ABn0 = vfmaq_lane_f32(vacc89ABn0, vi89AB, vw, 0);
            vaccCDEFn0 = vfmaq_lane_f32(vaccCDEFn0, viCDEF, vw, 0);
            vacc0123n1 = vfmaq_lane_f32(vacc0123n1, vi0123, vw, 1);
            vacc4567n1 = vfmaq_lane_f32(vacc4567n1, vi4567, vw, 1);
            vacc89ABn1 = vfmaq_lane_f32(vacc89ABn1, vi89AB, vw, 1);
            vaccCDEFn1 = vfmaq_lane_f32(vaccCDEFn1, viCDEF, vw, 1);
          } while (--nnz != 0);
        }
        float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
        float32x4_t vout4567n0 = vminq_f32(vacc4567n0, vmax);
        float32x4_t vout89ABn0 = vminq_f32(vacc89ABn0, vmax);
        float32x4_t voutCDEFn0 = vminq_f32(vaccCDEFn0, vmax);
        float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);
        float32x4_t vout4567n1 = vminq_f32(vacc4567n1, vmax);
        float32x4_t vout89ABn1 = vminq_f32(vacc89ABn1, vmax);
        float32x4_t voutCDEFn1 = vminq_f32(vaccCDEFn1, vmax);

        vout0123n0 = vmaxq_f32(vout0123n0, vmin);
        vout4567n0 = vmaxq_f32(vout4567n0, vmin);
        vout89ABn0 = vmaxq_f32(vout89ABn0, vmin);
        voutCDEFn0 = vmaxq_f32(voutCDEFn0, vmin);
        vout0123n1 = vmaxq_f32(vout0123n1, vmin);
        vout4567n1 = vmaxq_f32(vout4567n1, vmin);
        vout89ABn1 = vmaxq_f32(vout89ABn1, vmin);
        voutCDEFn1 = vmaxq_f32(voutCDEFn1, vmin);

        vst1q_f32(output + 0, vout0123n0);
        vst1q_f32(output + 4, vout4567n0);
        vst1q_f32(output + 8, vout89ABn0);
        vst1q_f32(output + 12, voutCDEFn0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n1);
        vst1q_f32(output + 4, vout4567n1);
        vst1q_f32(output + 8, vout89ABn1);
        vst1q_f32(output + 12, voutCDEFn1);
        output = (float*) ((uintptr_t) output + output_stride);
        n -= 2;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
          float32x4_t vacc4567 = vacc0123;
          float32x4_t vacc89AB = vacc0123;
          float32x4_t vaccCDEF = vacc0123;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x4_t vi0123 = vld1q_f32(input);
              const float32x4_t vi4567 = vld1q_f32(input + 4);
              const float32x4_t vi89AB = vld1q_f32(input + 8);
              const float32x4_t viCDEF = vld1q_f32(input + 12);
              input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
              const float32x4_t vw = vld1q_dup_f32(w); w += 1;
              vacc0123 = vfmaq_f32(vacc0123, vi0123, vw);
              vacc4567 = vfmaq_f32(vacc4567, vi4567, vw);
              vacc89AB = vfmaq_f32(vacc89AB, vi89AB, vw);
              vaccCDEF = vfmaq_f32(vaccCDEF, viCDEF, vw);
            } while (--nnz != 0);
          }
          float32x4_t vout0123 = vminq_f32(vacc0123, vmax);
          float32x4_t vout4567 = vminq_f32(vacc4567, vmax);
          float32x4_t vout89AB = vminq_f32(vacc89AB, vmax);
          float32x4_t voutCDEF = vminq_f32(vaccCDEF, vmax);

          vout0123 = vmaxq_f32(vout0123, vmin);
          vout4567 = vmaxq_f32(vout4567, vmin);
          vout89AB = vmaxq_f32(vout89AB, vmin);
          voutCDEF = vmaxq_f32(voutCDEF, vmin);

          vst1q_f32(output + 0, vout0123);
          vst1q_f32(output + 4, vout4567);
          vst1q_f32(output + 8, vout89AB);
          vst1q_f32(output + 12, voutCDEF);
          output = (float*) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 16;
    }
    output_decrement += 8 * sizeof(float);
    if (mc & (8 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 2) {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123n0 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n0 = vacc0123n0;
        float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n1 = vacc0123n1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw = vld1_f32(w); w += 2;

            vacc0123n0 = vfmaq_lane_f32(vacc0123n0, vi0123, vw, 0);
            vacc4567n0 = vfmaq_lane_f32(vacc4567n0, vi4567, vw, 0);
            vacc0123n1 = vfmaq_lane_f32(vacc0123n1, vi0123, vw, 1);
            vacc4567n1 = vfmaq_lane_f32(vacc4567n1, vi4567, vw, 1);
          } while (--nnz != 0);
        }
        float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
        float32x4_t vout4567n0 = vminq_f32(vacc4567n0, vmax);
        float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);
        float32x4_t vout4567n1 = vminq_f32(vacc4567n1, vmax);

        vout0123n0 = vmaxq_f32(vout0123n0, vmin);
        vout4567n0 = vmaxq_f32(vout4567n0, vmin);
        vout0123n1 = vmaxq_f32(vout0123n1, vmin);
        vout4567n1 = vmaxq_f32(vout4567n1, vmin);

        vst1q_f32(output + 0, vout0123n0);
        vst1q_f32(output + 4, vout4567n0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n1);
        vst1q_f32(output + 4, vout4567n1);
        output = (float*) ((uintptr_t) output + output_stride);
        n -= 2;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
          float32x4_t vacc4567 = vacc0123;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x4_t vi0123 = vld1q_f32(input);
              const float32x4_t vi4567 = vld1q_f32(input + 4);
              input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
              const float32x4_t vw = vld1q_dup_f32(w); w += 1;
              vacc0123 = vfmaq_f32(vacc0123, vi0123, vw);
              vacc4567 = vfmaq_f32(vacc4567, vi4567, vw);
            } while (--nnz != 0);
          }
          float32x4_t vout0123 = vminq_f32(vacc0123, vmax);
          float32x4_t vout4567 = vminq_f32(vacc4567, vmax);

          vout0123 = vmaxq_f32(vout0123, vmin);
          vout4567 = vmaxq_f32(vout4567, vmin);

          vst1q_f32(output + 0, vout0123);
          vst1q_f32(output + 4, vout4567);
          output = (float*) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 8;
    }
    output_decrement += 4 * sizeof(float);
    if (mc & (4 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 2) {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123n0 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw = vld1_f32(w); w += 2;

            vacc0123n0 = vfmaq_lane_f32(vacc0123n0, vi0123, vw, 0);
            vacc0123n1 = vfmaq_lane_f32(vacc0123n1, vi0123, vw, 1);
          } while (--nnz != 0);
        }
        float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
        float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);

        vout0123n0 = vmaxq_f32(vout0123n0, vmin);
        vout0123n1 = vmaxq_f32(vout0123n1, vmin);

        vst1q_f32(output + 0, vout0123n0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n1);
        output = (float*) ((uintptr_t) output + output_stride);
        n -= 2;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x4_t vi0123 = vld1q_f32(input);
              input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
              const float32x4_t vw = vld1q_dup_f32(w); w += 1;
              vacc0123 = vfmaq_f32(vacc0123, vi0123, vw);
            } while (--nnz != 0);
          }
          float32x4_t vout0123 = vminq_f32(vacc0123, vmax);

          vout0123 = vmaxq_f32(vout0123, vmin);

          vst1q_f32(output + 0, vout0123);
          output = (float*) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 4;
    }
    output_decrement += 2 * sizeof(float);
    if (mc & (2 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 2) {
        uint32_t nnz = *nnzmap++;
        float32x2_t vacc01n0 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc01n1 = vld1_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t vi01 = vld1_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw = vld1_f32(w); w += 2;

            vacc01n0 = vfma_lane_f32(vacc01n0, vi01, vw, 0);
            vacc01n1 = vfma_lane_f32(vacc01n1, vi01, vw, 1);
          } while (--nnz != 0);
        }
        float32x2_t vout01n0 = vmin_f32(vacc01n0, vget_low_f32(vmax));
        float32x2_t vout01n1 = vmin_f32(vacc01n1, vget_low_f32(vmax));

        vout01n0 = vmax_f32(vout01n0, vget_low_f32(vmin));
        vout01n1 = vmax_f32(vout01n1, vget_low_f32(vmin));

        vst1_f32(output + 0, vout01n0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1_f32(output + 0, vout01n1);
        output = (float*) ((uintptr_t) output + output_stride);
        n -= 2;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x2_t vacc01 = vld1_dup_f32(w); w += 1;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x2_t vi01 = vld1_f32(input);
              input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
              const float32x2_t vw = vld1_dup_f32(w); w += 1;
              vacc01 = vfma_f32(vacc01, vi01, vw);
            } while (--nnz != 0);
          }
          float32x2_t vout01 = vmin_f32(vacc01, vget_low_f32(vmax));
          vout01 = vmax_f32(vout01, vget_low_f32(vmin));

          vst1_f32(output, vout01);
          output = (float*) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 2;
    }
    output_decrement += 1 * sizeof(float);
    if (mc & (1 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 2) {
        uint32_t nnz = *nnzmap++;
        float32x2_t vacc0n0 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc0n1 = vld1_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t vi0 = vld1_dup_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const float32x2_t vw = vld1_f32(w); w += 2;

            vacc0n0 = vfma_lane_f32(vacc0n0, vi0, vw, 0);
            vacc0n1 = vfma_lane_f32(vacc0n1, vi0, vw, 1);
          } while (--nnz != 0);
        }
        float32x2_t vout0n0 = vmin_f32(vacc0n0, vget_low_f32(vmax));
        float32x2_t vout0n1 = vmin_f32(vacc0n1, vget_low_f32(vmax));

        vout0n0 = vmax_f32(vout0n0, vget_low_f32(vmin));
        vout0n1 = vmax_f32(vout0n1, vget_low_f32(vmin));

        vst1_lane_f32(output + 0, vout0n0, 0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1_lane_f32(output + 0, vout0n1, 0);
        output = (float*) ((uintptr_t) output + output_stride);
        n -= 2;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x2_t vacc0 = vld1_dup_f32(w); w += 1;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x2_t vi0 = vld1_dup_f32(input);
              input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
              const float32x2_t vw = vld1_dup_f32(w); w += 1;
              vacc0 = vfma_f32(vacc0, vi0, vw);
            } while (--nnz != 0);
          }
          float32x2_t vout0 = vmin_f32(vacc0, vget_low_f32(vmax));
          vout0 = vmax_f32(vout0, vget_low_f32(vmin));

          vst1_lane_f32(output, vout0, 1);
          output = (float*) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 1;
    }
    }
}

void xnn_f32_spmm_minmax_ukernel_32x4__aarch64_neonfma(
    size_t mc,
    size_t nc,
    const float* input,
    const float* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    float* output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mc != 0);
  assert(mc % sizeof(float) == 0);
  assert(nc != 0);

  #if XNN_ARCH_ARM64
    const float32x4x2_t vminmax = vld2q_dup_f32(&params->scalar.min);
    const float32x4_t vmin = vminmax.val[0];
    const float32x4_t vmax = vminmax.val[1];
  #else
    const float32x2x2_t vminmax = vld2_dup_f32(&params->scalar.min);
    const float32x4_t vmin = vcombine_f32(vminmax.val[0],vminmax.val[0]);
    const float32x4_t vmax = vcombine_f32(vminmax.val[1],vminmax.val[1]);
  #endif

  size_t output_decrement = output_stride * nc - 32 * sizeof(float);
  while XNN_LIKELY(mc >= 32 * sizeof(float)) {
    const float* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    while (n >= 4) {
      uint32_t nnz = *nnzmap++;
      float32x4_t vacc0123n0 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567n0 = vacc0123n0;
      float32x4_t vacc89ABn0 = vacc0123n0;
      float32x4_t vaccCDEFn0 = vacc0123n0;
      float32x4_t vaccGHIJn0 = vacc0123n0;
      float32x4_t vaccKLMNn0 = vacc0123n0;
      float32x4_t vaccOPQRn0 = vacc0123n0;
      float32x4_t vaccSTUVn0 = vacc0123n0;
      float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567n1 = vacc0123n1;
      float32x4_t vacc89ABn1 = vacc0123n1;
      float32x4_t vaccCDEFn1 = vacc0123n1;
      float32x4_t vaccGHIJn1 = vacc0123n1;
      float32x4_t vaccKLMNn1 = vacc0123n1;
      float32x4_t vaccOPQRn1 = vacc0123n1;
      float32x4_t vaccSTUVn1 = vacc0123n1;
      float32x4_t vacc0123n2 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567n2 = vacc0123n2;
      float32x4_t vacc89ABn2 = vacc0123n2;
      float32x4_t vaccCDEFn2 = vacc0123n2;
      float32x4_t vaccGHIJn2 = vacc0123n2;
      float32x4_t vaccKLMNn2 = vacc0123n2;
      float32x4_t vaccOPQRn2 = vacc0123n2;
      float32x4_t vaccSTUVn2 = vacc0123n2;
      float32x4_t vacc0123n3 = vld1q_dup_f32(w); w += 1;
      float32x4_t vacc4567n3 = vacc0123n3;
      float32x4_t vacc89ABn3 = vacc0123n3;
      float32x4_t vaccCDEFn3 = vacc0123n3;
      float32x4_t vaccGHIJn3 = vacc0123n3;
      float32x4_t vaccKLMNn3 = vacc0123n3;
      float32x4_t vaccOPQRn3 = vacc0123n3;
      float32x4_t vaccSTUVn3 = vacc0123n3;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const float32x4_t vi0123 = vld1q_f32(input);
          const float32x4_t vi4567 = vld1q_f32(input + 4);
          const float32x4_t vi89AB = vld1q_f32(input + 8);
          const float32x4_t viCDEF = vld1q_f32(input + 12);
          const float32x4_t viGHIJ = vld1q_f32(input + 16);
          const float32x4_t viKLMN = vld1q_f32(input + 20);
          const float32x4_t viOPQR = vld1q_f32(input + 24);
          const float32x4_t viSTUV = vld1q_f32(input + 28);
          input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
          xnn_prefetch_to_l1(input + 16);
          xnn_prefetch_to_l1(input + 32);
          const float32x4_t vw = vld1q_f32(w); w += 4;
          xnn_prefetch_to_l1(w + 32);
          vacc0123n0 = vfmaq_laneq_f32(vacc0123n0, vi0123, vw, 0);
          vacc4567n0 = vfmaq_laneq_f32(vacc4567n0, vi4567, vw, 0);
          vacc89ABn0 = vfmaq_laneq_f32(vacc89ABn0, vi89AB, vw, 0);
          vaccCDEFn0 = vfmaq_laneq_f32(vaccCDEFn0, viCDEF, vw, 0);
          vaccGHIJn0 = vfmaq_laneq_f32(vaccGHIJn0, viGHIJ, vw, 0);
          vaccKLMNn0 = vfmaq_laneq_f32(vaccKLMNn0, viKLMN, vw, 0);
          vaccOPQRn0 = vfmaq_laneq_f32(vaccOPQRn0, viOPQR, vw, 0);
          vaccSTUVn0 = vfmaq_laneq_f32(vaccSTUVn0, viSTUV, vw, 0);
          vacc0123n1 = vfmaq_laneq_f32(vacc0123n1, vi0123, vw, 1);
          vacc4567n1 = vfmaq_laneq_f32(vacc4567n1, vi4567, vw, 1);
          vacc89ABn1 = vfmaq_laneq_f32(vacc89ABn1, vi89AB, vw, 1);
          vaccCDEFn1 = vfmaq_laneq_f32(vaccCDEFn1, viCDEF, vw, 1);
          vaccGHIJn1 = vfmaq_laneq_f32(vaccGHIJn1, viGHIJ, vw, 1);
          vaccKLMNn1 = vfmaq_laneq_f32(vaccKLMNn1, viKLMN, vw, 1);
          vaccOPQRn1 = vfmaq_laneq_f32(vaccOPQRn1, viOPQR, vw, 1);
          vaccSTUVn1 = vfmaq_laneq_f32(vaccSTUVn1, viSTUV, vw, 1);
          vacc0123n2 = vfmaq_laneq_f32(vacc0123n2, vi0123, vw, 2);
          vacc4567n2 = vfmaq_laneq_f32(vacc4567n2, vi4567, vw, 2);
          vacc89ABn2 = vfmaq_laneq_f32(vacc89ABn2, vi89AB, vw, 2);
          vaccCDEFn2 = vfmaq_laneq_f32(vaccCDEFn2, viCDEF, vw, 2);
          vaccGHIJn2 = vfmaq_laneq_f32(vaccGHIJn2, viGHIJ, vw, 2);
          vaccKLMNn2 = vfmaq_laneq_f32(vaccKLMNn2, viKLMN, vw, 2);
          vaccOPQRn2 = vfmaq_laneq_f32(vaccOPQRn2, viOPQR, vw, 2);
          vaccSTUVn2 = vfmaq_laneq_f32(vaccSTUVn2, viSTUV, vw, 2);
          vacc0123n3 = vfmaq_laneq_f32(vacc0123n3, vi0123, vw, 3);
          vacc4567n3 = vfmaq_laneq_f32(vacc4567n3, vi4567, vw, 3);
          vacc89ABn3 = vfmaq_laneq_f32(vacc89ABn3, vi89AB, vw, 3);
          vaccCDEFn3 = vfmaq_laneq_f32(vaccCDEFn3, viCDEF, vw, 3);
          vaccGHIJn3 = vfmaq_laneq_f32(vaccGHIJn3, viGHIJ, vw, 3);
          vaccKLMNn3 = vfmaq_laneq_f32(vaccKLMNn3, viKLMN, vw, 3);
          vaccOPQRn3 = vfmaq_laneq_f32(vaccOPQRn3, viOPQR, vw, 3);
          vaccSTUVn3 = vfmaq_laneq_f32(vaccSTUVn3, viSTUV, vw, 3);
        } while (--nnz != 0);
      }
      float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
      float32x4_t vout4567n0 = vminq_f32(vacc4567n0, vmax);
      float32x4_t vout89ABn0 = vminq_f32(vacc89ABn0, vmax);
      float32x4_t voutCDEFn0 = vminq_f32(vaccCDEFn0, vmax);
      float32x4_t voutGHIJn0 = vminq_f32(vaccGHIJn0, vmax);
      float32x4_t voutKLMNn0 = vminq_f32(vaccKLMNn0, vmax);
      float32x4_t voutOPQRn0 = vminq_f32(vaccOPQRn0, vmax);
      float32x4_t voutSTUVn0 = vminq_f32(vaccSTUVn0, vmax);
      float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);
      float32x4_t vout4567n1 = vminq_f32(vacc4567n1, vmax);
      float32x4_t vout89ABn1 = vminq_f32(vacc89ABn1, vmax);
      float32x4_t voutCDEFn1 = vminq_f32(vaccCDEFn1, vmax);
      float32x4_t voutGHIJn1 = vminq_f32(vaccGHIJn1, vmax);
      float32x4_t voutKLMNn1 = vminq_f32(vaccKLMNn1, vmax);
      float32x4_t voutOPQRn1 = vminq_f32(vaccOPQRn1, vmax);
      float32x4_t voutSTUVn1 = vminq_f32(vaccSTUVn1, vmax);
      float32x4_t vout0123n2 = vminq_f32(vacc0123n2, vmax);
      float32x4_t vout4567n2 = vminq_f32(vacc4567n2, vmax);
      float32x4_t vout89ABn2 = vminq_f32(vacc89ABn2, vmax);
      float32x4_t voutCDEFn2 = vminq_f32(vaccCDEFn2, vmax);
      float32x4_t voutGHIJn2 = vminq_f32(vaccGHIJn2, vmax);
      float32x4_t voutKLMNn2 = vminq_f32(vaccKLMNn2, vmax);
      float32x4_t voutOPQRn2 = vminq_f32(vaccOPQRn2, vmax);
      float32x4_t voutSTUVn2 = vminq_f32(vaccSTUVn2, vmax);
      float32x4_t vout0123n3 = vminq_f32(vacc0123n3, vmax);
      float32x4_t vout4567n3 = vminq_f32(vacc4567n3, vmax);
      float32x4_t vout89ABn3 = vminq_f32(vacc89ABn3, vmax);
      float32x4_t voutCDEFn3 = vminq_f32(vaccCDEFn3, vmax);
      float32x4_t voutGHIJn3 = vminq_f32(vaccGHIJn3, vmax);
      float32x4_t voutKLMNn3 = vminq_f32(vaccKLMNn3, vmax);
      float32x4_t voutOPQRn3 = vminq_f32(vaccOPQRn3, vmax);
      float32x4_t voutSTUVn3 = vminq_f32(vaccSTUVn3, vmax);

      vout0123n0 = vmaxq_f32(vout0123n0, vmin);
      vout4567n0 = vmaxq_f32(vout4567n0, vmin);
      vout89ABn0 = vmaxq_f32(vout89ABn0, vmin);
      voutCDEFn0 = vmaxq_f32(voutCDEFn0, vmin);
      voutGHIJn0 = vmaxq_f32(voutGHIJn0, vmin);
      voutKLMNn0 = vmaxq_f32(voutKLMNn0, vmin);
      voutOPQRn0 = vmaxq_f32(voutOPQRn0, vmin);
      voutSTUVn0 = vmaxq_f32(voutSTUVn0, vmin);
      vout0123n1 = vmaxq_f32(vout0123n1, vmin);
      vout4567n1 = vmaxq_f32(vout4567n1, vmin);
      vout89ABn1 = vmaxq_f32(vout89ABn1, vmin);
      voutCDEFn1 = vmaxq_f32(voutCDEFn1, vmin);
      voutGHIJn1 = vmaxq_f32(voutGHIJn1, vmin);
      voutKLMNn1 = vmaxq_f32(voutKLMNn1, vmin);
      voutOPQRn1 = vmaxq_f32(voutOPQRn1, vmin);
      voutSTUVn1 = vmaxq_f32(voutSTUVn1, vmin);
      vout0123n2 = vmaxq_f32(vout0123n2, vmin);
      vout4567n2 = vmaxq_f32(vout4567n2, vmin);
      vout89ABn2 = vmaxq_f32(vout89ABn2, vmin);
      voutCDEFn2 = vmaxq_f32(voutCDEFn2, vmin);
      voutGHIJn2 = vmaxq_f32(voutGHIJn2, vmin);
      voutKLMNn2 = vmaxq_f32(voutKLMNn2, vmin);
      voutOPQRn2 = vmaxq_f32(voutOPQRn2, vmin);
      voutSTUVn2 = vmaxq_f32(voutSTUVn2, vmin);
      vout0123n3 = vmaxq_f32(vout0123n3, vmin);
      vout4567n3 = vmaxq_f32(vout4567n3, vmin);
      vout89ABn3 = vmaxq_f32(vout89ABn3, vmin);
      voutCDEFn3 = vmaxq_f32(voutCDEFn3, vmin);
      voutGHIJn3 = vmaxq_f32(voutGHIJn3, vmin);
      voutKLMNn3 = vmaxq_f32(voutKLMNn3, vmin);
      voutOPQRn3 = vmaxq_f32(voutOPQRn3, vmin);
      voutSTUVn3 = vmaxq_f32(voutSTUVn3, vmin);

      vst1q_f32(output + 0, vout0123n0);
      vst1q_f32(output + 4, vout4567n0);
      vst1q_f32(output + 8, vout89ABn0);
      vst1q_f32(output + 12, voutCDEFn0);
      vst1q_f32(output + 16, voutGHIJn0);
      vst1q_f32(output + 20, voutKLMNn0);
      vst1q_f32(output + 24, voutOPQRn0);
      vst1q_f32(output + 28, voutSTUVn0);
      output = (float*) ((uintptr_t) output + output_stride);
      vst1q_f32(output + 0, vout0123n1);
      vst1q_f32(output + 4, vout4567n1);
      vst1q_f32(output + 8, vout89ABn1);
      vst1q_f32(output + 12, voutCDEFn1);
      vst1q_f32(output + 16, voutGHIJn1);
      vst1q_f32(output + 20, voutKLMNn1);
      vst1q_f32(output + 24, voutOPQRn1);
      vst1q_f32(output + 28, voutSTUVn1);
      output = (float*) ((uintptr_t) output + output_stride);
      vst1q_f32(output + 0, vout0123n2);
      vst1q_f32(output + 4, vout4567n2);
      vst1q_f32(output + 8, vout89ABn2);
      vst1q_f32(output + 12, voutCDEFn2);
      vst1q_f32(output + 16, voutGHIJn2);
      vst1q_f32(output + 20, voutKLMNn2);
      vst1q_f32(output + 24, voutOPQRn2);
      vst1q_f32(output + 28, voutSTUVn2);
      output = (float*) ((uintptr_t) output + output_stride);
      vst1q_f32(output + 0, vout0123n3);
      vst1q_f32(output + 4, vout4567n3);
      vst1q_f32(output + 8, vout89ABn3);
      vst1q_f32(output + 12, voutCDEFn3);
      vst1q_f32(output + 16, voutGHIJn3);
      vst1q_f32(output + 20, voutKLMNn3);
      vst1q_f32(output + 24, voutOPQRn3);
      vst1q_f32(output + 28, voutSTUVn3);
      output = (float*) ((uintptr_t) output + output_stride);
      n -= 4;
    }

    // clean up loop, fall back to nr=1
    if XNN_UNLIKELY(n != 0) {
      do {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567 = vacc0123;
        float32x4_t vacc89AB = vacc0123;
        float32x4_t vaccCDEF = vacc0123;
        float32x4_t vaccGHIJ = vacc0123;
        float32x4_t vaccKLMN = vacc0123;
        float32x4_t vaccOPQR = vacc0123;
        float32x4_t vaccSTUV = vacc0123;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            const float32x4_t vi89AB = vld1q_f32(input + 8);
            const float32x4_t viCDEF = vld1q_f32(input + 12);
            const float32x4_t viGHIJ = vld1q_f32(input + 16);
            const float32x4_t viKLMN = vld1q_f32(input + 20);
            const float32x4_t viOPQR = vld1q_f32(input + 24);
            const float32x4_t viSTUV = vld1q_f32(input + 28);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            xnn_prefetch_to_l1(input + 16);
            xnn_prefetch_to_l1(input + 32);
            const float32x4_t vw = vld1q_dup_f32(w); w += 1;
            xnn_prefetch_to_l1(w + 32);
            vacc0123 = vfmaq_f32(vacc0123, vi0123, vw);
            vacc4567 = vfmaq_f32(vacc4567, vi4567, vw);
            vacc89AB = vfmaq_f32(vacc89AB, vi89AB, vw);
            vaccCDEF = vfmaq_f32(vaccCDEF, viCDEF, vw);
            vaccGHIJ = vfmaq_f32(vaccGHIJ, viGHIJ, vw);
            vaccKLMN = vfmaq_f32(vaccKLMN, viKLMN, vw);
            vaccOPQR = vfmaq_f32(vaccOPQR, viOPQR, vw);
            vaccSTUV = vfmaq_f32(vaccSTUV, viSTUV, vw);
          } while (--nnz != 0);
        }
        float32x4_t vout0123 = vminq_f32(vacc0123, vmax);
        float32x4_t vout4567 = vminq_f32(vacc4567, vmax);
        float32x4_t vout89AB = vminq_f32(vacc89AB, vmax);
        float32x4_t voutCDEF = vminq_f32(vaccCDEF, vmax);
        float32x4_t voutGHIJ = vminq_f32(vaccGHIJ, vmax);
        float32x4_t voutKLMN = vminq_f32(vaccKLMN, vmax);
        float32x4_t voutOPQR = vminq_f32(vaccOPQR, vmax);
        float32x4_t voutSTUV = vminq_f32(vaccSTUV, vmax);

        vout0123 = vmaxq_f32(vout0123, vmin);
        vout4567 = vmaxq_f32(vout4567, vmin);
        vout89AB = vmaxq_f32(vout89AB, vmin);
        voutCDEF = vmaxq_f32(voutCDEF, vmin);
        voutGHIJ = vmaxq_f32(voutGHIJ, vmin);
        voutKLMN = vmaxq_f32(voutKLMN, vmin);
        voutOPQR = vmaxq_f32(voutOPQR, vmin);
        voutSTUV = vmaxq_f32(voutSTUV, vmin);

        vst1q_f32(output + 0, vout0123);
        vst1q_f32(output + 4, vout4567);
        vst1q_f32(output + 8, vout89AB);
        vst1q_f32(output + 12, voutCDEF);
        vst1q_f32(output + 16, voutGHIJ);
        vst1q_f32(output + 20, voutKLMN);
        vst1q_f32(output + 24, voutOPQR);
        vst1q_f32(output + 28, voutSTUV);
        output = (float*) ((uintptr_t) output + output_stride);
        n -= 1;
      } while (n != 0);
    }
    output = (float*) ((uintptr_t) output - output_decrement);
    input += 32;
    mc -= 32 * sizeof(float);
  }
  if XNN_UNLIKELY(mc != 0) {
    output_decrement += 16 * sizeof(float);
    if (mc & (16 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 4) {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123n0 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n0 = vacc0123n0;
        float32x4_t vacc89ABn0 = vacc0123n0;
        float32x4_t vaccCDEFn0 = vacc0123n0;
        float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n1 = vacc0123n1;
        float32x4_t vacc89ABn1 = vacc0123n1;
        float32x4_t vaccCDEFn1 = vacc0123n1;
        float32x4_t vacc0123n2 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n2 = vacc0123n2;
        float32x4_t vacc89ABn2 = vacc0123n2;
        float32x4_t vaccCDEFn2 = vacc0123n2;
        float32x4_t vacc0123n3 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n3 = vacc0123n3;
        float32x4_t vacc89ABn3 = vacc0123n3;
        float32x4_t vaccCDEFn3 = vacc0123n3;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            const float32x4_t vi89AB = vld1q_f32(input + 8);
            const float32x4_t viCDEF = vld1q_f32(input + 12);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const float32x4_t vw = vld1q_f32(w); w += 4;

            vacc0123n0 = vfmaq_laneq_f32(vacc0123n0, vi0123, vw, 0);
            vacc4567n0 = vfmaq_laneq_f32(vacc4567n0, vi4567, vw, 0);
            vacc89ABn0 = vfmaq_laneq_f32(vacc89ABn0, vi89AB, vw, 0);
            vaccCDEFn0 = vfmaq_laneq_f32(vaccCDEFn0, viCDEF, vw, 0);
            vacc0123n1 = vfmaq_laneq_f32(vacc0123n1, vi0123, vw, 1);
            vacc4567n1 = vfmaq_laneq_f32(vacc4567n1, vi4567, vw, 1);
            vacc89ABn1 = vfmaq_laneq_f32(vacc89ABn1, vi89AB, vw, 1);
            vaccCDEFn1 = vfmaq_laneq_f32(vaccCDEFn1, viCDEF, vw, 1);
            vacc0123n2 = vfmaq_laneq_f32(vacc0123n2, vi0123, vw, 2);
            vacc4567n2 = vfmaq_laneq_f32(vacc4567n2, vi4567, vw, 2);
            vacc89ABn2 = vfmaq_laneq_f32(vacc89ABn2, vi89AB, vw, 2);
            vaccCDEFn2 = vfmaq_laneq_f32(vaccCDEFn2, viCDEF, vw, 2);
            vacc0123n3 = vfmaq_laneq_f32(vacc0123n3, vi0123, vw, 3);
            vacc4567n3 = vfmaq_laneq_f32(vacc4567n3, vi4567, vw, 3);
            vacc89ABn3 = vfmaq_laneq_f32(vacc89ABn3, vi89AB, vw, 3);
            vaccCDEFn3 = vfmaq_laneq_f32(vaccCDEFn3, viCDEF, vw, 3);
          } while (--nnz != 0);
        }
        float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
        float32x4_t vout4567n0 = vminq_f32(vacc4567n0, vmax);
        float32x4_t vout89ABn0 = vminq_f32(vacc89ABn0, vmax);
        float32x4_t voutCDEFn0 = vminq_f32(vaccCDEFn0, vmax);
        float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);
        float32x4_t vout4567n1 = vminq_f32(vacc4567n1, vmax);
        float32x4_t vout89ABn1 = vminq_f32(vacc89ABn1, vmax);
        float32x4_t voutCDEFn1 = vminq_f32(vaccCDEFn1, vmax);
        float32x4_t vout0123n2 = vminq_f32(vacc0123n2, vmax);
        float32x4_t vout4567n2 = vminq_f32(vacc4567n2, vmax);
        float32x4_t vout89ABn2 = vminq_f32(vacc89ABn2, vmax);
        float32x4_t voutCDEFn2 = vminq_f32(vaccCDEFn2, vmax);
        float32x4_t vout0123n3 = vminq_f32(vacc0123n3, vmax);
        float32x4_t vout4567n3 = vminq_f32(vacc4567n3, vmax);
        float32x4_t vout89ABn3 = vminq_f32(vacc89ABn3, vmax);
        float32x4_t voutCDEFn3 = vminq_f32(vaccCDEFn3, vmax);

        vout0123n0 = vmaxq_f32(vout0123n0, vmin);
        vout4567n0 = vmaxq_f32(vout4567n0, vmin);
        vout89ABn0 = vmaxq_f32(vout89ABn0, vmin);
        voutCDEFn0 = vmaxq_f32(voutCDEFn0, vmin);
        vout0123n1 = vmaxq_f32(vout0123n1, vmin);
        vout4567n1 = vmaxq_f32(vout4567n1, vmin);
        vout89ABn1 = vmaxq_f32(vout89ABn1, vmin);
        voutCDEFn1 = vmaxq_f32(voutCDEFn1, vmin);
        vout0123n2 = vmaxq_f32(vout0123n2, vmin);
        vout4567n2 = vmaxq_f32(vout4567n2, vmin);
        vout89ABn2 = vmaxq_f32(vout89ABn2, vmin);
        voutCDEFn2 = vmaxq_f32(voutCDEFn2, vmin);
        vout0123n3 = vmaxq_f32(vout0123n3, vmin);
        vout4567n3 = vmaxq_f32(vout4567n3, vmin);
        vout89ABn3 = vmaxq_f32(vout89ABn3, vmin);
        voutCDEFn3 = vmaxq_f32(voutCDEFn3, vmin);

        vst1q_f32(output + 0, vout0123n0);
        vst1q_f32(output + 4, vout4567n0);
        vst1q_f32(output + 8, vout89ABn0);
        vst1q_f32(output + 12, voutCDEFn0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n1);
        vst1q_f32(output + 4, vout4567n1);
        vst1q_f32(output + 8, vout89ABn1);
        vst1q_f32(output + 12, voutCDEFn1);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n2);
        vst1q_f32(output + 4, vout4567n2);
        vst1q_f32(output + 8, vout89ABn2);
        vst1q_f32(output + 12, voutCDEFn2);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n3);
        vst1q_f32(output + 4, vout4567n3);
        vst1q_f32(output + 8, vout89ABn3);
        vst1q_f32(output + 12, voutCDEFn3);
        output = (float*) ((uintptr_t) output + output_stride);
        n -= 4;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
          float32x4_t vacc4567 = vacc0123;
          float32x4_t vacc89AB = vacc0123;
          float32x4_t vaccCDEF = vacc0123;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x4_t vi0123 = vld1q_f32(input);
              const float32x4_t vi4567 = vld1q_f32(input + 4);
              const float32x4_t vi89AB = vld1q_f32(input + 8);
              const float32x4_t viCDEF = vld1q_f32(input + 12);
              input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
              const float32x4_t vw = vld1q_dup_f32(w); w += 1;
              vacc0123 = vfmaq_f32(vacc0123, vi0123, vw);
              vacc4567 = vfmaq_f32(vacc4567, vi4567, vw);
              vacc89AB = vfmaq_f32(vacc89AB, vi89AB, vw);
              vaccCDEF = vfmaq_f32(vaccCDEF, viCDEF, vw);
            } while (--nnz != 0);
          }
          float32x4_t vout0123 = vminq_f32(vacc0123, vmax);
          float32x4_t vout4567 = vminq_f32(vacc4567, vmax);
          float32x4_t vout89AB = vminq_f32(vacc89AB, vmax);
          float32x4_t voutCDEF = vminq_f32(vaccCDEF, vmax);

          vout0123 = vmaxq_f32(vout0123, vmin);
          vout4567 = vmaxq_f32(vout4567, vmin);
          vout89AB = vmaxq_f32(vout89AB, vmin);
          voutCDEF = vmaxq_f32(voutCDEF, vmin);

          vst1q_f32(output + 0, vout0123);
          vst1q_f32(output + 4, vout4567);
          vst1q_f32(output + 8, vout89AB);
          vst1q_f32(output + 12, voutCDEF);
          output = (float*) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 16;
    }
    output_decrement += 8 * sizeof(float);
    if (mc & (8 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 4) {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123n0 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n0 = vacc0123n0;
        float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n1 = vacc0123n1;
        float32x4_t vacc0123n2 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n2 = vacc0123n2;
        float32x4_t vacc0123n3 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc4567n3 = vacc0123n3;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            const float32x4_t vi4567 = vld1q_f32(input + 4);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const float32x4_t vw = vld1q_f32(w); w += 4;

            vacc0123n0 = vfmaq_laneq_f32(vacc0123n0, vi0123, vw, 0);
            vacc4567n0 = vfmaq_laneq_f32(vacc4567n0, vi4567, vw, 0);
            vacc0123n1 = vfmaq_laneq_f32(vacc0123n1, vi0123, vw, 1);
            vacc4567n1 = vfmaq_laneq_f32(vacc4567n1, vi4567, vw, 1);
            vacc0123n2 = vfmaq_laneq_f32(vacc0123n2, vi0123, vw, 2);
            vacc4567n2 = vfmaq_laneq_f32(vacc4567n2, vi4567, vw, 2);
            vacc0123n3 = vfmaq_laneq_f32(vacc0123n3, vi0123, vw, 3);
            vacc4567n3 = vfmaq_laneq_f32(vacc4567n3, vi4567, vw, 3);
          } while (--nnz != 0);
        }
        float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
        float32x4_t vout4567n0 = vminq_f32(vacc4567n0, vmax);
        float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);
        float32x4_t vout4567n1 = vminq_f32(vacc4567n1, vmax);
        float32x4_t vout0123n2 = vminq_f32(vacc0123n2, vmax);
        float32x4_t vout4567n2 = vminq_f32(vacc4567n2, vmax);
        float32x4_t vout0123n3 = vminq_f32(vacc0123n3, vmax);
        float32x4_t vout4567n3 = vminq_f32(vacc4567n3, vmax);

        vout0123n0 = vmaxq_f32(vout0123n0, vmin);
        vout4567n0 = vmaxq_f32(vout4567n0, vmin);
        vout0123n1 = vmaxq_f32(vout0123n1, vmin);
        vout4567n1 = vmaxq_f32(vout4567n1, vmin);
        vout0123n2 = vmaxq_f32(vout0123n2, vmin);
        vout4567n2 = vmaxq_f32(vout4567n2, vmin);
        vout0123n3 = vmaxq_f32(vout0123n3, vmin);
        vout4567n3 = vmaxq_f32(vout4567n3, vmin);

        vst1q_f32(output + 0, vout0123n0);
        vst1q_f32(output + 4, vout4567n0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n1);
        vst1q_f32(output + 4, vout4567n1);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n2);
        vst1q_f32(output + 4, vout4567n2);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n3);
        vst1q_f32(output + 4, vout4567n3);
        output = (float*) ((uintptr_t) output + output_stride);
        n -= 4;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
          float32x4_t vacc4567 = vacc0123;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x4_t vi0123 = vld1q_f32(input);
              const float32x4_t vi4567 = vld1q_f32(input + 4);
              input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
              const float32x4_t vw = vld1q_dup_f32(w); w += 1;
              vacc0123 = vfmaq_f32(vacc0123, vi0123, vw);
              vacc4567 = vfmaq_f32(vacc4567, vi4567, vw);
            } while (--nnz != 0);
          }
          float32x4_t vout0123 = vminq_f32(vacc0123, vmax);
          float32x4_t vout4567 = vminq_f32(vacc4567, vmax);

          vout0123 = vmaxq_f32(vout0123, vmin);
          vout4567 = vmaxq_f32(vout4567, vmin);

          vst1q_f32(output + 0, vout0123);
          vst1q_f32(output + 4, vout4567);
          output = (float*) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 8;
    }
    output_decrement += 4 * sizeof(float);
    if (mc & (4 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 4) {
        uint32_t nnz = *nnzmap++;
        float32x4_t vacc0123n0 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc0123n1 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc0123n2 = vld1q_dup_f32(w); w += 1;
        float32x4_t vacc0123n3 = vld1q_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x4_t vi0123 = vld1q_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const float32x4_t vw = vld1q_f32(w); w += 4;

            vacc0123n0 = vfmaq_laneq_f32(vacc0123n0, vi0123, vw, 0);
            vacc0123n1 = vfmaq_laneq_f32(vacc0123n1, vi0123, vw, 1);
            vacc0123n2 = vfmaq_laneq_f32(vacc0123n2, vi0123, vw, 2);
            vacc0123n3 = vfmaq_laneq_f32(vacc0123n3, vi0123, vw, 3);
          } while (--nnz != 0);
        }
        float32x4_t vout0123n0 = vminq_f32(vacc0123n0, vmax);
        float32x4_t vout0123n1 = vminq_f32(vacc0123n1, vmax);
        float32x4_t vout0123n2 = vminq_f32(vacc0123n2, vmax);
        float32x4_t vout0123n3 = vminq_f32(vacc0123n3, vmax);

        vout0123n0 = vmaxq_f32(vout0123n0, vmin);
        vout0123n1 = vmaxq_f32(vout0123n1, vmin);
        vout0123n2 = vmaxq_f32(vout0123n2, vmin);
        vout0123n3 = vmaxq_f32(vout0123n3, vmin);

        vst1q_f32(output + 0, vout0123n0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n1);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n2);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1q_f32(output + 0, vout0123n3);
        output = (float*) ((uintptr_t) output + output_stride);
        n -= 4;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x4_t vacc0123 = vld1q_dup_f32(w); w += 1;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x4_t vi0123 = vld1q_f32(input);
              input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
              const float32x4_t vw = vld1q_dup_f32(w); w += 1;
              vacc0123 = vfmaq_f32(vacc0123, vi0123, vw);
            } while (--nnz != 0);
          }
          float32x4_t vout0123 = vminq_f32(vacc0123, vmax);

          vout0123 = vmaxq_f32(vout0123, vmin);

          vst1q_f32(output + 0, vout0123);
          output = (float*) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 4;
    }
    output_decrement += 2 * sizeof(float);
    if (mc & (2 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 4) {
        uint32_t nnz = *nnzmap++;
        float32x2_t vacc01n0 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc01n1 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc01n2 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc01n3 = vld1_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t vi01 = vld1_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const float32x4_t vw = vld1q_f32(w); w += 4;

            vacc01n0 = vfma_laneq_f32(vacc01n0, vi01, vw, 0);
            vacc01n1 = vfma_laneq_f32(vacc01n1, vi01, vw, 1);
            vacc01n2 = vfma_laneq_f32(vacc01n2, vi01, vw, 2);
            vacc01n3 = vfma_laneq_f32(vacc01n3, vi01, vw, 3);
          } while (--nnz != 0);
        }
        float32x2_t vout01n0 = vmin_f32(vacc01n0, vget_low_f32(vmax));
        float32x2_t vout01n1 = vmin_f32(vacc01n1, vget_low_f32(vmax));
        float32x2_t vout01n2 = vmin_f32(vacc01n2, vget_low_f32(vmax));
        float32x2_t vout01n3 = vmin_f32(vacc01n3, vget_low_f32(vmax));

        vout01n0 = vmax_f32(vout01n0, vget_low_f32(vmin));
        vout01n1 = vmax_f32(vout01n1, vget_low_f32(vmin));
        vout01n2 = vmax_f32(vout01n2, vget_low_f32(vmin));
        vout01n3 = vmax_f32(vout01n3, vget_low_f32(vmin));

        vst1_f32(output + 0, vout01n0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1_f32(output + 0, vout01n1);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1_f32(output + 0, vout01n2);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1_f32(output + 0, vout01n3);
        output = (float*) ((uintptr_t) output + output_stride);
        n -= 4;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x2_t vacc01 = vld1_dup_f32(w); w += 1;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x2_t vi01 = vld1_f32(input);
              input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
              const float32x2_t vw = vld1_dup_f32(w); w += 1;
              vacc01 = vfma_f32(vacc01, vi01, vw);
            } while (--nnz != 0);
          }
          float32x2_t vout01 = vmin_f32(vacc01, vget_low_f32(vmax));
          vout01 = vmax_f32(vout01, vget_low_f32(vmin));

          vst1_f32(output, vout01);
          output = (float*) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 2;
    }
    output_decrement += 1 * sizeof(float);
    if (mc & (1 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      while (n >= 4) {
        uint32_t nnz = *nnzmap++;
        float32x2_t vacc0n0 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc0n1 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc0n2 = vld1_dup_f32(w); w += 1;
        float32x2_t vacc0n3 = vld1_dup_f32(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const float32x2_t vi0 = vld1_dup_f32(input);
            input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
            const float32x4_t vw = vld1q_f32(w); w += 4;

            vacc0n0 = vfma_laneq_f32(vacc0n0, vi0, vw, 0);
            vacc0n1 = vfma_laneq_f32(vacc0n1, vi0, vw, 1);
            vacc0n2 = vfma_laneq_f32(vacc0n2, vi0, vw, 2);
            vacc0n3 = vfma_laneq_f32(vacc0n3, vi0, vw, 3);
          } while (--nnz != 0);
        }
        float32x2_t vout0n0 = vmin_f32(vacc0n0, vget_low_f32(vmax));
        float32x2_t vout0n1 = vmin_f32(vacc0n1, vget_low_f32(vmax));
        float32x2_t vout0n2 = vmin_f32(vacc0n2, vget_low_f32(vmax));
        float32x2_t vout0n3 = vmin_f32(vacc0n3, vget_low_f32(vmax));

        vout0n0 = vmax_f32(vout0n0, vget_low_f32(vmin));
        vout0n1 = vmax_f32(vout0n1, vget_low_f32(vmin));
        vout0n2 = vmax_f32(vout0n2, vget_low_f32(vmin));
        vout0n3 = vmax_f32(vout0n3, vget_low_f32(vmin));

        vst1_lane_f32(output + 0, vout0n0, 0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1_lane_f32(output + 0, vout0n1, 0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1_lane_f32(output + 0, vout0n2, 0);
        output = (float*) ((uintptr_t) output + output_stride);
        vst1_lane_f32(output + 0, vout0n3, 0);
        output = (float*) ((uintptr_t) output + output_stride);
        n -= 4;
      }

      // clean up loop, fall back to nr=1
      if XNN_UNLIKELY(n != 0) {
        do {
          uint32_t nnz = *nnzmap++;
          float32x2_t vacc0 = vld1_dup_f32(w); w += 1;
          if XNN_LIKELY(nnz != 0) {
            do {
              const intptr_t diff = *dmap++;
              const float32x2_t vi0 = vld1_dup_f32(input);
              input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
              const float32x2_t vw = vld1_dup_f32(w); w += 1;
              vacc0 = vfma_f32(vacc0, vi0, vw);
            } while (--nnz != 0);
          }
          float32x2_t vout0 = vmin_f32(vacc0, vget_low_f32(vmax));
          vout0 = vmax_f32(vout0, vget_low_f32(vmin));

          vst1_lane_f32(output, vout0, 1);
          output = (float*) ((uintptr_t) output + output_stride);
          n -= 1;
        } while (n != 0);
      }
      output = (float*) ((uintptr_t) output - output_decrement);
      input += 1;
    }
    }
}

void xnn_f32_vdiv_minmax_ukernel__aarch64_neon_x8(
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

void xnn_f32_vdivc_minmax_ukernel__aarch64_neon_x8(
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

void xnn_f32_vrdivc_minmax_ukernel__aarch64_neon_x8(
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

void xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_x4(
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
    size_t block_height,
    const union xnn_x24_transpose_params* params) XNN_OOB_READS
{
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

  const uint8x16_t vperm0 = vld1q_u8(params->neon_tbl128.pos0);
  const uint8x16_t vperm1 = vld1q_u8(params->neon_tbl128.pos1);
  const uint8x16_t vperm2 = vld1q_u8(params->neon_tbl128.pos2);
  const uint8x16_t vperm3 = vld1q_u8(params->neon_tbl128.pos3);
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
    size_t block_height,
    const union xnn_x32_transpose_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_stride >= block_height * sizeof(uint32_t));
  assert(input_stride >= block_width * sizeof(uint32_t));

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

  const uint8x16_t vperm0 = vld1q_u8(params->neon_tbl128.pos0);
  const uint8x16_t vperm1 = vld1q_u8(params->neon_tbl128.pos1);
  const uint8x16_t vperm2 = vld1q_u8(params->neon_tbl128.pos2);
  const uint8x16_t vperm3 = vld1q_u8(params->neon_tbl128.pos3);
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

void xnn_x8_lut_ukernel__aarch64_neon_tbx128x4_x64(
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
