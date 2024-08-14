// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Generator: tools/update-microkernels.py -a

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/conv.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/gemm.h"
#include "xnnpack/igemm.h"
#include "xnnpack/math.h"
#include "xnnpack/microparams.h"
#include "xnnpack/prefetch.h"
#include "xnnpack/spmm.h"
#include "xnnpack/vunary.h"


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

void xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128(
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
    float32x4_t vacc0x0 = vld1q_f32(w); w += 4;
    float32x4_t vacc0x1 = vld1q_f32(w); w += 4;

    size_t k = kc;
    if XNN_LIKELY(k >= 4 * sizeof(float)) {
      do {
        const float32x4_t va0 = vld1q_f32(a0); a0 += 4;


        const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c0, vget_low_f32(va0), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c0, vget_low_f32(va0), 0);

        const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c1, vget_low_f32(va0), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c1, vget_low_f32(va0), 1);

        const float32x4_t vb0123c2 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c2 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c2, vget_high_f32(va0), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c2, vget_high_f32(va0), 0);

        const float32x4_t vb0123c3 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c3 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c3, vget_high_f32(va0), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c3, vget_high_f32(va0), 1);
        k -= 4 * sizeof(float);
      } while (k >= 4 * sizeof(float));
    }

    if XNN_UNLIKELY(k != 0) {
      if XNN_UNLIKELY(k & (2 * sizeof(float))) {
        const float32x2_t va0 = vld1_f32(a0); a0 += 2;


        const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c0, va0, 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c0, va0, 0);

        const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c1, va0, 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c1, va0, 1);
      }
      if XNN_UNLIKELY(k & (1 * sizeof(float))) {
        const float32x4_t va0 = vld1q_dup_f32(a0); a0 += 1;

        const float32x4_t vb0123 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567);
      }
    }
    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;

    } else {
      if (nc & 4) {
        vst1q_f32(c0, vacc0x0); c0 += 4;

        vacc0x0 = vacc0x1;
      }
      float32x2_t vacc0 = vget_low_f32(vacc0x0);
      if (nc & 2) {
        vst1_f32(c0, vacc0); c0 += 2;

        vacc0 = vget_high_f32(vacc0x0);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vacc0, 0);
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
    const float* restrict w,
    float* restrict c,
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

      const float32x4_t vb01c01 = vld1q_f32(w); w += 4;
      const float32x2_t vb01c0 = vget_low_f32(vb01c01);
      const float32x2_t vb01c1 = vget_high_f32(vb01c01);
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

void xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128(
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
    float32x4_t vacc0x0 = vld1q_f32(w); w += 4;
    float32x4_t vacc0x1 = vld1q_f32(w); w += 4;
    float32x4_t vacc1x0 = vacc0x0;
    float32x4_t vacc1x1 = vacc0x1;
    float32x4_t vacc2x0 = vacc0x0;
    float32x4_t vacc2x1 = vacc0x1;
    float32x4_t vacc3x0 = vacc0x0;
    float32x4_t vacc3x1 = vacc0x1;
    float32x4_t vacc4x0 = vacc0x0;
    float32x4_t vacc4x1 = vacc0x1;
    float32x4_t vacc5x0 = vacc0x0;
    float32x4_t vacc5x1 = vacc0x1;

    size_t k = kc;
    if XNN_LIKELY(k >= 4 * sizeof(float)) {
      do {
        const float32x4_t va0 = vld1q_f32(a0); a0 += 4;
        const float32x4_t va1 = vld1q_f32(a1); a1 += 4;
        const float32x4_t va2 = vld1q_f32(a2); a2 += 4;
        const float32x4_t va3 = vld1q_f32(a3); a3 += 4;
        const float32x4_t va4 = vld1q_f32(a4); a4 += 4;
        const float32x4_t va5 = vld1q_f32(a5); a5 += 4;


        const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c0, vget_low_f32(va0), 0);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c0, vget_low_f32(va1), 0);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c0, vget_low_f32(va2), 0);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c0, vget_low_f32(va3), 0);
        vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0123c0, vget_low_f32(va4), 0);
        vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0123c0, vget_low_f32(va5), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c0, vget_low_f32(va0), 0);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c0, vget_low_f32(va1), 0);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c0, vget_low_f32(va2), 0);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c0, vget_low_f32(va3), 0);
        vacc4x1 = vfmaq_lane_f32(vacc4x1, vb4567c0, vget_low_f32(va4), 0);
        vacc5x1 = vfmaq_lane_f32(vacc5x1, vb4567c0, vget_low_f32(va5), 0);

        const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c1, vget_low_f32(va0), 1);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c1, vget_low_f32(va1), 1);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c1, vget_low_f32(va2), 1);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c1, vget_low_f32(va3), 1);
        vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0123c1, vget_low_f32(va4), 1);
        vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0123c1, vget_low_f32(va5), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c1, vget_low_f32(va0), 1);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c1, vget_low_f32(va1), 1);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c1, vget_low_f32(va2), 1);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c1, vget_low_f32(va3), 1);
        vacc4x1 = vfmaq_lane_f32(vacc4x1, vb4567c1, vget_low_f32(va4), 1);
        vacc5x1 = vfmaq_lane_f32(vacc5x1, vb4567c1, vget_low_f32(va5), 1);

        const float32x4_t vb0123c2 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c2 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c2, vget_high_f32(va0), 0);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c2, vget_high_f32(va1), 0);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c2, vget_high_f32(va2), 0);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c2, vget_high_f32(va3), 0);
        vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0123c2, vget_high_f32(va4), 0);
        vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0123c2, vget_high_f32(va5), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c2, vget_high_f32(va0), 0);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c2, vget_high_f32(va1), 0);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c2, vget_high_f32(va2), 0);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c2, vget_high_f32(va3), 0);
        vacc4x1 = vfmaq_lane_f32(vacc4x1, vb4567c2, vget_high_f32(va4), 0);
        vacc5x1 = vfmaq_lane_f32(vacc5x1, vb4567c2, vget_high_f32(va5), 0);

        const float32x4_t vb0123c3 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c3 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c3, vget_high_f32(va0), 1);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c3, vget_high_f32(va1), 1);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c3, vget_high_f32(va2), 1);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c3, vget_high_f32(va3), 1);
        vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0123c3, vget_high_f32(va4), 1);
        vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0123c3, vget_high_f32(va5), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c3, vget_high_f32(va0), 1);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c3, vget_high_f32(va1), 1);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c3, vget_high_f32(va2), 1);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c3, vget_high_f32(va3), 1);
        vacc4x1 = vfmaq_lane_f32(vacc4x1, vb4567c3, vget_high_f32(va4), 1);
        vacc5x1 = vfmaq_lane_f32(vacc5x1, vb4567c3, vget_high_f32(va5), 1);
        k -= 4 * sizeof(float);
      } while (k >= 4 * sizeof(float));
    }

    if XNN_UNLIKELY(k != 0) {
      if XNN_UNLIKELY(k & (2 * sizeof(float))) {
        const float32x2_t va0 = vld1_f32(a0); a0 += 2;
        const float32x2_t va1 = vld1_f32(a1); a1 += 2;
        const float32x2_t va2 = vld1_f32(a2); a2 += 2;
        const float32x2_t va3 = vld1_f32(a3); a3 += 2;
        const float32x2_t va4 = vld1_f32(a4); a4 += 2;
        const float32x2_t va5 = vld1_f32(a5); a5 += 2;


        const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c0, va0, 0);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c0, va1, 0);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c0, va2, 0);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c0, va3, 0);
        vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0123c0, va4, 0);
        vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0123c0, va5, 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c0, va0, 0);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c0, va1, 0);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c0, va2, 0);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c0, va3, 0);
        vacc4x1 = vfmaq_lane_f32(vacc4x1, vb4567c0, va4, 0);
        vacc5x1 = vfmaq_lane_f32(vacc5x1, vb4567c0, va5, 0);

        const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c1, va0, 1);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c1, va1, 1);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c1, va2, 1);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c1, va3, 1);
        vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0123c1, va4, 1);
        vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0123c1, va5, 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c1, va0, 1);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c1, va1, 1);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c1, va2, 1);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c1, va3, 1);
        vacc4x1 = vfmaq_lane_f32(vacc4x1, vb4567c1, va4, 1);
        vacc5x1 = vfmaq_lane_f32(vacc5x1, vb4567c1, va5, 1);
      }
      if XNN_UNLIKELY(k & (1 * sizeof(float))) {
        const float32x4_t va0 = vld1q_dup_f32(a0); a0 += 1;
        const float32x4_t va1 = vld1q_dup_f32(a1); a1 += 1;
        const float32x4_t va2 = vld1q_dup_f32(a2); a2 += 1;
        const float32x4_t va3 = vld1q_dup_f32(a3); a3 += 1;
        const float32x4_t va4 = vld1q_dup_f32(a4); a4 += 1;
        const float32x4_t va5 = vld1q_dup_f32(a5); a5 += 1;

        const float32x4_t vb0123 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123);
        vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123);
        vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123);
        vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123);
        vacc4x0 = vfmaq_f32(vacc4x0, va4, vb0123);
        vacc5x0 = vfmaq_f32(vacc5x0, va5, vb0123);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567);
        vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567);
        vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567);
        vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567);
        vacc4x1 = vfmaq_f32(vacc4x1, va4, vb4567);
        vacc5x1 = vfmaq_f32(vacc5x1, va5, vb4567);
      }
    }
    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc1x0 = vminq_f32(vacc1x0, vmax);
    vacc2x0 = vminq_f32(vacc2x0, vmax);
    vacc3x0 = vminq_f32(vacc3x0, vmax);
    vacc4x0 = vminq_f32(vacc4x0, vmax);
    vacc5x0 = vminq_f32(vacc5x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);
    vacc1x1 = vminq_f32(vacc1x1, vmax);
    vacc2x1 = vminq_f32(vacc2x1, vmax);
    vacc3x1 = vminq_f32(vacc3x1, vmax);
    vacc4x1 = vminq_f32(vacc4x1, vmax);
    vacc5x1 = vminq_f32(vacc5x1, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc1x0 = vmaxq_f32(vacc1x0, vmin);
    vacc2x0 = vmaxq_f32(vacc2x0, vmin);
    vacc3x0 = vmaxq_f32(vacc3x0, vmin);
    vacc4x0 = vmaxq_f32(vacc4x0, vmin);
    vacc5x0 = vmaxq_f32(vacc5x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);
    vacc1x1 = vmaxq_f32(vacc1x1, vmin);
    vacc2x1 = vmaxq_f32(vacc2x1, vmin);
    vacc3x1 = vmaxq_f32(vacc3x1, vmin);
    vacc4x1 = vmaxq_f32(vacc4x1, vmin);
    vacc5x1 = vmaxq_f32(vacc5x1, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      vst1q_f32(c1, vacc1x0);
      vst1q_f32(c1 + 4, vacc1x1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1q_f32(c2, vacc2x0);
      vst1q_f32(c2 + 4, vacc2x1);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1q_f32(c3, vacc3x0);
      vst1q_f32(c3 + 4, vacc3x1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      vst1q_f32(c4, vacc4x0);
      vst1q_f32(c4 + 4, vacc4x1);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      vst1q_f32(c5, vacc5x0);
      vst1q_f32(c5 + 4, vacc5x1);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);

      nc -= 8;

    } else {
      if (nc & 4) {
        vst1q_f32(c0, vacc0x0); c0 += 4;
        vst1q_f32(c1, vacc1x0); c1 += 4;
        vst1q_f32(c2, vacc2x0); c2 += 4;
        vst1q_f32(c3, vacc3x0); c3 += 4;
        vst1q_f32(c4, vacc4x0); c4 += 4;
        vst1q_f32(c5, vacc5x0); c5 += 4;

        vacc0x0 = vacc0x1;
        vacc1x0 = vacc1x1;
        vacc2x0 = vacc2x1;
        vacc3x0 = vacc3x1;
        vacc4x0 = vacc4x1;
        vacc5x0 = vacc5x1;
      }
      float32x2_t vacc0 = vget_low_f32(vacc0x0);
      float32x2_t vacc1 = vget_low_f32(vacc1x0);
      float32x2_t vacc2 = vget_low_f32(vacc2x0);
      float32x2_t vacc3 = vget_low_f32(vacc3x0);
      float32x2_t vacc4 = vget_low_f32(vacc4x0);
      float32x2_t vacc5 = vget_low_f32(vacc5x0);
      if (nc & 2) {
        vst1_f32(c0, vacc0); c0 += 2;
        vst1_f32(c1, vacc1); c1 += 2;
        vst1_f32(c2, vacc2); c2 += 2;
        vst1_f32(c3, vacc3); c3 += 2;
        vst1_f32(c4, vacc4); c4 += 2;
        vst1_f32(c5, vacc5); c5 += 2;

        vacc0 = vget_high_f32(vacc0x0);
        vacc1 = vget_high_f32(vacc1x0);
        vacc2 = vget_high_f32(vacc2x0);
        vacc3 = vget_high_f32(vacc3x0);
        vacc4 = vget_high_f32(vacc4x0);
        vacc5 = vget_high_f32(vacc5x0);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vacc0, 0);
        vst1_lane_f32(c1, vacc1, 0);
        vst1_lane_f32(c2, vacc2, 0);
        vst1_lane_f32(c3, vacc3, 0);
        vst1_lane_f32(c4, vacc4, 0);
        vst1_lane_f32(c5, vacc5, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
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
    float32x4_t vacc0x0 = vld1q_f32(w); w += 4;
    float32x4_t vacc0x1 = vld1q_f32(w); w += 4;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
        const float32x4_t va0 = vld1q_f32(a0); a0 += 4;


        const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c0, vget_low_f32(va0), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c0, vget_low_f32(va0), 0);

        const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c1, vget_low_f32(va0), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c1, vget_low_f32(va0), 1);

        const float32x4_t vb0123c2 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c2 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c2, vget_high_f32(va0), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c2, vget_high_f32(va0), 0);

        const float32x4_t vb0123c3 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c3 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c3, vget_high_f32(va0), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c3, vget_high_f32(va0), 1);
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const float32x4_t va0 = vld1q_dup_f32(a0); a0 += 1;

          const float32x4_t vb0123 = vld1q_f32(w); w += 4;
          const float32x4_t vb4567 = vld1q_f32(w); w += 4;

          vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123);
          vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567);

          k -= sizeof(float);
        } while (k != 0);
      }

      p -= 1 * sizeof(void*);
    } while (p != 0);

    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        vst1q_f32(c0, vacc0x0); c0 += 4;

        vacc0x0 = vacc0x1;
      }
      float32x2_t vacc0 = vget_low_f32(vacc0x0);
      if (nc & 2) {
        vst1_f32(c0, vacc0); c0 += 2;

        vacc0 = vget_high_f32(vacc0x0);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vacc0, 0);
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
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
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

void xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
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
    float32x4_t vacc0x0 = vld1q_f32(w); w += 4;
    float32x4_t vacc0x1 = vld1q_f32(w); w += 4;
    float32x4_t vacc1x0 = vacc0x0;
    float32x4_t vacc1x1 = vacc0x1;
    float32x4_t vacc2x0 = vacc0x0;
    float32x4_t vacc2x1 = vacc0x1;
    float32x4_t vacc3x0 = vacc0x0;
    float32x4_t vacc3x1 = vacc0x1;
    float32x4_t vacc4x0 = vacc0x0;
    float32x4_t vacc4x1 = vacc0x1;
    float32x4_t vacc5x0 = vacc0x0;
    float32x4_t vacc5x1 = vacc0x1;

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
      for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
        const float32x4_t va0 = vld1q_f32(a0); a0 += 4;
        const float32x4_t va1 = vld1q_f32(a1); a1 += 4;
        const float32x4_t va2 = vld1q_f32(a2); a2 += 4;
        const float32x4_t va3 = vld1q_f32(a3); a3 += 4;
        const float32x4_t va4 = vld1q_f32(a4); a4 += 4;
        const float32x4_t va5 = vld1q_f32(a5); a5 += 4;


        const float32x4_t vb0123c0 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c0 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c0, vget_low_f32(va0), 0);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c0, vget_low_f32(va1), 0);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c0, vget_low_f32(va2), 0);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c0, vget_low_f32(va3), 0);
        vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0123c0, vget_low_f32(va4), 0);
        vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0123c0, vget_low_f32(va5), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c0, vget_low_f32(va0), 0);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c0, vget_low_f32(va1), 0);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c0, vget_low_f32(va2), 0);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c0, vget_low_f32(va3), 0);
        vacc4x1 = vfmaq_lane_f32(vacc4x1, vb4567c0, vget_low_f32(va4), 0);
        vacc5x1 = vfmaq_lane_f32(vacc5x1, vb4567c0, vget_low_f32(va5), 0);

        const float32x4_t vb0123c1 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c1 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c1, vget_low_f32(va0), 1);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c1, vget_low_f32(va1), 1);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c1, vget_low_f32(va2), 1);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c1, vget_low_f32(va3), 1);
        vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0123c1, vget_low_f32(va4), 1);
        vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0123c1, vget_low_f32(va5), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c1, vget_low_f32(va0), 1);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c1, vget_low_f32(va1), 1);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c1, vget_low_f32(va2), 1);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c1, vget_low_f32(va3), 1);
        vacc4x1 = vfmaq_lane_f32(vacc4x1, vb4567c1, vget_low_f32(va4), 1);
        vacc5x1 = vfmaq_lane_f32(vacc5x1, vb4567c1, vget_low_f32(va5), 1);

        const float32x4_t vb0123c2 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c2 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c2, vget_high_f32(va0), 0);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c2, vget_high_f32(va1), 0);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c2, vget_high_f32(va2), 0);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c2, vget_high_f32(va3), 0);
        vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0123c2, vget_high_f32(va4), 0);
        vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0123c2, vget_high_f32(va5), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c2, vget_high_f32(va0), 0);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c2, vget_high_f32(va1), 0);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c2, vget_high_f32(va2), 0);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c2, vget_high_f32(va3), 0);
        vacc4x1 = vfmaq_lane_f32(vacc4x1, vb4567c2, vget_high_f32(va4), 0);
        vacc5x1 = vfmaq_lane_f32(vacc5x1, vb4567c2, vget_high_f32(va5), 0);

        const float32x4_t vb0123c3 = vld1q_f32(w); w += 4;
        const float32x4_t vb4567c3 = vld1q_f32(w); w += 4;

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c3, vget_high_f32(va0), 1);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c3, vget_high_f32(va1), 1);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c3, vget_high_f32(va2), 1);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c3, vget_high_f32(va3), 1);
        vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0123c3, vget_high_f32(va4), 1);
        vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0123c3, vget_high_f32(va5), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c3, vget_high_f32(va0), 1);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c3, vget_high_f32(va1), 1);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c3, vget_high_f32(va2), 1);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c3, vget_high_f32(va3), 1);
        vacc4x1 = vfmaq_lane_f32(vacc4x1, vb4567c3, vget_high_f32(va4), 1);
        vacc5x1 = vfmaq_lane_f32(vacc5x1, vb4567c3, vget_high_f32(va5), 1);
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const float32x4_t va0 = vld1q_dup_f32(a0); a0 += 1;
          const float32x4_t va1 = vld1q_dup_f32(a1); a1 += 1;
          const float32x4_t va2 = vld1q_dup_f32(a2); a2 += 1;
          const float32x4_t va3 = vld1q_dup_f32(a3); a3 += 1;
          const float32x4_t va4 = vld1q_dup_f32(a4); a4 += 1;
          const float32x4_t va5 = vld1q_dup_f32(a5); a5 += 1;

          const float32x4_t vb0123 = vld1q_f32(w); w += 4;
          const float32x4_t vb4567 = vld1q_f32(w); w += 4;

          vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123);
          vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123);
          vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123);
          vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123);
          vacc4x0 = vfmaq_f32(vacc4x0, va4, vb0123);
          vacc5x0 = vfmaq_f32(vacc5x0, va5, vb0123);
          vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567);
          vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567);
          vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567);
          vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567);
          vacc4x1 = vfmaq_f32(vacc4x1, va4, vb4567);
          vacc5x1 = vfmaq_f32(vacc5x1, va5, vb4567);

          k -= sizeof(float);
        } while (k != 0);
      }

      p -= 6 * sizeof(void*);
    } while (p != 0);

    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc1x0 = vminq_f32(vacc1x0, vmax);
    vacc2x0 = vminq_f32(vacc2x0, vmax);
    vacc3x0 = vminq_f32(vacc3x0, vmax);
    vacc4x0 = vminq_f32(vacc4x0, vmax);
    vacc5x0 = vminq_f32(vacc5x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);
    vacc1x1 = vminq_f32(vacc1x1, vmax);
    vacc2x1 = vminq_f32(vacc2x1, vmax);
    vacc3x1 = vminq_f32(vacc3x1, vmax);
    vacc4x1 = vminq_f32(vacc4x1, vmax);
    vacc5x1 = vminq_f32(vacc5x1, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc1x0 = vmaxq_f32(vacc1x0, vmin);
    vacc2x0 = vmaxq_f32(vacc2x0, vmin);
    vacc3x0 = vmaxq_f32(vacc3x0, vmin);
    vacc4x0 = vmaxq_f32(vacc4x0, vmin);
    vacc5x0 = vmaxq_f32(vacc5x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);
    vacc1x1 = vmaxq_f32(vacc1x1, vmin);
    vacc2x1 = vmaxq_f32(vacc2x1, vmin);
    vacc3x1 = vmaxq_f32(vacc3x1, vmin);
    vacc4x1 = vmaxq_f32(vacc4x1, vmin);
    vacc5x1 = vmaxq_f32(vacc5x1, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c5, vacc5x0);
      vst1q_f32(c5 + 4, vacc5x1);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      vst1q_f32(c4, vacc4x0);
      vst1q_f32(c4 + 4, vacc4x1);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      vst1q_f32(c3, vacc3x0);
      vst1q_f32(c3 + 4, vacc3x1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      vst1q_f32(c2, vacc2x0);
      vst1q_f32(c2 + 4, vacc2x1);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1q_f32(c1, vacc1x0);
      vst1q_f32(c1 + 4, vacc1x1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1q_f32(c0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        vst1q_f32(c5, vacc5x0); c5 += 4;
        vst1q_f32(c4, vacc4x0); c4 += 4;
        vst1q_f32(c3, vacc3x0); c3 += 4;
        vst1q_f32(c2, vacc2x0); c2 += 4;
        vst1q_f32(c1, vacc1x0); c1 += 4;
        vst1q_f32(c0, vacc0x0); c0 += 4;

        vacc5x0 = vacc5x1;
        vacc4x0 = vacc4x1;
        vacc3x0 = vacc3x1;
        vacc2x0 = vacc2x1;
        vacc1x0 = vacc1x1;
        vacc0x0 = vacc0x1;
      }
      float32x2_t vacc5 = vget_low_f32(vacc5x0);
      float32x2_t vacc4 = vget_low_f32(vacc4x0);
      float32x2_t vacc3 = vget_low_f32(vacc3x0);
      float32x2_t vacc2 = vget_low_f32(vacc2x0);
      float32x2_t vacc1 = vget_low_f32(vacc1x0);
      float32x2_t vacc0 = vget_low_f32(vacc0x0);
      if (nc & 2) {
        vst1_f32(c5, vacc5); c5 += 2;
        vst1_f32(c4, vacc4); c4 += 2;
        vst1_f32(c3, vacc3); c3 += 2;
        vst1_f32(c2, vacc2); c2 += 2;
        vst1_f32(c1, vacc1); c1 += 2;
        vst1_f32(c0, vacc0); c0 += 2;

        vacc5 = vget_high_f32(vacc5x0);
        vacc4 = vget_high_f32(vacc4x0);
        vacc3 = vget_high_f32(vacc3x0);
        vacc2 = vget_high_f32(vacc2x0);
        vacc1 = vget_high_f32(vacc1x0);
        vacc0 = vget_high_f32(vacc0x0);
      }
      if (nc & 1) {
        vst1_lane_f32(c5, vacc5, 0);
        vst1_lane_f32(c4, vacc4, 0);
        vst1_lane_f32(c3, vacc3, 0);
        vst1_lane_f32(c2, vacc2, 0);
        vst1_lane_f32(c1, vacc1, 0);
        vst1_lane_f32(c0, vacc0, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc4w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
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
  const int32x4_t vminus_kernel_zero_point = vld1q_dup_s32(&params->scalar.minus_kernel_zero_point);
  const uint16x8_t vmask = vmovq_n_u16(UINT16_C(0xF));

  do {
    float32x4_t vacc0x0 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vacc0x1 = vld1q_f32(w); w = (const float*) w + 4;

    size_t k = kc;
    if XNN_LIKELY(k >= 4 * sizeof(float)) {
      do {
        const float32x4_t va0 = vld1q_f32(a0); a0 += 4;


        const uint8x16_t vw01234567c0123 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint16x8_t vxw01234567c01 = vmovl_u8(vget_low_u8(vw01234567c0123));
        const uint16x8_t vxw01234567c23 = vmovl_u8(vget_high_u8(vw01234567c0123));
        const uint16x8_t vxw01234567c0 = vandq_u16(vxw01234567c01, vmask);
        const uint16x8_t vxw01234567c1 = vshrq_n_u16(vxw01234567c01, 4);
        const uint16x8_t vxw01234567c2 = vandq_u16(vxw01234567c23, vmask);
        const uint16x8_t vxw01234567c3 = vshrq_n_u16(vxw01234567c23, 4);
        const int32x4_t vxw0123c0 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c0)));
        const int32x4_t vxw4567c0 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c0)));
        const float32x4_t vb0123c0 = vcvtq_f32_s32(vxw0123c0);
        const float32x4_t vb4567c0 = vcvtq_f32_s32(vxw4567c0);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c0, vget_low_f32(va0), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c0, vget_low_f32(va0), 0);

        const int32x4_t vxw0123c1 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c1)));
        const int32x4_t vxw4567c1 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c1)));
        const float32x4_t vb0123c1 = vcvtq_f32_s32(vxw0123c1);
        const float32x4_t vb4567c1 = vcvtq_f32_s32(vxw4567c1);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c1, vget_low_f32(va0), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c1, vget_low_f32(va0), 1);

        const int32x4_t vxw0123c2 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c2)));
        const int32x4_t vxw4567c2 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c2)));
        const float32x4_t vb0123c2 = vcvtq_f32_s32(vxw0123c2);
        const float32x4_t vb4567c2 = vcvtq_f32_s32(vxw4567c2);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c2, vget_high_f32(va0), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c2, vget_high_f32(va0), 0);

        const int32x4_t vxw0123c3 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c3)));
        const int32x4_t vxw4567c3 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c3)));
        const float32x4_t vb0123c3 = vcvtq_f32_s32(vxw0123c3);
        const float32x4_t vb4567c3 = vcvtq_f32_s32(vxw4567c3);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c3, vget_high_f32(va0), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c3, vget_high_f32(va0), 1);
        k -= 4 * sizeof(float);
      } while (k >= 4 * sizeof(float));
    }

    if XNN_UNLIKELY(k != 0) {
      if XNN_UNLIKELY(k & (2 * sizeof(float))) {
        const float32x2_t va0 = vld1_f32(a0); a0 += 2;


        const uint8x8_t vw01234567c01 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const uint16x8_t vxw01234567c01 = vmovl_u8(vw01234567c01);
        const uint16x8_t vxw01234567c0 = vandq_u16(vxw01234567c01, vmask);
        const uint16x8_t vxw01234567c1 = vshrq_n_u16(vxw01234567c01, 4);
        const int32x4_t vxw0123c0 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c0)));
        const int32x4_t vxw4567c0 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c0)));
        const float32x4_t vb0123c0 = vcvtq_f32_s32(vxw0123c0);
        const float32x4_t vb4567c0 = vcvtq_f32_s32(vxw4567c0);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c0, va0, 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c0, va0, 0);

        const int32x4_t vxw0123c1 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c1)));
        const int32x4_t vxw4567c1 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c1)));
        const float32x4_t vb0123c1 = vcvtq_f32_s32(vxw0123c1);
        const float32x4_t vb4567c1 = vcvtq_f32_s32(vxw4567c1);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c1, va0, 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c1, va0, 1);
      }
      if XNN_UNLIKELY(k & (1 * sizeof(float))) {
        const float32x4_t va0 = vld1q_dup_f32(a0); a0 += 1;

        const uint8x8_t vw01234567 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const uint16x8_t vxw01234567 = vmovl_u8(vw01234567);
        const int32x4_t vxw0123 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567)));
        const int32x4_t vxw4567 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567)));
        const float32x4_t vb0123 = vcvtq_f32_s32(vxw0123);
        const float32x4_t vb4567 = vcvtq_f32_s32(vxw4567);

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567);
      }
    }
    const float32x4_t vscale0123 = vld1q_f32(w); w = ((const float*) w + 4);
    const float32x4_t vscale4567 = vld1q_f32(w); w = ((const float*) w + 4);
    vacc0x0 = vmulq_f32(vacc0x0, vscale0123);
    vacc0x1 = vmulq_f32(vacc0x1, vscale4567);
    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;

    } else {
      if (nc & 4) {
        vst1q_f32(c0, vacc0x0); c0 += 4;

        vacc0x0 = vacc0x1;
      }
      float32x2_t vacc0 = vget_low_f32(vacc0x0);
      if (nc & 2) {
        vst1_f32(c0, vacc0); c0 += 2;

        vacc0 = vget_high_f32(vacc0x0);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vacc0, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc4w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
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
  const int32x4_t vminus_kernel_zero_point = vld1q_dup_s32(&params->scalar.minus_kernel_zero_point);
  const uint16x8_t vmask = vmovq_n_u16(UINT16_C(0xF));

  do {
    float32x4_t vacc0x0 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vacc0x1 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vacc1x0 = vacc0x0;
    float32x4_t vacc1x1 = vacc0x1;
    float32x4_t vacc2x0 = vacc0x0;
    float32x4_t vacc2x1 = vacc0x1;
    float32x4_t vacc3x0 = vacc0x0;
    float32x4_t vacc3x1 = vacc0x1;

    size_t k = kc;
    if XNN_LIKELY(k >= 4 * sizeof(float)) {
      do {
        const float32x4_t va0 = vld1q_f32(a0); a0 += 4;
        const float32x4_t va1 = vld1q_f32(a1); a1 += 4;
        const float32x4_t va2 = vld1q_f32(a2); a2 += 4;
        const float32x4_t va3 = vld1q_f32(a3); a3 += 4;


        const uint8x16_t vw01234567c0123 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint16x8_t vxw01234567c01 = vmovl_u8(vget_low_u8(vw01234567c0123));
        const uint16x8_t vxw01234567c23 = vmovl_u8(vget_high_u8(vw01234567c0123));
        const uint16x8_t vxw01234567c0 = vandq_u16(vxw01234567c01, vmask);
        const uint16x8_t vxw01234567c1 = vshrq_n_u16(vxw01234567c01, 4);
        const uint16x8_t vxw01234567c2 = vandq_u16(vxw01234567c23, vmask);
        const uint16x8_t vxw01234567c3 = vshrq_n_u16(vxw01234567c23, 4);
        const int32x4_t vxw0123c0 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c0)));
        const int32x4_t vxw4567c0 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c0)));
        const float32x4_t vb0123c0 = vcvtq_f32_s32(vxw0123c0);
        const float32x4_t vb4567c0 = vcvtq_f32_s32(vxw4567c0);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c0, vget_low_f32(va0), 0);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c0, vget_low_f32(va1), 0);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c0, vget_low_f32(va2), 0);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c0, vget_low_f32(va3), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c0, vget_low_f32(va0), 0);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c0, vget_low_f32(va1), 0);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c0, vget_low_f32(va2), 0);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c0, vget_low_f32(va3), 0);

        const int32x4_t vxw0123c1 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c1)));
        const int32x4_t vxw4567c1 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c1)));
        const float32x4_t vb0123c1 = vcvtq_f32_s32(vxw0123c1);
        const float32x4_t vb4567c1 = vcvtq_f32_s32(vxw4567c1);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c1, vget_low_f32(va0), 1);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c1, vget_low_f32(va1), 1);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c1, vget_low_f32(va2), 1);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c1, vget_low_f32(va3), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c1, vget_low_f32(va0), 1);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c1, vget_low_f32(va1), 1);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c1, vget_low_f32(va2), 1);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c1, vget_low_f32(va3), 1);

        const int32x4_t vxw0123c2 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c2)));
        const int32x4_t vxw4567c2 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c2)));
        const float32x4_t vb0123c2 = vcvtq_f32_s32(vxw0123c2);
        const float32x4_t vb4567c2 = vcvtq_f32_s32(vxw4567c2);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c2, vget_high_f32(va0), 0);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c2, vget_high_f32(va1), 0);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c2, vget_high_f32(va2), 0);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c2, vget_high_f32(va3), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c2, vget_high_f32(va0), 0);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c2, vget_high_f32(va1), 0);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c2, vget_high_f32(va2), 0);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c2, vget_high_f32(va3), 0);

        const int32x4_t vxw0123c3 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c3)));
        const int32x4_t vxw4567c3 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c3)));
        const float32x4_t vb0123c3 = vcvtq_f32_s32(vxw0123c3);
        const float32x4_t vb4567c3 = vcvtq_f32_s32(vxw4567c3);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c3, vget_high_f32(va0), 1);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c3, vget_high_f32(va1), 1);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c3, vget_high_f32(va2), 1);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c3, vget_high_f32(va3), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c3, vget_high_f32(va0), 1);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c3, vget_high_f32(va1), 1);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c3, vget_high_f32(va2), 1);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c3, vget_high_f32(va3), 1);
        k -= 4 * sizeof(float);
      } while (k >= 4 * sizeof(float));
    }

    if XNN_UNLIKELY(k != 0) {
      if XNN_UNLIKELY(k & (2 * sizeof(float))) {
        const float32x2_t va0 = vld1_f32(a0); a0 += 2;
        const float32x2_t va1 = vld1_f32(a1); a1 += 2;
        const float32x2_t va2 = vld1_f32(a2); a2 += 2;
        const float32x2_t va3 = vld1_f32(a3); a3 += 2;


        const uint8x8_t vw01234567c01 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const uint16x8_t vxw01234567c01 = vmovl_u8(vw01234567c01);
        const uint16x8_t vxw01234567c0 = vandq_u16(vxw01234567c01, vmask);
        const uint16x8_t vxw01234567c1 = vshrq_n_u16(vxw01234567c01, 4);
        const int32x4_t vxw0123c0 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c0)));
        const int32x4_t vxw4567c0 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c0)));
        const float32x4_t vb0123c0 = vcvtq_f32_s32(vxw0123c0);
        const float32x4_t vb4567c0 = vcvtq_f32_s32(vxw4567c0);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c0, va0, 0);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c0, va1, 0);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c0, va2, 0);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c0, va3, 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c0, va0, 0);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c0, va1, 0);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c0, va2, 0);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c0, va3, 0);

        const int32x4_t vxw0123c1 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c1)));
        const int32x4_t vxw4567c1 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c1)));
        const float32x4_t vb0123c1 = vcvtq_f32_s32(vxw0123c1);
        const float32x4_t vb4567c1 = vcvtq_f32_s32(vxw4567c1);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c1, va0, 1);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c1, va1, 1);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c1, va2, 1);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c1, va3, 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c1, va0, 1);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c1, va1, 1);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c1, va2, 1);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c1, va3, 1);
      }
      if XNN_UNLIKELY(k & (1 * sizeof(float))) {
        const float32x4_t va0 = vld1q_dup_f32(a0); a0 += 1;
        const float32x4_t va1 = vld1q_dup_f32(a1); a1 += 1;
        const float32x4_t va2 = vld1q_dup_f32(a2); a2 += 1;
        const float32x4_t va3 = vld1q_dup_f32(a3); a3 += 1;

        const uint8x8_t vw01234567 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const uint16x8_t vxw01234567 = vmovl_u8(vw01234567);
        const int32x4_t vxw0123 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567)));
        const int32x4_t vxw4567 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567)));
        const float32x4_t vb0123 = vcvtq_f32_s32(vxw0123);
        const float32x4_t vb4567 = vcvtq_f32_s32(vxw4567);

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123);
        vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123);
        vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123);
        vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567);
        vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567);
        vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567);
        vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567);
      }
    }
    const float32x4_t vscale0123 = vld1q_f32(w); w = ((const float*) w + 4);
    const float32x4_t vscale4567 = vld1q_f32(w); w = ((const float*) w + 4);
    vacc0x0 = vmulq_f32(vacc0x0, vscale0123);
    vacc1x0 = vmulq_f32(vacc1x0, vscale0123);
    vacc2x0 = vmulq_f32(vacc2x0, vscale0123);
    vacc3x0 = vmulq_f32(vacc3x0, vscale0123);
    vacc0x1 = vmulq_f32(vacc0x1, vscale4567);
    vacc1x1 = vmulq_f32(vacc1x1, vscale4567);
    vacc2x1 = vmulq_f32(vacc2x1, vscale4567);
    vacc3x1 = vmulq_f32(vacc3x1, vscale4567);
    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc1x0 = vminq_f32(vacc1x0, vmax);
    vacc2x0 = vminq_f32(vacc2x0, vmax);
    vacc3x0 = vminq_f32(vacc3x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);
    vacc1x1 = vminq_f32(vacc1x1, vmax);
    vacc2x1 = vminq_f32(vacc2x1, vmax);
    vacc3x1 = vminq_f32(vacc3x1, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc1x0 = vmaxq_f32(vacc1x0, vmin);
    vacc2x0 = vmaxq_f32(vacc2x0, vmin);
    vacc3x0 = vmaxq_f32(vacc3x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);
    vacc1x1 = vmaxq_f32(vacc1x1, vmin);
    vacc2x1 = vmaxq_f32(vacc2x1, vmin);
    vacc3x1 = vmaxq_f32(vacc3x1, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      vst1q_f32(c1, vacc1x0);
      vst1q_f32(c1 + 4, vacc1x1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1q_f32(c2, vacc2x0);
      vst1q_f32(c2 + 4, vacc2x1);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1q_f32(c3, vacc3x0);
      vst1q_f32(c3 + 4, vacc3x1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);

      nc -= 8;

    } else {
      if (nc & 4) {
        vst1q_f32(c0, vacc0x0); c0 += 4;
        vst1q_f32(c1, vacc1x0); c1 += 4;
        vst1q_f32(c2, vacc2x0); c2 += 4;
        vst1q_f32(c3, vacc3x0); c3 += 4;

        vacc0x0 = vacc0x1;
        vacc1x0 = vacc1x1;
        vacc2x0 = vacc2x1;
        vacc3x0 = vacc3x1;
      }
      float32x2_t vacc0 = vget_low_f32(vacc0x0);
      float32x2_t vacc1 = vget_low_f32(vacc1x0);
      float32x2_t vacc2 = vget_low_f32(vacc2x0);
      float32x2_t vacc3 = vget_low_f32(vacc3x0);
      if (nc & 2) {
        vst1_f32(c0, vacc0); c0 += 2;
        vst1_f32(c1, vacc1); c1 += 2;
        vst1_f32(c2, vacc2); c2 += 2;
        vst1_f32(c3, vacc3); c3 += 2;

        vacc0 = vget_high_f32(vacc0x0);
        vacc1 = vget_high_f32(vacc1x0);
        vacc2 = vget_high_f32(vacc2x0);
        vacc3 = vget_high_f32(vacc3x0);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vacc0, 0);
        vst1_lane_f32(c1, vacc1, 0);
        vst1_lane_f32(c2, vacc2, 0);
        vst1_lane_f32(c3, vacc3, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc4w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
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
  const int32x4_t vminus_kernel_zero_point = vld1q_dup_s32(&params->scalar.minus_kernel_zero_point);
  const uint16x8_t vmask = vmovq_n_u16(UINT16_C(0xF));

  do {
    float32x4_t vacc0x0 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vacc0x1 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vacc1x0 = vacc0x0;
    float32x4_t vacc1x1 = vacc0x1;
    float32x4_t vacc2x0 = vacc0x0;
    float32x4_t vacc2x1 = vacc0x1;
    float32x4_t vacc3x0 = vacc0x0;
    float32x4_t vacc3x1 = vacc0x1;
    float32x4_t vacc4x0 = vacc0x0;
    float32x4_t vacc4x1 = vacc0x1;
    float32x4_t vacc5x0 = vacc0x0;
    float32x4_t vacc5x1 = vacc0x1;

    size_t k = kc;
    if XNN_LIKELY(k >= 4 * sizeof(float)) {
      do {
        const float32x4_t va0 = vld1q_f32(a0); a0 += 4;
        const float32x4_t va1 = vld1q_f32(a1); a1 += 4;
        const float32x4_t va2 = vld1q_f32(a2); a2 += 4;
        const float32x4_t va3 = vld1q_f32(a3); a3 += 4;
        const float32x4_t va4 = vld1q_f32(a4); a4 += 4;
        const float32x4_t va5 = vld1q_f32(a5); a5 += 4;


        const uint8x16_t vw01234567c0123 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint16x8_t vxw01234567c01 = vmovl_u8(vget_low_u8(vw01234567c0123));
        const uint16x8_t vxw01234567c23 = vmovl_u8(vget_high_u8(vw01234567c0123));
        const uint16x8_t vxw01234567c0 = vandq_u16(vxw01234567c01, vmask);
        const uint16x8_t vxw01234567c1 = vshrq_n_u16(vxw01234567c01, 4);
        const uint16x8_t vxw01234567c2 = vandq_u16(vxw01234567c23, vmask);
        const uint16x8_t vxw01234567c3 = vshrq_n_u16(vxw01234567c23, 4);
        const int32x4_t vxw0123c0 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c0)));
        const int32x4_t vxw4567c0 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c0)));
        const float32x4_t vb0123c0 = vcvtq_f32_s32(vxw0123c0);
        const float32x4_t vb4567c0 = vcvtq_f32_s32(vxw4567c0);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c0, vget_low_f32(va0), 0);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c0, vget_low_f32(va1), 0);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c0, vget_low_f32(va2), 0);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c0, vget_low_f32(va3), 0);
        vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0123c0, vget_low_f32(va4), 0);
        vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0123c0, vget_low_f32(va5), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c0, vget_low_f32(va0), 0);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c0, vget_low_f32(va1), 0);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c0, vget_low_f32(va2), 0);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c0, vget_low_f32(va3), 0);
        vacc4x1 = vfmaq_lane_f32(vacc4x1, vb4567c0, vget_low_f32(va4), 0);
        vacc5x1 = vfmaq_lane_f32(vacc5x1, vb4567c0, vget_low_f32(va5), 0);

        const int32x4_t vxw0123c1 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c1)));
        const int32x4_t vxw4567c1 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c1)));
        const float32x4_t vb0123c1 = vcvtq_f32_s32(vxw0123c1);
        const float32x4_t vb4567c1 = vcvtq_f32_s32(vxw4567c1);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c1, vget_low_f32(va0), 1);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c1, vget_low_f32(va1), 1);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c1, vget_low_f32(va2), 1);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c1, vget_low_f32(va3), 1);
        vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0123c1, vget_low_f32(va4), 1);
        vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0123c1, vget_low_f32(va5), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c1, vget_low_f32(va0), 1);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c1, vget_low_f32(va1), 1);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c1, vget_low_f32(va2), 1);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c1, vget_low_f32(va3), 1);
        vacc4x1 = vfmaq_lane_f32(vacc4x1, vb4567c1, vget_low_f32(va4), 1);
        vacc5x1 = vfmaq_lane_f32(vacc5x1, vb4567c1, vget_low_f32(va5), 1);

        const int32x4_t vxw0123c2 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c2)));
        const int32x4_t vxw4567c2 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c2)));
        const float32x4_t vb0123c2 = vcvtq_f32_s32(vxw0123c2);
        const float32x4_t vb4567c2 = vcvtq_f32_s32(vxw4567c2);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c2, vget_high_f32(va0), 0);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c2, vget_high_f32(va1), 0);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c2, vget_high_f32(va2), 0);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c2, vget_high_f32(va3), 0);
        vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0123c2, vget_high_f32(va4), 0);
        vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0123c2, vget_high_f32(va5), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c2, vget_high_f32(va0), 0);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c2, vget_high_f32(va1), 0);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c2, vget_high_f32(va2), 0);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c2, vget_high_f32(va3), 0);
        vacc4x1 = vfmaq_lane_f32(vacc4x1, vb4567c2, vget_high_f32(va4), 0);
        vacc5x1 = vfmaq_lane_f32(vacc5x1, vb4567c2, vget_high_f32(va5), 0);

        const int32x4_t vxw0123c3 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c3)));
        const int32x4_t vxw4567c3 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c3)));
        const float32x4_t vb0123c3 = vcvtq_f32_s32(vxw0123c3);
        const float32x4_t vb4567c3 = vcvtq_f32_s32(vxw4567c3);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c3, vget_high_f32(va0), 1);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c3, vget_high_f32(va1), 1);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c3, vget_high_f32(va2), 1);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c3, vget_high_f32(va3), 1);
        vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0123c3, vget_high_f32(va4), 1);
        vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0123c3, vget_high_f32(va5), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c3, vget_high_f32(va0), 1);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c3, vget_high_f32(va1), 1);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c3, vget_high_f32(va2), 1);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c3, vget_high_f32(va3), 1);
        vacc4x1 = vfmaq_lane_f32(vacc4x1, vb4567c3, vget_high_f32(va4), 1);
        vacc5x1 = vfmaq_lane_f32(vacc5x1, vb4567c3, vget_high_f32(va5), 1);
        k -= 4 * sizeof(float);
      } while (k >= 4 * sizeof(float));
    }

    if XNN_UNLIKELY(k != 0) {
      if XNN_UNLIKELY(k & (2 * sizeof(float))) {
        const float32x2_t va0 = vld1_f32(a0); a0 += 2;
        const float32x2_t va1 = vld1_f32(a1); a1 += 2;
        const float32x2_t va2 = vld1_f32(a2); a2 += 2;
        const float32x2_t va3 = vld1_f32(a3); a3 += 2;
        const float32x2_t va4 = vld1_f32(a4); a4 += 2;
        const float32x2_t va5 = vld1_f32(a5); a5 += 2;


        const uint8x8_t vw01234567c01 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const uint16x8_t vxw01234567c01 = vmovl_u8(vw01234567c01);
        const uint16x8_t vxw01234567c0 = vandq_u16(vxw01234567c01, vmask);
        const uint16x8_t vxw01234567c1 = vshrq_n_u16(vxw01234567c01, 4);
        const int32x4_t vxw0123c0 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c0)));
        const int32x4_t vxw4567c0 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c0)));
        const float32x4_t vb0123c0 = vcvtq_f32_s32(vxw0123c0);
        const float32x4_t vb4567c0 = vcvtq_f32_s32(vxw4567c0);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c0, va0, 0);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c0, va1, 0);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c0, va2, 0);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c0, va3, 0);
        vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0123c0, va4, 0);
        vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0123c0, va5, 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c0, va0, 0);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c0, va1, 0);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c0, va2, 0);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c0, va3, 0);
        vacc4x1 = vfmaq_lane_f32(vacc4x1, vb4567c0, va4, 0);
        vacc5x1 = vfmaq_lane_f32(vacc5x1, vb4567c0, va5, 0);

        const int32x4_t vxw0123c1 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567c1)));
        const int32x4_t vxw4567c1 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567c1)));
        const float32x4_t vb0123c1 = vcvtq_f32_s32(vxw0123c1);
        const float32x4_t vb4567c1 = vcvtq_f32_s32(vxw4567c1);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c1, va0, 1);
        vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c1, va1, 1);
        vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c1, va2, 1);
        vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c1, va3, 1);
        vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0123c1, va4, 1);
        vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0123c1, va5, 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c1, va0, 1);
        vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c1, va1, 1);
        vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c1, va2, 1);
        vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c1, va3, 1);
        vacc4x1 = vfmaq_lane_f32(vacc4x1, vb4567c1, va4, 1);
        vacc5x1 = vfmaq_lane_f32(vacc5x1, vb4567c1, va5, 1);
      }
      if XNN_UNLIKELY(k & (1 * sizeof(float))) {
        const float32x4_t va0 = vld1q_dup_f32(a0); a0 += 1;
        const float32x4_t va1 = vld1q_dup_f32(a1); a1 += 1;
        const float32x4_t va2 = vld1q_dup_f32(a2); a2 += 1;
        const float32x4_t va3 = vld1q_dup_f32(a3); a3 += 1;
        const float32x4_t va4 = vld1q_dup_f32(a4); a4 += 1;
        const float32x4_t va5 = vld1q_dup_f32(a5); a5 += 1;

        const uint8x8_t vw01234567 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const uint16x8_t vxw01234567 = vmovl_u8(vw01234567);
        const int32x4_t vxw0123 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_low_u16(vxw01234567)));
        const int32x4_t vxw4567 = vaddw_s16(vminus_kernel_zero_point, vreinterpret_s16_u16(vget_high_u16(vxw01234567)));
        const float32x4_t vb0123 = vcvtq_f32_s32(vxw0123);
        const float32x4_t vb4567 = vcvtq_f32_s32(vxw4567);

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123);
        vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123);
        vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123);
        vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123);
        vacc4x0 = vfmaq_f32(vacc4x0, va4, vb0123);
        vacc5x0 = vfmaq_f32(vacc5x0, va5, vb0123);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567);
        vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567);
        vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567);
        vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567);
        vacc4x1 = vfmaq_f32(vacc4x1, va4, vb4567);
        vacc5x1 = vfmaq_f32(vacc5x1, va5, vb4567);
      }
    }
    const float32x4_t vscale0123 = vld1q_f32(w); w = ((const float*) w + 4);
    const float32x4_t vscale4567 = vld1q_f32(w); w = ((const float*) w + 4);
    vacc0x0 = vmulq_f32(vacc0x0, vscale0123);
    vacc1x0 = vmulq_f32(vacc1x0, vscale0123);
    vacc2x0 = vmulq_f32(vacc2x0, vscale0123);
    vacc3x0 = vmulq_f32(vacc3x0, vscale0123);
    vacc4x0 = vmulq_f32(vacc4x0, vscale0123);
    vacc5x0 = vmulq_f32(vacc5x0, vscale0123);
    vacc0x1 = vmulq_f32(vacc0x1, vscale4567);
    vacc1x1 = vmulq_f32(vacc1x1, vscale4567);
    vacc2x1 = vmulq_f32(vacc2x1, vscale4567);
    vacc3x1 = vmulq_f32(vacc3x1, vscale4567);
    vacc4x1 = vmulq_f32(vacc4x1, vscale4567);
    vacc5x1 = vmulq_f32(vacc5x1, vscale4567);
    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc1x0 = vminq_f32(vacc1x0, vmax);
    vacc2x0 = vminq_f32(vacc2x0, vmax);
    vacc3x0 = vminq_f32(vacc3x0, vmax);
    vacc4x0 = vminq_f32(vacc4x0, vmax);
    vacc5x0 = vminq_f32(vacc5x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);
    vacc1x1 = vminq_f32(vacc1x1, vmax);
    vacc2x1 = vminq_f32(vacc2x1, vmax);
    vacc3x1 = vminq_f32(vacc3x1, vmax);
    vacc4x1 = vminq_f32(vacc4x1, vmax);
    vacc5x1 = vminq_f32(vacc5x1, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc1x0 = vmaxq_f32(vacc1x0, vmin);
    vacc2x0 = vmaxq_f32(vacc2x0, vmin);
    vacc3x0 = vmaxq_f32(vacc3x0, vmin);
    vacc4x0 = vmaxq_f32(vacc4x0, vmin);
    vacc5x0 = vmaxq_f32(vacc5x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);
    vacc1x1 = vmaxq_f32(vacc1x1, vmin);
    vacc2x1 = vmaxq_f32(vacc2x1, vmin);
    vacc3x1 = vmaxq_f32(vacc3x1, vmin);
    vacc4x1 = vmaxq_f32(vacc4x1, vmin);
    vacc5x1 = vmaxq_f32(vacc5x1, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      vst1q_f32(c1, vacc1x0);
      vst1q_f32(c1 + 4, vacc1x1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1q_f32(c2, vacc2x0);
      vst1q_f32(c2 + 4, vacc2x1);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1q_f32(c3, vacc3x0);
      vst1q_f32(c3 + 4, vacc3x1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      vst1q_f32(c4, vacc4x0);
      vst1q_f32(c4 + 4, vacc4x1);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      vst1q_f32(c5, vacc5x0);
      vst1q_f32(c5 + 4, vacc5x1);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);

      nc -= 8;

    } else {
      if (nc & 4) {
        vst1q_f32(c0, vacc0x0); c0 += 4;
        vst1q_f32(c1, vacc1x0); c1 += 4;
        vst1q_f32(c2, vacc2x0); c2 += 4;
        vst1q_f32(c3, vacc3x0); c3 += 4;
        vst1q_f32(c4, vacc4x0); c4 += 4;
        vst1q_f32(c5, vacc5x0); c5 += 4;

        vacc0x0 = vacc0x1;
        vacc1x0 = vacc1x1;
        vacc2x0 = vacc2x1;
        vacc3x0 = vacc3x1;
        vacc4x0 = vacc4x1;
        vacc5x0 = vacc5x1;
      }
      float32x2_t vacc0 = vget_low_f32(vacc0x0);
      float32x2_t vacc1 = vget_low_f32(vacc1x0);
      float32x2_t vacc2 = vget_low_f32(vacc2x0);
      float32x2_t vacc3 = vget_low_f32(vacc3x0);
      float32x2_t vacc4 = vget_low_f32(vacc4x0);
      float32x2_t vacc5 = vget_low_f32(vacc5x0);
      if (nc & 2) {
        vst1_f32(c0, vacc0); c0 += 2;
        vst1_f32(c1, vacc1); c1 += 2;
        vst1_f32(c2, vacc2); c2 += 2;
        vst1_f32(c3, vacc3); c3 += 2;
        vst1_f32(c4, vacc4); c4 += 2;
        vst1_f32(c5, vacc5); c5 += 2;

        vacc0 = vget_high_f32(vacc0x0);
        vacc1 = vget_high_f32(vacc1x0);
        vacc2 = vget_high_f32(vacc2x0);
        vacc3 = vget_high_f32(vacc3x0);
        vacc4 = vget_high_f32(vacc4x0);
        vacc5 = vget_high_f32(vacc5x0);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vacc0, 0);
        vst1_lane_f32(c1, vacc1, 0);
        vst1_lane_f32(c2, vacc2, 0);
        vst1_lane_f32(c3, vacc3, 0);
        vst1_lane_f32(c4, vacc4, 0);
        vst1_lane_f32(c5, vacc5, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
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
    float32x4_t vacc0x0 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vacc0x1 = vld1q_f32(w); w = (const float*) w + 4;

    size_t k = kc;
    for (; k >= 2 * sizeof(float); k -= 2 * sizeof(float)) {
      const float32x2_t va0 = vld1_f32(a0); a0 += 2;

      const int8x8_t vw01234567c0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vw01234567c1 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int16x8_t vxw01234567c0 = vmovl_s8(vw01234567c0);
      const int16x8_t vxw01234567c1 = vmovl_s8(vw01234567c1);
      const int32x4_t vxw0123c0 = vmovl_s16(vget_low_s16(vxw01234567c0));
      const int32x4_t vxw4567c0 = vmovl_s16(vget_high_s16(vxw01234567c0));
      const int32x4_t vxw0123c1 = vmovl_s16(vget_low_s16(vxw01234567c1));
      const int32x4_t vxw4567c1 = vmovl_s16(vget_high_s16(vxw01234567c1));
      const float32x4_t vb0123c0 = vcvtq_f32_s32(vxw0123c0);
      const float32x4_t vb0123c1 = vcvtq_f32_s32(vxw0123c1);
      const float32x4_t vb4567c0 = vcvtq_f32_s32(vxw4567c0);
      const float32x4_t vb4567c1 = vcvtq_f32_s32(vxw4567c1);

      vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c0, va0, 0);
      vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c0, va0, 0);
      vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c1, va0, 1);
      vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c1, va0, 1);
    }
    if XNN_UNLIKELY(k != 0) {
      const float32x4_t va0 = vld1q_dup_f32(a0); a0 += 1;

      const int8x8_t vw01230123 = vreinterpret_s8_u32(vld1_dup_u32(w)); w = (const int8_t*) w + 4;
      const int8x8_t vw45674567 = vreinterpret_s8_u32(vld1_dup_u32(w)); w = (const int8_t*) w + 4;
      const int16x8_t vxw01230123 = vmovl_s8(vw01230123);
      const int16x8_t vxw45674567 = vmovl_s8(vw45674567);
      const int32x4_t vxw0123 = vmovl_s16(vget_low_s16(vxw01230123));
      const int32x4_t vxw4567 = vmovl_s16(vget_low_s16(vxw45674567));
      const float32x4_t vb0123 = vcvtq_f32_s32(vxw0123);
      const float32x4_t vb4567 = vcvtq_f32_s32(vxw4567);

      vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123);
      vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567);
    }
    const float32x4_t vscale0123 = vld1q_f32(w); w = (const float*) w + 4;
    vacc0x0 = vmulq_f32(vacc0x0, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32(w); w = (const float*) w + 4;
    vacc0x1 = vmulq_f32(vacc0x1, vscale4567);
    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;

    } else {
      if (nc & 4) {
        vst1q_f32(c0, vacc0x0); c0 += 4;

        vacc0x0 = vacc0x1;
      }
      float32x2_t vacc0 = vget_low_f32(vacc0x0);
      if (nc & 2) {
        vst1_f32(c0, vacc0); c0 += 2;

        vacc0 = vget_high_f32(vacc0x0);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vacc0, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
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
    float32x4_t vacc0x0 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vacc0x1 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vacc1x0 = vacc0x0;
    float32x4_t vacc1x1 = vacc0x1;
    float32x4_t vacc2x0 = vacc0x0;
    float32x4_t vacc2x1 = vacc0x1;
    float32x4_t vacc3x0 = vacc0x0;
    float32x4_t vacc3x1 = vacc0x1;
    float32x4_t vacc4x0 = vacc0x0;
    float32x4_t vacc4x1 = vacc0x1;
    float32x4_t vacc5x0 = vacc0x0;
    float32x4_t vacc5x1 = vacc0x1;

    size_t k = kc;
    for (; k >= 2 * sizeof(float); k -= 2 * sizeof(float)) {
      const float32x2_t va0 = vld1_f32(a0); a0 += 2;
      const float32x2_t va1 = vld1_f32(a1); a1 += 2;
      const float32x2_t va2 = vld1_f32(a2); a2 += 2;
      const float32x2_t va3 = vld1_f32(a3); a3 += 2;
      const float32x2_t va4 = vld1_f32(a4); a4 += 2;
      const float32x2_t va5 = vld1_f32(a5); a5 += 2;

      const int8x8_t vw01234567c0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vw01234567c1 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int16x8_t vxw01234567c0 = vmovl_s8(vw01234567c0);
      const int16x8_t vxw01234567c1 = vmovl_s8(vw01234567c1);
      const int32x4_t vxw0123c0 = vmovl_s16(vget_low_s16(vxw01234567c0));
      const int32x4_t vxw4567c0 = vmovl_s16(vget_high_s16(vxw01234567c0));
      const int32x4_t vxw0123c1 = vmovl_s16(vget_low_s16(vxw01234567c1));
      const int32x4_t vxw4567c1 = vmovl_s16(vget_high_s16(vxw01234567c1));
      const float32x4_t vb0123c0 = vcvtq_f32_s32(vxw0123c0);
      const float32x4_t vb0123c1 = vcvtq_f32_s32(vxw0123c1);
      const float32x4_t vb4567c0 = vcvtq_f32_s32(vxw4567c0);
      const float32x4_t vb4567c1 = vcvtq_f32_s32(vxw4567c1);

      vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c0, va0, 0);
      vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c0, va1, 0);
      vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c0, va2, 0);
      vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c0, va3, 0);
      vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0123c0, va4, 0);
      vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0123c0, va5, 0);
      vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c0, va0, 0);
      vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c0, va1, 0);
      vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c0, va2, 0);
      vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c0, va3, 0);
      vacc4x1 = vfmaq_lane_f32(vacc4x1, vb4567c0, va4, 0);
      vacc5x1 = vfmaq_lane_f32(vacc5x1, vb4567c0, va5, 0);
      vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c1, va0, 1);
      vacc1x0 = vfmaq_lane_f32(vacc1x0, vb0123c1, va1, 1);
      vacc2x0 = vfmaq_lane_f32(vacc2x0, vb0123c1, va2, 1);
      vacc3x0 = vfmaq_lane_f32(vacc3x0, vb0123c1, va3, 1);
      vacc4x0 = vfmaq_lane_f32(vacc4x0, vb0123c1, va4, 1);
      vacc5x0 = vfmaq_lane_f32(vacc5x0, vb0123c1, va5, 1);
      vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c1, va0, 1);
      vacc1x1 = vfmaq_lane_f32(vacc1x1, vb4567c1, va1, 1);
      vacc2x1 = vfmaq_lane_f32(vacc2x1, vb4567c1, va2, 1);
      vacc3x1 = vfmaq_lane_f32(vacc3x1, vb4567c1, va3, 1);
      vacc4x1 = vfmaq_lane_f32(vacc4x1, vb4567c1, va4, 1);
      vacc5x1 = vfmaq_lane_f32(vacc5x1, vb4567c1, va5, 1);
    }
    if XNN_UNLIKELY(k != 0) {
      const float32x4_t va0 = vld1q_dup_f32(a0); a0 += 1;
      const float32x4_t va1 = vld1q_dup_f32(a1); a1 += 1;
      const float32x4_t va2 = vld1q_dup_f32(a2); a2 += 1;
      const float32x4_t va3 = vld1q_dup_f32(a3); a3 += 1;
      const float32x4_t va4 = vld1q_dup_f32(a4); a4 += 1;
      const float32x4_t va5 = vld1q_dup_f32(a5); a5 += 1;

      const int8x8_t vw01230123 = vreinterpret_s8_u32(vld1_dup_u32(w)); w = (const int8_t*) w + 4;
      const int8x8_t vw45674567 = vreinterpret_s8_u32(vld1_dup_u32(w)); w = (const int8_t*) w + 4;
      const int16x8_t vxw01230123 = vmovl_s8(vw01230123);
      const int16x8_t vxw45674567 = vmovl_s8(vw45674567);
      const int32x4_t vxw0123 = vmovl_s16(vget_low_s16(vxw01230123));
      const int32x4_t vxw4567 = vmovl_s16(vget_low_s16(vxw45674567));
      const float32x4_t vb0123 = vcvtq_f32_s32(vxw0123);
      const float32x4_t vb4567 = vcvtq_f32_s32(vxw4567);

      vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123);
      vacc1x0 = vfmaq_f32(vacc1x0, va1, vb0123);
      vacc2x0 = vfmaq_f32(vacc2x0, va2, vb0123);
      vacc3x0 = vfmaq_f32(vacc3x0, va3, vb0123);
      vacc4x0 = vfmaq_f32(vacc4x0, va4, vb0123);
      vacc5x0 = vfmaq_f32(vacc5x0, va5, vb0123);
      vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567);
      vacc1x1 = vfmaq_f32(vacc1x1, va1, vb4567);
      vacc2x1 = vfmaq_f32(vacc2x1, va2, vb4567);
      vacc3x1 = vfmaq_f32(vacc3x1, va3, vb4567);
      vacc4x1 = vfmaq_f32(vacc4x1, va4, vb4567);
      vacc5x1 = vfmaq_f32(vacc5x1, va5, vb4567);
    }
    const float32x4_t vscale0123 = vld1q_f32(w); w = (const float*) w + 4;
    vacc0x0 = vmulq_f32(vacc0x0, vscale0123);
    vacc1x0 = vmulq_f32(vacc1x0, vscale0123);
    vacc2x0 = vmulq_f32(vacc2x0, vscale0123);
    vacc3x0 = vmulq_f32(vacc3x0, vscale0123);
    vacc4x0 = vmulq_f32(vacc4x0, vscale0123);
    vacc5x0 = vmulq_f32(vacc5x0, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32(w); w = (const float*) w + 4;
    vacc0x1 = vmulq_f32(vacc0x1, vscale4567);
    vacc1x1 = vmulq_f32(vacc1x1, vscale4567);
    vacc2x1 = vmulq_f32(vacc2x1, vscale4567);
    vacc3x1 = vmulq_f32(vacc3x1, vscale4567);
    vacc4x1 = vmulq_f32(vacc4x1, vscale4567);
    vacc5x1 = vmulq_f32(vacc5x1, vscale4567);
    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc1x0 = vminq_f32(vacc1x0, vmax);
    vacc2x0 = vminq_f32(vacc2x0, vmax);
    vacc3x0 = vminq_f32(vacc3x0, vmax);
    vacc4x0 = vminq_f32(vacc4x0, vmax);
    vacc5x0 = vminq_f32(vacc5x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);
    vacc1x1 = vminq_f32(vacc1x1, vmax);
    vacc2x1 = vminq_f32(vacc2x1, vmax);
    vacc3x1 = vminq_f32(vacc3x1, vmax);
    vacc4x1 = vminq_f32(vacc4x1, vmax);
    vacc5x1 = vminq_f32(vacc5x1, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc1x0 = vmaxq_f32(vacc1x0, vmin);
    vacc2x0 = vmaxq_f32(vacc2x0, vmin);
    vacc3x0 = vmaxq_f32(vacc3x0, vmin);
    vacc4x0 = vmaxq_f32(vacc4x0, vmin);
    vacc5x0 = vmaxq_f32(vacc5x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);
    vacc1x1 = vmaxq_f32(vacc1x1, vmin);
    vacc2x1 = vmaxq_f32(vacc2x1, vmin);
    vacc3x1 = vmaxq_f32(vacc3x1, vmin);
    vacc4x1 = vmaxq_f32(vacc4x1, vmin);
    vacc5x1 = vmaxq_f32(vacc5x1, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      vst1q_f32(c1, vacc1x0);
      vst1q_f32(c1 + 4, vacc1x1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      vst1q_f32(c2, vacc2x0);
      vst1q_f32(c2 + 4, vacc2x1);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      vst1q_f32(c3, vacc3x0);
      vst1q_f32(c3 + 4, vacc3x1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      vst1q_f32(c4, vacc4x0);
      vst1q_f32(c4 + 4, vacc4x1);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      vst1q_f32(c5, vacc5x0);
      vst1q_f32(c5 + 4, vacc5x1);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);

      nc -= 8;

    } else {
      if (nc & 4) {
        vst1q_f32(c0, vacc0x0); c0 += 4;
        vst1q_f32(c1, vacc1x0); c1 += 4;
        vst1q_f32(c2, vacc2x0); c2 += 4;
        vst1q_f32(c3, vacc3x0); c3 += 4;
        vst1q_f32(c4, vacc4x0); c4 += 4;
        vst1q_f32(c5, vacc5x0); c5 += 4;

        vacc0x0 = vacc0x1;
        vacc1x0 = vacc1x1;
        vacc2x0 = vacc2x1;
        vacc3x0 = vacc3x1;
        vacc4x0 = vacc4x1;
        vacc5x0 = vacc5x1;
      }
      float32x2_t vacc0 = vget_low_f32(vacc0x0);
      float32x2_t vacc1 = vget_low_f32(vacc1x0);
      float32x2_t vacc2 = vget_low_f32(vacc2x0);
      float32x2_t vacc3 = vget_low_f32(vacc3x0);
      float32x2_t vacc4 = vget_low_f32(vacc4x0);
      float32x2_t vacc5 = vget_low_f32(vacc5x0);
      if (nc & 2) {
        vst1_f32(c0, vacc0); c0 += 2;
        vst1_f32(c1, vacc1); c1 += 2;
        vst1_f32(c2, vacc2); c2 += 2;
        vst1_f32(c3, vacc3); c3 += 2;
        vst1_f32(c4, vacc4); c4 += 2;
        vst1_f32(c5, vacc5); c5 += 2;

        vacc0 = vget_high_f32(vacc0x0);
        vacc1 = vget_high_f32(vacc1x0);
        vacc2 = vget_high_f32(vacc2x0);
        vacc3 = vget_high_f32(vacc3x0);
        vacc4 = vget_high_f32(vacc4x0);
        vacc5 = vget_high_f32(vacc5x0);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vacc0, 0);
        vst1_lane_f32(c1, vacc1, 0);
        vst1_lane_f32(c2, vacc2, 0);
        vst1_lane_f32(c3, vacc3, 0);
        vst1_lane_f32(c4, vacc4, 0);
        vst1_lane_f32(c5, vacc5, 0);
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
    const float32x4_t vmin = vcombine_f32(vminmax.val[0], vminmax.val[0]);
    const float32x4_t vmax = vcombine_f32(vminmax.val[1], vminmax.val[1]);
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
    const float32x4_t vmin = vcombine_f32(vminmax.val[0], vminmax.val[0]);
    const float32x4_t vmax = vcombine_f32(vminmax.val[1], vminmax.val[1]);
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

void xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  const float32x4_t vsat_cutoff = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.sat_cutoff);
  const float32x4_t vminus_log2e = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.minus_log2e);

  const float32x4_t vmagic_bias = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.magic_bias);

  const float32x4_t vln2 = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.ln2);

  const float32x4_t vc6 = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.c6);
  const float32x4_t vc5 = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.c5);
  const float32x4_t vc4 = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.c4);
  const float32x4_t vc3 = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.c3);
  const float32x4_t vc2 = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.c2);

  const float32x4_t vone = vmovq_n_f32(1.0f);
  const float32x4_t vtwo = vmovq_n_f32(2.0f);

  const uint32x4_t vsign_mask = vmovq_n_u32(UINT32_C(0x80000000));

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(input); input += 4;
    const float32x4_t vx4567 = vld1q_f32(input); input += 4;
    const float32x4_t vx89AB = vld1q_f32(input); input += 4;
    const float32x4_t vxCDEF = vld1q_f32(input); input += 4;

    float32x4_t vz0123 = vabsq_f32(vx0123);
    float32x4_t vz4567 = vabsq_f32(vx4567);
    float32x4_t vz89AB = vabsq_f32(vx89AB);
    float32x4_t vzCDEF = vabsq_f32(vxCDEF);
    vz0123 = vminq_f32(vz0123, vsat_cutoff);
    vz4567 = vminq_f32(vz4567, vsat_cutoff);
    vz89AB = vminq_f32(vz89AB, vsat_cutoff);
    vzCDEF = vminq_f32(vzCDEF, vsat_cutoff);

    float32x4_t vn0123 = vfmaq_f32(vmagic_bias, vz0123, vminus_log2e);
    float32x4_t vn4567 = vfmaq_f32(vmagic_bias, vz4567, vminus_log2e);
    float32x4_t vn89AB = vfmaq_f32(vmagic_bias, vz89AB, vminus_log2e);
    float32x4_t vnCDEF = vfmaq_f32(vmagic_bias, vzCDEF, vminus_log2e);

    const float32x4_t vs0123 = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn0123), 23));
    const float32x4_t vs4567 = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn4567), 23));
    const float32x4_t vs89AB = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn89AB), 23));
    const float32x4_t vsCDEF = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vnCDEF), 23));

    vn0123 = vsubq_f32(vn0123, vmagic_bias);
    vn4567 = vsubq_f32(vn4567, vmagic_bias);
    vn89AB = vsubq_f32(vn89AB, vmagic_bias);
    vnCDEF = vsubq_f32(vnCDEF, vmagic_bias);

    const float32x4_t vt0123 = vfmaq_f32(vz0123, vn0123, vln2);
    const float32x4_t vt4567 = vfmaq_f32(vz4567, vn4567, vln2);
    const float32x4_t vt89AB = vfmaq_f32(vz89AB, vn89AB, vln2);
    const float32x4_t vtCDEF = vfmaq_f32(vzCDEF, vnCDEF, vln2);

    float32x4_t vp0123 = vfmaq_f32(vc5, vc6, vt0123);
    float32x4_t vp4567 = vfmaq_f32(vc5, vc6, vt4567);
    float32x4_t vp89AB = vfmaq_f32(vc5, vc6, vt89AB);
    float32x4_t vpCDEF = vfmaq_f32(vc5, vc6, vtCDEF);
    vp0123 = vfmaq_f32(vc4, vp0123, vt0123);
    vp0123 = vfmaq_f32(vc3, vp0123, vt0123);
    vp0123 = vfmaq_f32(vc2, vp0123, vt0123);
    vp4567 = vfmaq_f32(vc4, vp4567, vt4567);
    vp4567 = vfmaq_f32(vc3, vp4567, vt4567);
    vp4567 = vfmaq_f32(vc2, vp4567, vt4567);
    vp89AB = vfmaq_f32(vc4, vp89AB, vt89AB);
    vp89AB = vfmaq_f32(vc3, vp89AB, vt89AB);
    vp89AB = vfmaq_f32(vc2, vp89AB, vt89AB);
    vpCDEF = vfmaq_f32(vc4, vpCDEF, vtCDEF);
    vpCDEF = vfmaq_f32(vc3, vpCDEF, vtCDEF);
    vpCDEF = vfmaq_f32(vc2, vpCDEF, vtCDEF);
    vp0123 = vfmsq_f32(vtwo, vp0123, vt0123);
    vp4567 = vfmsq_f32(vtwo, vp4567, vt4567);
    vp89AB = vfmsq_f32(vtwo, vp89AB, vt89AB);
    vpCDEF = vfmsq_f32(vtwo, vpCDEF, vtCDEF);

    const float32x4_t vts0123 = vmulq_f32(vt0123, vs0123);
    const float32x4_t vsmo0123 = vsubq_f32(vs0123, vone);
    const float32x4_t vts4567 = vmulq_f32(vt4567, vs4567);
    const float32x4_t vsmo4567 = vsubq_f32(vs4567, vone);
    const float32x4_t vts89AB = vmulq_f32(vt89AB, vs89AB);
    const float32x4_t vsmo89AB = vsubq_f32(vs89AB, vone);
    const float32x4_t vtsCDEF = vmulq_f32(vtCDEF, vsCDEF);
    const float32x4_t vsmoCDEF = vsubq_f32(vsCDEF, vone);
    const float32x4_t vemo0123 = vfmsq_f32(vsmo0123, vp0123, vts0123);
    const float32x4_t vemo4567 = vfmsq_f32(vsmo4567, vp4567, vts4567);
    const float32x4_t vemo89AB = vfmsq_f32(vsmo89AB, vp89AB, vts89AB);
    const float32x4_t vemoCDEF = vfmsq_f32(vsmoCDEF, vpCDEF, vtsCDEF);

    const float32x4_t vepo0123 = vaddq_f32(vemo0123, vtwo);
    const float32x4_t vepo4567 = vaddq_f32(vemo4567, vtwo);
    const float32x4_t vepo89AB = vaddq_f32(vemo89AB, vtwo);
    const float32x4_t vepoCDEF = vaddq_f32(vemoCDEF, vtwo);

    float32x4_t vy0123 = vdivq_f32(vemo0123, vepo0123);
    float32x4_t vy4567 = vdivq_f32(vemo4567, vepo4567);
    float32x4_t vy89AB = vdivq_f32(vemo89AB, vepo89AB);
    float32x4_t vyCDEF = vdivq_f32(vemoCDEF, vepoCDEF);

    vy0123 = vbslq_f32(vsign_mask, vx0123, vy0123);
    vy4567 = vbslq_f32(vsign_mask, vx4567, vy4567);
    vy89AB = vbslq_f32(vsign_mask, vx89AB, vy89AB);
    vyCDEF = vbslq_f32(vsign_mask, vxCDEF, vyCDEF);

    vst1q_f32(output, vy0123); output += 4;
    vst1q_f32(output, vy4567); output += 4;
    vst1q_f32(output, vy89AB); output += 4;
    vst1q_f32(output, vyCDEF); output += 4;
  }

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;

    float32x4_t vz = vabsq_f32(vx);
    vz = vminq_f32(vz, vsat_cutoff);

    float32x4_t vn = vfmaq_f32(vmagic_bias, vz, vminus_log2e);

    const float32x4_t vs = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));

    vn = vsubq_f32(vn, vmagic_bias);

    const float32x4_t vt = vfmaq_f32(vz, vn, vln2);

    float32x4_t vp = vfmaq_f32(vc5, vc6, vt);
    vp = vfmaq_f32(vc4, vp, vt);
    vp = vfmaq_f32(vc3, vp, vt);
    vp = vfmaq_f32(vc2, vp, vt);
    vp = vfmsq_f32(vtwo, vp, vt);

    const float32x4_t vts = vmulq_f32(vt, vs);
    const float32x4_t vsmo = vsubq_f32(vs, vone);
    const float32x4_t vemo = vfmsq_f32(vsmo, vp, vts);

    const float32x4_t vepo = vaddq_f32(vemo, vtwo);

    float32x4_t vy = vdivq_f32(vemo, vepo);

    vy = vbslq_f32(vsign_mask, vx, vy);
    vst1q_f32(output, vy); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = vld1q_f32(input);

    float32x4_t vz = vabsq_f32(vx);
    vz = vminq_f32(vz, vsat_cutoff);

    float32x4_t vn = vfmaq_f32(vmagic_bias, vz, vminus_log2e);

    const float32x4_t vs = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));

    vn = vsubq_f32(vn, vmagic_bias);

    const float32x4_t vt = vfmaq_f32(vz, vn, vln2);

    float32x4_t vp = vfmaq_f32(vc5, vc6, vt);
    vp = vfmaq_f32(vc4, vp, vt);
    vp = vfmaq_f32(vc3, vp, vt);
    vp = vfmaq_f32(vc2, vp, vt);
    vp = vfmsq_f32(vtwo, vp, vt);

    const float32x4_t vts = vmulq_f32(vt, vs);
    const float32x4_t vsmo = vsubq_f32(vs, vone);
    const float32x4_t vemo = vfmsq_f32(vsmo, vp, vts);

    const float32x4_t vepo = vaddq_f32(vemo, vtwo);

    float32x4_t vy = vdivq_f32(vemo, vepo);

    vy = vbslq_f32(vsign_mask, vx, vy);

    float32x2_t vy_low = vget_low_f32(vy);

    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vy_low); output += 2;
      vy_low = vget_high_f32(vy);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vy_low, 0);
    }
  }
}
