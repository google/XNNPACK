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


void xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_1x4_acc2(
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

  float* o0 = output;

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i2 = zero;
    }

    float32x4_t vi0x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi1x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi2x0123 = vmovq_n_f32(0.0f);

    float32x4_t vi0x4567 = vld1q_f32(i0); i0 += 4;
    float32x4_t vi1x4567 = vld1q_f32(i1); i1 += 4;
    float32x4_t vi2x4567 = vld1q_f32(i2); i2 += 4;

    size_t w = input_width;
    for (; w > 4 * sizeof(float); w -= 4 * sizeof(float)) {
      float32x4_t vo0p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);

      const float32x4_t vi0x89AB = vld1q_f32(i0); i0 += 4;
      const float32x4_t vi1x89AB = vld1q_f32(i1); i1 += 4;
      const float32x4_t vi2x89AB = vld1q_f32(i2); i2 += 4;

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x4567, vget_high_f32(vw0123), 0);

      float32x4_t vo0p1 = vmulq_lane_f32(vi1x4567, vget_low_f32(vw4567), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x4567, vw89, 0);

      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi0x3456, vget_low_f32(vw0123), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x3456, vget_low_f32(vw4567), 0);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi2x3456, vget_high_f32(vw4567), 1);

      vi0x0123 = vi0x4567;
      vi1x0123 = vi1x4567;
      vi2x0123 = vi2x4567;

      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vi0x89AB, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vi1x89AB, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vi2x89AB, 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x5678, vget_high_f32(vw0123), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi1x5678, vget_high_f32(vw4567), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x5678, vw89, 1);

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;

      vo0p0 = vaddq_f32(vo0p0, vo0p1);

      float32x4_t vo0 = vmaxq_f32(vo0p0, vmin);

      vo0 = vminq_f32(vo0, vmax);

      vst1q_f32(o0, vo0); o0 += 4;
    }
    // Always process the last block of 1..4 pixels.
    assert(w >= 1 * sizeof(float));
    assert(w <= 4 * sizeof(float));
    {
      float32x4_t vo0p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);

      vi0x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi0x4567)));
      vi1x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi1x4567)));
      vi2x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi2x4567)));

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x4567, vget_high_f32(vw0123), 0);

      float32x4_t vo0p1 = vmulq_lane_f32(vi1x4567, vget_low_f32(vw4567), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x4567, vw89, 0);

      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi0x3456, vget_low_f32(vw0123), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x3456, vget_low_f32(vw4567), 0);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi2x3456, vget_high_f32(vw4567), 1);

      const float32x4_t vzero = vmovq_n_f32(0.0f);
      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vzero, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vzero, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vzero, 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x5678, vget_high_f32(vw0123), 1);

      vo0p1 = vmlaq_lane_f32(vo0p1, vi1x5678, vget_high_f32(vw4567), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x5678, vw89, 1);

      vo0p0 = vaddq_f32(vo0p0, vo0p1);

      float32x4_t vo0 = vmaxq_f32(vo0p0, vmin);

      vo0 = vminq_f32(vo0, vmax);

      if XNN_LIKELY(w == 4 * sizeof(float)) {
        vst1q_f32(o0, vo0); o0 += 4;
      } else {
        float32x2_t vo0_lo = vget_low_f32(vo0);
        if (w & (2 * sizeof(float))) {
          vst1_f32(o0, vo0_lo); o0 += 2;

          vo0_lo = vget_high_f32(vo0);
        }
        if (w & (1 * sizeof(float))) {
          vst1_lane_f32(o0, vo0_lo, 0); o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i1 - input_decrement);
    i1 = (const float*) ((uintptr_t) i2 - input_decrement);
    i2 = (const float*) ((uintptr_t) i1 + input_width);


  } while (--output_height != 0);
}
