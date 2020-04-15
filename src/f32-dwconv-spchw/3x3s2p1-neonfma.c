// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>


void xnn_f32_dwconv_spchw_ukernel_3x3s2p1__neonfma(
    size_t m,
    size_t n,
    const float* input,
    const float* weights,
    float* output,
    size_t input_tuple_stride,
    size_t output_tuple_stride,
    size_t input_width_stride,
    size_t output_width_stride,
    const union xnn_f32_spchw_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);

  const uint32x4_t vmask_even = vld1q_u32(params->neon.mask_even);
  const uint32x4_t vmask_odd  = vld1q_u32(params->neon.mask_odd);
  const float32x4_t vmax = vld1q_dup_f32(&params->neon.max);
  const float32x4_t vmin = vld1q_dup_f32(&params->neon.min);

  const size_t input_width_increment = input_width_stride * 2 - n / 8 * input_tuple_stride * 2;
  const size_t output_width_increment = output_width_stride - n / 8 * output_tuple_stride;

  // No vertical padding.
  const float* i0 = input;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_width_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width_stride);

  const float32x4_t vw0123 = vld1q_f32(weights);
  const float32x4_t vw4567 = vld1q_f32(weights + 4);
  const float32x2_t vw89 = vld1_f32(weights + 8);

  do {
    float32x4_t vi0x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi1x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi2x0123 = vmovq_n_f32(0.0f);

    size_t k = n;
    for (; k >= 8; k -= 8) {
      float32x4_t vo468Ap0 = vdupq_laneq_f32(vw0123, 0);

      const float32x4_t vi0x4567 = vld1q_f32(i0); i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      const float32x4_t vi1x4567 = vld1q_f32(i1); i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      const float32x4_t vi2x4567 = vld1q_f32(i2); i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);

      const float32x4_t vi0x89AB = vld1q_f32(i0); i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      const float32x4_t vi1x89AB = vld1q_f32(i1); i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      const float32x4_t vi2x89AB = vld1q_f32(i2); i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);

      const float32x4_t vi0x468A = vuzp1q_f32(vi0x4567, vi0x89AB);
      const float32x4_t vi0x579B = vuzp2q_f32(vi0x4567, vi0x89AB);
      const float32x4_t vi1x468A = vuzp1q_f32(vi1x4567, vi1x89AB);
      const float32x4_t vi1x579B = vuzp2q_f32(vi1x4567, vi1x89AB);
      const float32x4_t vi2x468A = vuzp1q_f32(vi2x4567, vi2x89AB);
      const float32x4_t vi2x579B = vuzp2q_f32(vi2x4567, vi2x89AB);
      // add bias only to first row, it will then get added
      // to the final result
      // multiply each row by corresponding row of center column of filter
      vo468Ap0 = vfmaq_laneq_f32(vo468Ap0, vi0x468A, vw0123, 2);
      float32x4_t vo468Ap1 = vmulq_laneq_f32(vi1x468A, vw4567, 1);
      float32x4_t vo468Ap2 = vmulq_lane_f32(vi2x468A, vw89, 0);

      // grab the values corresponding the left filter tap
      const float32x4_t vi0x3579 = vextq_f32(vi0x0123, vi0x579B, 3);
      const float32x4_t vi1x3579 = vextq_f32(vi1x0123, vi1x579B, 3);
      const float32x4_t vi2x3579 = vextq_f32(vi2x0123, vi2x579B, 3);

      vi0x0123 = vi0x89AB;
      vi1x0123 = vi1x89AB;
      vi2x0123 = vi2x89AB;

      vo468Ap0 = vfmaq_laneq_f32(vo468Ap0, vi0x3579, vw0123, 1);
      vo468Ap1 = vfmaq_laneq_f32(vo468Ap1, vi1x3579, vw4567, 0);
      vo468Ap2 = vfmaq_laneq_f32(vo468Ap2, vi2x3579, vw4567, 3);

      // Do multiplication by right filter tap.
      vo468Ap0 = vfmaq_laneq_f32(vo468Ap0, vi0x579B, vw0123, 3);
      vo468Ap1 = vfmaq_laneq_f32(vo468Ap1, vi1x579B, vw4567, 2);
      vo468Ap2 = vfmaq_lane_f32 (vo468Ap2, vi2x579B, vw89, 1);

      // Add up across rows to get the final outputs.
      float32x4_t vo = vaddq_f32(vo468Ap0, vo468Ap1);
      vo = vaddq_f32(vo, vo468Ap2);

      vo = vmaxq_f32(vo, vmin);
      vo = vminq_f32(vo, vmax);

      vst1q_f32(output, vo); output = (float*) ((uintptr_t) output + output_tuple_stride);
    }
    // Last block has 0-7 pixels to process.
    assert(k < 8);
    if XNN_LIKELY(k != 0) {
      float32x4_t vo468Ap0 = vdupq_laneq_f32(vw0123, 0);

      const float32x4_t vi0x4567 = vld1q_f32(i0);
      const float32x4_t vi1x4567 = vld1q_f32(i1);
      const float32x4_t vi2x4567 = vld1q_f32(i2);

      const float32x4_t vi0x89AB = vld1q_f32((const float*) ((uintptr_t) i0 + input_tuple_stride));
      const float32x4_t vi1x89AB = vld1q_f32((const float*) ((uintptr_t) i1 + input_tuple_stride));
      const float32x4_t vi2x89AB = vld1q_f32((const float*) ((uintptr_t) i2 + input_tuple_stride));

      const float32x4_t vi0x468A = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vuzp1q_f32(vi0x4567, vi0x89AB))));
      const float32x4_t vi0x579B = vreinterpretq_f32_u32(vandq_u32(vmask_odd,  vreinterpretq_u32_f32(vuzp2q_f32(vi0x4567, vi0x89AB))));
      const float32x4_t vi1x468A = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vuzp1q_f32(vi1x4567, vi1x89AB))));
      const float32x4_t vi1x579B = vreinterpretq_f32_u32(vandq_u32(vmask_odd,  vreinterpretq_u32_f32(vuzp2q_f32(vi1x4567, vi1x89AB))));
      const float32x4_t vi2x468A = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vuzp1q_f32(vi2x4567, vi2x89AB))));
      const float32x4_t vi2x579B = vreinterpretq_f32_u32(vandq_u32(vmask_odd,  vreinterpretq_u32_f32(vuzp2q_f32(vi2x4567, vi2x89AB))));
      // add bias only to first row, it will then get added
      // to the final result
      // multiply each row by corresponding row of center column of filter
      vo468Ap0 = vfmaq_laneq_f32(vo468Ap0, vi0x468A, vw0123, 2);
      float32x4_t vo468Ap1 = vmulq_laneq_f32(vi1x468A, vw4567, 1);
      float32x4_t vo468Ap2 = vmulq_lane_f32(vi2x468A, vw89, 0);

      // grab the values corresponding the left filter tap
      const float32x4_t vi0x3579 = vextq_f32(vi0x0123, vi0x579B, 3);
      const float32x4_t vi1x3579 = vextq_f32(vi1x0123, vi1x579B, 3);
      const float32x4_t vi2x3579 = vextq_f32(vi2x0123, vi2x579B, 3);

      vo468Ap0 = vfmaq_laneq_f32(vo468Ap0, vi0x3579, vw0123, 1);
      vo468Ap1 = vfmaq_laneq_f32(vo468Ap1, vi1x3579, vw4567, 0);
      vo468Ap2 = vfmaq_laneq_f32(vo468Ap2, vi2x3579, vw4567, 3);

      // do multiplication by right filter tap
      vo468Ap0 = vfmaq_laneq_f32(vo468Ap0, vi0x579B, vw0123, 3);
      vo468Ap1 = vfmaq_laneq_f32(vo468Ap1, vi1x579B, vw4567, 2);
      vo468Ap2 = vfmaq_lane_f32 (vo468Ap2, vi2x579B, vw89, 1);

      // add up across rows to get the final outputs
      float32x4_t vo = vaddq_f32(vo468Ap0, vo468Ap1);
      vo = vaddq_f32(vo, vo468Ap2);

      vo = vmaxq_f32(vo, vmin);
      vo = vminq_f32(vo, vmax);

      k += 1;
      if (k & 8) {
        vst1q_f32(output, vo);
      } else {
        float* output_lo = output;
        float32x2_t vo_lo = vget_low_f32(vo);
        if (k & 4) {
          vst1_f32(output_lo, vo_lo); output_lo += 2;
          vo_lo = vget_high_f32(vo);
        }
        if (k & 2) {
          vst1_lane_f32(output_lo, vo_lo, 0);
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i0 + input_width_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_width_increment);
    i2 = (const float*) ((uintptr_t) i2 + input_width_increment);
    output = (float*) ((uintptr_t) output + output_width_increment);
  } while (--m != 0);
}
