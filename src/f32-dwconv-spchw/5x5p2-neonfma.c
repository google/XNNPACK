// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>


void xnn_f32_dwconv_spchw_ukernel_5x5p2__neonfma(
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

  const uint32x4_t vmask = vld1q_u32(params->neon.mask);
  const float32x4_t vmax = vld1q_dup_f32(&params->neon.max);
  const float32x4_t vmin = vld1q_dup_f32(&params->neon.min);

  const size_t input_width_increment_triple = 3  * input_width_stride - round_up_po2(n, 4) / 4 * input_tuple_stride;
  const size_t output_width_increment_single = output_width_stride - (n - 1) / 4 * output_tuple_stride;

  // No vertical padding.
  const float* i0 = input;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_width_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width_stride);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width_stride);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_width_stride);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_width_stride);

  float* output0 = output;
  float* output1 = (float *)((uintptr_t) output0 + output_width_stride);
  float* output2 = (float *)((uintptr_t) output1 + output_width_stride);

  const float32x4_t vw0123 = vld1q_f32(weights);
  const float32x4_t vw4567 = vld1q_f32(weights + 4);
  const float32x4_t vw89AB = vld1q_f32(weights + 8);
  const float32x4_t vwCDEF = vld1q_f32(weights + 12);
  const float32x4_t vwGHIJ = vld1q_f32(weights + 16);
  const float32x4_t vwKLMN = vld1q_f32(weights + 20);
  const float32x2_t vwOP   = vld1_f32( weights + 24);

  while (m >= 3) {
    float32x4_t vi0x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi1x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi2x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi3x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi4x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi5x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi6x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi0x4567 = vld1q_f32(i0); i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
    float32x4_t vi1x4567 = vld1q_f32(i1); i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
    float32x4_t vi2x4567 = vld1q_f32(i2); i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);
    float32x4_t vi3x4567 = vld1q_f32(i3); i3 = (const float*) ((uintptr_t) i3 + input_tuple_stride);
    float32x4_t vi4x4567 = vld1q_f32(i4); i4 = (const float*) ((uintptr_t) i4 + input_tuple_stride);
    float32x4_t vi5x4567 = vld1q_f32(i5); i5 = (const float*) ((uintptr_t) i5 + input_tuple_stride);
    float32x4_t vi6x4567 = vld1q_f32(i6); i6 = (const float*) ((uintptr_t) i6 + input_tuple_stride);

    size_t k = n;
    for (; k > 8; k -= 4) {
      float32x4_t vo4567p00 = vdupq_laneq_f32(vw0123, 0);
      float32x4_t vo4567p10 = vdupq_laneq_f32(vw0123, 0);
      float32x4_t vo4567p20 = vdupq_laneq_f32(vw0123, 0);

      const float32x4_t vi0x89AB = vld1q_f32(i0); i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      const float32x4_t vi1x89AB = vld1q_f32(i1); i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      const float32x4_t vi2x89AB = vld1q_f32(i2); i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);
      const float32x4_t vi3x89AB = vld1q_f32(i3); i3 = (const float*) ((uintptr_t) i3 + input_tuple_stride);
      const float32x4_t vi4x89AB = vld1q_f32(i4); i4 = (const float*) ((uintptr_t) i4 + input_tuple_stride);
      const float32x4_t vi5x89AB = vld1q_f32(i5); i5 = (const float*) ((uintptr_t) i5 + input_tuple_stride);
      const float32x4_t vi6x89AB = vld1q_f32(i6); i6 = (const float*) ((uintptr_t) i6 + input_tuple_stride);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x4567, vw0123, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x4567, vw0123, 3);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi2x4567, vw0123, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x4567, vw89AB, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x4567, vw89AB, 0);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi3x4567, vw89AB, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x4567, vwCDEF, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x4567, vwCDEF, 1);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi4x4567, vwCDEF, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x4567, vwGHIJ, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x4567, vwGHIJ, 2);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi5x4567, vwGHIJ, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x4567, vwKLMN, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi5x4567, vwKLMN, 3);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi6x4567, vwKLMN, 3);


      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);
      const float32x4_t vi3x3456 = vextq_f32(vi3x0123, vi3x4567, 3);
      const float32x4_t vi4x3456 = vextq_f32(vi4x0123, vi4x4567, 3);
      const float32x4_t vi5x3456 = vextq_f32(vi5x0123, vi5x4567, 3);
      const float32x4_t vi6x3456 = vextq_f32(vi6x0123, vi6x4567, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x3456, vw0123, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x3456, vw0123, 2);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi2x3456, vw0123, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x3456, vw4567, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x3456, vw4567, 3);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi3x3456, vw4567, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x3456, vwCDEF, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x3456, vwCDEF, 0);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi4x3456, vwCDEF, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x3456, vwGHIJ, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x3456, vwGHIJ, 1);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi5x3456, vwGHIJ, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x3456, vwKLMN, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi5x3456, vwKLMN, 2);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi6x3456, vwKLMN, 2);

      const float32x4_t vi0x2345 = vextq_f32(vi0x0123, vi0x4567, 2);
      const float32x4_t vi1x2345 = vextq_f32(vi1x0123, vi1x4567, 2);
      const float32x4_t vi2x2345 = vextq_f32(vi2x0123, vi2x4567, 2);
      const float32x4_t vi3x2345 = vextq_f32(vi3x0123, vi3x4567, 2);
      const float32x4_t vi4x2345 = vextq_f32(vi4x0123, vi4x4567, 2);
      const float32x4_t vi5x2345 = vextq_f32(vi5x0123, vi5x4567, 2);
      const float32x4_t vi6x2345 = vextq_f32(vi6x0123, vi6x4567, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x2345, vw0123, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x2345, vw0123, 1);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi2x2345, vw0123, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x2345, vw4567, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x2345, vw4567, 2);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi3x2345, vw4567, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x2345, vw89AB, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x2345, vw89AB, 3);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi4x2345, vw89AB, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x2345, vwGHIJ, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x2345, vwGHIJ, 0);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi5x2345, vwGHIJ, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x2345, vwKLMN, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi5x2345, vwKLMN, 1);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi6x2345, vwKLMN, 1);

      vi0x0123 = vi0x4567;
      vi1x0123 = vi1x4567;
      vi2x0123 = vi2x4567;
      vi3x0123 = vi3x4567;
      vi4x0123 = vi4x4567;
      vi5x0123 = vi5x4567;
      vi6x0123 = vi6x4567;

      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vi0x89AB, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vi1x89AB, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vi2x89AB, 1);
      const float32x4_t vi3x5678 = vextq_f32(vi3x4567, vi3x89AB, 1);
      const float32x4_t vi4x5678 = vextq_f32(vi4x4567, vi4x89AB, 1);
      const float32x4_t vi5x5678 = vextq_f32(vi5x4567, vi5x89AB, 1);
      const float32x4_t vi6x5678 = vextq_f32(vi6x4567, vi6x89AB, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x5678, vw4567, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x5678, vw4567, 0);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi2x5678, vw4567, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x5678, vw89AB, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x5678, vw89AB, 1);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi3x5678, vw89AB, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x5678, vwCDEF, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x5678, vwCDEF, 2);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi4x5678, vwCDEF, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x5678, vwGHIJ, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x5678, vwGHIJ, 3);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi5x5678, vwGHIJ, 3);

      vo4567p00 = vfmaq_lane_f32( vo4567p00, vi4x5678, vwOP, 0);
      vo4567p10 = vfmaq_lane_f32( vo4567p10, vi5x5678, vwOP, 0);
      vo4567p20 = vfmaq_lane_f32( vo4567p20, vi6x5678, vwOP, 0);

      const float32x4_t vi0x6789 = vextq_f32(vi0x4567, vi0x89AB, 2);
      const float32x4_t vi1x6789 = vextq_f32(vi1x4567, vi1x89AB, 2);
      const float32x4_t vi2x6789 = vextq_f32(vi2x4567, vi2x89AB, 2);
      const float32x4_t vi3x6789 = vextq_f32(vi3x4567, vi3x89AB, 2);
      const float32x4_t vi4x6789 = vextq_f32(vi4x4567, vi4x89AB, 2);
      const float32x4_t vi5x6789 = vextq_f32(vi5x4567, vi5x89AB, 2);
      const float32x4_t vi6x6789 = vextq_f32(vi6x4567, vi6x89AB, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x6789, vw4567, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x6789, vw4567, 1);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi2x6789, vw4567, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x6789, vw89AB, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x6789, vw89AB, 2);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi3x6789, vw89AB, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x6789, vwCDEF, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x6789, vwCDEF, 3);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi4x6789, vwCDEF, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x6789, vwKLMN, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x6789, vwKLMN, 0);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi5x6789, vwKLMN, 0);

      vo4567p00 = vfmaq_lane_f32( vo4567p00, vi4x6789, vwOP, 1);
      vo4567p10 = vfmaq_lane_f32( vo4567p10, vi5x6789, vwOP, 1);
      vo4567p20 = vfmaq_lane_f32( vo4567p20, vi6x6789, vwOP, 1);

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;
      vi4x4567 = vi4x89AB;
      vi5x4567 = vi5x89AB;
      vi6x4567 = vi6x89AB;

      float32x4_t vo0 = vo4567p00;
      float32x4_t vo1 = vo4567p10;
      float32x4_t vo2 = vo4567p20;

      vo0 = vmaxq_f32(vo0, vmin);
      vo0 = vminq_f32(vo0, vmax);
      vo1 = vmaxq_f32(vo1, vmin);
      vo1 = vminq_f32(vo1, vmax);
      vo2 = vmaxq_f32(vo2, vmin);
      vo2 = vminq_f32(vo2, vmax);

      vst1q_f32(output0, vo0); output0 = (float*) ((uintptr_t) output0 + output_tuple_stride);
      vst1q_f32(output1, vo1); output1 = (float*) ((uintptr_t) output1 + output_tuple_stride);
      vst1q_f32(output2, vo2); output2 = (float*) ((uintptr_t) output2 + output_tuple_stride);
    }
    // Always process the last block of 5..8 pixels.
    if XNN_LIKELY(k > 4)
    {
      float32x4_t vo4567p00 = vdupq_laneq_f32(vw0123, 0);
      float32x4_t vo4567p10 = vdupq_laneq_f32(vw0123, 0);
      float32x4_t vo4567p20 = vdupq_laneq_f32(vw0123, 0);

      float32x4_t vi0x89AB = vld1q_f32(i0); i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      float32x4_t vi1x89AB = vld1q_f32(i1); i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      float32x4_t vi2x89AB = vld1q_f32(i2); i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);
      float32x4_t vi3x89AB = vld1q_f32(i3); i3 = (const float*) ((uintptr_t) i3 + input_tuple_stride);
      float32x4_t vi4x89AB = vld1q_f32(i4); i4 = (const float*) ((uintptr_t) i4 + input_tuple_stride);
      float32x4_t vi5x89AB = vld1q_f32(i5); i5 = (const float*) ((uintptr_t) i5 + input_tuple_stride);
      float32x4_t vi6x89AB = vld1q_f32(i6); i6 = (const float*) ((uintptr_t) i6 + input_tuple_stride);

      vi0x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi0x89AB)));
      vi1x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi1x89AB)));
      vi2x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi2x89AB)));
      vi3x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi3x89AB)));
      vi4x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi4x89AB)));
      vi5x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi5x89AB)));
      vi6x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi6x89AB)));

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x4567, vw0123, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x4567, vw0123, 3);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi2x4567, vw0123, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x4567, vw89AB, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x4567, vw89AB, 0);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi3x4567, vw89AB, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x4567, vwCDEF, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x4567, vwCDEF, 1);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi4x4567, vwCDEF, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x4567, vwGHIJ, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x4567, vwGHIJ, 2);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi5x4567, vwGHIJ, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x4567, vwKLMN, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi5x4567, vwKLMN, 3);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi6x4567, vwKLMN, 3);


      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);
      const float32x4_t vi3x3456 = vextq_f32(vi3x0123, vi3x4567, 3);
      const float32x4_t vi4x3456 = vextq_f32(vi4x0123, vi4x4567, 3);
      const float32x4_t vi5x3456 = vextq_f32(vi5x0123, vi5x4567, 3);
      const float32x4_t vi6x3456 = vextq_f32(vi6x0123, vi6x4567, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x3456, vw0123, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x3456, vw0123, 2);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi2x3456, vw0123, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x3456, vw4567, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x3456, vw4567, 3);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi3x3456, vw4567, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x3456, vwCDEF, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x3456, vwCDEF, 0);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi4x3456, vwCDEF, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x3456, vwGHIJ, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x3456, vwGHIJ, 1);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi5x3456, vwGHIJ, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x3456, vwKLMN, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi5x3456, vwKLMN, 2);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi6x3456, vwKLMN, 2);


      const float32x4_t vi0x2345 = vextq_f32(vi0x0123, vi0x4567, 2);
      const float32x4_t vi1x2345 = vextq_f32(vi1x0123, vi1x4567, 2);
      const float32x4_t vi2x2345 = vextq_f32(vi2x0123, vi2x4567, 2);
      const float32x4_t vi3x2345 = vextq_f32(vi3x0123, vi3x4567, 2);
      const float32x4_t vi4x2345 = vextq_f32(vi4x0123, vi4x4567, 2);
      const float32x4_t vi5x2345 = vextq_f32(vi5x0123, vi5x4567, 2);
      const float32x4_t vi6x2345 = vextq_f32(vi6x0123, vi6x4567, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x2345, vw0123, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x2345, vw0123, 1);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi2x2345, vw0123, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x2345, vw4567, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x2345, vw4567, 2);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi3x2345, vw4567, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x2345, vw89AB, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x2345, vw89AB, 3);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi4x2345, vw89AB, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x2345, vwGHIJ, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x2345, vwGHIJ, 0);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi5x2345, vwGHIJ, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x2345, vwKLMN, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi5x2345, vwKLMN, 1);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi6x2345, vwKLMN, 1);


      vi0x0123 = vi0x4567;
      vi1x0123 = vi1x4567;
      vi2x0123 = vi2x4567;
      vi3x0123 = vi3x4567;
      vi4x0123 = vi4x4567;
      vi5x0123 = vi5x4567;
      vi6x0123 = vi6x4567;

      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vi0x89AB, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vi1x89AB, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vi2x89AB, 1);
      const float32x4_t vi3x5678 = vextq_f32(vi3x4567, vi3x89AB, 1);
      const float32x4_t vi4x5678 = vextq_f32(vi4x4567, vi4x89AB, 1);
      const float32x4_t vi5x5678 = vextq_f32(vi5x4567, vi5x89AB, 1);
      const float32x4_t vi6x5678 = vextq_f32(vi6x4567, vi6x89AB, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x5678, vw4567, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x5678, vw4567, 0);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi2x5678, vw4567, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x5678, vw89AB, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x5678, vw89AB, 1);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi3x5678, vw89AB, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x5678, vwCDEF, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x5678, vwCDEF, 2);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi4x5678, vwCDEF, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x5678, vwGHIJ, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x5678, vwGHIJ, 3);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi5x5678, vwGHIJ, 3);

      vo4567p00 = vfmaq_lane_f32( vo4567p00, vi4x5678, vwOP, 0);
      vo4567p10 = vfmaq_lane_f32( vo4567p10, vi5x5678, vwOP, 0);
      vo4567p20 = vfmaq_lane_f32( vo4567p20, vi6x5678, vwOP, 0);

      const float32x4_t vi0x6789 = vextq_f32(vi0x4567, vi0x89AB, 2);
      const float32x4_t vi1x6789 = vextq_f32(vi1x4567, vi1x89AB, 2);
      const float32x4_t vi2x6789 = vextq_f32(vi2x4567, vi2x89AB, 2);
      const float32x4_t vi3x6789 = vextq_f32(vi3x4567, vi3x89AB, 2);
      const float32x4_t vi4x6789 = vextq_f32(vi4x4567, vi4x89AB, 2);
      const float32x4_t vi5x6789 = vextq_f32(vi5x4567, vi5x89AB, 2);
      const float32x4_t vi6x6789 = vextq_f32(vi6x4567, vi6x89AB, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x6789, vw4567, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x6789, vw4567, 1);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi2x6789, vw4567, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x6789, vw89AB, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x6789, vw89AB, 2);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi3x6789, vw89AB, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x6789, vwCDEF, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x6789, vwCDEF, 3);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi4x6789, vwCDEF, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x6789, vwKLMN, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x6789, vwKLMN, 0);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi5x6789, vwKLMN, 0);

      vo4567p00 = vfmaq_lane_f32( vo4567p00, vi4x6789, vwOP, 1);
      vo4567p10 = vfmaq_lane_f32( vo4567p10, vi5x6789, vwOP, 1);
      vo4567p20 = vfmaq_lane_f32( vo4567p20, vi6x6789, vwOP, 1);

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;
      vi4x4567 = vi4x89AB;
      vi5x4567 = vi5x89AB;
      vi6x4567 = vi6x89AB;

      float32x4_t vo0 = vo4567p00;
      float32x4_t vo1 = vo4567p10;
      float32x4_t vo2 = vo4567p20;

      vo0 = vmaxq_f32(vo0, vmin);
      vo0 = vminq_f32(vo0, vmax);
      vo1 = vmaxq_f32(vo1, vmin);
      vo1 = vminq_f32(vo1, vmax);
      vo2 = vmaxq_f32(vo2, vmin);
      vo2 = vminq_f32(vo2, vmax);

      vst1q_f32(output0, vo0); output0 = (float*) ((uintptr_t) output0 + output_tuple_stride);
      vst1q_f32(output1, vo1); output1 = (float*) ((uintptr_t) output1 + output_tuple_stride);
      vst1q_f32(output2, vo2); output2 = (float*) ((uintptr_t) output2 + output_tuple_stride);
      k -= 4;
    }
    assert(k >= 1);
    assert(k <= 4);
    {
      float32x4_t vo4567p00 = vdupq_laneq_f32(vw0123, 0);
      float32x4_t vo4567p10 = vdupq_laneq_f32(vw0123, 0);
      float32x4_t vo4567p20 = vdupq_laneq_f32(vw0123, 0);

      // This might have already happened if there are more than 4 pixels, but
      // we can't count on it.
      vi0x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi0x4567)));
      vi1x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi1x4567)));
      vi2x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi2x4567)));
      vi3x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi3x4567)));
      vi4x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi4x4567)));
      vi5x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi5x4567)));
      vi6x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi6x4567)));

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x4567, vw0123, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x4567, vw0123, 3);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi2x4567, vw0123, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x4567, vw89AB, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x4567, vw89AB, 0);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi3x4567, vw89AB, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x4567, vwCDEF, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x4567, vwCDEF, 1);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi4x4567, vwCDEF, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x4567, vwGHIJ, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x4567, vwGHIJ, 2);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi5x4567, vwGHIJ, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x4567, vwKLMN, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi5x4567, vwKLMN, 3);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi6x4567, vwKLMN, 3);


      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);
      const float32x4_t vi3x3456 = vextq_f32(vi3x0123, vi3x4567, 3);
      const float32x4_t vi4x3456 = vextq_f32(vi4x0123, vi4x4567, 3);
      const float32x4_t vi5x3456 = vextq_f32(vi5x0123, vi5x4567, 3);
      const float32x4_t vi6x3456 = vextq_f32(vi6x0123, vi6x4567, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x3456, vw0123, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x3456, vw0123, 2);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi2x3456, vw0123, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x3456, vw4567, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x3456, vw4567, 3);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi3x3456, vw4567, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x3456, vwCDEF, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x3456, vwCDEF, 0);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi4x3456, vwCDEF, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x3456, vwGHIJ, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x3456, vwGHIJ, 1);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi5x3456, vwGHIJ, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x3456, vwKLMN, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi5x3456, vwKLMN, 2);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi6x3456, vwKLMN, 2);

      const float32x4_t vi0x2345 = vextq_f32(vi0x0123, vi0x4567, 2);
      const float32x4_t vi1x2345 = vextq_f32(vi1x0123, vi1x4567, 2);
      const float32x4_t vi2x2345 = vextq_f32(vi2x0123, vi2x4567, 2);
      const float32x4_t vi3x2345 = vextq_f32(vi3x0123, vi3x4567, 2);
      const float32x4_t vi4x2345 = vextq_f32(vi4x0123, vi4x4567, 2);
      const float32x4_t vi5x2345 = vextq_f32(vi5x0123, vi5x4567, 2);
      const float32x4_t vi6x2345 = vextq_f32(vi6x0123, vi6x4567, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x2345, vw0123, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x2345, vw0123, 1);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi2x2345, vw0123, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x2345, vw4567, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x2345, vw4567, 2);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi3x2345, vw4567, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x2345, vw89AB, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x2345, vw89AB, 3);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi4x2345, vw89AB, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x2345, vwGHIJ, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x2345, vwGHIJ, 0);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi5x2345, vwGHIJ, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x2345, vwKLMN, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi5x2345, vwKLMN, 1);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi6x2345, vwKLMN, 1);


      const float32x4_t vzero = vmovq_n_f32(0.0f);
      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vzero, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vzero, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vzero, 1);
      const float32x4_t vi3x5678 = vextq_f32(vi3x4567, vzero, 1);
      const float32x4_t vi4x5678 = vextq_f32(vi4x4567, vzero, 1);
      const float32x4_t vi5x5678 = vextq_f32(vi5x4567, vzero, 1);
      const float32x4_t vi6x5678 = vextq_f32(vi6x4567, vzero, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x5678, vw4567, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x5678, vw4567, 0);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi2x5678, vw4567, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x5678, vw89AB, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x5678, vw89AB, 1);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi3x5678, vw89AB, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x5678, vwCDEF, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x5678, vwCDEF, 2);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi4x5678, vwCDEF, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x5678, vwGHIJ, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x5678, vwGHIJ, 3);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi5x5678, vwGHIJ, 3);

      vo4567p00 = vfmaq_lane_f32( vo4567p00, vi4x5678, vwOP, 0);
      vo4567p10 = vfmaq_lane_f32( vo4567p10, vi5x5678, vwOP, 0);
      vo4567p20 = vfmaq_lane_f32( vo4567p20, vi6x5678, vwOP, 0);

      const float32x4_t vi0x6789 = vextq_f32(vi0x4567, vzero, 2);
      const float32x4_t vi1x6789 = vextq_f32(vi1x4567, vzero, 2);
      const float32x4_t vi2x6789 = vextq_f32(vi2x4567, vzero, 2);
      const float32x4_t vi3x6789 = vextq_f32(vi3x4567, vzero, 2);
      const float32x4_t vi4x6789 = vextq_f32(vi4x4567, vzero, 2);
      const float32x4_t vi5x6789 = vextq_f32(vi5x4567, vzero, 2);
      const float32x4_t vi6x6789 = vextq_f32(vi6x4567, vzero, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x6789, vw4567, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x6789, vw4567, 1);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi2x6789, vw4567, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x6789, vw89AB, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x6789, vw89AB, 2);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi3x6789, vw89AB, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x6789, vwCDEF, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x6789, vwCDEF, 3);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi4x6789, vwCDEF, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x6789, vwKLMN, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x6789, vwKLMN, 0);
      vo4567p20 = vfmaq_laneq_f32(vo4567p20, vi5x6789, vwKLMN, 0);

      vo4567p00 = vfmaq_lane_f32( vo4567p00, vi4x6789, vwOP, 1);
      vo4567p10 = vfmaq_lane_f32( vo4567p10, vi5x6789, vwOP, 1);
      vo4567p20 = vfmaq_lane_f32( vo4567p20, vi6x6789, vwOP, 1);

      float32x4_t vo0 = vo4567p00;
      float32x4_t vo1 = vo4567p10;
      float32x4_t vo2 = vo4567p20;

      vo0 = vmaxq_f32(vo0, vmin);
      vo0 = vminq_f32(vo0, vmax);
      vo1 = vmaxq_f32(vo1, vmin);
      vo1 = vminq_f32(vo1, vmax);
      vo2 = vmaxq_f32(vo2, vmin);
      vo2 = vminq_f32(vo2, vmax);

      if XNN_LIKELY(k & 4) {
        vst1q_f32(output0, vo0);
        vst1q_f32(output1, vo1);
        vst1q_f32(output2, vo2);
      } else {
        float* output0_lo = output0;
        float* output1_lo = output1;
        float* output2_lo = output2;
        float32x2_t vo0_lo = vget_low_f32(vo0);
        float32x2_t vo1_lo = vget_low_f32(vo1);
        float32x2_t vo2_lo = vget_low_f32(vo2);
        if (k & 2) {
          vst1_f32(output0_lo, vo0_lo); output0_lo += 2;
          vst1_f32(output1_lo, vo1_lo); output1_lo += 2;
          vst1_f32(output2_lo, vo2_lo); output2_lo += 2;
          vo0_lo = vget_high_f32(vo0);
          vo1_lo = vget_high_f32(vo1);
          vo2_lo = vget_high_f32(vo2);
        }
        if (k & 1) {
          vst1_lane_f32(output0_lo, vo0_lo, 0);
          vst1_lane_f32(output1_lo, vo1_lo, 0);
          vst1_lane_f32(output2_lo, vo2_lo, 0);
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i0 + input_width_increment_triple);
    i1 = (const float*) ((uintptr_t) i1 + input_width_increment_triple);
    i2 = (const float*) ((uintptr_t) i2 + input_width_increment_triple);
    i3 = (const float*) ((uintptr_t) i3 + input_width_increment_triple);
    i4 = (const float*) ((uintptr_t) i4 + input_width_increment_triple);
    i5 = (const float*) ((uintptr_t) i5 + input_width_increment_triple);
    i6 = (const float*) ((uintptr_t) i6 + input_width_increment_triple);
    output0 = (float*) ((uintptr_t) output2 + output_width_increment_single);
    output1 = (float*) ((uintptr_t) output0 + output_width_stride);
    output2 = (float*) ((uintptr_t) output1 + output_width_stride);
    m -= 3;
  }

  if (m == 2) {
    float32x4_t vi0x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi1x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi2x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi3x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi4x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi5x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi0x4567 = vld1q_f32(i0); i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
    float32x4_t vi1x4567 = vld1q_f32(i1); i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
    float32x4_t vi2x4567 = vld1q_f32(i2); i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);
    float32x4_t vi3x4567 = vld1q_f32(i3); i3 = (const float*) ((uintptr_t) i3 + input_tuple_stride);
    float32x4_t vi4x4567 = vld1q_f32(i4); i4 = (const float*) ((uintptr_t) i4 + input_tuple_stride);
    float32x4_t vi5x4567 = vld1q_f32(i5); i5 = (const float*) ((uintptr_t) i5 + input_tuple_stride);

    size_t k = n;
    for (; k > 8; k -= 4) {
      float32x4_t vo4567p00 = vdupq_laneq_f32(vw0123, 0);
      float32x4_t vo4567p10 = vdupq_laneq_f32(vw0123, 0);

      const float32x4_t vi0x89AB = vld1q_f32(i0); i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      const float32x4_t vi1x89AB = vld1q_f32(i1); i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      const float32x4_t vi2x89AB = vld1q_f32(i2); i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);
      const float32x4_t vi3x89AB = vld1q_f32(i3); i3 = (const float*) ((uintptr_t) i3 + input_tuple_stride);
      const float32x4_t vi4x89AB = vld1q_f32(i4); i4 = (const float*) ((uintptr_t) i4 + input_tuple_stride);
      const float32x4_t vi5x89AB = vld1q_f32(i5); i5 = (const float*) ((uintptr_t) i5 + input_tuple_stride);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x4567, vw0123, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x4567, vw0123, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x4567, vw89AB, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x4567, vw89AB, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x4567, vwCDEF, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x4567, vwCDEF, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x4567, vwGHIJ, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x4567, vwGHIJ, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x4567, vwKLMN, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi5x4567, vwKLMN, 3);


      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);
      const float32x4_t vi3x3456 = vextq_f32(vi3x0123, vi3x4567, 3);
      const float32x4_t vi4x3456 = vextq_f32(vi4x0123, vi4x4567, 3);
      const float32x4_t vi5x3456 = vextq_f32(vi5x0123, vi5x4567, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x3456, vw0123, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x3456, vw0123, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x3456, vw4567, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x3456, vw4567, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x3456, vwCDEF, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x3456, vwCDEF, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x3456, vwGHIJ, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x3456, vwGHIJ, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x3456, vwKLMN, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi5x3456, vwKLMN, 2);

      const float32x4_t vi0x2345 = vextq_f32(vi0x0123, vi0x4567, 2);
      const float32x4_t vi1x2345 = vextq_f32(vi1x0123, vi1x4567, 2);
      const float32x4_t vi2x2345 = vextq_f32(vi2x0123, vi2x4567, 2);
      const float32x4_t vi3x2345 = vextq_f32(vi3x0123, vi3x4567, 2);
      const float32x4_t vi4x2345 = vextq_f32(vi4x0123, vi4x4567, 2);
      const float32x4_t vi5x2345 = vextq_f32(vi5x0123, vi5x4567, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x2345, vw0123, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x2345, vw0123, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x2345, vw4567, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x2345, vw4567, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x2345, vw89AB, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x2345, vw89AB, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x2345, vwGHIJ, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x2345, vwGHIJ, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x2345, vwKLMN, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi5x2345, vwKLMN, 1);

      vi0x0123 = vi0x4567;
      vi1x0123 = vi1x4567;
      vi2x0123 = vi2x4567;
      vi3x0123 = vi3x4567;
      vi4x0123 = vi4x4567;
      vi5x0123 = vi5x4567;

      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vi0x89AB, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vi1x89AB, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vi2x89AB, 1);
      const float32x4_t vi3x5678 = vextq_f32(vi3x4567, vi3x89AB, 1);
      const float32x4_t vi4x5678 = vextq_f32(vi4x4567, vi4x89AB, 1);
      const float32x4_t vi5x5678 = vextq_f32(vi5x4567, vi5x89AB, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x5678, vw4567, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x5678, vw4567, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x5678, vw89AB, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x5678, vw89AB, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x5678, vwCDEF, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x5678, vwCDEF, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x5678, vwGHIJ, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x5678, vwGHIJ, 3);

      vo4567p00 = vfmaq_lane_f32( vo4567p00, vi4x5678, vwOP, 0);
      vo4567p10 = vfmaq_lane_f32( vo4567p10, vi5x5678, vwOP, 0);

      const float32x4_t vi0x6789 = vextq_f32(vi0x4567, vi0x89AB, 2);
      const float32x4_t vi1x6789 = vextq_f32(vi1x4567, vi1x89AB, 2);
      const float32x4_t vi2x6789 = vextq_f32(vi2x4567, vi2x89AB, 2);
      const float32x4_t vi3x6789 = vextq_f32(vi3x4567, vi3x89AB, 2);
      const float32x4_t vi4x6789 = vextq_f32(vi4x4567, vi4x89AB, 2);
      const float32x4_t vi5x6789 = vextq_f32(vi5x4567, vi5x89AB, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x6789, vw4567, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x6789, vw4567, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x6789, vw89AB, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x6789, vw89AB, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x6789, vwCDEF, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x6789, vwCDEF, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x6789, vwKLMN, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x6789, vwKLMN, 0);

      vo4567p00 = vfmaq_lane_f32( vo4567p00, vi4x6789, vwOP, 1);
      vo4567p10 = vfmaq_lane_f32( vo4567p10, vi5x6789, vwOP, 1);

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;
      vi4x4567 = vi4x89AB;
      vi5x4567 = vi5x89AB;

      float32x4_t vo0 = vo4567p00;
      float32x4_t vo1 = vo4567p10;

      vo0 = vmaxq_f32(vo0, vmin);
      vo0 = vminq_f32(vo0, vmax);
      vo1 = vmaxq_f32(vo1, vmin);
      vo1 = vminq_f32(vo1, vmax);

      vst1q_f32(output0, vo0); output0 = (float*) ((uintptr_t) output0 + output_tuple_stride);
      vst1q_f32(output1, vo1); output1 = (float*) ((uintptr_t) output1 + output_tuple_stride);
    }
    // Always process the last block of 5..8 pixels.
    if XNN_LIKELY(k > 4)
    {
      float32x4_t vo4567p00 = vdupq_laneq_f32(vw0123, 0);
      float32x4_t vo4567p10 = vdupq_laneq_f32(vw0123, 0);

      float32x4_t vi0x89AB = vld1q_f32(i0); i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      float32x4_t vi1x89AB = vld1q_f32(i1); i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      float32x4_t vi2x89AB = vld1q_f32(i2); i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);
      float32x4_t vi3x89AB = vld1q_f32(i3); i3 = (const float*) ((uintptr_t) i3 + input_tuple_stride);
      float32x4_t vi4x89AB = vld1q_f32(i4); i4 = (const float*) ((uintptr_t) i4 + input_tuple_stride);
      float32x4_t vi5x89AB = vld1q_f32(i5); i5 = (const float*) ((uintptr_t) i5 + input_tuple_stride);

      vi0x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi0x89AB)));
      vi1x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi1x89AB)));
      vi2x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi2x89AB)));
      vi3x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi3x89AB)));
      vi4x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi4x89AB)));
      vi5x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi5x89AB)));

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x4567, vw0123, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x4567, vw0123, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x4567, vw89AB, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x4567, vw89AB, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x4567, vwCDEF, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x4567, vwCDEF, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x4567, vwGHIJ, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x4567, vwGHIJ, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x4567, vwKLMN, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi5x4567, vwKLMN, 3);


      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);
      const float32x4_t vi3x3456 = vextq_f32(vi3x0123, vi3x4567, 3);
      const float32x4_t vi4x3456 = vextq_f32(vi4x0123, vi4x4567, 3);
      const float32x4_t vi5x3456 = vextq_f32(vi5x0123, vi5x4567, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x3456, vw0123, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x3456, vw0123, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x3456, vw4567, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x3456, vw4567, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x3456, vwCDEF, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x3456, vwCDEF, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x3456, vwGHIJ, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x3456, vwGHIJ, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x3456, vwKLMN, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi5x3456, vwKLMN, 2);


      const float32x4_t vi0x2345 = vextq_f32(vi0x0123, vi0x4567, 2);
      const float32x4_t vi1x2345 = vextq_f32(vi1x0123, vi1x4567, 2);
      const float32x4_t vi2x2345 = vextq_f32(vi2x0123, vi2x4567, 2);
      const float32x4_t vi3x2345 = vextq_f32(vi3x0123, vi3x4567, 2);
      const float32x4_t vi4x2345 = vextq_f32(vi4x0123, vi4x4567, 2);
      const float32x4_t vi5x2345 = vextq_f32(vi5x0123, vi5x4567, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x2345, vw0123, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x2345, vw0123, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x2345, vw4567, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x2345, vw4567, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x2345, vw89AB, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x2345, vw89AB, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x2345, vwGHIJ, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x2345, vwGHIJ, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x2345, vwKLMN, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi5x2345, vwKLMN, 1);


      vi0x0123 = vi0x4567;
      vi1x0123 = vi1x4567;
      vi2x0123 = vi2x4567;
      vi3x0123 = vi3x4567;
      vi4x0123 = vi4x4567;
      vi5x0123 = vi5x4567;

      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vi0x89AB, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vi1x89AB, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vi2x89AB, 1);
      const float32x4_t vi3x5678 = vextq_f32(vi3x4567, vi3x89AB, 1);
      const float32x4_t vi4x5678 = vextq_f32(vi4x4567, vi4x89AB, 1);
      const float32x4_t vi5x5678 = vextq_f32(vi5x4567, vi5x89AB, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x5678, vw4567, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x5678, vw4567, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x5678, vw89AB, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x5678, vw89AB, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x5678, vwCDEF, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x5678, vwCDEF, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x5678, vwGHIJ, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x5678, vwGHIJ, 3);

      vo4567p00 = vfmaq_lane_f32( vo4567p00, vi4x5678, vwOP, 0);
      vo4567p10 = vfmaq_lane_f32( vo4567p10, vi5x5678, vwOP, 0);

      const float32x4_t vi0x6789 = vextq_f32(vi0x4567, vi0x89AB, 2);
      const float32x4_t vi1x6789 = vextq_f32(vi1x4567, vi1x89AB, 2);
      const float32x4_t vi2x6789 = vextq_f32(vi2x4567, vi2x89AB, 2);
      const float32x4_t vi3x6789 = vextq_f32(vi3x4567, vi3x89AB, 2);
      const float32x4_t vi4x6789 = vextq_f32(vi4x4567, vi4x89AB, 2);
      const float32x4_t vi5x6789 = vextq_f32(vi5x4567, vi5x89AB, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x6789, vw4567, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x6789, vw4567, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x6789, vw89AB, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x6789, vw89AB, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x6789, vwCDEF, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x6789, vwCDEF, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x6789, vwKLMN, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x6789, vwKLMN, 0);

      vo4567p00 = vfmaq_lane_f32( vo4567p00, vi4x6789, vwOP, 1);
      vo4567p10 = vfmaq_lane_f32( vo4567p10, vi5x6789, vwOP, 1);

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;
      vi4x4567 = vi4x89AB;
      vi5x4567 = vi5x89AB;

      float32x4_t vo0 = vo4567p00;
      float32x4_t vo1 = vo4567p10;

      vo0 = vmaxq_f32(vo0, vmin);
      vo0 = vminq_f32(vo0, vmax);
      vo1 = vmaxq_f32(vo1, vmin);
      vo1 = vminq_f32(vo1, vmax);

      vst1q_f32(output0, vo0); output0 = (float*) ((uintptr_t) output0 + output_tuple_stride);
      vst1q_f32(output1, vo1); output1 = (float*) ((uintptr_t) output1 + output_tuple_stride);
      k -= 4;
    }
    assert(k >= 1);
    assert(k <= 4);
    {
      float32x4_t vo4567p00 = vdupq_laneq_f32(vw0123, 0);
      float32x4_t vo4567p10 = vdupq_laneq_f32(vw0123, 0);

      // This might have already happened if there are more than 4 pixels, but
      // we can't count on it.
      vi0x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi0x4567)));
      vi1x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi1x4567)));
      vi2x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi2x4567)));
      vi3x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi3x4567)));
      vi4x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi4x4567)));
      vi5x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi5x4567)));

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x4567, vw0123, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x4567, vw0123, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x4567, vw89AB, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x4567, vw89AB, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x4567, vwCDEF, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x4567, vwCDEF, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x4567, vwGHIJ, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x4567, vwGHIJ, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x4567, vwKLMN, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi5x4567, vwKLMN, 3);


      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);
      const float32x4_t vi3x3456 = vextq_f32(vi3x0123, vi3x4567, 3);
      const float32x4_t vi4x3456 = vextq_f32(vi4x0123, vi4x4567, 3);
      const float32x4_t vi5x3456 = vextq_f32(vi5x0123, vi5x4567, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x3456, vw0123, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x3456, vw0123, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x3456, vw4567, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x3456, vw4567, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x3456, vwCDEF, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x3456, vwCDEF, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x3456, vwGHIJ, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x3456, vwGHIJ, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x3456, vwKLMN, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi5x3456, vwKLMN, 2);

      const float32x4_t vi0x2345 = vextq_f32(vi0x0123, vi0x4567, 2);
      const float32x4_t vi1x2345 = vextq_f32(vi1x0123, vi1x4567, 2);
      const float32x4_t vi2x2345 = vextq_f32(vi2x0123, vi2x4567, 2);
      const float32x4_t vi3x2345 = vextq_f32(vi3x0123, vi3x4567, 2);
      const float32x4_t vi4x2345 = vextq_f32(vi4x0123, vi4x4567, 2);
      const float32x4_t vi5x2345 = vextq_f32(vi5x0123, vi5x4567, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x2345, vw0123, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x2345, vw0123, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x2345, vw4567, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x2345, vw4567, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x2345, vw89AB, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x2345, vw89AB, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x2345, vwGHIJ, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x2345, vwGHIJ, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x2345, vwKLMN, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi5x2345, vwKLMN, 1);

      const float32x4_t vzero = vmovq_n_f32(0.0f);
      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vzero, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vzero, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vzero, 1);
      const float32x4_t vi3x5678 = vextq_f32(vi3x4567, vzero, 1);
      const float32x4_t vi4x5678 = vextq_f32(vi4x4567, vzero, 1);
      const float32x4_t vi5x5678 = vextq_f32(vi5x4567, vzero, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x5678, vw4567, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x5678, vw4567, 0);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x5678, vw89AB, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x5678, vw89AB, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x5678, vwCDEF, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x5678, vwCDEF, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x5678, vwGHIJ, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x5678, vwGHIJ, 3);

      vo4567p00 = vfmaq_lane_f32( vo4567p00, vi4x5678, vwOP, 0);
      vo4567p10 = vfmaq_lane_f32( vo4567p10, vi5x5678, vwOP, 0);

      const float32x4_t vi0x6789 = vextq_f32(vi0x4567, vzero, 2);
      const float32x4_t vi1x6789 = vextq_f32(vi1x4567, vzero, 2);
      const float32x4_t vi2x6789 = vextq_f32(vi2x4567, vzero, 2);
      const float32x4_t vi3x6789 = vextq_f32(vi3x4567, vzero, 2);
      const float32x4_t vi4x6789 = vextq_f32(vi4x4567, vzero, 2);
      const float32x4_t vi5x6789 = vextq_f32(vi5x4567, vzero, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x6789, vw4567, 1);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi1x6789, vw4567, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x6789, vw89AB, 2);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi2x6789, vw89AB, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x6789, vwCDEF, 3);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi3x6789, vwCDEF, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x6789, vwKLMN, 0);
      vo4567p10 = vfmaq_laneq_f32(vo4567p10, vi4x6789, vwKLMN, 0);

      vo4567p00 = vfmaq_lane_f32( vo4567p00, vi4x6789, vwOP, 1);
      vo4567p10 = vfmaq_lane_f32( vo4567p10, vi5x6789, vwOP, 1);

      float32x4_t vo0 = vo4567p00;
      float32x4_t vo1 = vo4567p10;

      vo0 = vmaxq_f32(vo0, vmin);
      vo0 = vminq_f32(vo0, vmax);
      vo1 = vmaxq_f32(vo1, vmin);
      vo1 = vminq_f32(vo1, vmax);

      if XNN_LIKELY(k & 4) {
        vst1q_f32(output0, vo0);
        vst1q_f32(output1, vo1);
      } else {
        float* output0_lo = output0;
        float* output1_lo = output1;
        float32x2_t vo0_lo = vget_low_f32(vo0);
        float32x2_t vo1_lo = vget_low_f32(vo1);
        if (k & 2) {
          vst1_f32(output0_lo, vo0_lo); output0_lo += 2;
          vst1_f32(output1_lo, vo1_lo); output1_lo += 2;
          vo0_lo = vget_high_f32(vo0);
          vo1_lo = vget_high_f32(vo1);
        }
        if (k & 1) {
          vst1_lane_f32(output0_lo, vo0_lo, 0);
          vst1_lane_f32(output1_lo, vo1_lo, 0);
        }
      }
    }
  }
  else if (m == 1) {
    float32x4_t vi0x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi1x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi2x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi3x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi4x0123 = vmovq_n_f32(0.0f);
    float32x4_t vi0x4567 = vld1q_f32(i0); i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
    float32x4_t vi1x4567 = vld1q_f32(i1); i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
    float32x4_t vi2x4567 = vld1q_f32(i2); i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);
    float32x4_t vi3x4567 = vld1q_f32(i3); i3 = (const float*) ((uintptr_t) i3 + input_tuple_stride);
    float32x4_t vi4x4567 = vld1q_f32(i4); i4 = (const float*) ((uintptr_t) i4 + input_tuple_stride);

    size_t k = n;
    for (; k > 8; k -= 4) {
      float32x4_t vo4567p00 = vdupq_laneq_f32(vw0123, 0);

      const float32x4_t vi0x89AB = vld1q_f32(i0); i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      const float32x4_t vi1x89AB = vld1q_f32(i1); i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      const float32x4_t vi2x89AB = vld1q_f32(i2); i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);
      const float32x4_t vi3x89AB = vld1q_f32(i3); i3 = (const float*) ((uintptr_t) i3 + input_tuple_stride);
      const float32x4_t vi4x89AB = vld1q_f32(i4); i4 = (const float*) ((uintptr_t) i4 + input_tuple_stride);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x4567, vw0123, 3);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x4567, vw89AB, 0);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x4567, vwCDEF, 1);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x4567, vwGHIJ, 2);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x4567, vwKLMN, 3);

      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);
      const float32x4_t vi3x3456 = vextq_f32(vi3x0123, vi3x4567, 3);
      const float32x4_t vi4x3456 = vextq_f32(vi4x0123, vi4x4567, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x3456, vw0123, 2);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x3456, vw4567, 3);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x3456, vwCDEF, 0);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x3456, vwGHIJ, 1);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x3456, vwKLMN, 2);

      const float32x4_t vi0x2345 = vextq_f32(vi0x0123, vi0x4567, 2);
      const float32x4_t vi1x2345 = vextq_f32(vi1x0123, vi1x4567, 2);
      const float32x4_t vi2x2345 = vextq_f32(vi2x0123, vi2x4567, 2);
      const float32x4_t vi3x2345 = vextq_f32(vi3x0123, vi3x4567, 2);
      const float32x4_t vi4x2345 = vextq_f32(vi4x0123, vi4x4567, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x2345, vw0123, 1);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x2345, vw4567, 2);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x2345, vw89AB, 3);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x2345, vwGHIJ, 0);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x2345, vwKLMN, 1);

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

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x5678, vw4567, 0);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x5678, vw89AB, 1);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x5678, vwCDEF, 2);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x5678, vwGHIJ, 3);
      vo4567p00 = vfmaq_lane_f32( vo4567p00, vi4x5678, vwOP, 0);

      const float32x4_t vi0x6789 = vextq_f32(vi0x4567, vi0x89AB, 2);
      const float32x4_t vi1x6789 = vextq_f32(vi1x4567, vi1x89AB, 2);
      const float32x4_t vi2x6789 = vextq_f32(vi2x4567, vi2x89AB, 2);
      const float32x4_t vi3x6789 = vextq_f32(vi3x4567, vi3x89AB, 2);
      const float32x4_t vi4x6789 = vextq_f32(vi4x4567, vi4x89AB, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x6789, vw4567, 1);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x6789, vw89AB, 2);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x6789, vwCDEF, 3);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x6789, vwKLMN, 0);
      vo4567p00 = vfmaq_lane_f32( vo4567p00, vi4x6789, vwOP, 1);


      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;
      vi4x4567 = vi4x89AB;

      float32x4_t vo0 = vo4567p00;

      vo0 = vmaxq_f32(vo0, vmin);
      vo0 = vminq_f32(vo0, vmax);

      vst1q_f32(output0, vo0); output0 = (float*) ((uintptr_t) output0 + output_tuple_stride);
    }
    // Always process the last block of 5..8 pixels.
    if XNN_LIKELY(k > 4)
    {
      float32x4_t vo4567p00 = vdupq_laneq_f32(vw0123, 0);

      float32x4_t vi0x89AB = vld1q_f32(i0); i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      float32x4_t vi1x89AB = vld1q_f32(i1); i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      float32x4_t vi2x89AB = vld1q_f32(i2); i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);
      float32x4_t vi3x89AB = vld1q_f32(i3); i3 = (const float*) ((uintptr_t) i3 + input_tuple_stride);
      float32x4_t vi4x89AB = vld1q_f32(i4); i4 = (const float*) ((uintptr_t) i4 + input_tuple_stride);

      vi0x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi0x89AB)));
      vi1x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi1x89AB)));
      vi2x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi2x89AB)));
      vi3x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi3x89AB)));
      vi4x89AB = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi4x89AB)));

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x4567, vw0123, 3);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x4567, vw89AB, 0);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x4567, vwCDEF, 1);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x4567, vwGHIJ, 2);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x4567, vwKLMN, 3);

      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);
      const float32x4_t vi3x3456 = vextq_f32(vi3x0123, vi3x4567, 3);
      const float32x4_t vi4x3456 = vextq_f32(vi4x0123, vi4x4567, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x3456, vw0123, 2);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x3456, vw4567, 3);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x3456, vwCDEF, 0);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x3456, vwGHIJ, 1);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x3456, vwKLMN, 2);

      const float32x4_t vi0x2345 = vextq_f32(vi0x0123, vi0x4567, 2);
      const float32x4_t vi1x2345 = vextq_f32(vi1x0123, vi1x4567, 2);
      const float32x4_t vi2x2345 = vextq_f32(vi2x0123, vi2x4567, 2);
      const float32x4_t vi3x2345 = vextq_f32(vi3x0123, vi3x4567, 2);
      const float32x4_t vi4x2345 = vextq_f32(vi4x0123, vi4x4567, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x2345, vw0123, 1);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x2345, vw4567, 2);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x2345, vw89AB, 3);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x2345, vwGHIJ, 0);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x2345, vwKLMN, 1);

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

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x5678, vw4567, 0);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x5678, vw89AB, 1);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x5678, vwCDEF, 2);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x5678, vwGHIJ, 3);
      vo4567p00 = vfmaq_lane_f32( vo4567p00, vi4x5678, vwOP, 0);

      const float32x4_t vi0x6789 = vextq_f32(vi0x4567, vi0x89AB, 2);
      const float32x4_t vi1x6789 = vextq_f32(vi1x4567, vi1x89AB, 2);
      const float32x4_t vi2x6789 = vextq_f32(vi2x4567, vi2x89AB, 2);
      const float32x4_t vi3x6789 = vextq_f32(vi3x4567, vi3x89AB, 2);
      const float32x4_t vi4x6789 = vextq_f32(vi4x4567, vi4x89AB, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x6789, vw4567, 1);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x6789, vw89AB, 2);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x6789, vwCDEF, 3);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x6789, vwKLMN, 0);
      vo4567p00 = vfmaq_lane_f32( vo4567p00, vi4x6789, vwOP, 1);

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;
      vi4x4567 = vi4x89AB;

      float32x4_t vo0 = vo4567p00;

      vo0 = vmaxq_f32(vo0, vmin);
      vo0 = vminq_f32(vo0, vmax);

      vst1q_f32(output0, vo0); output0 = (float*) ((uintptr_t) output0 + output_tuple_stride);
      k -= 4;
    }
    assert(k >= 1);
    assert(k <= 4);
    {
      float32x4_t vo4567p00 = vdupq_laneq_f32(vw0123, 0);

      // This might have already happened if there are more than 4 pixels, but
      // we can't count on it.
      vi0x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi0x4567)));
      vi1x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi1x4567)));
      vi2x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi2x4567)));
      vi3x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi3x4567)));
      vi4x4567 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi4x4567)));

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x4567, vw0123, 3);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x4567, vw89AB, 0);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x4567, vwCDEF, 1);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x4567, vwGHIJ, 2);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x4567, vwKLMN, 3);


      const float32x4_t vi0x3456 = vextq_f32(vi0x0123, vi0x4567, 3);
      const float32x4_t vi1x3456 = vextq_f32(vi1x0123, vi1x4567, 3);
      const float32x4_t vi2x3456 = vextq_f32(vi2x0123, vi2x4567, 3);
      const float32x4_t vi3x3456 = vextq_f32(vi3x0123, vi3x4567, 3);
      const float32x4_t vi4x3456 = vextq_f32(vi4x0123, vi4x4567, 3);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x3456, vw0123, 2);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x3456, vw4567, 3);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x3456, vwCDEF, 0);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x3456, vwGHIJ, 1);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x3456, vwKLMN, 2);

      const float32x4_t vi0x2345 = vextq_f32(vi0x0123, vi0x4567, 2);
      const float32x4_t vi1x2345 = vextq_f32(vi1x0123, vi1x4567, 2);
      const float32x4_t vi2x2345 = vextq_f32(vi2x0123, vi2x4567, 2);
      const float32x4_t vi3x2345 = vextq_f32(vi3x0123, vi3x4567, 2);
      const float32x4_t vi4x2345 = vextq_f32(vi4x0123, vi4x4567, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x2345, vw0123, 1);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x2345, vw4567, 2);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x2345, vw89AB, 3);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x2345, vwGHIJ, 0);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi4x2345, vwKLMN, 1);

      const float32x4_t vzero = vmovq_n_f32(0.0f);
      const float32x4_t vi0x5678 = vextq_f32(vi0x4567, vzero, 1);
      const float32x4_t vi1x5678 = vextq_f32(vi1x4567, vzero, 1);
      const float32x4_t vi2x5678 = vextq_f32(vi2x4567, vzero, 1);
      const float32x4_t vi3x5678 = vextq_f32(vi3x4567, vzero, 1);
      const float32x4_t vi4x5678 = vextq_f32(vi4x4567, vzero, 1);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x5678, vw4567, 0);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x5678, vw89AB, 1);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x5678, vwCDEF, 2);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x5678, vwGHIJ, 3);
      vo4567p00 = vfmaq_lane_f32( vo4567p00, vi4x5678, vwOP, 0);

      const float32x4_t vi0x6789 = vextq_f32(vi0x4567, vzero, 2);
      const float32x4_t vi1x6789 = vextq_f32(vi1x4567, vzero, 2);
      const float32x4_t vi2x6789 = vextq_f32(vi2x4567, vzero, 2);
      const float32x4_t vi3x6789 = vextq_f32(vi3x4567, vzero, 2);
      const float32x4_t vi4x6789 = vextq_f32(vi4x4567, vzero, 2);

      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi0x6789, vw4567, 1);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi1x6789, vw89AB, 2);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi2x6789, vwCDEF, 3);
      vo4567p00 = vfmaq_laneq_f32(vo4567p00, vi3x6789, vwKLMN, 0);
      vo4567p00 = vfmaq_lane_f32( vo4567p00, vi4x6789, vwOP, 1);

      float32x4_t vo0 = vo4567p00;

      vo0 = vmaxq_f32(vo0, vmin);
      vo0 = vminq_f32(vo0, vmax);

      if XNN_LIKELY(k & 4) {
        vst1q_f32(output0, vo0);
      } else {
        float* output0_lo = output0;
        float32x2_t vo0_lo = vget_low_f32(vo0);
        if (k & 2) {
          vst1_f32(output0_lo, vo0_lo); output0_lo += 2;
          vo0_lo = vget_high_f32(vo0);
        }
        if (k & 1) {
          vst1_lane_f32(output0_lo, vo0_lo, 0);
        }
      }
    }
  }
}
