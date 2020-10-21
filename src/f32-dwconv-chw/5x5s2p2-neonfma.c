// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>


void xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma(
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
  assert(padding_top >= 1);
  assert(padding_top <= 2);

  const uint32x4_t vmask_even = vld1q_u32(params->neon.mask_even);
  const uint32x4_t vmask_odd = vld1q_u32(params->neon.mask_odd);
  const float32x4_t vmax = vld1q_dup_f32(&params->neon.max);
  const float32x4_t vmin = vld1q_dup_f32(&params->neon.min);

  const float32x4_t vw0123 = vld1q_f32(weights);
  const float32x4_t vw4567 = vld1q_f32(weights + 4);
  const float32x4_t vw89AB = vld1q_f32(weights + 8);
  const float32x4_t vwCDEF = vld1q_f32(weights + 12);
  const float32x4_t vwGHIJ = vld1q_f32(weights + 16);
  const float32x4_t vwKLMN = vld1q_f32(weights + 20);
  const float32x2_t vwOP   = vld1_f32(weights + 24);

  const uint32_t padding_top_less_1 = padding_top - 1;
  const size_t input_width_stride = input_width * sizeof(float);
  const size_t input_decrement = (round_down_po2(input_width - 1, 4) + 4) * sizeof(float);

  const float* i0 = zero;
  const float* i1 = (const float*) ((uintptr_t) input - ((-padding_top_less_1) & input_width_stride));
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width_stride);
  if XNN_UNPREDICTABLE(padding_top_less_1 != 0) {
    i1 = zero;
  }
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width_stride);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width_stride);

  size_t padded_input_height = input_height + (padding_top_less_1 + 1) + 2 /* padding bottom */;
  size_t output_height = (padded_input_height - 5 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height <= 6) {
      i4 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 6) {
      i3 = zero;
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
    for (; w > 8; w -= 8) {
      float32x4_t vo468Ap00 = vdupq_laneq_f32(vw0123, 0);

      float32x4_t vi0x89AB;
      float32x4_t vi1x89AB;
      float32x4_t vi2x89AB;
      float32x4_t vi3x89AB;
      float32x4_t vi4x89AB;

      vi0x89AB = vld1q_f32(i0); i0 += 4;
      vi1x89AB = vld1q_f32(i1); i1 += 4;
      vi2x89AB = vld1q_f32(i2); i2 += 4;
      vi3x89AB = vld1q_f32(i3); i3 += 4;
      vi4x89AB = vld1q_f32(i4); i4 += 4;

      float32x4_t vi0xCDEF;
      float32x4_t vi1xCDEF;
      float32x4_t vi2xCDEF;
      float32x4_t vi3xCDEF;
      float32x4_t vi4xCDEF;

      vi0xCDEF = vld1q_f32(i0); i0 += 4;
      vi1xCDEF = vld1q_f32(i1); i1 += 4;
      vi2xCDEF = vld1q_f32(i2); i2 += 4;
      vi3xCDEF = vld1q_f32(i3); i3 += 4;
      vi4xCDEF = vld1q_f32(i4); i4 += 4;

      float32x4_t vi0x468A = vuzp1q_f32(vi0x4567, vi0x89AB);
      float32x4_t vi0x579B = vuzp2q_f32(vi0x4567, vi0x89AB);
      float32x4_t vi1x468A = vuzp1q_f32(vi1x4567, vi1x89AB);
      float32x4_t vi1x579B = vuzp2q_f32(vi1x4567, vi1x89AB);
      float32x4_t vi2x468A = vuzp1q_f32(vi2x4567, vi2x89AB);
      float32x4_t vi2x579B = vuzp2q_f32(vi2x4567, vi2x89AB);
      float32x4_t vi3x468A = vuzp1q_f32(vi3x4567, vi3x89AB);
      float32x4_t vi3x579B = vuzp2q_f32(vi3x4567, vi3x89AB);
      float32x4_t vi4x468A = vuzp1q_f32(vi4x4567, vi4x89AB);
      float32x4_t vi4x579B = vuzp2q_f32(vi4x4567, vi4x89AB);

      // middle tap
      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi0x468A, vw0123, 3);
      float32x4_t vo468Ap01 = vmulq_laneq_f32(vi1x468A, vw89AB, 0);
      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi2x468A, vwCDEF, 1);
      vo468Ap01 = vfmaq_laneq_f32(vo468Ap01, vi3x468A, vwGHIJ, 2);
      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi4x468A, vwKLMN, 3);

      // one left
      const float32x4_t vi0x3579 = vextq_f32(vi0x0123, vi0x579B, 3);
      const float32x4_t vi1x3579 = vextq_f32(vi1x0123, vi1x579B, 3);
      const float32x4_t vi2x3579 = vextq_f32(vi2x0123, vi2x579B, 3);
      const float32x4_t vi3x3579 = vextq_f32(vi3x0123, vi3x579B, 3);
      const float32x4_t vi4x3579 = vextq_f32(vi4x0123, vi4x579B, 3);

      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi0x3579, vw0123, 2);
      vo468Ap01 = vfmaq_laneq_f32(vo468Ap01, vi1x3579, vw4567, 3);
      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi2x3579, vwCDEF, 0);
      vo468Ap01 = vfmaq_laneq_f32(vo468Ap01, vi3x3579, vwGHIJ, 1);
      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi4x3579, vwKLMN, 2);

      // two left
      // getting the vector to use for the far left tap is annoying
      // as we can't ext anything we currently have to get it.
      // To do this, we get a bit ugly.  Interpret the float 32x4
      // vector as int 64x2. Then left shift by 32.  Interpret
      // again as float 32x4.  Now the right most bits are what we
      // want them to be for the following ext.
      const float32x4_t vi0x0012 = vreinterpretq_f32_u64(vshlq_n_u64(vreinterpretq_u64_f32(vi0x0123), 32));
      const float32x4_t vi1x0012 = vreinterpretq_f32_u64(vshlq_n_u64(vreinterpretq_u64_f32(vi1x0123), 32));
      const float32x4_t vi2x0012 = vreinterpretq_f32_u64(vshlq_n_u64(vreinterpretq_u64_f32(vi2x0123), 32));
      const float32x4_t vi3x0012 = vreinterpretq_f32_u64(vshlq_n_u64(vreinterpretq_u64_f32(vi3x0123), 32));
      const float32x4_t vi4x0012 = vreinterpretq_f32_u64(vshlq_n_u64(vreinterpretq_u64_f32(vi4x0123), 32));

      const float32x4_t vi0x2468 = vextq_f32(vi0x0012, vi0x468A, 3);
      const float32x4_t vi1x2468 = vextq_f32(vi1x0012, vi1x468A, 3);
      const float32x4_t vi2x2468 = vextq_f32(vi2x0012, vi2x468A, 3);
      const float32x4_t vi3x2468 = vextq_f32(vi3x0012, vi3x468A, 3);
      const float32x4_t vi4x2468 = vextq_f32(vi4x0012, vi4x468A, 3);

      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi0x2468, vw0123, 1);
      vo468Ap01 = vfmaq_laneq_f32(vo468Ap01, vi1x2468, vw4567, 2);
      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi2x2468, vw89AB, 3);
      vo468Ap01 = vfmaq_laneq_f32(vo468Ap01, vi3x2468, vwGHIJ, 0);
      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi4x2468, vwKLMN, 1);

      vi0x0123 = vi0x89AB;
      vi1x0123 = vi1x89AB;
      vi2x0123 = vi2x89AB;
      vi3x0123 = vi3x89AB;
      vi4x0123 = vi4x89AB;

      // one right
      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi0x579B, vw4567, 0);
      vo468Ap01 = vfmaq_laneq_f32(vo468Ap01, vi1x579B, vw89AB, 1);
      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi2x579B, vwCDEF, 2);
      vo468Ap01 = vfmaq_laneq_f32(vo468Ap01, vi3x579B, vwGHIJ, 3);
      vo468Ap00 = vfmaq_lane_f32( vo468Ap00, vi4x579B, vwOP, 0);

      // two right
      const float32x4_t vi0x68AC = vextq_f32(vi0x468A, vi0xCDEF, 1);
      const float32x4_t vi1x68AC = vextq_f32(vi1x468A, vi1xCDEF, 1);
      const float32x4_t vi2x68AC = vextq_f32(vi2x468A, vi2xCDEF, 1);
      const float32x4_t vi3x68AC = vextq_f32(vi3x468A, vi3xCDEF, 1);
      const float32x4_t vi4x68AC = vextq_f32(vi4x468A, vi4xCDEF, 1);

      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi0x68AC, vw4567, 1);
      vo468Ap01 = vfmaq_laneq_f32(vo468Ap01, vi1x68AC, vw89AB, 2);
      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi2x68AC, vwCDEF, 3);
      vo468Ap01 = vfmaq_laneq_f32(vo468Ap01, vi3x68AC, vwKLMN, 0);
      vo468Ap00 = vfmaq_lane_f32( vo468Ap00, vi4x68AC, vwOP, 1);

      vi0x4567 = vi0xCDEF;
      vi1x4567 = vi1xCDEF;
      vi2x4567 = vi2xCDEF;
      vi3x4567 = vi3xCDEF;
      vi4x4567 = vi4xCDEF;

      float32x4_t vo0 = vaddq_f32(vo468Ap00, vo468Ap01);

      vo0 = vmaxq_f32(vo0, vmin);
      vo0 = vminq_f32(vo0, vmax);

      size_t w_tmp = (w + 1) / 2;
      if XNN_LIKELY(w_tmp >= 4) {
        vst1q_f32(output, vo0); output += 4;
      } else {
        float* output_tmp = output;
        float32x2_t vo0_lo = vget_low_f32(vo0);
        if (w_tmp & 2) {
          vst1_f32(output_tmp, vo0_lo); output_tmp += 2;
          vo0_lo = vget_high_f32(vo0);
        }
        if (w_tmp & 1) {
          vst1_lane_f32(output_tmp, vo0_lo, 0);
        }
      }
    }

    {
      float32x4_t vo468Ap00 = vdupq_laneq_f32(vw0123, 0);

      float32x4_t vi0x89AB;
      float32x4_t vi1x89AB;
      float32x4_t vi2x89AB;
      float32x4_t vi3x89AB;
      float32x4_t vi4x89AB;

      if XNN_LIKELY(w > 4) {
        vi0x89AB = vld1q_f32(i0); i0 += 4;
        vi1x89AB = vld1q_f32(i1); i1 += 4;
        vi2x89AB = vld1q_f32(i2); i2 += 4;
        vi3x89AB = vld1q_f32(i3); i3 += 4;
        vi4x89AB = vld1q_f32(i4); i4 += 4;
      } else {
        vi0x89AB = vmovq_n_f32(0.f);
        vi1x89AB = vmovq_n_f32(0.f);
        vi2x89AB = vmovq_n_f32(0.f);
        vi3x89AB = vmovq_n_f32(0.f);
        vi4x89AB = vmovq_n_f32(0.f);
      }

      float32x4_t vi0xCDEF;
      float32x4_t vi1xCDEF;
      float32x4_t vi2xCDEF;
      float32x4_t vi3xCDEF;
      float32x4_t vi4xCDEF;

      if XNN_LIKELY(w > 8) {
        vi0xCDEF = vld1q_f32(i0); i0 += 4;
        vi1xCDEF = vld1q_f32(i1); i1 += 4;
        vi2xCDEF = vld1q_f32(i2); i2 += 4;
        vi3xCDEF = vld1q_f32(i3); i3 += 4;
        vi4xCDEF = vld1q_f32(i4); i4 += 4;
      } else {
        vi0xCDEF = vmovq_n_f32(0.f);
        vi1xCDEF = vmovq_n_f32(0.f);
        vi2xCDEF = vmovq_n_f32(0.f);
        vi3xCDEF = vmovq_n_f32(0.f);
        vi4xCDEF = vmovq_n_f32(0.f);
      }
      float32x4_t vi0x468A = vuzp1q_f32(vi0x4567, vi0x89AB);
      float32x4_t vi0x579B = vuzp2q_f32(vi0x4567, vi0x89AB);
      float32x4_t vi1x468A = vuzp1q_f32(vi1x4567, vi1x89AB);
      float32x4_t vi1x579B = vuzp2q_f32(vi1x4567, vi1x89AB);
      float32x4_t vi2x468A = vuzp1q_f32(vi2x4567, vi2x89AB);
      float32x4_t vi2x579B = vuzp2q_f32(vi2x4567, vi2x89AB);
      float32x4_t vi3x468A = vuzp1q_f32(vi3x4567, vi3x89AB);
      float32x4_t vi3x579B = vuzp2q_f32(vi3x4567, vi3x89AB);
      float32x4_t vi4x468A = vuzp1q_f32(vi4x4567, vi4x89AB);
      float32x4_t vi4x579B = vuzp2q_f32(vi4x4567, vi4x89AB);

      vi0x468A = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi0x468A)));
      vi1x468A = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi1x468A)));
      vi2x468A = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi2x468A)));
      vi3x468A = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi3x468A)));
      vi4x468A = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi4x468A)));

      vi0x579B = vreinterpretq_f32_u32(vandq_u32(vmask_odd, vreinterpretq_u32_f32(vi0x579B)));
      vi1x579B = vreinterpretq_f32_u32(vandq_u32(vmask_odd, vreinterpretq_u32_f32(vi1x579B)));
      vi2x579B = vreinterpretq_f32_u32(vandq_u32(vmask_odd, vreinterpretq_u32_f32(vi2x579B)));
      vi3x579B = vreinterpretq_f32_u32(vandq_u32(vmask_odd, vreinterpretq_u32_f32(vi3x579B)));
      vi4x579B = vreinterpretq_f32_u32(vandq_u32(vmask_odd, vreinterpretq_u32_f32(vi4x579B)));

      // middle tap
      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi0x468A, vw0123, 3);
      float32x4_t vo468Ap01 = vmulq_laneq_f32(vi1x468A, vw89AB, 0);
      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi2x468A, vwCDEF, 1);
      vo468Ap01 = vfmaq_laneq_f32(vo468Ap01, vi3x468A, vwGHIJ, 2);
      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi4x468A, vwKLMN, 3);

      // one left
      const float32x4_t vi0x3579 = vextq_f32(vi0x0123, vi0x579B, 3);
      const float32x4_t vi1x3579 = vextq_f32(vi1x0123, vi1x579B, 3);
      const float32x4_t vi2x3579 = vextq_f32(vi2x0123, vi2x579B, 3);
      const float32x4_t vi3x3579 = vextq_f32(vi3x0123, vi3x579B, 3);
      const float32x4_t vi4x3579 = vextq_f32(vi4x0123, vi4x579B, 3);

      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi0x3579, vw0123, 2);
      vo468Ap01 = vfmaq_laneq_f32(vo468Ap01, vi1x3579, vw4567, 3);
      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi2x3579, vwCDEF, 0);
      vo468Ap01 = vfmaq_laneq_f32(vo468Ap01, vi3x3579, vwGHIJ, 1);
      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi4x3579, vwKLMN, 2);

      // two left
      // getting the vector to use for the far left tap is annoying
      // as we can't ext anything we currently have to get it.
      // To do this, we get a bit ugly.  Interpret the float 32x4
      // vector as int 64x2. Then left shift by 32.  Interpret
      // again as float 32x4.  Now the right most bits are what we
      // want them to be for the following ext.
      const float32x4_t vi0x0012 = vreinterpretq_f32_u64(vshlq_n_u64(vreinterpretq_u64_f32(vi0x0123), 32));
      const float32x4_t vi1x0012 = vreinterpretq_f32_u64(vshlq_n_u64(vreinterpretq_u64_f32(vi1x0123), 32));
      const float32x4_t vi2x0012 = vreinterpretq_f32_u64(vshlq_n_u64(vreinterpretq_u64_f32(vi2x0123), 32));
      const float32x4_t vi3x0012 = vreinterpretq_f32_u64(vshlq_n_u64(vreinterpretq_u64_f32(vi3x0123), 32));
      const float32x4_t vi4x0012 = vreinterpretq_f32_u64(vshlq_n_u64(vreinterpretq_u64_f32(vi4x0123), 32));

      const float32x4_t vi0x2468 = vextq_f32(vi0x0012, vi0x468A, 3);
      const float32x4_t vi1x2468 = vextq_f32(vi1x0012, vi1x468A, 3);
      const float32x4_t vi2x2468 = vextq_f32(vi2x0012, vi2x468A, 3);
      const float32x4_t vi3x2468 = vextq_f32(vi3x0012, vi3x468A, 3);
      const float32x4_t vi4x2468 = vextq_f32(vi4x0012, vi4x468A, 3);

      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi0x2468, vw0123, 1);
      vo468Ap01 = vfmaq_laneq_f32(vo468Ap01, vi1x2468, vw4567, 2);
      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi2x2468, vw89AB, 3);
      vo468Ap01 = vfmaq_laneq_f32(vo468Ap01, vi3x2468, vwGHIJ, 0);
      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi4x2468, vwKLMN, 1);

      vi0x0123 = vi0x89AB;
      vi1x0123 = vi1x89AB;
      vi2x0123 = vi2x89AB;
      vi3x0123 = vi3x89AB;
      vi4x0123 = vi4x89AB;

      // one right
      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi0x579B, vw4567, 0);
      vo468Ap01 = vfmaq_laneq_f32(vo468Ap01, vi1x579B, vw89AB, 1);
      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi2x579B, vwCDEF, 2);
      vo468Ap01 = vfmaq_laneq_f32(vo468Ap01, vi3x579B, vwGHIJ, 3);
      vo468Ap00 = vfmaq_lane_f32( vo468Ap00, vi4x579B, vwOP, 0);

      // two right
      const float32x4_t vi0x68AC = vextq_f32(vi0x468A, vi0xCDEF, 1);
      const float32x4_t vi1x68AC = vextq_f32(vi1x468A, vi1xCDEF, 1);
      const float32x4_t vi2x68AC = vextq_f32(vi2x468A, vi2xCDEF, 1);
      const float32x4_t vi3x68AC = vextq_f32(vi3x468A, vi3xCDEF, 1);
      const float32x4_t vi4x68AC = vextq_f32(vi4x468A, vi4xCDEF, 1);

      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi0x68AC, vw4567, 1);
      vo468Ap01 = vfmaq_laneq_f32(vo468Ap01, vi1x68AC, vw89AB, 2);
      vo468Ap00 = vfmaq_laneq_f32(vo468Ap00, vi2x68AC, vwCDEF, 3);
      vo468Ap01 = vfmaq_laneq_f32(vo468Ap01, vi3x68AC, vwKLMN, 0);
      vo468Ap00 = vfmaq_lane_f32( vo468Ap00, vi4x68AC, vwOP, 1);

      vi0x4567 = vi0xCDEF;
      vi1x4567 = vi1xCDEF;
      vi2x4567 = vi2xCDEF;
      vi3x4567 = vi3xCDEF;
      vi4x4567 = vi4xCDEF;

      float32x4_t vo0 = vaddq_f32(vo468Ap00, vo468Ap01);

      vo0 = vmaxq_f32(vo0, vmin);
      vo0 = vminq_f32(vo0, vmax);

      size_t w_tmp = (w + 1) / 2;
      if XNN_LIKELY(w_tmp >= 4) {
        vst1q_f32(output, vo0); output += 4;
      } else {
        float32x2_t vo0_lo = vget_low_f32(vo0);
        if (w_tmp & 2) {
          vst1_f32(output, vo0_lo); output += 2;
          vo0_lo = vget_high_f32(vo0);
        }
        if (w_tmp & 1) {
          vst1_lane_f32(output, vo0_lo, 0); output += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i2 - input_decrement);
    i1 = (const float*) ((uintptr_t) i3 - input_decrement);
    i2 = (const float*) ((uintptr_t) i4 - input_decrement);
    i3 = (const float*) ((uintptr_t) i2 + input_width_stride);
    i4 = (const float*) ((uintptr_t) i3 + input_width_stride);

    output_height -= 1;
    padded_input_height -= 2;
  } while (output_height != 0);
}
