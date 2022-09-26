// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv2d-chw/5x5s2p2-neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>


void xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_1x8_acc3(
    size_t input_height,
    size_t input_width,
    const void* input,
    const void* weights,
    const void* zero,
    void* output,
    uint32_t padding_top,
    const union xnn_f16_chw_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(__fp16) == 0);
  assert(padding_top >= 1);
  assert(padding_top <= 2);

  const uint16x8_t vmask_even = vld1q_u16(params->neonfp16arith.maskx8_even);
  const uint16x8_t vmask_odd = vld1q_u16(params->neonfp16arith.maskx8_odd);
  const float16x8_t vmax = vld1q_dup_f16(&params->neonfp16arith.max);
  const float16x8_t vmin = vld1q_dup_f16(&params->neonfp16arith.min);

  const __fp16* w0 = (const __fp16*)weights;
  const float16x8_t vw01234567 = vld1q_f16(w0);
  const float16x8_t vw89ABCDEF = vld1q_f16(w0 + 8);
  const float16x8_t vwGHIJKLMN = vld1q_f16(w0 + 16);
  const float16x4_t vwOP = vreinterpret_f16_u32(vld1_dup_u32((const void*)(w0 + 24)));

  const uint32_t padding_top_less_1 = padding_top - 1;
  const size_t input_decrement = round_up_po2(input_width, 16 * sizeof(__fp16));

  const __fp16* i0 = zero;
  const __fp16* i1 = (const __fp16*) ((uintptr_t) input - ((-padding_top_less_1) & input_width));
  const __fp16* i2 = (const __fp16*) ((uintptr_t) i1 + input_width);
  if XNN_UNPREDICTABLE(padding_top_less_1 != 0) {
    i1 = zero;
  }
  const __fp16* i3 = (const __fp16*) ((uintptr_t) i2 + input_width);
  const __fp16* i4 = (const __fp16*) ((uintptr_t) i3 + input_width);


  __fp16* o0 = output;

  size_t padded_input_height = input_height + (padding_top_less_1 + 1) + 2 /* padding bottom */;
  size_t output_height = (padded_input_height - 5 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 6) {
      i3 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 7) {
      i4 = zero;
    }

    float16x8_t vi0x0246 = vmovq_n_f16(0);
    float16x8_t vi1x0246 = vmovq_n_f16(0);
    float16x8_t vi2x0246 = vmovq_n_f16(0);
    float16x8_t vi3x0246 = vmovq_n_f16(0);
    float16x8_t vi4x0246 = vmovq_n_f16(0);

    float16x8_t vi0x1357 = vmovq_n_f16(0);
    float16x8_t vi1x1357 = vmovq_n_f16(0);
    float16x8_t vi2x1357 = vmovq_n_f16(0);
    float16x8_t vi3x1357 = vmovq_n_f16(0);
    float16x8_t vi4x1357 = vmovq_n_f16(0);

    float16x8x2_t vi0x8ACE9BDF = vld2q_f16(i0); i0 += 16;
    float16x8x2_t vi1x8ACE9BDF = vld2q_f16(i1); i1 += 16;
    float16x8x2_t vi2x8ACE9BDF = vld2q_f16(i2); i2 += 16;
    float16x8x2_t vi3x8ACE9BDF = vld2q_f16(i3); i3 += 16;
    float16x8x2_t vi4x8ACE9BDF = vld2q_f16(i4); i4 += 16;

    size_t w = input_width;
    for (; w > 16 * sizeof(__fp16); w -= 16 * sizeof(__fp16)) {
      float16x8_t vo0p0 = vdupq_laneq_f16(vw01234567, 0);

      // Center column
      float16x8_t vo0p1 = vmulq_laneq_f16(vi0x8ACE9BDF.val[0], vw01234567, 3);

      float16x8_t vo0p2 = vmulq_laneq_f16(vi1x8ACE9BDF.val[0], vw89ABCDEF, 0);

      vo0p1 = vfmaq_laneq_f16(vo0p1, vi2x8ACE9BDF.val[0], vw89ABCDEF, 5);

      vo0p2 = vfmaq_laneq_f16(vo0p2, vi3x8ACE9BDF.val[0], vwGHIJKLMN, 2);

      vo0p0 = vfmaq_laneq_f16(vo0p0, vi4x8ACE9BDF.val[0], vwGHIJKLMN, 7);

      // Right by 2 column
      vo0p1 = vfmaq_laneq_f16(vo0p1, vi0x8ACE9BDF.val[1], vw01234567, 4);

      vo0p2 = vfmaq_laneq_f16(vo0p2, vi1x8ACE9BDF.val[1], vw89ABCDEF, 1);

      vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x8ACE9BDF.val[1], vw89ABCDEF, 6);

      vo0p1 = vfmaq_laneq_f16(vo0p1, vi3x8ACE9BDF.val[1], vwGHIJKLMN, 3);

      vo0p2 = vfmaq_lane_f16(vo0p2, vi4x8ACE9BDF.val[1], vwOP, 0);

      // Left by 2 column
      const float16x8_t vi0x68AC = vextq_f16(vi0x0246, vi0x8ACE9BDF.val[0], 7);
      vi0x0246 = vi0x8ACE9BDF.val[0];
      const float16x8_t vi1x68AC = vextq_f16(vi1x0246, vi1x8ACE9BDF.val[0], 7);
      vi1x0246 = vi1x8ACE9BDF.val[0];
      const float16x8_t vi2x68AC = vextq_f16(vi2x0246, vi2x8ACE9BDF.val[0], 7);
      vi2x0246 = vi2x8ACE9BDF.val[0];
      const float16x8_t vi3x68AC = vextq_f16(vi3x0246, vi3x8ACE9BDF.val[0], 7);
      vi3x0246 = vi3x8ACE9BDF.val[0];
      const float16x8_t vi4x68AC = vextq_f16(vi4x0246, vi4x8ACE9BDF.val[0], 7);
      vi4x0246 = vi4x8ACE9BDF.val[0];

      vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x68AC, vw01234567, 1);

      vo0p1 = vfmaq_laneq_f16(vo0p1, vi1x68AC, vw01234567, 6);

      vo0p2 = vfmaq_laneq_f16(vo0p2, vi2x68AC, vw89ABCDEF, 3);

      vo0p0 = vfmaq_laneq_f16(vo0p0, vi3x68AC, vwGHIJKLMN, 0);

      vo0p1 = vfmaq_laneq_f16(vo0p1, vi4x68AC, vwGHIJKLMN, 5);

      // Left by 1 column, s1
      const float16x8_t vi0x79BD = vextq_f16(vi0x1357, vi0x8ACE9BDF.val[1], 7);
      vi0x1357 = vi0x8ACE9BDF.val[1];
      const float16x8_t vi1x79BD = vextq_f16(vi1x1357, vi1x8ACE9BDF.val[1], 7);
      vi1x1357 = vi1x8ACE9BDF.val[1];
      const float16x8_t vi2x79BD = vextq_f16(vi2x1357, vi2x8ACE9BDF.val[1], 7);
      vi2x1357 = vi2x8ACE9BDF.val[1];
      const float16x8_t vi3x79BD = vextq_f16(vi3x1357, vi3x8ACE9BDF.val[1], 7);
      vi3x1357 = vi3x8ACE9BDF.val[1];
      const float16x8_t vi4x79BD = vextq_f16(vi4x1357, vi4x8ACE9BDF.val[1], 7);
      vi4x1357 = vi4x8ACE9BDF.val[1];

      const float16x8x2_t vi0xGIKMHJLN = vld2q_f16(i0); i0 += 16;
      const float16x8x2_t vi1xGIKMHJLN = vld2q_f16(i1); i1 += 16;
      const float16x8x2_t vi2xGIKMHJLN = vld2q_f16(i2); i2 += 16;
      const float16x8x2_t vi3xGIKMHJLN = vld2q_f16(i3); i3 += 16;
      const float16x8x2_t vi4xGIKMHJLN = vld2q_f16(i4); i4 += 16;

      vo0p2 = vfmaq_laneq_f16(vo0p2, vi0x79BD, vw01234567, 2);

      vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x79BD, vw01234567, 7);

      vo0p1 = vfmaq_laneq_f16(vo0p1, vi2x79BD, vw89ABCDEF, 4);

      vo0p2 = vfmaq_laneq_f16(vo0p2, vi3x79BD, vwGHIJKLMN, 1);

      vo0p0 = vfmaq_laneq_f16(vo0p0, vi4x79BD, vwGHIJKLMN, 6);

      // Right by 1 column, s0
      const float16x8_t vi0xACEG = vextq_f16(vi0x8ACE9BDF.val[0], vi0xGIKMHJLN.val[0], 1);
      vi0x8ACE9BDF = vi0xGIKMHJLN;
      const float16x8_t vi1xACEG = vextq_f16(vi1x8ACE9BDF.val[0], vi1xGIKMHJLN.val[0], 1);
      vi1x8ACE9BDF = vi1xGIKMHJLN;
      const float16x8_t vi2xACEG = vextq_f16(vi2x8ACE9BDF.val[0], vi2xGIKMHJLN.val[0], 1);
      vi2x8ACE9BDF = vi2xGIKMHJLN;
      const float16x8_t vi3xACEG = vextq_f16(vi3x8ACE9BDF.val[0], vi3xGIKMHJLN.val[0], 1);
      vi3x8ACE9BDF = vi3xGIKMHJLN;
      const float16x8_t vi4xACEG = vextq_f16(vi4x8ACE9BDF.val[0], vi4xGIKMHJLN.val[0], 1);
      vi4x8ACE9BDF = vi4xGIKMHJLN;

      vo0p1 = vfmaq_laneq_f16(vo0p1, vi0xACEG, vw01234567, 5);

      vo0p2 = vfmaq_laneq_f16(vo0p2, vi1xACEG, vw89ABCDEF, 2);

      vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xACEG, vw89ABCDEF, 7);

      vo0p1 = vfmaq_laneq_f16(vo0p1, vi3xACEG, vwGHIJKLMN, 4);

      vo0p2 = vfmaq_lane_f16(vo0p2, vi4xACEG, vwOP, 1);

      vo0p0 = vaddq_f16(vo0p0, vo0p1);
      vo0p0 = vaddq_f16(vo0p0, vo0p2);

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);

      vo0 = vminq_f16(vo0, vmax);

      vst1q_f16(o0, vo0); o0 += 8;
    }

    // Last block has 1-16 pixels to process.
    assert(w <= 16 * sizeof(__fp16));
    assert(w >= 1 * sizeof(__fp16));
    {
      float16x8_t vo0p0 = vdupq_laneq_f16(vw01234567, 0);

      const float16x8_t vi0x8ACE = vreinterpretq_f16_u16(vandq_u16(vmask_even, vreinterpretq_u16_f16(vi0x8ACE9BDF.val[0])));
      const float16x8_t vi1x8ACE = vreinterpretq_f16_u16(vandq_u16(vmask_even, vreinterpretq_u16_f16(vi1x8ACE9BDF.val[0])));
      const float16x8_t vi2x8ACE = vreinterpretq_f16_u16(vandq_u16(vmask_even, vreinterpretq_u16_f16(vi2x8ACE9BDF.val[0])));
      const float16x8_t vi3x8ACE = vreinterpretq_f16_u16(vandq_u16(vmask_even, vreinterpretq_u16_f16(vi3x8ACE9BDF.val[0])));
      const float16x8_t vi4x8ACE = vreinterpretq_f16_u16(vandq_u16(vmask_even, vreinterpretq_u16_f16(vi4x8ACE9BDF.val[0])));

      const float16x8_t vi0x9BDF = vreinterpretq_f16_u16(vandq_u16(vmask_odd, vreinterpretq_u16_f16(vi0x8ACE9BDF.val[1])));
      const float16x8_t vi1x9BDF = vreinterpretq_f16_u16(vandq_u16(vmask_odd, vreinterpretq_u16_f16(vi1x8ACE9BDF.val[1])));
      const float16x8_t vi2x9BDF = vreinterpretq_f16_u16(vandq_u16(vmask_odd, vreinterpretq_u16_f16(vi2x8ACE9BDF.val[1])));
      const float16x8_t vi3x9BDF = vreinterpretq_f16_u16(vandq_u16(vmask_odd, vreinterpretq_u16_f16(vi3x8ACE9BDF.val[1])));
      const float16x8_t vi4x9BDF = vreinterpretq_f16_u16(vandq_u16(vmask_odd, vreinterpretq_u16_f16(vi4x8ACE9BDF.val[1])));

      // Center column
      float16x8_t vo0p1 = vmulq_laneq_f16(vi0x8ACE, vw01234567, 3);

      float16x8_t vo0p2 = vmulq_laneq_f16(vi1x8ACE, vw89ABCDEF, 0);

      vo0p1 = vfmaq_laneq_f16(vo0p1, vi2x8ACE, vw89ABCDEF, 5);

      vo0p2 = vfmaq_laneq_f16(vo0p2, vi3x8ACE, vwGHIJKLMN, 2);

      vo0p0 = vfmaq_laneq_f16(vo0p0, vi4x8ACE, vwGHIJKLMN, 7);

      // Right by 1 column
      vo0p1 = vfmaq_laneq_f16(vo0p1, vi0x9BDF, vw01234567, 4);

      vo0p2 = vfmaq_laneq_f16(vo0p2, vi1x9BDF, vw89ABCDEF, 1);

      vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x9BDF, vw89ABCDEF, 6);

      vo0p1 = vfmaq_laneq_f16(vo0p1, vi3x9BDF, vwGHIJKLMN, 3);

      vo0p2 = vfmaq_lane_f16(vo0p2, vi4x9BDF, vwOP, 0);

      // Left by 2 columns
      const float16x8_t vi0x68AC = vextq_f16(vi0x0246, vi0x8ACE, 7);
      const float16x8_t vi1x68AC = vextq_f16(vi1x0246, vi1x8ACE, 7);
      const float16x8_t vi2x68AC = vextq_f16(vi2x0246, vi2x8ACE, 7);
      const float16x8_t vi3x68AC = vextq_f16(vi3x0246, vi3x8ACE, 7);
      const float16x8_t vi4x68AC = vextq_f16(vi4x0246, vi4x8ACE, 7);

      vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x68AC, vw01234567, 1);

      vo0p1 = vfmaq_laneq_f16(vo0p1, vi1x68AC, vw01234567, 6);

      vo0p2 = vfmaq_laneq_f16(vo0p2, vi2x68AC, vw89ABCDEF, 3);

      vo0p0 = vfmaq_laneq_f16(vo0p0, vi3x68AC, vwGHIJKLMN, 0);

      vo0p1 = vfmaq_laneq_f16(vo0p1, vi4x68AC, vwGHIJKLMN, 5);

      // Left by 1 column
      const float16x8_t vi0x79BD = vextq_f16(vi0x1357, vi0x9BDF, 7);
      const float16x8_t vi1x79BD = vextq_f16(vi1x1357, vi1x9BDF, 7);
      const float16x8_t vi2x79BD = vextq_f16(vi2x1357, vi2x9BDF, 7);
      const float16x8_t vi3x79BD = vextq_f16(vi3x1357, vi3x9BDF, 7);
      const float16x8_t vi4x79BD = vextq_f16(vi4x1357, vi4x9BDF, 7);

      vo0p2 = vfmaq_laneq_f16(vo0p2, vi0x79BD, vw01234567, 2);

      vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x79BD, vw01234567, 7);

      vo0p1 = vfmaq_laneq_f16(vo0p1, vi2x79BD, vw89ABCDEF, 4);

      vo0p2 = vfmaq_laneq_f16(vo0p2, vi3x79BD, vwGHIJKLMN, 1);

      vo0p0 = vfmaq_laneq_f16(vo0p0, vi4x79BD, vwGHIJKLMN, 6);

      // Right by 2 columns
      const float16x8_t vzero = vmovq_n_f16(0);
      const float16x8_t vi0xACEG = vextq_f16(vi0x8ACE, vzero, 1);
      const float16x8_t vi1xACEG = vextq_f16(vi1x8ACE, vzero, 1);
      const float16x8_t vi2xACEG = vextq_f16(vi2x8ACE, vzero, 1);
      const float16x8_t vi3xACEG = vextq_f16(vi3x8ACE, vzero, 1);
      const float16x8_t vi4xACEG = vextq_f16(vi4x8ACE, vzero, 1);

      vo0p1 = vfmaq_laneq_f16(vo0p1, vi0xACEG, vw01234567, 5);

      vo0p2 = vfmaq_laneq_f16(vo0p2, vi1xACEG, vw89ABCDEF, 2);

      vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xACEG, vw89ABCDEF, 7);

      vo0p1 = vfmaq_laneq_f16(vo0p1, vi3xACEG, vwGHIJKLMN, 4);

      vo0p2 = vfmaq_lane_f16(vo0p2, vi4xACEG, vwOP, 1);

      vo0p0 = vaddq_f16(vo0p0, vo0p1);
      vo0p0 = vaddq_f16(vo0p0, vo0p2);

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);

      vo0 = vminq_f16(vo0, vmax);

      const size_t w_tmp = (w + 1 * sizeof(__fp16)) / (2 * sizeof(__fp16));

      if XNN_LIKELY(w_tmp == 8) {
        vst1q_f16(o0, vo0); o0 += 8;
      } else {
        float16x4_t vo0_lo = vget_low_f16(vo0);

        if (w_tmp & 4) {
         vst1_f16(o0, vo0_lo); o0 += 4;

          vo0_lo = vget_high_f16(vo0);
        }
        if (w_tmp & 2) {
          vst1_lane_u32((void*) o0, vreinterpret_u32_f16(vo0_lo), 0); o0 += 2;

          vo0_lo = vext_f16(vo0_lo, vo0_lo, 2);
        }
        if (w_tmp & 1) {
          vst1_lane_f16(o0, vo0_lo, 0); o0 += 1;
        }
      }
    }

    i0 = (const __fp16*) ((uintptr_t) i2 - input_decrement);
    i1 = (const __fp16*) ((uintptr_t) i3 - input_decrement);
    i2 = (const __fp16*) ((uintptr_t) i4 - input_decrement);
    i3 = (const __fp16*) ((uintptr_t) i2 + input_width);
    i4 = (const __fp16*) ((uintptr_t) i3 + input_width);


    output_height -= 1;
    padded_input_height -= 2;
  } while (output_height != 0);
}
