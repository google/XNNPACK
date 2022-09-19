// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv2d-chw/3x3p1-neonfp16arith.c.in
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


void xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_1x8_acc3(
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
  assert(padding_top == 1);

  const uint16x8_t vmask = vld1q_u16(params->neonfp16arith.maskx8);
  const float16x8_t vmax = vld1q_dup_f16(&params->neonfp16arith.max);
  const float16x8_t vmin = vld1q_dup_f16(&params->neonfp16arith.min);

  const __fp16* w0 = (const __fp16*)weights;
  const float16x8_t vw01234567 = vld1q_f16(w0);
  const float16x4_t vw89 = vreinterpret_f16_u32(vld1_dup_u32((const void*)(w0 + 8)));

  const size_t input_decrement = round_up_po2(input_width, 8 * sizeof(__fp16));

  const __fp16* i0 = zero;
  const __fp16* i1 = input;
  const __fp16* i2 = (const __fp16*) ((uintptr_t) i1 + input_width);

  __fp16* o0 = output;

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i2 = zero;
    }

    float16x8_t vi0x01234567 = vmovq_n_f16(0);
    float16x8_t vi1x01234567 = vmovq_n_f16(0);
    float16x8_t vi2x01234567 = vmovq_n_f16(0);

    float16x8_t vi0x89ABCDEF = vld1q_f16(i0); i0 += 8;
    float16x8_t vi1x89ABCDEF = vld1q_f16(i1); i1 += 8;
    float16x8_t vi2x89ABCDEF = vld1q_f16(i2); i2 += 8;

    size_t w = input_width;
    for (; w > 8 * sizeof(__fp16); w -= 8 * sizeof(__fp16)) {
      float16x8_t vo0p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);

      const float16x8_t vi0xGHIJKLMN = vld1q_f16(i0); i0 += 8;
      const float16x8_t vi1xGHIJKLMN = vld1q_f16(i1); i1 += 8;
      const float16x8_t vi2xGHIJKLMN = vld1q_f16(i2); i2 += 8;

      // Center column
      vo0p0 = vfmaq_lane_f16(vo0p0, vi0x89ABCDEF, vget_low_f16(vw01234567), 2);

      float16x8_t vo0p1 = vmulq_lane_f16(vi1x89ABCDEF, vget_high_f16(vw01234567), 1);

      float16x8_t vo0p2 = vmulq_lane_f16(vi2x89ABCDEF, vw89, 0);

      // Left column
      const float16x8_t vi0x789ABCDE = vextq_f16(vi0x01234567, vi0x89ABCDEF, 7);
      const float16x8_t vi1x789ABCDE = vextq_f16(vi1x01234567, vi1x89ABCDEF, 7);
      const float16x8_t vi2x789ABCDE = vextq_f16(vi2x01234567, vi2x89ABCDEF, 7);

      vo0p0 = vfmaq_lane_f16(vo0p0, vi0x789ABCDE, vget_low_f16(vw01234567), 1);

      vo0p1 = vfmaq_lane_f16(vo0p1, vi1x789ABCDE, vget_high_f16(vw01234567), 0);

      vo0p2 = vfmaq_lane_f16(vo0p2, vi2x789ABCDE, vget_high_f16(vw01234567), 3);

      vi0x01234567 = vi0x89ABCDEF;
      vi1x01234567 = vi1x89ABCDEF;
      vi2x01234567 = vi2x89ABCDEF;

      // Right column
      const float16x8_t vi0x9ABCDEFG = vextq_f16(vi0x89ABCDEF, vi0xGHIJKLMN, 1);
      const float16x8_t vi1x9ABCDEFG = vextq_f16(vi1x89ABCDEF, vi1xGHIJKLMN, 1);
      const float16x8_t vi2x9ABCDEFG = vextq_f16(vi2x89ABCDEF, vi2xGHIJKLMN, 1);

      vo0p0 = vfmaq_lane_f16(vo0p0, vi0x9ABCDEFG, vget_low_f16(vw01234567), 3);

      vo0p1 = vfmaq_lane_f16(vo0p1, vi1x9ABCDEFG, vget_high_f16(vw01234567), 2);

      vo0p2 = vfmaq_lane_f16(vo0p2, vi2x9ABCDEFG, vw89, 1);

      vi0x89ABCDEF = vi0xGHIJKLMN;
      vi1x89ABCDEF = vi1xGHIJKLMN;
      vi2x89ABCDEF = vi2xGHIJKLMN;

      vo0p0 = vaddq_f16(vo0p0, vo0p1);
      vo0p0 = vaddq_f16(vo0p0, vo0p2);

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);

      vo0 = vminq_f16(vo0, vmax);

      vst1q_f16(o0, vo0); o0 += 8;
    }

    // Always process the last block of 1..8 pixels.
    assert(w >= 1 * sizeof(__fp16));
    assert(w <= 8 * sizeof(__fp16));
    {
      float16x8_t vo0p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);

      vi0x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi0x89ABCDEF)));
      vi1x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi1x89ABCDEF)));
      vi2x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi2x89ABCDEF)));

      // Center column
      vo0p0 = vfmaq_lane_f16(vo0p0, vi0x89ABCDEF, vget_low_f16(vw01234567), 2);

      float16x8_t vo0p1 = vmulq_lane_f16(vi1x89ABCDEF, vget_high_f16(vw01234567), 1);

      float16x8_t vo0p2 = vmulq_lane_f16(vi2x89ABCDEF, vw89, 0);

      // Left column
      const float16x8_t vi0x789ABCDE = vextq_f16(vi0x01234567, vi0x89ABCDEF, 7);
      const float16x8_t vi1x789ABCDE = vextq_f16(vi1x01234567, vi1x89ABCDEF, 7);
      const float16x8_t vi2x789ABCDE = vextq_f16(vi2x01234567, vi2x89ABCDEF, 7);

      vo0p0 = vfmaq_lane_f16(vo0p0, vi0x789ABCDE, vget_low_f16(vw01234567), 1);

      vo0p1 = vfmaq_lane_f16(vo0p1, vi1x789ABCDE, vget_high_f16(vw01234567), 0);

      vo0p2 = vfmaq_lane_f16(vo0p2, vi2x789ABCDE, vget_high_f16(vw01234567), 3);

      // Right column
      const float16x8_t vzero = vmovq_n_f16(0);
      const float16x8_t vi0x9ABCDEFG = vextq_f16(vi0x89ABCDEF, vzero, 1);
      const float16x8_t vi1x9ABCDEFG = vextq_f16(vi1x89ABCDEF, vzero, 1);
      const float16x8_t vi2x9ABCDEFG = vextq_f16(vi2x89ABCDEF, vzero, 1);

      vo0p0 = vfmaq_lane_f16(vo0p0, vi0x9ABCDEFG, vget_low_f16(vw01234567), 3);

      vo0p1 = vfmaq_lane_f16(vo0p1, vi1x9ABCDEFG, vget_high_f16(vw01234567), 2);

      vo0p2 = vfmaq_lane_f16(vo0p2, vi2x9ABCDEFG, vw89, 1);

      vo0p0 = vaddq_f16(vo0p0, vo0p1);
      vo0p0 = vaddq_f16(vo0p0, vo0p2);

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);

      vo0 = vminq_f16(vo0, vmax);

      if XNN_LIKELY(w == 8 * sizeof(__fp16)) {
        vst1q_f16(o0, vo0); o0 += 8;
      } else {
        float16x4_t vo0_lo = vget_low_f16(vo0);

        if (w & (4 * sizeof(__fp16))) {
         vst1_f16(o0, vo0_lo); o0 += 4;

          vo0_lo = vget_high_f16(vo0);
        }
        if (w & (2 * sizeof(__fp16))) {
          vst1_lane_u32((void*) o0, vreinterpret_u32_f16(vo0_lo), 0); o0 += 2;

          vo0_lo = vext_f16(vo0_lo, vo0_lo, 2);
        }
        if (w & (1 * sizeof(__fp16))) {
          vst1_lane_f16(o0, vo0_lo, 0); o0 += 1;
        }
      }
    }

    i0 = (const __fp16*) ((uintptr_t) i1 - input_decrement);
    i1 = (const __fp16*) ((uintptr_t) i2 - input_decrement);
    i2 = (const __fp16*) ((uintptr_t) i1 + input_width);


  } while (--output_height != 0);
}
