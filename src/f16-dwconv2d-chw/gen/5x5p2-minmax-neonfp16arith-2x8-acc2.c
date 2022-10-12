// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv2d-chw/5x5p2-neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>


void xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_2x8_acc2(
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
  assert(padding_top == 2);

  const uint16x8_t vmask = vld1q_u16(params->neonfp16arith.mask);
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->neonfp16arith.max));
  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->neonfp16arith.min));

  const __fp16* w0 = (const __fp16*)weights;
  const float16x8_t vw01234567 = vld1q_f16(w0);
  const float16x8_t vw89ABCDEF = vld1q_f16(w0 + 8);
  const float16x8_t vwGHIJKLMN = vld1q_f16(w0 + 16);
  const float16x4_t vwOP = vreinterpret_f16_u32(vld1_dup_u32((const void*)(w0 + 24)));

  const size_t input_decrement = round_up_po2(input_width, 8 * sizeof(__fp16));

  const __fp16* i0 = zero;
  const __fp16* i1 = zero;
  const __fp16* i2 = input;
  const __fp16* i3 = (const __fp16*) ((uintptr_t) i2 + input_width);
  const __fp16* i4 = (const __fp16*) ((uintptr_t) i3 + input_width);
  const __fp16* i5 = (const __fp16*) ((uintptr_t) i4 + input_width);

  __fp16* o0 = output;
  __fp16* o1 = (__fp16*) ((uintptr_t) o0 + input_width);

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i3 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(output_height < 3) {
      i4 = zero;
    }
    if XNN_UNPREDICTABLE(output_height < 4) {
      i5 = zero;
    }

    float16x8_t vi0x01234567 = vmovq_n_f16(0);
    float16x8_t vi1x01234567 = vmovq_n_f16(0);
    float16x8_t vi2x01234567 = vmovq_n_f16(0);
    float16x8_t vi3x01234567 = vmovq_n_f16(0);
    float16x8_t vi4x01234567 = vmovq_n_f16(0);
    float16x8_t vi5x01234567 = vmovq_n_f16(0);

    float16x8_t vi0x89ABCDEF = vld1q_f16(i0); i0 += 8;
    float16x8_t vi1x89ABCDEF = vld1q_f16(i1); i1 += 8;
    float16x8_t vi2x89ABCDEF = vld1q_f16(i2); i2 += 8;
    float16x8_t vi3x89ABCDEF = vld1q_f16(i3); i3 += 8;
    float16x8_t vi4x89ABCDEF = vld1q_f16(i4); i4 += 8;
    float16x8_t vi5x89ABCDEF = vld1q_f16(i5); i5 += 8;

    size_t w = input_width;
    for (; w > 16 * sizeof(__fp16); w -= 8 * sizeof(__fp16)) {
      float16x8_t vo0p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);
      float16x8_t vo1p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);

      const float16x8_t vi0xGHIJKLMN = vld1q_f16(i0); i0 += 8;
      const float16x8_t vi1xGHIJKLMN = vld1q_f16(i1); i1 += 8;
      const float16x8_t vi2xGHIJKLMN = vld1q_f16(i2); i2 += 8;
      const float16x8_t vi3xGHIJKLMN = vld1q_f16(i3); i3 += 8;
      const float16x8_t vi4xGHIJKLMN = vld1q_f16(i4); i4 += 8;
      const float16x8_t vi5xGHIJKLMN = vld1q_f16(i5); i5 += 8;

      // Center column
      float16x8_t vo0p1 = vmulq_lane_f16(vi0x89ABCDEF, vget_low_f16(vw01234567), 3);
      float16x8_t vo1p1 = vmulq_lane_f16(vi1x89ABCDEF, vget_low_f16(vw01234567), 3);
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x89ABCDEF, vw89ABCDEF, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x89ABCDEF, vget_low_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2x89ABCDEF, vw89ABCDEF, 0);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2x89ABCDEF, vget_low_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x89ABCDEF, vw89ABCDEF, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x89ABCDEF, vget_high_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi3x89ABCDEF, vw89ABCDEF, 5);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3x89ABCDEF, vget_high_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi3x89ABCDEF, vwGHIJKLMN, 2);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi3x89ABCDEF, vget_low_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi4x89ABCDEF, vwGHIJKLMN, 2);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi4x89ABCDEF, vget_low_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4x89ABCDEF, vwGHIJKLMN, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4x89ABCDEF, vget_high_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi5x89ABCDEF, vwGHIJKLMN, 7);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi5x89ABCDEF, vget_high_f16(vwGHIJKLMN), 3);
      #endif
      // Left by 1 column
      const float16x8_t vi0x789ABCDE = vextq_f16(vi0x01234567, vi0x89ABCDEF, 7);
      const float16x8_t vi1x789ABCDE = vextq_f16(vi1x01234567, vi1x89ABCDEF, 7);
      const float16x8_t vi2x789ABCDE = vextq_f16(vi2x01234567, vi2x89ABCDEF, 7);
      const float16x8_t vi3x789ABCDE = vextq_f16(vi3x01234567, vi3x89ABCDEF, 7);
      const float16x8_t vi4x789ABCDE = vextq_f16(vi4x01234567, vi4x89ABCDEF, 7);
      const float16x8_t vi5x789ABCDE = vextq_f16(vi5x01234567, vi5x89ABCDEF, 7);

      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi0x789ABCDE, vw01234567, 2);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi0x789ABCDE, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi1x789ABCDE, vw01234567, 2);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi1x789ABCDE, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x789ABCDE, vw01234567, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x789ABCDE, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2x789ABCDE, vw01234567, 7);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2x789ABCDE, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi2x789ABCDE, vw89ABCDEF, 4);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi2x789ABCDE, vget_high_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi3x789ABCDE, vw89ABCDEF, 4);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi3x789ABCDE, vget_high_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3x789ABCDE, vwGHIJKLMN, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3x789ABCDE, vget_low_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi4x789ABCDE, vwGHIJKLMN, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi4x789ABCDE, vget_low_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi4x789ABCDE, vwGHIJKLMN, 6);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi4x789ABCDE, vget_high_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi5x789ABCDE, vwGHIJKLMN, 6);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi5x789ABCDE, vget_high_f16(vwGHIJKLMN), 2);
      #endif
      // Left by 2 column
      const float16x8_t vi0x6789ABCD = vextq_f16(vi0x01234567, vi0x89ABCDEF, 6);
      vi0x01234567 = vi0x89ABCDEF;
      const float16x8_t vi1x6789ABCD = vextq_f16(vi1x01234567, vi1x89ABCDEF, 6);
      vi1x01234567 = vi1x89ABCDEF;
      const float16x8_t vi2x6789ABCD = vextq_f16(vi2x01234567, vi2x89ABCDEF, 6);
      vi2x01234567 = vi2x89ABCDEF;
      const float16x8_t vi3x6789ABCD = vextq_f16(vi3x01234567, vi3x89ABCDEF, 6);
      vi3x01234567 = vi3x89ABCDEF;
      const float16x8_t vi4x6789ABCD = vextq_f16(vi4x01234567, vi4x89ABCDEF, 6);
      vi4x01234567 = vi4x89ABCDEF;
      const float16x8_t vi5x6789ABCD = vextq_f16(vi5x01234567, vi5x89ABCDEF, 6);
      vi5x01234567 = vi5x89ABCDEF;

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x6789ABCD, vw01234567, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0x6789ABCD, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi1x6789ABCD, vw01234567, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi1x6789ABCD, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi1x6789ABCD, vw01234567, 6);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi1x6789ABCD, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi2x6789ABCD, vw01234567, 6);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi2x6789ABCD, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x6789ABCD, vw89ABCDEF, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x6789ABCD, vget_low_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi3x6789ABCD, vw89ABCDEF, 3);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3x6789ABCD, vget_low_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi3x6789ABCD, vwGHIJKLMN, 0);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi3x6789ABCD, vget_low_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi4x6789ABCD, vwGHIJKLMN, 0);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi4x6789ABCD, vget_low_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4x6789ABCD, vwGHIJKLMN, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4x6789ABCD, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi5x6789ABCD, vwGHIJKLMN, 5);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi5x6789ABCD, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      // Right by 1 column
      const float16x8_t vi0x9ABCDEFG = vextq_f16(vi0x89ABCDEF, vi0xGHIJKLMN, 1);
      const float16x8_t vi1x9ABCDEFG = vextq_f16(vi1x89ABCDEF, vi1xGHIJKLMN, 1);
      const float16x8_t vi2x9ABCDEFG = vextq_f16(vi2x89ABCDEF, vi2xGHIJKLMN, 1);
      const float16x8_t vi3x9ABCDEFG = vextq_f16(vi3x89ABCDEF, vi3xGHIJKLMN, 1);
      const float16x8_t vi4x9ABCDEFG = vextq_f16(vi4x89ABCDEF, vi4xGHIJKLMN, 1);
      const float16x8_t vi5x9ABCDEFG = vextq_f16(vi5x89ABCDEF, vi5xGHIJKLMN, 1);

      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi0x9ABCDEFG, vw01234567, 4);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi0x9ABCDEFG, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi1x9ABCDEFG, vw01234567, 4);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi1x9ABCDEFG, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x9ABCDEFG, vw89ABCDEF, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x9ABCDEFG, vget_low_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2x9ABCDEFG, vw89ABCDEF, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2x9ABCDEFG, vget_low_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi2x9ABCDEFG, vw89ABCDEF, 6);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi2x9ABCDEFG, vget_high_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi3x9ABCDEFG, vw89ABCDEF, 6);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi3x9ABCDEFG, vget_high_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3x9ABCDEFG, vwGHIJKLMN, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3x9ABCDEFG, vget_low_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi4x9ABCDEFG, vwGHIJKLMN, 3);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi4x9ABCDEFG, vget_low_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_lane_f16(vo0p1, vi4x9ABCDEFG, vwOP, 0);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi4x9ABCDEFG, vwOP, 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_lane_f16(vo1p1, vi5x9ABCDEFG, vwOP, 0);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi5x9ABCDEFG, vwOP, 0);
      #endif
      // Right by 2 column
      const float16x8_t vi0xABCDEFGH = vextq_f16(vi0x89ABCDEF, vi0xGHIJKLMN, 2);
      vi0x89ABCDEF = vi0xGHIJKLMN;
      const float16x8_t vi1xABCDEFGH = vextq_f16(vi1x89ABCDEF, vi1xGHIJKLMN, 2);
      vi1x89ABCDEF = vi1xGHIJKLMN;
      const float16x8_t vi2xABCDEFGH = vextq_f16(vi2x89ABCDEF, vi2xGHIJKLMN, 2);
      vi2x89ABCDEF = vi2xGHIJKLMN;
      const float16x8_t vi3xABCDEFGH = vextq_f16(vi3x89ABCDEF, vi3xGHIJKLMN, 2);
      vi3x89ABCDEF = vi3xGHIJKLMN;
      const float16x8_t vi4xABCDEFGH = vextq_f16(vi4x89ABCDEF, vi4xGHIJKLMN, 2);
      vi4x89ABCDEF = vi4xGHIJKLMN;
      const float16x8_t vi5xABCDEFGH = vextq_f16(vi5x89ABCDEF, vi5xGHIJKLMN, 2);
      vi5x89ABCDEF = vi5xGHIJKLMN;

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xABCDEFGH, vw01234567, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xABCDEFGH, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi1xABCDEFGH, vw01234567, 5);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi1xABCDEFGH, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi1xABCDEFGH, vw89ABCDEF, 2);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi1xABCDEFGH, vget_low_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi2xABCDEFGH, vw89ABCDEF, 2);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi2xABCDEFGH, vget_low_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xABCDEFGH, vw89ABCDEF, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xABCDEFGH, vget_high_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi3xABCDEFGH, vw89ABCDEF, 7);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3xABCDEFGH, vget_high_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi3xABCDEFGH, vwGHIJKLMN, 4);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi3xABCDEFGH, vget_high_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi4xABCDEFGH, vwGHIJKLMN, 4);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi4xABCDEFGH, vget_high_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vi4xABCDEFGH, vwOP, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xABCDEFGH, vwOP, 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_lane_f16(vo1p0, vi5xABCDEFGH, vwOP, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi5xABCDEFGH, vwOP, 1);
      #endif
      vo0p0 = vaddq_f16(vo0p0, vo0p1);
      vo1p0 = vaddq_f16(vo1p0, vo1p1);

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);
      float16x8_t vo1 = vmaxq_f16(vo1p0, vmin);

      vo0 = vminq_f16(vo0, vmax);
      vo1 = vminq_f16(vo1, vmax);

      vst1q_f16(o1, vo1); o1 += 8;
      vst1q_f16(o0, vo0); o0 += 8;
    }

    // Always process the last block of 5..16 pixels.
    assert(w <= 16 * sizeof(__fp16));
    if XNN_LIKELY(w > 8 * sizeof(__fp16)) {
      float16x8_t vo0p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);
      float16x8_t vo1p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);

      float16x8_t vi0xGHIJKLMN = vld1q_f16(i0); i0 += 8;
      float16x8_t vi1xGHIJKLMN = vld1q_f16(i1); i1 += 8;
      float16x8_t vi2xGHIJKLMN = vld1q_f16(i2); i2 += 8;
      float16x8_t vi3xGHIJKLMN = vld1q_f16(i3); i3 += 8;
      float16x8_t vi4xGHIJKLMN = vld1q_f16(i4); i4 += 8;
      float16x8_t vi5xGHIJKLMN = vld1q_f16(i5); i5 += 8;

      vi0xGHIJKLMN = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi0xGHIJKLMN)));
      vi1xGHIJKLMN = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi1xGHIJKLMN)));
      vi2xGHIJKLMN = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi2xGHIJKLMN)));
      vi3xGHIJKLMN = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi3xGHIJKLMN)));
      vi4xGHIJKLMN = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi4xGHIJKLMN)));
      vi5xGHIJKLMN = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi5xGHIJKLMN)));

      // Center column
      float16x8_t vo0p1 = vmulq_lane_f16(vi0x89ABCDEF, vget_low_f16(vw01234567), 3);
      float16x8_t vo1p1 = vmulq_lane_f16(vi1x89ABCDEF, vget_low_f16(vw01234567), 3);
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x89ABCDEF, vw89ABCDEF, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x89ABCDEF, vget_low_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2x89ABCDEF, vw89ABCDEF, 0);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2x89ABCDEF, vget_low_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x89ABCDEF, vw89ABCDEF, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x89ABCDEF, vget_high_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi3x89ABCDEF, vw89ABCDEF, 5);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3x89ABCDEF, vget_high_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi3x89ABCDEF, vwGHIJKLMN, 2);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi3x89ABCDEF, vget_low_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi4x89ABCDEF, vwGHIJKLMN, 2);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi4x89ABCDEF, vget_low_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4x89ABCDEF, vwGHIJKLMN, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4x89ABCDEF, vget_high_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi5x89ABCDEF, vwGHIJKLMN, 7);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi5x89ABCDEF, vget_high_f16(vwGHIJKLMN), 3);
      #endif
      // Left by 1 column
      const float16x8_t vi0x789ABCDE = vextq_f16(vi0x01234567, vi0x89ABCDEF, 7);
      const float16x8_t vi1x789ABCDE = vextq_f16(vi1x01234567, vi1x89ABCDEF, 7);
      const float16x8_t vi2x789ABCDE = vextq_f16(vi2x01234567, vi2x89ABCDEF, 7);
      const float16x8_t vi3x789ABCDE = vextq_f16(vi3x01234567, vi3x89ABCDEF, 7);
      const float16x8_t vi4x789ABCDE = vextq_f16(vi4x01234567, vi4x89ABCDEF, 7);
      const float16x8_t vi5x789ABCDE = vextq_f16(vi5x01234567, vi5x89ABCDEF, 7);

      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi0x789ABCDE, vw01234567, 2);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi0x789ABCDE, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi1x789ABCDE, vw01234567, 2);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi1x789ABCDE, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x789ABCDE, vw01234567, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x789ABCDE, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2x789ABCDE, vw01234567, 7);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2x789ABCDE, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi2x789ABCDE, vw89ABCDEF, 4);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi2x789ABCDE, vget_high_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi3x789ABCDE, vw89ABCDEF, 4);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi3x789ABCDE, vget_high_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3x789ABCDE, vwGHIJKLMN, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3x789ABCDE, vget_low_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi4x789ABCDE, vwGHIJKLMN, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi4x789ABCDE, vget_low_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi4x789ABCDE, vwGHIJKLMN, 6);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi4x789ABCDE, vget_high_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi5x789ABCDE, vwGHIJKLMN, 6);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi5x789ABCDE, vget_high_f16(vwGHIJKLMN), 2);
      #endif
      // Left by 2 column
      const float16x8_t vi0x6789ABCD = vextq_f16(vi0x01234567, vi0x89ABCDEF, 6);
      vi0x01234567 = vi0x89ABCDEF;
      const float16x8_t vi1x6789ABCD = vextq_f16(vi1x01234567, vi1x89ABCDEF, 6);
      vi1x01234567 = vi1x89ABCDEF;
      const float16x8_t vi2x6789ABCD = vextq_f16(vi2x01234567, vi2x89ABCDEF, 6);
      vi2x01234567 = vi2x89ABCDEF;
      const float16x8_t vi3x6789ABCD = vextq_f16(vi3x01234567, vi3x89ABCDEF, 6);
      vi3x01234567 = vi3x89ABCDEF;
      const float16x8_t vi4x6789ABCD = vextq_f16(vi4x01234567, vi4x89ABCDEF, 6);
      vi4x01234567 = vi4x89ABCDEF;
      const float16x8_t vi5x6789ABCD = vextq_f16(vi5x01234567, vi5x89ABCDEF, 6);
      vi5x01234567 = vi5x89ABCDEF;

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x6789ABCD, vw01234567, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0x6789ABCD, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi1x6789ABCD, vw01234567, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi1x6789ABCD, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi1x6789ABCD, vw01234567, 6);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi1x6789ABCD, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi2x6789ABCD, vw01234567, 6);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi2x6789ABCD, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x6789ABCD, vw89ABCDEF, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x6789ABCD, vget_low_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi3x6789ABCD, vw89ABCDEF, 3);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3x6789ABCD, vget_low_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi3x6789ABCD, vwGHIJKLMN, 0);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi3x6789ABCD, vget_low_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi4x6789ABCD, vwGHIJKLMN, 0);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi4x6789ABCD, vget_low_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4x6789ABCD, vwGHIJKLMN, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4x6789ABCD, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi5x6789ABCD, vwGHIJKLMN, 5);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi5x6789ABCD, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      // Right by 1 column
      const float16x8_t vi0x9ABCDEFG = vextq_f16(vi0x89ABCDEF, vi0xGHIJKLMN, 1);
      const float16x8_t vi1x9ABCDEFG = vextq_f16(vi1x89ABCDEF, vi1xGHIJKLMN, 1);
      const float16x8_t vi2x9ABCDEFG = vextq_f16(vi2x89ABCDEF, vi2xGHIJKLMN, 1);
      const float16x8_t vi3x9ABCDEFG = vextq_f16(vi3x89ABCDEF, vi3xGHIJKLMN, 1);
      const float16x8_t vi4x9ABCDEFG = vextq_f16(vi4x89ABCDEF, vi4xGHIJKLMN, 1);
      const float16x8_t vi5x9ABCDEFG = vextq_f16(vi5x89ABCDEF, vi5xGHIJKLMN, 1);

      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi0x9ABCDEFG, vw01234567, 4);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi0x9ABCDEFG, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi1x9ABCDEFG, vw01234567, 4);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi1x9ABCDEFG, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x9ABCDEFG, vw89ABCDEF, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x9ABCDEFG, vget_low_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2x9ABCDEFG, vw89ABCDEF, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2x9ABCDEFG, vget_low_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi2x9ABCDEFG, vw89ABCDEF, 6);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi2x9ABCDEFG, vget_high_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi3x9ABCDEFG, vw89ABCDEF, 6);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi3x9ABCDEFG, vget_high_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3x9ABCDEFG, vwGHIJKLMN, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3x9ABCDEFG, vget_low_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi4x9ABCDEFG, vwGHIJKLMN, 3);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi4x9ABCDEFG, vget_low_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_lane_f16(vo0p1, vi4x9ABCDEFG, vwOP, 0);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi4x9ABCDEFG, vwOP, 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_lane_f16(vo1p1, vi5x9ABCDEFG, vwOP, 0);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi5x9ABCDEFG, vwOP, 0);
      #endif
      // Right by 2 column
      const float16x8_t vi0xABCDEFGH = vextq_f16(vi0x89ABCDEF, vi0xGHIJKLMN, 2);
      vi0x89ABCDEF = vi0xGHIJKLMN;
      const float16x8_t vi1xABCDEFGH = vextq_f16(vi1x89ABCDEF, vi1xGHIJKLMN, 2);
      vi1x89ABCDEF = vi1xGHIJKLMN;
      const float16x8_t vi2xABCDEFGH = vextq_f16(vi2x89ABCDEF, vi2xGHIJKLMN, 2);
      vi2x89ABCDEF = vi2xGHIJKLMN;
      const float16x8_t vi3xABCDEFGH = vextq_f16(vi3x89ABCDEF, vi3xGHIJKLMN, 2);
      vi3x89ABCDEF = vi3xGHIJKLMN;
      const float16x8_t vi4xABCDEFGH = vextq_f16(vi4x89ABCDEF, vi4xGHIJKLMN, 2);
      vi4x89ABCDEF = vi4xGHIJKLMN;
      const float16x8_t vi5xABCDEFGH = vextq_f16(vi5x89ABCDEF, vi5xGHIJKLMN, 2);
      vi5x89ABCDEF = vi5xGHIJKLMN;

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xABCDEFGH, vw01234567, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xABCDEFGH, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi1xABCDEFGH, vw01234567, 5);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi1xABCDEFGH, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi1xABCDEFGH, vw89ABCDEF, 2);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi1xABCDEFGH, vget_low_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi2xABCDEFGH, vw89ABCDEF, 2);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi2xABCDEFGH, vget_low_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xABCDEFGH, vw89ABCDEF, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xABCDEFGH, vget_high_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi3xABCDEFGH, vw89ABCDEF, 7);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3xABCDEFGH, vget_high_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi3xABCDEFGH, vwGHIJKLMN, 4);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi3xABCDEFGH, vget_high_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi4xABCDEFGH, vwGHIJKLMN, 4);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi4xABCDEFGH, vget_high_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vi4xABCDEFGH, vwOP, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xABCDEFGH, vwOP, 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_lane_f16(vo1p0, vi5xABCDEFGH, vwOP, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi5xABCDEFGH, vwOP, 1);
      #endif
      vo0p0 = vaddq_f16(vo0p0, vo0p1);
      vo1p0 = vaddq_f16(vo1p0, vo1p1);

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);
      float16x8_t vo1 = vmaxq_f16(vo1p0, vmin);

      vo0 = vminq_f16(vo0, vmax);
      vo1 = vminq_f16(vo1, vmax);

      vst1q_f16(o1, vo1); o1 += 8;
      vst1q_f16(o0, vo0); o0 += 8;

      w -= 8 * sizeof(__fp16);
    }

    assert(w >= 1 * sizeof(__fp16));
    assert(w <= 8 * sizeof(__fp16));
    {
      float16x8_t vo0p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);
      float16x8_t vo1p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);

      vi0x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi0x89ABCDEF)));
      vi1x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi1x89ABCDEF)));
      vi2x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi2x89ABCDEF)));
      vi3x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi3x89ABCDEF)));
      vi4x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi4x89ABCDEF)));
      vi5x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi5x89ABCDEF)));

      // Center column
      float16x8_t vo0p1 = vmulq_lane_f16(vi0x89ABCDEF, vget_low_f16(vw01234567), 3);
      float16x8_t vo1p1 = vmulq_lane_f16(vi1x89ABCDEF, vget_low_f16(vw01234567), 3);
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x89ABCDEF, vw89ABCDEF, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x89ABCDEF, vget_low_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2x89ABCDEF, vw89ABCDEF, 0);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2x89ABCDEF, vget_low_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x89ABCDEF, vw89ABCDEF, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x89ABCDEF, vget_high_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi3x89ABCDEF, vw89ABCDEF, 5);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3x89ABCDEF, vget_high_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi3x89ABCDEF, vwGHIJKLMN, 2);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi3x89ABCDEF, vget_low_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi4x89ABCDEF, vwGHIJKLMN, 2);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi4x89ABCDEF, vget_low_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4x89ABCDEF, vwGHIJKLMN, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4x89ABCDEF, vget_high_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi5x89ABCDEF, vwGHIJKLMN, 7);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi5x89ABCDEF, vget_high_f16(vwGHIJKLMN), 3);
      #endif
      // Left by 1 column
      const float16x8_t vi0x789ABCDE = vextq_f16(vi0x01234567, vi0x89ABCDEF, 7);
      const float16x8_t vi1x789ABCDE = vextq_f16(vi1x01234567, vi1x89ABCDEF, 7);
      const float16x8_t vi2x789ABCDE = vextq_f16(vi2x01234567, vi2x89ABCDEF, 7);
      const float16x8_t vi3x789ABCDE = vextq_f16(vi3x01234567, vi3x89ABCDEF, 7);
      const float16x8_t vi4x789ABCDE = vextq_f16(vi4x01234567, vi4x89ABCDEF, 7);
      const float16x8_t vi5x789ABCDE = vextq_f16(vi5x01234567, vi5x89ABCDEF, 7);

      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi0x789ABCDE, vw01234567, 2);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi0x789ABCDE, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi1x789ABCDE, vw01234567, 2);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi1x789ABCDE, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x789ABCDE, vw01234567, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x789ABCDE, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2x789ABCDE, vw01234567, 7);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2x789ABCDE, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi2x789ABCDE, vw89ABCDEF, 4);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi2x789ABCDE, vget_high_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi3x789ABCDE, vw89ABCDEF, 4);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi3x789ABCDE, vget_high_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3x789ABCDE, vwGHIJKLMN, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3x789ABCDE, vget_low_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi4x789ABCDE, vwGHIJKLMN, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi4x789ABCDE, vget_low_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi4x789ABCDE, vwGHIJKLMN, 6);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi4x789ABCDE, vget_high_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi5x789ABCDE, vwGHIJKLMN, 6);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi5x789ABCDE, vget_high_f16(vwGHIJKLMN), 2);
      #endif
      // Left by 2 column
      const float16x8_t vi0x6789ABCD = vextq_f16(vi0x01234567, vi0x89ABCDEF, 6);
      const float16x8_t vi1x6789ABCD = vextq_f16(vi1x01234567, vi1x89ABCDEF, 6);
      const float16x8_t vi2x6789ABCD = vextq_f16(vi2x01234567, vi2x89ABCDEF, 6);
      const float16x8_t vi3x6789ABCD = vextq_f16(vi3x01234567, vi3x89ABCDEF, 6);
      const float16x8_t vi4x6789ABCD = vextq_f16(vi4x01234567, vi4x89ABCDEF, 6);
      const float16x8_t vi5x6789ABCD = vextq_f16(vi5x01234567, vi5x89ABCDEF, 6);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0x6789ABCD, vw01234567, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0x6789ABCD, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi1x6789ABCD, vw01234567, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi1x6789ABCD, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi1x6789ABCD, vw01234567, 6);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi1x6789ABCD, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi2x6789ABCD, vw01234567, 6);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi2x6789ABCD, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2x6789ABCD, vw89ABCDEF, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2x6789ABCD, vget_low_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi3x6789ABCD, vw89ABCDEF, 3);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3x6789ABCD, vget_low_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi3x6789ABCD, vwGHIJKLMN, 0);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi3x6789ABCD, vget_low_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi4x6789ABCD, vwGHIJKLMN, 0);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi4x6789ABCD, vget_low_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4x6789ABCD, vwGHIJKLMN, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4x6789ABCD, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi5x6789ABCD, vwGHIJKLMN, 5);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi5x6789ABCD, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      // Right by 1 column
      const float16x8_t vzero = vmovq_n_f16(0);
      const float16x8_t vi0x9ABCDEFG = vextq_f16(vi0x89ABCDEF, vzero, 1);
      const float16x8_t vi1x9ABCDEFG = vextq_f16(vi1x89ABCDEF, vzero, 1);
      const float16x8_t vi2x9ABCDEFG = vextq_f16(vi2x89ABCDEF, vzero, 1);
      const float16x8_t vi3x9ABCDEFG = vextq_f16(vi3x89ABCDEF, vzero, 1);
      const float16x8_t vi4x9ABCDEFG = vextq_f16(vi4x89ABCDEF, vzero, 1);
      const float16x8_t vi5x9ABCDEFG = vextq_f16(vi5x89ABCDEF, vzero, 1);

      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi0x9ABCDEFG, vw01234567, 4);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi0x9ABCDEFG, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi1x9ABCDEFG, vw01234567, 4);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi1x9ABCDEFG, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1x9ABCDEFG, vw89ABCDEF, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1x9ABCDEFG, vget_low_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2x9ABCDEFG, vw89ABCDEF, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2x9ABCDEFG, vget_low_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi2x9ABCDEFG, vw89ABCDEF, 6);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi2x9ABCDEFG, vget_high_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi3x9ABCDEFG, vw89ABCDEF, 6);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi3x9ABCDEFG, vget_high_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3x9ABCDEFG, vwGHIJKLMN, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3x9ABCDEFG, vget_low_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi4x9ABCDEFG, vwGHIJKLMN, 3);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi4x9ABCDEFG, vget_low_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_lane_f16(vo0p1, vi4x9ABCDEFG, vwOP, 0);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi4x9ABCDEFG, vwOP, 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_lane_f16(vo1p1, vi5x9ABCDEFG, vwOP, 0);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi5x9ABCDEFG, vwOP, 0);
      #endif
      // Right by 2 column
      const float16x8_t vi0xABCDEFGH = vextq_f16(vi0x9ABCDEFG, vzero, 1);
      const float16x8_t vi1xABCDEFGH = vextq_f16(vi1x9ABCDEFG, vzero, 1);
      const float16x8_t vi2xABCDEFGH = vextq_f16(vi2x9ABCDEFG, vzero, 1);
      const float16x8_t vi3xABCDEFGH = vextq_f16(vi3x9ABCDEFG, vzero, 1);
      const float16x8_t vi4xABCDEFGH = vextq_f16(vi4x9ABCDEFG, vzero, 1);
      const float16x8_t vi5xABCDEFGH = vextq_f16(vi5x9ABCDEFG, vzero, 1);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xABCDEFGH, vw01234567, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xABCDEFGH, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi1xABCDEFGH, vw01234567, 5);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi1xABCDEFGH, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi1xABCDEFGH, vw89ABCDEF, 2);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi1xABCDEFGH, vget_low_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi2xABCDEFGH, vw89ABCDEF, 2);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi2xABCDEFGH, vget_low_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xABCDEFGH, vw89ABCDEF, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xABCDEFGH, vget_high_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi3xABCDEFGH, vw89ABCDEF, 7);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3xABCDEFGH, vget_high_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi3xABCDEFGH, vwGHIJKLMN, 4);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi3xABCDEFGH, vget_high_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p1 = vfmaq_laneq_f16(vo1p1, vi4xABCDEFGH, vwGHIJKLMN, 4);
      #else
        vo1p1 = vmlaq_lane_f16(vo1p1, vi4xABCDEFGH, vget_high_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vi4xABCDEFGH, vwOP, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xABCDEFGH, vwOP, 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_lane_f16(vo1p0, vi5xABCDEFGH, vwOP, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi5xABCDEFGH, vwOP, 1);
      #endif
      vo0p0 = vaddq_f16(vo0p0, vo0p1);
      vo1p0 = vaddq_f16(vo1p0, vo1p1);

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);
      float16x8_t vo1 = vmaxq_f16(vo1p0, vmin);

      vo0 = vminq_f16(vo0, vmax);
      vo1 = vminq_f16(vo1, vmax);

      if XNN_LIKELY(w == 8 * sizeof(__fp16)) {
        vst1q_f16(o1, vo1); o1 += 8;
        vst1q_f16(o0, vo0); o0 += 8;
      } else {
        float16x4_t vo1_lo = vget_low_f16(vo1);
        float16x4_t vo0_lo = vget_low_f16(vo0);

        if (w & (4 * sizeof(__fp16))) {
         vst1_f16(o1, vo1_lo); o1 += 4;
         vst1_f16(o0, vo0_lo); o0 += 4;

          vo1_lo = vget_high_f16(vo1);
          vo0_lo = vget_high_f16(vo0);
        }
        if (w & (2 * sizeof(__fp16))) {
          vst1_lane_u32((void*) o1, vreinterpret_u32_f16(vo1_lo), 0); o1 += 2;
          vst1_lane_u32((void*) o0, vreinterpret_u32_f16(vo0_lo), 0); o0 += 2;

          vo0_lo = vext_f16(vo0_lo, vo0_lo, 2);
          vo1_lo = vext_f16(vo1_lo, vo1_lo, 2);
        }
        if (w & (1 * sizeof(__fp16))) {
          vst1_lane_f16(o1, vo1_lo, 0); o1 += 1;
          vst1_lane_f16(o0, vo0_lo, 0); o0 += 1;
        }
      }
    }

    i0 = (const __fp16*) ((uintptr_t) i2 - input_decrement);
    i1 = (const __fp16*) ((uintptr_t) i3 - input_decrement);
    i2 = (const __fp16*) ((uintptr_t) i1 + input_width);
    i3 = (const __fp16*) ((uintptr_t) i2 + input_width);
    i4 = (const __fp16*) ((uintptr_t) i3 + input_width);
    i5 = (const __fp16*) ((uintptr_t) i4 + input_width);

    o0 = o1;
    o1 = (__fp16*) ((uintptr_t) o0 + input_width);

    output_height = doz(output_height, 2);
  } while (output_height != 0);
}
