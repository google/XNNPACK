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

#include "xnnpack/dwconv.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"


void xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_4x8_acc2(
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
  assert(input_width % sizeof(uint16_t) == 0);
  assert(padding_top == 2);

  #if XNN_ARCH_ARM64
    const uint16x8x2_t vminmax = vld2q_dup_u16(&params->neonfp16arith_stride1.min);
    const float16x8_t vmin = vreinterpretq_f16_u16(vminmax.val[0]);
    const float16x8_t vmax = vreinterpretq_f16_u16(vminmax.val[1]);
  #else
    // vld2_dup is to work around aarch32 clang bug with vld1q_dup
    const uint16x4x2_t vminmax = vld2_dup_u16(&params->neonfp16arith_stride1.min);
    const float16x8_t vmin = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[0], vminmax.val[0]));
    const float16x8_t vmax = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[1], vminmax.val[1]));
  #endif
  const uint16x8_t vmask = vld1q_u16(params->neonfp16arith_stride1.mask);

  const uint16_t* w = (const uint16_t*) weights;
  const float16x8_t vw01234567 = vreinterpretq_f16_u16(vld1q_u16(w));
  const float16x8_t vw89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w + 8));
  const float16x8_t vwGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(w + 16));
  const float16x4_t vwOP = vreinterpret_f16_u32(vld1_dup_u32((const void*) (w + 24)));

  const size_t input_decrement = round_up_po2(input_width, 8 * sizeof(uint16_t));

  const uint16_t* i0 = zero;
  const uint16_t* i1 = zero;
  const uint16_t* i2 = input;
  const uint16_t* i3 = (const uint16_t*) ((uintptr_t) i2 + input_width);
  const uint16_t* i4 = (const uint16_t*) ((uintptr_t) i3 + input_width);
  const uint16_t* i5 = (const uint16_t*) ((uintptr_t) i4 + input_width);
  const uint16_t* i6 = (const uint16_t*) ((uintptr_t) i5 + input_width);
  const uint16_t* i7 = (const uint16_t*) ((uintptr_t) i6 + input_width);

  uint16_t* o0 = output;
  uint16_t* o1 = (uint16_t*) ((uintptr_t) o0 + input_width);
  uint16_t* o2 = (uint16_t*) ((uintptr_t) o1 + input_width);
  uint16_t* o3 = (uint16_t*) ((uintptr_t) o2 + input_width);

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

    float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi5x01234567 = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi6x01234567 = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi7x01234567 = vreinterpretq_f16_u16(vmovq_n_u16(0));

    float16x8_t vi0x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
    float16x8_t vi1x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
    float16x8_t vi2x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
    float16x8_t vi3x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
    float16x8_t vi4x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
    float16x8_t vi5x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
    float16x8_t vi6x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
    float16x8_t vi7x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;

    size_t w = input_width;
    for (; w > 16 * sizeof(uint16_t); w -= 8 * sizeof(uint16_t)) {
      float16x8_t vo0p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);
      float16x8_t vo1p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);
      float16x8_t vo2p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);
      float16x8_t vo3p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);

      const float16x8_t vi0xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vi1xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      const float16x8_t vi2xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      const float16x8_t vi3xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
      const float16x8_t vi4xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
      const float16x8_t vi5xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
      const float16x8_t vi6xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
      const float16x8_t vi7xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;

      // Center column
      float16x8_t vo0p1 = vmulq_lane_f16(vi0x89ABCDEF, vget_low_f16(vw01234567), 3);
      float16x8_t vo1p1 = vmulq_lane_f16(vi1x89ABCDEF, vget_low_f16(vw01234567), 3);
      float16x8_t vo2p1 = vmulq_lane_f16(vi2x89ABCDEF, vget_low_f16(vw01234567), 3);
      float16x8_t vo3p1 = vmulq_lane_f16(vi3x89ABCDEF, vget_low_f16(vw01234567), 3);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi3x89ABCDEF, vw89ABCDEF, 0);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi3x89ABCDEF, vget_low_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi4x89ABCDEF, vw89ABCDEF, 0);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi4x89ABCDEF, vget_low_f16(vw89ABCDEF), 0);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi4x89ABCDEF, vw89ABCDEF, 5);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi4x89ABCDEF, vget_high_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi5x89ABCDEF, vw89ABCDEF, 5);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi5x89ABCDEF, vget_high_f16(vw89ABCDEF), 1);
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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi5x89ABCDEF, vwGHIJKLMN, 2);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi5x89ABCDEF, vget_low_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi6x89ABCDEF, vwGHIJKLMN, 2);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi6x89ABCDEF, vget_low_f16(vwGHIJKLMN), 2);
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
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi6x89ABCDEF, vwGHIJKLMN, 7);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi6x89ABCDEF, vget_high_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi7x89ABCDEF, vwGHIJKLMN, 7);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi7x89ABCDEF, vget_high_f16(vwGHIJKLMN), 3);
      #endif
      // Left by 1 column
      const float16x8_t vi0x789ABCDE = vextq_f16(vi0x01234567, vi0x89ABCDEF, 7);
      const float16x8_t vi1x789ABCDE = vextq_f16(vi1x01234567, vi1x89ABCDEF, 7);
      const float16x8_t vi2x789ABCDE = vextq_f16(vi2x01234567, vi2x89ABCDEF, 7);
      const float16x8_t vi3x789ABCDE = vextq_f16(vi3x01234567, vi3x89ABCDEF, 7);
      const float16x8_t vi4x789ABCDE = vextq_f16(vi4x01234567, vi4x89ABCDEF, 7);
      const float16x8_t vi5x789ABCDE = vextq_f16(vi5x01234567, vi5x89ABCDEF, 7);
      const float16x8_t vi6x789ABCDE = vextq_f16(vi6x01234567, vi6x89ABCDEF, 7);
      const float16x8_t vi7x789ABCDE = vextq_f16(vi7x01234567, vi7x89ABCDEF, 7);

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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi2x789ABCDE, vw01234567, 2);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi2x789ABCDE, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi3x789ABCDE, vw01234567, 2);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi3x789ABCDE, vget_low_f16(vw01234567), 2);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi3x789ABCDE, vw01234567, 7);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi3x789ABCDE, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi4x789ABCDE, vw01234567, 7);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi4x789ABCDE, vget_high_f16(vw01234567), 3);
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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi4x789ABCDE, vw89ABCDEF, 4);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi4x789ABCDE, vget_high_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi5x789ABCDE, vw89ABCDEF, 4);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi5x789ABCDE, vget_high_f16(vw89ABCDEF), 0);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi5x789ABCDE, vwGHIJKLMN, 1);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi5x789ABCDE, vget_low_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi6x789ABCDE, vwGHIJKLMN, 1);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi6x789ABCDE, vget_low_f16(vwGHIJKLMN), 1);
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
      #if XNN_ARCH_ARM64
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi6x789ABCDE, vwGHIJKLMN, 6);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi6x789ABCDE, vget_high_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi7x789ABCDE, vwGHIJKLMN, 6);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi7x789ABCDE, vget_high_f16(vwGHIJKLMN), 2);
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
      const float16x8_t vi6x6789ABCD = vextq_f16(vi6x01234567, vi6x89ABCDEF, 6);
      vi6x01234567 = vi6x89ABCDEF;
      const float16x8_t vi7x6789ABCD = vextq_f16(vi7x01234567, vi7x89ABCDEF, 6);
      vi7x01234567 = vi7x89ABCDEF;

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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi2x6789ABCD, vw01234567, 1);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi2x6789ABCD, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi3x6789ABCD, vw01234567, 1);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi3x6789ABCD, vget_low_f16(vw01234567), 1);
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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi3x6789ABCD, vw01234567, 6);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi3x6789ABCD, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi4x6789ABCD, vw01234567, 6);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi4x6789ABCD, vget_high_f16(vw01234567), 2);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi4x6789ABCD, vw89ABCDEF, 3);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi4x6789ABCD, vget_low_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi5x6789ABCD, vw89ABCDEF, 3);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi5x6789ABCD, vget_low_f16(vw89ABCDEF), 3);
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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi5x6789ABCD, vwGHIJKLMN, 0);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi5x6789ABCD, vget_low_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi6x6789ABCD, vwGHIJKLMN, 0);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi6x6789ABCD, vget_low_f16(vwGHIJKLMN), 0);
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
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi6x6789ABCD, vwGHIJKLMN, 5);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi6x6789ABCD, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi7x6789ABCD, vwGHIJKLMN, 5);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi7x6789ABCD, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      // Right by 1 column
      const float16x8_t vi0x9ABCDEFG = vextq_f16(vi0x89ABCDEF, vi0xGHIJKLMN, 1);
      const float16x8_t vi1x9ABCDEFG = vextq_f16(vi1x89ABCDEF, vi1xGHIJKLMN, 1);
      const float16x8_t vi2x9ABCDEFG = vextq_f16(vi2x89ABCDEF, vi2xGHIJKLMN, 1);
      const float16x8_t vi3x9ABCDEFG = vextq_f16(vi3x89ABCDEF, vi3xGHIJKLMN, 1);
      const float16x8_t vi4x9ABCDEFG = vextq_f16(vi4x89ABCDEF, vi4xGHIJKLMN, 1);
      const float16x8_t vi5x9ABCDEFG = vextq_f16(vi5x89ABCDEF, vi5xGHIJKLMN, 1);
      const float16x8_t vi6x9ABCDEFG = vextq_f16(vi6x89ABCDEF, vi6xGHIJKLMN, 1);
      const float16x8_t vi7x9ABCDEFG = vextq_f16(vi7x89ABCDEF, vi7xGHIJKLMN, 1);

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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi2x9ABCDEFG, vw01234567, 4);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi2x9ABCDEFG, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi3x9ABCDEFG, vw01234567, 4);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi3x9ABCDEFG, vget_high_f16(vw01234567), 0);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi3x9ABCDEFG, vw89ABCDEF, 1);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi3x9ABCDEFG, vget_low_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi4x9ABCDEFG, vw89ABCDEF, 1);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi4x9ABCDEFG, vget_low_f16(vw89ABCDEF), 1);
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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi4x9ABCDEFG, vw89ABCDEF, 6);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi4x9ABCDEFG, vget_high_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi5x9ABCDEFG, vw89ABCDEF, 6);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi5x9ABCDEFG, vget_high_f16(vw89ABCDEF), 2);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi5x9ABCDEFG, vwGHIJKLMN, 3);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi5x9ABCDEFG, vget_low_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi6x9ABCDEFG, vwGHIJKLMN, 3);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi6x9ABCDEFG, vget_low_f16(vwGHIJKLMN), 3);
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
      #if XNN_ARCH_ARM64
        vo2p1 = vfmaq_lane_f16(vo2p1, vi6x9ABCDEFG, vwOP, 0);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi6x9ABCDEFG, vwOP, 0);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_lane_f16(vo3p1, vi7x9ABCDEFG, vwOP, 0);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi7x9ABCDEFG, vwOP, 0);
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
      const float16x8_t vi6xABCDEFGH = vextq_f16(vi6x89ABCDEF, vi6xGHIJKLMN, 2);
      vi6x89ABCDEF = vi6xGHIJKLMN;
      const float16x8_t vi7xABCDEFGH = vextq_f16(vi7x89ABCDEF, vi7xGHIJKLMN, 2);
      vi7x89ABCDEF = vi7xGHIJKLMN;

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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi2xABCDEFGH, vw01234567, 5);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi2xABCDEFGH, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi3xABCDEFGH, vw01234567, 5);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi3xABCDEFGH, vget_high_f16(vw01234567), 1);
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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi3xABCDEFGH, vw89ABCDEF, 2);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi3xABCDEFGH, vget_low_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi4xABCDEFGH, vw89ABCDEF, 2);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi4xABCDEFGH, vget_low_f16(vw89ABCDEF), 2);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi4xABCDEFGH, vw89ABCDEF, 7);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi4xABCDEFGH, vget_high_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi5xABCDEFGH, vw89ABCDEF, 7);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi5xABCDEFGH, vget_high_f16(vw89ABCDEF), 3);
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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi5xABCDEFGH, vwGHIJKLMN, 4);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi5xABCDEFGH, vget_high_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi6xABCDEFGH, vwGHIJKLMN, 4);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi6xABCDEFGH, vget_high_f16(vwGHIJKLMN), 0);
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
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_lane_f16(vo2p0, vi6xABCDEFGH, vwOP, 1);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi6xABCDEFGH, vwOP, 1);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_lane_f16(vo3p0, vi7xABCDEFGH, vwOP, 1);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi7xABCDEFGH, vwOP, 1);
      #endif
      vo0p0 = vaddq_f16(vo0p0, vo0p1);
      vo1p0 = vaddq_f16(vo1p0, vo1p1);
      vo2p0 = vaddq_f16(vo2p0, vo2p1);
      vo3p0 = vaddq_f16(vo3p0, vo3p1);

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);
      float16x8_t vo1 = vmaxq_f16(vo1p0, vmin);
      float16x8_t vo2 = vmaxq_f16(vo2p0, vmin);
      float16x8_t vo3 = vmaxq_f16(vo3p0, vmin);

      vo0 = vminq_f16(vo0, vmax);
      vo1 = vminq_f16(vo1, vmax);
      vo2 = vminq_f16(vo2, vmax);
      vo3 = vminq_f16(vo3, vmax);

      vst1q_u16(o3, vreinterpretq_u16_f16(vo3)); o3 += 8;
      vst1q_u16(o2, vreinterpretq_u16_f16(vo2)); o2 += 8;
      vst1q_u16(o1, vreinterpretq_u16_f16(vo1)); o1 += 8;
      vst1q_u16(o0, vreinterpretq_u16_f16(vo0)); o0 += 8;
    }

    // Always process the last block of 5..16 pixels.
    assert(w <= 16 * sizeof(uint16_t));
    if XNN_LIKELY(w > 8 * sizeof(uint16_t)) {
      float16x8_t vo0p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);
      float16x8_t vo1p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);
      float16x8_t vo2p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);
      float16x8_t vo3p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);

      float16x8_t vi0xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      float16x8_t vi1xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      float16x8_t vi2xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      float16x8_t vi3xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
      float16x8_t vi4xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
      float16x8_t vi5xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
      float16x8_t vi6xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
      float16x8_t vi7xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;

      vi0xGHIJKLMN = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi0xGHIJKLMN)));
      vi1xGHIJKLMN = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi1xGHIJKLMN)));
      vi2xGHIJKLMN = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi2xGHIJKLMN)));
      vi3xGHIJKLMN = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi3xGHIJKLMN)));
      vi4xGHIJKLMN = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi4xGHIJKLMN)));
      vi5xGHIJKLMN = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi5xGHIJKLMN)));
      vi6xGHIJKLMN = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi6xGHIJKLMN)));
      vi7xGHIJKLMN = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi7xGHIJKLMN)));

      // Center column
      float16x8_t vo0p1 = vmulq_lane_f16(vi0x89ABCDEF, vget_low_f16(vw01234567), 3);
      float16x8_t vo1p1 = vmulq_lane_f16(vi1x89ABCDEF, vget_low_f16(vw01234567), 3);
      float16x8_t vo2p1 = vmulq_lane_f16(vi2x89ABCDEF, vget_low_f16(vw01234567), 3);
      float16x8_t vo3p1 = vmulq_lane_f16(vi3x89ABCDEF, vget_low_f16(vw01234567), 3);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi3x89ABCDEF, vw89ABCDEF, 0);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi3x89ABCDEF, vget_low_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi4x89ABCDEF, vw89ABCDEF, 0);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi4x89ABCDEF, vget_low_f16(vw89ABCDEF), 0);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi4x89ABCDEF, vw89ABCDEF, 5);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi4x89ABCDEF, vget_high_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi5x89ABCDEF, vw89ABCDEF, 5);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi5x89ABCDEF, vget_high_f16(vw89ABCDEF), 1);
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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi5x89ABCDEF, vwGHIJKLMN, 2);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi5x89ABCDEF, vget_low_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi6x89ABCDEF, vwGHIJKLMN, 2);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi6x89ABCDEF, vget_low_f16(vwGHIJKLMN), 2);
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
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi6x89ABCDEF, vwGHIJKLMN, 7);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi6x89ABCDEF, vget_high_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi7x89ABCDEF, vwGHIJKLMN, 7);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi7x89ABCDEF, vget_high_f16(vwGHIJKLMN), 3);
      #endif
      // Left by 1 column
      const float16x8_t vi0x789ABCDE = vextq_f16(vi0x01234567, vi0x89ABCDEF, 7);
      const float16x8_t vi1x789ABCDE = vextq_f16(vi1x01234567, vi1x89ABCDEF, 7);
      const float16x8_t vi2x789ABCDE = vextq_f16(vi2x01234567, vi2x89ABCDEF, 7);
      const float16x8_t vi3x789ABCDE = vextq_f16(vi3x01234567, vi3x89ABCDEF, 7);
      const float16x8_t vi4x789ABCDE = vextq_f16(vi4x01234567, vi4x89ABCDEF, 7);
      const float16x8_t vi5x789ABCDE = vextq_f16(vi5x01234567, vi5x89ABCDEF, 7);
      const float16x8_t vi6x789ABCDE = vextq_f16(vi6x01234567, vi6x89ABCDEF, 7);
      const float16x8_t vi7x789ABCDE = vextq_f16(vi7x01234567, vi7x89ABCDEF, 7);

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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi2x789ABCDE, vw01234567, 2);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi2x789ABCDE, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi3x789ABCDE, vw01234567, 2);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi3x789ABCDE, vget_low_f16(vw01234567), 2);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi3x789ABCDE, vw01234567, 7);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi3x789ABCDE, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi4x789ABCDE, vw01234567, 7);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi4x789ABCDE, vget_high_f16(vw01234567), 3);
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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi4x789ABCDE, vw89ABCDEF, 4);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi4x789ABCDE, vget_high_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi5x789ABCDE, vw89ABCDEF, 4);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi5x789ABCDE, vget_high_f16(vw89ABCDEF), 0);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi5x789ABCDE, vwGHIJKLMN, 1);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi5x789ABCDE, vget_low_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi6x789ABCDE, vwGHIJKLMN, 1);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi6x789ABCDE, vget_low_f16(vwGHIJKLMN), 1);
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
      #if XNN_ARCH_ARM64
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi6x789ABCDE, vwGHIJKLMN, 6);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi6x789ABCDE, vget_high_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi7x789ABCDE, vwGHIJKLMN, 6);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi7x789ABCDE, vget_high_f16(vwGHIJKLMN), 2);
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
      const float16x8_t vi6x6789ABCD = vextq_f16(vi6x01234567, vi6x89ABCDEF, 6);
      vi6x01234567 = vi6x89ABCDEF;
      const float16x8_t vi7x6789ABCD = vextq_f16(vi7x01234567, vi7x89ABCDEF, 6);
      vi7x01234567 = vi7x89ABCDEF;

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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi2x6789ABCD, vw01234567, 1);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi2x6789ABCD, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi3x6789ABCD, vw01234567, 1);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi3x6789ABCD, vget_low_f16(vw01234567), 1);
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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi3x6789ABCD, vw01234567, 6);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi3x6789ABCD, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi4x6789ABCD, vw01234567, 6);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi4x6789ABCD, vget_high_f16(vw01234567), 2);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi4x6789ABCD, vw89ABCDEF, 3);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi4x6789ABCD, vget_low_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi5x6789ABCD, vw89ABCDEF, 3);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi5x6789ABCD, vget_low_f16(vw89ABCDEF), 3);
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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi5x6789ABCD, vwGHIJKLMN, 0);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi5x6789ABCD, vget_low_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi6x6789ABCD, vwGHIJKLMN, 0);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi6x6789ABCD, vget_low_f16(vwGHIJKLMN), 0);
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
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi6x6789ABCD, vwGHIJKLMN, 5);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi6x6789ABCD, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi7x6789ABCD, vwGHIJKLMN, 5);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi7x6789ABCD, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      // Right by 1 column
      const float16x8_t vi0x9ABCDEFG = vextq_f16(vi0x89ABCDEF, vi0xGHIJKLMN, 1);
      const float16x8_t vi1x9ABCDEFG = vextq_f16(vi1x89ABCDEF, vi1xGHIJKLMN, 1);
      const float16x8_t vi2x9ABCDEFG = vextq_f16(vi2x89ABCDEF, vi2xGHIJKLMN, 1);
      const float16x8_t vi3x9ABCDEFG = vextq_f16(vi3x89ABCDEF, vi3xGHIJKLMN, 1);
      const float16x8_t vi4x9ABCDEFG = vextq_f16(vi4x89ABCDEF, vi4xGHIJKLMN, 1);
      const float16x8_t vi5x9ABCDEFG = vextq_f16(vi5x89ABCDEF, vi5xGHIJKLMN, 1);
      const float16x8_t vi6x9ABCDEFG = vextq_f16(vi6x89ABCDEF, vi6xGHIJKLMN, 1);
      const float16x8_t vi7x9ABCDEFG = vextq_f16(vi7x89ABCDEF, vi7xGHIJKLMN, 1);

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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi2x9ABCDEFG, vw01234567, 4);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi2x9ABCDEFG, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi3x9ABCDEFG, vw01234567, 4);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi3x9ABCDEFG, vget_high_f16(vw01234567), 0);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi3x9ABCDEFG, vw89ABCDEF, 1);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi3x9ABCDEFG, vget_low_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi4x9ABCDEFG, vw89ABCDEF, 1);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi4x9ABCDEFG, vget_low_f16(vw89ABCDEF), 1);
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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi4x9ABCDEFG, vw89ABCDEF, 6);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi4x9ABCDEFG, vget_high_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi5x9ABCDEFG, vw89ABCDEF, 6);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi5x9ABCDEFG, vget_high_f16(vw89ABCDEF), 2);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi5x9ABCDEFG, vwGHIJKLMN, 3);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi5x9ABCDEFG, vget_low_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi6x9ABCDEFG, vwGHIJKLMN, 3);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi6x9ABCDEFG, vget_low_f16(vwGHIJKLMN), 3);
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
      #if XNN_ARCH_ARM64
        vo2p1 = vfmaq_lane_f16(vo2p1, vi6x9ABCDEFG, vwOP, 0);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi6x9ABCDEFG, vwOP, 0);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_lane_f16(vo3p1, vi7x9ABCDEFG, vwOP, 0);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi7x9ABCDEFG, vwOP, 0);
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
      const float16x8_t vi6xABCDEFGH = vextq_f16(vi6x89ABCDEF, vi6xGHIJKLMN, 2);
      vi6x89ABCDEF = vi6xGHIJKLMN;
      const float16x8_t vi7xABCDEFGH = vextq_f16(vi7x89ABCDEF, vi7xGHIJKLMN, 2);
      vi7x89ABCDEF = vi7xGHIJKLMN;

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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi2xABCDEFGH, vw01234567, 5);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi2xABCDEFGH, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi3xABCDEFGH, vw01234567, 5);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi3xABCDEFGH, vget_high_f16(vw01234567), 1);
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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi3xABCDEFGH, vw89ABCDEF, 2);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi3xABCDEFGH, vget_low_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi4xABCDEFGH, vw89ABCDEF, 2);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi4xABCDEFGH, vget_low_f16(vw89ABCDEF), 2);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi4xABCDEFGH, vw89ABCDEF, 7);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi4xABCDEFGH, vget_high_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi5xABCDEFGH, vw89ABCDEF, 7);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi5xABCDEFGH, vget_high_f16(vw89ABCDEF), 3);
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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi5xABCDEFGH, vwGHIJKLMN, 4);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi5xABCDEFGH, vget_high_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi6xABCDEFGH, vwGHIJKLMN, 4);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi6xABCDEFGH, vget_high_f16(vwGHIJKLMN), 0);
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
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_lane_f16(vo2p0, vi6xABCDEFGH, vwOP, 1);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi6xABCDEFGH, vwOP, 1);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_lane_f16(vo3p0, vi7xABCDEFGH, vwOP, 1);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi7xABCDEFGH, vwOP, 1);
      #endif
      vo0p0 = vaddq_f16(vo0p0, vo0p1);
      vo1p0 = vaddq_f16(vo1p0, vo1p1);
      vo2p0 = vaddq_f16(vo2p0, vo2p1);
      vo3p0 = vaddq_f16(vo3p0, vo3p1);

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);
      float16x8_t vo1 = vmaxq_f16(vo1p0, vmin);
      float16x8_t vo2 = vmaxq_f16(vo2p0, vmin);
      float16x8_t vo3 = vmaxq_f16(vo3p0, vmin);

      vo0 = vminq_f16(vo0, vmax);
      vo1 = vminq_f16(vo1, vmax);
      vo2 = vminq_f16(vo2, vmax);
      vo3 = vminq_f16(vo3, vmax);

      vst1q_u16(o3, vreinterpretq_u16_f16(vo3)); o3 += 8;
      vst1q_u16(o2, vreinterpretq_u16_f16(vo2)); o2 += 8;
      vst1q_u16(o1, vreinterpretq_u16_f16(vo1)); o1 += 8;
      vst1q_u16(o0, vreinterpretq_u16_f16(vo0)); o0 += 8;

      w -= 8 * sizeof(uint16_t);
    }

    assert(w >= 1 * sizeof(uint16_t));
    assert(w <= 8 * sizeof(uint16_t));
    {
      float16x8_t vo0p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);
      float16x8_t vo1p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);
      float16x8_t vo2p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);
      float16x8_t vo3p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);

      vi0x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi0x89ABCDEF)));
      vi1x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi1x89ABCDEF)));
      vi2x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi2x89ABCDEF)));
      vi3x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi3x89ABCDEF)));
      vi4x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi4x89ABCDEF)));
      vi5x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi5x89ABCDEF)));
      vi6x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi6x89ABCDEF)));
      vi7x89ABCDEF = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi7x89ABCDEF)));

      // Center column
      float16x8_t vo0p1 = vmulq_lane_f16(vi0x89ABCDEF, vget_low_f16(vw01234567), 3);
      float16x8_t vo1p1 = vmulq_lane_f16(vi1x89ABCDEF, vget_low_f16(vw01234567), 3);
      float16x8_t vo2p1 = vmulq_lane_f16(vi2x89ABCDEF, vget_low_f16(vw01234567), 3);
      float16x8_t vo3p1 = vmulq_lane_f16(vi3x89ABCDEF, vget_low_f16(vw01234567), 3);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi3x89ABCDEF, vw89ABCDEF, 0);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi3x89ABCDEF, vget_low_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi4x89ABCDEF, vw89ABCDEF, 0);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi4x89ABCDEF, vget_low_f16(vw89ABCDEF), 0);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi4x89ABCDEF, vw89ABCDEF, 5);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi4x89ABCDEF, vget_high_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi5x89ABCDEF, vw89ABCDEF, 5);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi5x89ABCDEF, vget_high_f16(vw89ABCDEF), 1);
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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi5x89ABCDEF, vwGHIJKLMN, 2);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi5x89ABCDEF, vget_low_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi6x89ABCDEF, vwGHIJKLMN, 2);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi6x89ABCDEF, vget_low_f16(vwGHIJKLMN), 2);
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
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi6x89ABCDEF, vwGHIJKLMN, 7);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi6x89ABCDEF, vget_high_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi7x89ABCDEF, vwGHIJKLMN, 7);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi7x89ABCDEF, vget_high_f16(vwGHIJKLMN), 3);
      #endif
      // Left by 1 column
      const float16x8_t vi0x789ABCDE = vextq_f16(vi0x01234567, vi0x89ABCDEF, 7);
      const float16x8_t vi1x789ABCDE = vextq_f16(vi1x01234567, vi1x89ABCDEF, 7);
      const float16x8_t vi2x789ABCDE = vextq_f16(vi2x01234567, vi2x89ABCDEF, 7);
      const float16x8_t vi3x789ABCDE = vextq_f16(vi3x01234567, vi3x89ABCDEF, 7);
      const float16x8_t vi4x789ABCDE = vextq_f16(vi4x01234567, vi4x89ABCDEF, 7);
      const float16x8_t vi5x789ABCDE = vextq_f16(vi5x01234567, vi5x89ABCDEF, 7);
      const float16x8_t vi6x789ABCDE = vextq_f16(vi6x01234567, vi6x89ABCDEF, 7);
      const float16x8_t vi7x789ABCDE = vextq_f16(vi7x01234567, vi7x89ABCDEF, 7);

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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi2x789ABCDE, vw01234567, 2);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi2x789ABCDE, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi3x789ABCDE, vw01234567, 2);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi3x789ABCDE, vget_low_f16(vw01234567), 2);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi3x789ABCDE, vw01234567, 7);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi3x789ABCDE, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi4x789ABCDE, vw01234567, 7);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi4x789ABCDE, vget_high_f16(vw01234567), 3);
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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi4x789ABCDE, vw89ABCDEF, 4);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi4x789ABCDE, vget_high_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi5x789ABCDE, vw89ABCDEF, 4);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi5x789ABCDE, vget_high_f16(vw89ABCDEF), 0);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi5x789ABCDE, vwGHIJKLMN, 1);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi5x789ABCDE, vget_low_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi6x789ABCDE, vwGHIJKLMN, 1);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi6x789ABCDE, vget_low_f16(vwGHIJKLMN), 1);
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
      #if XNN_ARCH_ARM64
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi6x789ABCDE, vwGHIJKLMN, 6);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi6x789ABCDE, vget_high_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi7x789ABCDE, vwGHIJKLMN, 6);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi7x789ABCDE, vget_high_f16(vwGHIJKLMN), 2);
      #endif
      // Left by 2 column
      const float16x8_t vi0x6789ABCD = vextq_f16(vi0x01234567, vi0x89ABCDEF, 6);
      const float16x8_t vi1x6789ABCD = vextq_f16(vi1x01234567, vi1x89ABCDEF, 6);
      const float16x8_t vi2x6789ABCD = vextq_f16(vi2x01234567, vi2x89ABCDEF, 6);
      const float16x8_t vi3x6789ABCD = vextq_f16(vi3x01234567, vi3x89ABCDEF, 6);
      const float16x8_t vi4x6789ABCD = vextq_f16(vi4x01234567, vi4x89ABCDEF, 6);
      const float16x8_t vi5x6789ABCD = vextq_f16(vi5x01234567, vi5x89ABCDEF, 6);
      const float16x8_t vi6x6789ABCD = vextq_f16(vi6x01234567, vi6x89ABCDEF, 6);
      const float16x8_t vi7x6789ABCD = vextq_f16(vi7x01234567, vi7x89ABCDEF, 6);

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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi2x6789ABCD, vw01234567, 1);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi2x6789ABCD, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi3x6789ABCD, vw01234567, 1);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi3x6789ABCD, vget_low_f16(vw01234567), 1);
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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi3x6789ABCD, vw01234567, 6);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi3x6789ABCD, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi4x6789ABCD, vw01234567, 6);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi4x6789ABCD, vget_high_f16(vw01234567), 2);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi4x6789ABCD, vw89ABCDEF, 3);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi4x6789ABCD, vget_low_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi5x6789ABCD, vw89ABCDEF, 3);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi5x6789ABCD, vget_low_f16(vw89ABCDEF), 3);
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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi5x6789ABCD, vwGHIJKLMN, 0);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi5x6789ABCD, vget_low_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi6x6789ABCD, vwGHIJKLMN, 0);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi6x6789ABCD, vget_low_f16(vwGHIJKLMN), 0);
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
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi6x6789ABCD, vwGHIJKLMN, 5);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi6x6789ABCD, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi7x6789ABCD, vwGHIJKLMN, 5);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi7x6789ABCD, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      // Right by 1 column
      const float16x8_t vzero = vreinterpretq_f16_u16(vmovq_n_u16(0));
      const float16x8_t vi0x9ABCDEFG = vextq_f16(vi0x89ABCDEF, vzero, 1);
      const float16x8_t vi1x9ABCDEFG = vextq_f16(vi1x89ABCDEF, vzero, 1);
      const float16x8_t vi2x9ABCDEFG = vextq_f16(vi2x89ABCDEF, vzero, 1);
      const float16x8_t vi3x9ABCDEFG = vextq_f16(vi3x89ABCDEF, vzero, 1);
      const float16x8_t vi4x9ABCDEFG = vextq_f16(vi4x89ABCDEF, vzero, 1);
      const float16x8_t vi5x9ABCDEFG = vextq_f16(vi5x89ABCDEF, vzero, 1);
      const float16x8_t vi6x9ABCDEFG = vextq_f16(vi6x89ABCDEF, vzero, 1);
      const float16x8_t vi7x9ABCDEFG = vextq_f16(vi7x89ABCDEF, vzero, 1);

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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi2x9ABCDEFG, vw01234567, 4);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi2x9ABCDEFG, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi3x9ABCDEFG, vw01234567, 4);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi3x9ABCDEFG, vget_high_f16(vw01234567), 0);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi3x9ABCDEFG, vw89ABCDEF, 1);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi3x9ABCDEFG, vget_low_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi4x9ABCDEFG, vw89ABCDEF, 1);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi4x9ABCDEFG, vget_low_f16(vw89ABCDEF), 1);
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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi4x9ABCDEFG, vw89ABCDEF, 6);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi4x9ABCDEFG, vget_high_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi5x9ABCDEFG, vw89ABCDEF, 6);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi5x9ABCDEFG, vget_high_f16(vw89ABCDEF), 2);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi5x9ABCDEFG, vwGHIJKLMN, 3);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi5x9ABCDEFG, vget_low_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi6x9ABCDEFG, vwGHIJKLMN, 3);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi6x9ABCDEFG, vget_low_f16(vwGHIJKLMN), 3);
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
      #if XNN_ARCH_ARM64
        vo2p1 = vfmaq_lane_f16(vo2p1, vi6x9ABCDEFG, vwOP, 0);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi6x9ABCDEFG, vwOP, 0);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_lane_f16(vo3p1, vi7x9ABCDEFG, vwOP, 0);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi7x9ABCDEFG, vwOP, 0);
      #endif
      // Right by 2 column
      const float16x8_t vi0xABCDEFGH = vextq_f16(vi0x9ABCDEFG, vzero, 1);
      const float16x8_t vi1xABCDEFGH = vextq_f16(vi1x9ABCDEFG, vzero, 1);
      const float16x8_t vi2xABCDEFGH = vextq_f16(vi2x9ABCDEFG, vzero, 1);
      const float16x8_t vi3xABCDEFGH = vextq_f16(vi3x9ABCDEFG, vzero, 1);
      const float16x8_t vi4xABCDEFGH = vextq_f16(vi4x9ABCDEFG, vzero, 1);
      const float16x8_t vi5xABCDEFGH = vextq_f16(vi5x9ABCDEFG, vzero, 1);
      const float16x8_t vi6xABCDEFGH = vextq_f16(vi6x9ABCDEFG, vzero, 1);
      const float16x8_t vi7xABCDEFGH = vextq_f16(vi7x9ABCDEFG, vzero, 1);

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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi2xABCDEFGH, vw01234567, 5);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi2xABCDEFGH, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi3xABCDEFGH, vw01234567, 5);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi3xABCDEFGH, vget_high_f16(vw01234567), 1);
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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi3xABCDEFGH, vw89ABCDEF, 2);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi3xABCDEFGH, vget_low_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi4xABCDEFGH, vw89ABCDEF, 2);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi4xABCDEFGH, vget_low_f16(vw89ABCDEF), 2);
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
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi4xABCDEFGH, vw89ABCDEF, 7);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi4xABCDEFGH, vget_high_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_laneq_f16(vo3p0, vi5xABCDEFGH, vw89ABCDEF, 7);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi5xABCDEFGH, vget_high_f16(vw89ABCDEF), 3);
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
        vo2p1 = vfmaq_laneq_f16(vo2p1, vi5xABCDEFGH, vwGHIJKLMN, 4);
      #else
        vo2p1 = vmlaq_lane_f16(vo2p1, vi5xABCDEFGH, vget_high_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo3p1 = vfmaq_laneq_f16(vo3p1, vi6xABCDEFGH, vwGHIJKLMN, 4);
      #else
        vo3p1 = vmlaq_lane_f16(vo3p1, vi6xABCDEFGH, vget_high_f16(vwGHIJKLMN), 0);
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
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_lane_f16(vo2p0, vi6xABCDEFGH, vwOP, 1);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi6xABCDEFGH, vwOP, 1);
      #endif
      #if XNN_ARCH_ARM64
        vo3p0 = vfmaq_lane_f16(vo3p0, vi7xABCDEFGH, vwOP, 1);
      #else
        vo3p0 = vmlaq_lane_f16(vo3p0, vi7xABCDEFGH, vwOP, 1);
      #endif
      vo0p0 = vaddq_f16(vo0p0, vo0p1);
      vo1p0 = vaddq_f16(vo1p0, vo1p1);
      vo2p0 = vaddq_f16(vo2p0, vo2p1);
      vo3p0 = vaddq_f16(vo3p0, vo3p1);

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);
      float16x8_t vo1 = vmaxq_f16(vo1p0, vmin);
      float16x8_t vo2 = vmaxq_f16(vo2p0, vmin);
      float16x8_t vo3 = vmaxq_f16(vo3p0, vmin);

      vo0 = vminq_f16(vo0, vmax);
      vo1 = vminq_f16(vo1, vmax);
      vo2 = vminq_f16(vo2, vmax);
      vo3 = vminq_f16(vo3, vmax);

      if XNN_LIKELY(w == 8 * sizeof(uint16_t)) {
        vst1q_u16(o3, vreinterpretq_u16_f16(vo3)); o3 += 8;
        vst1q_u16(o2, vreinterpretq_u16_f16(vo2)); o2 += 8;
        vst1q_u16(o1, vreinterpretq_u16_f16(vo1)); o1 += 8;
        vst1q_u16(o0, vreinterpretq_u16_f16(vo0)); o0 += 8;
      } else {
        float16x4_t vo3_lo = vget_low_f16(vo3);
        float16x4_t vo2_lo = vget_low_f16(vo2);
        float16x4_t vo1_lo = vget_low_f16(vo1);
        float16x4_t vo0_lo = vget_low_f16(vo0);

        if (w & (4 * sizeof(uint16_t))) {
         vst1_u16(o3, vreinterpret_u16_f16(vo3_lo)); o3 += 4;
         vst1_u16(o2, vreinterpret_u16_f16(vo2_lo)); o2 += 4;
         vst1_u16(o1, vreinterpret_u16_f16(vo1_lo)); o1 += 4;
         vst1_u16(o0, vreinterpret_u16_f16(vo0_lo)); o0 += 4;

          vo3_lo = vget_high_f16(vo3);
          vo2_lo = vget_high_f16(vo2);
          vo1_lo = vget_high_f16(vo1);
          vo0_lo = vget_high_f16(vo0);
        }
        if (w & (2 * sizeof(uint16_t))) {
          vst1_lane_u32((void*) o3, vreinterpret_u32_f16(vo3_lo), 0); o3 += 2;
          vst1_lane_u32((void*) o2, vreinterpret_u32_f16(vo2_lo), 0); o2 += 2;
          vst1_lane_u32((void*) o1, vreinterpret_u32_f16(vo1_lo), 0); o1 += 2;
          vst1_lane_u32((void*) o0, vreinterpret_u32_f16(vo0_lo), 0); o0 += 2;

          vo0_lo = vext_f16(vo0_lo, vo0_lo, 2);
          vo1_lo = vext_f16(vo1_lo, vo1_lo, 2);
          vo2_lo = vext_f16(vo2_lo, vo2_lo, 2);
          vo3_lo = vext_f16(vo3_lo, vo3_lo, 2);
        }
        if (w & (1 * sizeof(uint16_t))) {
          vst1_lane_u16(o3, vreinterpret_u16_f16(vo3_lo), 0); o3 += 1;
          vst1_lane_u16(o2, vreinterpret_u16_f16(vo2_lo), 0); o2 += 1;
          vst1_lane_u16(o1, vreinterpret_u16_f16(vo1_lo), 0); o1 += 1;
          vst1_lane_u16(o0, vreinterpret_u16_f16(vo0_lo), 0); o0 += 1;
        }
      }
    }

    i0 = (const uint16_t*) ((uintptr_t) i4 - input_decrement);
    i1 = (const uint16_t*) ((uintptr_t) i5 - input_decrement);
    i2 = (const uint16_t*) ((uintptr_t) i1 + input_width);
    i3 = (const uint16_t*) ((uintptr_t) i2 + input_width);
    i4 = (const uint16_t*) ((uintptr_t) i3 + input_width);
    i5 = (const uint16_t*) ((uintptr_t) i4 + input_width);
    i6 = (const uint16_t*) ((uintptr_t) i5 + input_width);
    i7 = (const uint16_t*) ((uintptr_t) i6 + input_width);

    o0 = o3;
    o1 = (uint16_t*) ((uintptr_t) o0 + input_width);
    o2 = (uint16_t*) ((uintptr_t) o1 + input_width);
    o3 = (uint16_t*) ((uintptr_t) o2 + input_width);

    output_height = doz(output_height, 4);
  } while (output_height != 0);
}
