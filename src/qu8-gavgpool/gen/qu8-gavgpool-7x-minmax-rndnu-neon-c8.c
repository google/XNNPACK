// Auto-generated file. Do not edit!
//   Template: src/qs8-gavgpool/unipass-neon.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/gavgpool.h"


void xnn_qu8_gavgpool_minmax_rndnu_ukernel_7x__neon_c8(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint8_t* output,
    const union xnn_qu8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const uint8_t* i0 = input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  const uint8_t* i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  const uint8_t* i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  const uint8_t* i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const int32x4_t vinit_bias = vld1q_dup_s32(&params->rndnu_neon.init_bias);
  const int32x4_t vleft_pre_shift = vld1q_dup_s32(&params->rndnu_neon.left_pre_shift);
  const int32x4_t vmultiplier = vld1q_dup_s32(&params->rndnu_neon.multiplier);
  const int32x4_t vleft_post_shift = vld1q_dup_s32(&params->rndnu_neon.left_post_shift);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->rndnu_neon.output_zero_point);
  const uint8x8_t voutput_min = vld1_dup_u8(&params->rndnu_neon.output_min);
  const uint8x8_t voutput_max = vld1_dup_u8(&params->rndnu_neon.output_max);
  for (; channels >= 8; channels -= 8) {
    const uint8x8_t vi0x01234567 = vld1_u8(i0); i0 += 8;
    const uint8x8_t vi1x01234567 = vld1_u8(i1); i1 += 8;

    const uint8x8_t vi2x01234567 = vld1_u8(i2); i2 += 8;
    uint16x8_t vsum01234567 = vaddl_u8(vi0x01234567, vi1x01234567);

    const uint8x8_t vi3x01234567 = vld1_u8(i3); i3 += 8;
    vsum01234567 = vaddw_u8(vsum01234567, vi2x01234567);
    const uint8x8_t vi4x01234567 = vld1_u8(i4); i4 += 8;
    vsum01234567 = vaddw_u8(vsum01234567, vi3x01234567);
    const uint8x8_t vi5x01234567 = vld1_u8(i5); i5 += 8;
    vsum01234567 = vaddw_u8(vsum01234567, vi4x01234567);
    const uint8x8_t vi6x01234567 = vld1_u8(i6); i6 += 8;
    vsum01234567 = vaddw_u8(vsum01234567, vi5x01234567);
    vsum01234567 = vaddw_u8(vsum01234567, vi6x01234567);

    int32x4_t vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vinit_bias), vget_low_u16(vsum01234567)));
    int32x4_t vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vinit_bias), vget_high_u16(vsum01234567)));

    vacc0123 = vqshlq_s32(vacc0123, vleft_pre_shift);
    vacc4567 = vqshlq_s32(vacc4567, vleft_pre_shift);

    vacc0123 = vqdmulhq_s32(vacc0123, vmultiplier);
    vacc4567 = vqdmulhq_s32(vacc4567, vmultiplier);

    vacc0123 = vrshlq_s32(vacc0123, vleft_post_shift);
    vacc4567 = vrshlq_s32(vacc4567, vleft_post_shift);

    #if XNN_ARCH_ARM64
      int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
    #else  // !XNN_ARCH_ARM64
      int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
    #endif  // !XNN_ARCH_ARM64

    vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);

    #if XNN_ARCH_ARM64
      uint8x8_t vout01234567 = vqmovun_s16(vacc01234567);
    #else  // !XNN_ARCH_ARM64
      uint8x8_t vout01234567 = vqmovun_s16(vacc01234567);
    #endif  // !XNN_ARCH_ARM64

    vout01234567 = vmax_u8(vout01234567, voutput_min);

    vout01234567 = vmin_u8(vout01234567, voutput_max);

    vst1_u8(output, vout01234567); output += 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {
      const uint8x8_t vi0x01234567 = vld1_u8(i0); i0 += 8;
      const uint8x8_t vi1x01234567 = vld1_u8(i1); i1 += 8;
      const uint8x8_t vi2x01234567 = vld1_u8(i2); i2 += 8;
      uint16x8_t vsum01234567 = vaddl_u8(vi0x01234567, vi1x01234567);

      const uint8x8_t vi3x01234567 = vld1_u8(i3); i3 += 8;
      vsum01234567 = vaddw_u8(vsum01234567, vi2x01234567);
      const uint8x8_t vi4x01234567 = vld1_u8(i4); i4 += 8;
      vsum01234567 = vaddw_u8(vsum01234567, vi3x01234567);
      const uint8x8_t vi5x01234567 = vld1_u8(i5); i5 += 8;
      vsum01234567 = vaddw_u8(vsum01234567, vi4x01234567);
      const uint8x8_t vi6x01234567 = vld1_u8(i6); i6 += 8;
      vsum01234567 = vaddw_u8(vsum01234567, vi5x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi6x01234567);

      int32x4_t vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vinit_bias), vget_low_u16(vsum01234567)));
      int32x4_t vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vinit_bias), vget_high_u16(vsum01234567)));

      vacc0123 = vqshlq_s32(vacc0123, vleft_pre_shift);
      vacc4567 = vqshlq_s32(vacc4567, vleft_pre_shift);

      vacc0123 = vqdmulhq_s32(vacc0123, vmultiplier);
      vacc4567 = vqdmulhq_s32(vacc4567, vmultiplier);

      vacc0123 = vrshlq_s32(vacc0123, vleft_post_shift);
      vacc4567 = vrshlq_s32(vacc4567, vleft_post_shift);

      #if XNN_ARCH_ARM64
        int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
      #else
        int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
      #endif
      vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);

      uint8x8_t vout01234567 = vqmovun_s16(vacc01234567);
      vout01234567 = vmax_u8(vout01234567, voutput_min);
      vout01234567 = vmin_u8(vout01234567, voutput_max);

      if (channels & 4) {
        vst1_lane_u32((void*) output, vreinterpret_u32_u8(vout01234567), 0); output += 4;
        vout01234567 = vext_u8(vout01234567, vout01234567, 4);
      }
      if (channels & 2) {
        vst1_lane_u16((void*) output, vreinterpret_u16_u8(vout01234567), 0); output += 2;
        vout01234567 = vext_u8(vout01234567, vout01234567, 2);
      }
      if (channels & 1) {
        vst1_lane_u8(output, vout01234567, 0);
      }
    }
  }
}
