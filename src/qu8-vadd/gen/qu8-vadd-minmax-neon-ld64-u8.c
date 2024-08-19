// Auto-generated file. Do not edit!
//   Template: src/qs8-vadd/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/vbinary.h"


void xnn_qu8_vadd_minmax_ukernel__neon_ld64_u8(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint8x8_t va_zero_point = vld1_dup_u8(&params->scalar.a_zero_point);
  const uint8x8_t vb_zero_point = vld1_dup_u8(&params->scalar.b_zero_point);
  const int32x4_t va_multiplier = vld1q_dup_s32(&params->scalar.a_multiplier);
  const int32x4_t vb_multiplier = vld1q_dup_s32(&params->scalar.b_multiplier);
  const int32x4_t vright_shift = vdupq_n_s32(-params->scalar.shift);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->scalar.output_zero_point);
  const uint8x8_t voutput_min = vld1_dup_u8(&params->scalar.output_min);
  const uint8x8_t voutput_max = vld1_dup_u8(&params->scalar.output_max);

  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    const uint8x8_t va01234567 = vld1_u8(input_a); input_a += 8;
    const uint8x8_t vb01234567 = vld1_u8(input_b); input_b += 8;

    const int16x8_t vxa01234567 = vreinterpretq_s16_u16(vsubl_u8(va01234567, va_zero_point));
    const int16x8_t vxb01234567 = vreinterpretq_s16_u16(vsubl_u8(vb01234567, vb_zero_point));

    int32x4_t vacc0123 = vmulq_s32(vmovl_s16(vget_low_s16(vxa01234567)), va_multiplier);
    int32x4_t vacc4567 = vmulq_s32(vmovl_s16(vget_high_s16(vxa01234567)), va_multiplier);

    vacc0123 = vmlaq_s32(vacc0123, vmovl_s16(vget_low_s16(vxb01234567)), vb_multiplier);
    vacc4567 = vmlaq_s32(vacc4567, vmovl_s16(vget_high_s16(vxb01234567)), vb_multiplier);

    vacc0123 = vrshlq_s32(vacc0123, vright_shift);
    vacc4567 = vrshlq_s32(vacc4567, vright_shift);

    const int16x8_t vacc01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567)), voutput_zero_point);

    uint8x8_t vout01234567 = vqmovun_s16(vacc01234567);

    vout01234567 = vmax_u8(vout01234567, voutput_min);

    vout01234567 = vmin_u8(vout01234567, voutput_max);

    vst1_u8(output, vout01234567); output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      const uint8x8_t va01234567 = vld1_u8(input_a);
      const uint8x8_t vb01234567 = vld1_u8(input_b);

      const int16x8_t vxa01234567 = vreinterpretq_s16_u16(vsubl_u8(va01234567, va_zero_point));
      const int16x8_t vxb01234567 = vreinterpretq_s16_u16(vsubl_u8(vb01234567, vb_zero_point));

      int32x4_t vacc0123 = vmulq_s32(vmovl_s16(vget_low_s16(vxa01234567)), va_multiplier);
      int32x4_t vacc4567 = vmulq_s32(vmovl_s16(vget_high_s16(vxa01234567)), va_multiplier);

      vacc0123 = vmlaq_s32(vacc0123, vmovl_s16(vget_low_s16(vxb01234567)), vb_multiplier);
      vacc4567 = vmlaq_s32(vacc4567, vmovl_s16(vget_high_s16(vxb01234567)), vb_multiplier);

      vacc0123 = vrshlq_s32(vacc0123, vright_shift);
      vacc4567 = vrshlq_s32(vacc4567, vright_shift);

      const int16x8_t vacc01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567)), voutput_zero_point);

      uint8x8_t vout01234567 = vqmovun_s16(vacc01234567);
      vout01234567 = vmax_u8(vout01234567, voutput_min);
      vout01234567 = vmin_u8(vout01234567, voutput_max);

      if (batch & (4 * sizeof(uint8_t))) {
        vst1_lane_u32((void*) output, vreinterpret_u32_u8(vout01234567), 0); output += 4;
        vout01234567 = vext_u8(vout01234567, vout01234567, 4);
      }
      if (batch & (2 * sizeof(uint8_t))) {
        vst1_lane_u16((void*) output, vreinterpret_u16_u8(vout01234567), 0); output += 2;
        vout01234567 = vext_u8(vout01234567, vout01234567, 2);
      }
      if (batch & (1 * sizeof(uint8_t))) {
        vst1_lane_u8(output, vout01234567, 0);
      }
    }
  }
}
