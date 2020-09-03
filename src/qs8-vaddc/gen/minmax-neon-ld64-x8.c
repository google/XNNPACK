// Auto-generated file. Do not edit!
//   Template: src/qs8-vaddc/neon-ld64.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/vadd.h>

#include <stdio.h>
#include <inttypes.h>


void xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x8(
    size_t n,
    const int8_t* input_x,
    const int8_t* input_y,
    int8_t* output,
    const union xnn_qs8_add_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  const int8x8_t vx_zero_point = vld1_dup_s8(&params->neon.x_zero_point);
  const int32x4_t vx_multiplier = vld1q_dup_s32(&params->neon.x_multiplier);
  const int32x4_t vright_shift = vld1q_dup_s32(&params->neon.right_shift);
  const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->neon.output_zero_point);
  const int8x16_t voutput_min = vld1q_dup_s8(&params->neon.output_min);
  const int8x16_t voutput_max = vld1q_dup_s8(&params->neon.output_max);

  const int32_t vey = (int32_t) *input_y - (int32_t) params->neon.y_zero_point;
  const int32_t vy_multiplier = params->neon.y_multiplier;
  const int32x4_t vy_bias = vdupq_n_s32(vey * vy_multiplier);

  for (; n >= 8 * sizeof(int8_t); n -= 8 * sizeof(int8_t)) {
    const int8x8_t vx01234567 = vld1_s8(input_x); input_x += 8;

    const int16x8_t vex01234567 = vsubl_s8(vx01234567, vx_zero_point);

    int32x4_t vacc0123 = vmlaq_s32(vy_bias, vmovl_s16(vget_low_s16(vex01234567)), vx_multiplier);
    int32x4_t vacc4567 = vmlaq_s32(vy_bias, vmovl_s16(vget_high_s16(vex01234567)), vx_multiplier);

    vacc0123 = vsraq_n_s32(vacc0123, vbicq_s32(vacc0123, vzero_shift_mask), 31);
    vacc4567 = vsraq_n_s32(vacc4567, vbicq_s32(vacc4567, vzero_shift_mask), 31);

    vacc0123 = vrshlq_s32(vacc0123, vright_shift);
    vacc4567 = vrshlq_s32(vacc4567, vright_shift);

    const int16x8_t vacc01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567)), voutput_zero_point);

    int8x8_t vout01234567 = vqmovn_s16(vacc01234567);

    vout01234567 = vmax_s8(vout01234567, vget_low_s8(voutput_min));

    vout01234567 = vmin_s8(vout01234567, vget_low_s8(voutput_max));

    vst1_s8(output, vout01234567); output += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    {
      const int8x8_t vx01234567 = vld1_s8(input_x);

      const int16x8_t vex01234567 = vsubl_s8(vx01234567, vx_zero_point);

      int32x4_t vacc0123 = vmlaq_s32(vy_bias, vmovl_s16(vget_low_s16(vex01234567)), vx_multiplier);
      int32x4_t vacc4567 = vmlaq_s32(vy_bias, vmovl_s16(vget_high_s16(vex01234567)), vx_multiplier);

      vacc0123 = vsraq_n_s32(vacc0123, vbicq_s32(vacc0123, vzero_shift_mask), 31);
      vacc4567 = vsraq_n_s32(vacc4567, vbicq_s32(vacc4567, vzero_shift_mask), 31);

      vacc0123 = vrshlq_s32(vacc0123, vright_shift);
      vacc4567 = vrshlq_s32(vacc4567, vright_shift);

      const int16x8_t vacc01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567)), voutput_zero_point);

      int8x8_t vout01234567 = vqmovn_s16(vacc01234567);
      vout01234567 = vmax_s8(vout01234567, vget_low_s8(voutput_min));
      vout01234567 = vmin_s8(vout01234567, vget_low_s8(voutput_max));

      if (n & (4 * sizeof(int8_t))) {
        vst1_lane_u32(__builtin_assume_aligned(output, 1), vreinterpret_u32_s8(vout01234567), 0); output += 4;
        vout01234567 = vext_s8(vout01234567, vout01234567, 4);
      }
      if (n & (2 * sizeof(int8_t))) {
        vst1_lane_u16(__builtin_assume_aligned(output, 1), vreinterpret_u16_s8(vout01234567), 0); output += 2;
        vout01234567 = vext_s8(vout01234567, vout01234567, 2);
      }
      if (n & (1 * sizeof(int8_t))) {
        vst1_lane_s8(output, vout01234567, 0);
      }
    }
  }
}
