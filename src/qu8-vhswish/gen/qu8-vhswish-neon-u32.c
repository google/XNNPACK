// Auto-generated file. Do not edit!
//   Template: src/qs8-vhswish/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/vhswish.h"


void xnn_qu8_vhswish_ukernel__neon_u32(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_hswish_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16x8_t vinput_zero_point = vld1q_dup_u16(&params->scalar.input_zero_point);
  const int16x8_t vinput_scale_div_exp = 	vdupq_n_s16(params->scalar.input_scale_div_exp + 15);
  const int16x8_t vinput_scale_div_mantissa = vld1q_dup_s16(&params->scalar.input_scale_div_mantissa);
  const int16x8_t vscale_ratio = vld1q_dup_s16(&params->scalar.scale_ratio);
  const int16x8_t vhalf = vdupq_n_s16(16384);
  const int16x8_t vzero = vdupq_n_s16(0);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->scalar.output_zero_point);
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    const uint8x16_t vx0 = vld1q_u8(input); input += 16;
    const uint8x16_t vx1 = vld1q_u8(input); input += 16;
    int16x8_t vacc0 = vreinterpretq_s16_u16(vsubw_u8(vinput_zero_point, vget_low_u8(vx0)));
    int16x8_t vacc1 = vreinterpretq_s16_u16(vsubw_u8(vinput_zero_point, vget_high_u8(vx0)));
    int16x8_t vacc2 = vreinterpretq_s16_u16(vsubw_u8(vinput_zero_point, vget_low_u8(vx1)));
    int16x8_t vacc3 = vreinterpretq_s16_u16(vsubw_u8(vinput_zero_point, vget_high_u8(vx1)));
    vacc0 = vshlq_n_s16(vacc0, 7);
    vacc1 = vshlq_n_s16(vacc1, 7);
    vacc2 = vshlq_n_s16(vacc2, 7);
    vacc3 = vshlq_n_s16(vacc3, 7);
    int16x8_t vin0 = vqrdmulhq_s16(vacc0, vinput_scale_div_mantissa);
    int16x8_t vin1 = vqrdmulhq_s16(vacc1, vinput_scale_div_mantissa);
    int16x8_t vin2 = vqrdmulhq_s16(vacc2, vinput_scale_div_mantissa);
    int16x8_t vin3 = vqrdmulhq_s16(vacc3, vinput_scale_div_mantissa);
    vin0 = vqshlq_s16(vin0, vinput_scale_div_exp);
    vin1 = vqshlq_s16(vin1, vinput_scale_div_exp);
    vin2 = vqshlq_s16(vin2, vinput_scale_div_exp);
    vin3 = vqshlq_s16(vin3, vinput_scale_div_exp);
    vin0 = vqsubq_s16(vin0, vhalf);
    vin1 = vqsubq_s16(vin1, vhalf);
    vin2 = vqsubq_s16(vin2, vhalf);
    vin3 = vqsubq_s16(vin3, vhalf);
    vin0 = vminq_s16(vin0, vzero);
    vin1 = vminq_s16(vin1, vzero);
    vin2 = vminq_s16(vin2, vzero);
    vin3 = vminq_s16(vin3, vzero);
    int16x8_t vout0 = vqrdmulhq_s16(vacc0, vscale_ratio);
    int16x8_t vout1 = vqrdmulhq_s16(vacc1, vscale_ratio);
    int16x8_t vout2 = vqrdmulhq_s16(vacc2, vscale_ratio);
    int16x8_t vout3 = vqrdmulhq_s16(vacc3, vscale_ratio);
    vout0 = vqrdmulhq_s16(vout0, vin0);
    vout1 = vqrdmulhq_s16(vout1, vin1);
    vout2 = vqrdmulhq_s16(vout2, vin2);
    vout3 = vqrdmulhq_s16(vout3, vin3);
    vout0 = vqaddq_s16(vout0, voutput_zero_point);
    vout1 = vqaddq_s16(vout1, voutput_zero_point);
    vout2 = vqaddq_s16(vout2, voutput_zero_point);
    vout3 = vqaddq_s16(vout3, voutput_zero_point);
    const uint8x16_t vy0 = vcombine_u8(vqmovun_s16(vout0), vqmovun_s16(vout1));
    const uint8x16_t vy1 = vcombine_u8(vqmovun_s16(vout2), vqmovun_s16(vout3));
    vst1q_u8(output, vy0); output += 16;
    vst1q_u8(output, vy1); output += 16;
  }
  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    const uint8x8_t vx = vld1_u8(input); input += 8;
    int16x8_t vacc = vreinterpretq_s16_u16(vsubw_u8(vinput_zero_point, vx));
    vacc = vshlq_n_s16(vacc, 7);
    int16x8_t vin = vqrdmulhq_s16(vacc, vinput_scale_div_mantissa);
    vin = vqshlq_s16(vin, vinput_scale_div_exp);
    vin = vqsubq_s16(vin, vhalf);
    vin = vminq_s16(vin, vzero);
    int16x8_t vout = vqrdmulhq_s16(vacc, vscale_ratio);
    vout = vqrdmulhq_s16(vout, vin);
    vout = vqaddq_s16(vout, voutput_zero_point);
    const uint8x8_t vy = vqmovun_s16(vout);
    vst1_u8(output, vy); output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 7 * sizeof(uint8_t));

    const uint8x8_t vx = vld1_u8(input);
    int16x8_t vacc = vreinterpretq_s16_u16(vsubw_u8(vinput_zero_point, vx));
    vacc = vshlq_n_s16(vacc, 7);
    int16x8_t vin = vqrdmulhq_s16(vacc, vinput_scale_div_mantissa);
    vin = vqshlq_s16(vin, vinput_scale_div_exp);
    vin = vqsubq_s16(vin, vhalf);
    vin = vminq_s16(vin, vzero);
    int16x8_t vout = vqrdmulhq_s16(vacc, vscale_ratio);
    vout = vqrdmulhq_s16(vout, vin);
    vout = vqaddq_s16(vout, voutput_zero_point);
    uint8x8_t vy = vqmovun_s16(vout);

    if (batch & (4 * sizeof(uint8_t))) {
      vst1_lane_u32((void*) output, vreinterpret_u32_u8(vy), 0); output += 4;
      vy = vext_u8(vy, vy, 4);
    }
    if (batch & (2 * sizeof(uint8_t))) {
      vst1_lane_u16((void*) output, vreinterpret_u16_u8(vy), 0); output += 2;
      vy = vext_u8(vy, vy, 2);
    }
    if (batch & (1 * sizeof(uint8_t))) {
      vst1_lane_u8(output, vy, 0);
    }
  }
}
