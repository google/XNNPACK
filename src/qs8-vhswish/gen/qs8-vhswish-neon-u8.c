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
#include "xnnpack/vunary.h"


void xnn_qs8_vhswish_ukernel__neon_u8(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_hswish_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int16x8_t vinput_zero_point = vld1q_dup_s16(&params->scalar.input_zero_point);
  const int16x8_t vinput_scale_div_exp = 	vdupq_n_s16(params->scalar.input_scale_div_exp + 15);
  const int16x8_t vinput_scale_div_mantissa = vld1q_dup_s16(&params->scalar.input_scale_div_mantissa);
  const int16x8_t vscale_ratio = vld1q_dup_s16(&params->scalar.scale_ratio);
  const int16x8_t vhalf = vdupq_n_s16(16384);
  const int16x8_t vzero = vdupq_n_s16(0);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->scalar.output_zero_point);
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    const int8x8_t vx = vld1_s8(input); input += 8;
    int16x8_t vacc = vsubw_s8(vinput_zero_point, vx);
    vacc = vshlq_n_s16(vacc, 7);
    int16x8_t vin = vqrdmulhq_s16(vacc, vinput_scale_div_mantissa);
    vin = vqshlq_s16(vin, vinput_scale_div_exp);
    vin = vqsubq_s16(vin, vhalf);
    vin = vminq_s16(vin, vzero);
    int16x8_t vout = vqrdmulhq_s16(vacc, vscale_ratio);
    vout = vqrdmulhq_s16(vout, vin);
    vout = vqaddq_s16(vout, voutput_zero_point);
    const int8x8_t vy = vqmovn_s16(vout);
    vst1_s8(output, vy); output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    const int8x8_t vx = vld1_s8(input);
    int16x8_t vacc = vsubw_s8(vinput_zero_point, vx);
    vacc = vshlq_n_s16(vacc, 7);
    int16x8_t vin = vqrdmulhq_s16(vacc, vinput_scale_div_mantissa);
    vin = vqshlq_s16(vin, vinput_scale_div_exp);
    vin = vqsubq_s16(vin, vhalf);
    vin = vminq_s16(vin, vzero);
    int16x8_t vout = vqrdmulhq_s16(vacc, vscale_ratio);
    vout = vqrdmulhq_s16(vout, vin);
    vout = vqaddq_s16(vout, voutput_zero_point);
    int8x8_t vy = vqmovn_s16(vout);

    if (batch & (4 * sizeof(int8_t))) {
      vst1_lane_u32((void*) output, vreinterpret_u32_s8(vy), 0); output += 4;
      vy = vext_s8(vy, vy, 4);
    }
    if (batch & (2 * sizeof(int8_t))) {
      vst1_lane_u16((void*) output, vreinterpret_u16_s8(vy), 0); output += 2;
      vy = vext_s8(vy, vy, 2);
    }
    if (batch & (1 * sizeof(int8_t))) {
      vst1_lane_s8(output, vy, 0);
    }
  }
}
