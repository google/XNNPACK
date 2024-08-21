// Auto-generated file. Do not edit!
//   Template: src/qs8-vhswish/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/vhswish.h"


void xnn_qu8_vhswish_ukernel__wasmsimd_u16(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_hswish_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int16_t shift_max = (int16_t) 1 << (15 - params->scalar.input_scale_div_exp);
  const v128_t vinput_zero_point = wasm_v128_load16_splat(&params->scalar.input_zero_point);
  const v128_t voutput_zero_point = wasm_v128_load16_splat(&params->scalar.output_zero_point);
  const v128_t vinput_scale_div_mantissa = wasm_v128_load16_splat(&params->scalar.input_scale_div_mantissa);
  const v128_t vshift_max = wasm_i16x8_splat(shift_max);
  const v128_t vshift_min = wasm_i16x8_splat(-shift_max);
  const v128_t vscale_ratio = wasm_v128_load16_splat(&params->scalar.scale_ratio);
  const v128_t vmax_val = wasm_u16x8_const_splat(0x7FFF);
  const v128_t vmin_val = wasm_u16x8_const_splat(0x8000);
  const v128_t vhalf = wasm_u16x8_const_splat(0x4000);
  const v128_t vzero = wasm_u16x8_const_splat(0);
  const v128_t vinput_scale_div_exp = wasm_i16x8_splat(1 << params->scalar.input_scale_div_exp);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(vinput_scale_div_mantissa);
  XNN_FORCE_REALIZATION(vshift_max);
  XNN_FORCE_REALIZATION(vshift_min);
  XNN_FORCE_REALIZATION(vscale_ratio);
  XNN_FORCE_REALIZATION(vmax_val);
  XNN_FORCE_REALIZATION(vmin_val);
  XNN_FORCE_REALIZATION(vhalf);
  XNN_FORCE_REALIZATION(vzero);
  XNN_FORCE_REALIZATION(vinput_scale_div_exp);

  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    v128_t vacc0 = wasm_u16x8_load8x8(input);
    v128_t vacc1 = wasm_u16x8_load8x8(input + 8);
    input += 16;

    vacc0 = wasm_i16x8_sub(vinput_zero_point, vacc0);
    vacc1 = wasm_i16x8_sub(vinput_zero_point, vacc1);

    vacc0 = wasm_i16x8_shl(vacc0, 7);
    vacc1 = wasm_i16x8_shl(vacc1, 7);

    const v128_t vbase0 = wasm_i16x8_q15mulr_sat(vacc0, vinput_scale_div_mantissa);
    const v128_t vbase1 = wasm_i16x8_q15mulr_sat(vacc1, vinput_scale_div_mantissa);

    const v128_t vshifted0 = wasm_i16x8_mul(vbase0, vinput_scale_div_exp);
    const v128_t vshifted1 = wasm_i16x8_mul(vbase1, vinput_scale_div_exp);

    const v128_t pos_mask0 = wasm_i16x8_ge(vbase0, vshift_max);
    v128_t vin0 = wasm_v128_bitselect(vmax_val, vshifted0, pos_mask0);
    const v128_t pos_mask1 = wasm_i16x8_ge(vbase1, vshift_max);
    v128_t vin1 = wasm_v128_bitselect(vmax_val, vshifted1, pos_mask1);

    const v128_t neg_mask0 = wasm_i16x8_le(vbase0, vshift_min);
    vin0 = wasm_v128_bitselect(vmin_val, vin0, neg_mask0);
    const v128_t neg_mask1 = wasm_i16x8_le(vbase1, vshift_min);
    vin1 = wasm_v128_bitselect(vmin_val, vin1, neg_mask1);

    vin0 = wasm_i16x8_sub_sat(vin0, vhalf);
    vin1 = wasm_i16x8_sub_sat(vin1, vhalf);

    vin0 = wasm_i16x8_min(vin0, vzero);
    vin1 = wasm_i16x8_min(vin1, vzero);

    v128_t vout0 = wasm_i16x8_q15mulr_sat(vacc0, vscale_ratio);
    v128_t vout1 = wasm_i16x8_q15mulr_sat(vacc1, vscale_ratio);

    vout0 = wasm_i16x8_q15mulr_sat(vin0, vout0);
    vout1 = wasm_i16x8_q15mulr_sat(vin1, vout1);

    vout0 = wasm_i16x8_add_sat(vout0, voutput_zero_point);
    vout1 = wasm_i16x8_add_sat(vout1, voutput_zero_point);

    const v128_t vy0 = wasm_u8x16_narrow_i16x8(vout0, vout1);

    wasm_v128_store(output, vy0);
    output += 16;
  }
  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    v128_t vacc = wasm_u16x8_load8x8(input);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vacc = wasm_i16x8_shl(vacc, 7);
    const v128_t vbase = wasm_i16x8_q15mulr_sat(vacc, vinput_scale_div_mantissa);
    const v128_t vshifted = wasm_i16x8_mul(vbase, vinput_scale_div_exp);
    const v128_t pos_mask = wasm_i16x8_ge(vbase, vshift_max);
    v128_t vin = wasm_v128_bitselect(vmax_val, vshifted, pos_mask);
    const v128_t neg_mask = wasm_i16x8_le(vbase, vshift_min);
    vin = wasm_v128_bitselect(vmin_val, vin, neg_mask);
    vin = wasm_i16x8_sub_sat(vin, vhalf);
    vin = wasm_i16x8_min(vin, vzero);
    v128_t vout = wasm_i16x8_q15mulr_sat(vacc, vscale_ratio);
    vout = wasm_i16x8_q15mulr_sat(vin, vout);
    vout = wasm_i16x8_add_sat(vout, voutput_zero_point);
    input += 8;

    const v128_t vy = wasm_u8x16_narrow_i16x8(vout, vout);
    wasm_v128_store64_lane(output, vy, 0);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 7 * sizeof(uint8_t));

    v128_t vacc = wasm_u16x8_load8x8(input);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vacc = wasm_i16x8_shl(vacc, 7);
    const v128_t vbase = wasm_i16x8_q15mulr_sat(vacc, vinput_scale_div_mantissa);
    const v128_t vshifted = wasm_i16x8_mul(vbase, vinput_scale_div_exp);
    const v128_t pos_mask = wasm_i16x8_ge(vbase, vshift_max);
    v128_t vin = wasm_v128_bitselect(vmax_val, vshifted, pos_mask);
    const v128_t neg_mask = wasm_i16x8_le(vbase, vshift_min);
    vin = wasm_v128_bitselect(vmin_val, vin, neg_mask);
    vin = wasm_i16x8_sub_sat(vin, vhalf);
    vin = wasm_i16x8_min(vin, vzero);
    v128_t vout = wasm_i16x8_q15mulr_sat(vacc, vscale_ratio);
    vout = wasm_i16x8_q15mulr_sat(vin, vout);
    vout = wasm_i16x8_add_sat(vout, voutput_zero_point);

    v128_t vy = wasm_u8x16_narrow_i16x8(vout, vout);
    if (batch & (4 * sizeof(uint8_t))) {
      wasm_v128_store32_lane(output, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(uint8_t))) {
      wasm_v128_store16_lane(output, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      wasm_v128_store8_lane(output, vy, 0);
    }
  }
}
