// Auto-generated file. Do not edit!
//   Template: src/qs8-vlrelu/wasmsimd-arm.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/vcvt.h"


void xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_arm_u32(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vinput_zero_point = wasm_v128_load16_splat(&params->scalar.input_zero_point);
  const v128_t vpositive_multiplier = wasm_i16x8_splat(-params->scalar.positive_multiplier);
  const v128_t vnegative_multiplier = wasm_i16x8_splat(-params->scalar.negative_multiplier);
  const v128_t voutput_zero_point = wasm_v128_load16_splat(&params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(vpositive_multiplier);
  XNN_FORCE_REALIZATION(vnegative_multiplier);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    v128_t vx0 = wasm_v128_load(input);
    v128_t vx1 = wasm_v128_load(input + 16);
    input += 32;

    v128_t vacc0 = wasm_i16x8_sub(vinput_zero_point, wasm_u16x8_extend_low_u8x16(vx0));
    v128_t vacc1 = wasm_i16x8_sub(vinput_zero_point, wasm_u16x8_extend_high_u8x16(vx0));
    v128_t vmultiplier0 = wasm_i16x8_shr(vacc0, 15);
    v128_t vmultiplier1 = wasm_i16x8_shr(vacc1, 15);
    v128_t vacc2 = wasm_i16x8_sub(vinput_zero_point, wasm_u16x8_extend_low_u8x16(vx1));
    v128_t vacc3 = wasm_i16x8_sub(vinput_zero_point, wasm_u16x8_extend_high_u8x16(vx1));
    v128_t vmultiplier2 = wasm_i16x8_shr(vacc2, 15);
    v128_t vmultiplier3 = wasm_i16x8_shr(vacc3, 15);

    vacc0 = wasm_i16x8_shl(vacc0, 7);
    vmultiplier0 = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier0);
    vacc1 = wasm_i16x8_shl(vacc1, 7);
    vmultiplier1 = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier1);
    vacc2 = wasm_i16x8_shl(vacc2, 7);
    vmultiplier2 = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier2);
    vacc3 = wasm_i16x8_shl(vacc3, 7);
    vmultiplier3 = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier3);

    vacc0 = wasm_i16x8_relaxed_q15mulr(vacc0, vmultiplier0);
    vacc1 = wasm_i16x8_relaxed_q15mulr(vacc1, vmultiplier1);
    vacc2 = wasm_i16x8_relaxed_q15mulr(vacc2, vmultiplier2);
    vacc3 = wasm_i16x8_relaxed_q15mulr(vacc3, vmultiplier3);

    vacc0 = wasm_i16x8_add_sat(vacc0, voutput_zero_point);
    vacc1 = wasm_i16x8_add_sat(vacc1, voutput_zero_point);
    vacc2 = wasm_i16x8_add_sat(vacc2, voutput_zero_point);
    vacc3 = wasm_i16x8_add_sat(vacc3, voutput_zero_point);

    const v128_t vy0 = wasm_u8x16_narrow_i16x8(vacc0, vacc1);
    const v128_t vy1 = wasm_u8x16_narrow_i16x8(vacc2, vacc3);

    wasm_v128_store(output, vy0);
    wasm_v128_store((output + 16), vy1);
    output += 32;
  }
  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    const v128_t vx = wasm_u16x8_load8x8(input);
    v128_t vacc = wasm_i16x8_sub(vinput_zero_point, vx);
    v128_t vmultiplier = wasm_i16x8_shr(vacc, 15);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier);
    vacc = wasm_i16x8_relaxed_q15mulr(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);
    input += 8;

    const v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);
    wasm_v128_store64_lane(output, vy, 0);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 7 * sizeof(uint8_t));

    const v128_t vx = wasm_u16x8_load8x8(input);
    v128_t vacc = wasm_i16x8_sub(vinput_zero_point, vx);
    v128_t vmultiplier = wasm_i16x8_shr(vacc, 15);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier);
    vacc = wasm_i16x8_relaxed_q15mulr(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);

    v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);
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
