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


void xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u16(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
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
  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    v128_t vx0 = wasm_v128_load(input);
    input += 16;

    v128_t vacc0 = wasm_i16x8_sub(vinput_zero_point, wasm_i16x8_extend_low_i8x16(vx0));
    v128_t vacc1 = wasm_i16x8_sub(vinput_zero_point, wasm_i16x8_extend_high_i8x16(vx0));
    v128_t vmultiplier0 = wasm_i16x8_shr(vacc0, 15);
    v128_t vmultiplier1 = wasm_i16x8_shr(vacc1, 15);

    vacc0 = wasm_i16x8_shl(vacc0, 7);
    vmultiplier0 = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier0);
    vacc1 = wasm_i16x8_shl(vacc1, 7);
    vmultiplier1 = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier1);

    vacc0 = wasm_i16x8_relaxed_q15mulr(vacc0, vmultiplier0);
    vacc1 = wasm_i16x8_relaxed_q15mulr(vacc1, vmultiplier1);

    vacc0 = wasm_i16x8_add_sat(vacc0, voutput_zero_point);
    vacc1 = wasm_i16x8_add_sat(vacc1, voutput_zero_point);

    const v128_t vy0 = wasm_i8x16_narrow_i16x8(vacc0, vacc1);

    wasm_v128_store(output, vy0);
    output += 16;
  }
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    const v128_t vx = wasm_i16x8_load8x8(input);
    v128_t vacc = wasm_i16x8_sub(vinput_zero_point, vx);
    v128_t vmultiplier = wasm_i16x8_shr(vacc, 15);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier);
    vacc = wasm_i16x8_relaxed_q15mulr(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);
    input += 8;

    const v128_t vy = wasm_i8x16_narrow_i16x8(vacc, vacc);
    wasm_v128_store64_lane(output, vy, 0);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    const v128_t vx = wasm_i16x8_load8x8(input);
    v128_t vacc = wasm_i16x8_sub(vinput_zero_point, vx);
    v128_t vmultiplier = wasm_i16x8_shr(vacc, 15);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier);
    vacc = wasm_i16x8_relaxed_q15mulr(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);

    v128_t vy = wasm_i8x16_narrow_i16x8(vacc, vacc);
    if (batch & (4 * sizeof(int8_t))) {
      wasm_v128_store32_lane(output, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(int8_t))) {
      wasm_v128_store16_lane(output, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      wasm_v128_store8_lane(output, vy, 0);
    }
  }
}
