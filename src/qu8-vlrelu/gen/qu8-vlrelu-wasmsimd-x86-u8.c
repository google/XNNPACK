// Auto-generated file. Do not edit!
//   Template: src/qs8-vlrelu/wasmsimd-x86.c.in
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


void xnn_qu8_vlrelu_ukernel__wasmsimd_x86_u8(
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
  const v128_t vmultiplier_diff = wasm_i16x8_splat(-params->scalar.negative_multiplier ^ -params->scalar.positive_multiplier);
  const v128_t vmultiplier_base = wasm_i16x8_splat(-params->scalar.negative_multiplier);
  const v128_t voutput_zero_point = wasm_v128_load16_splat(&params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    v128_t vacc = wasm_u16x8_load8x8(input);
    v128_t vmultiplier = wasm_i16x8_gt(vacc, vinput_zero_point);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vmultiplier = wasm_v128_and(vmultiplier, vmultiplier_diff);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_v128_xor(vmultiplier, vmultiplier_base);
    vacc = wasm_i16x8_q15mulr_sat(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);
    input += 8;

    const v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);
    wasm_v128_store64_lane(output, vy, 0);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 7 * sizeof(uint8_t));

    v128_t vacc = wasm_u16x8_load8x8(input);
    v128_t vmultiplier = wasm_i16x8_gt(vacc, vinput_zero_point);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vmultiplier = wasm_v128_and(vmultiplier, vmultiplier_diff);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_v128_xor(vmultiplier, vmultiplier_base);
    vacc = wasm_i16x8_q15mulr_sat(vacc, vmultiplier);
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
