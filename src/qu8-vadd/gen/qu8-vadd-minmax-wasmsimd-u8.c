// Auto-generated file. Do not edit!
//   Template: src/qs8-vadd/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/vbinary.h"


void xnn_qu8_vadd_minmax_ukernel__wasmsimd_u8(
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

  const v128_t vbias = wasm_v128_load32_splat(&params->scalar.bias);
  const v128_t va_multiplier = wasm_v128_load32_splat(&params->scalar.a_multiplier);
  const v128_t vb_multiplier = wasm_v128_load32_splat(&params->scalar.b_multiplier);
  const uint32_t vshift = params->scalar.shift;
  const v128_t voutput_zero_point = wasm_v128_load16_splat(&params->scalar.output_zero_point);
  const v128_t voutput_min = wasm_v128_load8_splat(&params->scalar.output_min);
  const v128_t voutput_max = wasm_v128_load8_splat(&params->scalar.output_max);

  XNN_FORCE_REALIZATION(vbias);
  XNN_FORCE_REALIZATION(va_multiplier);
  XNN_FORCE_REALIZATION(vb_multiplier);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    const v128_t va01234567 = wasm_u16x8_load8x8(input_a);
    const v128_t vb01234567 = wasm_u16x8_load8x8(input_b);
    input_a += 8;
    input_b += 8;

    v128_t vacc0123 = wasm_i32x4_add(vbias, wasm_i32x4_mul(wasm_u32x4_extend_low_u16x8(va01234567), va_multiplier));
    v128_t vacc4567 = wasm_i32x4_add(vbias, wasm_i32x4_mul(wasm_u32x4_extend_high_u16x8(va01234567), va_multiplier));

    vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_mul(wasm_u32x4_extend_low_u16x8(vb01234567), vb_multiplier));
    vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_mul(wasm_u32x4_extend_high_u16x8(vb01234567), vb_multiplier));

    vacc0123 = wasm_i32x4_shr(vacc0123, vshift);
    vacc4567 = wasm_i32x4_shr(vacc4567, vshift);

    v128_t vout01234567 = wasm_i16x8_add_sat(wasm_i16x8_narrow_i32x4(vacc0123, vacc4567), voutput_zero_point);

    v128_t vout0123456701234567 = wasm_u8x16_narrow_i16x8(vout01234567, vout01234567);

    vout0123456701234567 = wasm_u8x16_max(vout0123456701234567, voutput_min);

    vout0123456701234567 = wasm_u8x16_min(vout0123456701234567, voutput_max);

    wasm_v128_store64_lane(output, vout0123456701234567, 0);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      const v128_t va01234567 = wasm_u16x8_load8x8(input_a);
      const v128_t vb01234567 = wasm_u16x8_load8x8(input_b);

      v128_t vacc0123 = wasm_i32x4_add(vbias, wasm_i32x4_mul(wasm_u32x4_extend_low_u16x8(va01234567), va_multiplier));
      v128_t vacc4567 = wasm_i32x4_add(vbias, wasm_i32x4_mul(wasm_u32x4_extend_high_u16x8(va01234567), va_multiplier));

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_mul(wasm_u32x4_extend_low_u16x8(vb01234567), vb_multiplier));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_mul(wasm_u32x4_extend_high_u16x8(vb01234567), vb_multiplier));

      vacc0123 = wasm_i32x4_shr(vacc0123, vshift);
      vacc4567 = wasm_i32x4_shr(vacc4567, vshift);

      v128_t vout01234567 = wasm_i16x8_add_sat(wasm_i16x8_narrow_i32x4(vacc0123, vacc4567), voutput_zero_point);

      v128_t vout0123456701234567 = wasm_u8x16_narrow_i16x8(vout01234567, vout01234567);
      vout0123456701234567 = wasm_u8x16_max(vout0123456701234567, voutput_min);
      vout0123456701234567 = wasm_u8x16_min(vout0123456701234567, voutput_max);

      if (batch & (4 * sizeof(uint8_t))) {
        wasm_v128_store32_lane(output, vout0123456701234567, 0);
        vout0123456701234567 = wasm_u64x2_shr(vout0123456701234567, 32);
        output += 4;
      }
      if (batch & (2 * sizeof(uint8_t))) {
        wasm_v128_store16_lane(output, vout0123456701234567, 0);
        vout0123456701234567 = wasm_u32x4_shr(vout0123456701234567, 16);
        output += 2;
      }
      if (batch & (1 * sizeof(uint8_t))) {
        wasm_v128_store8_lane(output, vout0123456701234567, 0);
      }
    }
  }
}
