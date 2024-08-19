// Auto-generated file. Do not edit!
//   Template: src/qs8-vmulc/wasmsimd-mul32-ld64.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/vbinary.h"


void xnn_qu8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u16(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float output_min_less_zero_point = (float) ((int32_t) params->scalar.output_min - (int32_t) params->scalar.output_zero_point);
  const v128_t va_zero_point = wasm_i16x8_splat(params->scalar.a_zero_point);
  const v128_t vscale = wasm_v128_load32_splat(&params->scalar.scale);
  const v128_t vmagic_bias = wasm_f32x4_splat(12582912.0f);
  const v128_t vmagic_min = wasm_i32x4_splat((int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point));
  const v128_t vmagic_bias_less_output_zero_point = wasm_i32x4_splat(INT32_C(0x4B400000) - (int32_t) params->scalar.output_zero_point);
  const v128_t voutput_max = wasm_v128_load8_splat(&params->scalar.output_max);
  const v128_t vxb = wasm_i16x8_splat((int16_t) *input_b - params->scalar.b_zero_point);

  XNN_FORCE_REALIZATION(va_zero_point);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vmagic_min);
  XNN_FORCE_REALIZATION(vmagic_bias_less_output_zero_point);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    const v128_t va01234567 = wasm_u16x8_load8x8(input_a);
    const v128_t va89ABCDEF = wasm_u16x8_load8x8(input_a + 8);
    input_a += 16;

    const v128_t vxa01234567 = wasm_i16x8_sub(va01234567, va_zero_point);
    const v128_t vxa89ABCDEF = wasm_i16x8_sub(va89ABCDEF, va_zero_point);

    v128_t vacc0123 = wasm_i32x4_extmul_low_i16x8(vxa01234567, vxb);
    v128_t vacc4567 = wasm_i32x4_extmul_high_i16x8(vxa01234567, vxb);
    v128_t vacc89AB = wasm_i32x4_extmul_low_i16x8(vxa89ABCDEF, vxb);
    v128_t vaccCDEF = wasm_i32x4_extmul_high_i16x8(vxa89ABCDEF, vxb);

    vacc0123 = wasm_f32x4_convert_i32x4(vacc0123);
    vacc4567 = wasm_f32x4_convert_i32x4(vacc4567);
    vacc89AB = wasm_f32x4_convert_i32x4(vacc89AB);
    vaccCDEF = wasm_f32x4_convert_i32x4(vaccCDEF);

    vacc0123 = wasm_f32x4_mul(vacc0123, vscale);
    vacc4567 = wasm_f32x4_mul(vacc4567, vscale);
    vacc89AB = wasm_f32x4_mul(vacc89AB, vscale);
    vaccCDEF = wasm_f32x4_mul(vaccCDEF, vscale);

    vacc0123 = wasm_f32x4_add(vacc0123, vmagic_bias);
    vacc4567 = wasm_f32x4_add(vacc4567, vmagic_bias);
    vacc89AB = wasm_f32x4_add(vacc89AB, vmagic_bias);
    vaccCDEF = wasm_f32x4_add(vaccCDEF, vmagic_bias);

    vacc0123 = wasm_i32x4_max(vacc0123, vmagic_min);
    vacc4567 = wasm_i32x4_max(vacc4567, vmagic_min);
    vacc89AB = wasm_i32x4_max(vacc89AB, vmagic_min);
    vaccCDEF = wasm_i32x4_max(vaccCDEF, vmagic_min);

    vacc0123 = wasm_i32x4_sub(vacc0123, vmagic_bias_less_output_zero_point);
    vacc4567 = wasm_i32x4_sub(vacc4567, vmagic_bias_less_output_zero_point);
    vacc89AB = wasm_i32x4_sub(vacc89AB, vmagic_bias_less_output_zero_point);
    vaccCDEF = wasm_i32x4_sub(vaccCDEF, vmagic_bias_less_output_zero_point);

    v128_t vout01234567 = wasm_i16x8_narrow_i32x4(vacc0123, vacc4567);
    v128_t vout89ABCDEF = wasm_i16x8_narrow_i32x4(vacc89AB, vaccCDEF);

    v128_t vout0123456789ABCDEF = wasm_u8x16_narrow_i16x8(vout01234567, vout89ABCDEF);

    vout0123456789ABCDEF = wasm_u8x16_min(vout0123456789ABCDEF, voutput_max);

    wasm_v128_store(output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const v128_t va01234567 = wasm_u16x8_load8x8(input_a);
      input_a += 8;

      const v128_t vxa01234567 = wasm_i16x8_sub(va01234567, va_zero_point);

      v128_t vacc0123 = wasm_i32x4_extmul_low_i16x8(vxa01234567, vxb);
      v128_t vacc4567 = wasm_i32x4_extmul_high_i16x8(vxa01234567, vxb);

      vacc0123 = wasm_f32x4_convert_i32x4(vacc0123);
      vacc4567 = wasm_f32x4_convert_i32x4(vacc4567);

      vacc0123 = wasm_f32x4_mul(vacc0123, vscale);
      vacc4567 = wasm_f32x4_mul(vacc4567, vscale);

      vacc0123 = wasm_f32x4_add(vacc0123, vmagic_bias);
      vacc4567 = wasm_f32x4_add(vacc4567, vmagic_bias);

      vacc0123 = wasm_i32x4_max(vacc0123, vmagic_min);
      vacc4567 = wasm_i32x4_max(vacc4567, vmagic_min);

      vacc0123 = wasm_i32x4_sub(vacc0123, vmagic_bias_less_output_zero_point);
      vacc4567 = wasm_i32x4_sub(vacc4567, vmagic_bias_less_output_zero_point);

      v128_t vout01234567 = wasm_i16x8_narrow_i32x4(vacc0123, vacc4567);
      v128_t vout0123456701234567 = wasm_u8x16_narrow_i16x8(vout01234567, vout01234567);
      vout0123456701234567 = wasm_u8x16_min(vout0123456701234567, voutput_max);

      if XNN_LIKELY(batch >= (8 * sizeof(uint8_t))) {
        wasm_v128_store64_lane(output, vout0123456701234567, 0);
        output += 8;
        batch -= 8 * sizeof(uint8_t);
      } else {
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
        batch = 0;
      }
    } while (batch != 0);
  }
}
