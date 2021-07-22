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

#include <xnnpack/vadd.h>


void xnn_qs8_vadd_minmax_ukernel__wasmsimd_x32(
    size_t n,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN XNN_DISABLE_MSAN
{
  const v128_t vbias = wasm_v128_load(params->wasmsimd.bias);
  const v128_t va_multiplier = wasm_v128_load(params->wasmsimd.a_multiplier);
  const v128_t vb_multiplier = wasm_v128_load(params->wasmsimd.b_multiplier);
  const v128_t vrounding = wasm_v128_load(params->wasmsimd.rounding);
  const int32_t vshift = params->wasmsimd.shift;
  const v128_t voutput_zero_point = wasm_v128_load(params->wasmsimd.output_zero_point);
  const v128_t voutput_min = wasm_v128_load(params->wasmsimd.output_min);
  const v128_t voutput_max = wasm_v128_load(params->wasmsimd.output_max);

  for (; n >= 32 * sizeof(int8_t); n -= 32 * sizeof(int8_t)) {
    const v128_t va01234567 = wasm_i16x8_load8x8(input_a);
    const v128_t vb01234567 = wasm_i16x8_load8x8(input_b);
    const v128_t va89ABCDEF = wasm_i16x8_load8x8(input_a + 8);
    const v128_t vb89ABCDEF = wasm_i16x8_load8x8(input_b + 8);
    const v128_t vaGHIJKLMN = wasm_i16x8_load8x8(input_a + 16);
    const v128_t vbGHIJKLMN = wasm_i16x8_load8x8(input_b + 16);
    const v128_t vaOPQRSTUV = wasm_i16x8_load8x8(input_a + 24);
    const v128_t vbOPQRSTUV = wasm_i16x8_load8x8(input_b + 24);
    input_a += 32;
    input_b += 32;

    v128_t vacc0123 = wasm_i32x4_add(vbias, wasm_i32x4_mul(wasm_i32x4_extend_low_i16x8(va01234567), va_multiplier));
    v128_t vacc4567 = wasm_i32x4_add(vbias, wasm_i32x4_mul(wasm_i32x4_extend_high_i16x8(va01234567), va_multiplier));
    v128_t vacc89AB = wasm_i32x4_add(vbias, wasm_i32x4_mul(wasm_i32x4_extend_low_i16x8(va89ABCDEF), va_multiplier));
    v128_t vaccCDEF = wasm_i32x4_add(vbias, wasm_i32x4_mul(wasm_i32x4_extend_high_i16x8(va89ABCDEF), va_multiplier));
    v128_t vaccGHIJ = wasm_i32x4_add(vbias, wasm_i32x4_mul(wasm_i32x4_extend_low_i16x8(vaGHIJKLMN), va_multiplier));
    v128_t vaccKLMN = wasm_i32x4_add(vbias, wasm_i32x4_mul(wasm_i32x4_extend_high_i16x8(vaGHIJKLMN), va_multiplier));
    v128_t vaccOPQR = wasm_i32x4_add(vbias, wasm_i32x4_mul(wasm_i32x4_extend_low_i16x8(vaOPQRSTUV), va_multiplier));
    v128_t vaccSTUV = wasm_i32x4_add(vbias, wasm_i32x4_mul(wasm_i32x4_extend_high_i16x8(vaOPQRSTUV), va_multiplier));

    vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_mul(wasm_i32x4_extend_low_i16x8(vb01234567), vb_multiplier));
    vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_mul(wasm_i32x4_extend_high_i16x8(vb01234567), vb_multiplier));
    vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_mul(wasm_i32x4_extend_low_i16x8(vb89ABCDEF), vb_multiplier));
    vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_mul(wasm_i32x4_extend_high_i16x8(vb89ABCDEF), vb_multiplier));
    vaccGHIJ = wasm_i32x4_add(vaccGHIJ, wasm_i32x4_mul(wasm_i32x4_extend_low_i16x8(vbGHIJKLMN), vb_multiplier));
    vaccKLMN = wasm_i32x4_add(vaccKLMN, wasm_i32x4_mul(wasm_i32x4_extend_high_i16x8(vbGHIJKLMN), vb_multiplier));
    vaccOPQR = wasm_i32x4_add(vaccOPQR, wasm_i32x4_mul(wasm_i32x4_extend_low_i16x8(vbOPQRSTUV), vb_multiplier));
    vaccSTUV = wasm_i32x4_add(vaccSTUV, wasm_i32x4_mul(wasm_i32x4_extend_high_i16x8(vbOPQRSTUV), vb_multiplier));

    vacc0123 = wasm_i32x4_shr(wasm_i32x4_add(vacc0123, vrounding), vshift);
    vacc4567 = wasm_i32x4_shr(wasm_i32x4_add(vacc4567, vrounding), vshift);
    vacc89AB = wasm_i32x4_shr(wasm_i32x4_add(vacc89AB, vrounding), vshift);
    vaccCDEF = wasm_i32x4_shr(wasm_i32x4_add(vaccCDEF, vrounding), vshift);
    vaccGHIJ = wasm_i32x4_shr(wasm_i32x4_add(vaccGHIJ, vrounding), vshift);
    vaccKLMN = wasm_i32x4_shr(wasm_i32x4_add(vaccKLMN, vrounding), vshift);
    vaccOPQR = wasm_i32x4_shr(wasm_i32x4_add(vaccOPQR, vrounding), vshift);
    vaccSTUV = wasm_i32x4_shr(wasm_i32x4_add(vaccSTUV, vrounding), vshift);

    v128_t vout01234567 = wasm_i16x8_add_sat(wasm_i16x8_narrow_i32x4(vacc0123, vacc4567), voutput_zero_point);
    v128_t vout89ABCDEF = wasm_i16x8_add_sat(wasm_i16x8_narrow_i32x4(vacc89AB, vaccCDEF), voutput_zero_point);
    v128_t voutGHIJKLMN = wasm_i16x8_add_sat(wasm_i16x8_narrow_i32x4(vaccGHIJ, vaccKLMN), voutput_zero_point);
    v128_t voutOPQRSTUV = wasm_i16x8_add_sat(wasm_i16x8_narrow_i32x4(vaccOPQR, vaccSTUV), voutput_zero_point);

    v128_t vout0123456789ABCDEF = wasm_i8x16_narrow_i16x8(vout01234567, vout89ABCDEF);
    v128_t voutGHIJKLMNOPQRSTUV = wasm_i8x16_narrow_i16x8(voutGHIJKLMN, voutOPQRSTUV);

    vout0123456789ABCDEF = wasm_i8x16_max(vout0123456789ABCDEF, voutput_min);
    voutGHIJKLMNOPQRSTUV = wasm_i8x16_max(voutGHIJKLMNOPQRSTUV, voutput_min);

    vout0123456789ABCDEF = wasm_i8x16_min(vout0123456789ABCDEF, voutput_max);
    voutGHIJKLMNOPQRSTUV = wasm_i8x16_min(voutGHIJKLMNOPQRSTUV, voutput_max);

    wasm_v128_store(output, vout0123456789ABCDEF);
    wasm_v128_store(output + 16, voutGHIJKLMNOPQRSTUV);
    output += 32;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const v128_t va01234567 = wasm_i16x8_load8x8(input_a);
      const v128_t vb01234567 = wasm_i16x8_load8x8(input_b);
      input_a += 8;
      input_b += 8;

      v128_t vacc0123 = wasm_i32x4_add(vbias, wasm_i32x4_mul(wasm_i32x4_extend_low_i16x8(va01234567), va_multiplier));
      v128_t vacc4567 = wasm_i32x4_add(vbias, wasm_i32x4_mul(wasm_i32x4_extend_high_i16x8(va01234567), va_multiplier));

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_mul(wasm_i32x4_extend_low_i16x8(vb01234567), vb_multiplier));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_mul(wasm_i32x4_extend_high_i16x8(vb01234567), vb_multiplier));

      vacc0123 = wasm_i32x4_shr(wasm_i32x4_add(vacc0123, vrounding), vshift);
      vacc4567 = wasm_i32x4_shr(wasm_i32x4_add(vacc4567, vrounding), vshift);

      v128_t vout01234567 = wasm_i16x8_add_sat(wasm_i16x8_narrow_i32x4(vacc0123, vacc4567), voutput_zero_point);

      v128_t vout0123456701234567 = wasm_i8x16_narrow_i16x8(vout01234567, vout01234567);
      vout0123456701234567 = wasm_i8x16_max(vout0123456701234567, voutput_min);
      vout0123456701234567 = wasm_i8x16_min(vout0123456701234567, voutput_max);

      if XNN_LIKELY(n >= (8 * sizeof(int8_t))) {
        *((double*) output) = wasm_f64x2_extract_lane(vout0123456701234567, 0);
        output += 8;
        n -= 8 * sizeof(int8_t);
      } else {
        if (n & (4 * sizeof(int8_t))) {
          *((uint32_t*) output) = (uint32_t) wasm_i32x4_extract_lane(vout0123456701234567, 0);
          vout0123456701234567 = wasm_u64x2_shr(vout0123456701234567, 32);
          output += 4;
        }
        if (n & (2 * sizeof(int8_t))) {
          *((uint16_t*) output) = (uint16_t) wasm_i16x8_extract_lane(vout0123456701234567, 0);
          vout0123456701234567 = wasm_u32x4_shr(vout0123456701234567, 16);
          output += 2;
        }
        if (n & (1 * sizeof(int8_t))) {
          *output = (int8_t) wasm_i8x16_extract_lane(vout0123456701234567, 0);
        }
        n = 0;
      }
    } while (n != 0);
  }
}
