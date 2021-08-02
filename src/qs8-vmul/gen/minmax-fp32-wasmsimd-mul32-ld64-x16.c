// Auto-generated file. Do not edit!
//   Template: src/qs8-vmul/wasmsimd-mul32-ld64.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/vmul.h>


void xnn_qs8_vmul_minmax_fp32_ukernel__wasmsimd_mul32_ld64_x16(
    size_t n,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN XNN_DISABLE_MSAN

{
  const v128_t va_zero_point = wasm_v128_load(params->fp32_wasmsimd.a_zero_point);
  const v128_t vb_zero_point = wasm_v128_load(params->fp32_wasmsimd.b_zero_point);
  const v128_t vscale = wasm_v128_load(params->fp32_wasmsimd.scale);
  const v128_t voutput_min_less_zero_point = wasm_v128_load(params->fp32_wasmsimd.output_min_less_zero_point);
  const v128_t voutput_max_less_zero_point = wasm_v128_load(params->fp32_wasmsimd.output_max_less_zero_point);
  const v128_t vmagic_bias = wasm_v128_load(params->fp32_wasmsimd.magic_bias);
  const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load(params->fp32_wasmsimd.magic_bias_less_output_zero_point);

  for (; n >= 16 * sizeof(int8_t); n -= 16 * sizeof(int8_t)) {
    const v128_t va01234567 = wasm_i16x8_load8x8(input_a);
    const v128_t vb01234567 = wasm_i16x8_load8x8(input_b);
    const v128_t va89ABCDEF = wasm_i16x8_load8x8(input_a + 8);
    const v128_t vb89ABCDEF = wasm_i16x8_load8x8(input_b + 8);
    input_a += 16;
    input_b += 16;

    const v128_t vxa01234567 = wasm_i16x8_sub(va01234567, va_zero_point);
    const v128_t vxb01234567 = wasm_i16x8_sub(vb01234567, vb_zero_point);
    const v128_t vxa89ABCDEF = wasm_i16x8_sub(va89ABCDEF, va_zero_point);
    const v128_t vxb89ABCDEF = wasm_i16x8_sub(vb89ABCDEF, vb_zero_point);

    v128_t vacc0123 = wasm_i32x4_mul(wasm_i32x4_extend_low_i16x8(vxa01234567), wasm_i32x4_extend_low_i16x8(vxb01234567));
    v128_t vacc4567 = wasm_i32x4_mul(wasm_i32x4_extend_high_i16x8(vxa01234567), wasm_i32x4_extend_high_i16x8(vxb01234567));
    v128_t vacc89AB = wasm_i32x4_mul(wasm_i32x4_extend_low_i16x8(vxa89ABCDEF), wasm_i32x4_extend_low_i16x8(vxb89ABCDEF));
    v128_t vaccCDEF = wasm_i32x4_mul(wasm_i32x4_extend_high_i16x8(vxa89ABCDEF), wasm_i32x4_extend_high_i16x8(vxb89ABCDEF));

    vacc0123 = wasm_f32x4_convert_i32x4(vacc0123);
    vacc4567 = wasm_f32x4_convert_i32x4(vacc4567);
    vacc89AB = wasm_f32x4_convert_i32x4(vacc89AB);
    vaccCDEF = wasm_f32x4_convert_i32x4(vaccCDEF);

    vacc0123 = wasm_f32x4_mul(vacc0123, vscale);
    vacc4567 = wasm_f32x4_mul(vacc4567, vscale);
    vacc89AB = wasm_f32x4_mul(vacc89AB, vscale);
    vaccCDEF = wasm_f32x4_mul(vaccCDEF, vscale);

    vacc0123 = wasm_f32x4_max(vacc0123, voutput_min_less_zero_point);
    vacc4567 = wasm_f32x4_max(vacc4567, voutput_min_less_zero_point);
    vacc89AB = wasm_f32x4_max(vacc89AB, voutput_min_less_zero_point);
    vaccCDEF = wasm_f32x4_max(vaccCDEF, voutput_min_less_zero_point);

    vacc0123 = wasm_f32x4_min(vacc0123, voutput_max_less_zero_point);
    vacc4567 = wasm_f32x4_min(vacc4567, voutput_max_less_zero_point);
    vacc89AB = wasm_f32x4_min(vacc89AB, voutput_max_less_zero_point);
    vaccCDEF = wasm_f32x4_min(vaccCDEF, voutput_max_less_zero_point);

    vacc0123 = wasm_f32x4_add(vacc0123, vmagic_bias);
    vacc4567 = wasm_f32x4_add(vacc4567, vmagic_bias);
    vacc89AB = wasm_f32x4_add(vacc89AB, vmagic_bias);
    vaccCDEF = wasm_f32x4_add(vaccCDEF, vmagic_bias);

    vacc0123 = wasm_i32x4_sub(vacc0123, vmagic_bias_less_output_zero_point);
    vacc4567 = wasm_i32x4_sub(vacc4567, vmagic_bias_less_output_zero_point);
    vacc89AB = wasm_i32x4_sub(vacc89AB, vmagic_bias_less_output_zero_point);
    vaccCDEF = wasm_i32x4_sub(vaccCDEF, vmagic_bias_less_output_zero_point);

    v128_t vout01234567 = wasm_v16x8_shuffle(vacc0123, vacc4567, 0, 2, 4, 6, 8, 10, 12, 14);
    v128_t vout89ABCDEF = wasm_v16x8_shuffle(vacc89AB, vaccCDEF, 0, 2, 4, 6, 8, 10, 12, 14);

    v128_t vout0123456789ABCDEF = wasm_v8x16_shuffle(vout01234567, vout89ABCDEF, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);

    wasm_v128_store(output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const v128_t va01234567 = wasm_i16x8_load8x8(input_a);
      const v128_t vb01234567 = wasm_i16x8_load8x8(input_b);
      input_a += 8;
      input_b += 8;

      const v128_t vxa01234567 = wasm_i16x8_sub(va01234567, va_zero_point);
      const v128_t vxb01234567 = wasm_i16x8_sub(vb01234567, vb_zero_point);

      v128_t vacc0123 = wasm_i32x4_mul(wasm_i32x4_extend_low_i16x8(vxa01234567), wasm_i32x4_extend_low_i16x8(vxb01234567));
      v128_t vacc4567 = wasm_i32x4_mul(wasm_i32x4_extend_high_i16x8(vxa01234567), wasm_i32x4_extend_high_i16x8(vxb01234567));

      vacc0123 = wasm_f32x4_convert_i32x4(vacc0123);
      vacc4567 = wasm_f32x4_convert_i32x4(vacc4567);

      vacc0123 = wasm_f32x4_mul(vacc0123, vscale);
      vacc4567 = wasm_f32x4_mul(vacc4567, vscale);

      vacc0123 = wasm_f32x4_max(vacc0123, voutput_min_less_zero_point);
      vacc4567 = wasm_f32x4_max(vacc4567, voutput_min_less_zero_point);

      vacc0123 = wasm_f32x4_min(vacc0123, voutput_max_less_zero_point);
      vacc4567 = wasm_f32x4_min(vacc4567, voutput_max_less_zero_point);

      vacc0123 = wasm_f32x4_add(vacc0123, vmagic_bias);
      vacc4567 = wasm_f32x4_add(vacc4567, vmagic_bias);

      vacc0123 = wasm_i32x4_sub(vacc0123, vmagic_bias_less_output_zero_point);
      vacc4567 = wasm_i32x4_sub(vacc4567, vmagic_bias_less_output_zero_point);

      v128_t vout01234567 = wasm_v16x8_shuffle(vacc0123, vacc4567, 0, 2, 4, 6, 8, 10, 12, 14);
      v128_t vout0123456701234567 = wasm_v8x16_shuffle(vout01234567, vout01234567, 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14);

      if XNN_LIKELY(n >= (8 * sizeof(int8_t))) {
        *((double*) output) = wasm_f64x2_extract_lane(vout0123456701234567, 0);
        output += 8;
        n -= 8 * sizeof(int8_t);
      } else {
        if (n & (4 * sizeof(int8_t))) {
          *((float*) output) = wasm_f32x4_extract_lane(vout0123456701234567, 0);
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
