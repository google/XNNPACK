// Auto-generated file. Do not edit!
//   Template: src/qs8-vaddc/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/vaddsub.h>


void xnn_qu8_vaddc_minmax_ukernel__wasmsimd_x8(
    size_t n,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_addsub_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  const v128_t va_multiplier = wasm_v128_load64_splat(params->wasmsimd.a_multiplier);
  const uint32_t vshift = params->wasmsimd.shift;
  const v128_t voutput_zero_point = wasm_v128_load64_splat(params->wasmsimd.output_zero_point);
  const v128_t voutput_min = wasm_v128_load64_splat(params->wasmsimd.output_min);
  const v128_t voutput_max = wasm_v128_load64_splat(params->wasmsimd.output_max);

  v128_t vbias = wasm_i32x4_splat((int32_t) *input_b * params->wasmsimd.b_multiplier[0]);
  vbias = wasm_i32x4_add(vbias, wasm_v128_load64_splat(params->wasmsimd.bias));

  for (; n >= 8 * sizeof(uint8_t); n -= 8 * sizeof(uint8_t)) {
    const v128_t va01234567 = wasm_u16x8_load8x8(input_a);
    input_a += 8;

    v128_t vacc0123 = wasm_i32x4_add(vbias, wasm_i32x4_mul(wasm_u32x4_extend_low_u16x8(va01234567), va_multiplier));
    v128_t vacc4567 = wasm_i32x4_add(vbias, wasm_i32x4_mul(wasm_u32x4_extend_high_u16x8(va01234567), va_multiplier));

    vacc0123 = wasm_i32x4_shr(vacc0123, vshift);
    vacc4567 = wasm_i32x4_shr(vacc4567, vshift);

    v128_t vout01234567 = wasm_i16x8_add_sat(wasm_i16x8_narrow_i32x4(vacc0123, vacc4567), voutput_zero_point);

    v128_t vout0123456701234567 = wasm_u8x16_narrow_i16x8(vout01234567, vout01234567);

    vout0123456701234567 = wasm_u8x16_max(vout0123456701234567, voutput_min);

    vout0123456701234567 = wasm_u8x16_min(vout0123456701234567, voutput_max);

    *((double*) output) = wasm_f64x2_extract_lane(vout0123456701234567, 0);
    output += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    {
      const v128_t va01234567 = wasm_u16x8_load8x8(input_a);

      v128_t vacc0123 = wasm_i32x4_add(vbias, wasm_i32x4_mul(wasm_u32x4_extend_low_u16x8(va01234567), va_multiplier));
      v128_t vacc4567 = wasm_i32x4_add(vbias, wasm_i32x4_mul(wasm_u32x4_extend_high_u16x8(va01234567), va_multiplier));

      vacc0123 = wasm_i32x4_shr(vacc0123, vshift);
      vacc4567 = wasm_i32x4_shr(vacc4567, vshift);

      v128_t vout01234567 = wasm_i16x8_add_sat(wasm_i16x8_narrow_i32x4(vacc0123, vacc4567), voutput_zero_point);

      v128_t vout0123456701234567 = wasm_u8x16_narrow_i16x8(vout01234567, vout01234567);
      vout0123456701234567 = wasm_u8x16_max(vout0123456701234567, voutput_min);
      vout0123456701234567 = wasm_u8x16_min(vout0123456701234567, voutput_max);

      if (n & (4 * sizeof(uint8_t))) {
        *((float*) output) = (float) wasm_f32x4_extract_lane(vout0123456701234567, 0);
        vout0123456701234567 = wasm_u64x2_shr(vout0123456701234567, 32);
        output += 4;
      }
      uint32_t vout0123 = wasm_i32x4_extract_lane(vout0123456701234567, 0);
      if (n & (2 * sizeof(uint8_t))) {
        *((uint16_t*) output) = (uint16_t) vout0123;
        vout0123 >>= 16;
        output += 2;
      }
      if (n & (1 * sizeof(uint8_t))) {
        *output = (uint8_t) vout0123;
      }
    }
  }
}
