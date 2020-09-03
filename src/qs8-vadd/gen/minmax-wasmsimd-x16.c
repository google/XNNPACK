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


void xnn_qs8_vadd_minmax_ukernel__wasmsimd_x16(
    size_t n,
    const int8_t* input_x,
    const int8_t* input_y,
    int8_t* output,
    const union xnn_qs8_add_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  const v128_t vzero_point_product = wasm_v128_load(params->wasmsimd.zero_point_product);
  const v128_t vx_multiplier = wasm_v128_load(params->wasmsimd.x_multiplier);
  const v128_t vy_multiplier = wasm_v128_load(params->wasmsimd.y_multiplier);
  const v128_t vremainder_mask = wasm_v128_load(params->wasmsimd.remainder_mask);
  const v128_t vremainder_threshold = wasm_v128_load(params->wasmsimd.remainder_threshold);
  const int32_t vshift = params->wasmsimd.shift;
  const v128_t voutput_zero_point = wasm_v128_load(params->wasmsimd.output_zero_point);
  const v128_t voutput_min = wasm_v128_load(params->wasmsimd.output_min);
  const v128_t voutput_max = wasm_v128_load(params->wasmsimd.output_max);

  for (; n >= 16 * sizeof(int8_t); n -= 16 * sizeof(int8_t)) {
    const v128_t vx01234567 = wasm_i16x8_load_8x8(input_x);
    const v128_t vy01234567 = wasm_i16x8_load_8x8(input_y);
    const v128_t vx89ABCDEF = wasm_i16x8_load_8x8(input_x + 8);
    const v128_t vy89ABCDEF = wasm_i16x8_load_8x8(input_y + 8);
    input_x += 16;
    input_y += 16;

    v128_t vacc0123 = wasm_i32x4_add(vzero_point_product, wasm_i32x4_mul(wasm_i32x4_widen_low_i16x8(vx01234567), vx_multiplier));
    v128_t vacc4567 = wasm_i32x4_add(vzero_point_product, wasm_i32x4_mul(wasm_i32x4_widen_high_i16x8(vx01234567), vx_multiplier));
    v128_t vacc89AB = wasm_i32x4_add(vzero_point_product, wasm_i32x4_mul(wasm_i32x4_widen_low_i16x8(vx89ABCDEF), vx_multiplier));
    v128_t vaccCDEF = wasm_i32x4_add(vzero_point_product, wasm_i32x4_mul(wasm_i32x4_widen_high_i16x8(vx89ABCDEF), vx_multiplier));

    vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_mul(wasm_i32x4_widen_low_i16x8(vy01234567), vy_multiplier));
    vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_mul(wasm_i32x4_widen_high_i16x8(vy01234567), vy_multiplier));
    vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_mul(wasm_i32x4_widen_low_i16x8(vy89ABCDEF), vy_multiplier));
    vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_mul(wasm_i32x4_widen_high_i16x8(vy89ABCDEF), vy_multiplier));

    const v128_t vrem0123 = wasm_i32x4_add(wasm_v128_and(vacc0123, vremainder_mask), wasm_i32x4_shr(vacc0123, 31));
    const v128_t vrem4567 = wasm_i32x4_add(wasm_v128_and(vacc4567, vremainder_mask), wasm_i32x4_shr(vacc4567, 31));
    const v128_t vrem89AB = wasm_i32x4_add(wasm_v128_and(vacc89AB, vremainder_mask), wasm_i32x4_shr(vacc89AB, 31));
    const v128_t vremCDEF = wasm_i32x4_add(wasm_v128_and(vaccCDEF, vremainder_mask), wasm_i32x4_shr(vaccCDEF, 31));

    vacc0123 = wasm_i32x4_sub(wasm_i32x4_shr(vacc0123, vshift), wasm_i32x4_gt(vrem0123, vremainder_threshold));
    vacc4567 = wasm_i32x4_sub(wasm_i32x4_shr(vacc4567, vshift), wasm_i32x4_gt(vrem4567, vremainder_threshold));
    vacc89AB = wasm_i32x4_sub(wasm_i32x4_shr(vacc89AB, vshift), wasm_i32x4_gt(vrem89AB, vremainder_threshold));
    vaccCDEF = wasm_i32x4_sub(wasm_i32x4_shr(vaccCDEF, vshift), wasm_i32x4_gt(vremCDEF, vremainder_threshold));

    v128_t vout01234567 = wasm_i16x8_add_saturate(wasm_i16x8_narrow_i32x4(vacc0123, vacc4567), voutput_zero_point);
    v128_t vout89ABCDEF = wasm_i16x8_add_saturate(wasm_i16x8_narrow_i32x4(vacc89AB, vaccCDEF), voutput_zero_point);

    v128_t vout0123456789ABCDEF = wasm_i8x16_narrow_i16x8(vout01234567, vout89ABCDEF);

    vout0123456789ABCDEF = wasm_i8x16_max(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = wasm_i8x16_min(vout0123456789ABCDEF, voutput_max);

    wasm_v128_store(output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const v128_t vx01234567 = wasm_i16x8_load_8x8(input_x);
      const v128_t vy01234567 = wasm_i16x8_load_8x8(input_y);
      input_x += 8;
      input_y += 8;

      v128_t vacc0123 = wasm_i32x4_add(vzero_point_product, wasm_i32x4_mul(wasm_i32x4_widen_low_i16x8(vx01234567), vx_multiplier));
      v128_t vacc4567 = wasm_i32x4_add(vzero_point_product, wasm_i32x4_mul(wasm_i32x4_widen_high_i16x8(vx01234567), vx_multiplier));

      vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_mul(wasm_i32x4_widen_low_i16x8(vy01234567), vy_multiplier));
      vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_mul(wasm_i32x4_widen_high_i16x8(vy01234567), vy_multiplier));

      const v128_t vrem0123 = wasm_i32x4_add(wasm_v128_and(vacc0123, vremainder_mask), wasm_i32x4_shr(vacc0123, 31));
      const v128_t vrem4567 = wasm_i32x4_add(wasm_v128_and(vacc4567, vremainder_mask), wasm_i32x4_shr(vacc4567, 31));

      vacc0123 = wasm_i32x4_sub(wasm_i32x4_shr(vacc0123, vshift), wasm_i32x4_gt(vrem0123, vremainder_threshold));
      vacc4567 = wasm_i32x4_sub(wasm_i32x4_shr(vacc4567, vshift), wasm_i32x4_gt(vrem4567, vremainder_threshold));

      v128_t vout01234567 = wasm_i16x8_add_saturate(wasm_i16x8_narrow_i32x4(vacc0123, vacc4567), voutput_zero_point);

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
          *output = wasm_i8x16_extract_lane(vout0123456701234567, 0);
        }
        n = 0;
      }
    } while (n != 0);
  }
}
