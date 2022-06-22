// Auto-generated file. Do not edit!
//   Template: src/qs8-vcvt/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/common.h>
#include <xnnpack/vcvt.h>


void xnn_qu8_vcvt_ukernel__wasmsimd_x16(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const union xnn_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(uint8_t) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const v128_t vinput_zero_point = wasm_v128_load64_splat(params->wasmsimd.input_zero_point);
  const v128_t vmultiplier = wasm_v128_load64_splat(params->wasmsimd.multiplier);
  const v128_t voutput_zero_point = wasm_v128_load64_splat(params->wasmsimd.output_zero_point);
  for (; n >= 16 * sizeof(uint8_t); n -= 16 * sizeof(uint8_t)) {
    v128_t vacc0 = wasm_u16x8_load8x8(x);
    v128_t vacc1 = wasm_u16x8_load8x8(x + 8);
    x += 16;

    vacc0 = wasm_i16x8_sub(vinput_zero_point, vacc0);
    vacc1 = wasm_i16x8_sub(vinput_zero_point, vacc1);

    vacc0 = wasm_i16x8_shl(vacc0, 7);
    vacc1 = wasm_i16x8_shl(vacc1, 7);

    vacc0 = wasm_i16x8_q15mulr_sat(vacc0, vmultiplier);
    vacc1 = wasm_i16x8_q15mulr_sat(vacc1, vmultiplier);

    vacc0 = wasm_i16x8_add_sat(vacc0, voutput_zero_point);
    vacc1 = wasm_i16x8_add_sat(vacc1, voutput_zero_point);

    const v128_t vy0 = wasm_u8x16_narrow_i16x8(vacc0, vacc1);

    wasm_v128_store(y, vy0);
    y += 16;
  }
  for (; n >= 8 * sizeof(uint8_t); n -= 8 * sizeof(uint8_t)) {
    v128_t vacc = wasm_u16x8_load8x8(x);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vacc = wasm_i16x8_shl(vacc, 7);
    vacc = wasm_i16x8_q15mulr_sat(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);
    x += 8;

    const v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);
    wasm_v128_store64_lane(y, vy, 0);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(uint8_t));
    assert(n <= 7 * sizeof(uint8_t));

    v128_t vacc = wasm_u16x8_load8x8(x);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vacc = wasm_i16x8_shl(vacc, 7);
    vacc = wasm_i16x8_q15mulr_sat(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);

    v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);
    if (n & (4 * sizeof(uint8_t))) {
      wasm_v128_store32_lane(y, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      y += 4;
    }
    if (n & (2 * sizeof(uint8_t))) {
      wasm_v128_store16_lane(y, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      y += 2;
    }
    if (n & (1 * sizeof(uint8_t))) {
      wasm_v128_store8_lane(y, vy, 0);
    }
  }
}
