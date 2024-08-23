// Auto-generated file. Do not edit!
//   Template: src/qs16-qs8-vcvt/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/vcvt.h"


void xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u32(
    size_t batch,
    const int16_t* input,
    int8_t* output,
    const union xnn_qs16_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vmultiplier = wasm_v128_load32_splat(&params->scalar.multiplier);
  const v128_t vbias = wasm_i64x2_splat(((int32_t) params->scalar.output_zero_point << 16) + INT32_C(0x8000));
  XNN_FORCE_REALIZATION(vmultiplier);
  XNN_FORCE_REALIZATION(vbias);
  for (; batch >= 32 * sizeof(int16_t); batch -= 32 * sizeof(int16_t)) {
    const v128_t vx0 = wasm_i32x4_load16x4(input); input += 4;
    const v128_t vx1 = wasm_i32x4_load16x4(input); input += 4;
    const v128_t vx2 = wasm_i32x4_load16x4(input); input += 4;
    const v128_t vx3 = wasm_i32x4_load16x4(input); input += 4;
    const v128_t vx4 = wasm_i32x4_load16x4(input); input += 4;
    const v128_t vx5 = wasm_i32x4_load16x4(input); input += 4;
    const v128_t vx6 = wasm_i32x4_load16x4(input); input += 4;
    const v128_t vx7 = wasm_i32x4_load16x4(input); input += 4;

    v128_t vacc0lo = wasm_i64x2_extmul_low_i32x4(vx0, vmultiplier);
    v128_t vacc0hi = wasm_i64x2_extmul_high_i32x4(vx0, vmultiplier);
    v128_t vacc1lo = wasm_i64x2_extmul_low_i32x4(vx1, vmultiplier);
    v128_t vacc1hi = wasm_i64x2_extmul_high_i32x4(vx1, vmultiplier);
    v128_t vacc2lo = wasm_i64x2_extmul_low_i32x4(vx2, vmultiplier);
    v128_t vacc2hi = wasm_i64x2_extmul_high_i32x4(vx2, vmultiplier);
    v128_t vacc3lo = wasm_i64x2_extmul_low_i32x4(vx3, vmultiplier);
    v128_t vacc3hi = wasm_i64x2_extmul_high_i32x4(vx3, vmultiplier);
    v128_t vacc4lo = wasm_i64x2_extmul_low_i32x4(vx4, vmultiplier);
    v128_t vacc4hi = wasm_i64x2_extmul_high_i32x4(vx4, vmultiplier);
    v128_t vacc5lo = wasm_i64x2_extmul_low_i32x4(vx5, vmultiplier);
    v128_t vacc5hi = wasm_i64x2_extmul_high_i32x4(vx5, vmultiplier);
    v128_t vacc6lo = wasm_i64x2_extmul_low_i32x4(vx6, vmultiplier);
    v128_t vacc6hi = wasm_i64x2_extmul_high_i32x4(vx6, vmultiplier);
    v128_t vacc7lo = wasm_i64x2_extmul_low_i32x4(vx7, vmultiplier);
    v128_t vacc7hi = wasm_i64x2_extmul_high_i32x4(vx7, vmultiplier);

    vacc0lo = wasm_i64x2_add(vacc0lo, vbias);
    vacc0hi = wasm_i64x2_add(vacc0hi, vbias);
    vacc1lo = wasm_i64x2_add(vacc1lo, vbias);
    vacc1hi = wasm_i64x2_add(vacc1hi, vbias);
    vacc2lo = wasm_i64x2_add(vacc2lo, vbias);
    vacc2hi = wasm_i64x2_add(vacc2hi, vbias);
    vacc3lo = wasm_i64x2_add(vacc3lo, vbias);
    vacc3hi = wasm_i64x2_add(vacc3hi, vbias);
    vacc4lo = wasm_i64x2_add(vacc4lo, vbias);
    vacc4hi = wasm_i64x2_add(vacc4hi, vbias);
    vacc5lo = wasm_i64x2_add(vacc5lo, vbias);
    vacc5hi = wasm_i64x2_add(vacc5hi, vbias);
    vacc6lo = wasm_i64x2_add(vacc6lo, vbias);
    vacc6hi = wasm_i64x2_add(vacc6hi, vbias);
    vacc7lo = wasm_i64x2_add(vacc7lo, vbias);
    vacc7hi = wasm_i64x2_add(vacc7hi, vbias);

    vacc0lo = wasm_i64x2_shr(vacc0lo, 16);
    vacc0hi = wasm_i64x2_shr(vacc0hi, 16);
    vacc1lo = wasm_i64x2_shr(vacc1lo, 16);
    vacc1hi = wasm_i64x2_shr(vacc1hi, 16);
    vacc2lo = wasm_i64x2_shr(vacc2lo, 16);
    vacc2hi = wasm_i64x2_shr(vacc2hi, 16);
    vacc3lo = wasm_i64x2_shr(vacc3lo, 16);
    vacc3hi = wasm_i64x2_shr(vacc3hi, 16);
    vacc4lo = wasm_i64x2_shr(vacc4lo, 16);
    vacc4hi = wasm_i64x2_shr(vacc4hi, 16);
    vacc5lo = wasm_i64x2_shr(vacc5lo, 16);
    vacc5hi = wasm_i64x2_shr(vacc5hi, 16);
    vacc6lo = wasm_i64x2_shr(vacc6lo, 16);
    vacc6hi = wasm_i64x2_shr(vacc6hi, 16);
    vacc7lo = wasm_i64x2_shr(vacc7lo, 16);
    vacc7hi = wasm_i64x2_shr(vacc7hi, 16);

    v128_t vacc0 = wasm_v32x4_shuffle(vacc0lo, vacc0hi, 0, 2, 4, 6);
    v128_t vacc1 = wasm_v32x4_shuffle(vacc1lo, vacc1hi, 0, 2, 4, 6);
    v128_t vacc2 = wasm_v32x4_shuffle(vacc2lo, vacc2hi, 0, 2, 4, 6);
    v128_t vacc3 = wasm_v32x4_shuffle(vacc3lo, vacc3hi, 0, 2, 4, 6);
    v128_t vacc4 = wasm_v32x4_shuffle(vacc4lo, vacc4hi, 0, 2, 4, 6);
    v128_t vacc5 = wasm_v32x4_shuffle(vacc5lo, vacc5hi, 0, 2, 4, 6);
    v128_t vacc6 = wasm_v32x4_shuffle(vacc6lo, vacc6hi, 0, 2, 4, 6);
    v128_t vacc7 = wasm_v32x4_shuffle(vacc7lo, vacc7hi, 0, 2, 4, 6);

    vacc0 = wasm_i16x8_narrow_i32x4(vacc0, vacc0);
    vacc1 = wasm_i16x8_narrow_i32x4(vacc1, vacc1);
    vacc2 = wasm_i16x8_narrow_i32x4(vacc2, vacc2);
    vacc3 = wasm_i16x8_narrow_i32x4(vacc3, vacc3);
    vacc4 = wasm_i16x8_narrow_i32x4(vacc4, vacc4);
    vacc5 = wasm_i16x8_narrow_i32x4(vacc5, vacc5);
    vacc6 = wasm_i16x8_narrow_i32x4(vacc6, vacc6);
    vacc7 = wasm_i16x8_narrow_i32x4(vacc7, vacc7);

    const v128_t vy0 = wasm_i8x16_narrow_i16x8(vacc0, vacc0);
    const v128_t vy1 = wasm_i8x16_narrow_i16x8(vacc1, vacc1);
    const v128_t vy2 = wasm_i8x16_narrow_i16x8(vacc2, vacc2);
    const v128_t vy3 = wasm_i8x16_narrow_i16x8(vacc3, vacc3);
    const v128_t vy4 = wasm_i8x16_narrow_i16x8(vacc4, vacc4);
    const v128_t vy5 = wasm_i8x16_narrow_i16x8(vacc5, vacc5);
    const v128_t vy6 = wasm_i8x16_narrow_i16x8(vacc6, vacc6);
    const v128_t vy7 = wasm_i8x16_narrow_i16x8(vacc7, vacc7);

    wasm_v128_store32_lane(output, vy0, 0);  output += 4;
    wasm_v128_store32_lane(output, vy1, 0);  output += 4;
    wasm_v128_store32_lane(output, vy2, 0);  output += 4;
    wasm_v128_store32_lane(output, vy3, 0);  output += 4;
    wasm_v128_store32_lane(output, vy4, 0);  output += 4;
    wasm_v128_store32_lane(output, vy5, 0);  output += 4;
    wasm_v128_store32_lane(output, vy6, 0);  output += 4;
    wasm_v128_store32_lane(output, vy7, 0);  output += 4;
  }
  for (; batch >= 4 * sizeof(int16_t); batch -= 4 * sizeof(int16_t)) {
    const v128_t vx = wasm_i32x4_load16x4(input); input += 4;
    v128_t vacclo = wasm_i64x2_extmul_low_i32x4(vx, vmultiplier);
    v128_t vacchi = wasm_i64x2_extmul_high_i32x4(vx, vmultiplier);
    vacclo = wasm_i64x2_add(vacclo, vbias);
    vacchi = wasm_i64x2_add(vacchi, vbias);
    vacclo = wasm_i64x2_shr(vacclo, 16);
    vacchi = wasm_i64x2_shr(vacchi, 16);
    v128_t vacc = wasm_v32x4_shuffle(vacclo, vacchi, 0, 2, 4, 6);
    vacc = wasm_i16x8_narrow_i32x4(vacc, vacc);
    const v128_t vy = wasm_i8x16_narrow_i16x8(vacc, vacc);
    wasm_v128_store32_lane(output, vy, 0);  output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int16_t));
    assert(batch <= 3 * sizeof(int16_t));

    const v128_t vx = wasm_i32x4_load16x4(input);
    v128_t vacclo = wasm_i64x2_extmul_low_i32x4(vx, vmultiplier);
    v128_t vacchi = wasm_i64x2_extmul_high_i32x4(vx, vmultiplier);
    vacclo = wasm_i64x2_add(vacclo, vbias);
    vacchi = wasm_i64x2_add(vacchi, vbias);
    vacclo = wasm_i64x2_shr(vacclo, 16);
    vacchi = wasm_i64x2_shr(vacchi, 16);
    v128_t vacc = wasm_v32x4_shuffle(vacclo, vacchi, 0, 2, 4, 6);
    vacc = wasm_i16x8_narrow_i32x4(vacc, vacc);
    v128_t vy = wasm_i8x16_narrow_i16x8(vacc, vacc);

    if (batch & (2 * sizeof(int16_t))) {
      wasm_v128_store16_lane(output, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(int16_t))) {
      wasm_v128_store8_lane(output, vy, 0);
    }
  }
}
